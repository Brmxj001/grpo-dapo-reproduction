"""GRPO 训练循环（独立版）。

采样 → 奖励 → 组内优势 → 旧策略 log prob → GRPO loss → 反向传播 →
优化器更新 → 记录指标。

已支持（config 里的对应字段都会被读取）：
- LoRA：`lora.enabled=true` 时套 adapter，参考策略用 `disable_adapter()`
  复用同一模型，不额外占一份权重；
- 梯度累积：`training.grad_accum_steps`，每个优化步采样
  `prompts_per_batch × grad_accum_steps` 道题，但显存只按 `prompts_per_batch` 计；
- LR 调度：`training.warmup_ratio` > 0 时线性 warmup，之后恒定；
- 优化器超参：`weight_decay` / `betas` / `adam_eps`；
- 数据打乱：`training.shuffle`（默认 false，保持与作者基线一致）；
- 断点续训：`save_checkpoint()` 会存优化器/调度器/步数，`load_checkpoint()` 恢复；
- 诊断指标：策略熵、截断率。

不包含：分布式训练、训练中评测（用 scripts/evaluate_gsm8k.py 单独评）。
模型加载、Prompt 与奖励逻辑复用项目已有的 src/ 模块，避免重复实现。
"""

from __future__ import annotations

import copy
import random
import time
from pathlib import Path

import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .advantage import compute_group_advantages
from .loss import GRPOLossConfig, compute_grpo_loss
from .metrics import MetricsLogger, plot_metrics, reward_statistics
from .rollout import (
    HFGenerator,
    PromptGroup,
    RolloutBuffer,
    RolloutSample,
    SamplingParams,
    aligned_log_probs_and_entropy,
    aligned_token_log_probs,
    chunked_aligned_token_log_probs,
)

from src.data.prompt import build_prompt
from src.model_loader import load_model, load_tokenizer, resolve_device
from src.rewards.reward import RewardConfig, compute_reward
from src.utils.config import load_config, resolve_path
from src.utils.seed import set_seed


class GRPOTrainer:
    def __init__(self, config: dict, model=None, tokenizer=None, generator=None, ref_model=None):
        self.config = config
        set_seed(int(config.get("seed", 42)))

        model_cfg = config.get("model", {})
        self.device = resolve_device(model_cfg.get("device", "auto"))
        self.tokenizer = tokenizer or load_tokenizer(model_cfg.get("local_path"))
        self.model = model or load_model(model_cfg.get("local_path"), device=self.device)

        # LoRA：只在没有外部注入模型时应用，避免测试里被二次包装
        self._is_peft = False
        lora_cfg = config.get("lora", {})
        if lora_cfg.get("enabled", False) and model is None:
            from peft import LoraConfig, get_peft_model

            peft_config = LoraConfig(
                r=int(lora_cfg.get("r", 16)),
                lora_alpha=int(lora_cfg.get("alpha", 32)),
                lora_dropout=float(lora_cfg.get("dropout", 0.0)),
                target_modules=list(lora_cfg.get("target_modules", [])),
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, peft_config)
            self._is_peft = True
            if hasattr(self.model, "print_trainable_parameters"):
                self.model.print_trainable_parameters()

        self.model.to(self.device)

        rollout_cfg = config.get("rollout", {})
        self.num_generations = int(rollout_cfg.get("num_generations", 4))
        self.sampling = SamplingParams(
            num_generations=self.num_generations,
            max_new_tokens=int(rollout_cfg.get("max_new_tokens", 512)),
            temperature=float(rollout_cfg.get("temperature", 1.0)),
            top_p=float(rollout_cfg.get("top_p", 1.0)),
            top_k=int(rollout_cfg.get("top_k", -1)),
        )
        self.generator = generator or HFGenerator(
            self.model, self.tokenizer, self.sampling, device=self.device
        )

        algorithm = config.get("algorithm", {})
        self.loss_config = GRPOLossConfig(
            clip_ratio=float(algorithm.get("clip_ratio", algorithm.get("clip_ratio_low", 0.2))),
            kl_coef=float(algorithm.get("kl_coef", 0.04)),
            kl_estimator=str(algorithm.get("kl_estimator", "k3")),
            normalization=str(algorithm.get("normalization", "sequence")),
        )
        reward_cfg = config.get("reward", {})
        self.reward_config = RewardConfig(
            correct_reward=float(reward_cfg.get("correct_reward", 1.0)),
            incorrect_reward=float(reward_cfg.get("incorrect_reward", 0.0)),
            format_weight=float(reward_cfg.get("format_weight", 0.0)),
        )

        self.ref_model = ref_model
        if self.ref_model is None and self.loss_config.kl_coef != 0.0:
            if self._is_peft:
                # LoRA：基座权重被冻结且不变，参考策略 = 关掉 adapter 的同一模型。
                # 因此不需要 deepcopy，KL 计算时临时 disable_adapter() 即可，省一份权重。
                pass
            else:
                # 全参微调：参考模型与策略同尺寸，显存/内存约翻倍
                self.ref_model = copy.deepcopy(self.model)
        if self.ref_model is not None:
            self.ref_model.to(self.device).eval()
            for parameter in self.ref_model.parameters():
                parameter.requires_grad_(False)

        data_cfg = config.get("data", {})
        frame = pd.read_parquet(
            resolve_path(data_cfg.get("train_file", "data/processed/gsm8k/train.parquet"))
        )
        missing = {"question", "ground_truth"} - set(frame.columns)
        if missing:
            raise ValueError(f"训练数据缺少字段：{missing}")
        self.problems = list(zip(frame["question"].tolist(), frame["ground_truth"].tolist()))

        training_cfg = config.get("training", {})
        self.max_steps = int(training_cfg.get("max_steps", 500))
        self.prompts_per_batch = int(training_cfg.get("prompts_per_batch", 1))
        # 梯度累积：一个优化步里跑几轮"采样+前向+反向"。
        # 显存按 prompts_per_batch 计，等效 batch = prompts_per_batch × grad_accum_steps。
        self.grad_accum_steps = max(1, int(training_cfg.get("grad_accum_steps", 1)))
        self.max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
        self.save_every = int(training_cfg.get("save_every", 0))
        self.log_every = max(1, int(training_cfg.get("log_every", 1)))
        self.warmup_ratio = float(training_cfg.get("warmup_ratio", 0.0))
        self.shuffle = bool(training_cfg.get("shuffle", False))

        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(training_cfg.get("learning_rate", 1e-6)),
            betas=tuple(training_cfg.get("betas", (0.9, 0.999))),
            eps=float(training_cfg.get("adam_eps", 1e-8)),
            weight_decay=float(training_cfg.get("weight_decay", 0.0)),
        )
        self.scheduler = self._build_scheduler()

        # 数据顺序：默认按文件顺序循环（与作者基线一致）；
        # shuffle=true 时用固定 seed 打乱一次，保证可复现。
        if self.shuffle:
            rng = random.Random(int(config.get("seed", 42)))
            self._order = list(range(len(self.problems)))
            rng.shuffle(self._order)
        else:
            self._order = None

        self.output_dir = resolve_path(training_cfg.get("output_dir", "outputs/GRPO"))
        self.logger = MetricsLogger(self.output_dir / "logs")
        self._cursor = 0
        self._step = 0

    def _build_scheduler(self):
        """线性 warmup → 恒定 LR；warmup_ratio<=0 时不建调度器。"""
        if self.warmup_ratio <= 0:
            return None
        warmup_steps = max(1, int(self.max_steps * self.warmup_ratio))

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            return 1.0

        return LambdaLR(self.optimizer, lr_lambda)

    @classmethod
    def from_yaml(cls, path, **kwargs) -> "GRPOTrainer":
        return cls(load_config(path), **kwargs)

    def rollout_group(self, question: str, ground_truth: str) -> PromptGroup:
        """对一道题采样 G 条回答并计算奖励。"""
        prompt = build_prompt(question, self.tokenizer)
        outputs = self.generator.generate(prompt, self.sampling)
        group = PromptGroup(
            prompt_ids=outputs[0].prompt_ids,
            prompt_text=prompt,
            question=question,
            ground_truth=ground_truth,
        )
        for output in outputs:
            result = compute_reward(
                output.response_text, ground_truth, config=self.reward_config
            )
            group.add_sample(
                RolloutSample(
                    response_ids=output.response_ids,
                    response_text=output.response_text,
                    reward=result.reward,
                    correct=result.correct,
                    truncated=output.truncated,
                )
            )
        return group

    def collect_batch(self) -> RolloutBuffer:
        buffer = RolloutBuffer()
        for _ in range(self.prompts_per_batch):
            position = self._cursor % len(self.problems)
            index = self._order[position] if self._order is not None else position
            question, ground_truth = self.problems[index]
            self._cursor += 1
            buffer.add_group(self.rollout_group(question, ground_truth))
        return buffer

    def _old_and_ref_log_probs(self, buffer: RolloutBuffer) -> dict:
        pad_token_id = self.tokenizer.pad_token_id
        batch = buffer.to_batch(pad_token_id=pad_token_id if pad_token_id is not None else 0)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        self.model.eval()
        with torch.inference_mode():
            # 同一次前向顺带算出策略熵（诊断指标，不建图）
            old_log_probs, policy_entropy = aligned_log_probs_and_entropy(
                self.model, input_ids, attention_mask
            )
            batch["old_log_probs"] = old_log_probs.float().cpu()
            batch["policy_entropy"] = policy_entropy.float().cpu()
            if self.ref_model is not None:
                batch["ref_log_probs"] = aligned_token_log_probs(
                    self.ref_model, input_ids, attention_mask
                ).float().cpu()
            elif self._is_peft:
                # LoRA 的参考策略：临时禁用 adapter，得到基座模型的 log prob
                with self.model.disable_adapter():
                    batch["ref_log_probs"] = aligned_token_log_probs(
                        self.model, input_ids, attention_mask
                    ).float().cpu()
        self.model.train()
        return batch

    def _micro_step(self) -> dict:
        """采样一个 micro-batch、算损失并反传梯度（不更新参数）。

        返回原始数值，由 train_step 汇总成一个优化步的指标。
        """
        buffer = self.collect_batch()
        batch = self._old_and_ref_log_probs(buffer)

        rewards = buffer.rewards()
        group_count = buffer.num_groups
        if buffer.num_samples != group_count * self.num_generations:
            raise RuntimeError("每组回答数不等于 num_generations")
        advantage_result = compute_group_advantages(
            rewards.view(group_count, self.num_generations), self.num_generations
        )
        advantages = advantage_result.advantages.reshape(-1)

        mask = batch["response_mask"].to(self.device)
        # 训练前向必须建图，用分块版本：整份 [N, L, V] 的 fp32 log_softmax
        # 加 gather 反向梯度约 4 GB，长回答步会 OOM（见 rollout 里的说明）。
        policy_log_probs = chunked_aligned_token_log_probs(
            self.model,
            batch["input_ids"].to(self.device),
            batch["attention_mask"].to(self.device),
        )
        ref_log_probs = (
            None if batch["ref_log_probs"] is None
            else batch["ref_log_probs"].to(self.device)
        )
        output = compute_grpo_loss(
            policy_log_probs=policy_log_probs,
            old_log_probs=batch["old_log_probs"].to(self.device),
            advantages=advantages.to(self.device),
            response_mask=mask,
            ref_log_probs=ref_log_probs,
            config=self.loss_config,
        )

        # 除以累积步数：多轮梯度相加后 = 一个等效大 batch 的平均梯度
        (output.loss / self.grad_accum_steps).backward()

        entropy = batch["policy_entropy"].to(self.device)
        return {
            "loss_dict": output.as_dict(),
            "rewards": rewards.tolist(),
            "zero_variance": advantage_result.zero_variance.detach().cpu(),
            # 熵按 response token 数加权平均，不能简单对 micro-batch 的均值再平均
            "entropy_sum": float((entropy * mask).sum()),
            "entropy_count": float(mask.sum()),
            "avg_length": float(batch["response_lengths"].float().mean()),
            "truncation": float(batch["truncated"].float().mean()),
        }

    def train_step(self) -> dict[str, float]:
        """一次优化步：grad_accum_steps 个 micro-batch → 一次参数更新。"""
        start = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.model.train()

        micro_results = [self._micro_step() for _ in range(self.grad_accum_steps)]

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], self.max_grad_norm
        )
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        self._step += 1
        count = len(micro_results)

        # 各 micro-batch 样本数相同，所以这些指标取平均即为整体值
        metrics = {
            key: sum(result["loss_dict"][key] for result in micro_results) / count
            for key in ("loss", "policy_loss", "kl", "clip_fraction", "ratio_mean")
        }
        metrics["num_response_tokens"] = sum(
            result["loss_dict"]["num_response_tokens"] for result in micro_results
        )

        all_rewards = [value for result in micro_results for value in result["rewards"]]
        metrics.update(reward_statistics(all_rewards))
        metrics.update(
            zero_variance_ratio=float(
                torch.cat([result["zero_variance"] for result in micro_results]).float().mean()
            ),
            avg_response_length=sum(result["avg_length"] for result in micro_results) / count,
            truncation_rate=sum(result["truncation"] for result in micro_results) / count,
            policy_entropy=(
                sum(result["entropy_sum"] for result in micro_results)
                / max(sum(result["entropy_count"] for result in micro_results), 1.0)
            ),
            grad_norm=float(grad_norm),
            learning_rate=float(self.optimizer.param_groups[0]["lr"]),
            step_time=time.perf_counter() - start,
        )
        if torch.cuda.is_available():
            # 本步真实的张量分配峰值（不含 PyTorch 缓存），
            # 用来判断离 12 GB 上限还有多少余量。
            metrics["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)

        if self._step % self.log_every == 0 or self._step >= self.max_steps:
            self.logger.log(self._step, metrics)
        return metrics

    def save_checkpoint(self, step: int) -> None:
        """模型权重 + 优化器/调度器状态 + 步数，用于断点续训。"""
        path = self.output_dir / "checkpoints" / f"step_{step}"
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(path)
        torch.save(
            {
                "step": self._step,
                "cursor": self._cursor,
                "optimizer": self.optimizer.state_dict(),
                "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
                # RNG 状态：续训后同一步的采样才能与未中断时一致
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "python_rng": random.getstate(),
            },
            path / "trainer_state.pt",
        )
        print(f"检查点已保存：{path}")

    def load_checkpoint(self, path: str | Path) -> None:
        """从 save_checkpoint 产出的目录恢复训练状态。

        LoRA 只加载 adapter_model.safetensors；全参加载 model.safetensors。
        优化器状态会被搬到参数所在设备。
        """
        path = Path(resolve_path(path))
        if not path.exists():
            raise FileNotFoundError(f"检查点不存在：{path}")

        from safetensors.torch import load_file

        if self._is_peft:
            from peft import set_peft_model_state_dict

            set_peft_model_state_dict(self.model, load_file(str(path / "adapter_model.safetensors")))
        else:
            state_dict = load_file(str(path / "model.safetensors"))
            missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                print(f"注意：状态字典不完全匹配 missing={len(missing)} unexpected={len(unexpected)}")

        state_path = path / "trainer_state.pt"
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            self.optimizer.load_state_dict(state["optimizer"])
            if self.scheduler is not None and state.get("scheduler"):
                self.scheduler.load_state_dict(state["scheduler"])
            self._step = int(state.get("step", 0))
            self._cursor = int(state.get("cursor", 0))
            if state.get("torch_rng") is not None:
                torch.set_rng_state(state["torch_rng"])
            if state.get("cuda_rng"):
                torch.cuda.set_rng_state_all(state["cuda_rng"])
            if state.get("python_rng") is not None:
                random.setstate(state["python_rng"])
            # 回退点之后的指标对应的权重已被丢弃，续训会重算这些步
            removed = self.logger.truncate_after(self._step)
            if removed:
                print(f"指标日志已回退到第 {self._step} 步（删除 {removed} 条旧记录）")
        print(f"已从检查点恢复：{path}（第 {self._step} 步）")

    def train(self) -> list[dict]:
        start_step = self._step
        if start_step:
            print(f"从第 {start_step} 步继续训练")
        while self._step < self.max_steps:
            self.train_step()
            if self.save_every > 0 and self._step % self.save_every == 0:
                self.save_checkpoint(self._step)
        # 训练结束时补存一次：max_steps 不是 save_every 的整数倍时，
        # 否则最后几十步的状态会丢失，无法从最终步续训或评测。
        if self.save_every > 0 and self._step % self.save_every != 0:
            self.save_checkpoint(self._step)
        plot_metrics(self.logger.history, self.output_dir / "plots")
        self.logger.save()
        self.logger.close()
        print(f"训练结束：{self.max_steps} 步，日志位于 {self.output_dir}")
        return self.logger.history
