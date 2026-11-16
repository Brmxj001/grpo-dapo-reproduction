"""GRPO / DAPO 最小训练循环（真实训练需 GPU，本机未执行）。

流程：Rollout（每题 G 条）→ 规则奖励（可选 Overlong Shaping）→
Dynamic Sampling 过滤无效组 → 组内优势 → 旧策略 / Reference log prob →
GRPO / DAPO loss → 反传更新。

说明：本模块尚未在真实 Qwen2.5-0.5B 上运行过；CPU 冒烟测试见
tests/test_trainer_smoke.py（使用微型随机模型 + Mock 采样）。
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from ..algorithms.advantage import compute_group_advantages
from ..algorithms.dapo_loss import (
    DAPOConfig,
    dynamic_sampling_mask,
    shape_overlong_reward,
    to_grpo_loss_config,
)
from ..algorithms.grpo_loss import compute_grpo_loss
from ..data.dataset import load_train_and_test
from ..data.prompt import build_prompt
from ..model_loader import load_model, load_tokenizer, resolve_device
from ..rewards.reward import RewardConfig, compute_reward
from ..rollout.buffer import PromptGroup, RolloutBuffer, RolloutSample
from ..rollout.generator import HFPolicyGenerator, SamplingParams
from ..rollout.log_probs import aligned_token_log_probs
from ..utils.config import resolve_path
from ..utils.logger import TrainingLogger, plot_training_curves
from ..utils.seed import set_seed


class GRPODAPOTrainer:
    """Vanilla GRPO 与 DAPO 共用同一训练循环，由 config 中的四个开关控制差异。"""

    def __init__(self, config: dict, model=None, tokenizer=None, generator=None):
        self.config = config
        set_seed(int(config.get("seed", 42)))

        model_cfg = config["model"]
        self.device = resolve_device(model_cfg.get("device", "auto"))
        self.tokenizer = tokenizer or load_tokenizer(model_cfg.get("local_path"))
        self.model = model or load_model(model_cfg.get("local_path"), device=self.device)

        if config.get("lora", {}).get("enabled", False):
            self.model = self._apply_lora(self.model, config["lora"])
        self.model.to(self.device)

        rollout_cfg = config["rollout"]
        self.num_generations = int(rollout_cfg["num_generations"])
        self.sampling_params = SamplingParams(
            num_generations=self.num_generations,
            max_new_tokens=int(rollout_cfg["max_new_tokens"]),
            temperature=float(rollout_cfg["temperature"]),
            top_p=float(rollout_cfg["top_p"]),
            top_k=int(rollout_cfg["top_k"]),
        )
        self.generator = generator or HFPolicyGenerator(
            self.model, self.tokenizer, self.sampling_params, device=self.device
        )

        algorithm = config["algorithm"]
        reward_cfg = config["reward"]
        overlong_cfg = reward_cfg.get("overlong", {})
        self.reward_config = RewardConfig(
            correct_reward=float(reward_cfg.get("correct_reward", 1.0)),
            incorrect_reward=float(reward_cfg.get("incorrect_reward", 0.0)),
            format_weight=float(reward_cfg.get("format_weight", 0.0)),
        )
        self.dapo_config = DAPOConfig(
            use_clip_higher=bool(algorithm.get("use_clip_higher", False)),
            clip_ratio_low=float(algorithm.get("clip_ratio_low", 0.2)),
            clip_ratio_high=float(algorithm.get("clip_ratio_high", 0.2)),
            use_dynamic_sampling=bool(algorithm.get("use_dynamic_sampling", False)),
            max_sampling_rounds=int(algorithm.get("max_sampling_rounds", 10)),
            use_token_level_loss=bool(algorithm.get("use_token_level_loss", False)),
            use_overlong_shaping=bool(algorithm.get("use_overlong_shaping", False)),
            overlong_max_length=int(
                overlong_cfg.get("max_response_length", config["data"]["max_response_length"])
            ),
            overlong_soft_cache=int(overlong_cfg.get("soft_cache_length", 128)),
            overlong_penalty_scale=float(overlong_cfg.get("penalty_scale", 1.0)),
            kl_coef=float(algorithm.get("kl_coef", 0.0)),
            kl_estimator=str(algorithm.get("kl_estimator", "k3")),
            advantage_eps=float(algorithm.get("advantage_eps", 1e-4)),
        )
        self.loss_config = to_grpo_loss_config(self.dapo_config)

        self.ref_model = None
        if self.loss_config.kl_coef != 0.0:
            # Reference Model 与策略同尺寸，注意显存/内存翻倍
            self.ref_model = copy.deepcopy(self.model).eval()
            for parameter in self.ref_model.parameters():
                parameter.requires_grad_(False)

        data_cfg = config["data"]
        self.train_dataset, self.test_dataset = load_train_and_test(
            resolve_path(data_cfg["train_file"]), resolve_path(data_cfg["test_file"])
        )
        self._cursor = 0
        self._step_index = 0
        self._rollout_attempts = 0

        training_cfg = config["training"]
        self.max_steps = int(training_cfg["max_steps"])
        self.prompts_per_batch = int(training_cfg.get("prompts_per_batch", 1))
        self.max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))

        trainable = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if not trainable:
            raise RuntimeError("模型没有可训练参数（检查 LoRA 配置）")
        self.optimizer = AdamW(
            trainable,
            lr=float(training_cfg["learning_rate"]),
            betas=tuple(training_cfg.get("betas", (0.9, 0.999))),
            eps=float(training_cfg.get("adam_eps", 1e-8)),
            weight_decay=float(training_cfg.get("weight_decay", 0.0)),
        )
        warmup_steps = int(self.max_steps * float(training_cfg.get("warmup_ratio", 0.0)))
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: (step + 1) / max(1, warmup_steps) if step < warmup_steps else 1.0,
        )

        self.output_dir = resolve_path(training_cfg["output_dir"])
        self.log_every = int(training_cfg.get("log_every", 1))
        self.save_every = int(training_cfg.get("save_every", 0))
        self.logger = TrainingLogger(self.output_dir / "logs", use_tensorboard=False)

    @staticmethod
    def _apply_lora(model, lora_cfg: dict):
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=int(lora_cfg["r"]),
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg.get("dropout", 0.0)),
            target_modules=list(lora_cfg["target_modules"]),
            task_type="CAUSAL_LM",
        )
        return get_peft_model(model, lora_config)

    def rollout_group(self, problem) -> PromptGroup:
        """对一道题采样 G 个回答，计算 reward（含可选 Eq.13 超长惩罚）。"""
        prompt_text = build_prompt(problem.question, self.tokenizer)
        outputs = self.generator.generate(prompt_text, self.sampling_params)
        group = PromptGroup(
            prompt_ids=outputs[0].prompt_ids,
            prompt_text=prompt_text,
            question=problem.question,
            ground_truth=problem.ground_truth,
        )
        for output in outputs:
            reward_result = compute_reward(
                output.completion_text, problem.ground_truth, config=self.reward_config
            )
            reward = reward_result.reward
            if self.dapo_config.use_overlong_shaping:
                reward = shape_overlong_reward(
                    reward, len(output.completion_ids), self.dapo_config
                )
            group.add_sample(
                RolloutSample(
                    response_ids=output.completion_ids,
                    response_text=output.completion_text,
                    reward=reward,
                    correct=reward_result.correct,
                    format_ok=reward_result.format_ok,
                    answer=reward_result.answer,
                    truncated=output.truncated,
                )
            )
        return group

    def collect_rollout_batch(self) -> RolloutBuffer:
        """按 Dynamic Sampling 规则收集足够多“非全对/非全错”的题组。"""
        buffer = RolloutBuffer()
        required = self.prompts_per_batch
        max_attempts = required * max(self.dapo_config.max_sampling_rounds, 1)
        attempts = 0

        while buffer.num_groups < required and attempts < max_attempts:
            problem = self.train_dataset[self._cursor % len(self.train_dataset)]
            self._cursor += 1
            attempts += 1
            group = self.rollout_group(problem)

            valid = True
            if self.dapo_config.use_dynamic_sampling:
                flags = torch.tensor(
                    [float(sample.correct) for sample in group.samples]
                ).unsqueeze(0)
                valid = bool(dynamic_sampling_mask(flags, self.num_generations)[0])
            if valid:
                buffer.add_group(group)

        self._rollout_attempts = attempts
        if buffer.num_groups == 0:
            raise RuntimeError(
                "Dynamic Sampling 未收集到有效题组（全部全对或全错），"
                "可增大 max_sampling_rounds 或更换训练数据"
            )
        if buffer.num_groups < required:
            print(
                f"[warn] 达到最大采样轮数，仅收集到 "
                f"{buffer.num_groups}/{required} 个有效题组"
            )
        return buffer

    def _prepare_old_log_probs(self, buffer: RolloutBuffer) -> dict:
        """用采样时的策略计算 old log prob（可选 Reference Model）。"""
        pad_token_id = self.tokenizer.pad_token_id
        batch = buffer.to_batch(pad_token_id=pad_token_id if pad_token_id is not None else 0)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        self.model.eval()
        with torch.inference_mode():
            batch["old_log_probs"] = aligned_token_log_probs(
                self.model, input_ids, attention_mask
            ).float().cpu()
            if self.ref_model is not None:
                batch["ref_log_probs"] = aligned_token_log_probs(
                    self.ref_model, input_ids, attention_mask
                ).float().cpu()
        self.model.train()
        return batch

    def train_step(self) -> dict[str, float]:
        """一次完整更新：Rollout → 优势 → 前向 → 反传 → 优化器步。"""
        start = time.perf_counter()
        buffer = self.collect_rollout_batch()
        batch = self._prepare_old_log_probs(buffer)

        rewards = buffer.rewards()
        group_count = buffer.num_groups
        if buffer.num_samples != group_count * self.num_generations:
            raise RuntimeError("每组回答数不等于 num_generations")
        advantage_result = compute_group_advantages(
            rewards.view(group_count, self.num_generations),
            self.num_generations,
            eps=self.dapo_config.advantage_eps,
        )
        advantages = advantage_result.advantages.reshape(-1)

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        response_mask = batch["response_mask"].to(self.device)
        old_log_probs = batch["old_log_probs"].to(self.device)
        ref_log_probs = (
            None if batch["ref_log_probs"] is None
            else batch["ref_log_probs"].to(self.device)
        )

        policy_log_probs = aligned_token_log_probs(self.model, input_ids, attention_mask)
        metrics = compute_grpo_loss(
            policy_log_probs=policy_log_probs,
            old_log_probs=old_log_probs,
            advantages=advantages.to(self.device),
            response_mask=response_mask,
            ref_log_probs=ref_log_probs,
            config=self.loss_config,
        )

        metrics.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], self.max_grad_norm
        )
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)

        self._step_index += 1
        record = metrics.as_dict()
        record.update(
            reward_mean=float(rewards.mean()),
            reward_std=float(rewards.std(unbiased=False)),
            zero_variance_ratio=float(advantage_result.zero_variance.float().mean()),
            avg_response_length=float(batch["response_lengths"].float().mean()),
            effective_sample_ratio=buffer.num_groups / max(1, self._rollout_attempts),
            grad_norm=float(grad_norm),
            step_time=time.perf_counter() - start,
        )
        self.logger.log(self._step_index, record)
        return record

    def save_checkpoint(self, step: int) -> Path:
        checkpoint_dir = self.output_dir / "checkpoints" / f"step_{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(checkpoint_dir)
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(checkpoint_dir)
        torch.save(
            {"optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict()},
            checkpoint_dir / "training_state.pt",
        )
        with (checkpoint_dir / "config.yaml").open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.config, handle, allow_unicode=True, sort_keys=False)
        print(f"检查点已保存：{checkpoint_dir}")
        return checkpoint_dir

    def train(self) -> list[dict[str, float]]:
        for step in range(1, self.max_steps + 1):
            self.train_step()
            if self.save_every > 0 and step % self.save_every == 0:
                self.save_checkpoint(step)

        plot_training_curves(self.logger.history, self.output_dir / "plots")
        self.logger.save()
        self.logger.close()
        print(f"训练结束：{self.max_steps} 步，日志位于 {self.output_dir}")
        return self.logger.history
