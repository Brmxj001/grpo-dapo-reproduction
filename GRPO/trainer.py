"""最小 GRPO 训练循环（独立版，刻意保持简单）。

只做一件事：采样 → 奖励 → 组内优势 → 旧策略 log prob → GRPO loss →
反向传播 → 优化器更新 → 记录指标。

不包含：LoRA、梯度累积、LR 调度、分布式训练、评测。真实训练需 GPU。
模型加载、Prompt 与奖励逻辑复用项目已有的 src/ 模块，避免重复实现。
"""

from __future__ import annotations

import copy
import time

import pandas as pd
import torch
from torch.optim import AdamW

from .advantage import compute_group_advantages
from .loss import GRPOLossConfig, compute_grpo_loss
from .metrics import MetricsLogger, plot_metrics, reward_statistics
from .rollout import (
    HFGenerator,
    PromptGroup,
    RolloutBuffer,
    RolloutSample,
    SamplingParams,
    aligned_token_log_probs,
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
            # 计算 KL 需要参考模型；与策略同尺寸，显存/内存约翻倍
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
        self.max_grad_norm = float(training_cfg.get("max_grad_norm", 1.0))
        self.save_every = int(training_cfg.get("save_every", 0))
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=float(training_cfg.get("learning_rate", 1e-6)),
        )

        self.output_dir = resolve_path(training_cfg.get("output_dir", "outputs/GRPO"))
        self.logger = MetricsLogger(self.output_dir / "logs")
        self._cursor = 0
        self._step = 0

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
            question, ground_truth = self.problems[self._cursor % len(self.problems)]
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
        """一次更新：Rollout → 优势 → 前向 → 反传 → 优化器步。"""
        start = time.perf_counter()
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

        policy_log_probs = aligned_token_log_probs(
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
            response_mask=batch["response_mask"].to(self.device),
            ref_log_probs=ref_log_probs,
            config=self.loss_config,
        )

        output.loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], self.max_grad_norm
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        self._step += 1
        metrics = output.as_dict()
        metrics.update(reward_statistics(rewards.tolist()))
        metrics.update(
            zero_variance_ratio=float(advantage_result.zero_variance.float().mean()),
            avg_response_length=float(batch["response_lengths"].float().mean()),
            grad_norm=float(grad_norm),
            step_time=time.perf_counter() - start,
        )
        self.logger.log(self._step, metrics)
        return metrics

    def save_checkpoint(self, step: int) -> None:
        path = self.output_dir / "checkpoints" / f"step_{step}"
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        if hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(path)
        print(f"检查点已保存：{path}")

    def train(self) -> list[dict]:
        for step in range(1, self.max_steps + 1):
            self.train_step()
            if self.save_every > 0 and step % self.save_every == 0:
                self.save_checkpoint(step)
        plot_metrics(self.logger.history, self.output_dir / "plots")
        self.logger.save()
        self.logger.close()
        print(f"训练结束：{self.max_steps} 步，日志位于 {self.output_dir}")
        return self.logger.history
