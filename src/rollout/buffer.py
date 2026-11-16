"""Rollout 数据缓存：保存一道题的 G 个回答及其奖励 / 旧策略 log prob。

张量对齐约定（to_batch 输出，右 padding）：
- input_ids:        [N, L]   prompt + response
- attention_mask:   [N, L]   有效 token 为 1，padding 为 0
- response_mask:    [N, L]   response token 位置为 1
- old_log_probs:    [N, L]   与 input_ids 逐 token 对齐；非 response 位置为 0
- ref_log_probs:    [N, L]   同上（未启用 Reference Model 时全 0）
因此任意样本第 t 个位置的 ratio 可用 exp(policy[:,t] - old[:,t]) 直接计算。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class RolloutSample:
    response_ids: list[int]
    response_text: str
    reward: float
    correct: bool
    format_ok: bool
    answer: str | None
    old_log_probs: torch.Tensor | None = None
    ref_log_probs: torch.Tensor | None = None
    truncated: bool = False


@dataclass
class PromptGroup:
    prompt_ids: list[int]
    prompt_text: str
    question: str
    ground_truth: str
    samples: list[RolloutSample] = field(default_factory=list)

    def add_sample(self, sample: RolloutSample) -> None:
        response_length = len(sample.response_ids)
        for name in ("old_log_probs", "ref_log_probs"):
            value = getattr(sample, name)
            if value is not None and value.numel() != response_length:
                raise ValueError(
                    f"{name} 长度（{value.numel()}）与 response 长度（{response_length}）不一致"
                )
        self.samples.append(sample)


class RolloutBuffer:
    """一组题目（每个题目 G 个回答）的容器。"""

    def __init__(self) -> None:
        self.groups: list[PromptGroup] = []

    def add_group(self, group: PromptGroup) -> None:
        if not group.samples:
            raise ValueError("PromptGroup 至少需要一个回答")
        self.groups.append(group)

    def clear(self) -> None:
        self.groups.clear()

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    @property
    def num_samples(self) -> int:
        return sum(len(group.samples) for group in self.groups)

    def rewards(self) -> torch.Tensor:
        """[N] 按 flatten 顺序（group 优先）排列的奖励。"""
        return torch.tensor(
            [sample.reward for group in self.groups for sample in group.samples],
            dtype=torch.float32,
        )

    def correctness(self) -> torch.Tensor:
        return torch.tensor(
            [float(sample.correct) for group in self.groups for sample in group.samples],
            dtype=torch.float32,
        )

    def to_batch(self, pad_token_id: int = 0) -> dict[str, torch.Tensor]:
        """把全部样本 padding 成训练 batch；返回的 log prob 已按 token 位置对齐。"""
        entries = [
            (group_index, group, sample)
            for group_index, group in enumerate(self.groups)
            for sample in group.samples
        ]
        if not entries:
            raise ValueError("RolloutBuffer 为空")

        max_length = max(
            len(group.prompt_ids) + len(sample.response_ids)
            for _, group, sample in entries
        )

        num_samples = len(entries)
        input_ids = torch.full((num_samples, max_length), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((num_samples, max_length), dtype=torch.float32)
        response_mask = torch.zeros((num_samples, max_length), dtype=torch.float32)
        old_log_probs = torch.zeros((num_samples, max_length), dtype=torch.float32)
        ref_log_probs = torch.zeros((num_samples, max_length), dtype=torch.float32)
        rewards = torch.zeros(num_samples, dtype=torch.float32)
        correctness = torch.zeros(num_samples, dtype=torch.float32)
        group_ids = torch.zeros(num_samples, dtype=torch.long)
        prompt_lengths = torch.zeros(num_samples, dtype=torch.long)
        response_lengths = torch.zeros(num_samples, dtype=torch.long)
        truncated = torch.zeros(num_samples, dtype=torch.bool)
        has_ref_log_probs = any(sample.ref_log_probs is not None for _, _, sample in entries)

        for row, (group_index, group, sample) in enumerate(entries):
            prompt_length = len(group.prompt_ids)
            response_length = len(sample.response_ids)
            sequence = group.prompt_ids + sample.response_ids
            length = len(sequence)

            input_ids[row, :length] = torch.tensor(sequence, dtype=torch.long)
            attention_mask[row, :length] = 1.0
            response_mask[row, prompt_length:length] = 1.0
            prompt_lengths[row] = prompt_length
            response_lengths[row] = response_length
            rewards[row] = sample.reward
            correctness[row] = float(sample.correct)
            group_ids[row] = group_index
            truncated[row] = sample.truncated

            if sample.old_log_probs is not None:
                old_log_probs[row, prompt_length:length] = sample.old_log_probs.float()
            if sample.ref_log_probs is not None:
                ref_log_probs[row, prompt_length:length] = sample.ref_log_probs.float()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "old_log_probs": old_log_probs,
            "ref_log_probs": ref_log_probs if has_ref_log_probs else None,
            "rewards": rewards,
            "correctness": correctness,
            "group_ids": group_ids,
            "prompt_lengths": prompt_lengths,
            "response_lengths": response_lengths,
            "truncated": truncated,
        }
