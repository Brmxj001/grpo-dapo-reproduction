"""组内相对优势（GRPO 论文 Eq.4）。

A_i = (r_i - mean(r_1..r_G)) / (std(r_1..r_G) + eps)

约定：
- 标准差使用总体标准差（unbiased=False）；
- 原始论文公式不含 eps，这里加入 eps 仅为数值稳定；
- 组内奖励完全相同时优势显式置 0。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AdvantageResult:
    advantages: torch.Tensor      # [B, G]
    means: torch.Tensor           # [B]
    stds: torch.Tensor            # [B]
    zero_variance: torch.Tensor   # [B] bool


def compute_group_advantages(
    rewards: torch.Tensor | list[float],
    num_generations: int,
    eps: float = 1e-4,
) -> AdvantageResult:
    """rewards 支持 [B*G] 或 [B, G]。"""
    if num_generations < 1:
        raise ValueError("num_generations 必须 >= 1")

    rewards = torch.as_tensor(rewards, dtype=torch.float32)
    if rewards.dim() == 1:
        if rewards.numel() % num_generations != 0:
            raise ValueError("奖励数量不能被 num_generations 整除")
        rewards = rewards.view(-1, num_generations)
    elif rewards.dim() == 2:
        if rewards.shape[1] != num_generations:
            raise ValueError("rewards 第二维与 num_generations 不一致")
    else:
        raise ValueError("rewards 必须是一维或二维张量")
    if rewards.shape[0] == 0:
        raise ValueError("rewards 不能为空")

    means = rewards.mean(dim=1, keepdim=True)
    if num_generations > 1:
        stds = rewards.std(dim=1, unbiased=False, keepdim=True)
    else:
        stds = torch.zeros_like(means)

    advantages = (rewards - means) / (stds + eps)
    zero_variance = (rewards.max(dim=1).values - rewards.min(dim=1).values) <= eps
    advantages = torch.where(zero_variance.unsqueeze(1), torch.zeros_like(advantages), advantages)

    return AdvantageResult(
        advantages=advantages,
        means=means.squeeze(1),
        stds=stds.squeeze(1),
        zero_variance=zero_variance,
    )
