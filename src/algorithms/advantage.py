"""组内相对优势估计（GRPO 论文 Eq.4；DAPO 论文 Eq.9）。

A_i = (r_i - mean(r_1..r_G)) / (std(r_1..r_G) + eps)

实现约定：
- std 默认使用总体标准差（unbiased=False）；
- 原始论文公式不含 eps，这里加入 eps 仅作数值稳定，组方差为 0 时优势显式置 0；
- 所有回答共享同一个标量优势（Outcome Supervision）。
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
    unbiased_std: bool = False,
) -> AdvantageResult:
    """按组计算相对优势。

    Args:
        rewards: 展平的奖励 [B*G] 或已分组的 [B, G]。
        num_generations: 每题采样数 G。
        eps: 标准差数值稳定项。
    """
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
        stds = rewards.std(dim=1, unbiased=unbiased_std, keepdim=True)
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


def expand_advantages(
    advantages: torch.Tensor, response_mask: torch.Tensor
) -> torch.Tensor:
    """把 [N] 每条回答的标量优势扩展到每个 response token。"""
    if advantages.dim() == 1:
        if advantages.numel() != response_mask.size(0):
            raise ValueError("优势数量与 batch 大小不一致")
        # 非 response 位置显式置 0，避免 prompt / padding 被赋予优势
        return advantages.unsqueeze(1).expand_as(response_mask) * response_mask
    if advantages.shape != response_mask.shape:
        raise ValueError("逐 token 优势的形状必须与 response_mask 一致")
    return advantages
