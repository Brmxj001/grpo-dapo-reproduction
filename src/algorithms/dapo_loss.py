"""DAPO 四项优化（论文 Eq.8 - Eq.13），每项可独立开关。

① Clip-Higher（Eq.10）：非对称裁剪 [1-ε_low, 1+ε_high]，ε_high 通常取 0.28；
② Dynamic Sampling（Eq.11）：过滤组内全部正确 / 全部错误的题目；
③ Token-level Policy Gradient Loss（Eq.12）：以 Σ_i |o_i| 为分母统一归一化；
④ Overlong Reward Shaping（Eq.13）：长度软惩罚，加到规则奖励上。

论文 §2.3 明确移除了 KL 惩罚，因此默认 kl_coef = 0。

说明：Dynamic Sampling 与 Overlong Reward Shaping 不参与 loss 张量计算——
前者作用于 Rollout 采样阶段，后者作用于奖励计算阶段。本模块提供二者的
独立接口，保证四项优化可以分别启用/关闭。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ..rewards.reward import OverlongPenaltyConfig, overlong_penalty
from .advantage import compute_group_advantages
from .grpo_loss import GRPOLossConfig, LossMetrics, compute_grpo_loss


@dataclass
class DAPOConfig:
    """DAPO 四项优化开关与超参数。"""

    # ① Clip-Higher
    use_clip_higher: bool = True
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.28

    # ② Dynamic Sampling
    use_dynamic_sampling: bool = True
    max_sampling_rounds: int = 10

    # ③ Token-level Policy Gradient Loss
    use_token_level_loss: bool = True

    # ④ Overlong Reward Shaping
    use_overlong_shaping: bool = True
    overlong_max_length: int = 512
    overlong_soft_cache: int = 128
    overlong_penalty_scale: float = 1.0

    # 奖励/优势/KL
    kl_coef: float = 0.0
    kl_estimator: str = "k3"
    advantage_eps: float = 1e-4


def to_grpo_loss_config(config: DAPOConfig) -> GRPOLossConfig:
    """把 DAPO 开关映射为底层 Loss 配置。

    关闭 Clip-Higher 时退化为对称裁剪 [1-ε_low, 1+ε_low]（即使用 ε_low 作 ε）。
    """
    clip_high = config.clip_ratio_high if config.use_clip_higher else config.clip_ratio_low
    normalization = "token" if config.use_token_level_loss else "sequence"
    return GRPOLossConfig(
        clip_ratio_low=config.clip_ratio_low,
        clip_ratio_high=clip_high,
        kl_coef=config.kl_coef,
        kl_estimator=config.kl_estimator,
        normalization=normalization,
    )


def compute_dapo_loss(
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    ref_log_probs: torch.Tensor | None = None,
    config: DAPOConfig | None = None,
) -> LossMetrics:
    """按 DAPO 配置计算损失（四项优化由 config 独立控制）。"""
    cfg = config or DAPOConfig()
    return compute_grpo_loss(
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        response_mask=response_mask,
        ref_log_probs=ref_log_probs,
        config=to_grpo_loss_config(cfg),
    )


def dynamic_sampling_mask(
    correct_flags: torch.Tensor | Sequence[float] | Sequence[bool],
    num_generations: int,
    eps: float = 1e-4,
) -> torch.Tensor:
    """DAPO Eq.11 的过滤条件：保留 0 < 答对数 < G 的组。

    返回 [B] 布尔张量，True 表示该题可用于训练。
    """
    flags = torch.as_tensor(correct_flags, dtype=torch.float32)
    result = compute_group_advantages(flags, num_generations, eps=eps)
    return ~result.zero_variance


def shape_overlong_reward(
    reward: float, response_length: int, config: DAPOConfig
) -> float:
    """对单条样本应用 DAPO Eq.13 软惩罚；未启用时原样返回。"""
    if not config.use_overlong_shaping:
        return reward
    penalty_config = OverlongPenaltyConfig(
        max_response_length=config.overlong_max_length,
        soft_cache_length=config.overlong_soft_cache,
        penalty_scale=config.overlong_penalty_scale,
    )
    return reward + overlong_penalty(response_length, penalty_config)


def shape_overlong_rewards(
    rewards: Sequence[float],
    response_lengths: Sequence[int],
    config: DAPOConfig,
) -> list[float]:
    """批量应用 Eq.13；长度必须一一对应。"""
    if len(rewards) != len(response_lengths):
        raise ValueError("rewards 与 response_lengths 数量不一致")
    return [
        shape_overlong_reward(reward, length, config)
        for reward, length in zip(rewards, response_lengths)
    ]
