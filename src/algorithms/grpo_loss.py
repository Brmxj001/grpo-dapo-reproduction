"""Vanilla GRPO 损失（DeepSeekMath 论文 Eq.3 / Eq.21）。

目标（最大化）：
    J = E[ 1/G Σ_i 1/|o_i| Σ_t min( ρ_{i,t} A_i,
              clip(ρ_{i,t}, 1-ε, 1+ε) A_i ) - β KL(πθ || πref) ]
    ρ_{i,t} = πθ(o_{i,t}|q,o_{<t}) / πθ_old(o_{i,t}|q,o_{<t})

代码优化的是 -J，因此 loss 的符号为：
    loss = -policy_loss + β * kl_loss

归一化方式：
- "sequence"（GRPO 原始做法）：先对每条回答的 response token 求平均，再对回答求平均；
- "token"（DAPO 做法）：对整个 batch 的所有 response token 统一求平均。

本模块只做纯张量计算，不依赖模型，便于单元测试。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .advantage import expand_advantages
from .kl import kl_estimator


@dataclass
class GRPOLossConfig:
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    kl_coef: float = 0.0
    kl_estimator: str = "k3"
    normalization: str = "sequence"   # sequence | token
    eps: float = 1e-8                 # 归一化防除零

    def __post_init__(self) -> None:
        if not 0.0 <= self.clip_ratio_low < 1.0:
            raise ValueError("clip_ratio_low 必须在 [0, 1) 内")
        if self.clip_ratio_high < 0.0:
            raise ValueError("clip_ratio_high 必须 >= 0")
        if self.normalization not in ("sequence", "token"):
            raise ValueError("normalization 只能是 sequence 或 token")


@dataclass
class LossMetrics:
    loss: torch.Tensor
    policy_loss: torch.Tensor
    kl: torch.Tensor
    clip_fraction: torch.Tensor
    ratio_mean: torch.Tensor
    num_response_tokens: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        return {
            "loss": float(self.loss.detach()),
            "policy_loss": float(self.policy_loss.detach()),
            "kl": float(self.kl.detach()),
            "clip_fraction": float(self.clip_fraction.detach()),
            "ratio_mean": float(self.ratio_mean.detach()),
            "num_response_tokens": float(self.num_response_tokens.detach()),
        }


def _masked_reduce(
    values: torch.Tensor,
    mask: torch.Tensor,
    normalization: str,
    eps: float,
) -> torch.Tensor:
    """按 sequence 或 token 方式对 response token 求平均。"""
    if normalization == "sequence":
        per_sequence = (values * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return per_sequence.mean()
    if normalization == "token":
        return (values * mask).sum() / mask.sum().clamp(min=eps)
    raise ValueError(f"未知归一化方式：{normalization}")


def compute_grpo_loss(
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    ref_log_probs: torch.Tensor | None = None,
    config: GRPOLossConfig | None = None,
) -> LossMetrics:
    """计算 GRPO 损失与诊断指标。

    Args:
        policy_log_probs: [N, L] 当前策略逐 token log prob（与 input_ids 对齐）。
        old_log_probs: [N, L] 采样时旧策略的 log prob（无梯度）。
        advantages: [N] 每回答标量优势，或 [N, L] 逐 token 优势。
        response_mask: [N, L] response token 为 1、其余为 0。
        ref_log_probs: [N, L] Reference Model log prob；kl_coef>0 时必填。
        config: 损失配置。
    """
    cfg = config or GRPOLossConfig()
    if policy_log_probs.shape != old_log_probs.shape:
        raise ValueError("policy_log_probs 与 old_log_probs 形状不一致")
    if policy_log_probs.shape != response_mask.shape:
        raise ValueError("policy_log_probs 与 response_mask 形状不一致")

    mask = response_mask.to(policy_log_probs.dtype)
    advantages = expand_advantages(advantages, response_mask).to(policy_log_probs.dtype)

    # 重要性采样比率 ρ = πθ / πold
    ratio = torch.exp(policy_log_probs - old_log_probs)
    clipped_ratio = torch.clamp(
        ratio, 1.0 - cfg.clip_ratio_low, 1.0 + cfg.clip_ratio_high
    )
    # PPO/GRPO 的悲观下界：min(ρA, clip(ρ)A)
    surrogate = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    policy_loss = _masked_reduce(surrogate, mask, cfg.normalization, cfg.eps)

    kl = torch.zeros((), dtype=policy_log_probs.dtype, device=policy_log_probs.device)
    if cfg.kl_coef != 0.0:
        if ref_log_probs is None:
            raise ValueError("kl_coef != 0 时必须提供 ref_log_probs")
        kl_values = kl_estimator(policy_log_probs, ref_log_probs, cfg.kl_estimator)
        kl = _masked_reduce(kl_values, mask, cfg.normalization, cfg.eps)

    loss = -policy_loss + cfg.kl_coef * kl

    # 诊断指标统一用 token 级平均
    clip_indicator = ((ratio < 1.0 - cfg.clip_ratio_low) |
                      (ratio > 1.0 + cfg.clip_ratio_high)).to(policy_log_probs.dtype)
    clip_fraction = _masked_reduce(clip_indicator, mask, "token", cfg.eps)
    ratio_mean = _masked_reduce(ratio, mask, "token", cfg.eps)
    num_response_tokens = mask.sum()

    return LossMetrics(
        loss=loss,
        policy_loss=policy_loss,
        kl=kl,
        clip_fraction=clip_fraction,
        ratio_mean=ratio_mean,
        num_response_tokens=num_response_tokens,
    )
