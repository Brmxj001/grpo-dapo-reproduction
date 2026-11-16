"""KL 散度估计。

GRPO 论文（DeepSeekMath, Eq.21 附近）采用 Schulman (2020) 的无偏估计器 k3：

    KL(πθ || πref) ≈ πref/πθ - log(πref/πθ) - 1

其中比值为逐 token 概率比，实际实现用 log prob 计算：
    log_ratio = log πref - log πθ
    k3 = exp(log_ratio) - log_ratio - 1
k3 恒非负，适合作为逐 token 惩罚项。
"""

from __future__ import annotations

import torch


def kl_estimator(
    policy_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    kind: str = "k3",
) -> torch.Tensor:
    """返回逐 token KL 估计值（未乘 mask）。"""
    if policy_log_probs.shape != ref_log_probs.shape:
        raise ValueError("policy_log_probs 与 ref_log_probs 形状不一致")
    log_ratio = ref_log_probs - policy_log_probs

    if kind == "k3":
        return torch.exp(log_ratio) - log_ratio - 1.0
    if kind == "k1":
        # 采样估计 log πθ - log πref
        return -log_ratio
    if kind == "k2":
        return 0.5 * log_ratio.pow(2)
    raise ValueError(f"未知的 KL 估计器：{kind}（可选 k1/k2/k3）")
