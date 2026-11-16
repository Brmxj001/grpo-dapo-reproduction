"""计算给定序列的逐 Token Log Probability，供 PPO / GRPO 比率与 KL 使用。

位置约定（必须与 buffer 中的 response_mask 对齐）：
- 模型 forward 的 logits[:, t] 预测 token t+1；
- compute_token_log_probs 返回 [B, T-1]，第 i 列是 token i+1 的 log prob；
- aligned_token_log_probs 左移一列补 0，返回 [B, T]，
  即 aligned[:, t] = log π(x_t | x_<t)，t >= 1 才有意义。
"""

from __future__ import annotations

import torch


def compute_token_log_probs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """返回 [B, T-1] 的逐 token log prob（第 i 列对应 token i+1）。"""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :].float()
    labels = input_ids[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def aligned_token_log_probs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """返回 [B, T]，aligned[:, t] = log π(input_ids[:, t] | 前文)，t=0 位置填 0。"""
    token_log_probs = compute_token_log_probs(model, input_ids, attention_mask)
    aligned = torch.zeros(
        input_ids.shape, dtype=token_log_probs.dtype, device=token_log_probs.device
    )
    aligned[:, 1:] = token_log_probs
    return aligned


def masked_mean(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """按 mask 求均值（mask 需为 0/1 浮点或布尔张量）。"""
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp(min=eps)
