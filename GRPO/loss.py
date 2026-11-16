"""Vanilla GRPO 损失实现（独立版）。

本文件只做纯张量计算，不加载模型，方便单元测试和逐行阅读。
公式核对自 DeepSeekMath 论文（GRPO）Eq.3 / 附录 Eq.21。

======================================================================
一、优化目标（论文最大化形式）
======================================================================

    J(θ) = E[ (1/G) Σ_i (1/|o_i|) Σ_t
              min( ρ_{i,t} · A_i , clip(ρ_{i,t}, 1-ε, 1+ε) · A_i )
            - β · KL(πθ || πref) ]

其中：
- G：每道题采样几个回答（num_generations）；
- |o_i|：第 i 条回答的 token 数；
- ρ_{i,t} = πθ(o_{i,t} | q, o_{i,<t}) / πold(o_{i,t} | q, o_{i,<t})，
  是“新策略概率 / 采样时旧策略概率”的比值；
- A_i：组内相对优势，GRPO 的 outcome supervision 下，
  同一条回答的所有 token 共享同一个标量优势；
- β：KL 惩罚系数，论文实验取 0.04；
- KL 使用 k3 无偏估计（见 kl_k3 的注释）。

PyTorch 优化器做的是“最小化”，所以代码里实际优化 -J：

    loss = -policy_loss + kl_coef * kl

======================================================================
二、张量形状约定
======================================================================

- policy_log_probs / old_log_probs / ref_log_probs: [N, L]
    N = 题目数 B × 每题回答数 G，即一个 update 里的回答总条数；
    L = prompt + response 经过 padding 后的统一长度。
- response_mask: [N, L]
    response token 所在位置为 1，prompt / padding 位置为 0。
- advantages: [N] 或 [N, L]
    [N]：每条回答一个标量优势（本项目默认，Outcome Supervision）；
    [N, L]：逐 token 优势（为 process supervision 等扩展保留）。
- 所有 log prob 都已经按 token 位置对齐：
    第 t 列 = token t 的产生 log prob（第 0 列填 0），
    由 rollout.aligned_token_log_probs() 完成。

======================================================================
三、两种归一化方式
======================================================================

- sequence（GRPO 原始做法）：
    先对每条回答自己的 response token 求平均，再对 N 条回答求平均。
    每条回答等权；因此长回答里的单个 token 权重更小。
- token（DAPO 的 Token-level Policy Gradient Loss 做法）：
    把整个 batch 的所有 response token 放在一起求平均。
    每个有效 token 等权；长回答贡献更大。

本独立版默认 sequence，可通过 GRPOLossConfig.normalization 切换。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class GRPOLossConfig:
    """GRPO 损失超参数。"""

    # PPO 的裁剪范围 ε：ratio 被限制在 [1-ε, 1+ε]。
    # 论文取 0.2；这也是“信任域”的宽度。
    clip_ratio: float = 0.2

    # KL 惩罚系数 β。论文实验取 0.04。
    # 设为 0 时完全不需要 Reference Model，也能省一份模型显存。
    kl_coef: float = 0.04

    # KL 估计器类型。本文件只实现 k3（GRPO 论文采用的估计器），
    # 保留该字段是为了将来对比 k1/k2 时不用改调用方。
    kl_estimator: str = "k3"

    # 归一化方式：
    # "sequence" -> GRPO 论文的 1/G Σ_i 1/|o_i| Σ_t ...
    # "token"    -> DAPO 的 1/Σ|o_i| Σ_i Σ_t ...
    normalization: str = "sequence"

    # 归一化时分母的防除零项（注意：不是优势计算里的 eps）。
    eps: float = 1e-8

    def __post_init__(self) -> None:
        # 裁剪比例必须小于 1：1-ε > 0，否则下界会变成负数，
        # ratio 永远大于下界，下裁剪失去意义。
        if not 0.0 <= self.clip_ratio < 1.0:
            raise ValueError("clip_ratio 必须在 [0, 1) 内")
        if self.normalization not in ("sequence", "token"):
            raise ValueError("normalization 只能是 sequence 或 token")


@dataclass
class LossOutput:
    """损失函数返回的标量结果与诊断指标（都是 0 维张量）。"""

    # 用于 .backward() 的最终损失：loss = -policy_loss + kl_coef * kl
    loss: torch.Tensor

    # 策略梯度部分（PPO 裁剪目标的 batch 平均），不含 KL。
    # 正值表示当前代理目标为正收益；代码里对它取负号后最小化。
    policy_loss: torch.Tensor

    # batch 平均 KL；kl_coef=0 时恒为 0。
    kl: torch.Tensor

    # 被裁剪的 response token 占比：
    # ratio 落在 [1-ε, 1+ε] 之外的 token 比例。
    # 训练早期应接近 0；持续很高说明单步更新幅度过大/学习率过高。
    clip_fraction: torch.Tensor

    # 所有 response token 的平均 ratio。第一次更新时应正好为 1.0，
    # 因为此时 policy == old；偏离 1 表示策略已发生变化。
    ratio_mean: torch.Tensor

    # 参与损失的 response token 总数（不含 prompt / padding），
    # 可用于核对 mask 是否写对、以及做 token 级加权分析。
    num_response_tokens: torch.Tensor

    def as_dict(self) -> dict[str, float]:
        """转成普通 float 字典，直接交给日志系统。

        detach()：这些指标只用于展示，不能把计算图带进日志；
        float()：0 维张量转成 Python 数值。
        """
        return {
            "loss": float(self.loss.detach()),
            "policy_loss": float(self.policy_loss.detach()),
            "kl": float(self.kl.detach()),
            "clip_fraction": float(self.clip_fraction.detach()),
            "ratio_mean": float(self.ratio_mean.detach()),
            "num_response_tokens": float(self.num_response_tokens.detach()),
        }


def kl_k3(policy_log_probs: torch.Tensor, ref_log_probs: torch.Tensor) -> torch.Tensor:
    """逐 token 的 k3 KL 估计，返回与输入同形状的张量。

    推导：
    设 ratio = πref/πθ，则 k3 估计器为
        KL ≈ ratio - log(ratio) - 1
    用 log prob 表示：log_ratio = logπref - logπθ，
        KL ≈ exp(log_ratio) - log_ratio - 1

    为什么是 k3：
    - 对任意实数 x，e^x - x - 1 >= 0，在 x=0 处取等号，
      所以估计值恒非负，不会出现“负的 KL”；
    - 当 policy 与 ref 完全相同时 log_ratio=0，KL=0，符合预期；
    - GRPO 论文（Eq.21 附近）明确采用 Schulman (2020) 的这一无偏估计器。

    注意：这里的“逐 token”是指每个 response 位置单独估计，
    最后还需要用 response_mask 做平均（在 compute_grpo_loss 中完成）。
    """
    # log_ratio 的形状是 [N, L]，与两个输入一致
    log_ratio = ref_log_probs - policy_log_probs
    return torch.exp(log_ratio) - log_ratio - 1.0


def _masked_reduce(
    values: torch.Tensor,
    mask: torch.Tensor,
    normalization: str,
    eps: float,
) -> torch.Tensor:
    """把逐 token 的数值聚合成一个标量，按 response_mask 屏蔽 padding。

    Args:
        values: [N, L] 待聚合的逐 token 数值（如 surrogate、KL、ratio）。
        mask:   [N, L] 0/1 浮点 mask，response token 为 1。
        normalization: "sequence" 或 "token"，含义见文件头第三节。
        eps: 分母防除零，仅对 token 归一生效。

    两种归一化的数学形式：

    sequence：
        per_sequence_n = Σ_t values[n,t] * mask[n,t] / Σ_t mask[n,t]
        result         = (1/N) Σ_n per_sequence_n
        —— 先回答内平均，再回答间平均；每条回答等权。

    token：
        result = Σ_{n,t} values[n,t] * mask[n,t] / Σ_{n,t} mask[n,t]
        —— 所有有效 token 等权，长回答因为 token 多而权重更大。

    两种情况下 padding 位置的 mask 都是 0，既不进分子也不进分母。
    即使一条回答完全没有 response token（理论上不会发生），
    clamp(min=1.0) 也能避免除零产生 NaN。
    """
    if normalization == "sequence":
        # 每条回答的 response 平均：[N]
        per_sequence = (values * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        # 再对所有回答求平均：标量
        return per_sequence.mean()

    if normalization == "token":
        # 整个 batch 的 response 平均：标量
        return (values * mask).sum() / mask.sum().clamp(min=eps)

    raise ValueError(f"未知归一化方式：{normalization}")


def compute_grpo_loss(
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    ref_log_probs: torch.Tensor | None = None,
    config: GRPOLossConfig | None = None,
) -> LossOutput:
    """计算一次 GRPO 损失（可反向传播）。

    参数：
        policy_log_probs: [N, L] 当前策略的逐 token log prob（带梯度）。
        old_log_probs:    [N, L] 采样时旧策略的 log prob（常量，无梯度）。
        advantages:       [N] 每条回答一个标量优势，或 [N, L] 逐 token 优势。
        response_mask:    [N, L] response token 为 1、prompt/padding 为 0。
        ref_log_probs:    [N, L] Reference Model 的 log prob；
                          kl_coef != 0 时必须提供，否则 KL 无法计算。
        config:           超参数；None 时使用默认配置（ε=0.2, β=0.04）。

    返回：
        LossOutput，其中 loss 可直接 .backward()。

    ------------------------------------------------------------------
    数值示例（与 tests/test_grpo_standalone.py 的 test_loss_manual 对应）：
        ratio  = [[1.2, 0.8], [0.5, 1.0]]
        mask   = [[1,   1  ], [1,   0  ]]
        A      = [ 1.0,      -1.0 ]
        ε      = 0.2

        第 0 条（A=+1）：
            token0: min(1.2*1, clip(1.2)*1) = 1.2
            token1: min(0.8*1, clip(0.8)*1) = 0.8
        sequence 归一化：per-seq = (1.2+0.8)/2 = 1.0
        第 1 条（A=-1，mask 只有 token0）：
            ratio 0.5 被下裁剪到 0.8，
            min(0.5*-1, 0.8*-1) = min(-0.5, -0.8) = -0.8
        sequence 归一化：per-seq = -0.8

        policy_loss = (1.0 + (-0.8)) / 2 = 0.1
        loss        = -0.1

        若改用 token 归一化：
            policy_loss = (1.2 + 0.8 - 0.8) / 3 = 0.4
    """
    cfg = config or GRPOLossConfig()

    # ------------------------------------------------------------------
    # 步骤 1：形状校验。
    # 三个 [N, L] 张量必须完全对齐，否则 mask / ratio 会错位，
    # 这类错误不会报错但会静默训练出错误结果，所以必须显式检查。
    # ------------------------------------------------------------------
    if policy_log_probs.shape != old_log_probs.shape:
        raise ValueError("policy_log_probs 与 old_log_probs 形状不一致")
    if policy_log_probs.shape != response_mask.shape:
        raise ValueError("policy_log_probs 与 response_mask 形状不一致")

    # mask 转成和 log prob 相同的浮点类型，便于逐元素相乘。
    mask = response_mask.to(policy_log_probs.dtype)

    # ------------------------------------------------------------------
    # 步骤 2：把优势扩展到 [N, L]。
    #
    # GRPO 默认只有“整条回答”的标量优势 A_i（Outcome Supervision），
    # 同一条回答里每个生成 token 共享同一个 A_i。
    # 扩展后乘 mask：prompt / padding 位置的优势显式置 0，
    # 防止它们通过 unmasked 位置偷偷产生梯度。
    # ------------------------------------------------------------------
    if advantages.dim() == 1:
        if advantages.numel() != mask.size(0):
            raise ValueError("优势数量与 batch 大小不一致")
        adv = advantages.unsqueeze(1).expand_as(mask) * mask
    elif advantages.shape == mask.shape:
        adv = advantages
    else:
        raise ValueError("优势形状必须为 [N] 或 [N, L]")
    adv = adv.to(policy_log_probs.dtype)

    # ------------------------------------------------------------------
    # 步骤 3：重要性采样比率 ρ = πθ / πold。
    #
    # 用 log prob 相减再 exp，比直接“prob / prob”数值更稳定：
    # prob 直接相除在极小概率时容易下溢。
    # 第一次更新时 policy == old，ratio 应恒等于 1。
    # ------------------------------------------------------------------
    ratio = torch.exp(policy_log_probs - old_log_probs)

    # ------------------------------------------------------------------
    # 步骤 4：PPO 裁剪。
    #
    # clip_ratio=ε 时，ratio 被限制在 [1-ε, 1+ε]。
    # 目的：单次更新的策略变化不能过大，形成“信任域”。
    # ------------------------------------------------------------------
    clipped_ratio = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio)

    # ------------------------------------------------------------------
    # 步骤 5：PPO/GRPO 的悲观目标。
    #
    # min(未裁剪代理项, 裁剪代理项)：
    # - A > 0：ratio 过大时按裁剪值算，避免一次更新吃太多收益；
    # - A < 0：ratio 过小时按裁剪值算，避免一次更新受过多惩罚。
    # 这保证了 surrogate 对策略变化的收益是“保守”的。
    # ------------------------------------------------------------------
    surrogate = torch.minimum(ratio * adv, clipped_ratio * adv)

    # ------------------------------------------------------------------
    # 步骤 6：按 response_mask 聚合策略损失。
    # sequence（默认）：每条回答内平均，再回答间平均（GRPO 论文）。
    # token：整个 batch 的所有 response token 平均（DAPO 做法）。
    # ------------------------------------------------------------------
    policy_loss = _masked_reduce(surrogate, mask, cfg.normalization, cfg.eps)

    # ------------------------------------------------------------------
    # 步骤 7：KL 惩罚项（可选）。
    #
    # GRPO 把 KL 直接加在 loss 上（而不是加在 reward 里），
    # 这样优势 A_i 的计算不受 KL 影响，实现更简单。
    # KL 与策略损失使用完全相同的归一化方式，保证量纲一致。
    # kl_values 恒非负（k3 的性质）。
    # ------------------------------------------------------------------
    kl = torch.zeros((), dtype=policy_log_probs.dtype, device=policy_log_probs.device)
    if cfg.kl_coef != 0.0:
        if ref_log_probs is None:
            raise ValueError("kl_coef != 0 时必须提供 ref_log_probs")
        kl_values = kl_k3(policy_log_probs, ref_log_probs)
        kl = _masked_reduce(kl_values, mask, cfg.normalization, cfg.eps)

    # ------------------------------------------------------------------
    # 步骤 8：符号约定。
    #
    # 优化器最小化 loss；论文最大化 J，因此：
    #   loss = -policy_loss + β·KL
    # 若写成 loss = policy_loss + β·KL 就等价于“反向优化”，是常见错误。
    # ------------------------------------------------------------------
    loss = -policy_loss + cfg.kl_coef * kl

    # ------------------------------------------------------------------
    # 步骤 9：诊断指标（不参与梯度）。
    #
    # clip_fraction：多少比例的 token 的 ratio 落在裁剪区间外。
    # 这个指标用 token 级平均计算，与当前归一化方式无关，
    # 方便跨实验对比“更新被截断的程度”。
    # ------------------------------------------------------------------
    clip_indicator = (
        (ratio < 1.0 - cfg.clip_ratio) | (ratio > 1.0 + cfg.clip_ratio)
    ).to(policy_log_probs.dtype)
    clip_fraction = _masked_reduce(clip_indicator, mask, "token", cfg.eps)

    # ratio_mean：所有有效 response token 的平均比率。
    # 第 1 步训练时应约等于 1.0；若远大于 1，说明策略概率整体抬升过快。
    ratio_mean = _masked_reduce(ratio, mask, "token", cfg.eps)

    return LossOutput(
        loss=loss,
        policy_loss=policy_loss,
        kl=kl,
        clip_fraction=clip_fraction,
        ratio_mean=ratio_mean,
        num_response_tokens=mask.sum(),
    )
