"""奖励计算：答案解析 + 正确性验证 + 可配置奖励 + DAPO 超长惩罚接口。

默认奖励（按项目任务约定）：
- 答对 +1.0
- 答错 0.0
- 答案无法解析按答错处理（0.0）
- 可选格式奖励，默认关闭（format_weight=0.0）
- 可选 DAPO Soft Overlong Punishment，默认关闭

注意（与论文的差异，详见 docs/algorithm_spec.md）：
DAPO 论文 Eq.7 的二值奖励为 +1 / -1。本项目为保持任务约定，
incorrect_reward 默认取 0.0；需要严格对齐论文时可配置为 -1.0。
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .answer_parser import parse_answer
from .math_verifier import verify_answer

BOXED_FORMAT_RE = re.compile(r"\\boxed\s*\{")


@dataclass
class RewardConfig:
    """规则奖励配置。"""

    correct_reward: float = 1.0
    incorrect_reward: float = 0.0
    format_weight: float = 0.0


@dataclass
class OverlongPenaltyConfig:
    """DAPO Soft Overlong Punishment 配置（论文 Eq.13）。

    - soft_cache_length: 软惩罚区间长度 L_cache，惩罚从 L_max - L_cache 开始；
    - penalty_scale: 对论文惩罚值整体缩放，默认 1.0（与论文一致）。
    """

    max_response_length: int
    soft_cache_length: int
    penalty_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.max_response_length <= 0:
            raise ValueError("max_response_length 必须为正数")
        if not 0 < self.soft_cache_length < self.max_response_length:
            raise ValueError("soft_cache_length 必须满足 0 < L_cache < L_max")


@dataclass
class RewardResult:
    answer: str | None
    correct: bool
    format_ok: bool
    reward: float
    length_penalty: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def has_boxed_format(completion: str) -> bool:
    return BOXED_FORMAT_RE.search(completion) is not None


def overlong_penalty(response_length: int, config: OverlongPenaltyConfig) -> float:
    """DAPO Eq.13 的超长惩罚 R_length(y)，取值区间 [-1, 0]。

    R_length(y) =
        0                                      |y| <= L_max - L_cache
        ((L_max - L_cache) - |y|) / L_cache    L_max - L_cache < |y| <= L_max
        -1                                     |y| > L_max
    """
    boundary = config.max_response_length - config.soft_cache_length
    if response_length <= boundary:
        penalty = 0.0
    elif response_length <= config.max_response_length:
        penalty = (boundary - response_length) / config.soft_cache_length
    else:
        penalty = -1.0
    return config.penalty_scale * penalty


def compute_reward(
    completion: str,
    ground_truth: str,
    config: RewardConfig | None = None,
    overlong_config: OverlongPenaltyConfig | None = None,
    response_length: int | None = None,
) -> RewardResult:
    """计算单条 completion 的奖励。

    Args:
        completion: 模型生成文本。
        ground_truth: GSM8K 标准答案（#### 之后的字符串）。
        config: 奖励配置，默认 1.0 / 0.0 / 格式奖励 0.0。
        overlong_config: 启用 DAPO 超长惩罚时传入。
        response_length: 生成的 response token 数；启用超长惩罚时必填。
    """
    cfg = config or RewardConfig()
    answer = parse_answer(completion)
    correct = verify_answer(answer, ground_truth)
    format_ok = has_boxed_format(completion)

    reward = cfg.correct_reward if correct else cfg.incorrect_reward
    reward += cfg.format_weight * float(format_ok)

    length_penalty = 0.0
    if overlong_config is not None:
        if response_length is None:
            raise ValueError("启用 overlong_config 时必须提供 response_length")
        length_penalty = overlong_penalty(response_length, overlong_config)
        reward += length_penalty

    return RewardResult(
        answer=answer,
        correct=correct,
        format_ok=format_ok,
        reward=reward,
        length_penalty=length_penalty,
    )


def compute_rewards(
    completions: list[str],
    ground_truths: list[str],
    config: RewardConfig | None = None,
    overlong_config: OverlongPenaltyConfig | None = None,
    response_lengths: list[int] | None = None,
) -> list[RewardResult]:
    """批量计算奖励；长度必须一一对应。"""
    if len(completions) != len(ground_truths):
        raise ValueError(
            f"completion 数量（{len(completions)}）"
            f"与 ground_truth 数量（{len(ground_truths)}）不一致"
        )
    if overlong_config is not None and response_lengths is None:
        raise ValueError("启用 overlong_config 时必须提供 response_lengths")
    if response_lengths is not None and len(response_lengths) != len(completions):
        raise ValueError("response_lengths 与 completions 数量不一致")

    return [
        compute_reward(
            completion,
            ground_truth,
            config=config,
            overlong_config=overlong_config,
            response_length=None if response_lengths is None else response_lengths[index],
        )
        for index, (completion, ground_truth) in enumerate(zip(completions, ground_truths))
    ]
