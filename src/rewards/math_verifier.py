"""答案验证：判断模型答案与标准答案是否等价。

规则：
- 先做文本归一化（去 $、千分位逗号、LaTeX 包装等）；
- 字符串完全一致视为正确；
- 否则尝试解析为数值（支持 "72"、"0.75"、"3/4"、带千分位的 "1,080"），
  在相对容差内相等视为正确。
"""

from __future__ import annotations

import re
from fractions import Fraction

DEFAULT_TOLERANCE = 1e-4

_TEXT_WRAPPER_RE = re.compile(r"\\text\{([^{}]*)\}")
_FRAC_RE = re.compile(r"\\(?:d|t)?frac\{([^{}]+)\}\{([^{}]+)\}")
_BOXED_RE = re.compile(r"\\boxed\{(.+)\}")


def normalize_answer(answer: str) -> str:
    text = answer.strip()
    text = text.replace("$", "")
    text = _TEXT_WRAPPER_RE.sub(r"\1", text)
    text = _FRAC_RE.sub(r"\1/\2", text)
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\!", "").replace("\\ ", "")
    text = _BOXED_RE.sub(r"\1", text)
    text = text.replace(",", "")
    text = "".join(text.split())
    text = text.strip("{}")
    return text.rstrip(".")


def _to_float(text: str) -> float | None:
    try:
        return float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        pass
    try:
        return float(text)
    except ValueError:
        return None


def verify_answer(
    predicted: str | None,
    ground_truth: str | None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> bool:
    """判断 predicted 与 ground_truth 是否等价；任一为空返回 False。"""
    if predicted is None or ground_truth is None:
        return False

    pred = normalize_answer(predicted)
    truth = normalize_answer(ground_truth)
    if not pred or not truth:
        return False
    if pred == truth:
        return True

    pred_value = _to_float(pred)
    truth_value = _to_float(truth)
    if pred_value is None or truth_value is None:
        return False
    return abs(pred_value - truth_value) <= tolerance * max(1.0, abs(truth_value))
