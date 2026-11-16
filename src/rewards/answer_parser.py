"""从模型生成的长文本中提取最终答案。

按优先级依次尝试：
1. 最后一个完整的 \\boxed{...}（支持花括号嵌套，如 \\boxed{\\frac{1}{2}}）
2. GSM8K 风格的 #### 答案
3. "Answer: ..." / "The answer is ..." 形式

都找不到时返回 None，由上层决定是否给零奖励。
"""

from __future__ import annotations

import re

BOXED_TOKEN = "\\boxed"

_ANSWER_COLON_RE = re.compile(r"(?i)\*{0,2}answer\*{0,2}\s*[:：]\s*([^\n]+)")
_ANSWER_IS_RE = re.compile(
    r"(?i)(?:the\s+)?(?:final\s+)?answer\s+is\s*[:：]?\s*([^\n]+)"
)
_HASH_ANSWER_RE = re.compile(r"####\s*([^\n]+)")


def _extract_last_boxed(text: str) -> str | None:
    """从后往前找 \\boxed{...}，用花括号计数处理嵌套内容。"""
    pos = text.rfind(BOXED_TOKEN)
    while pos != -1:
        start = pos + len(BOXED_TOKEN)
        while start < len(text) and text[start] in " \t":
            start += 1
        if start < len(text) and text[start] == "{":
            content_start = start + 1
            depth = 1
            cursor = content_start
            while cursor < len(text):
                char = text[cursor]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return text[content_start:cursor].strip()
                cursor += 1
            # 花括号未闭合（常见于生成被截断），退化到取到行尾
            line_end = text.find("\n", content_start)
            if line_end == -1:
                line_end = len(text)
            return text[content_start:line_end].strip().rstrip("}").strip() or None
        pos = text.rfind(BOXED_TOKEN, 0, pos)
    return None


def _clean_answer(answer: str) -> str | None:
    answer = answer.strip().strip("$").strip()
    answer = answer.strip("*")
    answer = answer.rstrip(".,;。；")
    answer = answer.strip()
    return answer or None


def parse_answer(text: str | None) -> str | None:
    """提取最终答案；解析失败返回 None。"""
    if not text:
        return None

    boxed = _extract_last_boxed(text)
    if boxed:
        cleaned = _clean_answer(boxed)
        if cleaned:
            return cleaned

    matches = _HASH_ANSWER_RE.findall(text)
    if matches:
        cleaned = _clean_answer(matches[-1])
        if cleaned:
            return cleaned

    for pattern in (_ANSWER_COLON_RE, _ANSWER_IS_RE):
        matches = pattern.findall(text)
        if matches:
            cleaned = _clean_answer(matches[-1])
            if cleaned:
                return cleaned

    return None
