"""把数学题转换成 Qwen2.5 对话模型可接收的 Prompt。

Qwen2.5-Instruct 使用 ChatML 模板（tokenizer.chat_template），本模块只负责
组织 messages 并调用 apply_chat_template，不涉及 tokenize 与张量。
"""

from __future__ import annotations

from typing import Any

MATH_INSTRUCTION = (
    "Solve the following math problem step by step. "
    "Put the final answer in \\boxed{}."
)


def build_messages(
    question: str,
    instruction: str = MATH_INSTRUCTION,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    """把题目包装成 chat messages；system_prompt 为 None 时使用 Qwen 默认 system。"""
    question = question.strip()
    if not question:
        raise ValueError("题目不能为空")

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append(
        {"role": "user", "content": f"{instruction}\n\nProblem:\n{question}"}
    )
    return messages


def build_prompt(
    question: str,
    tokenizer: Any,
    instruction: str = MATH_INSTRUCTION,
    system_prompt: str | None = None,
    add_generation_prompt: bool = True,
) -> str:
    """渲染成 Qwen ChatML 文本 Prompt；推理时应保持 add_generation_prompt=True。"""
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Tokenizer 没有 chat_template，无法构建对话 Prompt")

    return tokenizer.apply_chat_template(
        build_messages(question, instruction, system_prompt),
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
