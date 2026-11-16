"""回答采样：同一道题生成 G 个回答。

- HFPolicyGenerator: 真实 HuggingFace 模型采样（需要模型与 GPU/CPU）。
- MockPolicyGenerator: 不加载模型，按预设回答返回结果，仅用于 CPU 测试。

两者实现相同的 generate() 接口，训练代码可无感切换。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence

import torch

from ..data.prompt import build_prompt


@dataclass
class SamplingParams:
    num_generations: int = 4
    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    do_sample: bool = True


@dataclass
class GenerationOutput:
    prompt_text: str
    prompt_ids: list[int]
    completion_ids: list[int]
    completion_text: str
    truncated: bool


class PolicyGenerator(Protocol):
    def generate(
        self, prompt_text: str, params: SamplingParams | None = None
    ) -> list[GenerationOutput]: ...


def _encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


class HFPolicyGenerator:
    """使用 transformers 的 model.generate 做组内采样。"""

    def __init__(self, model, tokenizer, params: SamplingParams, device: str | torch.device | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.params = params
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    def _generation_kwargs(self, params: SamplingParams) -> dict[str, object]:
        kwargs: dict[str, object] = {
            "max_new_tokens": params.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if params.do_sample and params.temperature > 0:
            kwargs.update(
                do_sample=True,
                temperature=params.temperature,
                top_p=params.top_p,
                top_k=params.top_k if params.top_k > 0 else 0,
            )
        else:
            kwargs["do_sample"] = False
        return kwargs

    def generate(
        self, prompt_text: str, params: SamplingParams | None = None
    ) -> list[GenerationOutput]:
        params = params or self.params
        encoded = self.tokenizer(
            prompt_text, return_tensors="pt", add_special_tokens=False
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        prompt_ids = input_ids[0].tolist()
        prompt_len = input_ids.shape[1]

        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids.repeat(params.num_generations, 1),
                attention_mask=attention_mask.repeat(params.num_generations, 1),
                **self._generation_kwargs(params),
            )

        results: list[GenerationOutput] = []
        eos_token_id = self.tokenizer.eos_token_id
        for row in output:
            completion_ids = row[prompt_len:].tolist()
            completion_ids, truncated = self._trim_at_eos(
                completion_ids, eos_token_id, params.max_new_tokens
            )
            results.append(
                GenerationOutput(
                    prompt_text=prompt_text,
                    prompt_ids=prompt_ids,
                    completion_ids=completion_ids,
                    completion_text=self.tokenizer.decode(
                        completion_ids, skip_special_tokens=True
                    ),
                    truncated=truncated,
                )
            )
        return results

    @staticmethod
    def _trim_at_eos(
        completion_ids: list[int], eos_token_id: int | None, max_new_tokens: int
    ) -> tuple[list[int], bool]:
        """截断到第一个 EOS（包含 EOS），并判断是否达到生成长度上限。"""
        if eos_token_id is not None and eos_token_id in completion_ids:
            return completion_ids[: completion_ids.index(eos_token_id) + 1], False
        return completion_ids, len(completion_ids) >= max_new_tokens


class MockPolicyGenerator:
    """不加载模型的确定性采样器（仅用于测试）。

    responses 支持三种形式：
    - Mapping[str, Sequence[str]]: prompt 包含某个 key 时使用对应回答；
    - Sequence[str]: 所有 prompt 共用一组回答；
    - Callable[[str], Sequence[str]]: 自定义生成逻辑。

    tokenizer 为 None 时使用基于字符编码的伪 token id，仅用于测试 Buffer，
    不能用于真实模型。
    """

    def __init__(
        self,
        responses: Mapping[str, Sequence[str]] | Sequence[str] | Callable[[str], Sequence[str]],
        tokenizer=None,
        params: SamplingParams | None = None,
    ):
        self.responses = responses
        self.tokenizer = tokenizer
        self.params = params or SamplingParams()

    def _lookup(self, prompt_text: str) -> list[str]:
        if callable(self.responses):
            return list(self.responses(prompt_text))
        if isinstance(self.responses, Mapping):
            for key, value in self.responses.items():
                if key in prompt_text:
                    return list(value)
            raise KeyError(f"MockPolicyGenerator 没有匹配的回答：{prompt_text[:80]!r}")
        return list(self.responses)

    def _encode(self, text: str) -> list[int]:
        if self.tokenizer is not None:
            return _encode(self.tokenizer, text)
        # 伪 token：可复现、无需词表，仅供 Buffer/对齐测试
        return [1 + (ord(char) % 97) for char in text]

    def generate(
        self, prompt_text: str, params: SamplingParams | None = None
    ) -> list[GenerationOutput]:
        params = params or self.params
        candidates = self._lookup(prompt_text)
        if not candidates:
            raise ValueError("MockPolicyGenerator 的回答列表为空")

        prompt_ids = self._encode(prompt_text)
        results: list[GenerationOutput] = []
        for index in range(params.num_generations):
            text = candidates[index % len(candidates)]
            completion_ids = self._encode(text)
            results.append(
                GenerationOutput(
                    prompt_text=prompt_text,
                    prompt_ids=prompt_ids,
                    completion_ids=completion_ids,
                    completion_text=text,
                    truncated=len(completion_ids) >= params.max_new_tokens,
                )
            )
        return results


def build_prompts(questions: Sequence[str], tokenizer) -> list[str]:
    """把题目批量渲染为 Qwen ChatML Prompt（训练入口使用）。"""
    return [build_prompt(question, tokenizer) for question in questions]
