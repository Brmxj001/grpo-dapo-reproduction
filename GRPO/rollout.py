"""Rollout：对每道题采样 G 个回答，计算逐 token log prob，并缓存为训练 batch。

张量对齐约定（RolloutBuffer.to_batch，右 padding）：
- input_ids:      [N, L]  prompt + response
- attention_mask: [N, L]  有效 token 为 1
- response_mask:  [N, L]  response token 为 1
- old_log_probs:  [N, L]  与 input_ids 逐 token 对齐，非 response 位置为 0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import torch


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
    response_ids: list[int]
    response_text: str
    truncated: bool


@dataclass
class RolloutSample:
    response_ids: list[int]
    response_text: str
    reward: float
    correct: bool
    old_log_probs: torch.Tensor | None = None
    ref_log_probs: torch.Tensor | None = None
    truncated: bool = False


@dataclass
class PromptGroup:
    prompt_ids: list[int]
    prompt_text: str
    question: str
    ground_truth: str
    samples: list[RolloutSample] = field(default_factory=list)

    def add_sample(self, sample: RolloutSample) -> None:
        response_length = len(sample.response_ids)
        for name in ("old_log_probs", "ref_log_probs"):
            value = getattr(sample, name)
            if value is not None and value.numel() != response_length:
                raise ValueError(
                    f"{name} 长度（{value.numel()}）"
                    f"与 response 长度（{response_length}）不一致"
                )
        self.samples.append(sample)


class RolloutBuffer:
    """一组题目（每题 G 个回答）的缓存。"""

    def __init__(self) -> None:
        self.groups: list[PromptGroup] = []

    def add_group(self, group: PromptGroup) -> None:
        if not group.samples:
            raise ValueError("PromptGroup 至少需要一个回答")
        self.groups.append(group)

    def clear(self) -> None:
        self.groups.clear()

    @property
    def num_groups(self) -> int:
        return len(self.groups)

    @property
    def num_samples(self) -> int:
        return sum(len(group.samples) for group in self.groups)

    def rewards(self) -> torch.Tensor:
        return torch.tensor(
            [sample.reward for group in self.groups for sample in group.samples],
            dtype=torch.float32,
        )

    def to_batch(self, pad_token_id: int = 0) -> dict[str, torch.Tensor]:
        entries = [
            (group_index, group, sample)
            for group_index, group in enumerate(self.groups)
            for sample in group.samples
        ]
        if not entries:
            raise ValueError("RolloutBuffer 为空")

        max_length = max(
            len(group.prompt_ids) + len(sample.response_ids)
            for _, group, sample in entries
        )
        num_samples = len(entries)

        input_ids = torch.full((num_samples, max_length), pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((num_samples, max_length), dtype=torch.float32)
        response_mask = torch.zeros((num_samples, max_length), dtype=torch.float32)
        old_log_probs = torch.zeros((num_samples, max_length), dtype=torch.float32)
        ref_log_probs = torch.zeros((num_samples, max_length), dtype=torch.float32)
        rewards = torch.zeros(num_samples, dtype=torch.float32)
        group_ids = torch.zeros(num_samples, dtype=torch.long)
        response_lengths = torch.zeros(num_samples, dtype=torch.long)
        truncated = torch.zeros(num_samples, dtype=torch.bool)
        has_ref = any(sample.ref_log_probs is not None for _, _, sample in entries)

        for row, (group_index, group, sample) in enumerate(entries):
            prompt_length = len(group.prompt_ids)
            response_length = len(sample.response_ids)
            sequence = group.prompt_ids + sample.response_ids
            length = len(sequence)

            input_ids[row, :length] = torch.tensor(sequence, dtype=torch.long)
            attention_mask[row, :length] = 1.0
            response_mask[row, prompt_length:length] = 1.0
            response_lengths[row] = response_length
            rewards[row] = sample.reward
            group_ids[row] = group_index
            truncated[row] = sample.truncated

            if sample.old_log_probs is not None:
                old_log_probs[row, prompt_length:length] = sample.old_log_probs.float()
            if sample.ref_log_probs is not None:
                ref_log_probs[row, prompt_length:length] = sample.ref_log_probs.float()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "old_log_probs": old_log_probs,
            "ref_log_probs": ref_log_probs if has_ref else None,
            "rewards": rewards,
            "group_ids": group_ids,
            "response_lengths": response_lengths,
            "truncated": truncated,
        }


def compute_token_log_probs(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """返回 [B, T-1]，第 i 列是 token i+1 的 log prob。"""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :].float()
    labels = input_ids[:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    return torch.gather(log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def aligned_token_log_probs(
    model, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """返回 [B, T]，aligned[:,t] = log π(x_t | x_<t)，第 0 列填 0。"""
    token_log_probs = compute_token_log_probs(model, input_ids, attention_mask)
    aligned = torch.zeros(
        input_ids.shape, dtype=token_log_probs.dtype, device=token_log_probs.device
    )
    aligned[:, 1:] = token_log_probs
    return aligned


class HFGenerator:
    """使用 transformers model.generate 做组内采样。"""

    def __init__(self, model, tokenizer, params: SamplingParams, device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.params = params
        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

    def _kwargs(self, params: SamplingParams) -> dict:
        kwargs = {
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
        encoded = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        prompt_ids = input_ids[0].tolist()
        prompt_length = input_ids.shape[1]

        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids.repeat(params.num_generations, 1),
                attention_mask=attention_mask.repeat(params.num_generations, 1),
                **self._kwargs(params),
            )

        eos_token_id = self.tokenizer.eos_token_id
        results: list[GenerationOutput] = []
        for row in output:
            response_ids = row[prompt_length:].tolist()
            if eos_token_id is not None and eos_token_id in response_ids:
                response_ids = response_ids[: response_ids.index(eos_token_id) + 1]
                truncated = False
            else:
                truncated = len(response_ids) >= params.max_new_tokens
            results.append(
                GenerationOutput(
                    prompt_text=prompt_text,
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_text=self.tokenizer.decode(
                        response_ids, skip_special_tokens=True
                    ),
                    truncated=truncated,
                )
            )
        return results


class MockGenerator:
    """不加载模型的确定性采样器，仅用于 CPU 测试。"""

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
            raise KeyError(f"MockGenerator 没有匹配的回答：{prompt_text[:80]!r}")
        return list(self.responses)

    def _encode(self, text: str) -> list[int]:
        if self.tokenizer is not None:
            return self.tokenizer.encode(text, add_special_tokens=False)
        return [1 + (ord(char) % 97) for char in text]

    def generate(
        self, prompt_text: str, params: SamplingParams | None = None
    ) -> list[GenerationOutput]:
        params = params or self.params
        candidates = self._lookup(prompt_text)
        if not candidates:
            raise ValueError("MockGenerator 的回答列表为空")

        prompt_ids = self._encode(prompt_text)
        results: list[GenerationOutput] = []
        for index in range(params.num_generations):
            text = candidates[index % len(candidates)]
            response_ids = self._encode(text)
            results.append(
                GenerationOutput(
                    prompt_text=prompt_text,
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_text=text,
                    truncated=len(response_ids) >= params.max_new_tokens,
                )
            )
        return results
