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
    """返回 [B, T-1]，第 i 列是 token i+1 的 log prob。

    显存提示：词表 V=151936，logits 形状是 [N, L, V]，每份 fp32 副本约
    4·N·L·V 字节。因此这里用 F.log_softmax(..., dtype=torch.float32) 直接
    在 fp32 下计算，而不是先 logits.float() 再 log_softmax —— 后者会同时
    多留一份同样大小的 fp32 张量（N=8、L=620 时约 3 GB），12 GB 卡上直接
    OOM。数学结果完全一致，只是少一份中间副本。
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]
    log_probs = torch.nn.functional.log_softmax(
        logits, dim=-1, dtype=torch.float32
    )
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


def chunked_token_log_probs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """训练用：分块计算 [B, T-1] 的 log prob，数学上与
    ``compute_token_log_probs`` 完全一致，但显存峰值低得多。

    为什么必须分块：V=151936 时 ``log_softmax`` 输出是 [N, L, V] 的 fp32
    张量（N=4、L≈800 时约 2 GB），它会被反向图保存；而 ``gather`` 的反向
    还要再分配一个同样大小的全零梯度张量。两者叠加约 4 GB，长回答步直接
    把 12 GB 卡打爆（实测第 12 步 backward 要 1.5 GiB 时 OOM）。

    分块后，反向图只保存每个 chunk 的 [N, chunk, V] 输出，反向梯度也按
    chunk 分配并立即释放，峰值从 ~4 GB 降到 ~2 GB。
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]

    chunks: list[torch.Tensor] = []
    for start in range(0, logits.shape[1], chunk_size):
        chunk = logits[:, start : start + chunk_size, :]
        chunk_log_probs = torch.nn.functional.log_softmax(
            chunk, dim=-1, dtype=torch.float32
        )
        chunks.append(
            torch.gather(
                chunk_log_probs,
                dim=-1,
                index=labels[:, start : start + chunk_size].unsqueeze(-1),
            ).squeeze(-1)
        )
    return torch.cat(chunks, dim=1)


def chunked_aligned_token_log_probs(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """``aligned_token_log_probs`` 的分块显存友好版本，用于训练前向。"""
    token_log_probs = chunked_token_log_probs(
        model, input_ids, attention_mask, chunk_size=chunk_size
    )
    aligned = torch.zeros(
        input_ids.shape, dtype=token_log_probs.dtype, device=token_log_probs.device
    )
    aligned[:, 1:] = token_log_probs
    return aligned


@torch.inference_mode()
def aligned_log_probs_and_entropy(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """一次前向同时返回对齐 log prob 与逐 token 策略熵，都是 [B, T]。

    熵定义为 H_t = -Σ_v p_v log p_v，衡量当前策略在位置 t 上的"犹豫程度"：
    - 训练早期通常在 1~3 之间并缓慢下降；
    - 若持续跌向 0，说明策略塌缩成近乎确定性的输出（RLHF 里典型的 entropy collapse）。

    为什么单独写一个函数而不是改 compute_token_log_probs：
    1. 熵需要整份 log_softmax，而 log prob 只需要被选中的那一个 token；
    2. 按序列分块（chunk_size 个位置一批）计算，避免同时存在多份
       [B, L, V] 的 fp32 张量 —— V=151936 时每份都是 GB 级；
    3. 本函数只用于诊断，全程 inference_mode，不建计算图。

    只用来打日志的指标，不参与梯度。
    """
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits[:, :-1, :]
    labels = input_ids[:, 1:]

    log_prob_chunks: list[torch.Tensor] = []
    entropy_chunks: list[torch.Tensor] = []
    for start in range(0, logits.shape[1], chunk_size):
        chunk = logits[:, start : start + chunk_size, :].float()
        log_probs = torch.log_softmax(chunk, dim=-1)
        picked = torch.gather(
            log_probs, dim=-1, index=labels[:, start : start + chunk_size].unsqueeze(-1)
        ).squeeze(-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        log_prob_chunks.append(picked)
        entropy_chunks.append(entropy)
        del chunk, log_probs

    token_log_probs = torch.cat(log_prob_chunks, dim=1)
    token_entropy = torch.cat(entropy_chunks, dim=1)

    aligned_log_probs = torch.zeros(
        input_ids.shape, dtype=token_log_probs.dtype, device=token_log_probs.device
    )
    aligned_entropy = torch.zeros(
        input_ids.shape, dtype=token_entropy.dtype, device=token_entropy.device
    )
    aligned_log_probs[:, 1:] = token_log_probs
    aligned_entropy[:, 1:] = token_entropy
    return aligned_log_probs, aligned_entropy


def _trim_response(
    response_ids: list[int], eos_token_id: int | None, max_new_tokens: int
) -> tuple[list[int], bool]:
    """截到第一个 EOS（含 EOS）；没有 EOS 时判断是否达到生成长度上限。"""
    if eos_token_id is not None and eos_token_id in response_ids:
        return response_ids[: response_ids.index(eos_token_id) + 1], False
    return response_ids, len(response_ids) >= max_new_tokens


def split_batched_output(
    output: torch.Tensor,
    prompt_texts: Sequence[str],
    prompt_ids_list: Sequence[Sequence[int]],
    padded_prompt_length: int,
    params: SamplingParams,
    tokenizer,
) -> list[list[GenerationOutput]]:
    """把一次批量 generate 的输出按题切开。

    批量生成使用**左 padding**，所以真实 prompt 位于每行
    ``[padded_prompt_length - prompt_len, padded_prompt_length)``，
    回答从 ``padded_prompt_length`` 开始。
    """
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    groups: list[list[GenerationOutput]] = []
    for index, prompt_text in enumerate(prompt_texts):
        prompt_ids = list(prompt_ids_list[index])
        rows: list[GenerationOutput] = []
        for offset in range(params.num_generations):
            row = output[index * params.num_generations + offset]
            response_ids, truncated = _trim_response(
                row[padded_prompt_length:].tolist(), eos_token_id, params.max_new_tokens
            )
            rows.append(
                GenerationOutput(
                    prompt_text=prompt_text,
                    prompt_ids=prompt_ids,
                    response_ids=response_ids,
                    response_text=tokenizer.decode(
                        response_ids, skip_special_tokens=True
                    ),
                    truncated=truncated,
                )
            )
        groups.append(rows)
    return groups


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
            response_ids, truncated = _trim_response(
                row[prompt_length:].tolist(), eos_token_id, params.max_new_tokens
            )
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

    def generate_batch(
        self, prompt_texts: Sequence[str], params: SamplingParams | None = None
    ) -> list[list[GenerationOutput]]:
        """为多道题一次性采样，返回与 ``prompt_texts`` 等长的分组列表。

        与逐题调用 ``generate()`` 的采样分布相同，但把 K 道题 × G 条回答
        放进同一个解码 batch：每个 token 的 kernel 从处理 G 行变成处理 K×G 行，
        kernel 启动与 Python 循环开销被摊薄（见 docs/perf_optimization_backlog.md P1）。

        使用**左 padding**：右 padding 会让生成从 pad 位置继续写，结果全错。
        """
        params = params or self.params
        if not prompt_texts:
            return []

        tokenizer = self.tokenizer
        padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            encoded = tokenizer(
                list(prompt_texts),
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            )
        finally:
            tokenizer.padding_side = padding_side

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        prompt_lengths = attention_mask.sum(dim=1).tolist()
        padded_prompt_length = input_ids.shape[1]
        prompt_ids_list = [
            input_ids[index, padded_prompt_length - int(length):].tolist()
            for index, length in enumerate(prompt_lengths)
        ]

        generations = params.num_generations
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids.repeat_interleave(generations, dim=0),
                attention_mask=attention_mask.repeat_interleave(generations, dim=0),
                **self._kwargs(params),
            )

        return split_batched_output(
            output,
            prompt_texts,
            prompt_ids_list,
            padded_prompt_length,
            params,
            tokenizer,
        )


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

    def generate_batch(
        self, prompt_texts: Sequence[str], params: SamplingParams | None = None
    ) -> list[list[GenerationOutput]]:
        """Mock 版本：逐题返回，接口与 HFGenerator.generate_batch 一致。"""
        return [self.generate(text, params) for text in prompt_texts]
