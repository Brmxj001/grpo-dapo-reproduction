"""Rollout 模块测试（Mock 采样 + 微型随机模型，不加载真实权重）：

    python tests/test_rollout.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.prompt import build_prompt  # noqa: E402
from src.rewards.reward import compute_reward  # noqa: E402
from src.rollout.buffer import PromptGroup, RolloutBuffer, RolloutSample  # noqa: E402
from src.rollout.generator import MockPolicyGenerator, SamplingParams  # noqa: E402
from src.rollout.log_probs import (  # noqa: E402
    aligned_token_log_probs,
    compute_token_log_probs,
    masked_mean,
)

PROCESSED_TEST = PROJECT_ROOT / "data/processed/gsm8k/test.parquet"
RAW_TEST = PROJECT_ROOT / "data/raw/gsm8k/main/test-00000-of-00001.parquet"

_TOKENIZER = None
_TINY_MODEL = None


def _tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from src.model_loader import load_tokenizer

        _TOKENIZER = load_tokenizer()
    return _TOKENIZER


def _tiny_model():
    """随机初始化的微型 Qwen2 模型（CPU，约几 MB），用于验证 log prob 计算。"""
    global _TINY_MODEL
    if _TINY_MODEL is None:
        from transformers import Qwen2Config, Qwen2ForCausalLM

        torch.manual_seed(0)
        config = Qwen2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
        _TINY_MODEL = Qwen2ForCausalLM(config).eval()
    return _TINY_MODEL


def _load_problem():
    from src.data.dataset import MathReasoningDataset

    if PROCESSED_TEST.exists():
        return MathReasoningDataset.from_processed(PROCESSED_TEST)[0]
    if RAW_TEST.exists():
        return MathReasoningDataset.from_raw(RAW_TEST)[0]
    return None


def test_mock_generator():
    tokenizer = _tokenizer()
    problem = _load_problem()
    if problem is None:
        print("  (跳过) GSM8K 测试集不存在")
        return

    prompt = build_prompt(problem.question, tokenizer)
    params = SamplingParams(num_generations=3)
    generator = MockPolicyGenerator(
        responses=[
            f"We compute step by step. \\boxed{{{problem.ground_truth}}}",
            "Wrong attempt. \\boxed{987654321}",
        ],
        tokenizer=tokenizer,
        params=params,
    )
    outputs = generator.generate(prompt)
    assert len(outputs) == 3
    assert len({tuple(item.prompt_ids) for item in outputs}) == 1
    assert all(item.prompt_ids == tokenizer.encode(prompt, add_special_tokens=False) for item in outputs)
    assert outputs[0].completion_text.startswith("We compute")
    assert outputs[1].completion_text.startswith("Wrong attempt")


def test_buffer_padding_and_alignment():
    buffer = RolloutBuffer()
    group = PromptGroup(
        prompt_ids=[10, 11, 12],
        prompt_text="prompt",
        question="2+3=?",
        ground_truth="5",
    )
    lengths = [2, 4, 1]
    for index, response_length in enumerate(lengths):
        response_ids = list(range(20 + index * 10, 20 + index * 10 + response_length))
        group.add_sample(
            RolloutSample(
                response_ids=response_ids,
                response_text=f"resp{index}",
                reward=float(index == 0),
                correct=index == 0,
                format_ok=True,
                answer="5" if index == 0 else "wrong",
                old_log_probs=torch.arange(response_length, dtype=torch.float32) * 0.1,
                truncated=index == 2,
            )
        )
    buffer.add_group(group)

    batch = buffer.to_batch(pad_token_id=0)
    assert batch["input_ids"].shape == (3, 7)  # prompt 3 + 最长 response 4
    assert batch["ref_log_probs"] is None
    assert batch["attention_mask"].sum(dim=1).tolist() == [5.0, 7.0, 4.0]
    assert batch["response_mask"].sum(dim=1).tolist() == [2.0, 4.0, 1.0]
    assert batch["response_lengths"].tolist() == lengths
    assert batch["rewards"].tolist() == [1.0, 0.0, 0.0]
    assert batch["truncated"].tolist() == [False, False, True]
    assert batch["group_ids"].tolist() == [0, 0, 0]

    for row, response_length in enumerate(lengths):
        prompt_length = 3
        assert batch["response_mask"][row, prompt_length:].sum().item() == response_length
        expected_positions = slice(prompt_length, prompt_length + response_length)
        expected = torch.arange(response_length, dtype=torch.float32) * 0.1
        assert torch.allclose(batch["old_log_probs"][row, expected_positions], expected)
        # padding 区域不参与 attention / response
        assert batch["attention_mask"][row, prompt_length + response_length:].sum() == 0
        assert batch["response_mask"][row, :prompt_length].sum() == 0


def test_log_probs_alignment():
    model = _tiny_model()
    input_ids = torch.tensor([[5, 9, 12, 7, 3, 4]])
    position = 4  # 验证 token 4 的 log prob

    aligned = aligned_token_log_probs(model, input_ids)
    token_log_probs = compute_token_log_probs(model, input_ids)
    assert aligned.shape == input_ids.shape
    assert token_log_probs.shape == (1, input_ids.shape[1] - 1)
    assert aligned[0, 0].item() == 0.0

    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    manual = F.log_softmax(logits[0, position - 1], dim=-1)[input_ids[0, position]]
    assert torch.isclose(aligned[0, position], manual, atol=1e-6)
    assert torch.isclose(token_log_probs[0, position - 1], manual, atol=1e-6)

    mask = torch.tensor([[0.0, 0.0, 0.0, 1.0, 1.0, 1.0]])
    mean = masked_mean(aligned, mask)
    expected = aligned[0, 3:].mean()
    assert torch.isclose(mean, expected)


def test_rollout_with_reward_and_buffer():
    tokenizer = _tokenizer()
    problem = _load_problem()
    if problem is None:
        print("  (跳过) GSM8K 测试集不存在")
        return

    prompt = build_prompt(problem.question, tokenizer)
    params = SamplingParams(num_generations=2)
    generator = MockPolicyGenerator(
        responses=[f"\\boxed{{{problem.ground_truth}}}", "\\boxed{987654321}"],
        tokenizer=tokenizer,
        params=params,
    )

    group = PromptGroup(
        prompt_ids=generator.generate(prompt)[0].prompt_ids,
        prompt_text=prompt,
        question=problem.question,
        ground_truth=problem.ground_truth,
    )
    for output in generator.generate(prompt):
        result = compute_reward(output.completion_text, problem.ground_truth)
        group.add_sample(
            RolloutSample(
                response_ids=output.completion_ids,
                response_text=output.completion_text,
                reward=result.reward,
                correct=result.correct,
                format_ok=result.format_ok,
                answer=result.answer,
                old_log_probs=torch.zeros(len(output.completion_ids)),
                truncated=output.truncated,
            )
        )

    buffer = RolloutBuffer()
    buffer.add_group(group)
    batch = buffer.to_batch(pad_token_id=tokenizer.pad_token_id or 0)

    assert buffer.num_groups == 1 and buffer.num_samples == 2
    assert batch["rewards"].tolist() == [1.0, 0.0]
    assert batch["correctness"].tolist() == [1.0, 0.0]
    assert batch["ref_log_probs"] is None
    assert torch.allclose(batch["old_log_probs"] * batch["response_mask"], batch["old_log_probs"])
    assert batch["response_mask"].sum().item() == sum(
        len(sample.response_ids) for sample in group.samples
    )


TESTS = [
    ("generator: Mock 组内采样", test_mock_generator),
    ("buffer: padding 与 logprob 对齐", test_buffer_padding_and_alignment),
    ("log_probs: 微型模型逐 token 对齐", test_log_probs_alignment),
    ("rollout: 采样 + 奖励 + Buffer 集成", test_rollout_with_reward_and_buffer),
]


def main() -> int:
    failures = []
    for name, test in TESTS:
        try:
            test()
        except Exception:
            failures.append(name)
            print(f"[FAIL] {name}")
            traceback.print_exc()
        else:
            print(f"[PASS] {name}")
    total = len(TESTS)
    print(f"\n结果：{total - len(failures)}/{total} 通过")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
