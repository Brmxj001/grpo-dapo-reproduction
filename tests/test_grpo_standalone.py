"""独立版 GRPO（GRPO/ 文件夹）的 CPU 测试：

    python tests/test_grpo_standalone.py

只使用 Mock 采样和微型随机模型，不加载 Qwen 权重，不进行真实训练。
"""

from __future__ import annotations

import math
import sys
import tempfile
import traceback
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from GRPO.advantage import compute_group_advantages  # noqa: E402
from GRPO.loss import GRPOLossConfig, compute_grpo_loss  # noqa: E402
from GRPO.metrics import MetricsLogger, reward_statistics  # noqa: E402
from GRPO.rollout import (  # noqa: E402
    MockGenerator,
    PromptGroup,
    RolloutBuffer,
    RolloutSample,
    SamplingParams,
    aligned_token_log_probs,
    chunked_aligned_token_log_probs,
    split_batched_output,
)

_TINY_MODEL = None


class FakeTokenizer:
    chat_template = "fake"
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return text + "\nassistant:" if add_generation_prompt else text

    def encode(self, text, add_special_tokens=False):
        return [2 + (ord(char) % 60) for char in text][:20]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(97 + int(token) % 26) for token in ids)


def _tiny_model():
    global _TINY_MODEL
    if _TINY_MODEL is None:
        from transformers import Qwen2Config, Qwen2ForCausalLM

        torch.manual_seed(0)
        _TINY_MODEL = Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=64,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=2,
                max_position_embeddings=64,
            )
        )
    return _TINY_MODEL


def test_advantage_manual():
    result = compute_group_advantages(torch.tensor([1.0, 3.0, 0.0, 0.0]), 2, eps=0.0)
    assert torch.allclose(result.advantages, torch.tensor([[-1.0, 1.0], [0.0, 0.0]]))
    assert result.zero_variance.tolist() == [False, True]


def test_loss_manual():
    ratios = torch.tensor([[1.2, 0.8], [0.5, 1.0]])
    policy = torch.log(ratios)
    old = torch.zeros_like(policy)
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    advantages = torch.tensor([1.0, -1.0])

    sequence = compute_grpo_loss(
        policy, old, advantages, mask,
        config=GRPOLossConfig(normalization="sequence", kl_coef=0.0),
    )
    assert math.isclose(float(sequence.policy_loss), 0.1, abs_tol=1e-6)
    assert math.isclose(float(sequence.loss), -0.1, abs_tol=1e-6)

    token = compute_grpo_loss(
        policy, old, advantages, mask,
        config=GRPOLossConfig(normalization="token", kl_coef=0.0),
    )
    assert math.isclose(float(token.policy_loss), 0.4, abs_tol=1e-6)

    zero_policy = torch.zeros_like(policy)
    ref = torch.full_like(policy, 0.1)
    kl = compute_grpo_loss(
        zero_policy, zero_policy, torch.zeros(2), mask, ref_log_probs=ref,
        config=GRPOLossConfig(kl_coef=1.0, normalization="token"),
    )
    assert math.isclose(float(kl.kl), math.exp(0.1) - 0.1 - 1.0, abs_tol=1e-6)


def test_buffer_and_log_probs():
    assert torch.isfinite(torch.tensor(1.0))
    buffer = RolloutBuffer()
    group = PromptGroup([10, 11], "prompt", "2+3=?", "5")
    for index, length in enumerate([2, 4]):
        group.add_sample(
            RolloutSample(
                response_ids=list(range(20, 20 + length)),
                response_text="x",
                reward=float(index == 0),
                correct=index == 0,
                old_log_probs=torch.zeros(length),
            )
        )
    buffer.add_group(group)
    batch = buffer.to_batch(pad_token_id=0)
    assert batch["input_ids"].shape == (2, 6)
    assert batch["response_mask"].sum(dim=1).tolist() == [2.0, 4.0]
    assert batch["attention_mask"].sum(dim=1).tolist() == [4.0, 6.0]

    model = _tiny_model()
    input_ids = torch.tensor([[5, 9, 12, 7, 3]])
    aligned = aligned_token_log_probs(model, input_ids)
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
    manual = torch.log_softmax(logits[0, 3], dim=-1)[input_ids[0, 4]]
    assert torch.isclose(aligned[0, 4], manual, atol=1e-6)


def test_chunked_log_probs_match_full():
    """分块实现必须与整份实现逐元素一致（只是显存策略不同）。"""
    model = _tiny_model()
    input_ids = torch.tensor([[5, 9, 12, 7, 3, 8, 11, 6, 4, 2]])
    full = aligned_token_log_probs(model, input_ids)
    for chunk_size in (1, 2, 3, 64):
        chunked = chunked_aligned_token_log_probs(
            model, input_ids, chunk_size=chunk_size
        )
        assert torch.allclose(full, chunked, atol=1e-6), f"chunk_size={chunk_size}"
        assert chunked.shape == input_ids.shape
        assert torch.equal(chunked[:, 0], torch.zeros_like(chunked[:, 0]))


def test_metrics_logger():
    with tempfile.TemporaryDirectory() as directory:
        logger = MetricsLogger(directory, verbose=False)
        logger.log(1, {"loss": 1.0, "reward_mean": 0.5})
        logger.log(2, {"loss": 0.5, "reward_mean": 0.75})
        assert (Path(directory) / "metrics.jsonl").exists()
        summary = logger.summary()
        assert math.isclose(summary["loss"], 0.75)
        logger.close()
    stats = reward_statistics([1.0, 0.0])
    assert math.isclose(stats["reward_mean"], 0.5)
    assert math.isclose(stats["reward_std"], 0.5)


def test_trainer_one_step():
    from GRPO.trainer import GRPOTrainer

    tokenizer = FakeTokenizer()
    model = _tiny_model()
    generator = MockGenerator(
        responses=["\\boxed{72}", "I am not sure."],
        tokenizer=tokenizer,
        params=SamplingParams(num_generations=2),
    )
    config = {
        "seed": 0,
        "model": {"device": "cpu"},
        "data": {"train_file": "data/processed/gsm8k/train.parquet"},
        "rollout": {"num_generations": 2, "max_new_tokens": 32},
        "training": {
            "learning_rate": 1e-3,
            "max_steps": 1,
            "prompts_per_batch": 1,
            "output_dir": "",
        },
        "algorithm": {"clip_ratio": 0.2, "kl_coef": 0.0},
        "reward": {"correct_reward": 1.0, "incorrect_reward": 0.0},
    }

    with tempfile.TemporaryDirectory() as directory:
        config["training"]["output_dir"] = str(Path(directory) / "run")
        trainer = GRPOTrainer(config, model=model, tokenizer=tokenizer, generator=generator)
        metrics = trainer.train_step()
        assert abs(metrics["reward_mean"] - 0.5) < 1e-6
        assert math.isfinite(metrics["loss"])
        assert metrics["grad_norm"] > 0
        assert (Path(directory) / "run/logs/metrics.jsonl").exists()
        trainer.logger.close()


def test_split_batched_output():
    """批量解码结果按题切分：左 padding 偏移、EOS 截断与长度上限。"""
    tokenizer = FakeTokenizer()
    params = SamplingParams(num_generations=2, max_new_tokens=3)
    # 两条 prompt 左 padding 到长度 3：p0=[5,6] → [0,5,6]；p1=[7] → [0,0,7]
    output = torch.tensor(
        [
            [0, 5, 6, 10, 11, 1],   # p0 / 回答 10,11,EOS
            [0, 5, 6, 12, 13, 14],  # p0 / 无 EOS，达到 3 token → 截断
            [0, 0, 7, 15, 1, 0],    # p1 / 回答 15,EOS
            [0, 0, 7, 16, 17, 18],  # p1 / 无 EOS，截断
        ]
    )
    groups = split_batched_output(
        output,
        prompt_texts=["p0", "p1"],
        prompt_ids_list=[[5, 6], [7]],
        padded_prompt_length=3,
        params=params,
        tokenizer=tokenizer,
    )
    assert [group[0].prompt_ids for group in groups] == [[5, 6], [7]]
    assert groups[0][0].response_ids == [10, 11, 1]
    assert groups[0][0].truncated is False
    assert groups[0][1].response_ids == [12, 13, 14]
    assert groups[0][1].truncated is True
    assert groups[1][0].response_ids == [15, 1]
    assert groups[1][0].prompt_ids == [7]
    assert groups[1][1].truncated is True


def test_mock_generate_batch_matches_single():
    tokenizer = FakeTokenizer()
    params = SamplingParams(num_generations=2)
    generator = MockGenerator(
        responses=["\\boxed{72}", "I am not sure."],
        tokenizer=tokenizer,
        params=params,
    )
    batched = generator.generate_batch(["p0", "p1", "p2"])
    assert len(batched) == 3
    for index, prompt in enumerate(["p0", "p1", "p2"]):
        single = generator.generate(prompt)
        assert [item.response_text for item in batched[index]] == [
            item.response_text for item in single
        ]
        assert [item.prompt_ids for item in batched[index]] == [
            item.prompt_ids for item in single
        ]


def _trainer_config(output_dir: str, **training) -> dict:
    config = {
        "seed": 0,
        "model": {"device": "cpu"},
        "data": {"train_file": "data/processed/gsm8k/train.parquet"},
        "rollout": {"num_generations": 2, "max_new_tokens": 32},
        "training": {
            "learning_rate": 1e-3,
            "max_steps": 1,
            "prompts_per_batch": 1,
            "grad_accum_steps": 1,
            "output_dir": output_dir,
        },
        "algorithm": {"clip_ratio": 0.2, "kl_coef": 0.0},
        "reward": {"correct_reward": 1.0, "incorrect_reward": 0.0},
    }
    config["training"].update(training)
    return config


def test_trainer_grad_accum_batches_rollout():
    """grad_accum_steps=2 时，一个优化步消费 2 道题、产生 2 个 micro-batch。"""
    from GRPO.trainer import GRPOTrainer

    tokenizer = FakeTokenizer()
    generator = MockGenerator(
        responses=["\\boxed{72}", "I am not sure."],
        tokenizer=tokenizer,
        params=SamplingParams(num_generations=2),
    )
    with tempfile.TemporaryDirectory() as directory:
        config = _trainer_config(
            str(Path(directory) / "run"), grad_accum_steps=2
        )
        trainer = GRPOTrainer(
            config, model=_tiny_model(), tokenizer=tokenizer, generator=generator
        )
        metrics = trainer.train_step()
        assert trainer._cursor == 2, "一个优化步应消费 prompts_per_batch × grad_accum_steps 道题"
        assert metrics["skipped_ratio"] == 0.0
        assert math.isfinite(metrics["loss"])
        assert metrics["num_response_tokens"] > 0
        trainer.logger.close()


def test_trainer_skips_zero_gradient_step():
    """零优势步：不反传、参数不变，但奖励/熵等诊断指标仍然记录。"""
    import copy

    from GRPO.trainer import GRPOTrainer

    tokenizer = FakeTokenizer()
    model = copy.deepcopy(_tiny_model())
    # 两条回答都对（第一题答案为 72）→ 组内奖励相同 → 优势全 0
    generator = MockGenerator(
        responses=["\\boxed{72}"],
        tokenizer=tokenizer,
        params=SamplingParams(num_generations=2),
    )
    with tempfile.TemporaryDirectory() as directory:
        config = _trainer_config(
            str(Path(directory) / "run"), skip_zero_gradient_steps=True
        )
        trainer = GRPOTrainer(
            config, model=model, tokenizer=tokenizer, generator=generator
        )
        before = [parameter.detach().clone() for parameter in model.parameters()]
        metrics = trainer.train_step()
        after = [parameter.detach() for parameter in model.parameters()]

        assert metrics["skipped_ratio"] == 1.0
        assert metrics["skipped_micro_batches"] == 1.0
        assert "loss" not in metrics, "跳过的步不应伪造 loss"
        assert metrics["grad_norm"] == 0.0
        assert math.isfinite(metrics["policy_entropy"])
        assert all(
            torch.equal(old, new) for old, new in zip(before, after)
        ), "零梯度步不应改变参数"
        trainer.logger.close()


TESTS = [
    ("advantage: 手算组内优势", test_advantage_manual),
    ("loss: sequence/token 与 KL", test_loss_manual),
    ("rollout: Buffer 对齐与 log prob", test_buffer_and_log_probs),
    ("metrics: 日志与奖励统计", test_metrics_logger),
    ("trainer: 微型模型单步更新", test_trainer_one_step),
    ("rollout: 批量解码按题切分", test_split_batched_output),
    ("rollout: Mock 批量生成与单题一致", test_mock_generate_batch_matches_single),
    ("trainer: 梯度累积消费多道题", test_trainer_grad_accum_batches_rollout),
    ("trainer: 零优势步跳过反传", test_trainer_skips_zero_gradient_step),
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
