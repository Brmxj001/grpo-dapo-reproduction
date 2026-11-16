"""训练循环 CPU 冒烟测试：微型随机模型 + Mock 采样，执行一次真实反传更新。

不加载 Qwen2.5-0.5B，不进行真实训练。

    python tests/test_trainer_smoke.py
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rollout.generator import MockPolicyGenerator, SamplingParams  # noqa: E402
from src.utils.config import load_config  # noqa: E402


class FakeTokenizer:
    """只实现训练循环用到的最小 Tokenizer 接口（伪 token，非真实词表）。"""

    chat_template = "fake-chat-template"
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        text = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        return text + "\nassistant:" if add_generation_prompt else text

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        # 截断到 20 个伪 token，保证 prompt + response 不超过微型模型的位置上限
        return [2 + (ord(char) % 60) for char in text][:20]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return "".join(chr(97 + (int(token) % 26)) for token in ids)


def _tiny_model():
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
    return Qwen2ForCausalLM(config)


def test_dry_run_script():
    result = subprocess.run(
        [sys.executable, "scripts/train.py", "--config", "configs/dapo.yaml", "--dry-run"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "dry-run 完成" in result.stdout
    assert "训练集 7473 条" in result.stdout


def test_one_train_step_with_tiny_model():
    from src.trainer import GRPODAPOTrainer

    config = load_config("configs/dapo.yaml")
    config["model"]["device"] = "cpu"
    config["lora"]["enabled"] = False
    config["rollout"]["num_generations"] = 2
    config["training"]["max_steps"] = 1
    config["training"]["prompts_per_batch"] = 1
    config["training"]["warmup_ratio"] = 0.0

    tokenizer = FakeTokenizer()
    model = _tiny_model()
    # GSM8K train 第一条题为 Natalia 卖 clip，标准答案 72。
    # 两条回答长度不同，避免一正一负优势在等长序列上梯度完全抵消。
    responses = ["\\boxed{72}", "I am not sure."]
    generator = MockPolicyGenerator(
        responses=responses,
        tokenizer=tokenizer,
        params=SamplingParams(num_generations=2),
    )

    with tempfile.TemporaryDirectory() as directory:
        config["training"]["output_dir"] = str(Path(directory) / "run")
        trainer = GRPODAPOTrainer(
            config, model=model, tokenizer=tokenizer, generator=generator
        )
        before = model.model.embed_tokens.weight.detach().clone()
        metrics = trainer.train_step()
        after = model.model.embed_tokens.weight.detach()

        assert abs(metrics["reward_mean"] - 0.5) < 1e-6
        assert abs(metrics["reward_std"] - 0.5) < 1e-6
        assert torch.isfinite(torch.tensor(metrics["loss"]))
        assert metrics["avg_response_length"] > 0
        assert metrics["grad_norm"] > 0, "长度不同的回答应产生非零梯度"
        assert not torch.allclose(before, after), "训练一步后参数应发生变化"
        assert (Path(directory) / "run/logs/metrics.jsonl").exists()
        trainer.logger.close()


TESTS = [
    ("训练入口: dry-run", test_dry_run_script),
    ("训练循环: 微型模型单步更新", test_one_train_step_with_tiny_model),
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
