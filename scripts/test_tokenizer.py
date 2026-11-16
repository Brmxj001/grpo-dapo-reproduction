#!/usr/bin/env python3
"""Qwen2.5 Tokenizer 功能测试：中英文、数学公式、特殊 Token、Chat Template 与解码。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_loader import load_tokenizer

TEXTS = {
    "中文": "强化学习是一种让智能体通过与环境交互来学习最优策略的方法。",
    "英文": "Reinforcement learning trains an agent by maximizing cumulative reward.",
    "数学公式": r"设 $x^2 + 2x + 1 = 0$，则 $(x+1)^2 = 0$，所以 $x = -1$。",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Qwen2.5 Tokenizer 测试")
    parser.add_argument("--model-path", default=None, help="本地模型目录或 HF repo id")
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.model_path)
    checks: list[tuple[str, bool, str]] = []

    def record(name: str, passed: bool, detail: str = "") -> None:
        checks.append((name, passed, detail))
        print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))

    print("=" * 70)
    print(f"Tokenizer 类型：{type(tokenizer).__name__}  词表大小：{len(tokenizer)}")
    print(f"EOS: {tokenizer.eos_token!r} ({tokenizer.eos_token_id})  "
          f"PAD: {tokenizer.pad_token!r} ({tokenizer.pad_token_id})")
    print("=" * 70)

    # 1) 中英文与数学公式编码 + 解码往返
    for label, text in TEXTS.items():
        ids = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=True)
        print(f"\n[{label}] 原文：{text}")
        print(f"[{label}] Token 数：{len(ids)}  ID 前 20 个：{ids[:20]}")
        print(f"[{label}] 解码：{decoded}")
        record(f"{label}编码非空", len(ids) > 0, f"{len(ids)} tokens")
        record(f"{label}解码往返一致", decoded == text)

    # 2) 特殊 Token
    print("\n特殊 Token 映射：")
    for name, token in tokenizer.special_tokens_map.items():
        print(f"  {name}: {token!r}")
    for special in ("<|im_start|>", "<|im_end|>", "<|endoftext|>"):
        ids = tokenizer.encode(special, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        record(
            f"特殊 Token {special} 编解码",
            len(ids) == 1 and decoded == special,
            f"ids={ids}, decoded={decoded!r}",
        )

    # 3) Chat Template（原生模板，多轮 + generation prompt）
    messages = [
        {"role": "system", "content": "你是一个简洁的数学助手。"},
        {"role": "user", "content": "2+3等于多少？"},
        {"role": "assistant", "content": "2+3等于5。"},
        {"role": "user", "content": "再加 4 呢？"},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    print("\nChat Template 渲染结果：")
    print(prompt_text)
    record("Chat Template 以 <|im_start|>system 开头", prompt_text.startswith("<|im_start|>system"))
    record(
        "Chat Template 以 <|im_start|>assistant 结尾",
        prompt_text.rstrip().endswith("<|im_start|>assistant"),
    )
    record("Chat Template Token 化非空", len(prompt_ids) > 0, f"{len(prompt_ids)} tokens")

    # 4) 解码显示控制：skip_special_tokens 两种模式
    simple_msgs = [{"role": "user", "content": "你好"}]
    ids = tokenizer.apply_chat_template(simple_msgs, tokenize=True, add_generation_prompt=True)
    raw = tokenizer.decode(ids, skip_special_tokens=False)
    clean = tokenizer.decode(ids, skip_special_tokens=True)
    print(f"\n含特殊 Token 解码：{raw!r}")
    print(f"跳过特殊 Token 解码：{clean!r}")
    record("skip_special_tokens 保留模板标记", "<|im_start|>" in raw)
    record("skip_special_tokens 去掉模板标记", "<|im_start|>" not in clean)

    failures = [name for name, passed, _ in checks if not passed]
    print("\n" + "=" * 70)
    if failures:
        print(f"Tokenizer 测试失败：{len(failures)}/{len(checks)} 项未通过")
        for name in failures:
            print(f"  - {name}")
        return 1
    print(f"Tokenizer 测试通过：{len(checks)}/{len(checks)} 项全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
