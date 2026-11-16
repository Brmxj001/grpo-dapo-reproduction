#!/usr/bin/env python3
"""使用 Qwen2.5-0.5B-Instruct 做 CPU 推理冒烟测试。

使用模型原生 Chat Template 构造输入，调用 generate() 生成回答，
并输出 Token 数量与推理耗时。本脚本不进行任何梯度更新。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from src.model_loader import get_model_info, load_model, load_tokenizer

QUESTIONS = [
    "2+3等于多少？",
    "请解释什么是强化学习。",
    "请简单解释 PPO 和 GRPO 的区别。",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen2.5-0.5B-Instruct CPU 推理测试")
    parser.add_argument(
        "--model-path",
        default=None,
        help="本地模型目录或 HF repo id（默认使用项目内 models/Qwen2.5-0.5B-Instruct）",
    )
    parser.add_argument("--max-new-tokens", type=int, default=192, help="每次生成的最大新 Token 数")
    parser.add_argument("--temperature", type=float, default=0.7, help="采样温度")
    parser.add_argument("--top-p", type=float, default=0.9, help="nucleus 采样阈值")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    tokenizer = load_tokenizer(args.model_path)
    model = load_model(args.model_path)
    device = next(model.parameters()).device

    info = get_model_info(model, tokenizer)
    print("=" * 70)
    print(f"模型：{info['model_class']} | 设备：{info['device']} | 精度：{info['dtype']}")
    print(f"参数量：{info['num_parameters']:,}（{info['num_parameters'] / 1e6:.1f}M）")
    print(f"生成参数：max_new_tokens={args.max_new_tokens}, "
          f"do_sample=True, temperature={args.temperature}, top_p={args.top_p}")
    print("=" * 70)

    failures = 0
    for index, question in enumerate(QUESTIONS, start=1):
        messages = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        start = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed = time.perf_counter() - start

        new_tokens = output_ids[0, input_length:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        hit_eos = bool(
            tokenizer.eos_token_id is not None
            and new_tokens.numel() > 0
            and new_tokens[-1].item() == tokenizer.eos_token_id
        )

        print(f"\n[问题 {index}] {question}")
        print(f"[输入 Token] {input_length}  [输出 Token] {new_tokens.numel()}  "
              f"[耗时] {elapsed:.2f}s  [速度] {new_tokens.numel() / elapsed:.2f} tok/s  "
              f"[EOS] {hit_eos}")
        print(f"[回答]\n{answer}")

        if not answer:
            failures += 1
            print("!! 警告：模型输出为空，生成失败。")

    print("\n" + "=" * 70)
    if failures:
        print(f"推理测试失败：{failures}/{len(QUESTIONS)} 个问题输出为空。")
        return 1
    print(f"推理测试通过：{len(QUESTIONS)}/{len(QUESTIONS)} 个问题均生成了非空回答。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
