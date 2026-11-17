"""在 GSM8K 测试集上评测模型（基线与 LoRA 训练后共用）。

用法示例：
    # 基座模型，全量测试集，贪心解码
    python scripts/evaluate_gsm8k.py --output outputs/eval_base

    # 训练后的 LoRA adapter
    python scripts/evaluate_gsm8k.py \
        --adapter outputs/E1_grpo_lora/checkpoints/step_500 \
        --output outputs/eval_lora_step500

    # 先跑 20 条探速度
    python scripts/evaluate_gsm8k.py --limit 20 --output /tmp/eval_bench

输出（都在 --output 目录下）：
    generations.jsonl  每个样本一行（题目、标准答案、模型回答、是否正确）
    summary.json       汇总指标（准确率、平均长度、截断率、耗时）

关于批处理：多道题（左 padding）一起送进 generate，吞吐比逐题高一个数量级。
左 padding 是必须的 —— 右 padding 时生成会从 pad 位置接下去，结果全错。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import torch

from src.data.prompt import build_prompt
from src.model_loader import load_model, load_tokenizer, resolve_device
from src.rewards.reward import RewardConfig, compute_reward
from src.utils.config import load_config, resolve_path
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GSM8K 测试集评测")
    parser.add_argument("--config", default="configs/grpo.yaml", help="读取数据路径与奖励配置")
    parser.add_argument("--model-path", default=None, help="默认取 config 里的 model.local_path")
    parser.add_argument("--adapter", default=None, help="PEFT adapter 目录（不填则评基座）")
    parser.add_argument("--test-file", default="data/processed/gsm8k/test.parquet")
    parser.add_argument("--limit", type=int, default=0, help="评测前 N 条，0 表示全量")
    parser.add_argument("--offset", type=int, default=0, help="从第几条开始")
    parser.add_argument("--batch-size", type=int, default=16, help="一次生成多少道题")
    parser.add_argument("--num-generations", type=int, default=1, help="每题采样几条（>1 时统计 pass@k）")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="0 表示贪心解码（可复现），>0 表示按该温度采样",
    )
    parser.add_argument("--output", default="outputs/eval_base")
    return parser.parse_args()


def strip_after_eos(token_ids: list[int], eos_token_id: int | None) -> tuple[list[int], bool]:
    """截断到首个 EOS（含），返回 (ids, 是否没遇到 EOS 就被截断)。"""
    if eos_token_id is not None and eos_token_id in token_ids:
        return token_ids[: token_ids.index(eos_token_id) + 1], False
    return token_ids, True


@torch.inference_mode()
def generate_batch(
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    num_generations: int,
    device,
) -> list[list[list[int]]]:
    """返回 [题目][候选回答][token ids]。"""
    results: list[list[list[int]]] = []
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = eos_token_id

    for start in range(0, len(prompts), batch_size):
        chunk = prompts[start : start + batch_size]
        # 每题重复 num_generations 次，interleave 后按题拆回来
        expanded = [text for text in chunk for _ in range(num_generations)]
        encoded = tokenizer(
            expanded, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(device)
        prompt_length = encoded["input_ids"].shape[1]

        outputs = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=1.0,
        )

        generated = outputs[:, prompt_length:].tolist()
        for offset in range(0, len(generated), num_generations):
            candidates = []
            for ids in generated[offset : offset + num_generations]:
                ids, _ = strip_after_eos(ids, eos_token_id)
                candidates.append(ids)
            results.append(candidates)
    return results


def main() -> None:
    args = parse_args()

    # 一次评测只用一套 RNG 状态，保证同参数下可复现
    set_seed(42)

    config = load_config(args.config)
    model_cfg = config.get("model", {})
    device = resolve_device(model_cfg.get("device", "auto"))

    model_path = args.model_path or model_cfg.get(
        "local_path", "models/Qwen2.5-0.5B-Instruct"
    )
    tokenizer = load_tokenizer(model_path)
    model = load_model(model_path, device=device)

    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(resolve_path(args.adapter)))
        model = model.merge_and_unload()
        print(f"已加载并合并 LoRA adapter：{args.adapter}")

    model.eval()
    # 左 padding：右 padding 会让生成从 pad 位置续写，批量结果全错
    tokenizer.padding_side = "left"

    reward_cfg = config.get("reward", {})
    reward_config = RewardConfig(
        correct_reward=float(reward_cfg.get("correct_reward", 1.0)),
        incorrect_reward=float(reward_cfg.get("incorrect_reward", 0.0)),
        format_weight=float(reward_cfg.get("format_weight", 0.0)),
    )

    frame = pd.read_parquet(resolve_path(args.test_file))
    total_available = len(frame)
    end = total_available if args.limit <= 0 else min(args.offset + args.limit, total_available)
    indices = list(range(args.offset, end))

    output_dir = resolve_path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    generations_path = output_dir / "generations.jsonl"

    print(
        f"评测 {len(indices)} / {total_available} 条 | batch={args.batch_size} "
        f"G={args.num_generations} max_new_tokens={args.max_new_tokens} "
        f"temperature={args.temperature}"
    )

    start = time.perf_counter()
    solved = 0
    num_rows = 0
    total_tokens = 0
    truncated_count = 0
    done = 0

    with generations_path.open("w", encoding="utf-8") as handle:
        for batch_start in range(0, len(indices), args.batch_size):
            batch_indices = indices[batch_start : batch_start + args.batch_size]
            prompts = [
                build_prompt(frame["question"][index], tokenizer) for index in batch_indices
            ]
            candidates = generate_batch(
                model,
                tokenizer,
                prompts,
                batch_size=len(prompts),
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                num_generations=args.num_generations,
                device=device,
            )

            for index, prompt, token_lists in zip(batch_indices, prompts, candidates):
                ground_truth = frame["ground_truth"][index]
                row_solved = False
                for candidate_index, ids in enumerate(token_lists):
                    response_text = tokenizer.decode(ids, skip_special_tokens=True)
                    result = compute_reward(
                        response_text, ground_truth, config=reward_config
                    )
                    record = {
                        "index": int(index),
                        "candidate": candidate_index,
                        "question": frame["question"][index],
                        "ground_truth": ground_truth,
                        "response": response_text,
                        "correct": bool(result.correct),
                        "reward": float(result.reward),
                        "response_len": len(ids),
                        "truncated": bool(len(ids) >= args.max_new_tokens),
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    num_rows += 1
                    total_tokens += len(ids)
                    truncated_count += int(record["truncated"])
                    row_solved = row_solved or result.correct
                solved += int(row_solved)
                done += 1

            handle.flush()
            elapsed = time.perf_counter() - start
            rate = done / elapsed if elapsed > 0 else 0.0
            remain = (len(indices) - done) / rate if rate > 0 else 0.0
            print(
                f"  {done}/{len(indices)} | 准确率 {solved/done:.3f} | "
                f"{rate:.2f} 题/秒 | 预计剩余 {remain/60:.1f} 分钟",
                flush=True,
            )

    elapsed = time.perf_counter() - start
    num_questions = len(indices)
    summary = {
        "model_path": model_path,
        "adapter": args.adapter,
        "test_file": args.test_file,
        "num_questions": num_questions,
        "num_candidates": num_rows,
        "num_generations": args.num_generations,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "solved_questions": solved,
        "accuracy": solved / num_questions if num_questions else 0.0,
        "avg_response_tokens": total_tokens / num_rows if num_rows else 0.0,
        "truncation_rate": truncated_count / num_rows if num_rows else 0.0,
        "elapsed_seconds": elapsed,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n=== 汇总 ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"结果已保存：{output_dir}")


if __name__ == "__main__":
    main()
