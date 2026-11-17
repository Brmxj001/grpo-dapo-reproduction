#!/usr/bin/env python3
"""GRPO/ 独立版训练入口。

仓库原本只有 scripts/train.py，它调用的是 src/trainer 的 GRPODAPOTrainer；
GRPO/ 目录下的独立实现没有入口脚本。本文件补上这个入口：

    python scripts/train_grpo_standalone.py --config configs/grpo.yaml

CPU 冒烟测试（几十秒，验证训练循环能跑通，不追求效果）：

    python scripts/train_grpo_standalone.py --config configs/grpo.yaml \
        --max-steps 1 --max-new-tokens 32 --num-generations 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO 独立版（GRPO/）训练入口")
    parser.add_argument("--config", required=True, help="YAML 配置路径")
    parser.add_argument("--max-steps", type=int, help="覆盖 training.max_steps")
    parser.add_argument("--prompts-per-batch", type=int, help="覆盖 training.prompts_per_batch")
    parser.add_argument("--max-new-tokens", type=int, help="覆盖 rollout.max_new_tokens")
    parser.add_argument("--num-generations", type=int, help="覆盖 rollout.num_generations")
    parser.add_argument(
        "--grad-accum-steps", type=int, help="覆盖 training.grad_accum_steps（梯度累积）"
    )
    parser.add_argument("--save-every", type=int, help="覆盖 training.save_every（0 = 不存检查点）")
    parser.add_argument("--output-dir", help="覆盖 training.output_dir")
    parser.add_argument("--resume", help="从检查点目录恢复训练，例如 outputs/.../checkpoints/step_100")
    parser.add_argument("--dry-run", action="store_true", help="只检查配置，不加载模型")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    # CLI 覆盖：便于先用极小配置做 CPU 冒烟测试
    if args.max_steps is not None:
        config.setdefault("training", {})["max_steps"] = args.max_steps
    if args.prompts_per_batch is not None:
        config.setdefault("training", {})["prompts_per_batch"] = args.prompts_per_batch
    if args.max_new_tokens is not None:
        config.setdefault("rollout", {})["max_new_tokens"] = args.max_new_tokens
    if args.num_generations is not None:
        config.setdefault("rollout", {})["num_generations"] = args.num_generations
    if args.grad_accum_steps is not None:
        config.setdefault("training", {})["grad_accum_steps"] = args.grad_accum_steps
    if args.save_every is not None:
        config.setdefault("training", {})["save_every"] = args.save_every
    if args.output_dir is not None:
        config.setdefault("training", {})["output_dir"] = args.output_dir

    experiment = config.get("experiment", {})
    training = config.get("training", {})
    rollout = config.get("rollout", {})
    effective_batch = int(training.get("prompts_per_batch", 1)) * int(
        training.get("grad_accum_steps", 1)
    )
    print(
        f"实验 {experiment.get('id', '?')} - {experiment.get('name', '?')} | "
        f"G={rollout.get('num_generations')} | "
        f"max_new_tokens={rollout.get('max_new_tokens')} | "
        f"B={training.get('prompts_per_batch')}×累积{training.get('grad_accum_steps', 1)}"
        f"=每步{effective_batch}题 | "
        f"max_steps={training.get('max_steps')}"
    )

    if args.dry_run:
        print("dry-run 完成：仅检查配置，未加载模型。")
        return 0

    from GRPO.trainer import GRPOTrainer

    trainer = GRPOTrainer(config)
    if args.resume:
        trainer.load_checkpoint(args.resume)
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
