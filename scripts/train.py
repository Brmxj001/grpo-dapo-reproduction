#!/usr/bin/env python3
"""GRPO / DAPO 训练入口。

本机（无 GPU）只验证 dry-run；真实训练请在租用的 GPU 机器上执行：

    python scripts/train.py --config configs/grpo.yaml
    python scripts/train.py --config configs/dapo.yaml

dry-run 只检查配置与数据，不加载模型、不训练：

    python scripts/train.py --config configs/grpo.yaml --dry-run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import load_train_and_test  # noqa: E402
from src.utils.config import load_config, resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRPO / DAPO 训练入口")
    parser.add_argument("--config", required=True, help="YAML 配置路径，例如 configs/grpo.yaml")
    parser.add_argument("--dry-run", action="store_true", help="只检查配置与数据，不训练")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    experiment = config.get("experiment", {})
    algorithm = config.get("algorithm", {})
    print(
        f"实验 {experiment.get('id', '?')} - {experiment.get('name', '?')} | "
        f"算法 {algorithm.get('name', '?')} | "
        f"G={config['rollout']['num_generations']} | "
        f"max_steps={config['training']['max_steps']}"
    )

    train_dataset, test_dataset = load_train_and_test(
        resolve_path(config["data"]["train_file"]),
        resolve_path(config["data"]["test_file"]),
    )
    print(f"训练集 {len(train_dataset)} 条，测试集 {len(test_dataset)} 条")

    if args.dry_run:
        print("dry-run 完成：配置与数据检查通过，未加载模型、未开始训练。")
        return 0

    from src.trainer import GRPODAPOTrainer

    trainer = GRPODAPOTrainer(config)
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
