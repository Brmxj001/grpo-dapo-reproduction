"""实时查看 GRPO 训练指标。

训练脚本每步都会把一行 JSON 追加并 flush 到
    <output_dir>/logs/metrics.jsonl
本脚本每隔几秒读一次这个文件，打印最新指标 + 最近若干步的滑动平均。

用法：
    python scripts/watch_training.py --output outputs/E1_grpo_lora
    python scripts/watch_training.py --output outputs/E1_grpo_lora --interval 5 --window 20

指标说明：
    kl            当前策略与参考策略的 k3 KL 估计（越大离基座越远）
    ent           策略熵（response token 平均），持续下降→策略在收敛，跌到 0 附近是塌缩
    rew           奖励均值（0~1，GSM8K 二进制题就是对的比例）
    zvar          组内零方差比例：该步所有回答奖励相同 → 没有学习信号
    ratio         πθ/πold 的均值，应贴近 1；远大于 1 说明单步更新过猛
    clip          被 PPO 裁剪的 token 比例，早期应接近 0
    grad          梯度范数（裁剪前），长期为 0 说明梯度断了
    len           平均回答长度（token）
    trunc         撞到 max_new_tokens 的比例，过高说明该加长度或题目太难
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="实时查看训练指标")
    parser.add_argument("--output", default="outputs/E1_grpo_lora", help="训练输出目录")
    parser.add_argument("--log", default=None, help="直接指定 metrics.jsonl 路径")
    parser.add_argument("--interval", type=float, default=5.0, help="刷新间隔（秒）")
    parser.add_argument("--window", type=int, default=20, help="滑动平均窗口（步）")
    parser.add_argument("--total-steps", type=int, default=500, help="用于计算进度与 ETA")
    return parser.parse_args()


def load_records(path: Path) -> list[dict]:
    """每次整读（500 行量级，开销可忽略），避免处理半行写入。"""
    records: list[dict] = []
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return records
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def mean_of(records: list[dict], key: str) -> float | None:
    values = [record[key] for record in records if key in record]
    if not values:
        return None
    return sum(values) / len(values)


def fmt(value: float | None, spec: str) -> str:
    return "-" if value is None else format(value, spec)


def main() -> None:
    args = parse_args()
    log_path = (
        Path(args.log) if args.log
        else resolve_path(args.output) / "logs" / "metrics.jsonl"
    )
    print(f"监听 {log_path}（每 {args.interval:g}s 刷新，窗口 {args.window} 步）")
    print(
        "step | 进度 | 剩余 | kl | ent | rew | zvar | ratio | clip | grad | len | trunc | s/step"
    )

    last_step = -1
    try:
        while True:
            records = load_records(log_path)
            if records:
                latest = records[-1]
                step = int(latest.get("step", 0))
                if step != last_step:
                    window = records[-args.window :]
                    step_times = [
                        record["step_time"] for record in window if "step_time" in record
                    ]
                    avg_step_time = (
                        sum(step_times) / len(step_times) if step_times else None
                    )
                    remaining = max(args.total_steps - step, 0)
                    eta = (
                        f"{remaining * avg_step_time / 3600:.2f}h"
                        if avg_step_time
                        else "-"
                    )
                    progress = 100.0 * step / args.total_steps if args.total_steps else 0.0
                    line = (
                        f"{step}/{args.total_steps} | {progress:5.1f}% | {eta:>5} | "
                        f"{fmt(latest.get('kl'), '.4f')} | "
                        f"{fmt(latest.get('policy_entropy'), '.3f')} | "
                        f"{fmt(mean_of(window, 'reward_mean'), '.2f')} | "
                        f"{fmt(mean_of(window, 'zero_variance_ratio'), '.2f')} | "
                        f"{fmt(latest.get('ratio_mean'), '.4f')} | "
                        f"{fmt(latest.get('clip_fraction'), '.4f')} | "
                        f"{fmt(mean_of(window, 'grad_norm'), '.3f')} | "
                        f"{fmt(mean_of(window, 'avg_response_length'), '.0f')} | "
                        f"{fmt(mean_of(window, 'truncation_rate'), '.2f')} | "
                        f"{fmt(avg_step_time, '.1f')}"
                    )
                    if sys.stdout.isatty():
                        sys.stdout.write("\r" + line + " " * 10 + "\n")
                    else:
                        sys.stdout.write(line + "\n")
                    sys.stdout.flush()
                    last_step = step
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n已停止监听（训练本身不受影响）")


if __name__ == "__main__":
    main()
