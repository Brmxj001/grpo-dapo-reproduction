"""汇总 metrics.jsonl：整体统计 + 分桶趋势，用于训练报告。"""
import argparse
import json
import statistics as st
from pathlib import Path

KEYS = [
    "reward_mean",
    "zero_variance_ratio",
    "policy_entropy",
    "avg_response_length",
    "truncation_rate",
    "kl",
    "clip_fraction",
    "grad_norm",
    "step_time",
]


def load(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics", help="metrics.jsonl 路径")
    ap.add_argument("--bucket", type=int, default=50, help="分桶步数")
    ap.add_argument("--json-out", default="", help="把汇总写成 json")
    args = ap.parse_args()

    rows = load(Path(args.metrics))
    if not rows:
        raise SystemExit("空文件")

    summary = {"steps": len(rows), "first_step": rows[0]["step"], "last_step": rows[-1]["step"]}
    print(f"steps={summary['steps']}  range=[{summary['first_step']}, {summary['last_step']}]")
    for key in KEYS:
        vals = [r[key] for r in rows if key in r]
        if not vals:
            continue
        summary[key] = {"mean": st.mean(vals), "last": vals[-1]}
        print(f"  {key:20s} mean={st.mean(vals):8.4f}  last={vals[-1]:8.4f}")

    if any("gpu_mem_gb" in r for r in rows):
        peak = max(r.get("gpu_mem_gb", 0) for r in rows)
        summary["gpu_mem_gb_peak"] = peak
        print(f"  {'gpu_mem_gb_peak':20s} {peak:8.2f}")

    total_h = sum(r.get("step_time", 0) for r in rows) / 3600
    summary["total_hours"] = total_h
    print(f"  {'total_hours':20s} {total_h:8.2f}")

    buckets = []
    start = rows[0]["step"]
    end = rows[-1]["step"]
    print("分桶趋势:")
    for lo in range(start, end + 1, args.bucket):
        b = [r for r in rows if lo <= r["step"] < lo + args.bucket]
        if not b:
            continue
        item = {
            "from": lo,
            "to": b[-1]["step"],
            "reward": st.mean(r["reward_mean"] for r in b),
            "zero_var": st.mean(r["zero_variance_ratio"] for r in b),
            "entropy": st.mean(r["policy_entropy"] for r in b),
            "length": st.mean(r["avg_response_length"] for r in b),
            "truncation": st.mean(r["truncation_rate"] for r in b),
            "rewarded": sum(1 for r in b if r["reward_mean"] > 0),
            "n": len(b),
        }
        buckets.append(item)
        print(
            f"  [{item['from']:3d}-{item['to']:3d}] rew={item['reward']:.3f} "
            f"zvar={item['zero_var']:.2f} ent={item['entropy']:.3f} "
            f"len={item['length']:.0f} trunc={item['truncation']:.2f} "
            f"有奖励步={item['rewarded']}/{item['n']}"
        )

    summary["buckets"] = buckets
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写入 {args.json_out}")


if __name__ == "__main__":
    main()
