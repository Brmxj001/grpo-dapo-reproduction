"""配对比较两次评测（同一批题）：准确率差、McNemar 检验、回答长度变化。

用法：
    python scripts/compare_evals.py outputs/eval_base outputs/eval_lora_step300
"""
import argparse
import json
from math import sqrt
from pathlib import Path


def load(path: Path):
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {int(r["index"]): r for r in rows}


def mcnemar_exact(b, c):
    """精确二项检验（双侧），b/c 为不一致配对数。"""
    n = b + c
    if n == 0:
        return 1.0
    # 双侧 p = 2 * P(X <= min(b,c)), X ~ Bin(n, 0.5)
    k = min(b, c)
    prob = sum(_comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * prob)


def _comb(n, k):
    from math import comb

    return comb(n, k)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_dir", help="基线评测目录（含 generations.jsonl）")
    ap.add_argument("new_dir", help="新模型评测目录")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    base = load(Path(args.base_dir) / "generations.jsonl")
    new = load(Path(args.new_dir) / "generations.jsonl")
    common = sorted(set(base) & set(new))
    if not common:
        raise SystemExit("两次评测没有共同的题目 index")

    both = sum(1 for i in common if base[i]["correct"] and new[i]["correct"])
    only_base = sum(1 for i in common if base[i]["correct"] and not new[i]["correct"])
    only_new = sum(1 for i in common if not base[i]["correct"] and new[i]["correct"])
    neither = len(common) - both - only_base - only_new

    acc_base = (both + only_base) / len(common)
    acc_new = (both + only_new) / len(common)
    delta = acc_new - acc_base
    se = sqrt((only_base + only_new) / len(common)) / len(common) ** 0.5
    p = mcnemar_exact(only_base, only_new)

    len_base = sum(base[i]["response_len"] for i in common) / len(common)
    len_new = sum(new[i]["response_len"] for i in common) / len(common)
    trunc_base = sum(1 for i in common if base[i].get("truncated")) / len(common)
    trunc_new = sum(1 for i in common if new[i].get("truncated")) / len(common)

    print(f"共同题目数        {len(common)}")
    print(f"基线准确率        {acc_base:.4f}  ({both + only_base})")
    print(f"新模型准确率      {acc_new:.4f}  ({both + only_new})")
    print(f"差值              {delta:+.4f}  (±{1.96 * se:.4f}, 95% CI 近似)")
    print(f"配对表            都对 {both} | 仅基线对 {only_base} | 仅新模型对 {only_new} | 都错 {neither}")
    print(f"McNemar 精确检验  p = {p:.4f}")
    print(f"平均回答长度      {len_base:.1f} → {len_new:.1f}  ({len_new - len_base:+.1f})")
    print(f"截断率            {trunc_base:.2%} → {trunc_new:.2%}")

    summary = {
        "n": len(common),
        "accuracy_base": acc_base,
        "accuracy_new": acc_new,
        "delta": delta,
        "ci95_halfwidth": 1.96 * se,
        "both_correct": both,
        "only_base": only_base,
        "only_new": only_new,
        "neither": neither,
        "mcnemar_p": p,
        "length_base": len_base,
        "length_new": len_new,
        "truncation_base": trunc_base,
        "truncation_new": trunc_new,
    }
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"已写入 {args.json_out}")


if __name__ == "__main__":
    main()
