"""训练指标：累计、JSONL 记录与曲线绘制。

trainer.py 每个更新步调用 MetricsLogger.log()，日志写入
``<output_dir>/logs/metrics.jsonl``，训练结束后可调用 plot_metrics() 生成 PNG。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

DEFAULT_PLOT_METRICS = (
    "loss",
    "policy_loss",
    "kl",
    "policy_entropy",
    "reward_mean",
    "reward_std",
    "clip_fraction",
    "ratio_mean",
    "avg_response_length",
    "truncation_rate",
    "grad_norm",
    "gpu_mem_gb",
)


class MetricsLogger:
    def __init__(self, log_dir: str | Path, verbose: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "metrics.jsonl"
        self.history: list[dict] = []
        self.verbose = verbose
        self._file = self.log_path.open("a", encoding="utf-8")

    def log(self, step: int, metrics: Mapping[str, float], split: str = "train") -> None:
        record: dict = {"step": int(step), "split": split}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                record[key] = float(value)
        self.history.append(record)
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

        if self.verbose:
            formatted = " ".join(
                f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
                for key, value in record.items()
            )
            # flush=True：stdout 重定向到文件时默认是块缓冲（4–8 KB），
            # 不显式 flush 会让 "实时日志" 延迟几十条才出现。
            print(f"[{split}] {formatted}", flush=True)

    def truncate_after(self, step: int) -> int:
        """只保留 step 及之前的记录，返回删除条数。

        断点续训时，权重会回退到检查点；继续训练会重新产生这些步的指标。
        若不清理，metrics.jsonl 里会出现重复/前后矛盾的步号，曲线也会错乱。
        同时把保留的记录载入 ``history``，续训结束后的曲线仍覆盖完整训练过程。
        """
        self._file.flush()
        records: list[dict] = []
        if self.log_path.exists():
            for line in self.log_path.read_text(encoding="utf-8").splitlines():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        kept = [record for record in records if int(record.get("step", 0)) <= step]
        removed = len(records) - len(kept)
        if removed:
            self._file.close()
            self.log_path.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept),
                encoding="utf-8",
            )
            self._file = self.log_path.open("a", encoding="utf-8")
        self.history = kept
        return removed

    def summary(self, split: str = "train") -> dict[str, float]:
        """返回当前 history 中各指标的平均值。"""
        records = [record for record in self.history if record.get("split") == split]
        if not records:
            return {}
        keys = [key for key in records[-1] if key not in ("step", "split")]
        return {
            key: sum(record.get(key, 0.0) for record in records) / len(records)
            for key in keys
        }

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else self.log_dir / "metrics_history.json"
        target.write_text(
            json.dumps(self.history, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return target

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


def reward_statistics(rewards: Sequence[float]) -> dict[str, float]:
    """奖励均值、标准差（总体）与零方差比例（按单条回答计算）。"""
    if not rewards:
        return {"reward_mean": 0.0, "reward_std": 0.0, "zero_variance_ratio": 0.0}
    mean = sum(rewards) / len(rewards)
    variance = sum((reward - mean) ** 2 for reward in rewards) / len(rewards)
    return {
        "reward_mean": float(mean),
        "reward_std": float(variance ** 0.5),
        "zero_variance_ratio": float(all(abs(r - rewards[0]) < 1e-8 for r in rewards)),
    }


def plot_metrics(
    history: Iterable[dict],
    output_dir: str | Path,
    metrics: Iterable[str] = DEFAULT_PLOT_METRICS,
    split: str = "train",
) -> list[Path]:
    """按指标逐张绘制 step-曲线；matplotlib 缺失或没有数据时返回空列表。"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = [record for record in history if record.get("split") == split]

    saved: list[Path] = []
    for metric in metrics:
        points = [(record["step"], record[metric]) for record in records if metric in record]
        if not points:
            continue
        steps, values = zip(*points)
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.plot(steps, values, marker="o", markersize=3)
        axis.set_xlabel("step")
        axis.set_ylabel(metric)
        axis.set_title(metric)
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        path = output_dir / f"{metric}.png"
        figure.savefig(path, dpi=120)
        plt.close(figure)
        saved.append(path)
    return saved
