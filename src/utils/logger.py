"""训练日志：JSONL 指标记录、控制台输出与训练曲线绘制。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

DEFAULT_PLOT_METRICS = (
    "reward_mean",
    "reward_std",
    "policy_entropy",
    "kl",
    "zero_variance_ratio",
    "avg_response_length",
    "policy_loss",
    "clip_fraction",
    "effective_sample_ratio",
)


class TrainingLogger:
    """把每次记录的指标追加写入 ``metrics.jsonl``，可选同步 TensorBoard。"""

    def __init__(self, log_dir: str | Path, use_tensorboard: bool = True, verbose: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.log_dir / "metrics.jsonl"
        self.history: list[dict[str, Any]] = []
        self.verbose = verbose
        self._file = self.metrics_path.open("a", encoding="utf-8")

        self._writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._writer = SummaryWriter(log_dir=str(self.log_dir / "tensorboard"))
            except Exception as exc:  # tensorboard 缺失不影响训练
                logger.info("TensorBoard 不可用，仅写 JSONL：%s", exc)

    def log(self, step: int, metrics: dict[str, Any], split: str = "train") -> None:
        record: dict[str, Any] = {"step": int(step), "split": split}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                record[key] = float(value)
        self.history.append(record)
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

        if self._writer is not None and split == "train":
            for key, value in record.items():
                if key not in ("step", "split"):
                    self._writer.add_scalar(f"{split}/{key}", value, step)

        if self.verbose:
            formatted = " ".join(
                f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}"
                for key, value in record.items()
            )
            print(f"[{split}] {formatted}")

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
        if self._writer is not None:
            self._writer.close()

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else self.log_dir / "metrics_history.json"
        target.write_text(json.dumps(self.history, ensure_ascii=False, indent=2), encoding="utf-8")
        return target


def plot_training_curves(
    history: Iterable[dict[str, Any]],
    output_dir: str | Path,
    metrics: Iterable[str] = DEFAULT_PLOT_METRICS,
    split: str = "train",
) -> list[Path]:
    """按指标逐张绘制 step-曲线，返回生成的 PNG 路径列表。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
