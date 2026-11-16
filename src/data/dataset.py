"""GSM8K 数据集封装：统一 Rollout / 训练使用的题目与标准答案格式。

处理后 parquet 应包含列：question、ground_truth（可选 solution）。
如需从原始 GSM8K parquet 直接构建，使用 ``MathReasoningDataset.from_raw``。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

REQUIRED_COLUMNS = ("question", "ground_truth")


@dataclass(frozen=True)
class MathProblem:
    index: int
    question: str
    ground_truth: str
    solution: str | None = None


class MathReasoningDataset(Dataset):
    """按行读取数学题数据集，供 Rollout 逐题采样。"""

    def __init__(self, frame: pd.DataFrame, source: str = ""):
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"数据集缺少字段：{missing}（来源：{source}）")
        if len(frame) == 0:
            raise ValueError(f"数据集为空：{source}")

        frame = frame.reset_index(drop=True)
        questions = frame["question"].astype(str).str.strip()
        ground_truths = frame["ground_truth"].astype(str).str.strip()
        if (questions == "").any() or (ground_truths == "").any():
            raise ValueError(f"存在空题目或空标准答案：{source}")

        solutions = (
            frame["solution"].astype(str) if "solution" in frame.columns else [None] * len(frame)
        )
        self.source = source
        self._problems = [
            MathProblem(
                index=index,
                question=questions.iloc[index],
                ground_truth=ground_truths.iloc[index],
                solution=None if solutions[index] is None else str(solutions[index]),
            )
            for index in range(len(frame))
        ]

    @classmethod
    def from_processed(cls, path: str | Path) -> "MathReasoningDataset":
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(
                f"预处理数据不存在：{path}\n"
                "请先运行：python -m src.data.preprocess"
            )
        return cls(pd.read_parquet(path), source=str(path))

    @classmethod
    def from_raw(cls, path: str | Path) -> "MathReasoningDataset":
        """直接从原始 GSM8K parquet 构建（内部调用 preprocess）。"""
        from .preprocess import preprocess_frame

        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"原始数据不存在：{path}")
        return cls(preprocess_frame(pd.read_parquet(path)), source=str(path))

    def __len__(self) -> int:
        return len(self._problems)

    def __getitem__(self, index: int) -> MathProblem:
        return self._problems[index]

    @property
    def questions(self) -> list[str]:
        return [problem.question for problem in self._problems]


def load_train_and_test(
    train_file: str | Path,
    test_file: str | Path,
) -> tuple[MathReasoningDataset, MathReasoningDataset]:
    """加载训练集/测试集，并检查题目不重叠，保证测试集不参与训练。"""
    train_dataset = MathReasoningDataset.from_processed(train_file)
    test_dataset = MathReasoningDataset.from_processed(test_file)

    overlap = set(train_dataset.questions) & set(test_dataset.questions)
    if overlap:
        raise ValueError(f"训练集与测试集存在 {len(overlap)} 道重复题目，请检查数据划分")
    return train_dataset, test_dataset


def create_dataloader(
    dataset: MathReasoningDataset,
    batch_size: int = 1,
    shuffle: bool = False,
    seed: int = 42,
) -> DataLoader:
    """Rollout 用的 DataLoader，batch 为 MathProblem 列表（不定长，不默认 collate）。"""
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda batch: list(batch),
        generator=generator,
    )
