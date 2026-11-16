"""GSM8K 数据预处理：读取原始 parquet，整理题目、参考解法与标准答案。

原始 answer 字段形如：
    Natalia sold 48/2 = <<48/2=24>>24 clips in May.
    ...
    #### 72

处理后输出三列：
- question: 题目文本
- solution: 去掉 <<...>> 计算器注解的参考解法（不含 #### 行）
- ground_truth: #### 之后的最终答案，保留原始写法（如 "1,080"）
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "gsm8k" / "main"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "gsm8k"

ANSWER_MARKER = "####"
REQUIRED_COLUMNS = ("question", "answer")
SPLITS = ("train", "test")
CALC_ANNOTATION_RE = re.compile(r"<<[^<>]*>>")


def extract_ground_truth(answer: str) -> str:
    """取 #### 之后的最终答案；缺失或为空时抛出 ValueError。"""
    if ANSWER_MARKER not in answer:
        raise ValueError(f"answer 缺少 {ANSWER_MARKER} 标记：{answer!r}")
    ground_truth = answer.rsplit(ANSWER_MARKER, 1)[1].strip()
    if not ground_truth:
        raise ValueError(f"#### 之后没有答案：{answer!r}")
    return ground_truth


def clean_solution(answer: str) -> str:
    """去掉 #### 行与 <<计算>> 注解，得到纯文本参考解法。"""
    solution = answer.rsplit(ANSWER_MARKER, 1)[0]
    solution = CALC_ANNOTATION_RE.sub("", solution)
    return solution.strip()


def preprocess_frame(df: pd.DataFrame) -> pd.DataFrame:
    """把含 question/answer 的原始 DataFrame 转成预处理后的三列数据。"""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"缺少字段：{missing}")
    if len(df) == 0:
        raise ValueError("输入数据为空")

    processed = pd.DataFrame(
        {
            "question": df["question"].str.strip(),
            "solution": df["answer"].map(clean_solution),
            "ground_truth": df["answer"].map(extract_ground_truth),
        }
    )
    if processed.isnull().any().any():
        raise ValueError("预处理结果中存在缺失值")
    return processed


def load_raw_split(split: str, data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    path = Path(data_dir) / f"{split}-00000-of-00001.parquet"
    if not path.exists():
        raise FileNotFoundError(f"原始数据不存在：{path}")
    return pd.read_parquet(path)


def process_split(
    split: str,
    data_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> Path:
    df = preprocess_frame(load_raw_split(split, data_dir))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}.parquet"
    df.to_parquet(output_path, index=False)
    print(f"[{split}] {len(df)} 条 -> {output_path}")
    print(f"  示例 ground_truth: {df['ground_truth'].iloc[0]}")
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GSM8K 数据预处理")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="原始 parquet 目录")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="预处理结果输出目录")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=list(SPLITS),
                        help="要处理的 split")
    args = parser.parse_args(argv)

    for split in args.splits:
        process_split(split, args.data_dir, args.output_dir)
    print("预处理完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
