
from pathlib import Path
import pandas as pd

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# GSM8K 数据集目录
DATA_DIR = PROJECT_ROOT / "data/raw/gsm8k/main"


def inspect_dataset(split):
    file_path = DATA_DIR / f"{split}-00000-of-00001.parquet"

    print("\n" + "=" * 60)
    print(f"检查 {split.upper()} 数据集")
    print("=" * 60)

    if not file_path.exists():
        print(f"文件不存在：{file_path}")
        return

    # 读取 Parquet 文件
    df = pd.read_parquet(file_path)

    # 基本信息
    print(f"\n文件路径：{file_path}")
    print(f"数据数量：{len(df)}")
    print(f"字段名称：{list(df.columns)}")

    # 检查缺失值
    print("\n缺失值统计：")
    print(df.isnull().sum())

    # 检查数据样本
    print("\n前 5 条数据：")

    for index, row in df.head(5).iterrows():
        print(f"\n---------- 样本 {index + 1} ----------")

        for column in df.columns:
            print(f"\n{column}：")
            print(row[column])

        # 提取 GSM8K 标准答案
        if "answer" in df.columns:
            answer = str(row["answer"])

            if "####" in answer:
                final_answer = answer.rsplit("####", 1)[1].strip()
                print(f"\n提取出的最终答案：{final_answer}")
            else:
                print("\n未找到 #### 答案标记")


if __name__ == "__main__":
    inspect_dataset("train")
    inspect_dataset("test")

    print("\n数据集检查完成！")

