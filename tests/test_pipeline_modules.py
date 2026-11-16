"""数据、Prompt 与奖励模块测试，不依赖 pytest：

    python tests/test_pipeline_modules.py
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import (  # noqa: E402
    MathProblem,
    MathReasoningDataset,
    create_dataloader,
)
from src.data.preprocess import (  # noqa: E402
    clean_solution,
    extract_ground_truth,
    load_raw_split,
    preprocess_frame,
)
from src.data.prompt import MATH_INSTRUCTION, build_messages, build_prompt  # noqa: E402
from src.rewards.answer_parser import parse_answer  # noqa: E402
from src.rewards.math_verifier import normalize_answer, verify_answer  # noqa: E402
from src.rewards.reward import (  # noqa: E402
    OverlongPenaltyConfig,
    RewardConfig,
    compute_reward,
    compute_rewards,
    overlong_penalty,
)

PROCESSED_DIR = PROJECT_ROOT / "data/processed/gsm8k"
RAW_DIR = PROJECT_ROOT / "data/raw/gsm8k/main"

_TOKENIZER = None


def _tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from src.model_loader import load_tokenizer

        _TOKENIZER = load_tokenizer()
    return _TOKENIZER


def _skip_if_no_raw() -> bool:
    if not (RAW_DIR / "train-00000-of-00001.parquet").exists():
        print("  (跳过) 原始 GSM8K 数据不存在")
        return True
    return False


def _load_test_dataset() -> MathReasoningDataset | None:
    processed = PROCESSED_DIR / "test.parquet"
    if processed.exists():
        return MathReasoningDataset.from_processed(processed)
    raw = RAW_DIR / "test-00000-of-00001.parquet"
    if raw.exists():
        return MathReasoningDataset.from_raw(raw)
    print("  (跳过) 测试集不存在")
    return None


def test_preprocess_helpers():
    assert extract_ground_truth("steps...\n#### 1,080") == "1,080"

    solution = clean_solution("She sold 48/2 = <<48/2=24>>24 clips.\n#### 72")
    assert "<<48/2=24>>" not in solution
    assert "24 clips" in solution
    assert "####" not in solution

    try:
        extract_ground_truth("no marker")
    except ValueError:
        pass
    else:
        raise AssertionError("缺少 #### 时应抛 ValueError")


def test_preprocess_real_data():
    if _skip_if_no_raw():
        return
    processed = preprocess_frame(load_raw_split("train").head(20))
    assert list(processed.columns) == ["question", "solution", "ground_truth"]
    assert len(processed) == 20
    assert not processed.isnull().any().any()
    assert all(processed["ground_truth"])
    assert all("<<" not in solution for solution in processed["solution"])


def test_dataset_loading():
    dataset = _load_test_dataset()
    if dataset is None:
        return
    assert len(dataset) == 1319
    problem = dataset[0]
    assert isinstance(problem, MathProblem)
    assert problem.question and problem.ground_truth

    train_path = PROCESSED_DIR / "train.parquet"
    raw_train = RAW_DIR / "train-00000-of-00001.parquet"
    train_dataset = (
        MathReasoningDataset.from_processed(train_path)
        if train_path.exists()
        else MathReasoningDataset.from_raw(raw_train)
    )
    assert len(train_dataset) == 7473
    overlap = set(train_dataset.questions) & set(dataset.questions)
    assert not overlap, f"训练/测试集题目重叠 {len(overlap)} 条"

    batch = next(iter(create_dataloader(dataset, batch_size=2, shuffle=False)))
    assert len(batch) == 2
    assert all(isinstance(item, MathProblem) for item in batch)


def test_prompt_building():
    messages = build_messages("2+3=?")
    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert MATH_INSTRUCTION in messages[0]["content"]
    assert "2+3=?" in messages[0]["content"]

    tokenizer = _tokenizer()
    prompt = build_prompt("2+3=?", tokenizer)
    assert "<|im_start|>user" in prompt
    assert "\\boxed" in prompt
    assert "2+3=?" in prompt
    assert prompt.endswith("<|im_start|>assistant\n")

    with_system = build_prompt("2+3=?", tokenizer, system_prompt="你是数学助手。")
    assert "你是数学助手。" in with_system


def test_answer_parser():
    assert parse_answer("blah blah \\boxed{72}") == "72"
    assert parse_answer("first \\boxed{1}, so \\boxed{2}") == "2"
    assert parse_answer("\\boxed{\\frac{3}{4}}") == "\\frac{3}{4}"
    assert parse_answer("steps...\n#### 1,080") == "1,080"
    assert parse_answer("Answer: 55.") == "55"
    assert parse_answer("Final Answer: 8") == "8"
    assert parse_answer("The final answer is 42.") == "42"
    assert parse_answer("\\boxed{5}.") == "5"
    assert parse_answer("so \\boxed{72") == "72"  # 生成截断、花括号未闭合
    assert parse_answer("no answer here") is None
    assert parse_answer("") is None
    assert parse_answer(None) is None


def test_math_verifier():
    assert verify_answer("72", "72")
    assert verify_answer("72.0", "72")
    assert verify_answer("-5", "-5")
    assert verify_answer("1,080", "1080")
    assert verify_answer("3/4", "0.75")
    assert verify_answer("\\frac{3}{4}", "0.75")
    assert verify_answer("$5", "5")
    assert verify_answer(" 55. ", "55")
    assert verify_answer("0.3333", "1/3")
    assert not verify_answer("71", "72")
    assert not verify_answer(None, "5")
    assert not verify_answer("", "5")
    assert not verify_answer("abc", "5")
    assert normalize_answer("$1,080.") == "1080"


def test_reward():
    good = "Natalia sold 48+24=72 clips.\n\\boxed{72}"
    result = compute_reward(good, "72")
    assert result.correct and result.format_ok
    assert result.reward == 1.0

    no_format = compute_reward("Final: #### 72", "72")
    assert no_format.correct and not no_format.format_ok
    assert no_format.reward == 1.0

    wrong = compute_reward("... \\boxed{71}", "72")
    assert not wrong.correct and wrong.format_ok
    assert wrong.reward == 0.0

    garbage = compute_reward("I don't know.", "72")
    assert garbage.answer is None
    assert garbage.reward == 0.0

    with_format_bonus = compute_reward(
        good, "72", config=RewardConfig(format_weight=0.1)
    )
    assert abs(with_format_bonus.reward - 1.1) < 1e-9

    paper_style = compute_reward(
        "... \\boxed{71}", "72", config=RewardConfig(incorrect_reward=-1.0)
    )
    assert paper_style.reward == -1.0

    results = compute_rewards([good, "... \\boxed{71}"], ["72", "72"])
    assert results[0].reward == 1.0 and results[1].reward == 0.0

    try:
        compute_rewards(["a"], ["1", "2"])
    except ValueError:
        pass
    else:
        raise AssertionError("长度不一致时应抛 ValueError")


def test_overlong_shaping():
    config = OverlongPenaltyConfig(max_response_length=100, soft_cache_length=20)
    assert overlong_penalty(80, config) == 0.0
    assert math.isclose(overlong_penalty(90, config), -0.5)
    assert math.isclose(overlong_penalty(100, config), -1.0)
    assert overlong_penalty(150, config) == -1.0

    result = compute_reward(
        "\\boxed{72}", "72", overlong_config=config, response_length=90
    )
    assert result.correct
    assert math.isclose(result.length_penalty, -0.5)
    assert math.isclose(result.reward, 0.5)

    scaled = OverlongPenaltyConfig(100, 20, penalty_scale=0.5)
    assert math.isclose(overlong_penalty(90, scaled), -0.25)

    try:
        OverlongPenaltyConfig(max_response_length=100, soft_cache_length=100)
    except ValueError:
        pass
    else:
        raise AssertionError("L_cache >= L_max 时应抛 ValueError")


def test_reward_on_real_samples():
    dataset = _load_test_dataset()
    if dataset is None:
        return
    num_samples = min(120, len(dataset))
    for index in range(num_samples):
        problem = dataset[index]
        completion = f"{problem.solution}\nThe final answer is \\boxed{{{problem.ground_truth}}}."
        result = compute_reward(completion, problem.ground_truth)
        assert result.correct, f"第 {index} 条正确答案未被识别：{problem.ground_truth}"
        assert result.reward == 1.0

        wrong_answer = "987654321" if problem.ground_truth != "987654321" else "123456789"
        wrong = compute_reward(f"\\boxed{{{wrong_answer}}}", problem.ground_truth)
        assert not wrong.correct and wrong.reward == 0.0

    print(f"  (已用 {num_samples} 道真实样本人为构造正确/错误回答验证奖励)")


def test_end_to_end():
    if _skip_if_no_raw():
        return
    row = preprocess_frame(load_raw_split("train").head(1)).iloc[0]

    prompt = build_prompt(row["question"], _tokenizer())
    assert row["question"][:30] in prompt

    completion = f"{row['solution']}\n\\boxed{{{row['ground_truth']}}}"
    result = compute_reward(completion, row["ground_truth"])
    assert result.correct and result.format_ok
    assert result.reward == 1.0


TESTS = [
    ("preprocess: 答案提取与解法清洗", test_preprocess_helpers),
    ("preprocess: 真实 GSM8K 预处理", test_preprocess_real_data),
    ("dataset: 训练/测试读取与隔离", test_dataset_loading),
    ("prompt: Qwen ChatML 构建", test_prompt_building),
    ("answer_parser: 多格式答案解析", test_answer_parser),
    ("math_verifier: 数值等价验证", test_math_verifier),
    ("reward: 1/0 奖励与可配置项", test_reward),
    ("reward: DAPO 超长惩罚", test_overlong_shaping),
    ("reward: 120 道真实样本验证", test_reward_on_real_samples),
    ("集成: 真实样本端到端", test_end_to_end),
]


def main() -> int:
    failures = []
    for name, test in TESTS:
        try:
            test()
        except Exception:
            failures.append(name)
            print(f"[FAIL] {name}")
            traceback.print_exc()
        else:
            print(f"[PASS] {name}")

    total = len(TESTS)
    print(f"\n结果：{total - len(failures)}/{total} 通过")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
