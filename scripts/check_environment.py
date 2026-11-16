#!/usr/bin/env python3
"""环境自检脚本：Python/PyTorch/Transformers 版本、设备、模型目录、Tokenizer/模型加载与 CPU 推理。

退出码 0 表示所有必需检查通过；任一必需项失败则返回 1。
"""

from __future__ import annotations

import argparse
import importlib.metadata as importlib_metadata
import os
import platform
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_loader import DEFAULT_LOCAL_DIR, get_model_info, load_model, load_tokenizer

REQUIRED_PACKAGES = [
    "torch",
    "transformers",
    "datasets",
    "peft",
    "accelerate",
    "numpy",
    "pandas",
    "matplotlib",
    "sentencepiece",
    "safetensors",
    "tqdm",
    "scipy",
    "tensorboard",
    "pyyaml",
]


class Report:
    """收集检查结果并统一格式化输出。"""

    def __init__(self) -> None:
        self.results: list[tuple[str, str, str]] = []

    def add(self, name: str, status: str, detail: str = "") -> None:
        assert status in {"PASS", "FAIL", "WARN", "INFO"}
        self.results.append((name, status, detail))
        print(f"[{status:4s}] {name}" + (f" | {detail}" if detail else ""))

    @property
    def failed(self) -> int:
        return sum(1 for _, status, _ in self.results if status == "FAIL")

    def summary(self) -> str:
        counts = {key: 0 for key in ("PASS", "FAIL", "WARN", "INFO")}
        for _, status, _ in self.results:
            counts[status] += 1
        return (
            f"PASS={counts['PASS']} FAIL={counts['FAIL']} "
            f"WARN={counts['WARN']} INFO={counts['INFO']}"
        )


def package_version(distribution: str) -> str:
    try:
        return importlib_metadata.version(distribution)
    except importlib_metadata.PackageNotFoundError:
        return "未安装"


def check_versions(report: Report) -> None:
    import torch
    import transformers

    report.add("Python 版本", "PASS" if sys.version_info >= (3, 10) else "WARN",
               f"{platform.python_version()}（{sys.executable}）")
    report.add("操作系统", "INFO", f"{platform.system()} {platform.release()} / {platform.machine()}")
    report.add("Conda 环境", "INFO", os.environ.get("CONDA_DEFAULT_ENV", "未激活 conda 环境"))
    report.add("PyTorch 版本", "PASS", torch.__version__)
    report.add("Transformers 版本", "PASS", transformers.__version__)

    missing = [name for name in REQUIRED_PACKAGES if package_version(name) == "未安装"]
    report.add(
        "必需依赖完整性",
        "PASS" if not missing else "FAIL",
        "全部已安装" if not missing else f"缺少：{', '.join(missing)}",
    )
    versions = ", ".join(f"{name}=={package_version(name)}" for name in REQUIRED_PACKAGES)
    report.add("依赖版本", "INFO", versions)


def check_device(report: Report) -> None:
    import torch

    cuda_available = torch.cuda.is_available()
    report.add("CUDA 可用性", "INFO", str(cuda_available))
    if cuda_available:
        device_count = torch.cuda.device_count()
        report.add("CUDA 版本", "INFO", str(torch.version.cuda))
        for index in range(device_count):
            props = torch.cuda.get_device_properties(index)
            report.add(
                f"GPU {index}",
                "INFO",
                f"{props.name}, 显存 {props.total_memory / 1024**3:.1f} GiB, "
                f"compute capability {props.major}.{props.minor}",
            )
    else:
        report.add("GPU 信息", "INFO", "未检测到可用的 NVIDIA GPU（将使用 CPU）")


def check_model_files(report: Report, model_path: Path) -> None:
    if model_path.is_dir():
        files = sorted(p.name for p in model_path.iterdir() if p.is_file())
        total_bytes = sum(p.stat().st_size for p in model_path.rglob("*") if p.is_file())
        report.add("模型目录", "PASS", f"{model_path}（{len(files)} 个文件，{total_bytes / 1024**3:.2f} GiB）")
        required = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}
        missing = sorted(required - set(files))
        report.add(
            "模型关键文件",
            "PASS" if not missing else "FAIL",
            "config.json / model.safetensors / tokenizer 文件齐全" if not missing
            else f"缺少：{', '.join(missing)}",
        )
    else:
        report.add("模型目录", "FAIL", f"不存在：{model_path}")


def check_tokenizer_and_model(report: Report, model_path: Path, max_new_tokens: int) -> None:
    import torch

    if not model_path.is_dir():
        report.add("Tokenizer 加载", "FAIL", "模型目录不存在，跳过")
        report.add("模型加载", "FAIL", "模型目录不存在，跳过")
        report.add("CPU 推理", "FAIL", "模型目录不存在，跳过")
        return

    try:
        tokenizer = load_tokenizer(model_path)
        report.add("Tokenizer 加载", "PASS", f"{type(tokenizer).__name__}, vocab={len(tokenizer)}")
    except Exception as exc:
        report.add("Tokenizer 加载", "FAIL", str(exc))
        report.add("模型加载", "FAIL", "Tokenizer 加载失败，跳过")
        report.add("CPU 推理", "FAIL", "Tokenizer 加载失败，跳过")
        return

    try:
        start = time.perf_counter()
        model = load_model(model_path)
        load_seconds = time.perf_counter() - start
        info = get_model_info(model, tokenizer)
        report.add(
            "模型加载",
            "PASS",
            f"{info['model_class']}, {info['num_parameters']:,} 参数, "
            f"{info['dtype']}, {info['device']}, 耗时 {load_seconds:.1f}s",
        )
    except Exception as exc:
        report.add("模型加载", "FAIL", str(exc))
        report.add("CPU 推理", "FAIL", "模型加载失败，跳过")
        return

    try:
        messages = [{"role": "user", "content": "用一句话说明什么是机器学习。"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(next(model.parameters()).device) for key, value in inputs.items()}
        start = time.perf_counter()
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                # 显式置空采样参数，避免与 generation_config 中的默认值冲突而告警
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
            )
        elapsed = time.perf_counter() - start
        new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        report.add(
            "CPU 推理",
            "PASS" if answer else "FAIL",
            f"{new_ids.numel()} tokens / {elapsed:.2f}s | 输出：{answer[:60]!r}",
        )
    except Exception as exc:
        report.add("CPU 推理", "FAIL", f"{type(exc).__name__}: {exc}")


def main() -> int:
    parser = argparse.ArgumentParser(description="GRPO+DAPO 项目环境自检")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_LOCAL_DIR, help="本地模型目录")
    parser.add_argument("--max-new-tokens", type=int, default=32, help="推理自检的最大新 Token 数")
    args = parser.parse_args()

    report = Report()
    print("=" * 78)
    print("GRPO + DAPO 复现项目 - 环境检查报告")
    print("=" * 78)
    print("\n[1] 版本与依赖")
    check_versions(report)
    print("\n[2] 设备")
    check_device(report)
    print("\n[3] 模型文件")
    check_model_files(report, args.model_path)
    print("\n[4] Tokenizer / 模型 / 推理")
    check_tokenizer_and_model(report, args.model_path, args.max_new_tokens)

    print("\n" + "=" * 78)
    print(f"汇总：{report.summary()}")
    if report.failed:
        print("结论：环境检查未通过，请根据上方 FAIL 项排查。")
        return 1
    print("结论：环境检查全部必需项通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
