"""Qwen2.5 模型与 Tokenizer 的统一加载入口。

设计要点：
- 默认从项目内 models/Qwen2.5-0.5B-Instruct 加载，导入本模块时不会下载任何文件；
- CPU 环境固定使用 float32；
- CUDA 环境优先 bfloat16，显卡不支持时退回 float16；
- 加载失败时抛出带排查提示的异常。
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_DIR = PROJECT_ROOT / "models" / "Qwen2.5-0.5B-Instruct"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """解析目标设备：None/"auto" 自动选择，否则校验请求的设备是否可用。"""
    if isinstance(device, torch.device):
        resolved = device
    elif device is None or device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved = torch.device(device)

    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "请求了 CUDA 设备，但当前环境 torch.cuda.is_available() 为 False。"
            "请确认已安装对应 CUDA 版本的 PyTorch 且驱动正常。"
        )
    if resolved.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("请求了 MPS 设备，但当前 PyTorch 不支持或未启用 MPS。")
    return resolved


def select_dtype(device: torch.device) -> torch.dtype:
    """按设备选择推理/训练精度：CPU 用 float32，CUDA 优先 bfloat16。"""
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _resolve_source(model_path: str | Path | None) -> str:
    """返回实际加载路径或 HF repo id。

    - model_path 为 None 时只使用本地默认目录，不存在则报错（不自动下载）；
    - 显式传入的路径存在则按本地目录加载，否则视为 HF repo id（由调用者主动触发下载）。
    """
    if model_path is None:
        if DEFAULT_LOCAL_DIR.is_dir():
            return str(DEFAULT_LOCAL_DIR)
        raise FileNotFoundError(
            f"本地模型目录不存在：{DEFAULT_LOCAL_DIR}\n"
            "请先下载模型，例如：\n"
            f"  huggingface-cli download {DEFAULT_MODEL_NAME} "
            f"--local-dir {DEFAULT_LOCAL_DIR}"
        )

    path = Path(model_path).expanduser()
    if path.exists():
        return str(path)
    return str(model_path)


def load_tokenizer(
    model_path: str | Path | None = None,
    use_fast: bool = True,
    local_files_only: bool | None = None,
) -> PreTrainedTokenizerBase:
    """加载 Tokenizer。

    Args:
        model_path: 本地模型目录或 HF repo id；None 表示项目内默认目录。
        use_fast: 是否使用 fast tokenizer。
        local_files_only: 是否只读本地文件；None 时对本地目录自动设为 True。
    """
    source = _resolve_source(model_path)
    is_local = Path(source).exists()
    if local_files_only is None:
        local_files_only = is_local

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            source,
            use_fast=use_fast,
            local_files_only=local_files_only,
            trust_remote_code=False,
        )
    except Exception as exc:
        hint = (
            "请检查模型目录是否完整（config.json / tokenizer_config.json / tokenizer.json 等），"
            "或网络连接是否正常。"
        )
        raise RuntimeError(f"Tokenizer 加载失败：{source}\n{hint}\n原始错误：{exc}") from exc

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer 未定义 pad_token，已使用 eos_token 代替。")
    return tokenizer


def load_model(
    model_path: str | Path | None = None,
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    eval_mode: bool = True,
) -> PreTrainedModel:
    """加载因果语言模型。

    Args:
        model_path: 本地模型目录或 HF repo id；None 表示项目内默认目录。
        device: 目标设备；None 自动选择（CUDA 可用则 CUDA，否则 CPU）。
        dtype: 权重精度；None 时由 select_dtype() 决定。
        eval_mode: 加载后是否切到 eval 模式（推理场景应为 True）。
    """
    source = _resolve_source(model_path)
    resolved_device = resolve_device(device)
    resolved_dtype = dtype if dtype is not None else select_dtype(resolved_device)
    is_local = Path(source).exists()

    logger.info("加载模型：%s（device=%s, dtype=%s）", source, resolved_device, resolved_dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            source,
            torch_dtype=resolved_dtype,
            local_files_only=is_local,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
        )
    except torch.cuda.OutOfMemoryError as exc:
        raise RuntimeError(
            f"加载模型时 CUDA 显存不足：{source}\n"
            "可尝试减小分辨率相关 batch、使用更小精度或换用显存更大的 GPU。"
        ) from exc
    except Exception as exc:
        hint = (
            "请检查模型目录是否包含完整的 safetensors 权重与 config.json，"
            "以及 transformers 版本是否支持该模型结构。"
        )
        raise RuntimeError(f"模型加载失败：{source}\n{hint}\n原始错误：{exc}") from exc

    model.to(resolved_device)
    if eval_mode:
        model.eval()

    if not isinstance(model, PreTrainedModel):
        raise RuntimeError(f"加载结果不是 PreTrainedModel：{type(model)}")
    return model


def get_model_info(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> dict[str, Any]:
    """汇总模型规模、精度、设备与关键配置，便于日志记录与审计。"""
    first_param = next(model.parameters())
    config = model.config

    info: dict[str, Any] = {
        "model_class": type(model).__name__,
        "device": str(first_param.device),
        "dtype": str(first_param.dtype),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "num_trainable_parameters": sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ),
        "config": {
            "model_type": getattr(config, "model_type", None),
            "architectures": getattr(config, "architectures", None),
            "hidden_size": getattr(config, "hidden_size", None),
            "num_hidden_layers": getattr(config, "num_hidden_layers", None),
            "num_attention_heads": getattr(config, "num_attention_heads", None),
            "num_key_value_heads": getattr(config, "num_key_value_heads", None),
            "vocab_size": getattr(config, "vocab_size", None),
            "max_position_embeddings": getattr(config, "max_position_embeddings", None),
            "tie_word_embeddings": getattr(config, "tie_word_embeddings", None),
        },
    }

    if tokenizer is not None:
        info["tokenizer"] = {
            "class": type(tokenizer).__name__,
            "vocab_size": len(tokenizer),
            "eos_token": tokenizer.eos_token,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token": tokenizer.pad_token,
            "pad_token_id": tokenizer.pad_token_id,
            "chat_template_available": tokenizer.chat_template is not None,
        }
    return info
