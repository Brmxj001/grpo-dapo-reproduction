"""统一随机种子，保证数据采样、Rollout 与训练可复现。"""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """设置 Python / NumPy / PyTorch 的随机种子。

    deterministic=True 时开启 cuDNN 确定性模式（CPU 环境无影响）。
    注意：不强制开启 torch.use_deterministic_algorithms，避免部分算子直接报错。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_torch_generator(seed: int | None = None) -> torch.Generator:
    generator = torch.Generator()
    if seed is not None:
        generator.manual_seed(seed)
    return generator
