"""YAML 配置加载与继承合并。

配置文件可包含 ``defaults`` 字段继承其他配置（相对路径，相对于当前文件所在目录）。
继承按列出顺序依次深合并，当前文件的键优先级最高。

示例（configs/experiments/E2_clip_higher.yaml）::

    defaults: [../grpo.yaml]
    algorithm:
      use_clip_higher: true
      clip_ratio_high: 0.28
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是字典：{path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """递归合并两个字典，override 优先；列表整体替换。"""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_with_defaults(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in stack + (path,))
        raise ValueError(f"配置存在循环继承：{chain}")

    data = load_yaml(path)
    defaults = data.pop("defaults", [])
    if isinstance(defaults, str):
        defaults = [defaults]
    if not isinstance(defaults, list):
        raise ValueError(f"defaults 必须是列表：{path}")

    merged: dict[str, Any] = {}
    for entry in defaults:
        dependency = (path.parent / str(entry)).resolve()
        merged = deep_merge(merged, _load_with_defaults(dependency, stack + (path,)))
    return deep_merge(merged, data)


def load_config(path: str | Path) -> dict[str, Any]:
    """加载配置并展开 defaults 继承；相对路径相对于项目根目录。"""
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    return _load_with_defaults(config_path, ())


def resolve_path(path: str | Path) -> Path:
    """把配置中的相对路径解析到项目根目录。"""
    resolved = Path(path).expanduser()
    return resolved if resolved.is_absolute() else PROJECT_ROOT / resolved
