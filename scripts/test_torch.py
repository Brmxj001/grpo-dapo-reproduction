#!/usr/bin/env python3
"""PyTorch 环境验证：Tensor、矩阵乘法、自动求导、反向传播与优化器更新。

仅使用小型张量与单层线性模型，不涉及大模型训练。
"""

from __future__ import annotations

import sys

import torch
from torch import nn


def main() -> int:
    checks: list[tuple[str, bool, str]] = []

    def record(name: str, passed: bool, detail: str = "") -> None:
        checks.append((name, passed, detail))
        print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" | {detail}" if detail else ""))

    print("=" * 70)
    print(f"PyTorch 版本：{torch.__version__}  CUDA 可用：{torch.cuda.is_available()}")
    print(f"默认设备：{torch.get_default_device() if hasattr(torch, 'get_default_device') else 'cpu'}")
    print("=" * 70)

    # 1) Tensor 创建
    zeros = torch.zeros(2, 3)
    ones = torch.ones(2, 3, dtype=torch.float32)
    randn = torch.randn(2, 3)
    record(
        "Tensor 创建",
        zeros.shape == (2, 3) and ones.dtype == torch.float32 and randn.shape == (2, 3),
        f"zeros={tuple(zeros.shape)}, ones.dtype={ones.dtype}, randn={tuple(randn.shape)}",
    )
    record(
        "Tensor 加法",
        torch.allclose(zeros + ones, ones),
        "zeros + ones == ones",
    )

    # 2) 矩阵乘法（与 NumPy 参考实现对比，避免引入额外依赖）
    torch.manual_seed(42)
    a = torch.randn(4, 5)
    b = torch.randn(5, 3)
    c = a @ b
    c_ref = torch.from_numpy(a.numpy() @ b.numpy())
    record(
        "矩阵乘法",
        c.shape == (4, 3) and torch.allclose(c, c_ref, atol=1e-6),
        f"shape={tuple(c.shape)}, max_abs_diff={(c - c_ref).abs().max().item():.2e}",
    )

    # 3) 自动求导与反向传播：y = x^3 + 2x，在 x=2 处 dy/dx = 3x^2 + 2 = 14
    x = torch.tensor(2.0, requires_grad=True)
    y = x**3 + 2 * x
    y.backward()
    record("自动求导 requires_grad", x.grad is not None)
    record(
        "反向传播梯度正确",
        x.grad is not None and abs(x.grad.item() - 14.0) < 1e-6,
        f"dy/dx={x.grad.item() if x.grad is not None else None:.6f}（期望 14）",
    )

    # 4) 优化器更新参数：拟合 y = 2.5x - 1
    torch.manual_seed(0)
    true_w, true_b = 2.5, -1.0
    inputs = torch.randn(256, 1)
    targets = true_w * inputs + true_b

    model = nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    weight_before = model.weight.detach().clone()
    initial_loss = criterion(model(inputs), targets).item()
    for _ in range(500):
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()
    final_loss = criterion(model(inputs), targets).item()
    weight_after = model.weight.detach().clone()

    record(
        "优化器更新参数",
        not torch.allclose(weight_before, weight_after),
        f"w: {weight_before.item():.4f} -> {weight_after.item():.4f}",
    )
    record(
        "线性模型损失下降",
        final_loss < initial_loss * 1e-3,
        f"loss {initial_loss:.6f} -> {final_loss:.8f}",
    )
    learned_w = model.weight.item()
    learned_b = model.bias.item()
    record(
        "学到的参数接近真实值",
        abs(learned_w - true_w) < 0.1 and abs(learned_b - true_b) < 0.1,
        f"w={learned_w:.4f}（期望 {true_w}）, b={learned_b:.4f}（期望 {true_b}）",
    )

    # 5) no_grad 下不构建计算图
    with torch.no_grad():
        out = model(inputs)
    record("no_grad 下 requires_grad 为 False", not out.requires_grad)

    failures = [name for name, passed, _ in checks if not passed]
    print("\n" + "=" * 70)
    if failures:
        print(f"PyTorch 测试失败：{len(failures)}/{len(checks)} 项未通过")
        for name in failures:
            print(f"  - {name}")
        return 1
    print(f"PyTorch 测试通过：{len(checks)}/{len(checks)} 项全部通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
