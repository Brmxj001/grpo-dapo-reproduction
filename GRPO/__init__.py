"""独立版 Vanilla GRPO 实现。

模块：
- rollout.py：组采样、逐 token log prob、RolloutBuffer
- advantage.py：组内相对优势
- loss.py：PPO clip + k3 KL 的 GRPO 损失
- trainer.py：最小训练循环
- metrics.py：训练指标记录与绘图
"""

__all__ = []
