# 训练吞吐优化 backlog

> 建立于 2026-11-18。**P1 / P2 已在 `GRPO/` 路径实现**（2026-11-18），
> `src/`（DAPO）路径尚未同步。
> 背景：E1 300 步实测 **60.2 response token/s**、22.5 s/步、显存峰值 10.77 / 11.63 GB。
> 显存接近上限，但算力大量闲置（当时 `nvidia-smi` 抽查约 23% 利用率）。
> 数据来源：`outputs/E1_grpo_lora/logs/metrics.jsonl`。

| 条目 | 状态 | 代码 |
|---|---|---|
| P1 rollout 批量解码 | ✅ 已实现并实测：E2 500 步吞吐 **524 tok/s**（E1 60.2），8.7× | `GRPO/rollout.py::generate_batch`、`GRPO/trainer.py::train_step` |
| P2 零梯度步跳过反传 | ✅ 已实现并实测：E2 跳过 744/2000 个 micro-batch（37.2%） | `GRPO/trainer.py::_update_micro_batch` |
| P1/P2 移植到 `src/`（DAPO 路径） | ⬜ 未开始；注意 dynamic sampling 的重采逻辑要一起改 | — |

前置条件（P0，不在下列条目内）：先补**分段计时**（rollout / old+ref log-prob / 前向反向 / 优化器）
和 **GPU 利用率采样**（`nvidia-smi --query-gpu=utilization.gpu -l 1` 或 pynvml），
否则下面两条都无法验收。

---

## P1. Rollout 按题批量解码（K 道题 × G=4 一起 generate）

**状态**：✅ 已实现（2026-11-18）。`HFGenerator.generate_batch()` 左 padding 批量解码，
`split_batched_output()` 按题切回；训练改为先采 `prompts_per_batch × grad_accum_steps`
道题再切 micro-batch。测试：`tests/test_grpo_standalone.py` 9/9；
并用本地 Qwen2.5-0.5B 在 CPU 上实跑 `generate_batch`（2 题 × G=2），
左 padding 切分结果与 `tokenizer.encode` 逐 token 一致。GPU 提速仍待 smoke。

**现状**
- `GRPO/trainer.py:211-219` 的 `collect_batch` 对每道题单独调一次 `rollout_group`；
- `GRPO/rollout.py:331-333` 把同一 prompt repeat `num_generations` 份；
- 结果：解码 batch 恒为 4 条序列，每个 token 的 kernel 只有 4 行，启动开销和访存延迟占主导。

**方案**
- 一次取 K 道题（建议 4–8），把 K×G 条序列左 padding 拼成一个 batch 调 `model.generate`，
  再按题切回各自的 `PromptGroup`；
- `scripts/evaluate_gsm8k.py` 里已有左 padding 批量生成 + 结果切分的现成实现，抽成公共函数复用；
- 训练阶段保持不变：仍按 micro-batch 反传，`prompts_per_batch=1` / `grad_accum_steps` 语义不动，
  避免 12 GB 卡 OOM（B=2 已实测 OOM）。

**预期收益**
- rollout 段提速 2–4 倍（粗估），单步 22.5 s → 10–12 s 量级；
- 显存代价可忽略：16 条 × 800 token 的 KV cache 约 150 MB。

**注意**
- 训练采样本身是随机的，批量带来的数值差异不影响正确性；
  只有**评测**必须锁死 batch size（见踩坑记录 #19）。
- 左 padding 要确认 `position_ids` / `attention_mask` 处理正确，右 padding 会导致从 pad 位置续写。

**验收**
- 分段计时里 rollout 段耗时下降 ≥ 2×；
- 同样步数下 reward / 长度 / KL 的分布与改造前无系统性偏移。

---

## P2. 零梯度步跳过反传

**状态**：✅ 已实现（2026-11-18），默认关闭。开启方式：
`training.skip_zero_gradient_steps: true`。跳过时仍做一次 inference 前向保留熵，
不记录 `loss`/`kl` 等梯度指标，改记 `skipped_micro_batches` / `skipped_ratio`。

**现状**
- `zero_variance_ratio` 均值 **0.53**：这些步里 G=4 条回答奖励全相同，组内优势全为 0；
- 但 `_micro_step` 仍然跑 old/ref 两次前向、一次 policy 前向和 backward（`GRPO/trainer.py:221-287`）；
- E1 中约 1 小时耗在这类空转步上。

**方案**
- 算出优势后判断：若本步所有组都零方差，直接跳过后续前向/反传，只记录指标；
- 数学上严格等价：loss = −A·ratio + β·KL，A=0 时策略项梯度为 0，
  KL 项对策略 log-prob 的导数也相互抵消，跳过不改变参数轨迹；
- 注意与 DAPO Dynamic Sampling 的区别：这里是"跳过这次更新"，
  DAPO 是"重采直到组内非全对/全错"，语义不同，不要混用。

**预期收益**
- 省掉约一半步的更新段耗时，按步耗构成粗估整体 15–25%。

**验收**
- 同 seed 下，跳过版与不跳过版的检查点（同一步数）逐位一致，或差异仅在浮点误差内。

---

## 暂不做

- **vLLM 替换 rollout**：batch 只有 16–32 时连续批处理的优势发挥不出来，
  且需要处理 LoRA 热加载，改造量大。等 P1 做完看剩余瓶颈再评估。
- **`torch.compile`**：服务器是 Linux 可以试，但收益不确定，排在 P1/P2 之后。
