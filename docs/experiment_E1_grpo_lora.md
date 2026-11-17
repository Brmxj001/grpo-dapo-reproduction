# 实验 E1：GRPO + LoRA，300 步

| 项 | 内容 |
|---|---|
| 日期 | 2026-11-17 |
| 平台 | AutoDL 单卡 RTX 3080 Ti 12 GB（可用 11.63 GiB） |
| 基座 | Qwen2.5-0.5B-Instruct |
| 训练方式 | LoRA（r=16, α=32, dropout=0.05, 7 个投影层，8,798,208 可训练参数 / 1.75%） |
| 数据 | GSM8K train 7,473 题，顺序读取前 300 题（不打乱） |
| 算法 | GRPO，组大小 G=4，ε=0.2，β=0.04（k3 KL），奖励 = 答案正确 |
| 配置 | `configs/grpo_lora.yaml`（`prompts_per_batch=1`, `grad_accum_steps=1`） |
| 训练量 | 300 个优化步 = 300 题 × 4 条回答 = 1,200 条 rollout |

---

## 1. 目的

在 12 GB 单卡上跑通 **真实规模的 GRPO**（不是 toy 数据），回答两个问题：

1. 现有代码链路（rollout → 奖励 → 优势 → 裁剪目标 → LoRA 反向 → 存档/续训）是否能长时间稳定运行；
2. 在这样小的算力预算下，GRPO 究竟能带来多少实际收益 —— 还是说开销主要花在"无学习信号"的步骤上。

---

## 2. 结果总览

### 2.1 训练曲线（300 步，50 步一桶）

| 步数 | reward_mean | zero_var 比例 | 熵 entropy | 平均长度 | 截断率 | 有奖励的步 |
|---|---|---|---|---|---|---|
| 1–50 | 0.210 | 0.48 | 0.986 | 346 | 0.18 | 27/50 |
| 51–100 | 0.215 | 0.60 | 0.540 | 308 | 0.07 | 22/50 |
| 101–150 | 0.315 | 0.44 | 0.487 | 335 | 0.16 | 31/50 |
| 151–200 | 0.285 | 0.48 | 0.394 | 321 | 0.10 | 29/50 |
| 201–250 | 0.170 | 0.70 | 0.717 | 374 | 0.27 | 17/50 |
| 251–300 | 0.300 | 0.48 | 0.401 | 344 | 0.12 | 28/50 |

整体：reward 均值 **0.249**，熵 0.986 → 0.401，KL 均值 0.085，`clip_fraction` 均值 0.0095，
`ratio_mean` 稳定在 1.00 附近，显存峰值 **10.77 GB**，总耗时 **1.87 小时**（22.5 s/步）。

### 2.2 评测（GSM8K test，1,319 题，贪心解码，batch=16）

| 模型 | 准确率 | 平均回答长度 | 截断率 |
|---|---|---|---|
| 基座（baseline） | **0.4306** (568/1319) | 319.9 | 6.75% |
| E1 step_300 | **0.4049** (534/1319) | 330.9 | 7.88% |
| Δ | **−0.0258** | **+11.0** | **+1.13 pp** |

配对检验（同一批 1319 题，`scripts/compare_evals.py`）：

```
都对 376 | 仅基线对 192 | 仅新模型对 158 | 都错 593
McNemar 精确检验  p = 0.0776
```

结论：**没有提升，且方向上略差**；但 p=0.078，严格说"当前样本量下无法断言显著变差"。

评测脚本与基线完全一致（`scripts/evaluate_gsm8k.py --batch-size 16`，temperature=0，max_new_tokens=512），
避免"批量 vs 逐题""采样 vs 贪心"带来的假提升。

---

## 3. 观察与解读

### 3.1 训练没有把准确率推上去

0.4306 → 0.4049（−2.58 个百分点）。McNemar p=0.078，
所以准确的说法是"**没有证据表明变好**"，严格下界也谈不上"显著变差"。
但结合 300 步里 reward 始终在 0.21–0.32 横盘，可以确定这不是随机抖动掩盖了进步。

### 3.2 为什么没效果：三个结构性原因

**① 数据量太少了。** 300 个优化步 × 每步 1 道题 = 300 道题，
仅占 GSM8K 训练集（7,473）的 **4%**，总共 1,200 条 rollout。
作为对比，DAPO 原论文每步就要几百个 prompt、每个 prompt 采样十几条。
在这里，模型被更新了 300 次，每次只看 4 条回答 —— 梯度噪声远大于信号。

**② 一半的步是空转。** `zero_variance_ratio` 均值 **0.53**：
这些步里 4 条回答奖励完全相同（通常全对或全错），组内优势全为 0，
这一整步的前向 + 反向 + 优化器更新**什么都没学到**，但照样花了 22 秒和同等显存。
1.87 小时里大约有 **1 小时是纯开销**。

**③ 熵塌缩让 ② 更严重。** `policy_entropy` 从 0.986 掉到 0.401。
策略越确定，4 条采样越容易给出相同答案 → 零方差比例上升：
第 201–250 桶的 `zero_variance_ratio` 正好是全实验最高的 0.70。
这是一个正反馈的恶性循环，也正是 DAPO 的 Clip-Higher 要打破的。

### 3.3 长度在涨，正确率没跟上

平均回答长度 +11 token（319.9 → 330.9），截断率 6.75% → 7.88%。
这是 GRPO 的经典倾向：**把答案写长**比"想对"更容易被奖励塑形捕捉到。
DAPO 的 overlong reward shaping 针对的正是这里。

### 3.4 一个正面的结论：工程链路是稳的

- 300 步全程无 OOM、无 NaN、无 `ratio` 爆炸（`ratio_mean` 稳定在 1.00 附近，`clip_fraction` 均值 0.0095）
- KL 从 0 平缓升到 0.09，符合"逐步偏离参考策略"的预期
- 断点续训后指标逐位可复现
- 显存峰值 10.77 GB，距 11.63 GiB 上限有约 0.9 GB 余量

也就是说：**代码没问题，问题在算法配方和算力预算。**

---

## 4. 与 DAPO 的对应关系

这次用的是**原版 GRPO**，因此 DAPO 论文指出的四类病灶都能在数据里看到：

| DAPO 组件 | 本实验的对应现象 | 数据证据 |
|---|---|---|
| Dynamic Sampling（丢弃零方差组再补采） | 53% 的步组内奖励全相同，优势全 0，该步不产生任何梯度 | `zero_variance_ratio` 均值 0.53 |
| Clip-Higher（解耦上下裁剪界） | 熵从 0.99 跌到 0.40，策略趋于确定 | `policy_entropy` 后 150 步稳定在 0.4 附近 |
| Token-level loss | 本实验按序列内 token 平均，长回答的梯度被稀释 | 未单独度量 |
| Overlong filtering / soft punishment | 截断率平均 14.9%，最大桶 27% | `truncation_rate` |

---

## 5. 复现

```bash
# 云端
export PATH=/root/miniconda3/bin:$PATH
python scripts/train_grpo_standalone.py --config configs/grpo_lora.yaml --max-steps 300
python scripts/evaluate_gsm8k.py --batch-size 16 \
    --adapter outputs/E1_grpo_lora/checkpoints/step_300 \
    --output outputs/eval_lora_step300
```

断点续训：

```bash
python scripts/train_grpo_standalone.py --config configs/grpo_lora.yaml \
    --max-steps 300 --resume outputs/E1_grpo_lora/checkpoints/step_150
```

---

## 6. 产物

| 文件 | 位置 | 说明 |
|---|---|---|
| LoRA 权重 | `outputs/E1_grpo_lora/checkpoints/step_300/adapter_model.safetensors` | 33.6 MB |
| 完整检查点 | 同目录 `trainer_state.pt` | 含优化器状态，67 MB，仅续训用 |
| 指标原始数据 | `outputs/E1_grpo_lora/logs/metrics.jsonl` | 每步一行 JSON |
| 指标汇总 | `outputs/E1_grpo_lora/logs/summary_E1.json` | `scripts/summarize_metrics.py` 生成 |
| 曲线图 | `outputs/E1_grpo_lora/plots/*.png` | 12 张 |
| 评测明细 | `outputs/eval_lora_step300/` | `summary.json` + `generations.jsonl` |
| 配对比较 | `outputs/eval_lora_step300/compare_vs_base.json` | `scripts/compare_evals.py` 生成（McNemar） |
| Release | GitHub `v0.1-E1-grpo-step300` | 含 `step_300_lora.zip` |
