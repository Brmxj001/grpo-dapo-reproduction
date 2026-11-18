# E2 实验记录：GRPO + LoRA，4 题 × G=6，500 步

> 运行日期：2026-11-18
> 配置：[`configs/grpo_lora_g6.yaml`](../configs/grpo_lora_g6.yaml)
> 指标：[`outputs/E2_grpo_lora_g6/logs/metrics.jsonl`](../outputs/E2_grpo_lora_g6/logs/metrics.jsonl)
> 曲线：[`outputs/E2_grpo_lora_g6/plots/`](../outputs/E2_grpo_lora_g6/plots/)

## 1. 目的

在 E1（1 题 × G=4 × 300 步，300 道题）基础上，回答两个问题：

1. **增加题目数能否带来可测的提升**：500 步 × 4 题 = 2,000 道题（GSM8K 训练集 27%），
   是 E1 的 6.7 倍；
2. **P1 / P2 两项吞吐改造在真实训练里值多少**：批量 rollout、零梯度步跳过反传。

## 2. 实验设置

| 项 | 值 |
|---|---|
| 模型 | Qwen2.5-0.5B-Instruct + LoRA（r=16, α=32, 7 个投影层，可训练 8.8 M / 1.75%） |
| 硬件 | AutoDL RTX 4090 D 24564 MiB，cgroup 16 核 / 80 GB，驱动 580.105.08 |
| 采样 | G=6，temperature=1.0，max_new_tokens=512 |
| 更新 | `prompts_per_batch=1`，`grad_accum_steps=4` → 每优化步 4 道题；显存只按 1 题 × 6 条计 |
| 优化 | AdamW，lr=1e-4，warmup 3%（15 步），clip 0.2，KL β=0.04 |
| 数据 | GSM8K train 前 2,000 道（`shuffle: false`），与 E1 同一顺序 |
| 环境变量 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| 吞吐改造 | P1 批量 rollout（4 题 × 6 = 24 条一次解码）；P2 零梯度 micro-batch 跳过反传 |

显存说明：N=12（2 题 × 6）在 23.5 GiB 上实测 OOM，原因见踩坑记录 #34
（Qwen2 的 lm_head 输出 fp32 logits，三份约 15 GB）。

## 3. 结果

### 3.1 吞吐与资源

| 指标 | E1（1×4×300） | E2（4×6×500） | 变化 |
|---|---|---|---|
| 训练时长 | 112.3 min | **127.5 min** | +13.5% |
| 题目数 / rollout 数 | 300 / 1,200 | **2,000 / 12,000** | 6.7× / 10× |
| 平均步时 | 22.5 s | **15.3 s**（中位 14.9，最大 24.2） | −32% |
| response token 总量 | 0.406 M | **4.01 M** | 9.9× |
| 吞吐 | 60.2 tok/s | **524 tok/s** | **8.7×** |
| 显存峰值（分配） | 10.77 GB | 16.27 GB | +5.5 GB |
| GPU 利用率（1 min 采样均值） | ~23%（抽查） | **40.5%**（峰值 89%） | — |

P2 效果：2,000 个 micro-batch 中跳过 **744 个（37.2%）**；整步全跳过的只有 7 步。

### 3.2 训练指标

| 指标 | 前 50 步 | 后 50 步 | 全程 |
|---|---|---|---|
| reward_mean | 0.323 | 0.341 | 0.404 |
| zero_variance_ratio | 0.410 | 0.420 | 0.372 |
| policy_entropy | 0.614 | 0.431 | 0.363 |
| avg_response_length | 339.8 | 377.2 | 334.3 |
| truncation_rate | 14.4% | 23.3% | 11.7% |
| kl | 0.062 | 0.183 | 0.110 |
| grad_norm | 0.347 | 0.327 | 0.306 |

按 100 步分段看 reward：

```
1-100: 0.365   101-200: 0.448   201-300: 0.378   301-400: 0.442   401-500: 0.387
```

## 4. 解读（截至当前数据）

- **吞吐改造达到预期**：题目数 ×6.7 的情况下总时长只多 13.5%，单位 token 吞吐提升 8.7×。
  瓶颈已从"空转反传"转为"小 batch 解码"，与踩坑 #33 的判断一致。
- **没有看到稳定的学习收益**：reward 在 0.36–0.45 之间震荡，500 步内没有趋势性上升；
  同时回答长度、截断率、KL 在后期同步上升（长度 339→377，截断 14%→23%，KL 0.06→0.18），
  与 E1 的"变长但没变对"是同一现象。
- **零方差步仍是主要约束**：37% 的 micro-batch 没有梯度；G=6 把比例从 E1 的 0.53 压到 0.37，
  但基座模型大量题目全错的现实没有改变。真正的解法是 DAPO dynamic sampling（重采非退化组），
  尚未移植到 `GRPO/`。
- **熵没有坍缩**：从 0.98 降到 0.27 后又回到 0.43 左右，全程最低 0.11 以上，未出现模式坍缩。

## 5. 尚未完成

- [ ] **评测**：本次只记录训练指标，没有跑 GSM8K test（同口径 batch=16）。
      权重在本地 `outputs/E2_grpo_lora_g6/checkpoints/step_500/`，
      评测命令见 [`server_setup.md`](server_setup.md) §4.4。
- [ ] **DAPO 对照**：`configs/dapo.yaml` 的四个开关尚未在同等规模下跑过。
- [ ] **P1/P2 移植到 `src/`**（DAPO 路径），并处理 dynamic sampling 的重采逻辑。

## 6. 复现

```bash
export PATH=/root/miniconda3/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /root/autodl-tmp/grpo-dapo-reproduction
setsid nohup python scripts/train_grpo_standalone.py \
    --config configs/grpo_lora_g6.yaml > outputs/E2_train.log 2>&1 < /dev/null &
```

评测（需 GPU）：

```bash
python scripts/evaluate_gsm8k.py --batch-size 16 \
    --adapter outputs/E2_grpo_lora_g6/checkpoints/step_500 \
    --output outputs/eval_E2_step500
```
