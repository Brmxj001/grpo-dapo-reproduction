# 云服务器环境还原指南（AutoDL）

> 目的：AutoDL 实例释放后机器状态全部消失。下次租新卡时，照着本文从
> **本地仓库 + 3 条命令**恢复到"能直接开训"的状态，不用重新踩坑。
>
> 实测环境：2026-11-17，AutoDL RTX 3080 Ti 12 GB / Ubuntu 22.04 / Python 3.12。
> 遇到的问题与解决办法见 [`environment_troubleshooting.md`](environment_troubleshooting.md)。

---

## 0. 一句话版本

本地保留 **代码（git）+ `data/processed`**，新机器上：

```bash
# 1) 传代码和数据（见 3.2）
# 2) 下载模型（见 3.3）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B-Instruct

# 3) 补依赖并验证（见 3.4 / 3.5）
pip install -r requirements-autodl.txt
python scripts/run_tests.py
```

然后按 4.1 开训。

---

## 1. 哪些东西必须保留，哪些可以重建

| 内容 | 位置（本地） | 大小 | 丢了怎么办 |
|---|---|---|---|
| 代码 | `D:\grpo-dapo-reproduction`（git 仓库） | ~6 MB | 唯一不可再生，**必须提交** |
| 已处理数据 | `data/processed/gsm8k/{train,test}.parquet` | 2.4 MB | 可由 `python -m src.data.preprocess` 从 raw 重建（raw 需另下 GSM8K，约 5 MB） |
| 模型 | `models/Qwen2.5-0.5B-Instruct/` | 954 MB | 用 3.3 的命令从 hf-mirror 重下，约 2–5 分钟 |
| 训练产物 | `outputs/E1_grpo_lora/` | 检查点 ~2 GB | **必须拉回本地**，机器释放即丢失 |
| 基线评测 | `outputs/eval_base/` | 2 MB | 已拉回本地：`D:\grpo-dapo-reproduction\outputs\eval_base\` |
| 环境 | 云机系统盘 | — | 按本文重建（镜像自带 torch，见第 5 节版本表） |

**结论**：本地只要保住 git 仓库 + `data/processed`；模型可重下；结果要及时 `scp` 回来。

---

## 2. 新实例要求

| 项 | 要求 | 实测 |
|---|---|---|
| GPU | ≥ 12 GB 显存；24 GB 更宽松 | RTX 3080 Ti 12288 MiB（驱动 595.71.05）；RTX 4090 D 24564 MiB（驱动 580.105.08） |
| 系统 | Ubuntu 22.04（AutoDL 默认 PyTorch 镜像） | Ubuntu 22.04.5，内核 5.15 |
| 数据盘 | ≥ 10 GB | `/root/autodl-tmp` 50 GB（项目放这里） |
| Python | 3.10+ | 3.12.3（miniconda base，conda 24.4.0） |
| 全参微调 | **不要在 12 GB 卡上做** | 实测差 1.39 GiB OOM，需 16–24 GB |

> AutoDL 系统盘 30 GB 也够用，但项目、模型、检查点一律放 `/root/autodl-tmp`。
>
> 24 GB 卡上 micro-batch 仍建议 **1 题**：Qwen2 的 lm_head 输出 fp32 logits，
> N=12 × L≈700 时三份 logits 就约 15 GB，实测 OOM（踩坑 #34）。
> 多题并行靠 `grad_accum_steps` 和 P1 批量 rollout，不靠加大 micro-batch。

---

## 3. 从零恢复（按顺序）

### 3.1 SSH 连接

新实例的 **host 和端口每次都会变**，更新本地 `C:\Users\EDY\.ssh\config`：

```
Host autodl
    HostName region-xx.autodl.com     # 以 AutoDL 控制台为准
    Port 12345                        # 以 AutoDL 控制台为准
    User root
    IdentityFile C:\Users\EDY\.ssh\autodl_ed25519
```

用密钥登录（生成命令在 troubleshooting 文档 #14/#15）。这样本地所有
`ssh autodl ...` / `scp autodl:...` 命令不用改。

验证：

```powershell
ssh autodl "nvidia-smi; df -h /root/autodl-tmp"
```

### 3.2 上传代码与数据

在本地 `D:\grpo-dapo-reproduction` 下执行（**不要传 models/ outputs/ data/raw/**）：

```powershell
ssh autodl "mkdir -p /root/autodl-tmp/grpo-dapo-reproduction"
scp -r GRPO configs data scripts src tests requirements.txt requirements-autodl.txt `
      README.md docs autodl:/root/autodl-tmp/grpo-dapo-reproduction/
```

- `data` 只带 `processed/gsm8k/*.parquet`（2.4 MB）。若本地没有，见第 1 节重建。
- `docs/` 传过去方便在云端查本文。

### 3.3 下载模型（约 954 MB）

```bash
export HF_ENDPOINT=https://hf-mirror.com      # 云机直连 huggingface.co 会失败
cd /root/autodl-tmp/grpo-dapo-reproduction
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B-Instruct
```

下载完确认 `model.safetensors` 约 988 MB、`config.json` 存在。
AutoDL 的"学术加速"对 HF 不可靠，用 hf-mirror 即可。

### 3.4 依赖

```bash
export PATH=/root/miniconda3/bin:$PATH       # 非交互 SSH 必须显式加 PATH
cd /root/autodl-tmp/grpo-dapo-reproduction
pip install -r requirements-autodl.txt
pip check
```

- **不要**用 `requirements.txt`（那是 macOS 环境的锁版本，会把 torch/numpy 降级）。
- 镜像一般已带 torch 2.8.0+cu128、transformers；上表只补齐/对齐版本。
- 如果 `peft` 已存在且版本一致，pip 会自动跳过。

### 3.5 验证（约 2 分钟）

```bash
export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/grpo-dapo-reproduction

# 单元测试：5 个文件全过
python scripts/run_tests.py

# 3 步真实冒烟：应看到 step 1 的 kl=0、grad_norm≈0.9、gpu_mem_gb≈9–11
python scripts/train_grpo_standalone.py --config configs/grpo_lora.yaml \
    --max-steps 3 --save-every 0 --output-dir outputs/smoke_check
```

判断标准（详见 troubleshooting「指标判读指南」）：
- step 1：`kl=0`、`ratio_mean=1.0`、`grad_norm≠0` —— 正常；
- `reward_mean` 单步波动很大 —— B=1、G=4，看滑动平均；
- `gpu_mem_gb` 若 > 11.5 且频繁出现，考虑加 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。

---

## 4. 训练 / 监控 / 续训 / 评测

### 4.1 正式训练（500 步，约 3.2 小时）

```bash
export PATH=/root/miniconda3/bin:$PATH
cd /root/autodl-tmp/grpo-dapo-reproduction
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p outputs
setsid nohup python scripts/train_grpo_standalone.py --config configs/grpo_lora.yaml \
    > outputs/train_lora.log 2>&1 < /dev/null &
```

- `setsid nohup`：断开 SSH 不影响训练。
- 每 25 步存一次检查点（`outputs/E1_grpo_lora/checkpoints/step_25` …），
  单步 16–25 s。
- 显存：实测分配峰值 8.4 GB → **10.7 GB**（回答最长的步，`gpu_mem_gb` 指标），
  12 GB 卡余量约 1.3 GB。`expandable_segments:True` 用来减少碎片导致的 OOM。
  真 OOM 也不怕：每 25 步有检查点，`--resume` 继续。

### 4.2 监控（三种，选一个）

```bash
# 每步一行
tail -f outputs/train_lora.log

# 看板：滑动平均 + ETA
python scripts/watch_training.py --output outputs/E1_grpo_lora

# 看门狗：异常/到里程碑时退出，交给调用方告警
bash scripts/watchdog_training.sh outputs/E1_grpo_lora outputs/train_lora.log 25 30
```

本地（Windows）：

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\pull_metrics.ps1 -IntervalSeconds 300
```

指标解释见 troubleshooting 的「训练时怎么实时看」与「指标判读指南」。

### 4.3 断点续训

```bash
python scripts/train_grpo_standalone.py --config configs/grpo_lora.yaml \
    --resume outputs/E1_grpo_lora/checkpoints/step_250
```

恢复内容：LoRA 权重、优化器、LR 调度器、当前步数、数据游标。日志会从第 N+1 步继续。

### 4.4 评测（训练后必须与基线同口径）

```bash
# 基线（已完成，结果在 outputs/eval_base）
python scripts/evaluate_gsm8k.py --batch-size 16 --output outputs/eval_base

# 训练后（同一个脚本、同一个 batch-size）
python scripts/evaluate_gsm8k.py --batch-size 16 \
    --adapter outputs/E1_grpo_lora/checkpoints/step_500 \
    --output outputs/eval_lora_step500
```

基线：GSM8K test 1319 题，**准确率 0.4306**，平均 319.9 token，截断率 6.75%。

---

## 5. 本次实测环境快照

| 组件 | 版本 |
|---|---|
| OS | Ubuntu 22.04.5 LTS，kernel 5.15.0-78 |
| GPU / 驱动 | RTX 3080 Ti 12288 MiB / 595.71.05 |
| CUDA（torch 内置） | 12.8 |
| Python | 3.12.3（`/root/miniconda3`，conda 24.4.0，base 环境） |
| torch | 2.8.0+cu128 |
| transformers | 4.46.3 |
| peft | 0.13.2 |
| accelerate | 1.1.1 |
| datasets | 3.2.0 |
| numpy / pandas / pyarrow | 2.3.2 / 3.1.1 / 26.0.1 |
| tokenizers / sentencepiece / safetensors | 0.20.3 / 0.2.3 / 0.8.0 |
| matplotlib / PyYAML / tqdm | 3.10.5 / 6.0.2 / 4.70.1 |
| huggingface_hub | 0.36.2 |

路径：

```
/root/autodl-tmp/grpo-dapo-reproduction/          # 项目
├── models/Qwen2.5-0.5B-Instruct/                  # 954 MB
├── data/processed/gsm8k/{train,test}.parquet      # 2.4 MB
├── configs/grpo_lora.yaml                         # 正式训练配置
└── outputs/
    ├── train_lora.log
    ├── E1_grpo_lora/{logs/metrics.jsonl, checkpoints/step_*, plots/}
    └── eval_base/{generations.jsonl, summary.json}
```

---

## 6. 已知坑（云机相关）

| 坑 | 处理 |
|---|---|
| 非交互 SSH 里 `python`/`pip` 找不到 | 每条命令前 `export PATH=/root/miniconda3/bin:$PATH` |
| 直连 huggingface.co 超时 | `export HF_ENDPOINT=https://hf-mirror.com` |
| `git clone` GitHub 返回 503（开学术加速后） | 改用本地 `scp` 上传代码 |
| 实例释放后所有数据丢失 | 代码在本地 git；检查点/结果定期 `scp` 回来 |
| B=2 训练 OOM | 保持 `prompts_per_batch: 1`（见 troubleshooting #16） |
| 12 GB 卡跑全参微调 OOM | 只用 LoRA（#17） |
| SSH 断开训练被杀 | `setsid nohup ... < /dev/null &` |
| 密码明文 / askpass 临时文件 | 用密钥登录；`%TEMP%\askpass.cmd` 用完删除，密码已建议轮换 |

更多（含每个问题的现象/分析/尝试/方案）：[`environment_troubleshooting.md`](environment_troubleshooting.md)。

---

## 7. 训练结束后要拉回本地的文件

```powershell
# 指标与日志
scp -r "autodl:/root/autodl-tmp/grpo-dapo-reproduction/outputs/E1_grpo_lora/logs" `
       "D:\grpo-dapo-reproduction\outputs\E1_grpo_lora\logs"
scp "autodl:/root/autodl-tmp/grpo-dapo-reproduction/outputs/train_lora.log" `
    "D:\grpo-dapo-reproduction\outputs\"

# 最终 adapter（约 35 MB，无优化器状态）
scp -r "autodl:/root/autodl-tmp/grpo-dapo-reproduction/outputs/E1_grpo_lora/checkpoints/step_500" `
       "D:\grpo-dapo-reproduction\outputs\E1_grpo_lora\checkpoints\"

# 训练后评测结果
scp -r "autodl:/root/autodl-tmp/grpo-dapo-reproduction/outputs/eval_lora_step500" `
       "D:\grpo-dapo-reproduction\outputs\"
```

> 检查点目录里的 `trainer_state.pt`（优化器状态，约 70 MB）只在续训时需要；
> 如果确定不再续训，可以只留 `adapter_model.safetensors` + `adapter_config.json`。
