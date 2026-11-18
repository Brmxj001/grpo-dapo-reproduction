# 会话状态快照 · 2026-11-17

> 用途：下次开新会话、或换一台全新服务器时，读这一份就能接上。
> 环境还原的**操作步骤**见 [`server_setup.md`](server_setup.md)，
> 踩过的坑见 [`environment_troubleshooting.md`](environment_troubleshooting.md)。

---

## 1. 这次做到了哪一步

| 阶段 | 状态 | 说明 |
|---|---|---|
| 本机搭环境（Windows + WSL2） | ✅ | 无 CUDA，只做代码验证与冒烟 |
| 云 GPU 环境（AutoDL 3080 Ti 12 GB） | ✅ | torch 2.8.0+cu128，依赖见 `requirements-autodl.txt` |
| 代码改造（可观测性 / 续训 / 分块 log-prob） | ✅ | 5 个测试文件全通过 |
| 基线评测（GSM8K test 1319 题） | ✅ | **0.4306**，平均 319.9 token，截断 6.75% |
| 冒烟：单步 / 梯度累积 / 断点续训 | ✅ | 续训后指标逐位可复现 |
| 正式训练 E1（GRPO + LoRA，300 步） | ✅ | 1.87 小时，显存峰值 10.77 GB，无 OOM |
| 训练后评测 | ✅ | **0.4049**（基线 0.4306，−2.58 点，McNemar p=0.078）；长度 +11 token |
| E2 实验（4 题 × G=6，500 步，含 P1/P2） | ✅ | 2026-11-18，127.5 分钟、2,000 道题、524 tok/s；**尚未评测**，见 [`experiment_E2_grpo_lora_g6.md`](experiment_E2_grpo_lora_g6.md) |
| DAPO 对照实验 | ⬜ | 未开始，见第 5 节 |

---

## 2. 服务器状态（训练结束时）

| 项 | 值 |
|---|---|
| 实例 | AutoDL，RTX 3080 Ti 12 GB（可用 11.63 GiB），80 vCPU，128 GB 内存 |
| 工作目录 | `/root/autodl-tmp/grpo-dapo-reproduction` |
| Python | conda `base`，`/root/miniconda3/bin/python`（非交互 SSH 必须显式 export PATH） |
| 渲染产物 | `outputs/E1_grpo_lora/`（checkpoints / logs / plots） |
| 数据盘 | `/root/autodl-tmp`（**实例释放即清空**） |

⚠️ AutoDL 的实例是**开机关机计费、释放即销毁**。本地仓库 + 本文件是唯一的长期记忆，
凡是"下次还要用"的东西都必须落到 `D:\grpo-dapo-reproduction\outputs\` 或 git。

---

## 3. 本轮的关键决策

| 决策 | 选择 | 原因 |
|---|---|---|
| 训练规模 | 300 步而不是 500 步 | 先验证链路与收益趋势，1.87 小时可承受；500 步约 3.1 小时 |
| batch | `prompts_per_batch=1`、`grad_accum_steps=1` | 12 GB 上 B=2 实测 OOM；与作者 E1 基线一致 |
| 显存优化 | log-prob 按 token 分块（`chunk_size=64`） | 全量 fp32 + gather 反向峰值约 4 GB，分块后降约 1 GB |
| 评测口径 | 批量贪心 batch=16，与基线完全一致 | 逐题评测要 2.7 小时，且与批量结果不完全一致 |
| 模型产物 | 权重走 GitHub Release，不进 git | 每次训练 commit 一个 35 MB 二进制会让仓库永久膨胀 |
| 结果入库 | 曲线 / metrics / 评测 json 进 git | 换服务器后靠 git 恢复全部实验记录 |

---

## 4. 需要恢复的本地文件

| 内容 | 本地路径 | 是否在 git |
|---|---|---|
| 代码 / 文档 / 配置 | 仓库根目录 | ✅ |
| 训练曲线与指标 | `outputs/E1_grpo_lora/logs|plots/` | ✅ |
| 评测结果 | `outputs/eval_base/`、`outputs/eval_lora_step300/` | ✅ |
| LoRA 权重 step_300 | `outputs/E1_grpo_lora/checkpoints/step_300/`（117 MB） | ❌ → GitHub Release |
| 原始对话记录 | `%USERPROFILE%\.claude\projects\D--\<session>.jsonl` | ❌ 本地留存 |

> 原始对话记录**不要**放进仓库：里面可能含明文凭据。

---

## 5. 下一步（按优先级）

1. **E1 评测结论已出**：0.4049 vs 基线 0.4306（−2.58 点，p=0.078），
   长度 +11 token、截断率 +1.1 pp —— 正是"变长但没变对"。详见
   [`experiment_E1_grpo_lora.md`](experiment_E1_grpo_lora.md)。
2. **跑 DAPO 对照**：`configs/dapo.yaml` 四个开关（clip-higher / dynamic sampling /
   token-level loss / overlong shaping）。建议同样 300 步，与 E1 严格同配置，只差这四个开关。
3. **加大训练量**：300 步只有 300 道题（占训练集 4%），梯度更新次数太少。
   若预算允许，下一步应优先提高**题目数**而不是调超参。
4. **训练吞吐**：P1（批量 rollout）与 P2（零梯度跳过）已于 2026-11-18 在 `GRPO/`
   实现并实测：E2 500 步 127.5 分钟、4.01 M token，吞吐 **524 tok/s**（E1 为 60.2），
   单位时间产出提升 8.7×；GPU 采样均值 40.5%、峰值 89%。
   详见 [`experiment_E2_grpo_lora_g6.md`](experiment_E2_grpo_lora_g6.md)、
   [`perf_optimization_backlog.md`](perf_optimization_backlog.md)、踩坑记录 #33/#34。
   `src/`（DAPO）路径尚未同步，跑 DAPO 前要先移植。
5. **E2 待办**：先跑同口径评测（batch=16）拿到 test 准确率，再决定是否作为新的基线；
   权重已在本地 `outputs/E2_grpo_lora_g6/checkpoints/step_500/`。

---

## 6. 安全事项

- [ ] 本轮为推送 GitHub 生成过一个 **Personal Access Token**（fine-grained，7 天有效）。
      用完请到 GitHub → Settings → Developer settings 里 **Revoke**。
- [ ] 令牌暂存在仓库根目录 `githubtoken.txt`（已在 `.gitignore` 里），**不要** `git add -f`。
- [ ] 更早一轮对话曾明文贴过 AutoDL 的 SSH 密码，**建议在 AutoDL 控制台重置密码**。
- [ ] 本地 `%TEMP%` 下若有遗留的 `askpass` 脚本，用完删除。

---

## 7. 换一台电脑怎么继续

**会跟着 git 走的**（`git clone` 就有）：代码、配置、文档、训练曲线、metrics、评测结果。

**不会跟着走的**（需要手动搬）：

| 内容 | 原位置 | 说明 |
|---|---|---|
| SSH 私钥 + 配置 | `C:\Users\EDY\.ssh\`（`autodl_ed25519`、`config`） | **最关键**，没有它连不上服务器 |
| GitHub 令牌 | 仓库根目录 `githubtoken.txt` | 已在 `.gitignore`，需重新创建或另存 |
| Claude 记忆 | `C:\Users\EDY\.claude\projects\D--\memory\` | 记录习惯等长期偏好 |
| 对话记录 | `C:\Users\EDY\.claude\projects\D--\*.jsonl` | 可选 |

**新电脑上的步骤**：

```powershell
# 1. 装 git 和 Claude Code；把旧机器的 .ssh 整个文件夹拷到 C:\Users\<你>\.ssh\

# 2. 克隆仓库（公开仓库，不需要令牌）
git clone https://github.com/Brmxj001/grpo-dapo-reproduction.git
cd grpo-dapo-reproduction

# 3. 验证能连服务器（前提：AutoDL 实例还活着）
ssh autodl "pgrep -af 'train_grpo|evaluate_gsm8k'"

# 4. 启动 Claude Code，让它读这两份文档再动手
claude
#   「先读 docs/session_state_2026-11-17.md 和 docs/experiment_E1_grpo_lora.md，然后继续 DAPO 实验」
```

权重两大来源：服务器上的 `outputs/E1_grpo_lora/checkpoints/step_300/`，
或 GitHub Release `v0.1-E1-grpo-step300` 里的 `step_300_lora.zip`。

⚠️ AutoDL 的**关机**保留数据盘，**释放**则全部清空。换电脑前确认实例状态；
若已释放，按 [`server_setup.md`](server_setup.md) 从零重建（约 30–40 分钟，主要是下模型）。

---

## 8. 快速接续

```powershell
# 1. 连服务器
ssh autodl

# 2. 看训练/评测是否还活着
pgrep -af "train_grpo|evaluate_gsm8k"

# 3. 拉最新指标到本地
powershell -ExecutionPolicy Bypass -File .\scripts\pull_metrics.ps1 -Once

# 4. 看曲线
Get-ChildItem outputs\E1_grpo_lora\plots
```
