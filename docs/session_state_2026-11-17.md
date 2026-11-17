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
4. 可选：换 vLLM 做 rollout（GPU 利用率目前只有 23%），但改造量大。

---

## 6. 安全事项

- [ ] 本轮为推送 GitHub 生成过一个 **Personal Access Token**（fine-grained，7 天有效）。
      用完请到 GitHub → Settings → Developer settings 里 **Revoke**。
- [ ] 令牌暂存在仓库根目录 `githubtoken.txt`（已在 `.gitignore` 里），**不要** `git add -f`。
- [ ] 更早一轮对话曾明文贴过 AutoDL 的 SSH 密码，**建议在 AutoDL 控制台重置密码**。
- [ ] 本地 `%TEMP%` 下若有遗留的 `askpass` 脚本，用完删除。

---

## 7. 快速接续

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
