# 项目协作约定

> 这份文件会被 Claude Code 自动加载。换电脑 clone 仓库后，这些偏好直接生效。

## 必须遵守

1. **问题必须记录**。每解决一个非平凡问题，立刻追加到
   [`docs/environment_troubleshooting.md`](docs/environment_troubleshooting.md)，格式：
   **现象 → 分析（排除了什么）→ 尝试（失败的路及原因）→ 最终方案 → 经验**。
   追加，不覆盖历史；会话结束前检查有没有漏记。
   目的：事后能复盘判断过程，而不只是看结果。
2. **长任务先报再跑**。预计超过 30 分钟、或会产生云 GPU 费用的任务，
   先给出耗时 / 显存 / 费用估算，等用户明确同意再启动。
3. **结果不能只留在临时服务器上**。AutoDL 实例释放即清空，
   训练产物必须落到本地 `outputs/` 并进 git（权重除外）。
4. **对外动作先确认**。推送 GitHub、发布 Release、花钱的操作，动手前说明要做什么。

## 项目背景

- 目标：从零复现 **GRPO**，并在此基础上做 **DAPO** 对照实验。
- 算法细节：[`docs/algorithm_spec.md`](docs/algorithm_spec.md)
- 当前进度与下一步：[`docs/session_state_2026-11-17.md`](docs/session_state_2026-11-17.md)
- E1 实验结果（GRPO + LoRA 300 步）：[`docs/experiment_E1_grpo_lora.md`](docs/experiment_E1_grpo_lora.md)
- 换服务器 / 环境还原：[`docs/server_setup.md`](docs/server_setup.md)

## 环境要点

- 本机 Windows + WSL2（无 CUDA，只做 CPU 验证）；训练在 AutoDL RTX 3080 Ti 12 GB。
- 本机 `git` 不在 PATH，用 `D:\Tools\PortableGit\cmd\git.exe`；直连 GitHub 被墙，
  需要 `-c http.proxy=http://127.0.0.1:7890`（见踩坑记录 #32）。
- 服务器非交互 SSH 必须先 `export PATH=/root/miniconda3/bin:$PATH`。
- 12 GB 卡上实测：LoRA + `prompts_per_batch=1` + 512 token，显存峰值 10.8 GB；
  B=2 会 OOM。

## 产出位置

| 类型 | 位置 | 进 git |
|---|---|---|
| 代码 / 配置 / 文档 | 仓库 | ✅ |
| 训练曲线、metrics、评测结果 | `outputs/` | ✅ |
| 模型权重、检查点 | `outputs/**/checkpoints/` | ❌（走 GitHub Release） |
