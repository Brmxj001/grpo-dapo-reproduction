# 环境搭建与运行问题记录

记录时间：2026-11-17
记录范围：从本机（Windows 10 + WSL2）到 AutoDL 云 GPU 的完整搭建过程。

> 说明：本文档是**操作过程记录**，不是项目算法文档。每条按「现象 → 分析 → 尝试 → 最终方案 → 经验」组织。

---

## 0. 环境概览

| 环境 | 配置 |
|---|---|
| 本机 | Windows 10 Pro 19045，Intel N95（4 核），16 GB 内存，Intel UHD 核显（无 CUDA），C: 剩 19 GB / D: 剩 126 GB |
| WSL | Ubuntu 24.04.5，Python 3.12.3，默认内存上限 7.6 GB（物理内存 50%） |
| 云 GPU | AutoDL，RTX 3080 Ti 12 GB，Ubuntu 22.04，Python 3.12.3，torch 2.8.0+cu128，125 GB 内存 |
| 网络 | Windows 有 Clash（7890/7891/7892）；WSL 与 AutoDL 均无法直接访问 huggingface.co |

---

## 问题清单

| # | 阶段 | 问题 | 最终方案 |
|---|---|---|---|
| 1 | 本机 | `git` 未安装，无 winget、无管理员权限 | 下载便携版 Git 解压到 D 盘，写入用户 PATH |
| 2 | 本机 | 目标仓库 404，无法克隆 | 仓库为私有；用户公开后克隆成功 |
| 3 | WSL | `claude` 报 command not found | 只有 Windows 版 `claude.exe`，Linux 不自动补 `.exe` |
| 4 | 本机 | venv 建在 `/mnt/d`，pip 装 60 个包耗时 10+ 分钟 | 教训：venv 应放 WSL ext4；本次硬扛完成 |
| 5 | 依赖 | `sentencepiece==0.1.99` 源码编译失败（退出码 127） | 改用 `sentencepiece==0.2.0` |
| 6 | 网络 | WSL 内 huggingface.co 直连失败 | 用 `hf-mirror.com` |
| 7 | 依赖 | `pyarrow 26` 要求 NumPy≥2，与 `numpy==1.26.4` 冲突，`read_parquet` 崩溃 | 降级 `pyarrow==17.0.0` |
| 8 | 依赖 | `datasets 5.0.1` 需要 `pyarrow>=21`，`import datasets` 报 `no attribute 'json_'` | 降级 `datasets==3.2.0` |
| 9 | 下载 | 10 个文件并发下载导致 `curl: (56) Connection reset` | 大文件单独下载 + 断点续传 + 重试 |
| 10 | 监控 | 后台等待循环盯错 PID，`pgrep -f` 匹配到自身导致死等 | 改用精确 PID 文件或 `run_in_background` |
| 11 | 训练 | WSL 7.6 GB 内存不足以跑 0.5B 全参微调（估算需 11 GB） | 本机只做冒烟；真实训练上云；或改用 LoRA |
| 12 | 云 | 非交互 SSH 下 `python`/`pip` 找不到（conda 未进 PATH） | 每次显式 `export PATH=/root/miniconda3/bin:$PATH` |
| 13 | 云 | 开学术加速后 `git clone` GitHub 返回 503 | 改用本地 `scp` 上传（仅 5.6 MB） |
| 14 | 云 | `ssh-keygen -N '""'` 在 PowerShell 中被当作字面密码 `""` | 用 `cmd /c` 传空密码参数 |
| 15 | 云 | SSH 密钥认证报 `Server accepts key` 后仍 Permission denied | 根因即 #14；重新生成空密码密钥后解决 |
| 16 | 训练 | LoRA + B=2 + 512 token 在 12 GB 卡上 OOM（11.09 GB 打满） | `log_softmax` 直接算 fp32（省一份副本）+ `prompts_per_batch` 降回 1 |
| 17 | 训练 | 全参微调 B=1 + 512 token 也 OOM（差 1.39 GiB） | 12 GB 卡不做全参；改用 LoRA（峰值 8.57 GB） |
| 18 | 判读 | 第 1 步 `loss=0 kl=0 ratio_mean=1`，`lora_A` 梯度为 0 | 不是 bug，见下方「指标判读指南」 |
| 19 | 评测 | 逐题评测 1319 题需 2.7 小时；且批量贪心与逐题贪心结果不完全一致 | 改批量（左 padding）提速 10 倍；基线与训练后用同一脚本/同一 batch 对比 |
| 20 | 流程 | 没拿到基线就启动了 3.2 小时付费训练 | 先评测后训练；长任务先报耗时/显存/费用再等确认 |
| 21 | 监控 | 训练日志里没有策略熵，也没有实时看板 | 新增 `aligned_log_probs_and_entropy`（复用前向、分块计算）+ `scripts/watch_training.py` |
| 22 | 训练 | 配置里一半字段是摆设：warmup、weight_decay、betas、adam_eps、log_every 都不生效；也没有梯度累积和断点续训 | 全部补齐，并新增 `grad_accum_steps` / `shuffle` / `--resume` |
| 23 | 本地 | `pull_metrics.ps1` 在 Windows PowerShell 5.1 下报「缺少字符串终止符」 | 无 BOM 的 UTF-8 被当成 GBK 解析；补上 UTF-8 BOM 后正常 |
| 24 | 版本控制 | 49 个已跟踪文件全部显示为「整文件修改」 | 工作区被转成 CRLF；统一转回 LF（与 HEAD 一致）后，真实改动只有 3 个文件 |
| 25 | 训练 | 断点续训冒烟失败：`checkpoints/step_2` 不存在 | `save_every=25` 时 4 步训练不会存档，且训练结束不补存；新增 `--save-every`、结尾补存、续训回退指标日志并恢复 RNG |
| 26 | 训练 | 正式训练第 12 步 backward OOM（要 1.50 GiB，只剩 1.45 GiB） | 长回答步的 log-prob 全量张量反向峰值太大；改成分块计算（`chunked_token_log_probs`），峰值降约 1 GB |
| 27 | 监控 | 看门狗在非交互 SSH 下 `python: command not found`，退化成误报 `process_gone` | 脚本内显式 `export PATH=/root/miniconda3/bin:$PATH` |
| 28 | 推送 | WSL 访问 `github.com` 超时（`api.github.com` 却 200），Windows 正常 | 代理只绑 127.0.0.1；改用 Windows 便携版 git + `GIT_ASKPASS` 推送 |
| 29 | 远程 | SSH 启动评测报 `setsid: command not found` | PowerShell 的双引号提前展开了 `$PATH`；外层改用单引号 |
| 30 | 远程 | 后台任务的日志文件"找不到"，但进程在跑 | `&` 优先级低于 `&&`，`tail` 在原 cwd 执行；一律用绝对路径 |
| 31 | 推送 | `git push` 返回 403 `Permission ... denied to Brmxj001`，但账号明明是仓库 owner | fine-grained token 只给了读权限；补上 `Contents: Read and write` 后正常 |
| 32 | 推送 | 令牌修好后 `git push` 报 `Failed to connect to github.com:443`，而 `Invoke-WebRequest` 却正常 | git 不走系统代理；给 git 显式指定 `-c http.proxy=http://127.0.0.1:7890` |

---

## 详细记录

### 1. 本机没有 git，且无管理员权限

**现象**：`git --version` 不存在；`winget` 不可用；`IsInRole(Administrator)` 为 False。

**分析**：不能装系统级软件，可选①便携版 ②安装程序（需 UAC）③不用 git。

**尝试与方案**：选**便携版 Git for Windows**：
1. 从 GitHub API 查最新版（v2.56.1），下载 `PortableGit-2.56.1-64-bit.7z.exe`（57 MB）；
2. 用 `-o<目录> -y` 参数自解压到 `D:\Tools\PortableGit`；
3. 用 `[Environment]::SetEnvironmentVariable(...,'User')` 把 `cmd` 和 `bin` 写入**用户** PATH（不动系统 PATH）。

**经验**：无管理员权限时，"解压即用 + 用户级 PATH"是最干净的方案。

---

### 2. 仓库 404，克隆要求输入用户名

**现象**：
```
fatal: could not read Username for 'https://github.com': terminal prompts disabled
```
GitHub API 查 `Brmxj001/grpo-dapo-reproduction` 返回 404，但该用户的 25 个公开仓库可正常列出。

**分析**：账号存在、仓库不存在于公开列表 → 只可能是**私有仓库**或名字写错。

**尝试**：
- 配置便携版自带的 `git-credential-manager`，想用浏览器 OAuth 登录；
- 失败：执行环境禁用了终端交互，GCM 弹不出登录窗口
  （`fatal: Cannot prompt because user interactivity has been disabled`）。

**最终方案**：用户把仓库设为公开后直接克隆成功。

**经验**：私有仓库的自动认证需要可交互环境；在非交互执行环境里，优先让用户公开仓库或用 SSH key 预置。

---

### 3. WSL 里 `claude` 找不到

**现象**：WSL 中 `claude --version` → `command not found`，但 `claude.exe --version` → `2.1.322`。

**分析**：机器上只有 Windows 版 `C:\Users\EDY\.local\bin\claude.exe`，且该目录已通过 `/mnt/c/...` 出现在 WSL 的 PATH 中。Linux 的 PATH 查找**不像 Windows 那样有 PATHEXT**，不会自动补 `.exe`。

**结论**：正确做法不是想办法跑 `claude.exe`（Windows 进程操作 WSL 文件系统会走 UNC 路径，慢且易出错），而是在 WSL 内装 Linux 原生版：
```bash
curl -fsSL https://claude.ai/install.sh | bash
```

---

### 4. venv 建在 `/mnt/d` 导致 pip 极慢

**现象**：`pip install -r requirements.txt` 安装约 60 个包，在 `Installing collected packages` 阶段卡了 10 分钟以上。

**分析**：`/mnt/d` 是 drvfs（9p 协议）挂载的 NTFS，逐个写小文件比 ext4 慢一个数量级，而 pip 安装恰好是大量小文件写入。

**当时的选择**：为节省 C 盘（仅剩 19 GB）而把 venv 放在 D 盘，代价是速度。

**更好的方案**（下次采用）：venv 放 WSL 家目录 `~/.venvs/`（ext4），只把**模型和输出**放 `/mnt/d` 或数据盘。本次已花掉的时间不值得推倒重来。

**经验**：WSL 里"代码/venv 放 ext4，数据放 NTFS"是通用原则。

---

### 5. sentencepiece 0.1.99 编译失败

**现象**：
```
subprocess.CalledProcessError: Command '['./build_bundled.sh', '0.1.99']' returned non-zero exit status 127
ERROR: Failed building wheel for sentencepiece
```

**分析**：`requirements.txt` 固定 `sentencepiece==0.1.99`，原因是作者在 macOS Intel 上 0.2.x 没有预编译 wheel。但该理由不适用于 Linux：0.1.99 没有 cp312 wheel，只能源码编译，而源码编译缺依赖（退出码 127 = command not found，缺 cmake/protobuf 工具链）。

**尝试与方案**：不改仓库文件，用 sed 生成一份临时 requirements 再安装：
```bash
sed 's/^sentencepiece==0.1.99/sentencepiece==0.2.0/' requirements.txt > /tmp/req-linux.txt
pip install -r /tmp/req-linux.txt
```
0.2.0 有 cp312 manylinux wheel，秒装。

**经验**：`requirements.txt` 里的"平台特定锁版本"注释要读，换平台时先看注释理由是否还成立。

---

### 6. WSL 无法访问 huggingface.co

**现象**：WSL 内 `curl https://huggingface.co` 立即失败（0.06 秒），而 Windows 上同一地址返回 200。

**分析**：Windows 的 Clash 走的是**系统代理**，WSL2 是独立虚拟机，不继承该代理设置。

**尝试过的思路**：
- 临时导出代理：`export https_proxy=http://$(ip route | awk '{print $3}'):7890`；
- 或在 `.wslconfig` 设 `networkingMode=mirrored`，让 `127.0.0.1:7890` 在 WSL 内可用。

**最终方案**：都不必要 —— `hf-mirror.com` 在 WSL 内直连可用（1.4 秒），设置 `HF_ENDPOINT=https://hf-mirror.com` 即可。

**经验**：国内环境下，"WSL 无代理 + 用镜像站"比"折腾代理穿透"更稳。

---

### 7. pyarrow 与 numpy 版本冲突（本次最隐蔽的坑）

**现象**：`scripts/train.py --dry-run` 报
```
ImportError: Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'.
```
直接 `import pyarrow` 才是真实错误：
```
ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4
```

**分析**：链式依赖冲突 ——
- `requirements.txt` 锁 `numpy==1.26.4`（为了 torch 2.2.2 的 NumPy 1.x ABI）；
- `datasets 5.0.1` 要求 `pyarrow>=21`；
- `pyarrow 26` 要求 `NumPy>=2`；
- 三者不可能同时满足。

**尝试**：
1. ~~升级 numpy 到 2.x~~ —— 不可行，torch 2.2.2 是按 NumPy 1.x ABI 编译的，作者文档明确警告会告警/崩溃；
2. 降 pyarrow 到支持 NumPy 1.x 的版本。

**最终方案**：
```bash
pip install "pyarrow==17.0.0"
```
同时发现 `import datasets` 仍报 `AttributeError: module 'pyarrow' has no attribute 'json_'`（datasets 5.0.1 用了 pyarrow 的新 API）。
查证项目代码**完全没有 `import datasets`**，只用 `pandas.read_parquet`，因此再降 datasets：
```bash
pip install "datasets==3.2.0"
```
最终 `pip check` 通过。

**经验**：`requirements.txt` 里固定版本是"作者在自己平台验证过的组合"，换平台时不能照抄；遇到 ImportError 要**直接 import 那个库**看真实报错，不要只看上层转述。

---

### 8. 并发下载 10 个文件导致连接被重置

**现象**：
```
curl: (56) Recv failure: Connection reset by peer
```
998 MB 的 `model.safetensors` 只下到 957 KB 就断了。

**分析**：一次性并发 10 个连接，把 hf-mirror 的连接数挤爆。

**尝试与方案**：大文件改为**单独一条连接**，并加断点续传与重试：
```bash
curl -L -C - --retry 15 --retry-delay 5 --retry-all-errors -o model.safetensors "$URL"
```
小文件用 `xargs -P 3` 限并发。

**经验**：镜像站的并发容忍度低，`-P 3` 比 `-P 10` 稳；大文件务必带 `-C -`。

---

### 9. 后台监控脚本盯错 PID

**现象**：启动脚本后，等待循环立即退出或永不退出。

**分析**：两个独立错误：
1. `nohup ... & echo $!` 拿到的是包装 shell 的 PID，不是脚本 PID；
2. 等待循环写成 `while pgrep -f setup_grpo.sh`，而**循环自身的命令行里也包含这个字符串**，`pgrep` 匹配到自己 → 永真。

**解决**：
- 用 `ps -eo pid,cmd | grep "[s]etup_grpo.sh"`（方括号技巧排除自身）；
- 更彻底：直接用工具的后台任务机制（`run_in_background`），由宿主跟踪进程，不自己写等待循环。

**经验**：不要用 `pgrep -f` 匹配会出现在自身命令行里的模式。

---

### 10. WSL 内存不足，0.5B 全参微调会 OOM

**现象（推算）**：WSL 默认内存上限为物理内存 50% = 7.6 GB。

**分析**：Qwen2.5-0.5B fp32 全参训练的显存/内存占用：

| 项目 | 估算 |
|---|---|
| 权重 494M × 4B | ~2.0 GB |
| 梯度 | ~2.0 GB |
| AdamW 两个动量 | ~4.0 GB |
| ref_model（KL 需 deepcopy） | +2.0 GB |
| logits `[N, L, 151936]` | ~0.7–1.5 GB |
| **合计** | **~11 GB** |

**实测佐证**：CPU 推理 31 token 用了 27 秒（≈1.15 tok/s）→ 单步生成 4×512 token 约需 30 分钟，500 步约 10 天。

**结论与方案**：
- 本机只做「代码正确性验证」（单元测试 + dry-run），实测全部通过；
- 真实训练必须上 GPU；
- 若必须在低显存环境训练 → 改用 LoRA：
  - 可训练参数 494M → ~8.8M（1.8%）
  - 优化器状态 4 GB → ~70 MB
  - 总占用从 ~11 GB 降到 ~4.5 GB

**经验**：先算账再动手，"能不能跑"是显存/内存问题，"值不值得跑"是算力问题，两者要分开判断。

---

### 11. AutoDL：非交互 SSH 找不到 python/pip

**现象**：
```
bash: line 3: pip: command not found
bash: line 5: python: command not found
```
但机器上 `/root/miniconda3/bin/python` 存在。

**分析**：`ssh host "command"` 启动的是**非交互、非登录** shell，不会加载 `~/.bashrc` 里 conda 的 PATH 注入。

**方案**：每条远程命令开头显式设置：
```bash
export PATH=/root/miniconda3/bin:$PATH
```
（或 `source /root/miniconda3/etc/profile.d/conda.sh && conda activate base`）

**经验**：所有自动化脚本都应在开头自己设置 PATH，不能依赖登录 shell 的配置。

---

### 12. 学术加速反而让 git clone 失败

**现象**：`source /etc/network_turbo` 后，`git clone` GitHub 返回
```
fatal: unable to access '...': The requested URL returned error: 503
```

**分析**：加速服务本身不稳定；而且 AutoDL 明确提示"开启加速后访问其他资源会更慢"。

**方案**：仓库只有 5.6 MB，直接从本机 `scp` 上传，绕开 GitHub：
```powershell
scp -r D:\grpo-dapo-reproduction\{configs,data,docs,GRPO,scripts,src,tests} autodl:/root/autodl-tmp/grpo-dapo-reproduction/
```
排除 `.venv` 与 954 MB 的 `models/`（模型在服务器上用 hf-mirror 单下）。

**经验**：小仓库"本地传"比"远端拉"更可靠。

---

### 13. SSH 密钥认证：`Server accepts key` 却是 Permission denied

**现象**：
```
debug1: Offering public key: ... id_ed25519 ED25519 SHA256:qMIWg...
debug1: Server accepts key: ... id_ed25519 ED25519 SHA256:qMIWg...
debug2: we did not send a packet, disable method
Permission denied (publickey,password).
```
服务端 `authorized_keys` 内容、权限（700/600）、公钥匹配均正常。

**分析思路**：
1. 排除服务端配置 → `authorized_keys` 内容正确；
2. 排除文件格式 → `od -c` 确认无 CRLF、结尾单个 `\n`；
3. 锁定 `we did not send a packet` = **客户端签名失败**；
4. 回到生成命令：`ssh-keygen -N '""'` —— 在 PowerShell 里 `'""'` 是**两个字面引号字符**，不是空字符串，于是私钥被设置了密码 `""`；
5. 非交互环境无法输入密码 → 签名失败。

**方案**：用 `cmd /c` 传递真正的空参数重新生成：
```powershell
cmd /c 'ssh-keygen -t ed25519 -f %USERPROFILE%\.ssh\autodl_ed25519 -N "" -C claude-code-autodl -q'
```
之后验证通过：`KEY_AUTH_OK`。

**经验**：PowerShell 传给原生 exe 的参数不走 shell 解析，`'""'` 不是空串；需要空参数时用 `cmd /c` 包装或 `--%` 停止解析。

---

### 16. LoRA + B=2 + 512 token 在 12 GB 卡上 OOM

**现象**：
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 90.00 MiB.
GPU 0 has a total capacity of 11.63 GiB of which 67.50 MiB is free.
... this process has 11.56 GiB memory in use. Of the allocated memory 11.09 GiB is allocated by PyTorch
```
配置：`prompts_per_batch=2`、`num_generations=4`、`max_new_tokens=512` → N=8，序列长 L≈620。

**分析**：账面算一下 `compute_token_log_probs` 的峰值。
词表 V=151936，logits 形状 `[N, L, V]`，fp32 每份 `4×8×620×151936 ≈ 3.0 GB`：

| 张量 | 大小 |
|---|---|
| 权重（bf16） | 1.0 GB |
| lm_head 输出 logits（bf16，被 autograd 持有） | 1.5 GB |
| `logits.float()` | 3.0 GB |
| `log_softmax` 输出（fp32，backward 需要） | 3.0 GB |
| `gather` 反向的零张量（fp32） | 3.0 GB |
| **合计** | **≈ 11.5 GB** |

与实测的 11.09 GB 吻合。原写法先 `.float()` 再 `log_softmax`，于是同一份数据在 fp32 下存在两遍。

**尝试**：
1. 一行等价优化：改用 `F.log_softmax(logits, dim=-1, dtype=torch.float32)`，
   直接在 fp32 下做归一化，不再额外生成 `logits.float()` 副本 —— 数学结果完全相同，省 3 GB；
2. 若还不够，再降 `prompts_per_batch`。

**最终方案**：两项一起做。
- `GRPO/rollout.py` 的 `compute_token_log_probs` 改为上面的一行写法；
- `configs/grpo_lora.yaml` 的 `prompts_per_batch` 改回 **1**（与作者原始 E1 基线一致）。

**实测**：B=1、G=4、512 token 时显存峰值 **8570 MiB / 11631**，单步 **22–25 秒**，500 步约 3.2 小时。

**经验**：
- 显存估算不能只算权重和优化器，**logits 张量是词表大小×序列长度，往往是最大项**；V=15 万时一个 `.float()` 就是 GB 级；
- 看到 OOM，先按「张量形状 × dtype 字节数」列账单，再去调 batch，比盲目试参数快得多。

---

### 17. 12 GB 卡上全参微调（非 LoRA）也 OOM

**现象**：同一份 3080 Ti 上跑 `configs/grpo.yaml`（全参，B=1、G=4、512 token）：
```
torch.OutOfMemoryError: Tried to allocate 1.39 GiB ... 9.86 GiB allocated by PyTorch
```
失败点是 `backward()` 里分配 1.39 GiB —— 正好对应 `[4, 620, 151936]` 的 fp32 反向缓冲区。

**分析**：全参相比 LoRA 多出来的开销：
- 梯度（bf16）≈ 1.0 GB
- AdamW 两个动量（fp32）≈ 4.0 GB
- 参考模型 deepcopy ≈ 1.0 GB
- 加回前面 3 GB 量级的 logits 缓冲区

即便把 `max_new_tokens` 砍半，也只剩 2～3 GB 余量，训练中极易因碎片再次 OOM。

**结论**：**这台 12 GB 卡上只有 LoRA 是稳妥路径**（峰值 8.57 GB，余量约 3 GB）。
若一定要做全参对照实验（E1 基线 vs LoRA），需要 24 GB 卡（4090 / A5000）或把 `max_new_tokens` 降到 256 并接受更多截断。

**经验**：先按「能不能跑」的账单选卡（这里全参需要 ~16 GB），再去谈性价比；反过来先租卡再发现跑不动，时间成本更高。

---

### 19. 评测脚本：逐题太慢，批量后结果又有细微差异

**现象**：`scripts/evaluate_gsm8k.py` 第一版逐题调用 `generate`，20 题耗时 146 秒
（7.3 秒/题）→ 全量 1319 题需要约 **2.7 小时**。

**分析**：逐题生成时 GPU 每次只处理一条序列，算力利用率极低；批量生成可以把
多条序列一起送进解码循环。

**尝试**：改成按 batch（默认 16）左 padding 批量生成。
- 必须用**左 padding**：右 padding 时生成会从 pad 位置继续往下写，结果全错；
- 批量后 32 题只用 30 秒（约 1.06 题/秒），全量约 21 分钟，**提速约 10 倍**。

**新发现**：同一批 20 题，逐题贪心与批量贪心的结果从第 100～600 个字符处开始分岔，
准确率 6/20 与 7/20 不同。排查后确认**不是左 padding 的 bug**——分岔后两边都是
通顺且同分布的续写；真正原因是批量 GEMM 与单条 GEMV 的数值差异（最后几位不同），
贪心解码在近似并列的候选上被这点差异翻转，随后级联放大。

**结论**：
- 评测统一用批量版（快 10 倍），并**固定 batch_size=16**；
- 基线与训练后的对比必须用**同一个脚本、同一个 batch size、同一个解码方式**，
  否则数值差异会混进"训练带来的提升"里；
- 报告准确率时把它当作统计量看，不要对单题差异做解读。

**经验**：批量推理的"确定性"只对同一 batch 形状成立，换 batch size 结果就会变；
跨实验对比要锁死评测配置。

---

### 20. 流程问题：没拿到基线就启动了 3.2 小时的付费训练

**现象**：冒烟测试通过后，我直接 `setsid nohup` 启动了 500 步训练（约 3.2 小时、
按小时计费）。用户随后要求"先用测试集测当前模型的能力、结果保存到本地、经我同意
再开始训练"，训练被 `pkill` 终止。

**分析**：冒烟测试只证明了"能跑"（不 OOM、梯度非零），没有证明"值得跑"——
没有基线就无法判断训练后到底是变好还是变差。此外长训练是**花钱且不可逆**的动作，
本应先确认。

**最终方案**：
1. 训练前先跑基线评测，结果落盘（`outputs/eval_base/`）；
2. 基线与训练后**用同一脚本、同一 batch size**评测；
3. 启动长任务前先得到用户明确同意。

**经验**：把"能跑通"和"要开跑"分成两道关。任何按小时计费的长任务，先给出耗时、
显存、费用估计，再等确认。

---

### 21. 训练监控缺策略熵，也没有实时看板

**现象**：清点落盘指标时发现只有 loss / policy_loss / kl / clip_fraction /
ratio_mean / reward_mean / reward_std / zero_variance_ratio / avg_response_length /
grad_norm / step_time，**没有策略熵**；实时查看只能 `tail` JSON 行，不好看趋势。

**分析**：
1. 熵要 `-Σ p log p`，需要整份 `log_softmax` 输出；而现有
   `compute_token_log_probs` 只 gather 被选中的那个 token，
   `[B, L, 151936]` 的 fp32 张量算完立刻释放 —— 这正是前面省显存的关键。
2. 所以不能直接在训练前向里加熵，否则又会引入 GB 级临时张量。

**尝试与方案**：
1. 新增 `GRPO/rollout.py::aligned_log_probs_and_entropy`，
   **复用计算 old_log_probs 的那次前向**，按序列分块（默认 64 个位置）
   算 `log_softmax` 与熵。分块后每次只存在 `[B, 64, 151936]` 的 fp32 ≈ 155 MB，
   且全程 `inference_mode` 不建图 —— **不增加前向次数**。
2. `GRPO/trainer.py` 的 `_old_and_ref_log_probs` 改用它，把 `policy_entropy`
   存进 batch；`train_step` 里按 response_mask 求平均后写进日志。
3. 顺带加了 `truncation_rate`（撞 512 token 上限的比例）。
4. `GRPO/metrics.py` 的 `DEFAULT_PLOT_METRICS` 加上这两个指标，结束时自动出图。
5. 新增 `scripts/watch_training.py`：读 `metrics.jsonl` 打印
   进度 / 剩余时间 / kl / ent / 奖励 / 梯度 / 长度 的滚动视图。

**验证**：`python scripts/run_tests.py` 五个测试文件全部通过。

**经验**：诊断指标和训练指标要分开设计 —— 诊断量用"复用前向 + 分块 + no_grad"，
不要为了多打一个数字而把主力前向的显存峰值抬上去。

---

### 22. 配置字段一半是摆设，且缺梯度累积与断点续训

**现象**：逐行核对 `GRPO/trainer.py` 读取的字段，发现 config 里有但代码从未读取的：
`warmup_ratio`、`weight_decay`、`betas`、`adam_eps`、`log_every`。
优化器实际就是 `AdamW(params, lr=lr)`；且文件头写明"不包含梯度累积、LR 调度"。

**分析**：这是"刻意保持简单"的代价。但一旦要做正式实验，后果很实际：
1. `configs/grpo.yaml` 里写着 `warmup_ratio: 0.03`，读者会以为有 warmup，**实际没有** ——
   实验记录会失真；
2. B=2 在 12 GB 卡上 OOM，而没有梯度累积就没有任何"提高等效 batch"的办法；
3. 3.2 小时的任务被打断（AutoDL 实例回收/网络断开）就得从零开始。

**方案**（`GRPO/trainer.py` + `scripts/train_grpo_standalone.py`）：
1. 优化器超参全部接上：`betas` / `eps` / `weight_decay`；
2. `warmup_ratio > 0` 时建 `LambdaLR`：线性 warmup 到基准 LR，之后恒定（GRPO 常用做法）；
3. `grad_accum_steps`：每个优化步跑多轮"采样→前向→反传"，损失除以累积步数。
   因为各组样本数相同，**多轮梯度之和 = 一次等效大 batch 的平均梯度**，数学等价；
4. `shuffle`：默认 false 保持与作者基线的数据顺序一致，需要时用固定 seed 打乱；
5. `save_checkpoint` 额外保存配置器/调度器状态与步数/数据游标；
   `load_checkpoint()` + `--resume` 续训；
6. 指标里新增 `learning_rate`，warmup 是否生效一眼可见。
7. CLI 打印改成 `B=1×累积1=每步1题`，避免"配置写 2、实际跑 1"这类误读。

**验证**：`python scripts/run_tests.py` 五个测试文件全部通过。

**经验**：配置项有两种坏法 —— 读不到（拼写/层级错）和**根本没实现**。
后者最危险，因为日志、文档、代码三者看起来完全一致，只有逐字段核对才发现。
做实验代码时，凡是 config 里出现的字段，要么真正生效，要么在文档里显式标注为"未实现"。

---

### 23. `pull_metrics.ps1` 在 Windows PowerShell 5.1 下解析失败

**现象**：
```
At scripts\pull_metrics.ps1:60 char:14
+ ... rite-Output "姣?$IntervalSeconds 绉掑悓姝ュ埌 ...
The string is missing the terminator: ".
```
脚本在 `pwsh 7`（Claude Code 的终端）里能跑，用系统自带的 `powershell`（5.1）就报错，
而且中文全变成乱码 `姣?`。

**分析**：文件是无 BOM 的 UTF-8。Windows PowerShell 5.1 **默认按 ANSI（本机 GBK）**
读取 .ps1，UTF-8 的中文多字节序列被拆错，其中某个字节恰好被解析成引号，导致字符串未闭合。
PowerShell 7 默认按 UTF-8 读，所以同一个文件在两处表现不同。

**尝试**：改成全英文提示 —— 能绕过，但用户看中文更顺，且同类问题以后还会出现。

**最终方案**：给 .ps1 加 UTF-8 BOM（`EF BB BF`），5.1 和 7 都能正确识别：
```powershell
$bytes = [IO.File]::ReadAllBytes($p)
[IO.File]::WriteAllBytes($p, [byte[]](0xEF,0xBB,0xBF) + $bytes)
```
**验证**：`powershell -File scripts\pull_metrics.ps1 -RemoteOut outputs/smoke_v2 -Once`
输出 `[14:52:28] 3 步 | kl=0.0094 ent=1.181 ...`，`metrics.csv` 也正常生成。

**经验**：Windows 上凡是带中文的 .ps1/.bat，一律存成 **UTF-8 with BOM**。
UTF-8 无 BOM 只对 PowerShell 7 安全。

---

### 24. 版本控制：49 个文件全是「整文件修改」，git 也不在 PATH

**现象**：提交前 `git status` 显示 49 个已跟踪文件全部被修改，
`git diff --stat` 有 6221 行增 / 5981 行删；而真正改过的只有 3 个文件。
另外在 PowerShell 里直接敲 `git` 报 `CommandNotFound`。

**分析**：
1. 工作区文件被转成了 CRLF（Git for Windows 在 `core.autocrlf` 未配置时，
   某些编辑器/复制操作会写成 CRLF），而仓库里存的是 LF → 每一行都算改动；
2. `git` 找不到是因为当前 PowerShell 进程启动早于用户 PATH 更新，
   便携版其实在 `D:\Tools\PortableGit\cmd\git.exe`。

**尝试**：直接提交 —— 能提交，但历史里会留下"整文件重写"的噪声 diff，
以后 blame / diff 都不可读。WSL 里已有 git 2.43，可直接操作同一个 `.git`。

**最终方案**：
```bash
# WSL 里把已跟踪的 CRLF 文件统一转回 LF
git ls-files -z | xargs -0 file | grep CRLF | cut -d: -f1 | tr '\n' '\0' \
  | xargs -0 sed -i 's/\r$//'
git add -A
```
`git diff` 立刻只剩真实改动（`GRPO/rollout.py`、`GRPO/trainer.py`、`GRPO/metrics.py`）
加 7 个新文件。提交用 WSL 的 git，不依赖 Windows PATH。

**经验**：
- 训练机是 Linux，**脚本必须 LF**（CRLF 的 `.sh` 还会让 bash 报 `\r: command not found`）；
- 换行符问题会伪装成"大改动"，提交前先看 `git diff --stat` 是否合理；
- 本机 `git` 和 WSL `git` 操作同一仓库时，注意 `core.autocrlf` 的差异。

---

### 25. 断点续训冒烟失败：`checkpoints/step_2` 不存在

**现象**：冒烟脚本先跑 4 步（期望在第 2 步存档），然后 `--resume .../step_2`：

```
FileNotFoundError: 检查点不存在：/root/autodl-tmp/grpo-dapo-reproduction/outputs/smoke_resume/checkpoints/step_2
```

**分析**：两个原因叠加。
1. `save_every` 来自配置（25），冒烟命令没有覆盖它，4 步里根本到不了第 25 步；
2. 即使到了终点，`train()` 也只在 `step % save_every == 0` 时存档，**训练结束不补存**。
   真实后果：若 `max_steps` 不是 `save_every` 的整数倍（或手动提前停止），
   最终模型不会落盘，既不能续训也不能评测。

**尝试**：只改冒烟命令的步数 —— 能绕开，但第 2 点仍是真 bug。

**最终方案**：
1. `scripts/train_grpo_standalone.py` 新增 `--save-every`，冒烟/试验可设 2 或 0；
2. `GRPO/trainer.py` 的 `train()` 在循环结束后补存一次：
   ```python
   if self.save_every > 0 and self._step % self.save_every != 0:
       self.save_checkpoint(self._step)
   ```
**验证**（两轮）：
1. 4 步 + `--save-every 2` → 存档 `step_2`、`step_4`；从 `step_2` 恢复后
   权重/优化器/步数/游标都正确恢复，但 `metrics.jsonl` 变成 `[1,2,3,4,3,4]` ——
   第 3、4 步被重算了一遍，旧记录还留着；
2. 增加 `truncate_after()` 与 RNG 状态保存后再测：
   续训时打印「指标日志已回退到第 2 步（删除 2 条旧记录）」，最终步数为 `[1,2,3,4]`，
   且第 3、4 步的所有指标（loss/kl/entropy/grad_norm/num_response_tokens/…）
   与未中断时**逐项一致**（step_time 除外），说明采样也可复现。

**经验**：检查点策略要同时覆盖"周期性"和"终点"两种情况；
冒烟测试要显式覆盖配置默认值，否则测的是配置、不是被测逻辑。

---

### 26. 正式训练第 12 步 backward OOM

**现象**：
```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.50 GiB.
GPU 0 has a total capacity of 11.63 GiB of which 1.45 GiB is free.
this process has 10.17 GiB memory in use. Of the allocated memory 9.72 GiB
is allocated by PyTorch, and 141.51 MiB is reserved but unallocated.
```
训练在第 12 步挂掉（前 11 步正常）。`gpu_mem_gb` 显示第 10/11 步
平均回答 501 token、75% 截断，峰值已到 10.86 GB。

**分析**：
1. **不是泄漏**：每步峰值随回答长度在 6–11 GB 之间正常起伏；
2. **不是碎片**：`reserved but unallocated` 只有 141 MiB；
3. **是数学上的必然峰值**：训练前向用 `compute_token_log_probs`，
   `log_softmax` 输出是 `[N, L, V]` 的 fp32 张量（N=4、L≈800、V=151936
   时约 **2 GB**），它要被反向图保存；`gather` 的反向还要再分配一个
   同尺寸的全零梯度张量（**又是 2 GB**）。长回答步这两块叠加约 4 GB，
   再加上权重、激活和生成期缓存，就顶到 11.63 GiB 了；
4. 注意 torch 报的是 **11.63 GiB 可用**（不是标称 12 GiB），余量比想象更小。

**尝试**：
- 已经开了 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，没用 —— 说明不是碎片问题；
- 降 `max_new_tokens` 或 G 都能省显存，但会改变实验口径（作者基线是 512 / G=4），放弃。

**最终方案**：新增
`GRPO/rollout.py: chunked_token_log_probs / chunked_aligned_token_log_probs`，
训练前向改用它（`trainer._micro_step`）。按 token 位置分块（默认 64），
每块单独算 `log_softmax` + `gather`：
- 反向图仍保存全部块输出（总大小不变），但 `gather` 的反向梯度只按块分配、
  用完即释放，峰值少了一份全尺寸梯度（约 2 GB）；
- 前向期间的瞬时副本也从 2 GB 降到 156 MB；
- 数学完全等价，新增测试 `test_chunked_log_probs_match_full` 对
  chunk_size = 1/2/3/64 四种情况与整份实现逐元素比对。

**验证**：
- 5 个测试文件全部通过（含新的一致性测试）；
- 重跑训练：第 12 步顺利通过；前 13 步峰值 **10.37 GB**（改前最高 10.86），
  同样 75% 截断的第 10/11 步从 10.86 降到 **9.86 GB**；
- 第 1 步 `ratio_mean=1.0000` 仍然成立，说明分块版与推理版 log prob 一致。

**经验**：报 OOM 时先看 `reserved but unallocated` —— 很小就是真实不够用，
很大才是碎片。词表大的模型（V≈15 万）最容易在 `log_softmax` + `gather`
这条路径上炸，训练侧一定要分块。

---

### 27. 看门狗误报 `process_gone`

**现象**：看门狗启动后每隔 30 秒打印 `python: command not found`，步数读成空串
（`[: : integer expression expected`），最后在训练明明还活着时报
`STATUS=process_gone`，退出码 2。

**分析**：非交互 SSH 不加载 conda 的 PATH，脚本内部调用的是 `python`。
主训练命令前有 `export PATH=...`，看门狗单独启动时没有。

**最终方案**：在 `scripts/watchdog_training.sh` 顶部加
`export PATH=/root/miniconda3/bin:$PATH`，不再依赖调用方环境。

**经验**：任何在云机上通过 `ssh host "..."` 直接运行的脚本，
都要自己保证 PATH，不要假设登录 shell 的配置生效。

---

## 评测脚本用法

```bash
# 基线（全量测试集，贪心解码，batch=16）
python scripts/evaluate_gsm8k.py --batch-size 16 --output outputs/eval_base

# 训练后的 LoRA adapter（同一套配置，才可比较）
python scripts/evaluate_gsm8k.py --batch-size 16 \
    --adapter outputs/E1_grpo_lora/checkpoints/step_500 \
    --output outputs/eval_lora_step500
```

输出：`generations.jsonl`（每题一行，含完整回答）+ `summary.json`（准确率、截断率、耗时）。

---

## 基线结果（2026-11-17，训练前）

| 项 | 值 |
|---|---|
| 模型 | `Qwen2.5-0.5B-Instruct`（未训练的原模型） |
| 测试集 | GSM8K test，**1319 题全量** |
| 解码 | 贪心（temperature=0），`max_new_tokens=512`，batch=16 |
| **准确率** | **0.4306**（568/1319） |
| 平均回答长度 | 319.9 token |
| 截断率 | 6.75%（89 题撞到 512 上限） |
| 耗时 | 984 s（16.4 分钟） |
| 结果 | 云端 `outputs/eval_base/`，本地 `D:\grpo-dapo-reproduction\outputs\eval_base\` |

这是训练后对比的唯一基准。对比时必须用**同一个脚本、同一个 batch size、同一个 `max_new_tokens`**，
因为批量贪心与逐题贪心会因 GEMM 数值差异在长回答上分叉（见问题 19）。

---

## 正式训练参数（LoRA，`configs/grpo_lora.yaml`）

| 项 | 值 | 说明 |
|---|---|---|
| `max_steps` | **500** | 优化器更新次数（不是 epoch） |
| `prompts_per_batch` | **1** | 每步只取 1 道题 |
| `num_generations` (G) | **4** | 同一道题采样 4 条回答 |
| 每步样本数 N | **4** | = 1 题 × 4 条，一次前向的序列数 |
| `max_new_tokens` | **512** | 单条回答最长 token；超长比例记在 `truncation_rate` |
| `temperature` / `top_p` / `top_k` | 1.0 / 1.0 / -1 | 纯温度采样，不截断 |
| 学习率 | **1e-4** | LoRA 专用（全参是 1e-6，因可训练参数只有 1.75%） |
| `clip_ratio` ε | 0.2 | PPO 对称裁剪 [0.8, 1.2] |
| `kl_coef` β | 0.04 | k3 估计器 |
| `normalization` | sequence | 先回答内平均，再回答间平均 |
| `max_grad_norm` | 1.0 | 梯度裁剪 |
| `save_every` | **25** | 每 25 步存一次（约 105 MB/个，20 个共 ~2 GB），最多丢 10 分钟 |
| LoRA | r=16, α=32, dropout=0.05 | q/k/v/o/gate/up/down 七个投影层 |
| 可训练参数 | 8,798,208 | 占全部 502,830,976 的 **1.75%** |
| 数据 | GSM8K train（7473 题） | **按文件顺序循环，不 shuffle** |
| 显存峰值 / 单步 | 8.4–10.7 GB / 16–25 s | 实测分配峰值（`gpu_mem_gb`）；500 步约 3.2 小时 |

**这些配置字段现在都真正生效了**（此前 `GRPO/trainer.py` 只读 `learning_rate`）：

| 字段 | 行为 |
|---|---|
| `warmup_ratio: 0.03` | 前 15 步线性 warmup 到 1e-4，之后恒定 |
| `weight_decay: 0.0` / `betas` / `adam_eps` | 传给 AdamW |
| `grad_accum_steps` | 梯度累积，等效 batch = `prompts_per_batch × grad_accum_steps`，**显存不变** |
| `shuffle` | 默认 false（与作者 E1 顺序一致）；true 时用 seed 固定打乱 |
| `log_every` | 每 N 步写一次日志 |

实时的 `learning_rate` 也进了指标，warmup 是否生效直接看得见。

---

## 训练时怎么实时看

三个层次，按需选：

1. **人类可读的每步一行**（训练启动时重定向到此文件）
   ```bash
   tail -f outputs/train_lora.log
   ```

2. **指标看板**（推荐）：读 `metrics.jsonl`，显示滑动平均与 ETA
   ```bash
   python scripts/watch_training.py --output outputs/E1_grpo_lora
   # 可选：--interval 5 --window 20 --total-steps 500
   ```
   每行含义：`step | 进度 | 剩余 | kl | ent | rew | zvar | ratio | clip | grad | len | trunc | s/step`

3. **原始数据**：`outputs/E1_grpo_lora/logs/metrics.jsonl`
   每步一行 JSON，训练中**每步 flush**，可随时 `scp` 回本地。

训练结束时会调用 `plot_metrics()`，在 `outputs/E1_grpo_lora/plots/` 生成
每个指标的 PNG 曲线（现在包含 `policy_entropy` 和 `truncation_rate`）。

| 指标 | 含义 | 健康信号 |
|---|---|---|
| `kl` | 当前策略与参考策略的 k3 KL | 从 0 缓慢上升；暴涨说明 lr 过大 |
| `policy_entropy` | 策略熵（response token 平均） | 缓慢下降；趋近 0 = 策略塌缩 |
| `reward_mean` / `reward_std` | 奖励均值 / 标准差 | 应逐步上升；单步波动大要加窗口看 |
| `zero_variance_ratio` | 组内奖励全相同的比例 | 1.0 表示该步无学习信号 |
| `ratio_mean` | πθ/πold 平均 | 贴近 1；远离 1 说明单步更新过猛 |
| `clip_fraction` | 被裁剪的 token 占比 | 早期接近 0；长期偏高说明 lr 大 |
| `grad_norm` | 裁剪前梯度范数 | 长期为 0 = 梯度断了 |
| `avg_response_length` | 平均回答长度 | 变化不宜剧烈 |
| `truncation_rate` | 撞到 512 token 上限的比例 | 过高说明该加长度或题目太难 |
| `gpu_mem_gb` | 本步张量分配峰值（不含缓存） | 稳态 ~9–11 GB；持续 >11.5 GB 要防 OOM |

---

### 28. WSL 打不开 github.com，Windows 却能打开

**现象**：训练结束要 `git push`，WSL 里报错：
```
fatal: unable to access 'https://github.com/...': GnuTLS recv error (-110)
```
同一时刻：
- WSL `curl https://github.com` → 超时（15 s 无响应）
- WSL `curl https://api.github.com` → **200**
- Windows `Invoke-WebRequest https://github.com` → **200**
- `Test-NetConnection github.com -Port 443` → True
- WSL DNS 解析出 `20.247.166.64`，地址是对的

**分析**：不是 DNS，也不是证书，是**出口路径**不同。本机代理（Clash）监听在
`127.0.0.1:7890`，只对本机 Windows 进程生效；WSL2 是独立 NAT 网络，
既拿不到 `127.0.0.1` 的代理，直连又被墙。`api.github.com` 恰好没被卡，所以出现
"一半能通用一半不通"的错觉。

**尝试**：
1. WSL 里设 `https_proxy=http://172.29.192.1:7890`（WSL 网关=Windows 宿主）→ 仍超时，
   因为 Clash 没开 `Allow LAN`，不监听 `0.0.0.0`；
2. `curl -4` 强制 IPv4 → 仍超时；
3. `netstat | findstr 7890` → 本机根本没有监听（代理当时未运行），
   注册表里的 `ProxyEnable=1` 只是残留设置。

**最终方案**：绕开 WSL 的网络，**用 Windows 上的便携版 git 推送**。
凭据不写进命令行（会进进程列表和对话记录），而是用 `GIT_ASKPASS`：
```powershell
$env:GIT_ASKPASS = "$env:TEMP\gh_askpass.cmd"   # 内容：用户名回 Brmxj001，密码 type 令牌文件
$env:GIT_TERMINAL_PROMPT = "0"
& D:\Tools\PortableGit\cmd\git.exe -c core.autocrlf=false push origin main
```

**经验**：
- "同一个网站，Windows 能开、WSL 打不开"几乎都是**代理只绑 127.0.0.1**；
  要么在代理里开 Allow LAN，要么换到 Windows 侧执行；
- `api.github.com` 通不代表 `github.com` 通，Git 走的是后者；
- 用 Windows git 操作仓库时务必带 `-c core.autocrlf=false`，否则又会引入 CRLF（见 #24）。

---

### 29. 远程命令里的 `$PATH` 被本地 PowerShell 抢先展开

**现象**：启动评测的命令执行后立刻失败：
```
bash: line 1: setsid: command not found
```
而同一份命令手动在服务器终端里能跑。

**分析**：命令写成
```powershell
ssh autodl "export PATH=/root/miniconda3/bin:\$PATH && setsid nohup python ..."
```
PowerShell 的转义符是反引号（backtick），**不是反斜杠**。`\$PATH` 里的 `\` 是字面字符，
`$PATH` 仍被 PowerShell 展开成 Windows 的 PATH，于是远端拿到的是
`PATH='/root/miniconda3/bin:\$PATH'` —— 一个只有 conda 目录的字面量路径，
`/usr/bin` 全部丢失，`setsid`、`nohup` 自然都找不到。

**尝试**：把 `\$PATH` 改成 `$env:PATH` 拼进去 —— 能跑，但会把 Windows 路径塞进 Linux 环境，
属于歪打正着。

**最终方案**：**外层命令一律用 PowerShell 单引号**，变量留给远端 bash 展开：
```powershell
ssh autodl 'export PATH=/root/miniconda3/bin:$PATH && setsid nohup python ...'
```
**验证**：`pgrep -af evaluate_gsm8k` 能看到进程，日志正常滚动。

**经验**：从 PowerShell 发远程命令，只要内容含 `$`，就用单引号包住；
这比记"几个反斜杠"可靠得多。同类坑还有 `!`（命令历史展开）和 `%`（部分场景）。

---

### 30. 后台训练的日志"不在那里"，进程却在跑

**现象**：
```powershell
ssh autodl 'cd /root/autodl-tmp/grpo-dapo-reproduction && ... nohup python train.py > outputs/x.log 2>&1 & sleep 8; tail outputs/x.log'
```
`tail` 报 `No such file or directory`，但 `pgrep` 显示进程活得好好的。

**分析**：shell 里 `&` 的优先级**低于** `&&`。`A && B & C` 实际是
`(A && B) &` 然后 `C`。C（也就是 `sleep` 和 `tail`）在**原 shell 的工作目录**（`/root`）执行，
所以它找的是 `/root/outputs/x.log`；而进程重定向的文件在 `cd` 之后的目录里。

**最终方案**：排查/查看日志时用**绝对路径**，启动脚本里也统一写绝对路径：
```bash
tail -f /root/autodl-tmp/grpo-dapo-reproduction/outputs/train_lora.log
```

**经验**：一条 SSH 命令里既 `cd` 又后台又查看日志时，等于把三个 shell 语义揉在一起。
要么拆成两条 ssh，要么一律绝对路径。

---

### 31. 令牌有账号权限却推不上去（403）

**现象**：
```
remote: Permission to Brmxj001/grpo-dapo-reproduction.git denied to Brmxj001.
fatal: unable to access '...': The requested URL returned error: 403
```
而 `gh auth status` 显示登录的就是仓库 owner `Brmxj001`，
GitHub API 返回的 `permissions.push` 也是 `true`。

**分析**：`permissions` 字段描述的是**这个账号在仓库里的角色**，不是**这个令牌被授予的权限**。
fine-grained token 生成时如果选的是 "Public repositories (read-only)"，
或者勾了仓库却没给 `Contents: write`，就出现"账号是管理员、令牌只能读"的组合。

**定位手段**（一眼区分读/写权限，且不会在日志里泄露令牌）：
```powershell
curl.exe -s -o NUL -w "%{http_code}\n" -u "Brmxj001:$TOK" `
  "https://github.com/<owner>/<repo>.git/info/refs?service=git-receive-pack"   # 403 = 无写权限
curl.exe -s -o NUL -w "%{http_code}\n" -u "Brmxj001:$TOK" `
  "https://github.com/<owner>/<repo>.git/info/refs?service=git-upload-pack"    # 200 = 可读
```

**最终方案**：在 token 设置页把 **Contents** 改成 `Read and write`（fine-grained token 编辑后令牌值不变）。

**经验**：遇到 403 先分清是**账号权限**还是**令牌范围**。API 的 `permissions` 字段会骗人，
直接打 git 的 `receive-pack` 端点最准。

---

### 32. 同一台机器：PowerShell 能访问 GitHub，git 却连不上

**现象**：
```powershell
Invoke-WebRequest https://github.com        # 200
git push                                     # Failed to connect to github.com:443 after 21s
```

**分析**：`Invoke-WebRequest`（.NET）默认走 **WinINET 系统代理**（本机是 `127.0.0.1:7890`），
而 Git for Windows 自带的 libcurl **不读系统代理**，直连 `github.com:443`。
代理开着时能通的地方，换成 git 就变成"直连被墙"。

**最终方案**：给 git 显式加代理（一次性的 `-c`，不改全局配置）：
```powershell
git -c http.proxy=http://127.0.0.1:7890 -c https.proxy=http://127.0.0.1:7890 push origin main
```
**验证**：`e67ea6b..5b12b6f  main -> main`。

**经验**：判断"是不是代理问题"最快的办法是看**谁在访问** —— 浏览器/.NET 走系统代理，
curl/git 走环境变量 `HTTP_PROXY`，Python `requests` 又走另一套。同一个 URL 结果不同是常态。

---

## 指标判读指南（哪些"看起来不对"其实是正常的）

训练早期几类"异常"数字，实测确认都是设计使然，不要误判成 bug：

| 现象 | 为什么正常 |
|---|---|
| 第 1 步 `loss=0`、`policy_loss=0` | 此时 `ratio≡1`，代理目标等于优势本身；而组内优势是零均值的（ΣA=0），所以按回答平均后恰好抵消为 0。**关键是 `grad_norm≠0`**（实测 0.91），说明梯度真实存在。 |
| 第 1 步 `kl=0`、`ratio_mean=1.0` | LoRA 的 `lora_B` 初始化为全 0，adapter 输出为 0，模型输出与原模型逐位相同，policy 与 ref 完全一致，KL 必然为 0。这同时也是**参考策略实现正确**的证据。 |
| 336 个 LoRA 参数中只有 168 个有非零梯度 | 168 个是 `lora_B`。`lora_A` 的梯度 = `Bᵀ·…`，而 B=0，所以第一步必然为 0 —— PEFT 的标准行为。 |
| 第 2 步起 `reward_mean=0` | 基座模型本身在多数 GSM8K 题上就是 0/4 全错。实测对照：题目 0 基座奖励 `[1,0,1,1]`（0.75），与 LoRA 训练第 1 步完全一致；题目 2/4 基座为 0/4。B=1、G=4 时奖励方差极大，需看滑动平均而非单步。 |
| `zero_variance_ratio=1.0` | 该步 4 条回答奖励全相同，优势全为 0，这一步不产生学习信号。vanilla GRPO 没有动态采样，这类步会白白浪费；DAPO 的 dynamic sampling 正是为此设计。 |
| **`loss` 300 步几乎不动（0.003 上下）** | GRPO 的 loss = `policy_loss + β·kl`。组内优势零均值，且起始时 `ratio≈1`，所以 `policy_loss≈0` 是**数学必然**；剩下的 `β·kl = 0.04×0.085 ≈ 0.0034` 就是实测的 0.0035。它是"约束记账"，**不是进度条**。判断好坏请看 `reward_mean`（升）和 `grad_norm`（非零），而不是 loss 下降。 |

---

## 决策记录（为什么这么做）

| 决策 | 选择 | 理由 | 代价 |
|---|---|---|---|
| git 安装方式 | 便携版 | 无管理员权限，不污染系统 | 需手动维护 PATH |
| venv 位置 | `/mnt/d` | 省 C 盘（仅剩 19 GB） | pip 慢 5–10 倍 |
| 模型下载源 | hf-mirror.com | WSL/云机直连 HF 均失败 | 镜像偶发断连 |
| numpy 策略 | 保持 1.26.4，降 pyarrow | torch 2.2.2 的 ABI 约束不能动 | datasets 需降到 3.2.0 |
| 训练方式 | 本机验证 + 云 GPU 训练 | 无 CUDA，CPU 仅 1.15 tok/s | 需租用云实例 |
| LoRA 的 ref_model | 用 `disable_adapter()` 而非 deepcopy | 基座冻结，参考策略=关掉 adapter 的同一模型 | 需改 3 处代码 |
| LoRA 的 batch | `prompts_per_batch=1` | 12 GB 卡上 B=2 实测 11.09 GB OOM；且与作者 E1 基线一致 | 单步梯度噪声大 |
| 本卡训练方式 | 只做 LoRA | 全参实测需 >11.6 GB，12 GB 卡放不下 | 无法做全参对照 |

---

## 待办 / 未验证项

- [x] 云上跑通完整测试套件（5 个测试文件全部通过）
- [x] 单步冒烟（真实 Qwen + 真实 rollout）：可训练参数 8,798,208（1.75%）、显存峰值 8.57 GB、单步 22–25 s
- [x] 实测 GRPO 单步耗时与显存峰值 → `G=4`、`prompts_per_batch=1`、`max_new_tokens=512`
- [x] LoRA 三处改动已完成（套 adapter / `disable_adapter()` 作参考策略 / lr=1e-4）
- [x] 12 GB 下 logits 占用问题：`F.log_softmax(dtype=float32)` 省一份 fp32 副本，B=2 仍 OOM → 定 B=1
- [x] 基线评测：GSM8K test 全量 1319 题，准确率 **0.4306**，平均 319.9 token，截断 6.75%
- [x] 冒烟 1：`policy_entropy` / `learning_rate` / `truncation_rate` / `gpu_mem_gb` 都正常落盘
- [x] 冒烟 2：`--grad-accum-steps 2` 等效 B=2，显存不涨、无 OOM，单步约 31 s
- [x] 冒烟 3：`--save-every 2` → 存档 step_2/step_4；从 step_2 续训，日志回退后重算，指标可复现
- [x] 完整训练 E1：300 步（1.87 小时，显存峰值 10.77 GB），结果见 `docs/experiment_E1_grpo_lora.md`
- [x] 训练后评测：0.4049 vs 基线 0.4306（−2.58 点，McNemar p=0.078），长度 +11 token
- [ ] DAPO 对照实验（同预算 300 步，只差四个开关）
- [ ] 扩大训练量：300 步只用了 4% 训练集，下一步优先加题目数
