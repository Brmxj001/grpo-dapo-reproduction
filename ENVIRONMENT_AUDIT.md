# 阶段 2 环境审计报告

- 审计日期：2026-11-14
- 项目根目录：`/Users/wangxiaobo/y研1/实习/Projects/qwen/grpo-dapo-reproduction`
- 说明：本报告只记录实际执行过的检查与实测结果；未完成项会明确标注。

## 1. 系统环境

| 项目 | 检查结果 |
| --- | --- |
| 操作系统 | macOS 15.7.9（Darwin 24.6.0，x86_64） |
| CPU | Intel Core i5-8257U @ 1.40 GHz，4 物理核 / 8 逻辑核 |
| 内存 | 8 GB（8,589,934,592 bytes） |
| GPU | Intel Iris Plus Graphics 645（集成显卡） |
| NVIDIA GPU | 无 |
| CUDA 可用性 | 不可用（`torch.cuda.is_available() == False`） |
| 磁盘空间 | 233 GiB 总量，约 11 GiB 可用（使用率 95%） |
| 当前工作目录 | `/Users/wangxiaobo/y研1/实习/Projects/qwen` |

## 2. 系统 Python 工具链（只检查，未修改）

| 项目 | 检查结果 |
| --- | --- |
| 系统 Python | 3.12.12（miniforge base：`/usr/local/Caskroom/miniforge/base/bin/python3`） |
| 系统 pip | 25.3 |
| Conda | 已安装，25.11.0，channel 为 conda-forge |
| 系统环境 PyTorch | 2.2.2（CPU），但 NumPy 2.4.3 与其 ABI 不兼容，导入时出现 `_ARRAY_API not found` 告警 |
| 处理方式 | 未修改、未删除系统环境；新建独立环境 `grpo-dapo` 规避该问题 |

## 3. 独立环境 grpo-dapo

| 项目 | 结果 |
| --- | --- |
| 环境名称 | `grpo-dapo` |
| 环境路径 | `/usr/local/Caskroom/miniforge/base/envs/grpo-dapo` |
| Python | 3.10.22 |
| pip | 26.2.1 |
| 依赖一致性 | `pip check`：No broken requirements found |
| 创建命令 | `conda create -n grpo-dapo python=3.10 -y` |

## 4. 依赖版本（实测）

### 4.1 直接依赖

| 包 | 版本 | 备注 |
| --- | --- | --- |
| torch | 2.2.2 | macOS Intel 官方最后支持版本，CPU 版，无 CUDA |
| numpy | 1.26.4 | 固定 <2，避免 torch 2.2.2 的 NumPy ABI 不兼容 |
| transformers | 4.46.3 | 支持 Qwen2.5 与 Chat Template |
| datasets | 5.0.1 | 仅安装，本阶段未下载任何数据集 |
| peft | 0.13.2 | |
| accelerate | 1.1.1 | |
| pandas | 2.3.3 | |
| matplotlib | 3.10.9 | |
| sentencepiece | 0.1.99 | 0.2.x 在 macOS x86_64 无 wheel，源码编译失败，降级解决 |
| safetensors | 0.8.0 | |
| tqdm | 4.70.1 | |
| scipy | 1.15.3 | |
| tensorboard | 2.21.0 | |
| pyyaml | 6.0.3 | |

### 4.2 关键传递依赖

| 包 | 版本 |
| --- | --- |
| huggingface-hub | 0.36.2 |
| tokenizers | 0.20.3 |
| hf-xet | 1.8.0 |
| pyarrow | 25.0.1 |
| fsspec | 2026.6.0 |
| regex | 2026.9.29 |
| jinja2 | 3.1.6 |
| typing-extensions | 4.16.0 |

所有 14 个直接依赖均实际导入成功（版本逐一核验），无导入失败项。

### 4.3 安装过程中遇到的问题

1. **sentencepiece 安装失败**：默认解析到 0.2.x，在 macOS x86_64 + Python 3.10 上没有预编译 wheel，
   源码编译 `build_bundled.sh` 返回非零。解决方案：固定 `sentencepiece==0.1.99`（有 x86_64 wheel）。
2. **NumPy 版本约束**：PyTorch 2.2.2 使用 NumPy 1.x ABI，安装 NumPy 2.x 会导致运行告警/崩溃，
   因此显式固定 `numpy==1.26.4`。
3. **PyTorch 版本上限**：PyTorch 2.3+ 不再发布 macOS x86_64 wheel，`torch==2.2.2` 是本机可用的最新版本。
   本阶段为 CPU 验证，不影响目标；迁移到 Linux CUDA 服务器时需重装对应版本。
4. 未安装 vLLM、DeepSpeed、FlashAttention 等训练加速组件（按阶段要求）。

## 5. 模型

| 项目 | 结果 |
| --- | --- |
| 来源 | Hugging Face `Qwen/Qwen2.5-0.5B-Instruct` |
| Revision | `7ae557604adf67be50417f59c2c2f167def9a775`（下载时固定 commit） |
| 本地路径 | `models/Qwen2.5-0.5B-Instruct/` |
| 目录大小 | 953 MiB |
| 文件 | config.json, generation_config.json, LICENSE, merges.txt, model.safetensors, README.md, tokenizer_config.json, tokenizer.json, vocab.json |
| 模型类型 | `qwen2`，架构 `Qwen2ForCausalLM`（支持因果语言建模） |
| 层数 / hidden / attention heads / KV heads | 24 / 896 / 14 / 2 |
| 词表大小（config） | 151,936 |
| 最大位置编码 | 32,768 |
| tie_word_embeddings | true |
| 权重 dtype（checkpoint） | bfloat16，加载时按设备转换（CPU -> float32） |
| Tokenizer | `Qwen2TokenizerFast`，len=151,665；eos=`<|im_end|>`(151645)，pad=`<|endoftext|>`(151643) |
| Chat Template | 可用（system/user/assistant，多轮与 generation prompt 均验证通过） |

## 6. 测试结果（实际运行）

### 6.1 PyTorch 测试 `scripts/test_torch.py`

- 命令：`python scripts/test_torch.py`
- 结果：**9/9 项通过，退出码 0**
- 覆盖：Tensor 创建、Tensor 加法、矩阵乘法（与 NumPy 结果最大误差 1.19e-07）、
  自动求导、反向传播（dy/dx=14.000000，符合解析解）、优化器参数更新、
  线性模型损失下降（12.597874 -> 0.00000000）、学到的参数 w=2.5000 / b=-1.0000、no_grad 行为。

### 6.2 Tokenizer 测试 `scripts/test_tokenizer.py`

- 命令：`python scripts/test_tokenizer.py`
- 结果：**14/14 项通过，退出码 0**
- 覆盖：中文（16 tokens）、英文（12 tokens）、数学公式（37 tokens）编码与解码往返一致；
  `<|im_start|>` / `<|im_end|>` / `<|endoftext|>` 特殊 Token 均编码为单 Token 且可还原；
  Chat Template 渲染以 `<|im_start|>system` 开头、以 `<|im_start|>assistant` 结尾；
  `skip_special_tokens` 两种解码模式行为正确。

### 6.3 CPU 推理测试 `scripts/test_inference.py`

- 命令：`python scripts/test_inference.py`（max_new_tokens=192, temperature=0.7, top_p=0.9, seed=42）
- 环境：CPU / float32 / 494,032,768 参数（494.0M）
- 结果：**3/3 个问题均生成非空回答，退出码 0**

| 问题 | 输入 Token | 输出 Token | 耗时 | 速度 | 是否命中 EOS |
| --- | --- | --- | --- | --- | --- |
| 2+3等于多少？ | 35 | 12 | 9.79 s | 1.23 tok/s | 是 |
| 请解释什么是强化学习。 | 35 | 192 | 37.51 s | 5.12 tok/s | 否（达到上限） |
| 请简单解释 PPO 和 GRPO 的区别。 | 40 | 192 | 38.79 s | 4.95 tok/s | 否（达到上限） |

第 1 题回答正确（“2 + 3 等于 5。”）。注意：第 3 题模型把 GRPO 错误解释成“生成对抗网络（GAN）”，
这是 0.5B 模型自身知识/能力不足导致的**回答质量问题**，不是推理管线故障；本测试只验证生成链路可用。
该现象也说明后续 GRPO 训练前模型基线较弱，评测时需以实际 Pass@1 为准。

### 6.4 环境自检 `scripts/check_environment.py`

- 命令：`python scripts/check_environment.py`
- 结果：**PASS=9，FAIL=0，WARN=0，退出码 0**
- 覆盖：Python 3.10.22、PyTorch 2.2.2、Transformers 4.46.3、14 个直接依赖全部已安装、
  CUDA 不可用（预期）、未检测到 NVIDIA GPU、模型目录 0.93 GiB 且关键文件齐全、
  Tokenizer 加载成功、模型加载成功（494,032,768 参数，float32，CPU，8.4 s）、
  CPU 贪心推理成功（31 tokens / 9.00 s）。模型加载耗时在不同运行间为 5-20 s（受磁盘缓存影响）。
- 日志：`outputs/phase2_env_check.log`；推理测试日志：`outputs/phase2_inference.log`。

## 7. 验收标准对照

| 验收项 | 状态 |
| --- | --- |
| 独立 Python 环境创建成功 | ✅ 完成（`grpo-dapo`，Python 3.10.22） |
| 依赖安装成功 | ✅ 完成（14 个直接依赖全部导入成功，pip check 通过） |
| requirements.txt 生成成功 | ✅ 完成（固定实际安装版本） |
| Qwen2.5-0.5B-Instruct 下载成功 | ✅ 完成（953 MiB，revision 已记录） |
| Tokenizer 加载成功 | ✅ 完成 |
| 模型加载成功 | ✅ 完成（Qwen2ForCausalLM，494,032,768 参数，CPU float32） |
| CPU 文本生成成功 | ✅ 完成（3/3 问题生成非空回答） |
| PyTorch 自动求导测试通过 | ✅ 完成（9/9） |
| 项目目录结构创建完成 | ✅ 完成 |
| 环境检查脚本可以运行 | ✅ 完成（PASS=9，FAIL=0，退出码 0） |
| ENVIRONMENT_AUDIT.md 内容完整 | ✅ 本文件 |

## 8. 已知限制

1. 本机无 NVIDIA GPU，只有 CPU 推理验证；CUDA 训练路径完全未验证。
2. 8 GB 内存：0.5B 模型 float32 推理可用，但无法在本机进行训练或更大模型实验。
3. 磁盘仅剩约 11 GiB，不足以容纳 GSM8K/MATH 与训练检查点，阶段 3 前需扩容。
4. 系统 base 环境中的 NumPy/PyTorch 兼容问题未处理（按要求不修改系统环境）；
   本项目一律使用 `grpo-dapo` 环境。
5. 未下载 GSM8K / MATH，未实现 GRPO / DAPO，未启动训练（符合阶段 2 范围）。

## 9. 下一阶段（阶段 3，未开始）

1. 数据准备：GSM8K / MATH 下载、清洗、prompt 模板与 train/test 划分。
2. 实现 rollout 采样（CPU 验证可先用 Transformers，云 GPU 阶段再评估 vLLM）。
3. 实现奖励函数（答案正确性 + 格式）与 Vanilla GRPO 目标。
4. 实现 DAPO 四项优化并接入配置开关。
5. 搭建训练/评测循环，记录策略熵、KL、奖励方差、Pass@1、Pass@8。
6. 设计并执行五组消融实验。
