# GRPO + DAPO 复现（Qwen2.5-0.5B-Instruct / GSM8K）

从零实现 Vanilla GRPO 与 DAPO 四项优化的核心训练代码，不依赖现成强化学习 Trainer。

**当前状态**
- 数据处理、奖励函数、Rollout、GRPO Loss、DAPO 四项优化、最小训练循环均已实现。
- 所有测试在 **CPU + Mock 采样 + 微型随机模型** 上通过（不加载 Qwen 权重，不进行真实训练）。
- **尚未用真实 Qwen2.5-0.5B 训练或评测**（本机无 GPU）。

## 项目结构

```
grpo-dapo-reproduction/
├── configs/
│   ├── model.yaml              # 模型与 LoRA 配置
│   ├── data.yaml               # 数据路径与最大长度
│   ├── grpo.yaml               # E1：Vanilla GRPO
│   └── dapo.yaml               # E5：完整 DAPO（四项优化全开）
├── data/
│   ├── raw/gsm8k/main/         # 原始 parquet（不可修改）
│   └── processed/gsm8k/        # train.parquet / test.parquet
├── src/
│   ├── data/
│   │   ├── preprocess.py       # 读取 GSM8K，提取 question / solution / ground_truth
│   │   ├── dataset.py          # 训练集/测试集封装与隔离检查
│   │   └── prompt.py           # Qwen ChatML 数学推理 Prompt
│   ├── rewards/
│   │   ├── answer_parser.py    # 从生成文本提取 \boxed / #### / Answer
│   │   ├── math_verifier.py    # 整数/小数/分数/负数/千分位等价判断
│   │   └── reward.py           # 1/0 奖励 + DAPO Eq.13 超长惩罚接口
│   ├── rollout/
│   │   ├── generator.py        # HF 采样 / Mock 采样
│   │   ├── log_probs.py        # 逐 token log prob 与对齐
│   │   └── buffer.py           # G 条回答、奖励、旧/参考 log prob 的缓存
│   ├── algorithms/
│   │   ├── advantage.py        # 组内相对优势 (r - mean) / std
│   │   ├── kl.py               # k3 无偏 KL 估计
│   │   ├── grpo_loss.py        # PPO clip + KL，sequence 归一化
│   │   └── dapo_loss.py        # Clip-Higher / Dynamic Sampling / Token-level / Overlong
│   ├── trainer/
│   │   └── trainer.py          # 最小训练循环（Rollout → Reward → Loss → 更新）
│   ├── utils/                  # YAML 配置、随机种子、日志与曲线
│   └── model_loader.py         # Qwen 模型 / Tokenizer 加载
├── scripts/
│   ├── run_tests.py            # 统一测试入口
│   ├── train.py                # 训练入口（GPU 使用）
│   ├── inspect_gsm8k.py        # 数据检查
│   └── check_environment.py    # 环境自检
├── tests/                      # CPU/Mock 单元测试
├── docs/algorithm_spec.md      # 公式、出处与实现约定
└── requirements.txt
```

## 环境安装

```bash
conda create -n grpo-dapo python=3.10 -y
conda activate grpo-dapo
pip install -r requirements.txt
```

本机为 macOS Intel：torch 固定 2.2.2（CPU）、numpy<2、sentencepiece 0.1.99。
迁移到 GPU 机器时按 CUDA 版本重装 torch：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

模型位于 `models/Qwen2.5-0.5B-Instruct/`（`configs/model.yaml` 已配置，不会自动下载）。

## 数据

GSM8K 原始数据已位于 `data/raw/gsm8k/main/`。生成预处理数据：

```bash
python -m src.data.preprocess
# train 7473 条 / test 1319 条 -> data/processed/gsm8k/
```

训练只读取 train split，加载时会校验训练集与测试集题目不重叠。

## 运行测试

```bash
python scripts/run_tests.py
```

覆盖内容：
- GSM8K 读取、数据划分隔离、Qwen Prompt 构建；
- 答案提取、数值验证、120 道真实样本人为构造回答的奖励验证；
- Mock Rollout、Buffer padding 与 log prob 对齐、微型模型逐 token log prob；
- GRPO 优势手算、Loss 手算（sequence/token）、k3 KL、Mask/Padding；
- DAPO 四项优化分别测试；
- 手工张量梯度 + 微型模型反向传播与优化器更新；
- 训练入口 dry-run 与微型模型单步训练。

## 算法配置

`configs/grpo.yaml`（基线）与 `configs/dapo.yaml`（四项全开）通过四个布尔开关控制差异：

| 开关 | GRPO | DAPO |
|---|---|---|
| `use_clip_higher` | false | true（ε_low=0.2, ε_high=0.28） |
| `use_dynamic_sampling` | false | true |
| `use_token_level_loss` | false | true |
| `use_overlong_shaping` | false | true |
| `kl_coef` | 0.04 | 0.0（DAPO 论文移除 KL） |

公式、归一化方式、与论文的差异见 `docs/algorithm_spec.md`。

## 后续在 GPU 上启动真实训练

> 以下流程尚未在 GPU 上实际验证；训练超参与数据规模仅为初始值，需要根据显存调整。

```bash
# 1. 确认配置与数据（不加载模型）
python scripts/train.py --config configs/grpo.yaml --dry-run

# 2. Vanilla GRPO
python scripts/train.py --config configs/grpo.yaml

# 3. 完整 DAPO
python scripts/train.py --config configs/dapo.yaml
```

- 日志与检查点输出到配置中的 `training.output_dir`（`metrics.jsonl`、`plots/`、`checkpoints/`）。
- 显存不足时在 `configs/model.yaml` 打开 LoRA：`lora.enabled: true`。
- `configs/dapo.yaml` 默认 `max_new_tokens=512`，可按显存与数据长度调整。

## 已知限制

- 无 GPU：训练循环仅用微型随机模型完成单步冒烟测试，未在真实 Qwen2.5-0.5B 上运行。
- 尚未实现独立评测模块（Pass@1/Pass@8），不在本阶段范围内。
- 项目奖励默认答错为 0（任务约定）；DAPO 论文 Eq.7 为 +1/-1，可通过 `reward.incorrect_reward: -1.0` 对齐，见算法文档。
- 原始数据 `data/raw/` 未被修改。
