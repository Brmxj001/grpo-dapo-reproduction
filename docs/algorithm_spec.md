# 核心算法说明（GRPO / DAPO）

本文档记录本项目采用的公式、代码位置、实现选择，以及与原始论文的差异。
公式均核对自原始论文；引用见文末。

## 1. 记号

- 每题采样 G 个回答，记为 $o_i$，长度 $|o_i|$；
- $\pi_\theta$：当前策略；$\pi_{\theta_{old}}$：采样时的旧策略；$\pi_{ref}$：参考策略；
- $\rho_{i,t} = \pi_\theta(o_{i,t}\mid q,o_{i,<t}) / \pi_{\theta_{old}}(o_{i,t}\mid q,o_{i,<t})$：重要性采样比率；
- $A_i$：组内相对优势；GRPO 中同一回答的所有 token 共享同一个标量优势（Outcome Supervision）。

## 2. PPO 裁剪目标（GRPO 的组件）

PPO（Schulman et al., 2017）对每个 token 使用：

$$
\mathcal{L}^{clip}_{i,t} = \min\left(\rho_{i,t}A_{i,t},\ \mathrm{clip}(\rho_{i,t},1-\varepsilon,1+\varepsilon)A_{i,t}\right)
$$

实现见 `src/algorithms/grpo_loss.py::compute_grpo_loss`。最小值结构保证更新是“悲观”的：
正优势时不能因比率过大获得超额收益，负优势时不能因比率过小获得超额惩罚。

## 3. Vanilla GRPO

### 3.1 组内优势

$$
A_i = \frac{r_i - \mathrm{mean}(r_1,\dots,r_G)}{\mathrm{std}(r_1,\dots,r_G)}
$$

对应 `src/algorithms/advantage.py::compute_group_advantages`。

实现约定：
- **标准差使用总体标准差**（`unbiased=False`）。论文未注明总体/样本，代码中显式固定，保证不同长度/批次下可复现；
- 公式分母加入 `advantage_eps=1e-4`（仅为数值稳定，论文无此 eps）。测试中可传 `eps=0` 严格对照公式；
- 组内奖励完全相同时，标准差为 0、分子也为 0，实现中显式将优势置 0。

### 3.2 GRPO 目标函数

DeepSeekMath 论文 Eq.3（附录 Eq.21）：

$$
\mathcal{J}_{GRPO}(\theta)=\mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|}
\min\left(\rho_{i,t}A_{i,t},\ \mathrm{clip}(\rho_{i,t},1-\varepsilon,1+\varepsilon)A_{i,t}\right)
-\beta\,\mathbb{D}_{KL}(\pi_\theta\|\pi_{ref})\right]
$$

实现优化的是 $-\mathcal{J}$，即：

```text
loss = -policy_loss + kl_coef * kl_loss
```

归一化采用论文的 **sequence** 方式：先对每条回答的 response token 求平均，再对回答求平均
（`GRPOLossConfig.normalization="sequence"`）。

### 3.3 KL 估计（k3）

GRPO 论文使用 Schulman (2020) 的无偏估计器 k3，逐 token 计算：

$$
\widehat{KL}_{i,t}=\frac{\pi_{ref}}{\pi_\theta}-\log\frac{\pi_{ref}}{\pi_\theta}-1
=\exp(\ell^{ref}_{i,t}-\ell^\theta_{i,t})-(\ell^{ref}_{i,t}-\ell^\theta_{i,t})-1
$$

实现见 `src/algorithms/kl.py::kl_estimator`。KL 与策略损失使用相同的归一化方式，
默认系数 `kl_coef=0.04`（DeepSeekMath 实验取值）。k3 恒非负。

## 4. DAPO

### 4.1 DAPO 目标

DAPO 论文 Eq.8 / Eq.10：

$$
\mathcal{J}_{DAPO}(\theta)=\mathbb{E}\left[\frac{1}{\sum_{i=1}^{G}|o_i|}\sum_{i=1}^{G}\sum_{t=1}^{|o_i|}
\min\left(\rho_{i,t}A_{i,t},\ \mathrm{clip}(\rho_{i,t},1-\varepsilon_{low},1+\varepsilon_{high})A_{i,t}\right)\right]
$$

与 GRPO 的两处核心差异：

1. 非对称裁剪（Clip-Higher）；
2. 分母为整个 batch 的 response token 总数 $\sum_i|o_i|$（Token-level loss，Eq.12）。

DAPO 论文 §2.3 明确 **移除 KL 惩罚**，因此 `configs/dapo.yaml` 中 `kl_coef: 0.0`。
实现中仍保留 KL 接口，便于消融，但默认关闭。

### 4.2 Clip-Higher（Eq.10）

上下裁剪范围解耦：$\varepsilon_{low}=0.2$，$\varepsilon_{high}=0.28$。
实现为 `GRPOLossConfig` 的 `clip_ratio_low / clip_ratio_high`，
`DAPOConfig.use_clip_higher=false` 时退化为对称裁剪（取 `clip_ratio_low` 作 $\varepsilon$）。
对应 `to_grpo_loss_config()`。

论文依据：上界过紧会限制低概率 token 的概率提升，导致熵崩缩。

### 4.3 Dynamic Sampling（Eq.11）

只保留满足下式的题组：

$$
0 < \left|\{o_i \mid \text{is\_equivalent}(a,o_i)\}\right| < G
$$

即过滤“全对”和“全错”的组。实现分两层：

- `src/algorithms/dapo_loss.py::dynamic_sampling_mask`：输入每题 G 个正确性标记，返回有效组掩码；
- `src/trainer/trainer.py::collect_rollout_batch`：被过滤的题目丢弃并继续采样，直到凑满配置的题组数。

**与论文的差异**：论文表述为“持续过采样直到 batch 填满”；实现中增加了
`max_sampling_rounds` 上限，避免模型能力极弱（全错）时训练死循环。这是工程保护，
不改变过滤条件本身。

注意：过滤依据是 **答案正确性**（0/1），不是经过长度惩罚后的奖励，符合 Eq.11 的约束。

### 4.4 Token-level Policy Gradient Loss（Eq.12）

两种归一化：

- **sequence（GRPO）**：$\frac{1}{G}\sum_i \frac{1}{|o_i|}\sum_t(\cdot)$，每条回答等权；
- **token（DAPO）**：$\frac{1}{\sum_i|o_i|}\sum_i\sum_t(\cdot)$，每个有效 response token 等权。

实现：`GRPOLossConfig.normalization` 取 `"sequence"` 或 `"token"`，
由 `DAPOConfig.use_token_level_loss` 控制。分母 clamp 到最小值 1，mask 外的 padding 不参与。

### 4.5 Overlong Reward Shaping（Eq.13）

DAPO 的软长度惩罚，**加到规则正确性奖励上**：

$$
R_{length}(y)=
\begin{cases}
0, & |y| \le L_{max}-L_{cache}\\
\dfrac{(L_{max}-L_{cache})-|y|}{L_{cache}}, & L_{max}-L_{cache}<|y|\le L_{max}\\
-1, & |y|>L_{max}
\end{cases}
$$

实现：
- `src/rewards/reward.py::overlong_penalty`：严格按上式，区间 $[-1,0]$；
- `src/algorithms/dapo_loss.py::shape_overlong_reward(s)`：`reward = 正确性奖励 + penalty`，
  仅在 `use_overlong_shaping=True` 时生效；
- 配置：`reward.overlong.{max_response_length, soft_cache_length, penalty_scale}`。
  `penalty_scale` 默认 1.0（与论文一致），仅用于额外消融。

边界：生成被截断时长度等于 `max_new_tokens`，若其等于 $L_{max}$，代入第二段得 -1，
与论文第三段保持一致。

### 4.6 与论文的一处已知差异：基础奖励

DAPO 论文 Eq.7 的有符号奖励为：

$$
R(\hat y, y)= \begin{cases}+1, & \text{is\_equivalent}(\hat y,y)\\ -1, & \text{otherwise}\end{cases}
$$

本项目按任务约定默认 **答对 +1、答错 0、无法解析 0**（`RewardConfig`）。
这不是笔误，而是需求规定的基线奖励。如需严格对齐 DAPO 论文，可在 YAML 中设置：

```yaml
reward:
  incorrect_reward: -1.0
```

该差异同时影响 E1 与 E5，因此做 GRPO/DAPO 对比时保持同一奖励设置即可。

## 5. Token 对齐与 Mask

- Rollout 记录 `prompt_ids + response_ids`，`response_mask` 在 response 位置为 1；
- 模型 logits 位置 $t$ 预测 token $t+1$，`aligned_token_log_probs` 将 log prob 左移一列，
  使 `aligned[:,t]` 对应 `input_ids[:,t]` 的产生概率（第 0 列填 0）；
- `RolloutBuffer.to_batch` 右 padding，`old_log_probs / ref_log_probs` 写在
  `[prompt_len, prompt_len+response_len)` 区间，与 response_mask 对齐；
- padding token 的 mask 为 0，不参与 loss / 梯度。

## 6. 代码映射

| 组件 | 代码 |
|---|---|
| 组内优势 | `src/algorithms/advantage.py::compute_group_advantages` |
| PPO/GRPO 损失 | `src/algorithms/grpo_loss.py::compute_grpo_loss` |
| k3 KL | `src/algorithms/kl.py::kl_estimator` |
| DAPO 四项开关 | `src/algorithms/dapo_loss.py::DAPOConfig` |
| Dynamic Sampling 掩码 | `src/algorithms/dapo_loss.py::dynamic_sampling_mask` |
| Overlong Eq.13 | `src/rewards/reward.py::overlong_penalty`、`dapo_loss.py::shape_overlong_reward` |
| 逐 token log prob | `src/rollout/log_probs.py::aligned_token_log_probs` |
| Rollout 缓存与对齐 | `src/rollout/buffer.py::RolloutBuffer.to_batch` |
| 训练循环 | `src/trainer/trainer.py::GRPODAPOTrainer.train_step` |

## 7. 验证状态

- 公式实现通过手工数值测试（优势、两种归一化、裁剪、KL、mask/padding）；
- 训练路径通过微型随机 Qwen2 模型的单步前向/反传/优化器更新测试；
- **未验证**：真实 Qwen2.5-0.5B 的生成、训练稳定性、显存占用与最终效果。
  训练超参数为初始值，需要在 GPU 上重新调参。

## 8. 参考文献

1. PPO: Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv:1707.06347, 2017.
2. GRPO: Shao et al., *DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models*, arXiv:2402.03300, 2024（§4.1, Eq.3/21）。
3. DAPO: Yu et al., *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*, arXiv:2503.14476, 2025（§2.3, §3.1–3.4, Eq.7–13）。
4. KL 估计器：Schulman, *Approximating KL Divergence*, 2020（k3 估计）。
