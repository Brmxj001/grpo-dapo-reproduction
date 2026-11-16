"""GRPO / DAPO 数值与梯度测试（CPU）：

    python tests/test_grpo_dapo.py
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.advantage import (  # noqa: E402
    compute_group_advantages,
    expand_advantages,
)
from src.algorithms.dapo_loss import (  # noqa: E402
    DAPOConfig,
    compute_dapo_loss,
    dynamic_sampling_mask,
    shape_overlong_reward,
    shape_overlong_rewards,
    to_grpo_loss_config,
)
from src.algorithms.grpo_loss import GRPOLossConfig, compute_grpo_loss  # noqa: E402
from src.algorithms.kl import kl_estimator  # noqa: E402
from src.rollout.log_probs import aligned_token_log_probs  # noqa: E402
from src.utils.config import load_config  # noqa: E402

_TINY_MODEL = None


def _manual_batch():
    """两条回答、长度 2/1 的手工 batch（ratio 已给定，便于数值验算）。"""
    ratios = torch.tensor([[1.2, 0.8], [0.5, 1.0]])
    policy_log_probs = torch.log(ratios)
    old_log_probs = torch.zeros_like(policy_log_probs)
    response_mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    advantages = torch.tensor([1.0, -1.0])
    return policy_log_probs, old_log_probs, advantages, response_mask


def _tiny_model():
    global _TINY_MODEL
    if _TINY_MODEL is None:
        from transformers import Qwen2Config, Qwen2ForCausalLM

        torch.manual_seed(0)
        config = Qwen2Config(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
        _TINY_MODEL = Qwen2ForCausalLM(config).train()
    return _TINY_MODEL


def test_advantage_manual():
    rewards = torch.tensor([1.0, 3.0, 0.0, 0.0])
    # eps=0 时严格等于论文公式 (r - mean) / std，便于手工校验
    result = compute_group_advantages(rewards, num_generations=2, eps=0.0)
    assert torch.allclose(result.advantages, torch.tensor([[-1.0, 1.0], [0.0, 0.0]]))
    assert torch.allclose(result.means, torch.tensor([2.0, 0.0]))
    assert torch.allclose(result.stds, torch.tensor([1.0, 0.0]))
    assert result.zero_variance.tolist() == [False, True]

    all_same = compute_group_advantages(torch.ones(8), num_generations=4)
    assert torch.allclose(all_same.advantages, torch.zeros(2, 4))
    assert bool(all_same.zero_variance.all())

    expanded = expand_advantages(
        torch.tensor([1.0, -1.0]),
        torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    )
    assert expanded.shape == (2, 3)
    assert torch.allclose(expanded[0], torch.tensor([1.0, 1.0, 0.0]))


def test_kl_estimator():
    policy = torch.zeros(3)
    ref = torch.full((3,), 0.1)
    k3 = kl_estimator(policy, ref, "k3")
    assert torch.allclose(k3, torch.full((3,), math.exp(0.1) - 0.1 - 1.0), atol=1e-6)
    assert (k3 >= 0).all()

    k1 = kl_estimator(policy, ref, "k1")
    assert torch.allclose(k1, torch.full((3,), -0.1), atol=1e-7)

    same = kl_estimator(ref, ref, "k3")
    assert torch.allclose(same, torch.zeros(3), atol=1e-7)


def test_grpo_loss_sequence_vs_token():
    policy, old, advantages, mask = _manual_batch()

    sequence = compute_grpo_loss(
        policy, old, advantages, mask,
        config=GRPOLossConfig(normalization="sequence"),
    )
    # ratio 1.2 -> 裁剪到 1.2；0.8 -> 0.8；样本 A 平均 1.0
    # 样本 B：ratio 0.5 被下裁剪到 0.8，A=-1 -> min(-0.5, -0.8) = -0.8
    # 两种归一化分别是 (1.0 + -0.8)/2 = 0.1 与 (1.2+0.8-0.8)/3 = 0.4
    assert math.isclose(float(sequence.policy_loss), 0.1, abs_tol=1e-6)
    assert math.isclose(float(sequence.loss), -0.1, abs_tol=1e-6)
    assert math.isclose(float(sequence.clip_fraction), 1.0 / 3.0, abs_tol=1e-6)
    assert math.isclose(float(sequence.ratio_mean), (1.2 + 0.8 + 0.5) / 3.0, abs_tol=1e-6)
    assert float(sequence.num_response_tokens) == 3.0

    token = compute_grpo_loss(
        policy, old, advantages, mask,
        config=GRPOLossConfig(normalization="token"),
    )
    assert math.isclose(float(token.policy_loss), 0.4, abs_tol=1e-6)
    assert math.isclose(float(token.loss), -0.4, abs_tol=1e-6)


def test_kl_term_in_loss():
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    policy = torch.zeros_like(mask)      # ratio = 1，确保策略损失为 0
    old = torch.zeros_like(mask)
    advantages = torch.zeros(2)
    ref = torch.full_like(policy, 0.1)
    metrics = compute_grpo_loss(
        policy, old, advantages, mask, ref_log_probs=ref,
        config=GRPOLossConfig(normalization="token", kl_coef=1.0),
    )
    expected_kl = math.exp(0.1) - 0.1 - 1.0
    assert math.isclose(float(metrics.policy_loss), 0.0, abs_tol=1e-7)
    assert math.isclose(float(metrics.kl), expected_kl, abs_tol=1e-6)
    assert math.isclose(float(metrics.loss), expected_kl, abs_tol=1e-6)

    try:
        compute_grpo_loss(
            policy, old, advantages, mask, ref_log_probs=None,
            config=GRPOLossConfig(kl_coef=0.04),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("kl_coef>0 且缺少 ref_log_probs 时应抛 ValueError")


def test_dapo_clip_higher():
    policy = torch.log(torch.tensor([[1.25]]))
    old = torch.zeros_like(policy)
    mask = torch.ones_like(policy)
    advantages = torch.tensor([1.0])

    symmetric = compute_grpo_loss(
        policy, old, advantages, mask,
        config=GRPOLossConfig(clip_ratio_low=0.2, clip_ratio_high=0.2, normalization="token"),
    )
    assert math.isclose(float(symmetric.policy_loss), 1.2, abs_tol=1e-6)

    clip_higher = compute_dapo_loss(
        policy, old, advantages, mask,
        config=DAPOConfig(
            use_clip_higher=True, use_token_level_loss=False,
            clip_ratio_low=0.2, clip_ratio_high=0.28,
        ),
    )
    assert math.isclose(float(clip_higher.policy_loss), 1.25, abs_tol=1e-6)

    clip_off = compute_dapo_loss(
        policy, old, advantages, mask,
        config=DAPOConfig(
            use_clip_higher=False, use_token_level_loss=False,
            clip_ratio_low=0.2, clip_ratio_high=0.28,
        ),
    )
    assert math.isclose(float(clip_off.policy_loss), 1.2, abs_tol=1e-6)

    # 负优势时下裁剪仍然生效：ratio 0.7 -> clip 到 0.8
    policy_neg = torch.log(torch.tensor([[0.7]]))
    negative = compute_dapo_loss(
        policy_neg, old, torch.tensor([-1.0]), mask,
        config=DAPOConfig(use_clip_higher=True, use_token_level_loss=False),
    )
    # min(0.7 * -1, clip(0.7) * -1) = min(-0.7, -0.8) = -0.8，loss = +0.8
    assert math.isclose(float(negative.policy_loss), -0.8, abs_tol=1e-6)
    assert math.isclose(float(negative.loss), 0.8, abs_tol=1e-6)


def test_dapo_token_level_switch():
    policy, old, advantages, mask = _manual_batch()
    sequence_cfg = DAPOConfig(use_clip_higher=False, use_token_level_loss=False)
    token_cfg = DAPOConfig(use_clip_higher=False, use_token_level_loss=True)

    sequence = compute_dapo_loss(policy, old, advantages, mask, config=sequence_cfg)
    token = compute_dapo_loss(policy, old, advantages, mask, config=token_cfg)
    assert math.isclose(float(sequence.policy_loss), 0.1, abs_tol=1e-6)
    assert math.isclose(float(token.policy_loss), 0.4, abs_tol=1e-6)

    assert to_grpo_loss_config(sequence_cfg).normalization == "sequence"
    assert to_grpo_loss_config(token_cfg).normalization == "token"


def test_dynamic_sampling():
    flags = torch.tensor([1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0], dtype=torch.float32)
    mask = dynamic_sampling_mask(flags, num_generations=4)
    assert mask.tolist() == [False, False, True]

    single = dynamic_sampling_mask(torch.tensor([1, 0]), num_generations=2)
    assert single.tolist() == [True]


def test_overlong_shaping():
    config = DAPOConfig(
        use_overlong_shaping=True,
        overlong_max_length=100,
        overlong_soft_cache=20,
    )
    assert shape_overlong_reward(1.0, 80, config) == 1.0
    assert math.isclose(shape_overlong_reward(1.0, 90, config), 0.5, abs_tol=1e-9)
    assert math.isclose(shape_overlong_reward(1.0, 100, config), 0.0, abs_tol=1e-9)
    assert math.isclose(shape_overlong_reward(1.0, 150, config), 0.0, abs_tol=1e-9)
    assert math.isclose(shape_overlong_reward(0.0, 90, config), -0.5, abs_tol=1e-9)

    disabled = DAPOConfig(use_overlong_shaping=False, overlong_max_length=100, overlong_soft_cache=20)
    assert shape_overlong_reward(0.0, 150, disabled) == 0.0

    batch = shape_overlong_rewards([1.0, 0.0], [80, 90], config)
    assert batch[0] == 1.0 and math.isclose(batch[1], -0.5, abs_tol=1e-9)

    try:
        shape_overlong_rewards([1.0], [80, 90], config)
    except ValueError:
        pass
    else:
        raise AssertionError("长度不一致时应抛 ValueError")


def test_padding_does_not_change_loss():
    policy, old, advantages, mask = _manual_batch()
    padded_policy = torch.nn.functional.pad(policy, (0, 1), value=0.0)
    padded_old = torch.zeros_like(padded_policy)
    padded_mask = torch.nn.functional.pad(mask, (0, 1), value=0.0)

    for normalization in ("sequence", "token"):
        config = GRPOLossConfig(normalization=normalization)
        original = compute_grpo_loss(policy, old, advantages, mask, config=config)
        padded = compute_grpo_loss(
            padded_policy, padded_old, advantages, padded_mask, config=config
        )
        assert math.isclose(float(original.loss), float(padded.loss), abs_tol=1e-6)

    try:
        compute_grpo_loss(policy, old, advantages[:, None], mask[:, :1])
    except ValueError:
        pass
    else:
        raise AssertionError("形状不一致时应抛 ValueError")


def test_gradient_backprop():
    policy, old, advantages, mask = _manual_batch()
    policy = policy.clone().requires_grad_(True)

    metrics = compute_grpo_loss(
        policy, old, advantages, mask,
        config=GRPOLossConfig(normalization="sequence"),
    )
    metrics.loss.backward()

    assert policy.grad is not None
    assert torch.isfinite(policy.grad).all()
    # 手动梯度：loss = -[(1.2+0.8)/2/2 + (-0.8)/2]
    # d/dlogp_A1 = -(1/2 样本)*(1/2 token)*1.2 = -0.3
    # d/dlogp_A2 = -0.2；B 的 ratio 落在裁剪区，梯度为 0
    expected = torch.tensor([[-0.3, -0.2], [0.0, 0.0]])
    assert torch.allclose(policy.grad, expected, atol=1e-6)
    assert policy.grad[1, 1].item() == 0.0


def test_tiny_model_backprop_and_optimizer_step():
    model = _tiny_model()
    generator = torch.Generator().manual_seed(1)
    input_ids = torch.randint(1, 60, (2, 8), generator=generator)
    attention_mask = torch.ones_like(input_ids)

    policy_log_probs = aligned_token_log_probs(model, input_ids, attention_mask)
    with torch.no_grad():
        old_log_probs = aligned_token_log_probs(model, input_ids, attention_mask)

    response_mask = torch.zeros_like(input_ids, dtype=torch.float32)
    # 两条回答长度不同（5 个 vs 3 个 response token），避免对称优势梯度抵消
    response_mask[0, 3:] = 1.0
    response_mask[1, 5:] = 1.0
    advantages = torch.tensor([1.0, -1.0])

    metrics = compute_grpo_loss(
        policy_log_probs, old_log_probs, advantages, response_mask,
        config=GRPOLossConfig(normalization="sequence"),
    )
    metrics.loss.backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients, "反向传播后没有任何梯度"
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    total_grad_norm = torch.sqrt(sum(gradient.pow(2).sum() for gradient in gradients))
    assert total_grad_norm > 0, "长度不同的回答应产生非零梯度"

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.model.embed_tokens.weight.detach().clone()
    optimizer.step()
    after = model.model.embed_tokens.weight.detach()
    assert not torch.allclose(before, after), "优化器更新后参数没有变化"
    model.zero_grad(set_to_none=True)


def test_configs_load_and_map():
    grpo = load_config("configs/grpo.yaml")
    dapo = load_config("configs/dapo.yaml")

    assert grpo["model"]["local_path"] == "models/Qwen2.5-0.5B-Instruct"
    assert grpo["data"]["train_file"].endswith("train.parquet")
    assert grpo["algorithm"]["kl_coef"] == 0.04
    for key in (
        "use_clip_higher", "use_dynamic_sampling",
        "use_token_level_loss", "use_overlong_shaping",
    ):
        assert grpo["algorithm"][key] is False

    assert dapo["algorithm"]["kl_coef"] == 0.0
    for key in (
        "use_clip_higher", "use_dynamic_sampling",
        "use_token_level_loss", "use_overlong_shaping",
    ):
        assert dapo["algorithm"][key] is True
    assert dapo["reward"]["overlong"]["soft_cache_length"] == 128

    loss_config = to_grpo_loss_config(
        DAPOConfig(
            use_clip_higher=dapo["algorithm"]["use_clip_higher"],
            clip_ratio_low=dapo["algorithm"]["clip_ratio_low"],
            clip_ratio_high=dapo["algorithm"]["clip_ratio_high"],
            use_token_level_loss=dapo["algorithm"]["use_token_level_loss"],
            kl_coef=dapo["algorithm"]["kl_coef"],
        )
    )
    assert loss_config.clip_ratio_low == 0.2
    assert loss_config.clip_ratio_high == 0.28
    assert loss_config.normalization == "token"
    assert loss_config.kl_coef == 0.0


TESTS = [
    ("advantage: 组内优势数值", test_advantage_manual),
    ("kl: k1/k3 估计器", test_kl_estimator),
    ("GRPO loss: sequence/token 归一化", test_grpo_loss_sequence_vs_token),
    ("GRPO loss: KL 项", test_kl_term_in_loss),
    ("DAPO: Clip-Higher", test_dapo_clip_higher),
    ("DAPO: Token-level loss", test_dapo_token_level_switch),
    ("DAPO: Dynamic Sampling", test_dynamic_sampling),
    ("DAPO: Overlong Reward Shaping", test_overlong_shaping),
    ("Mask/Padding: 不影响 loss", test_padding_does_not_change_loss),
    ("梯度: 手工张量反向传播", test_gradient_backprop),
    ("梯度: 微型模型 + 优化器更新", test_tiny_model_backprop_and_optimizer_step),
    ("配置: YAML 加载与开关映射", test_configs_load_and_map),
]


def main() -> int:
    failures = []
    for name, test in TESTS:
        try:
            test()
        except Exception:
            failures.append(name)
            print(f"[FAIL] {name}")
            traceback.print_exc()
        else:
            print(f"[PASS] {name}")
    total = len(TESTS)
    print(f"\n结果：{total - len(failures)}/{total} 通过")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
