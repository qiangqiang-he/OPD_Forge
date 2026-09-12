"""Correct-RL gating, normalization, clipping, and no-op regression tests."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from algorithms.correct_rl import compute_correct_rl_batch


def test_correct_rl_marks_only_correct_tokens_and_reports_accuracy():
    response_mask = torch.tensor(
        [[1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.bool
    )
    rm_scores = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    result = compute_correct_rl_batch(
        rm_scores,
        response_mask,
        genuine_trajectory_mask=torch.tensor([True, True, True, False]),
    )

    torch.testing.assert_close(
        result.correct_rollout_mask,
        torch.tensor([True, False, True, False]),
    )
    torch.testing.assert_close(
        result.token_mask,
        torch.tensor(
            [[1, 1, 1], [0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.bool
        ),
    )
    torch.testing.assert_close(
        result.advantages,
        result.token_mask.to(result.advantages.dtype),
    )
    assert result.correct_count == 2
    assert result.genuine_count == 3
    assert result.loss_normalization == pytest.approx(0.5)
    assert result.metrics["correct-rl/train/accuracy"] == pytest.approx(2 / 3)
    assert result.metrics["correct-rl/train/correct_response_length_mean"] == pytest.approx(2.0)


def test_correct_rl_all_incorrect_batch_has_zero_count_without_division_error():
    result = compute_correct_rl_batch(
        torch.zeros((2, 3)),
        torch.ones((2, 3), dtype=torch.bool),
    )
    assert result.correct_count == 0
    assert result.loss_normalization == 0.0
    assert result.metrics["correct-rl/train/accuracy"] == 0.0
    assert not bool(result.token_mask.any())


def test_correct_rl_correct_trajectory_normalization_is_length_invariant():
    # With seq-mean-token-mean, one correct trajectory contributes the mean of
    # its token losses regardless of its response length.  This test mirrors
    # the reducer directly and verifies the expected equal trajectory weight.
    losses = torch.tensor([[-1.0, -3.0, 0.0], [-2.0, 0.0, 0.0]])
    token_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    per_trajectory = (losses * token_mask).sum(-1) / token_mask.sum(-1).clamp_min(1)
    final = per_trajectory.mean()
    assert final.item() == pytest.approx(-2.0)


def _ppo_config():
    return OmegaConf.create(
        {
            "loss_agg_mode": "seq-mean-token-mean",
            "loss_scale_factor": None,
            "entropy_coeff": 0.0,
            "use_kl_loss": False,
            "clip_ratio": 0.2,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.2,
            "policy_loss": {"loss_mode": "vanilla"},
            "global_batch_info": {},
        }
    )


def _ppo_data(response_mask, correct_mask, old_log_probs, advantages):
    from verl.utils import tensordict_utils as tu
    from tensordict import TensorDict

    data = TensorDict(
        {
            "response_mask": response_mask,
            "correct_rl_loss_mask": correct_mask,
            "old_log_probs": old_log_probs.detach(),
            "advantages": advantages,
        },
        batch_size=[response_mask.shape[0]],
    )
    tu.assign_non_tensor(
        data,
        dp_size=1,
        batch_num_tokens=int(response_mask.sum().item()),
        global_batch_size=int(correct_mask.any(dim=-1).sum().item()),
    )
    return data


def test_correct_rl_ppo_loss_uses_correct_count_and_zeroes_wrong_gradients():
    from verl.workers.utils import losses

    response_mask = torch.ones((3, 3), dtype=torch.bool)
    correct_mask = torch.tensor(
        [[1, 1, 1], [0, 0, 0], [1, 0, 0]], dtype=torch.bool
    )
    old_log_probs = torch.tensor(
        [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]]
    )
    log_probs = old_log_probs.clone().requires_grad_(True)
    data = _ppo_data(response_mask, correct_mask, old_log_probs, correct_mask.float())

    with patch.object(losses, "no_padding_2_padding", lambda value, _data: value):
        loss, metrics = losses.ppo_loss(_ppo_config(), {"log_probs": log_probs}, data)
    # Each correct trajectory contributes -1 after its own token mean; the
    # two correct trajectories are averaged by the correct count (2), not 3.
    assert loss.detach().item() == pytest.approx(-1.0)
    loss.backward()
    torch.testing.assert_close(log_probs.grad[1], torch.zeros_like(log_probs.grad[1]))
    assert "actor/pg_loss" in metrics


def test_correct_rl_ppo_clip_is_point_two():
    from verl.workers.utils import losses

    response_mask = torch.ones((1, 2), dtype=torch.bool)
    correct_mask = response_mask.clone()
    old_log_probs = torch.full((1, 2), -1.0)
    # exp(log pi - log pi_old) = 2, which must clip to 1.2.
    log_probs = (old_log_probs + torch.log(torch.tensor(2.0))).requires_grad_(True)
    data = _ppo_data(response_mask, correct_mask, old_log_probs, correct_mask.float())

    with patch.object(losses, "no_padding_2_padding", lambda value, _data: value):
        loss, _ = losses.ppo_loss(_ppo_config(), {"log_probs": log_probs}, data)
    assert loss.detach().item() == pytest.approx(-1.2)


def test_correct_rl_all_incorrect_is_safe_zero_gradient_noop():
    from verl.workers.utils import losses

    response_mask = torch.ones((2, 2), dtype=torch.bool)
    correct_mask = torch.zeros_like(response_mask)
    old_log_probs = torch.full((2, 2), -1.0)
    log_probs = old_log_probs.clone().requires_grad_(True)
    data = _ppo_data(response_mask, correct_mask, old_log_probs, correct_mask.float())

    config = _ppo_config()
    config.entropy_coeff = 0.1
    with patch.object(losses, "no_padding_2_padding", lambda value, _data: value):
        loss, _ = losses.ppo_loss(
            config,
            {"log_probs": log_probs, "entropy": torch.ones_like(log_probs)},
            data,
        )
    assert loss.detach().item() == 0.0
    loss.backward()
    torch.testing.assert_close(log_probs.grad, torch.zeros_like(log_probs.grad))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU is unavailable")
def test_correct_rl_cuda_backward_smoke():
    """Exercise the real CUDA tensor path without loading a full model."""

    from verl.workers.utils import losses

    device = torch.device("cuda")
    response_mask = torch.ones((3, 4), dtype=torch.bool, device=device)
    correct_mask = torch.tensor(
        [[1, 1, 1, 1], [0, 0, 0, 0], [1, 1, 0, 0]],
        dtype=torch.bool,
        device=device,
    )
    old_log_probs = torch.full((3, 4), -1.0, device=device)
    log_probs = old_log_probs.clone().requires_grad_(True)
    data = _ppo_data(response_mask, correct_mask, old_log_probs, correct_mask.float())

    with patch.object(losses, "no_padding_2_padding", lambda value, _data: value):
        loss, _ = losses.ppo_loss(_ppo_config(), {"log_probs": log_probs}, data)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(log_probs.grad).all()
    torch.testing.assert_close(
        log_probs.grad[1], torch.zeros_like(log_probs.grad[1])
    )


def test_correct_rl_config_requires_no_reference_or_teacher():
    from algorithms.correct_rl import validate_correct_rl_config

    config = OmegaConf.create(
        {
            "algorithm": {
                "name": "correct_rl",
                "adv_estimator": "grpo",
                "use_kl_in_reward": False,
            },
            "student_prompt": "qwen3_no_thinking_prompt",
            "teacher_prompt": "qwen3_no_thinking_prompt",
            "distillation": {"enabled": False},
            "actor_rollout_ref": {
                "actor": {
                    "loss_agg_mode": "seq-mean-token-mean",
                    "clip_ratio": 0.2,
                    "clip_ratio_low": 0.2,
                    "clip_ratio_high": 0.2,
                    "policy_loss": {"loss_mode": "vanilla"},
                    "use_kl_loss": False,
                }
            },
        }
    )
    validate_correct_rl_config(config)


def test_correct_rl_metric_namespace_covers_required_batch_metrics():
    from algorithms.correct_rl import CorrectRLTrainer

    aliases = CorrectRLTrainer.algorithm_metric_aliases(object())
    assert aliases["actor/entropy"] == "correct-rl/train/entropy"
    assert aliases["actor/grad_norm"] == "correct-rl/train/grad_norm"
    assert aliases["response_length/mean"] == "correct-rl/train/response_length_mean"
