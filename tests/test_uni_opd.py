"""CPU-only regression tests for Uni-OPD and its production configs."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from algorithms.uni_opd import (
    compute_uni_opd_batch,
    select_correctness_balanced_rollouts,
    validate_uni_opd_config,
)
from utils.opd_runtime import configure_opd_data_parallel_batch
from verl.trainer import main_ppo_sync as verl_sync
from verl.trainer.distillation import losses
from verl.utils import tensordict_utils as tu
from verl.workers.config import DistillationLossConfig


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs" / "Uni_OPD"
LOSS_DEFAULTS = OmegaConf.load(
    PROJECT_ROOT
    / "verl"
    / "verl"
    / "trainer"
    / "config"
    / "distillation"
    / "distillation.yaml"
).distillation_loss


def _load_config(path: Path):
    config = OmegaConf.merge(
        OmegaConf.load(PROJECT_ROOT / "configs" / "base.yaml"),
        OmegaConf.load(path),
    )
    config.distillation.distillation_loss = OmegaConf.merge(
        LOSS_DEFAULTS, config.distillation.distillation_loss
    )
    return config


def test_uni_opd_means_tokens_shifts_group_margin_and_broadcasts():
    student = torch.zeros((4, 3), requires_grad=True)
    teacher = torch.tensor(
        [
            [0.1, 0.3, 99.0],  # G=0.2, correct
            [0.2, 0.2, 0.2],  # G=0.2, incorrect
            [0.8, 0.8, 0.8],  # G=0.8, correct
            [0.1, 0.1, 0.1],  # G=0.1, incorrect
        ],
        requires_grad=True,
    )
    response_mask = torch.tensor(
        [
            [True, True, False],
            [True, True, True],
            [True, True, True],
            [True, True, True],
        ]
    )
    result = compute_uni_opd_batch(
        student,
        teacher,
        response_mask,
        verifier_rewards=torch.tensor([1.0, 0.0, 1.0, 0.0]),
        prompt_group_ids=["a", "a", "b", "b"],
        margin_delta=0.4,
    )

    torch.testing.assert_close(
        result.raw_returns, torch.tensor([0.2, 0.2, 0.8, 0.1])
    )
    # Group a has margin 0 and is spread by +/-0.2.  Group b already has
    # margin 0.7 and remains unchanged.
    torch.testing.assert_close(
        result.calibrated_returns, torch.tensor([0.4, 0.0, 0.8, 0.1])
    )
    torch.testing.assert_close(
        result.token_advantages,
        torch.tensor(
            [
                [0.4, 0.4, 0.0],
                [0.0, 0.0, 0.0],
                [0.8, 0.8, 0.8],
                [0.1, 0.1, 0.1],
            ]
        ),
    )
    assert result.keep_mask.all()
    assert result.metrics["uni-opd/train/shifted_group_count"] == 1.0
    assert result.metrics["uni-opd/train/margin_after_mean"] == pytest.approx(
        0.55
    )
    assert not result.raw_returns.requires_grad
    assert not result.calibrated_returns.requires_grad
    assert not result.token_advantages.requires_grad


def test_correctness_balancing_is_uniform_across_prompt_groups():
    prompt_groups = [group for group in "abcd" for _ in range(4)]
    # Every group contains three correct and one incorrect trajectory.
    correctness = torch.tensor(
        [value for _ in "abcd" for value in (True, True, True, False)]
    )

    keep = select_correctness_balanced_rollouts(
        correctness, prompt_groups, seed=123
    )

    assert int(keep.sum()) == 8
    assert int((keep & correctness).sum()) == 4
    assert int((keep & ~correctness).sum()) == 4
    # The minority side is never discarded.
    assert keep[~correctness].all()
    # Majority quota is one per group, not concentrated in a few prompts.
    for group in "abcd":
        group_mask = torch.tensor([value == group for value in prompt_groups])
        assert int((keep & correctness & group_mask).sum()) == 1
    torch.testing.assert_close(
        keep,
        select_correctness_balanced_rollouts(
            correctness, prompt_groups, seed=123
        ),
    )


@pytest.mark.parametrize("correct", [False, True])
def test_single_class_batch_skips_balancing_and_margin(correct):
    result = compute_uni_opd_batch(
        student_log_probs=torch.zeros((4, 2)),
        teacher_log_probs=torch.tensor(
            [[0.1, 0.3], [0.2, 0.4], [0.5, 0.7], [0.6, 0.8]]
        ),
        response_mask=torch.ones((4, 2), dtype=torch.bool),
        verifier_rewards=torch.full((4,), float(correct)),
        prompt_group_ids=["a", "a", "b", "b"],
    )

    assert result.keep_mask.all()
    torch.testing.assert_close(result.calibrated_returns, result.raw_returns)
    assert result.metrics["uni-opd/train/mixed_group_count"] == 0.0
    assert result.metrics["uni-opd/train/shifted_group_count"] == 0.0


def test_mono_outcome_prompt_groups_skip_margin_in_a_mixed_batch():
    result = compute_uni_opd_batch(
        student_log_probs=torch.zeros((4, 1)),
        teacher_log_probs=torch.tensor([[0.1], [0.2], [0.7], [0.8]]),
        response_mask=torch.ones((4, 1), dtype=torch.bool),
        verifier_rewards=torch.tensor([1.0, 1.0, 0.0, 0.0]),
        prompt_group_ids=["all-correct", "all-correct", "all-wrong", "all-wrong"],
        margin_delta=0.4,
    )

    assert result.keep_mask.all()
    torch.testing.assert_close(result.calibrated_returns, result.raw_returns)
    assert result.metrics["uni-opd/train/mixed_group_count"] == 0.0
    assert result.metrics["uni-opd/train/shifted_group_count"] == 0.0


def test_margin_is_computed_after_global_subsampling():
    # Only one of the three correct trajectories survives balancing.  The
    # calibrated group mean therefore uses the retained row, not all rows.
    result = compute_uni_opd_batch(
        student_log_probs=torch.zeros((4, 1)),
        teacher_log_probs=torch.tensor([[0.0], [1.0], [2.0], [0.5]]),
        response_mask=torch.ones((4, 1), dtype=torch.bool),
        verifier_rewards=torch.tensor([1.0, 1.0, 1.0, 0.0]),
        prompt_group_ids=["a", "a", "a", "a"],
        margin_delta=0.4,
        seed=7,
    )

    assert int(result.keep_mask.sum()) == 2
    selected_correct = result.keep_mask & torch.tensor([True, True, True, False])
    selected_margin = (
        result.calibrated_returns[selected_correct].mean()
        - result.calibrated_returns[~torch.tensor([True, True, True, False])].mean()
    )
    assert float(selected_margin) >= 0.4 - 1.0e-6
    assert torch.equal(
        result.token_advantages[~result.keep_mask],
        torch.zeros_like(result.token_advantages[~result.keep_mask]),
    )


def test_uni_opd_rejects_nonbinary_rewards_and_empty_trajectories():
    common = dict(
        student_log_probs=torch.zeros((2, 2)),
        teacher_log_probs=torch.zeros((2, 2)),
        prompt_group_ids=["a", "a"],
    )
    with pytest.raises(ValueError, match="binary verifier rewards"):
        compute_uni_opd_batch(
            **common,
            response_mask=torch.ones((2, 2), dtype=torch.bool),
            verifier_rewards=torch.tensor([0.5, 0.0]),
        )
    with pytest.raises(ValueError, match="no sampled response tokens"):
        compute_uni_opd_batch(
            **common,
            response_mask=torch.tensor([[True, True], [False, False]]),
            verifier_rewards=torch.tensor([1.0, 0.0]),
        )


def test_uni_opd_loss_normalization_matches_deleting_dropped_tokens(monkeypatch):
    monkeypatch.setattr(losses, "no_padding_2_padding", lambda tensor, _data: tensor)
    log_probs = torch.tensor(
        [[-1.0, -2.0], [-3.0, -4.0]], requires_grad=True
    )
    data = TensorDict(
        {
            "old_log_probs": torch.zeros_like(log_probs),
            "response_mask": torch.ones_like(log_probs, dtype=torch.bool),
            # First trajectory is retained with tilde_G=0.4; second was
            # subsampled and therefore has zero algorithmic advantage.
            "uni_opd_advantages": torch.tensor([[0.4, 0.4], [0.0, 0.0]]),
        },
        batch_size=2,
    )
    tu.assign_non_tensor_data(
        data, "uni_opd_loss_normalization", 2.0
    )
    actor_config = SimpleNamespace(
        loss_agg_mode="token-mean",
        global_batch_info={
            "dp_size": 1,
            "batch_num_tokens": 4,
            "global_batch_size": 2,
        },
    )
    loss_config = SimpleNamespace(
        loss_mode="uni_opd",
        loss_max_clamp=None,
        use_policy_gradient=True,
        policy_loss_mode="reinforce",
        global_batch_info={},
    )

    loss, _ = losses.distillation_loss(
        actor_config,
        SimpleNamespace(distillation_loss=loss_config),
        {"log_probs": log_probs},
        data,
    )
    expected = -(torch.tensor(0.4) * log_probs[0].mean())
    torch.testing.assert_close(loss, expected)
    loss.backward()
    torch.testing.assert_close(
        log_probs.grad,
        torch.tensor([[-0.2, -0.2], [0.0, 0.0]]),
    )


def test_uni_opd_loss_mode_builds_as_sampled_token_estimator():
    config = DistillationLossConfig(
        loss_mode="uni_opd",
        loss_max_clamp=None,
        policy_loss_mode="reinforce",
    )

    assert config.loss_settings.use_estimator
    assert not config.loss_settings.use_topk


def test_uni_opd_grades_thinking_training_rollouts():
    config = OmegaConf.create(
        {
            "algorithm": {"name": "uni_opd"},
            "student_prompt": "qwen3_thinking_prompt",
            "teacher_prompt": "qwen3_thinking_prompt",
        }
    )
    assert verl_sync.should_track_opd_reward_metrics(config)


def test_six_uni_opd_configs_have_official_batch_lengths_and_persistence():
    paths = sorted(CONFIG_DIR.glob("*.yaml"))
    assert len(paths) == 6

    expected_pairs = {
        ("Qwen3-4B", "Qwen3-1.7B"),
        ("Qwen3-8B", "Qwen3-1.7B"),
        ("Qwen3-4B", "Qwen3-0.6B"),
    }
    observed_pairs = set()
    observed_modes = {"thinking": 0, "no_thinking": 0}
    for path in paths:
        config = _load_config(path)
        teacher = config.distillation.teacher_models.teacher_model
        teacher_name = Path(str(teacher.model_path)).name
        student_name = Path(str(config.actor_rollout_ref.model.path)).name
        observed_pairs.add((teacher_name, student_name))

        thinking = str(config.student_prompt) == "qwen3_thinking_prompt"
        mode = "thinking" if thinking else "no_thinking"
        observed_modes[mode] += 1
        train_tokens = 16384 if thinking else 4096
        val_tokens = 32768 if thinking else 16384
        train_context = 2048 + train_tokens + 1
        val_context = 2048 + val_tokens

        assert config.teacher_prompt == config.student_prompt
        assert config.rlvr_generation.train_max_new_tokens == train_tokens
        assert config.rlvr_generation.val_max_new_tokens == val_tokens
        assert config.data.max_response_length == val_tokens
        assert config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu == train_context
        assert config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu == train_context
        assert config.actor_rollout_ref.ref.log_prob_max_token_len_per_gpu == train_context
        assert config.actor_rollout_ref.rollout.max_model_len == val_context
        assert config.actor_rollout_ref.rollout.max_num_batched_tokens == val_context
        assert teacher.inference.response_length == train_tokens
        assert teacher.inference.max_model_len == train_context

        assert config.data.train_batch_size == 16
        assert config.actor_rollout_ref.actor.ppo_mini_batch_size == 16
        assert config.actor_rollout_ref.rollout.n == 16
        assert config.data.train_batch_size * config.actor_rollout_ref.rollout.n == 256
        assert config.actor_rollout_ref.rollout.val_kwargs.n == 16

        assert config.algorithm.name == "uni_opd"
        assert config.algorithm.uni_opd.target_correct_ratio == pytest.approx(0.5)
        assert config.algorithm.uni_opd.margin_scope == "group"
        assert config.algorithm.uni_opd.trajectory_reduce == "mean"
        assert config.algorithm.uni_opd.margin_direction == "spread"
        assert config.algorithm.uni_opd.margin_delta == pytest.approx(0.4)
        assert config.group_name == "UniOPD"
        assert config.actor_rollout_ref.actor.loss_agg_mode == "token-mean"
        assert config.distillation.distillation_loss.loss_mode == "uni_opd"
        assert config.distillation.distillation_loss.loss_max_clamp is None
        assert config.distillation.distillation_loss.policy_loss_mode == "reinforce"
        assert config.trainer.total_training_steps == 100
        assert config.trainer.test_freq == 20
        assert config.trainer.save_freq == 20
        validate_uni_opd_config(config)

        # The launch-time DP adjustment must preserve the requested 16 prompts
        # and corresponding 16-question actor mini-batch.
        assert configure_opd_data_parallel_batch(config) == 16
        assert config.data.train_batch_size == 16
        assert config.actor_rollout_ref.actor.ppo_mini_batch_size == 16

    assert observed_pairs == expected_pairs
    assert observed_modes == {"thinking": 3, "no_thinking": 3}
