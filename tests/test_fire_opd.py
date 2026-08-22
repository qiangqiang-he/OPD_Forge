"""CPU-only regression tests for FiRe-OPD and its six production configs."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from algorithms.fire_opd import compute_fire_opd_batch, validate_fire_opd_config
from verl.trainer.distillation import losses
from verl.utils import tensordict_utils as tu
from verl.utils.config import omega_conf_to_dataclass
from verl.workers.config import (
    DistillationLossConfig,
    DistillationTeacherModelConfig,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs" / "FiReOPD"
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


def test_fire_opd_filters_lowest_teacher_score_and_matches_weight_formula():
    old_log_probs = torch.zeros((5, 2), requires_grad=True)
    teacher_log_probs = torch.tensor(
        [[-5.0, -5.0], [-4.0, -4.0], [-3.0, -3.0], [-2.0, -2.0], [-1.0, -1.0]],
        requires_grad=True,
    )
    teacher_entropy = torch.tensor(
        [[99.0, 99.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]]
    )
    student_entropy = torch.tensor(
        [[99.0, 99.0], [8.0, 4.0], [6.0, 3.0], [4.0, 2.0], [2.0, 1.0]]
    )
    response_mask = torch.ones((5, 2), dtype=torch.bool)

    result = compute_fire_opd_batch(
        old_log_probs,
        teacher_log_probs,
        teacher_entropy,
        student_entropy,
        response_mask,
        filter_ratio=0.2,
        alpha=1.0,
        beta=1.0,
    )

    torch.testing.assert_close(
        result.trajectory_scores, torch.tensor([-5.0, -4.0, -3.0, -2.0, -1.0])
    )
    torch.testing.assert_close(
        result.keep_mask, torch.tensor([False, True, True, True, True])
    )
    teacher_confidence = 1.0 - teacher_entropy[1:] / 8.0
    student_confusion = student_entropy[1:] / 8.0
    raw_weights = (1.0 + teacher_confidence) * (1.0 + student_confusion)
    expected_weights = raw_weights / raw_weights.mean(dim=-1, keepdim=True)
    torch.testing.assert_close(result.normalized_weights[1:], expected_weights)
    torch.testing.assert_close(
        result.advantages[1:], expected_weights * teacher_log_probs.detach()[1:]
    )
    assert torch.equal(result.advantages[0], torch.zeros(2))
    torch.testing.assert_close(
        result.normalized_weights[1:].mean(dim=-1), torch.ones(4)
    )
    assert result.metrics["fire-opd/train/filtered_trajectory_count"] == 1.0
    assert result.metrics["fire-opd/train/token_weight_mean"] == pytest.approx(1.0)
    assert not result.advantages.requires_grad


def test_fire_opd_ignores_padding_trajectories_in_filter_and_entropy_maxima():
    result = compute_fire_opd_batch(
        old_log_probs=torch.zeros((6, 1)),
        teacher_log_probs=torch.tensor([[-5.0], [-4.0], [-3.0], [-2.0], [-1.0], [-100.0]]),
        teacher_entropy=torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [999.0]]),
        student_entropy=torch.tensor([[5.0], [4.0], [3.0], [2.0], [1.0], [999.0]]),
        response_mask=torch.ones((6, 1), dtype=torch.bool),
        genuine_trajectory_mask=torch.tensor([True, True, True, True, True, False]),
        filter_ratio=0.2,
    )

    torch.testing.assert_close(
        result.keep_mask,
        torch.tensor([False, True, True, True, True, False]),
    )
    assert result.metrics["fire-opd/train/teacher_entropy_max"] == 5.0
    assert result.metrics["fire-opd/train/student_entropy_max"] == 4.0


def test_fire_opd_loss_normalization_matches_deleting_filtered_trajectory(
    monkeypatch,
):
    monkeypatch.setattr(losses, "no_padding_2_padding", lambda tensor, _data: tensor)
    log_probs = torch.tensor(
        [[-1.0, -2.0], [-3.0, -4.0]], requires_grad=True
    )
    data = TensorDict(
        {
            "old_log_probs": torch.zeros_like(log_probs),
            "response_mask": torch.ones_like(log_probs, dtype=torch.bool),
            "fire_opd_advantages": torch.tensor([[0.4, 0.4], [0.0, 0.0]]),
        },
        batch_size=2,
    )
    tu.assign_non_tensor_data(data, "fire_opd_loss_normalization", 2.0)
    actor_config = SimpleNamespace(
        loss_agg_mode="seq-mean-token-mean",
        global_batch_info={
            "dp_size": 1,
            "batch_num_tokens": 4,
            "global_batch_size": 2,
        },
    )
    loss_config = SimpleNamespace(
        loss_mode="fire_opd",
        loss_max_clamp=20.0,
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


def test_fire_opd_loss_mode_requests_sampled_tokens_and_exact_teacher_entropy():
    config = DistillationLossConfig(
        loss_mode="fire_opd",
        topk=16,
        policy_loss_mode="reinforce",
    )

    assert config.loss_settings.use_estimator
    assert not config.loss_settings.use_topk
    assert config.loss_settings.use_full_vocab_teacher_entropy


def test_fire_opd_distillation_config_enables_bounded_teacher_entropy_channel():
    config = _load_config(
        CONFIG_DIR / "fire_opd_qwen3_4b_to_1p7b_thinking_100steps.yaml"
    )
    teacher_config = config.distillation.teacher_models.teacher_model
    teacher_config.inference.max_num_batched_tokens = 128
    teacher = omega_conf_to_dataclass(
        teacher_config,
        dataclass_type=DistillationTeacherModelConfig,
    )
    teacher.validate_and_prepare_for_distillation(
        use_topk=False,
        topk=16,
        use_full_vocab_teacher_entropy=True,
    )
    assert teacher.inference.engine_kwargs["vllm"]["max_logprobs"] == 17
    assert teacher.inference.engine_kwargs["vllm"]["full_vocab_entropy_topk"] == 16


def test_six_fire_opd_configs_have_models_lengths_defaults_and_persistence():
    paths = sorted(CONFIG_DIR.glob("fire_opd_*.yaml"))
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
        observed_modes["thinking" if thinking else "no_thinking"] += 1
        train_tokens = 8192 if thinking else 4096
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

        assert config.algorithm.name == "fire_opd"
        assert config.algorithm.fire_opd.filter_ratio == pytest.approx(0.2)
        assert config.algorithm.fire_opd.alpha == pytest.approx(1.0)
        assert config.algorithm.fire_opd.beta == pytest.approx(1.0)
        assert config.group_name == "FiReOPD"
        loss = config.distillation.distillation_loss
        assert loss.loss_mode == "fire_opd"
        assert loss.topk == 16
        assert loss.policy_loss_mode == "reinforce"
        vllm_kwargs = teacher.inference.engine_kwargs.vllm
        assert vllm_kwargs.max_logprobs == 17
        assert vllm_kwargs.full_vocab_entropy_topk == 16
        assert config.trainer.total_training_steps == 100
        assert config.trainer.test_freq == 20
        assert config.trainer.save_freq == 20
        validate_fire_opd_config(config)

    assert observed_pairs == expected_pairs
    assert observed_modes == {"thinking": 3, "no_thinking": 3}
