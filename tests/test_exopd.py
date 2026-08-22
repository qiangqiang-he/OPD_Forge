"""CPU-only regression tests for the ExOPD objective and configurations."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from algorithms.exopd import validate_exopd_config
from verl.trainer.distillation import losses
from verl.trainer.ppo.utils import need_reference_policy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs" / "ExOPD"
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
    # These values normally come from Hydra's /ppo_trainer component defaults.
    OmegaConf.update(config, "actor_rollout_ref.ref.strategy", "fsdp", force_add=True)
    OmegaConf.update(
        config,
        "actor_rollout_ref.ref.fsdp_config.forward_only",
        True,
        force_add=True,
    )
    return config


def test_exopd_advantage_matches_formula_and_is_stop_gradient():
    student = torch.tensor([[-2.0, -0.5]], requires_grad=True)
    teacher = torch.tensor([[-1.0, -1.5]], requires_grad=True)
    reference = torch.tensor([[-1.8, -1.0]], requires_grad=True)

    advantage = losses.compute_exopd_advantage(
        student, teacher, reference, exopd_lambda=1.25
    )

    expected = 1.25 * (teacher.detach() - reference.detach()) - (
        student.detach() - reference.detach()
    )
    torch.testing.assert_close(advantage, expected)
    assert not advantage.requires_grad

    loss = -(advantage * student).mean()
    loss.backward()
    torch.testing.assert_close(student.grad, -advantage / advantage.numel())
    assert teacher.grad is None
    assert reference.grad is None


def test_exopd_loss_returns_negative_advantage_for_shared_pg_path(monkeypatch):
    monkeypatch.setattr(losses, "no_padding_2_padding", lambda tensor, _data: tensor)
    student = torch.tensor([[-2.0, -0.5]], requires_grad=True)
    teacher = torch.tensor([[[-1.0], [-1.5]]])
    reference = torch.tensor([[-1.8, -1.0]])
    data = {
        "teacher_logprobs": teacher,
        "ref_log_prob": reference,
        "response_mask": torch.ones_like(student, dtype=torch.bool),
    }
    config = SimpleNamespace(distillation_loss=SimpleNamespace(exopd_lambda=1.25))

    per_token_loss, metrics = losses.compute_exopd_sampled_token_loss(
        None, config, {"log_probs": student}, data
    )

    expected_advantage = 1.25 * (teacher.squeeze(-1) - reference) - (
        student.detach() - reference
    )
    torch.testing.assert_close(per_token_loss, -expected_advantage)
    assert not per_token_loss.requires_grad
    assert "distillation/exopd_reference_sampled_token_prob" in metrics


def test_six_exopd_configs_have_required_models_lengths_and_persistence():
    paths = sorted(CONFIG_DIR.glob("exopd_*.yaml"))
    assert len(paths) == 6

    expected_pairs = {
        ("Qwen3-4B", "Qwen3-1.7B"),
        ("Qwen3-8B", "Qwen3-1.7B"),
        ("Qwen3-4B", "Qwen3-0.6B"),
    }
    observed = set()
    observed_modes = {"thinking": 0, "no_thinking": 0}
    for path in paths:
        config = _load_config(path)
        teacher_name = Path(
            str(config.distillation.teacher_models.teacher_model.model_path)
        ).name
        student_name = Path(str(config.actor_rollout_ref.model.path)).name
        observed.add((teacher_name, student_name))

        thinking = str(config.student_prompt) == "qwen3_thinking_prompt"
        mode = "thinking" if thinking else "no_thinking"
        observed_modes[mode] += 1
        assert config.teacher_prompt == config.student_prompt
        assert config.rlvr_generation.train_max_new_tokens == (
            8192 if thinking else 4096
        )
        assert config.rlvr_generation.val_max_new_tokens == (
            32768 if thinking else 16384
        )
        assert config.data.max_response_length == (32768 if thinking else 16384)

        assert config.algorithm.name == "exopd"
        assert config.group_name == "ExOPD"
        assert config.distillation.distillation_loss.exopd_lambda == pytest.approx(1.25)
        assert config.distillation.distillation_loss.loss_max_clamp is None
        assert config.actor_rollout_ref.actor.loss_agg_mode == "token-mean"
        assert config.trainer.total_training_steps == 100
        assert config.trainer.test_freq == 20
        assert config.trainer.save_freq == 20
        assert config.actor_rollout_ref.actor.fsdp_config.fsdp_size == 1
        assert config.actor_rollout_ref.ref.fsdp_config.fsdp_size == 1
        assert not config.actor_rollout_ref.actor.fsdp_config.param_offload
        assert not config.actor_rollout_ref.ref.fsdp_config.param_offload
        assert config.actor_rollout_ref.ref.fsdp_config.forward_only_keep_on_device
        assert need_reference_policy(config)
        validate_exopd_config(config)

    assert observed == expected_pairs
    assert observed_modes == {"thinking": 3, "no_thinking": 3}


def test_exopd_rejects_per_sequence_length_normalization():
    config = _load_config(
        CONFIG_DIR / "exopd_qwen3_4b_to_1p7b_no_thinking_100steps.yaml"
    )
    config.actor_rollout_ref.actor.loss_agg_mode = "seq-mean-token-mean"

    with pytest.raises(ValueError, match="loss_agg_mode=token-mean"):
        validate_exopd_config(config)
