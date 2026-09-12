"""Publication contracts for Correct-OPD No-Thinking configurations."""

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs" / "PUB_Correct_OPD_NoThinking"
CONFIG_NAMES = sorted(path.stem for path in CONFIG_DIR.glob("*.yaml"))


def test_correct_opd_publication_has_five_configs():
    assert len(CONFIG_NAMES) == 5


def _compose(config_name: str):
    from hydra import compose, initialize_config_dir

    search_path = (
        f"hydra.searchpath=[file://{PROJECT_ROOT / 'configs'},"
        "pkg://verl.trainer.config]"
    )
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name=config_name, overrides=[search_path])


@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_correct_opd_no_thinking_publication_config_contract(config_name):
    from omegaconf import OmegaConf

    config = _compose(config_name)
    OmegaConf.resolve(config)

    assert str(config.run_name) == config_name
    assert str(config.run_name_prefix) == config_name
    assert str(config.group_name) == "PUB_Correct_OPD_NoThinking"
    assert str(config.trainer.experiment_name) == config_name
    assert str(config.student_prompt) == "qwen3_no_thinking_prompt"
    assert str(config.teacher_prompt) == "qwen3_no_thinking_prompt"
    assert str(config.algorithm.name) == "correct_opd"
    assert int(config.rlvr_generation.train_max_new_tokens) == 10240
    assert int(config.rlvr_generation.val_max_new_tokens) == 16384
    assert int(config.data.max_response_length) == 16384

    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    assert int(actor.ppo_max_token_len_per_gpu) == 12289
    assert int(rollout.log_prob_max_token_len_per_gpu) == 12289
    assert int(config.actor_rollout_ref.ref.log_prob_max_token_len_per_gpu) == 12289
    assert int(rollout.max_model_len) == 18432

    teacher = config.distillation.teacher_models.teacher_model.inference
    assert int(teacher.prompt_length) == 2048
    assert int(teacher.response_length) == 10240
    assert int(teacher.max_model_len) == 12289

    loss = config.distillation.distillation_loss
    assert str(loss.loss_mode) == "correct_reverse_kl"
    assert bool(loss.use_policy_gradient)
    assert not bool(loss.use_task_rewards)
    assert str(loss.policy_loss_mode) == "reinforce"
    assert float(loss.selection_ratio) == 1.0
    assert str(loss.selection_method) == "random"
