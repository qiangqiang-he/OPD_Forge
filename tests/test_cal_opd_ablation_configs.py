"""Contracts for the four publication Cal-OPD prompt ablations."""

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs" / "PUB_Cal_OPD_Thinking_Ablations"
INTERVENTIONS = ("inst", "answer", "solution")


def _compose(intervention_type: str):
    from hydra import compose, initialize_config_dir

    config_name = (
        f"pub_cal_opd_{intervention_type}_qwen3_4b_thinking_2507_to_1p7b_"
        "thinking_len16k_100steps"
    )
    search_path = (
        f"hydra.searchpath=[file://{PROJECT_ROOT / 'configs'},"
        "pkg://verl.trainer.config]"
    )
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return config_name, compose(config_name=config_name, overrides=[search_path])


@pytest.mark.parametrize("intervention_type", INTERVENTIONS)
def test_cal_opd_ablation_config_contract(intervention_type):
    from algorithms import resolve_algorithm
    from utils.opd_runtime import configure_opd_defaults

    config_name, config = _compose(intervention_type)
    assert str(config.run_name) == config_name
    assert str(config.run_name_prefix) == config_name
    assert str(config.group_name) == "PUB_Cal_OPD_Thinking"
    assert str(config.cal_intervention_type) == intervention_type
    assert str(config.algorithm.name) == "cal_opd"
    assert str(config.student_prompt) == "qwen3_thinking_prompt"
    assert str(config.teacher_prompt) == "qwen3_thinking_prompt"
    assert str(config.actor_rollout_ref.model.path) == "./models/Qwen3-1.7B"
    assert (
        str(config.distillation.teacher_models.teacher_model.model_path)
        == "./models/Qwen3-4B-Thinking-2507"
    )
    assert list(config.data.train_files) == [
        "./data/DAPO-17k-English-Qwen3-4B-Instruct-2507-Correct.json"
    ]
    assert list(config.data.val_files) == ["./data/AMC-2023.json"]
    assert int(config.data.max_prompt_length) == 8192
    assert int(config.data.max_response_length) == 16384
    assert int(config.rlvr_generation.train_max_new_tokens) == 16384
    assert int(config.rlvr_generation.val_max_new_tokens) == 16384
    assert int(config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu) == 24577
    assert int(config.actor_rollout_ref.rollout.max_model_len) == 24576
    assert (
        int(config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu)
        == 24577
    )
    assert int(config.actor_rollout_ref.ref.log_prob_max_token_len_per_gpu) == 24577
    teacher = config.distillation.teacher_models.teacher_model.inference
    assert int(teacher.prompt_length) == 8192
    assert int(teacher.response_length) == 16384
    assert int(teacher.max_model_len) == 24577
    loss = config.distillation.distillation_loss
    assert str(loss.loss_mode) == "cal_reverse_kl"
    assert float(loss.cal_lambda) == 5.0
    assert str(loss.policy_loss_mode) == "reinforce"
    assert int(config.trainer.n_gpus_per_node) == 4
    assert int(config.distillation.n_gpus_per_node) == 4
    assert int(config.trainer.total_training_steps) == 100
    assert int(config.trainer.test_freq) == 20
    assert int(config.trainer.save_freq) == 20

    configure_opd_defaults(config)
    resolve_algorithm(config).validate(config)
