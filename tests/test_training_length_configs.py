"""Regression checks for production OPD sequence-length budgets."""

from pathlib import Path

from omegaconf import OmegaConf

from utils.opd_runtime import validate_opd_runtime_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOSS_DEFAULTS = OmegaConf.load(
    PROJECT_ROOT
    / "verl"
    / "verl"
    / "trainer"
    / "config"
    / "distillation"
    / "distillation.yaml"
).distillation_loss
CONFIG_DIRS = (
    PROJECT_ROOT / "configs" / "p1_pg_opd",
    PROJECT_ROOT / "configs" / "p2_gkd_opd",
)


def test_all_production_configs_use_the_complete_length_budget():
    base = OmegaConf.load(PROJECT_ROOT / "configs" / "base.yaml")
    paths = sorted(path for directory in CONFIG_DIRS for path in directory.glob("*.yaml"))
    assert len(paths) == 12

    for path in paths:
        config = OmegaConf.merge(base, OmegaConf.load(path))
        config.distillation.distillation_loss = OmegaConf.merge(
            LOSS_DEFAULTS, config.distillation.distillation_loss
        )
        rollout = config.actor_rollout_ref.rollout
        teacher = config.distillation.teacher_models.teacher_model.inference

        assert config.trainer.test_freq == 20, path
        assert config.data.max_prompt_length == 2048, path
        assert config.data.max_response_length == 16384, path
        assert config.rlvr_generation.train_max_new_tokens == 8192, path
        assert config.rlvr_generation.val_max_new_tokens == 16384, path

        assert rollout.max_model_len == 18432, path
        assert rollout.max_num_batched_tokens == 18432, path
        assert config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu == 10241, path
        assert rollout.log_prob_max_token_len_per_gpu == 10241, path
        assert config.actor_rollout_ref.ref.log_prob_max_token_len_per_gpu == 10241, path

        assert teacher.prompt_length == 2048, path
        assert teacher.response_length == 8192, path
        assert teacher.max_model_len == 10241, path
        validate_opd_runtime_config(config)
