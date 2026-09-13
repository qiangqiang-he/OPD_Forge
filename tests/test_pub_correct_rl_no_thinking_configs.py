"""Publication contracts for Correct-RL No-Thinking configurations."""

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs" / "PUB_Correct_RL_NoThinking"
CONFIG_NAMES = sorted(path.stem for path in CONFIG_DIR.glob("*.yaml"))


def _compose(config_name: str):
    from hydra import compose, initialize_config_dir

    search_path = (
        f"hydra.searchpath=[file://{PROJECT_ROOT / 'configs'},"
        "pkg://verl.trainer.config]"
    )
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name=config_name, overrides=[search_path])


def test_correct_rl_publication_has_five_configs():
    assert len(CONFIG_NAMES) == 5


@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_correct_rl_no_thinking_publication_contract(config_name):
    from omegaconf import OmegaConf

    from algorithms import resolve_algorithm
    from verl.trainer import main_ppo_sync as verl_sync

    config = _compose(config_name)
    OmegaConf.resolve(config)

    assert "advptgamma0p5_cliphigh0p27_lr5e-6_len10k_1000steps" in config_name
    assert str(config.run_name) == config_name
    assert str(config.run_name_prefix) == config_name
    assert str(config.group_name) == "PUB_Correct_RL_NoThinking"
    assert str(config.algorithm.name) == "correct_rl"
    assert str(config.algorithm.adv_estimator) == "grpo"
    assert float(config.algorithm.correct_rl_gamma) == pytest.approx(0.5)
    assert int(config.rlvr_generation.train_max_new_tokens) == 10240
    assert int(config.rlvr_generation.val_max_new_tokens) == 16384
    assert int(config.data.max_response_length) == 16384
    assert not bool(config.distillation.enabled)
    assert not bool(config.actor_rollout_ref.actor.use_kl_loss)
    assert not bool(config.algorithm.use_kl_in_reward)

    actor = config.actor_rollout_ref.actor
    rollout = config.actor_rollout_ref.rollout
    assert int(actor.ppo_max_token_len_per_gpu) == 12289
    assert str(actor.loss_agg_mode) == "seq-mean-token-mean"
    assert float(actor.clip_ratio) == pytest.approx(0.2)
    assert float(actor.clip_ratio_low) == pytest.approx(0.2)
    assert float(actor.clip_ratio_high) == pytest.approx(0.27)
    assert str(actor.policy_loss.loss_mode) == "vanilla"
    assert float(actor.optim.lr) == pytest.approx(5.0e-6)
    assert int(rollout.log_prob_max_token_len_per_gpu) == 12289
    assert int(rollout.max_model_len) == 18432
    assert int(rollout.max_num_batched_tokens) == 18432
    assert int(rollout.val_kwargs.n) == 16

    resolved = resolve_algorithm(config)
    assert resolved.name == "correct_rl"
    resolved.validate(config)
    assert not verl_sync.need_reference_policy(config)
    assert not verl_sync.need_teacher_policy(config)
    assert not verl_sync.need_critic(config)
    assert int(config.trainer.total_training_steps) == 1000
    assert int(config.trainer.save_freq) == 100
