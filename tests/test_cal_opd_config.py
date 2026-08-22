"""Contract checks for the formal Cal-OPD training configuration."""

from pathlib import Path

from omegaconf import OmegaConf
import pytest


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "p3_cal_opd"
    / "cal_opd_math33k_100steps.yaml"
)
LOSS_DEFAULTS = OmegaConf.load(
    CONFIG.parents[2]
    / "verl"
    / "verl"
    / "trainer"
    / "config"
    / "distillation"
    / "distillation.yaml"
).distillation_loss


def test_cal_opd_config_contract():
    config = OmegaConf.load(CONFIG)
    loss = config.distillation.distillation_loss
    effective_loss = OmegaConf.merge(LOSS_DEFAULTS, loss)

    assert config.algorithm.name == "cal_opd"
    assert config.run_name == "cal_opd_math33k_100steps"
    assert config.group_name == "cal_opd"
    assert loss.loss_mode == "cal_reverse_kl"
    assert "topk" not in loss
    assert "use_policy_gradient" not in loss
    assert effective_loss.topk is None
    assert effective_loss.use_policy_gradient is True
    assert loss.policy_loss_mode == "reinforce"
    assert config.trainer.total_training_steps == 100
    assert config.trainer.test_freq == 20
    assert config.rlvr_generation.train_max_new_tokens == 8192
    assert config.rlvr_generation.val_max_new_tokens == 16384
    assert list(config.data.train_files) == ["./data/nvidia_math_33k.json"]
    assert list(config.data.val_files) == [
        "./data/AMC-2023.json",
        "./data/AIME-2024.json",
        "./data/AIME-2025.json",
    ]


@pytest.mark.parametrize(
    "config_name",
    [
        "cal_opd_qwen3_1p7b_to_0p6b_thinking_200steps.yaml",
        "cal_opd_qwen3_4b_to_1p7b_no_thinking_200steps.yaml",
        "cal_opd_qwen3_4b_to_1p7b_thinking_200steps.yaml",
    ],
)
def test_p5_cal_opd_configs_set_default_lambda(config_name):
    config = OmegaConf.load(CONFIG.parents[1] / "p5" / config_name)

    assert config.distillation.distillation_loss.cal_lambda == 1.0
