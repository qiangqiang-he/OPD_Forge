"""Contract checks for the formal Cal-OPD training configuration."""

from pathlib import Path

from omegaconf import OmegaConf


CONFIG = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "p3_cal_opd"
    / "cal_opd_math33k_100steps.yaml"
)


def test_cal_opd_config_contract():
    config = OmegaConf.load(CONFIG)
    loss = config.distillation.distillation_loss

    assert config.algorithm.name == "cal_opd"
    assert config.run_name == "cal_opd_math33k_100steps"
    assert config.group_name == "cal_opd"
    assert loss.loss_mode == "cal_reverse_kl"
    assert loss.topk is None
    assert loss.use_policy_gradient is True
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
