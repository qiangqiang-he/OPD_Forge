"""Regression tests for the six P2 GKD-OPD training configurations."""

from pathlib import Path

from omegaconf import OmegaConf


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "p2_gkd_opd"
EXPECTED_CONFIGS = {
    "gkd_opd_math33k_100steps.yaml": ("gkd_opd", "reverse_kl", None),
    "random_gkd_opd_math33k_ratio0p05_100steps.yaml": (
        "random_gkd_opd",
        "random_reverse_kl",
        0.05,
    ),
    "random_gkd_opd_math33k_ratio0p10_100steps.yaml": (
        "random_gkd_opd",
        "random_reverse_kl",
        0.10,
    ),
    "random_gkd_opd_math33k_ratio0p20_100steps.yaml": (
        "random_gkd_opd",
        "random_reverse_kl",
        0.20,
    ),
    "random_gkd_opd_math33k_ratio0p40_100steps.yaml": (
        "random_gkd_opd",
        "random_reverse_kl",
        0.40,
    ),
    "topgap_gkd_opd_math33k_ratio0p20_100steps.yaml": (
        "topgap_gkd_opd",
        "topgap_reverse_kl",
        0.20,
    ),
}


def test_p2_gkd_opd_config_contracts():
    paths = sorted(CONFIG_DIR.glob("*.yaml"))
    assert {path.name for path in paths} == set(EXPECTED_CONFIGS)

    run_names = set()
    for path in paths:
        algorithm_name, loss_mode, ratio = EXPECTED_CONFIGS[path.name]
        config = OmegaConf.load(path)
        loss = config.distillation.distillation_loss

        assert config.algorithm.name == algorithm_name
        assert loss.loss_mode == loss_mode
        assert loss.policy_loss_mode == "vanilla"
        assert config.group_name == "GKD_OPD"
        assert config.trainer.total_training_steps == 100
        assert list(config.data.train_files) == ["./data/nvidia_math_33k.json"]
        assert list(config.data.val_files) == [
            "./data/AMC-2023.json",
            "./data/AIME-2024.json",
            "./data/AIME-2025.json",
        ]
        assert config.run_name
        assert config.run_name not in run_names
        run_names.add(config.run_name)

        if algorithm_name == "random_gkd_opd":
            assert loss.random_token_ratio == ratio
        elif algorithm_name == "topgap_gkd_opd":
            assert loss.topgap_token_ratio == ratio
            assert loss.topgap_selection == "top"
