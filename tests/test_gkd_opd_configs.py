"""Regression tests for the six P2 GKD-OPD training configurations."""

from pathlib import Path

from omegaconf import OmegaConf


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "p2_gkd_opd"
EXPECTED_CONFIGS = {
    "gkd_opd_math33k_100steps.yaml": (1.0, "random"),
    "random_gkd_opd_math33k_ratio0p05_100steps.yaml": (0.05, "random"),
    "random_gkd_opd_math33k_ratio0p10_100steps.yaml": (0.10, "random"),
    "random_gkd_opd_math33k_ratio0p20_100steps.yaml": (0.20, "random"),
    "random_gkd_opd_math33k_ratio0p40_100steps.yaml": (0.40, "random"),
    "topgap_gkd_opd_math33k_ratio0p20_100steps.yaml": (0.20, "topgap"),
}


def test_p2_gkd_opd_config_contracts():
    paths = sorted(CONFIG_DIR.glob("*.yaml"))
    assert {path.name for path in paths} == set(EXPECTED_CONFIGS)

    run_names = set()
    for path in paths:
        ratio, method = EXPECTED_CONFIGS[path.name]
        config = OmegaConf.load(path)
        loss = config.distillation.distillation_loss

        assert config.algorithm.name == "gkd_opd"
        assert loss.selection_ratio == ratio
        assert loss.selection_method == method
        assert "loss_mode" not in loss
        assert "topk" not in loss
        assert "policy_loss_mode" not in loss
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

        assert "random_token_ratio" not in loss
        assert "topgap_token_ratio" not in loss
        assert "topgap_selection" not in loss
