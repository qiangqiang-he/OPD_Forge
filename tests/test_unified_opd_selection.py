"""Contracts for unified GKD/PG token selection and reverse-KL-only OPD."""

from pathlib import Path

from omegaconf import OmegaConf

from utils.opd_runtime import SUPPORTED_LOSS_MODES, validate_token_selection_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_opd_runtime_supports_reverse_kl_modes_only():
    assert SUPPORTED_LOSS_MODES == {
        "reverse_kl",
        "cal_reverse_kl",
        "eopd",
        "exopd_reverse_kl",
        "ps_reverse_kl",
        "uni_opd",
        "fire_opd",
    }


def test_selection_contract_accepts_all_three_methods():
    for method in ("random", "topgap", "bottomgap"):
        validate_token_selection_config(
            OmegaConf.create({"selection_ratio": 0.5, "selection_method": method})
        )


def test_full_selection_ignores_method():
    validate_token_selection_config(
        OmegaConf.create({"selection_ratio": 1.0, "selection_method": "unused"})
    )


def test_requested_100step_configs_use_unified_selection_schema():
    paths = (
        PROJECT_ROOT / "configs" / "p2_gkd_opd" / "gkd_opd_math33k_100steps.yaml",
        PROJECT_ROOT
        / "configs"
        / "p4"
        / "pg_opd_qwen3_8b_to_1p7b_base_no_thinking_100steps.yaml",
    )
    expected_algorithms = ("gkd_opd", "pg_opd")

    for path, algorithm in zip(paths, expected_algorithms, strict=True):
        config = OmegaConf.load(path)
        loss = config.distillation.distillation_loss
        assert config.algorithm.name == algorithm
        assert config.trainer.total_training_steps == 100
        assert loss.selection_ratio == 1.0
        assert loss.selection_method == "random"
        assert "loss_mode" not in loss
        assert "topk" not in loss
