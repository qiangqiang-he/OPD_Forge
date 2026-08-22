"""GKD-OPD with configurable reverse-KL token selection."""

from __future__ import annotations

from utils.opd_runtime import (
    BaseOPDTrainer,
    validate_opd_runtime_config,
    validate_token_selection_config,
)


GKD_OPD_VARIANT = "gkd_opd"


def validate_gkd_opd_config(config) -> None:
    """Validate the complete GKD-OPD contract."""

    validate_opd_runtime_config(config)
    if str(config.algorithm.name) != GKD_OPD_VARIANT:
        raise ValueError(f"GKD-OPD requires algorithm.name={GKD_OPD_VARIANT}.")
    loss = config.distillation.distillation_loss
    if str(loss.loss_mode) != "reverse_kl":
        raise ValueError("GKD-OPD requires loss_mode=reverse_kl.")
    if str(loss.policy_loss_mode) != "vanilla":
        raise ValueError("GKD-OPD requires policy_loss_mode=vanilla.")
    validate_token_selection_config(loss)


class GKDOPDTrainer(BaseOPDTrainer):
    """Trainer implementing GKD-OPD on all or selected response tokens."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        validate_gkd_opd_config(config)
        super().__init__(*args, **kwargs)

    def algorithm_metric_aliases(self) -> dict[str, str]:
        return {
            "actor/distillation/reverse_kl_estimate": (
                "gkd-opd/train/reverse_kl_estimate"
            ),
            "actor/distillation/selected_token_ratio": (
                "gkd-opd/train/selected_token_ratio"
            ),
            "actor/distillation/selection_gap_mean": "gkd-opd/train/gap_mean",
            "actor/distillation/selected_gap_mean": (
                "gkd-opd/train/selected_gap_mean"
            ),
            "actor/distillation/selection_gradient_signal_relative_change": (
                "gkd-opd/train/gradient_signal_relative_change"
            ),
            "actor/distillation/loss": "gkd-opd/train/policy_loss",
        }
