"""TopGap GKD-OPD with absolute-gap selection and signed optimization."""

from __future__ import annotations

from utils.opd_runtime import BaseOPDTrainer, validate_opd_runtime_config


TOPGAP_GKD_OPD_VARIANT = "topgap_gkd_opd"


def validate_topgap_gkd_opd_config(config) -> None:
    """Validate the complete TopGap GKD-OPD contract."""

    validate_opd_runtime_config(config)
    if str(config.algorithm.name) != TOPGAP_GKD_OPD_VARIANT:
        raise ValueError(
            f"TopGap GKD-OPD requires algorithm.name={TOPGAP_GKD_OPD_VARIANT}."
        )
    loss = config.distillation.distillation_loss
    if str(loss.loss_mode) != "topgap_reverse_kl":
        raise ValueError("TopGap GKD-OPD requires loss_mode=topgap_reverse_kl.")
    if str(loss.policy_loss_mode) != "vanilla":
        raise ValueError("TopGap GKD-OPD requires policy_loss_mode=vanilla.")
    _validate_topgap_settings(loss)


def _validate_topgap_settings(loss) -> None:
    ratio = float(loss.topgap_token_ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"TopGap GKD-OPD token ratio must lie in [0, 1], got {ratio}.")
    selection = str(loss.topgap_selection)
    if selection not in {"top", "bottom"}:
        raise ValueError(
            f"TopGap GKD-OPD selection must be 'top' or 'bottom', got {selection!r}."
        )


class TopGapGKDOPDTrainer(BaseOPDTrainer):
    """Trainer ranking by absolute gap while retaining signed GKD signals."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        validate_topgap_gkd_opd_config(config)
        super().__init__(*args, **kwargs)

    def algorithm_metric_aliases(self) -> dict[str, str]:
        return {
            "actor/distillation/reverse_kl_estimate": (
                "topgap-gkd-opd/train/reverse_kl_estimate"
            ),
            "actor/distillation/topgap_selected_token_ratio": (
                "topgap-gkd-opd/train/selected_token_ratio"
            ),
            "actor/distillation/topgap_gap_mean": "topgap-gkd-opd/train/gap_mean",
            "actor/distillation/topgap_selected_gap_mean": (
                "topgap-gkd-opd/train/selected_gap_mean"
            ),
            "actor/distillation/topgap_gradient_signal_relative_change": (
                "topgap-gkd-opd/train/gradient_signal_relative_change"
            ),
            "actor/distillation/loss": "topgap-gkd-opd/train/policy_loss",
        }
