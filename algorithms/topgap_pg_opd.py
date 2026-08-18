"""TopGap PG-OPD with absolute-gap selection and signed optimization.

Tokens are ranked by ``abs(logT - logS)``.  Selected tokens retain the signed
REINFORCE advantage ``logT - logS``; the absolute value is never optimized.
"""

from __future__ import annotations

from utils.opd_runtime import BaseOPDTrainer, validate_opd_runtime_config


TOPGAP_PG_OPD_VARIANT = "topgap_pg_opd"


def validate_topgap_pg_opd_config(config) -> None:
    """Validate the complete TopGap PG-OPD contract."""

    validate_opd_runtime_config(config)
    if str(config.algorithm.name) != TOPGAP_PG_OPD_VARIANT:
        raise ValueError(
            f"TopGap PG-OPD requires algorithm.name={TOPGAP_PG_OPD_VARIANT}."
        )
    loss = config.distillation.distillation_loss
    if str(loss.loss_mode) != "topgap_reverse_kl":
        raise ValueError("TopGap PG-OPD requires loss_mode=topgap_reverse_kl.")
    if str(loss.policy_loss_mode) != "reinforce":
        raise ValueError("TopGap PG-OPD requires policy_loss_mode=reinforce.")
    _validate_topgap_settings(loss)


def _validate_topgap_settings(loss) -> None:
    ratio = float(loss.topgap_token_ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"TopGap PG-OPD token ratio must lie in [0, 1], got {ratio}.")
    selection = str(loss.topgap_selection)
    if selection not in {"top", "bottom"}:
        raise ValueError(
            f"TopGap PG-OPD selection must be 'top' or 'bottom', got {selection!r}."
        )


class TopGapPGOPDTrainer(BaseOPDTrainer):
    """Trainer ranking by absolute gap while retaining signed PG advantages."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        validate_topgap_pg_opd_config(config)
        super().__init__(*args, **kwargs)

    def algorithm_metric_aliases(self) -> dict[str, str]:
        return {
            "actor/distillation/reverse_kl_estimate": (
                "topgap-pg-opd/train/reverse_kl_estimate"
            ),
            "actor/distillation/topgap_selected_token_ratio": (
                "topgap-pg-opd/train/selected_token_ratio"
            ),
            "actor/distillation/topgap_gap_mean": "topgap-pg-opd/train/gap_mean",
            "actor/distillation/topgap_selected_gap_mean": (
                "topgap-pg-opd/train/selected_gap_mean"
            ),
            "actor/distillation/topgap_gradient_signal_relative_change": (
                "topgap-pg-opd/train/gradient_signal_relative_change"
            ),
            "actor/distillation/loss": "topgap-pg-opd/train/policy_loss",
        }
