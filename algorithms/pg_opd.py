"""Signed REINFORCE on-policy distillation with configurable token selection.

For student-sampled actions, PG-OPD uses

    A_opd = log pi_teacher(a|s) - log pi_student(a|s)
    L_pg-opd = -E[stop_gradient(A_opd) * log pi_student(a|s)].

The ``reverse_kl`` kernel returns ``logS - logT``.  The shared VERL
distillation path negates and detaches it before applying REINFORCE.
"""

from __future__ import annotations

from utils.opd_runtime import (
    BaseOPDTrainer,
    validate_opd_runtime_config,
    validate_token_selection_config,
)


PG_OPD_VARIANT = "pg_opd"


def validate_pg_opd_config(config) -> None:
    """Validate the complete signed PG-OPD contract."""

    validate_opd_runtime_config(config)
    if str(config.algorithm.name) != PG_OPD_VARIANT:
        raise ValueError(f"PG-OPD requires algorithm.name={PG_OPD_VARIANT}.")
    loss = config.distillation.distillation_loss
    if str(loss.loss_mode) != "reverse_kl":
        raise ValueError("PG-OPD requires loss_mode=reverse_kl.")
    if str(loss.policy_loss_mode) != "reinforce":
        raise ValueError("PG-OPD requires policy_loss_mode=reinforce.")
    validate_token_selection_config(loss)


class PGOPDTrainer(BaseOPDTrainer):
    """Trainer implementing signed detached-advantage PG-OPD."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        validate_pg_opd_config(config)
        super().__init__(*args, **kwargs)

    def algorithm_metric_aliases(self) -> dict[str, str]:
        return {
            "actor/distillation/reverse_kl_estimate": (
                "pg-opd/train/reverse_kl_estimate"
            ),
            "actor/distillation/selected_token_ratio": (
                "pg-opd/train/selected_token_ratio"
            ),
            "actor/distillation/selection_gap_mean": "pg-opd/train/gap_mean",
            "actor/distillation/selected_gap_mean": (
                "pg-opd/train/selected_gap_mean"
            ),
            "actor/distillation/selection_gradient_signal_relative_change": (
                "pg-opd/train/gradient_signal_relative_change"
            ),
            "actor/distillation/loss": "pg-opd/train/policy_loss",
        }
