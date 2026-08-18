"""Privilege-sensitive on-policy distillation (PS-OPD)."""

from __future__ import annotations

from utils.opd_runtime import BaseOPDTrainer, validate_opd_runtime_config


PS_OPD_VARIANT = "ps_opd"


def validate_ps_opd_config(config) -> None:
    """Validate the complete privilege-sensitive OPD contract."""

    validate_opd_runtime_config(config)
    name = str(config.algorithm.name)
    if name != PS_OPD_VARIANT:
        raise ValueError(
            f"PS-OPD requires algorithm.name={PS_OPD_VARIANT}; got {name}."
        )
    loss = config.distillation.distillation_loss
    if str(loss.loss_mode) != "ps_reverse_kl":
        raise ValueError("PS-OPD requires loss_mode=ps_reverse_kl.")
    if str(loss.policy_loss_mode) != "vanilla":
        raise ValueError("PS-OPD requires policy_loss_mode=vanilla.")
    if float(loss.sensitivity_threshold) < 0:
        raise ValueError("PS-OPD sensitivity_threshold must be non-negative.")
    if float(loss.w_sens) < 0 or float(loss.w_stable) < 0:
        raise ValueError(
            "PS-OPD token weights w_sens and w_stable must be non-negative."
        )


class PSOPDTrainer(BaseOPDTrainer):
    """Trainer implementing privilege-sensitive OPD."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        validate_ps_opd_config(config)
        super().__init__(*args, **kwargs)

    def algorithm_metric_aliases(self) -> dict[str, str]:
        return {
            "actor/distillation/ps_sensitive_token_ratio": (
                "ps-opd/train/sensitive_token_ratio"
            ),
            "actor/distillation/ps_sensitivity_mean": "ps-opd/train/sensitivity_mean",
            "actor/distillation/ps_gradient_signal_relative_change": (
                "ps-opd/train/gradient_signal_relative_change"
            ),
            "actor/distillation/loss": "ps-opd/train/policy_loss",
        }
