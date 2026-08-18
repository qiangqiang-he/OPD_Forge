"""Random-token GKD-OPD with signed reverse-KL signals."""

from __future__ import annotations

from utils.opd_runtime import BaseOPDTrainer, validate_opd_runtime_config


RANDOM_GKD_OPD_VARIANT = "random_gkd_opd"


def validate_random_gkd_opd_config(config) -> None:
    """Validate the complete Random GKD-OPD contract."""

    validate_opd_runtime_config(config)
    if str(config.algorithm.name) != RANDOM_GKD_OPD_VARIANT:
        raise ValueError(
            f"Random GKD-OPD requires algorithm.name={RANDOM_GKD_OPD_VARIANT}."
        )
    loss = config.distillation.distillation_loss
    if str(loss.loss_mode) != "random_reverse_kl":
        raise ValueError("Random GKD-OPD requires loss_mode=random_reverse_kl.")
    if str(loss.policy_loss_mode) != "vanilla":
        raise ValueError("Random GKD-OPD requires policy_loss_mode=vanilla.")
    ratio = float(loss.random_token_ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"Random GKD-OPD token ratio must lie in [0, 1], got {ratio}.")


class RandomGKDOPDTrainer(BaseOPDTrainer):
    """Trainer masking signed GKD signals with an independent random mask."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        validate_random_gkd_opd_config(config)
        super().__init__(*args, **kwargs)

    def algorithm_metric_aliases(self) -> dict[str, str]:
        return {
            "actor/distillation/reverse_kl_estimate": (
                "random-gkd-opd/train/reverse_kl_estimate"
            ),
            "actor/distillation/random_selected_token_ratio": (
                "random-gkd-opd/train/selected_token_ratio"
            ),
            "actor/distillation/random_gradient_signal_relative_change": (
                "random-gkd-opd/train/gradient_signal_relative_change"
            ),
            "actor/distillation/loss": "random-gkd-opd/train/policy_loss",
        }
