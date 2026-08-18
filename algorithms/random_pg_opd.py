"""Random-token signed REINFORCE on-policy distillation."""

from __future__ import annotations

from utils.opd_runtime import BaseOPDTrainer, validate_opd_runtime_config


RANDOM_PG_OPD_VARIANT = "random_pg_opd"


def validate_random_pg_opd_config(config) -> None:
    """Validate the complete Random PG-OPD contract."""

    validate_opd_runtime_config(config)
    if str(config.algorithm.name) != RANDOM_PG_OPD_VARIANT:
        raise ValueError(
            f"Random PG-OPD requires algorithm.name={RANDOM_PG_OPD_VARIANT}."
        )
    loss = config.distillation.distillation_loss
    if str(loss.loss_mode) != "random_reverse_kl":
        raise ValueError("Random PG-OPD requires loss_mode=random_reverse_kl.")
    if str(loss.policy_loss_mode) != "reinforce":
        raise ValueError("Random PG-OPD requires policy_loss_mode=reinforce.")
    ratio = float(loss.random_token_ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(f"Random PG-OPD token ratio must lie in [0, 1], got {ratio}.")


class RandomPGOPDTrainer(BaseOPDTrainer):
    """Trainer retaining a random subset of signed PG advantages."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        validate_random_pg_opd_config(config)
        super().__init__(*args, **kwargs)

    def algorithm_metric_aliases(self) -> dict[str, str]:
        return {
            "actor/distillation/reverse_kl_estimate": (
                "random-pg-opd/train/reverse_kl_estimate"
            ),
            "actor/distillation/random_selected_token_ratio": (
                "random-pg-opd/train/selected_token_ratio"
            ),
            "actor/distillation/random_gradient_signal_relative_change": (
                "random-pg-opd/train/gradient_signal_relative_change"
            ),
            "actor/distillation/loss": "random-pg-opd/train/policy_loss",
        }
