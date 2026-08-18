"""Calibrated policy-gradient on-policy distillation.

Cal-OPD teacher-forces the same student rollout with a base teacher prompt and
two feedback-conditioned prompts.  It subtracts only the teacher's largest
self-deviation toward the student from the signed teacher-student gap.  The
remaining advantage keeps its original sign and is clipped at zero magnitude.
"""

from __future__ import annotations

from utils.opd_runtime import BaseOPDTrainer, validate_opd_runtime_config
from utils.prompts import get_prompt_template


CAL_OPD_VARIANT = "cal_opd"
CAL_POSITIVE_PROMPT = "qwen3_response_extremely_correct_feedback"
CAL_NEGATIVE_PROMPT = "qwen3_response_extremely_incorrect_feedback"


def validate_cal_opd_config(config) -> None:
    """Validate the complete Cal-OPD configuration contract."""

    validate_opd_runtime_config(config)
    if str(config.algorithm.name) != CAL_OPD_VARIANT:
        raise ValueError(f"Cal-OPD requires algorithm.name={CAL_OPD_VARIANT}.")
    loss = config.distillation.distillation_loss
    if str(loss.loss_mode) != "cal_reverse_kl":
        raise ValueError("Cal-OPD requires loss_mode=cal_reverse_kl.")
    if str(loss.policy_loss_mode) != "reinforce":
        raise ValueError("Cal-OPD requires policy_loss_mode=reinforce.")
    get_prompt_template(CAL_POSITIVE_PROMPT)
    get_prompt_template(CAL_NEGATIVE_PROMPT)


class CalOPDTrainer(BaseOPDTrainer):
    """Trainer using the signed, non-flipping calibrated advantage."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        validate_cal_opd_config(config)
        super().__init__(*args, **kwargs)

    def algorithm_metric_aliases(self) -> dict[str, str]:
        return {
            "actor/distillation/cal_advantage_mean": "cal-opd/train/advantage_mean",
            "actor/distillation/cal_advantage_abs_mean": "cal-opd/train/advantage_abs_mean",
            "actor/distillation/cal_zero_token_ratio": "cal-opd/train/zero_token_ratio",
            "actor/distillation/cal_teacher_self_deviation_mean": (
                "cal-opd/train/teacher_self_deviation_mean"
            ),
            "actor/distillation/cal_retained_magnitude_ratio": (
                "cal-opd/train/retained_magnitude_ratio"
            ),
            "actor/distillation/loss": "cal-opd/train/policy_loss",
        }
