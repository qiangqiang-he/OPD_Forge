"""Answer-privileged signed REINFORCE on-policy distillation.

Pri-OPD uses the same sampled-token reverse-KL policy-gradient objective as
PG-OPD.  Its only algorithmic intervention is the Teacher context: the
Teacher receives a verified ground-truth answer while the Student rollout
prompt remains unchanged.  Both models score the Student's original sampled
response token IDs.
"""

from __future__ import annotations

from utils.opd_runtime import (
    BaseOPDTrainer,
    validate_opd_runtime_config,
    validate_token_selection_config,
)
from utils.prompts import get_pri_privileged_prompt_name


PRI_OPD_VARIANT = "pri_opd"


def validate_pri_opd_config(config) -> None:
    """Validate the complete answer-privileged PG-OPD contract."""

    validate_opd_runtime_config(config)
    if str(config.algorithm.name) != PRI_OPD_VARIANT:
        raise ValueError(f"Pri-OPD requires algorithm.name={PRI_OPD_VARIANT}.")

    loss = config.distillation.distillation_loss
    if str(loss.loss_mode) != "reverse_kl":
        raise ValueError("Pri-OPD requires loss_mode=reverse_kl.")
    if str(loss.policy_loss_mode) != "reinforce":
        raise ValueError("Pri-OPD requires policy_loss_mode=reinforce.")
    validate_token_selection_config(loss)

    student_prompt = str(config.student_prompt)
    teacher_prompt = str(config.teacher_prompt)
    expected_teacher_prompt = get_pri_privileged_prompt_name(student_prompt)
    if teacher_prompt != expected_teacher_prompt:
        raise ValueError(
            "Pri-OPD requires the mode-matched answer-privileged Teacher prompt; "
            f"Student prompt {student_prompt!r} requires {expected_teacher_prompt!r}, "
            f"got {teacher_prompt!r}."
        )


class PriOPDTrainer(BaseOPDTrainer):
    """Trainer using a ground-truth-answer-privileged Teacher context."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        validate_pri_opd_config(config)
        super().__init__(*args, **kwargs)

    def algorithm_metric_aliases(self) -> dict[str, str]:
        return {
            "actor/distillation/reverse_kl_estimate": (
                "pri-opd/train/reverse_kl_estimate"
            ),
            "actor/distillation/selected_token_ratio": (
                "pri-opd/train/selected_token_ratio"
            ),
            "actor/distillation/selection_gap_mean": "pri-opd/train/gap_mean",
            "actor/distillation/selected_gap_mean": (
                "pri-opd/train/selected_gap_mean"
            ),
            "actor/distillation/selection_gradient_signal_relative_change": (
                "pri-opd/train/gradient_signal_relative_change"
            ),
            "actor/distillation/loss": "pri-opd/train/policy_loss",
        }
