"""Solution-privileged signed REINFORCE on-policy distillation.

Sol-OPD trains on the Student's original sampled token IDs under a Teacher
that receives a reference solution.  A second, unprivileged Teacher forward
scores the same token IDs only for diagnostics against standard OPD.
"""

from __future__ import annotations

import math

from utils.opd_runtime import (
    BaseOPDTrainer,
    validate_opd_runtime_config,
    validate_token_selection_config,
)
from utils.prompts import get_sol_privileged_prompt_name


SOL_OPD_VARIANT = "sol_opd"
SOL_OPD_LOSS_MODE = "sol_reverse_kl"
SOL_OPD_WANDB_PREFIX = "sol-opd/train"


def validate_sol_opd_config(config) -> None:
    """Validate the complete solution-privileged OPD contract."""

    validate_opd_runtime_config(config)
    if str(config.algorithm.name) != SOL_OPD_VARIANT:
        raise ValueError(f"Sol-OPD requires algorithm.name={SOL_OPD_VARIANT}.")

    loss = config.distillation.distillation_loss
    if str(loss.loss_mode) != SOL_OPD_LOSS_MODE:
        raise ValueError(f"Sol-OPD requires loss_mode={SOL_OPD_LOSS_MODE}.")
    if str(loss.policy_loss_mode) != "reinforce":
        raise ValueError("Sol-OPD requires policy_loss_mode=reinforce.")
    validate_token_selection_config(loss)

    epsilon = float(loss.sol_opd_epsilon)
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(
            f"Sol-OPD requires a finite positive sol_opd_epsilon, got {epsilon}."
        )

    student_prompt = str(config.student_prompt)
    teacher_prompt = str(config.teacher_prompt)
    expected_teacher_prompt = get_sol_privileged_prompt_name(student_prompt)
    if teacher_prompt != expected_teacher_prompt:
        raise ValueError(
            "Sol-OPD requires the solution-privileged thinking Teacher prompt; "
            f"Student prompt {student_prompt!r} requires {expected_teacher_prompt!r}, "
            f"got {teacher_prompt!r}."
        )


class SolOPDTrainer(BaseOPDTrainer):
    """Trainer using solution privilege and an unprivileged OPD control pass."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        validate_sol_opd_config(config)
        super().__init__(*args, **kwargs)

    def algorithm_metric_aliases(self) -> dict[str, str]:
        prefix = SOL_OPD_WANDB_PREFIX
        return {
            "actor/distillation/reverse_kl_estimate": (
                f"{prefix}/reverse_kl_estimate"
            ),
            "actor/distillation/student_sampled_token_prob": (
                f"{prefix}/student_sampled_token_prob_mean"
            ),
            "actor/distillation/teacher_sampled_token_prob": (
                f"{prefix}/solution_teacher_sampled_token_prob_mean"
            ),
            "actor/distillation/sol_unprivileged_teacher_sampled_token_prob": (
                f"{prefix}/unprivileged_teacher_sampled_token_prob_mean"
            ),
            "actor/distillation/selected_token_ratio": (
                f"{prefix}/selected_token_ratio"
            ),
            "actor/distillation/selection_gap_mean": f"{prefix}/gap_mean",
            "actor/distillation/selected_gap_mean": (
                f"{prefix}/selected_gap_mean"
            ),
            "actor/distillation/selection_gradient_signal_relative_change": (
                f"{prefix}/gradient_signal_relative_change"
            ),
            "actor/distillation/loss": f"{prefix}/policy_loss",
        }

    def _add_opd_training_metrics(self, metrics) -> None:
        super()._add_opd_training_metrics(metrics)

        stats_prefix = "actor/distillation/sol_stats/"

        def raw(name: str, default=0.0):
            return metrics.get(f"{stats_prefix}{name}", default)

        if any(key.startswith(stats_prefix) for key in metrics):
            token_count = raw("token_count")
            rollout_count = raw("rollout_count")
            ratio_rollout_count = raw("ratio_rollout_count")
            same_direction_token_count = raw("same_direction_token_count")
            same_direction_rollout_count = raw("same_direction_rollout_count")
            token_prefix = f"{SOL_OPD_WANDB_PREFIX}/token"
            rollout_prefix = f"{SOL_OPD_WANDB_PREFIX}/rollout_mean"

            opd_abs_sum = raw("opd_abs_sum")
            token_ratio_defined = float(opd_abs_sum) > 0.0
            token_r_mag = self._safe_ratio(raw("sol_abs_sum"), opd_abs_sum)
            rollout_r_mag = self._safe_ratio(
                raw("rollout_magnitude_ratio_sum"), ratio_rollout_count
            )
            rollout_ratio_defined = float(ratio_rollout_count) > 0.0
            has_rollouts = float(rollout_count) > 0.0
            ratio_excluded_rate = (
                1.0 - self._safe_ratio(ratio_rollout_count, rollout_count)
                if has_rollouts
                else 0.0
            )
            same_direction_excluded_rate = (
                1.0
                - self._safe_ratio(
                    same_direction_rollout_count, rollout_count
                )
                if has_rollouts
                else 0.0
            )
            metrics.update(
                {
                    f"{token_prefix}/overall_magnitude_ratio": token_r_mag,
                    f"{token_prefix}/overall_magnitude_change_percent": (
                        (token_r_mag - 1.0) * 100.0
                        if token_ratio_defined
                        else 0.0
                    ),
                    f"{token_prefix}/amplification_rate": self._safe_ratio(
                        raw("amplified_count"), token_count
                    ),
                    f"{token_prefix}/reduction_rate": self._safe_ratio(
                        raw("reduced_count"), token_count
                    ),
                    f"{token_prefix}/equal_rate": self._safe_ratio(
                        raw("equal_count"), token_count
                    ),
                    f"{token_prefix}/same_direction_amplification_rate": (
                        self._safe_ratio(
                            raw("same_direction_amplified_count"),
                            same_direction_token_count,
                        )
                    ),
                    f"{token_prefix}/same_direction_reduction_rate": (
                        self._safe_ratio(
                            raw("same_direction_reduced_count"),
                            same_direction_token_count,
                        )
                    ),
                    f"{token_prefix}/same_direction_eligible_rate": (
                        self._safe_ratio(
                            same_direction_token_count, token_count
                        )
                    ),
                    f"{token_prefix}/sign_flip_rate": self._safe_ratio(
                        raw("sign_flipped_count"), token_count
                    ),
                    f"{token_prefix}/category_same_direction_amplified_rate": (
                        self._safe_ratio(raw("category_amplified_count"), token_count)
                    ),
                    f"{token_prefix}/category_same_direction_reduced_rate": (
                        self._safe_ratio(raw("category_reduced_count"), token_count)
                    ),
                    f"{token_prefix}/category_sign_flipped_rate": self._safe_ratio(
                        raw("category_sign_flipped_count"), token_count
                    ),
                    f"{token_prefix}/category_approximately_unchanged_rate": (
                        self._safe_ratio(raw("category_unchanged_count"), token_count)
                    ),
                    f"{token_prefix}/privilege_induced_deviation_ratio": (
                        self._safe_ratio(raw("deviation_abs_sum"), raw("opd_abs_sum"))
                    ),
                    f"{token_prefix}/opd_advantage_mean": self._safe_ratio(
                        raw("opd_advantage_sum"), token_count
                    ),
                    f"{token_prefix}/sol_advantage_mean": self._safe_ratio(
                        raw("sol_advantage_sum"), token_count
                    ),
                    f"{rollout_prefix}/overall_magnitude_ratio": rollout_r_mag,
                    f"{rollout_prefix}/overall_magnitude_change_percent": (
                        (rollout_r_mag - 1.0) * 100.0
                        if rollout_ratio_defined
                        else 0.0
                    ),
                    f"{rollout_prefix}/amplification_rate": self._safe_ratio(
                        raw("rollout_amplification_rate_sum"), rollout_count
                    ),
                    f"{rollout_prefix}/reduction_rate": self._safe_ratio(
                        raw("rollout_reduction_rate_sum"), rollout_count
                    ),
                    f"{rollout_prefix}/equal_rate": self._safe_ratio(
                        raw("rollout_equal_rate_sum"), rollout_count
                    ),
                    f"{rollout_prefix}/same_direction_amplification_rate": (
                        self._safe_ratio(
                            raw("rollout_same_direction_amplification_rate_sum"),
                            same_direction_rollout_count,
                        )
                    ),
                    f"{rollout_prefix}/same_direction_reduction_rate": (
                        self._safe_ratio(
                            raw("rollout_same_direction_reduction_rate_sum"),
                            same_direction_rollout_count,
                        )
                    ),
                    f"{rollout_prefix}/sign_flip_rate": self._safe_ratio(
                        raw("rollout_sign_flip_rate_sum"), rollout_count
                    ),
                    f"{rollout_prefix}/category_same_direction_amplified_rate": (
                        self._safe_ratio(
                            raw("rollout_category_amplified_rate_sum"), rollout_count
                        )
                    ),
                    f"{rollout_prefix}/category_same_direction_reduced_rate": (
                        self._safe_ratio(
                            raw("rollout_category_reduced_rate_sum"), rollout_count
                        )
                    ),
                    f"{rollout_prefix}/category_sign_flipped_rate": self._safe_ratio(
                        raw("rollout_category_sign_flipped_rate_sum"), rollout_count
                    ),
                    f"{rollout_prefix}/category_approximately_unchanged_rate": (
                        self._safe_ratio(
                            raw("rollout_category_unchanged_rate_sum"), rollout_count
                        )
                    ),
                    f"{rollout_prefix}/privilege_induced_deviation_ratio": (
                        self._safe_ratio(
                            raw("rollout_deviation_ratio_sum"), ratio_rollout_count
                        )
                    ),
                    f"{rollout_prefix}/ratio_excluded_rate": (
                        ratio_excluded_rate
                    ),
                    f"{rollout_prefix}/same_direction_excluded_rate": (
                        same_direction_excluded_rate
                    ),
                }
            )

        for key in [key for key in metrics if key.startswith(stats_prefix)]:
            metrics.pop(key)
