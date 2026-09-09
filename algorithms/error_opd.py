"""Error-only signed on-policy distillation.

Error-OPD is identical to PG-OPD on incorrect Student rollouts and skips
correct rollouts completely.  The binary verifier result is materialized on
the controller before actor micro-batching so filtering and loss
normalization use the complete rollout batch rather than a local slice.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tensordict import TensorDict

from utils.opd_runtime import (
    BaseOPDTrainer,
    validate_opd_runtime_config,
    validate_token_selection_config,
)
from verl.trainer import main_ppo_sync as verl_sync
from verl.workers.utils.padding import response_to_nested


ERROR_OPD_VARIANT = "error_opd"
ERROR_OPD_LOSS_MODE = "error_reverse_kl"


@dataclass(frozen=True)
class ErrorOPDBatchResult:
    """Controller-computed error gate and loss-normalization statistics."""

    error_rollout_mask: torch.Tensor
    token_mask: torch.Tensor
    loss_normalization: float
    metrics: dict[str, float]


def compute_error_opd_batch(
    rm_scores: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    genuine_trajectory_mask: torch.Tensor | None = None,
    loss_agg_mode: str = "seq-mean-token-mean",
) -> ErrorOPDBatchResult:
    """Gate OPD to verifier-incorrect trajectories.

    ``rm_scores`` follows VERL's token-level reward layout.  A sequence is
    correct iff its total score is greater than 0.5.  Consequently malformed
    or unextractable answers, whose math-verifier reward is zero, are treated
    as errors and retained for OPD.

    Correct trajectories remain in the physical actor batch with zero signal.
    ``loss_normalization`` corrects the unchanged global reducer denominator,
    making the result equivalent to deleting correct trajectories first.
    """

    if rm_scores.ndim != 2 or response_mask.ndim != 2:
        raise ValueError("Error-OPD rm_scores and response_mask must be 2-D tensors.")
    if rm_scores.shape != response_mask.shape:
        raise ValueError(
            "Error-OPD rm_scores and response_mask must have identical shapes; "
            f"got {tuple(rm_scores.shape)} and {tuple(response_mask.shape)}."
        )

    response_mask = response_mask.bool()
    batch_size = response_mask.shape[0]
    if genuine_trajectory_mask is None:
        genuine_trajectory_mask = torch.ones(
            batch_size, dtype=torch.bool, device=response_mask.device
        )
    elif (
        genuine_trajectory_mask.ndim != 1
        or genuine_trajectory_mask.shape[0] != batch_size
    ):
        raise ValueError(
            "Error-OPD genuine_trajectory_mask must have one entry per trajectory."
        )
    else:
        genuine_trajectory_mask = genuine_trajectory_mask.to(
            device=response_mask.device, dtype=torch.bool
        )

    genuine_count = int(genuine_trajectory_mask.sum().item())
    if genuine_count == 0:
        raise ValueError(
            "Error-OPD cannot operate on a batch with no genuine trajectories."
        )
    genuine_tokens = response_mask & genuine_trajectory_mask.unsqueeze(-1)
    if not bool(torch.isfinite(rm_scores[genuine_tokens]).all()):
        raise ValueError("Error-OPD verifier scores must be finite on valid tokens.")

    rollout_correct = rm_scores.float().sum(dim=-1) > 0.5
    correct_rollout_mask = genuine_trajectory_mask & rollout_correct
    error_rollout_mask = genuine_trajectory_mask & ~rollout_correct
    token_mask = response_mask & error_rollout_mask.unsqueeze(-1)

    error_count = int(error_rollout_mask.sum().item())
    correct_count = int(correct_rollout_mask.sum().item())
    genuine_token_count = int(genuine_tokens.sum().item())
    error_token_count = int(token_mask.sum().item())
    if loss_agg_mode == "token-mean":
        numerator = genuine_token_count
        denominator = error_token_count
    elif loss_agg_mode in {
        "seq-mean-token-mean",
        "seq-mean-token-sum",
        "seq-mean-token-sum-norm",
    }:
        numerator = genuine_count
        denominator = error_count
    else:
        raise ValueError(
            f"Error-OPD received unsupported loss_agg_mode={loss_agg_mode!r}."
        )

    # All-correct batches are intentional no-ops.  Keeping the physical batch
    # avoids empty actor micro-batches while a zero multiplier preserves a
    # valid autograd graph with exactly zero gradient.
    loss_normalization = (
        float(numerator) / float(denominator) if denominator > 0 else 0.0
    )
    metrics = {
        "error-opd/train/generated_trajectory_count": float(genuine_count),
        "error-opd/train/error_trajectory_count": float(error_count),
        "error-opd/train/error_trajectory_ratio": error_count / genuine_count,
        "error-opd/train/skipped_correct_trajectory_count": float(correct_count),
        "error-opd/train/error_token_count": float(error_token_count),
        "error-opd/train/loss_normalization": loss_normalization,
    }
    return ErrorOPDBatchResult(
        error_rollout_mask=error_rollout_mask,
        token_mask=token_mask,
        loss_normalization=loss_normalization,
        metrics=metrics,
    )


def validate_error_opd_config(config) -> None:
    """Validate the complete Error-OPD No-Thinking contract."""

    validate_opd_runtime_config(config)
    if str(config.algorithm.name) != ERROR_OPD_VARIANT:
        raise ValueError(f"Error-OPD requires algorithm.name={ERROR_OPD_VARIANT}.")
    if str(config.get("student_prompt", "")) != "qwen3_no_thinking_prompt":
        raise ValueError("Error-OPD requires the No-Thinking Student prompt.")
    if str(config.get("teacher_prompt", "")) != "qwen3_no_thinking_prompt":
        raise ValueError("Error-OPD requires the No-Thinking Teacher prompt.")

    loss = config.distillation.distillation_loss
    if str(loss.loss_mode) != ERROR_OPD_LOSS_MODE:
        raise ValueError(f"Error-OPD requires loss_mode={ERROR_OPD_LOSS_MODE}.")
    if str(loss.policy_loss_mode) != "reinforce":
        raise ValueError("Error-OPD requires policy_loss_mode=reinforce.")
    validate_token_selection_config(loss)


class ErrorOPDTrainer(BaseOPDTrainer):
    """PG-OPD trainer that skips verifier-correct Student rollouts."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        validate_error_opd_config(config)
        super().__init__(*args, **kwargs)

    def _compute_old_log_prob(self, batch, metrics):
        batch = super()._compute_old_log_prob(batch, metrics)
        data = verl_sync.tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["response_mask", "rm_scores"],
        )
        response_mask_nested = data["response_mask"]
        if not response_mask_nested.is_nested:
            raise RuntimeError(
                "Error-OPD controller expects jagged response masks from TransferQueue."
            )
        response_mask = response_mask_nested.to_padded_tensor(False).bool()
        rm_scores = data["rm_scores"]
        if rm_scores.is_nested:
            rm_scores = rm_scores.to_padded_tensor(0.0)
        genuine_mask = torch.tensor(
            [not bool(tag.get("is_padding", False)) for tag in batch.tags],
            dtype=torch.bool,
            device=response_mask.device,
        )
        result = compute_error_opd_batch(
            rm_scores=rm_scores,
            response_mask=response_mask,
            genuine_trajectory_mask=genuine_mask,
            loss_agg_mode=str(self.config.actor_rollout_ref.actor.loss_agg_mode),
        )

        output = TensorDict(
            {
                "error_opd_token_mask": response_to_nested(
                    result.token_mask, response_mask_nested
                ),
                "error_opd_loss_normalization": torch.full(
                    (len(batch),),
                    result.loss_normalization,
                    dtype=torch.float32,
                    device=response_mask.device,
                ),
            },
            batch_size=len(batch),
        )
        batch = verl_sync.tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=output,
        )
        metrics.update(result.metrics)
        return batch

    def algorithm_metric_aliases(self) -> dict[str, str]:
        return {
            "actor/distillation/reverse_kl_estimate": (
                "error-opd/train/reverse_kl_estimate"
            ),
            "actor/distillation/selected_token_ratio": (
                "error-opd/train/selected_token_ratio"
            ),
            "actor/distillation/selection_gap_mean": "error-opd/train/gap_mean",
            "actor/distillation/selected_gap_mean": (
                "error-opd/train/selected_gap_mean"
            ),
            "actor/distillation/selection_gradient_signal_relative_change": (
                "error-opd/train/gradient_signal_relative_change"
            ),
            "actor/distillation/loss": "error-opd/train/policy_loss",
        }


__all__ = [
    "ERROR_OPD_LOSS_MODE",
    "ERROR_OPD_VARIANT",
    "ErrorOPDBatchResult",
    "ErrorOPDTrainer",
    "compute_error_opd_batch",
    "validate_error_opd_config",
]
