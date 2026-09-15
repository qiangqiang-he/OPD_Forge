"""Difficulty-aware hybrid self-reinforcement and OPD (R²OPL-base).

For every prompt, R²OPL samples a complete response group from the Student,
uses the binary verifier outcomes to estimate prompt difficulty, and applies
one of two detached token-level signals:

* verifier-correct responses receive the fixed self-reinforcement advantage
  ``1``;
* verifier-incorrect responses receive ``lambda * (log pi_teacher -
  log pi_student)``.

Both branches are multiplied by the prompt-level difficulty ``1 - group_mean``.
The complete-batch group calculation lives on the controller, while the
Student log-probability in the OPD branch is recomputed in the actor forward
pass so the policy-gradient signal stays aligned with the current policy.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import torch
from tensordict import TensorDict

from utils.opd_runtime import BaseOPDTrainer, validate_opd_runtime_config
from verl.trainer import main_ppo_sync as verl_sync
from verl.workers.utils.padding import no_padding_2_padding, response_to_nested


R2OPL_BASE_VARIANT = "r2opl_base"
R2OPL_BASE_LOSS_MODE = "r2opl_base"
R2OPL_BASE_DEFAULT_LAMBDA = 1.0 / 20.0
R2OPL_BASE_ROLLOUTS_PER_PROMPT = 8
R2OPL_BASE_QUESTIONS_PER_BATCH = 32
R2OPL_BASE_TOTAL_ROLLOUTS = (
    R2OPL_BASE_ROLLOUTS_PER_PROMPT * R2OPL_BASE_QUESTIONS_PER_BATCH
)


@dataclass(frozen=True)
class R2OPLBaseBatchResult:
    """Controller-materialized R²OPL group statistics and token masks."""

    correct_mask: torch.Tensor
    error_mask: torch.Tensor
    difficulty: torch.Tensor
    correctness: torch.Tensor
    group_success: torch.Tensor
    metrics: dict[str, float]


def _safe_mean(values: torch.Tensor, *, zero: torch.Tensor) -> torch.Tensor:
    """Mean over a possibly empty tensor without creating NaNs."""

    return values.mean() if values.numel() else zero


def _trajectory_means(
    values: torch.Tensor,
    token_mask: torch.Tensor,
    trajectory_mask: torch.Tensor,
    *,
    zero: torch.Tensor,
) -> torch.Tensor:
    """Return one token-mean value for each selected trajectory.

    The R²OPL objective normalizes every response by its own valid token
    count.  Reporting the same sequence-level reduction prevents long
    responses from dominating the advantage diagnostics.
    """

    token_mask = token_mask.bool()
    counts = token_mask.sum(dim=-1)
    selected = trajectory_mask.bool()
    if not bool(selected.any()):
        return zero.reshape(1)[:0]
    if bool((selected & counts.eq(0)).any()):
        raise ValueError(
            "R²OPL-base cannot reduce a selected trajectory with no response tokens."
        )
    sequence_sums = (values * token_mask.to(values.dtype)).sum(dim=-1)
    sequence_means = sequence_sums / counts.clamp_min(1).to(values.dtype)
    return sequence_means[selected]


def _validate_binary_rewards(sequence_rewards: torch.Tensor) -> None:
    if not bool(torch.isfinite(sequence_rewards).all()):
        raise ValueError("R²OPL-base verifier rewards must be finite.")
    binary = sequence_rewards.eq(0.0) | sequence_rewards.eq(1.0)
    if not bool(binary.all()):
        invalid = sequence_rewards[~binary].detach().cpu().tolist()
        raise ValueError(
            "R²OPL-base requires binary verifier rewards in {0, 1}; "
            f"got invalid values {invalid[:5]}"
        )


def _ordered_groups(prompt_group_ids: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(str(group_id) for group_id in prompt_group_ids))


def compute_r2opl_base_batch(
    old_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    verifier_rewards: torch.Tensor,
    prompt_group_ids: Sequence[str],
    *,
    lambda_: float = R2OPL_BASE_DEFAULT_LAMBDA,
    genuine_trajectory_mask: torch.Tensor | None = None,
) -> R2OPLBaseBatchResult:
    """Compute complete-batch R²OPL group difficulty and branch masks.

    ``old_log_probs`` are used only for controller-side diagnostics.  The
    actual incorrect-trajectory OPD signal is recomputed from the current
    Student log-probabilities in :func:`compute_r2opl_base_loss`.
    """

    tensors = {
        "old_log_probs": old_log_probs,
        "teacher_log_probs": teacher_log_probs,
        "response_mask": response_mask,
    }
    if any(value.ndim != 2 for value in tensors.values()):
        raise ValueError("R²OPL-base token tensors must all be two-dimensional.")
    shapes = {tuple(value.shape) for value in tensors.values()}
    if len(shapes) != 1:
        details = ", ".join(f"{name}={tuple(value.shape)}" for name, value in tensors.items())
        raise ValueError(
            "R²OPL-base Student/Teacher log-probs and response mask must have "
            f"identical shapes; got {details}."
        )

    lambda_ = float(lambda_)
    if not math.isfinite(lambda_) or lambda_ < 0.0:
        raise ValueError(f"R²OPL-base lambda must be finite and non-negative; got {lambda_}.")

    response_mask = response_mask.bool()
    batch_size = response_mask.shape[0]
    if len(prompt_group_ids) != batch_size:
        raise ValueError(
            "R²OPL-base prompt_group_ids must have one entry per trajectory."
        )
    if genuine_trajectory_mask is None:
        genuine_trajectory_mask = torch.ones(
            batch_size, dtype=torch.bool, device=response_mask.device
        )
    elif genuine_trajectory_mask.ndim != 1 or genuine_trajectory_mask.shape[0] != batch_size:
        raise ValueError(
            "R²OPL-base genuine_trajectory_mask must have one entry per trajectory."
        )
    else:
        genuine_trajectory_mask = genuine_trajectory_mask.to(
            device=response_mask.device, dtype=torch.bool
        )

    if verifier_rewards.ndim == 2:
        if verifier_rewards.shape[0] != batch_size:
            raise ValueError("R²OPL-base verifier reward batch size must match trajectories.")
        sequence_rewards = verifier_rewards.float().sum(dim=-1)
    elif verifier_rewards.ndim == 1:
        sequence_rewards = verifier_rewards.float()
    else:
        raise ValueError("R²OPL-base verifier_rewards must be one- or two-dimensional.")
    if sequence_rewards.shape[0] != batch_size:
        raise ValueError("R²OPL-base verifier reward batch size must match trajectories.")

    genuine_tokens = response_mask & genuine_trajectory_mask.unsqueeze(-1)
    token_counts = genuine_tokens.sum(dim=-1)
    if bool((genuine_trajectory_mask & token_counts.eq(0)).any()):
        rows = (genuine_trajectory_mask & token_counts.eq(0)).nonzero(
            as_tuple=False
        ).flatten().tolist()
        raise ValueError(
            "R²OPL-base cannot score a genuine trajectory with no response tokens; "
            f"empty rows: {rows[:5]}"
        )
    _validate_binary_rewards(sequence_rewards[genuine_trajectory_mask])
    if not bool(torch.isfinite(old_log_probs[genuine_tokens]).all()):
        raise ValueError("R²OPL-base Student log-probabilities must be finite.")
    if not bool(torch.isfinite(teacher_log_probs[genuine_tokens]).all()):
        raise ValueError("R²OPL-base Teacher log-probabilities must be finite.")

    correctness = sequence_rewards.eq(1.0) & genuine_trajectory_mask
    difficulty = torch.zeros(
        batch_size, dtype=torch.float32, device=response_mask.device
    )
    group_success = []
    groups = _ordered_groups(prompt_group_ids)
    group_counts = Counter(
        str(group_id)
        for group_id, is_genuine in zip(prompt_group_ids, genuine_trajectory_mask.tolist(), strict=True)
        if is_genuine
    )
    for group_id in groups:
        indices = [
            index
            for index, row_group_id in enumerate(prompt_group_ids)
            if str(row_group_id) == group_id and bool(genuine_trajectory_mask[index])
        ]
        if not indices:
            continue
        index_tensor = torch.tensor(indices, dtype=torch.long, device=response_mask.device)
        success = correctness[index_tensor].float().mean()
        difficulty[index_tensor] = 1.0 - success
        group_success.append(success)

    if not group_success:
        raise ValueError("R²OPL-base requires at least one genuine prompt group.")

    correct_mask = response_mask & correctness.unsqueeze(-1)
    error_mask = response_mask & genuine_trajectory_mask.unsqueeze(-1) & ~correctness.unsqueeze(-1)
    teacher_minus_student = teacher_log_probs.float() - old_log_probs.float()
    zero = teacher_minus_student.new_zeros(())
    correct_scaled = difficulty.unsqueeze(-1).expand_as(teacher_minus_student)
    error_scaled = correct_scaled * lambda_ * teacher_minus_student

    valid_correct = correct_mask
    valid_error = error_mask
    # Match the requested objective and ``seq-mean-token-mean`` reducer:
    # first average tokens inside every trajectory, then average trajectories.
    # A response with more tokens therefore has no more influence on either
    # optimization or the corresponding W&B advantage metric.
    correct_mean = _safe_mean(
        _trajectory_means(
            correct_scaled,
            valid_correct,
            correctness,
            zero=zero,
        ),
        zero=zero,
    )
    error_trajectory_mask = genuine_trajectory_mask & ~correctness
    error_mean = _safe_mean(
        _trajectory_means(
            error_scaled,
            valid_error,
            error_trajectory_mask,
            zero=zero,
        ),
        zero=zero,
    )
    error_abs_mean = _safe_mean(
        _trajectory_means(
            error_scaled.abs(),
            valid_error,
            error_trajectory_mask,
            zero=zero,
        ),
        zero=zero,
    )
    raw_error_mean = _safe_mean(
        _trajectory_means(
            teacher_minus_student,
            valid_error,
            error_trajectory_mask,
            zero=zero,
        ),
        zero=zero,
    )
    raw_error_abs_mean = _safe_mean(
        _trajectory_means(
            teacher_minus_student.abs(),
            valid_error,
            error_trajectory_mask,
            zero=zero,
        ),
        zero=zero,
    )
    group_success_tensor = torch.stack(group_success)
    genuine_rewards = sequence_rewards[genuine_trajectory_mask]
    metrics = {
        "r2opl_base/train/mean_score": float(genuine_rewards.mean().item()),
        "r2opl_base/train/group_success_rate": float(group_success_tensor.mean().item()),
        "r2opl_base/train/prompt_difficulty_mean": float(
            (1.0 - group_success_tensor).mean().item()
        ),
        "r2opl_base/train/correct_trajectory_count": float(correctness.sum().item()),
        "r2opl_base/train/error_trajectory_count": float(
            (genuine_trajectory_mask & ~correctness).sum().item()
        ),
        "r2opl_base/train/correct_trajectory_ratio": float(
            correctness[genuine_trajectory_mask].float().mean().item()
        ),
        "r2opl_base/train/error_trajectory_ratio": float(
            (~correctness[genuine_trajectory_mask]).float().mean().item()
        ),
        # The requested advantage metrics are the final signals applied to the
        # policy loss.  Raw OPD values are also logged to make the lambda and
        # difficulty scaling directly auditable.
        "r2opl_base/train/correct_advantage_mean": float(correct_mean.item()),
        "r2opl_base/train/correct_raw_advantage_mean": 1.0 if bool(valid_correct.any()) else 0.0,
        "r2opl_base/train/error_opd_advantage_mean": float(error_mean.item()),
        "r2opl_base/train/error_opd_abs_advantage_mean": float(error_abs_mean.item()),
        "r2opl_base/train/error_raw_opd_mean": float(raw_error_mean.item()),
        "r2opl_base/train/error_raw_opd_abs_mean": float(raw_error_abs_mean.item()),
        "r2opl_base/train/correct_token_count": float(valid_correct.sum().item()),
        "r2opl_base/train/error_token_count": float(valid_error.sum().item()),
        "r2opl_base/train/lambda": lambda_,
    }
    return R2OPLBaseBatchResult(
        correct_mask=correct_mask,
        error_mask=error_mask,
        difficulty=difficulty,
        correctness=correctness,
        group_success=group_success_tensor,
        metrics=metrics,
    )


def validate_r2opl_base_config(config) -> None:
    """Validate the fixed R²OPL-base training contract."""

    validate_opd_runtime_config(config)
    if str(config.algorithm.name) != R2OPL_BASE_VARIANT:
        raise ValueError(
            f"R²OPL-base requires algorithm.name={R2OPL_BASE_VARIANT}."
        )
    if str(config.get("student_prompt", "")) != "qwen3_no_thinking_prompt":
        raise ValueError("R²OPL-base requires the No-Thinking Student prompt.")
    if str(config.get("teacher_prompt", "")) != "qwen3_no_thinking_prompt":
        raise ValueError("R²OPL-base requires the No-Thinking Teacher prompt.")

    loss = config.distillation.distillation_loss
    if str(loss.loss_mode) != R2OPL_BASE_LOSS_MODE:
        raise ValueError(
            f"R²OPL-base requires loss_mode={R2OPL_BASE_LOSS_MODE}."
        )
    if str(loss.policy_loss_mode) != "reinforce":
        raise ValueError("R²OPL-base requires policy_loss_mode=reinforce.")
    if not bool(loss.use_policy_gradient) or bool(loss.use_task_rewards):
        raise ValueError(
            "R²OPL-base requires use_policy_gradient=true and use_task_rewards=false."
        )
    if loss.topk is not None:
        raise ValueError("R²OPL-base uses sampled-token OPD and requires topk=null.")
    # Use the same optional numerical guard as standard OPD.  The shared
    # distillation reducer applies ``loss_max_clamp`` after this loss returns,
    # so a configured value clips the final detached token-level signal while
    # leaving the R²OPL branch construction unchanged.
    if loss.loss_max_clamp is not None:
        clamp_value = float(loss.loss_max_clamp)
        if not math.isfinite(clamp_value) or clamp_value < 0.0:
            raise ValueError(
                "R²OPL-base loss_max_clamp must be finite and non-negative when set."
            )
    if float(loss.selection_ratio) != 1.0:
        raise ValueError("R²OPL-base uses every valid response token; selection_ratio must be 1.0.")
    if str(config.actor_rollout_ref.actor.loss_agg_mode) != "seq-mean-token-mean":
        raise ValueError(
            "R²OPL-base requires loss_agg_mode=seq-mean-token-mean so each "
            "trajectory is averaged by its own response-token count."
        )

    settings = config.algorithm.get("r2opl_base", {})
    lambda_ = float(settings.get("lambda", R2OPL_BASE_DEFAULT_LAMBDA))
    if not math.isfinite(lambda_) or lambda_ < 0.0:
        raise ValueError(f"R²OPL-base lambda must be finite and non-negative; got {lambda_}.")

    rollout_n = int(config.actor_rollout_ref.rollout.n)
    question_count = int(config.data.train_batch_size)
    if rollout_n != R2OPL_BASE_ROLLOUTS_PER_PROMPT:
        raise ValueError(
            "R²OPL-base requires exactly 8 Student rollouts per training prompt."
        )
    if question_count != R2OPL_BASE_QUESTIONS_PER_BATCH:
        raise ValueError(
            "R²OPL-base requires exactly 32 training questions per step."
        )
    if question_count * rollout_n != R2OPL_BASE_TOTAL_ROLLOUTS:
        raise ValueError("R²OPL-base requires exactly 256 generated rollouts per step.")


class R2OPLBaseTrainer(BaseOPDTrainer):
    """Controller-side group difficulty plus detached hybrid OPD policy loss."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        validate_r2opl_base_config(config)
        super().__init__(*args, **kwargs)

    def _validate_complete_rollout_batch(
        self, batch, prompt_group_ids: Sequence[str]
    ) -> None:
        expected_questions = int(self.config.data.train_batch_size)
        expected_per_group = int(self.config.actor_rollout_ref.rollout.n)
        expected_rollouts = expected_questions * expected_per_group
        genuine_rollouts = sum(
            not bool(tag.get("is_padding", False)) for tag in batch.tags
        )
        if len(batch) != expected_rollouts or genuine_rollouts != expected_rollouts:
            raise RuntimeError(
                "R²OPL-base requires 256 genuine rollouts before training; got "
                f"batch_size={len(batch)}, genuine_rollouts={genuine_rollouts}."
            )
        counts = Counter(str(group_id) for group_id in prompt_group_ids)
        bad_counts = {
            group_id: count
            for group_id, count in counts.items()
            if count != expected_per_group
        }
        if len(counts) != expected_questions or bad_counts:
            raise RuntimeError(
                "R²OPL-base requires 32 prompt groups with 8 rollouts each; "
                f"group_count={len(counts)}, mismatched_counts={bad_counts}."
            )

    def _compute_old_log_prob(self, batch, metrics):
        batch = super()._compute_old_log_prob(batch, metrics)
        fields = [
            "uid",
            # ``no_padding_2_padding`` needs the original prompt/response
            # boundaries to align Teacher token log-probabilities with the
            # sampled response.  These fields must come from the same
            # TransferQueue snapshot as the log-probability tensors.
            "prompts",
            "responses",
            "response_mask",
            "rm_scores",
            "teacher_logprobs",
            "old_log_probs",
        ]
        data = verl_sync.tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=fields,
        )
        prompt_group_ids = [str(value) for value in data["uid"]]
        self._validate_complete_rollout_batch(batch, prompt_group_ids)

        response_mask_nested = data["response_mask"]
        if not response_mask_nested.is_nested:
            raise RuntimeError(
                "R²OPL-base controller expects jagged response masks from TransferQueue."
            )
        response_mask = response_mask_nested.to_padded_tensor(False).bool()
        old_log_probs = data["old_log_probs"]
        if old_log_probs.is_nested:
            old_log_probs = old_log_probs.to_padded_tensor(0.0)
        teacher_log_probs = no_padding_2_padding(
            data["teacher_logprobs"], data
        ).squeeze(-1)
        verifier_rewards = data["rm_scores"]
        if verifier_rewards.is_nested:
            verifier_rewards = verifier_rewards.to_padded_tensor(0.0)

        settings = self.config.algorithm.get("r2opl_base", {})
        lambda_ = float(settings.get("lambda", R2OPL_BASE_DEFAULT_LAMBDA))
        genuine_mask = torch.tensor(
            [not bool(tag.get("is_padding", False)) for tag in batch.tags],
            dtype=torch.bool,
            device=response_mask.device,
        )
        result = compute_r2opl_base_batch(
            old_log_probs=old_log_probs,
            teacher_log_probs=teacher_log_probs,
            response_mask=response_mask,
            verifier_rewards=verifier_rewards,
            prompt_group_ids=prompt_group_ids,
            lambda_=lambda_,
            genuine_trajectory_mask=genuine_mask,
        )

        difficulty_tokens = result.difficulty.unsqueeze(-1).expand_as(response_mask).clone()
        difficulty_tokens *= response_mask.to(difficulty_tokens.dtype)
        output = TensorDict(
            {
                "r2opl_base_correct_mask": response_to_nested(
                    result.correct_mask, response_mask_nested
                ),
                "r2opl_base_error_mask": response_to_nested(
                    result.error_mask, response_mask_nested
                ),
                "r2opl_base_difficulty": response_to_nested(
                    difficulty_tokens, response_mask_nested
                ),
                # A batch-aligned tensor keeps the algorithm hyperparameter
                # available after TransferQueue slicing into actor
                # micro-batches without coupling the shared VERL loss to the
                # full Hydra algorithm config.
                "r2opl_base_lambda": torch.full(
                    (len(batch),),
                    lambda_,
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
            "actor/entropy": "r2opl_base/train/entropy",
            "actor/grad_norm": "r2opl_base/train/total_grad_norm",
            "response_length/mean": "r2opl_base/train/response_length_mean",
            "response_length/max": "r2opl_base/train/response_length_max",
            "response_length/clip_ratio": (
                "r2opl_base/train/response_truncated_ratio"
            ),
            "perf/throughput": "r2opl_base/perf/tokens_per_second_per_gpu",
            "perf/time_per_step": "r2opl_base/perf/time_per_step",
            "actor/r2opl_base/correct_grad_norm": (
                "r2opl_base/train/correct_grad_norm"
            ),
            "actor/r2opl_base/error_grad_norm": (
                "r2opl_base/train/error_grad_norm"
            ),
            "actor/r2opl_base/total_grad_norm": (
                "r2opl_base/train/total_grad_norm"
            ),
            "actor/distillation/reverse_kl_estimate": (
                "r2opl_base/train/error_raw_opd_mean"
            ),
            "actor/distillation/loss": "r2opl_base/train/policy_loss",
        }


__all__ = [
    "R2OPL_BASE_DEFAULT_LAMBDA",
    "R2OPL_BASE_LOSS_MODE",
    "R2OPL_BASE_QUESTIONS_PER_BATCH",
    "R2OPL_BASE_ROLLOUTS_PER_PROMPT",
    "R2OPL_BASE_TOTAL_ROLLOUTS",
    "R2OPL_BASE_VARIANT",
    "R2OPLBaseBatchResult",
    "R2OPLBaseTrainer",
    "compute_r2opl_base_batch",
    "validate_r2opl_base_config",
]
