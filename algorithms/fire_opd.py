"""Filtering-and-reweighting on-policy distillation (FiRe-OPD).

FiRe-OPD is materialized once on the controller for the complete rollout
batch.  This is important because both trajectory filtering and entropy
normalization are batch-level operations; an actor micro-batch is generally
only a small, non-representative slice of that batch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from tensordict import TensorDict

from utils.opd_runtime import BaseOPDTrainer, validate_opd_runtime_config
from verl.trainer import main_ppo_sync as verl_sync
from verl.workers.utils.padding import no_padding_2_padding, response_to_nested


FIRE_OPD_VARIANT = "fire_opd"
FIRE_OPD_LOSS_MODE = "fire_opd"
FIRE_OPD_WANDB_GROUP = "FiReOPD"
FIRE_OPD_DEFAULT_FILTER_RATIO = 0.20
FIRE_OPD_DEFAULT_ALPHA = 1.0
FIRE_OPD_DEFAULT_BETA = 1.0


@dataclass(frozen=True)
class FiReOPDBatchResult:
    """Complete-batch FiRe-OPD tensors and controller-side metrics."""

    trajectory_scores: torch.Tensor
    keep_mask: torch.Tensor
    token_mask: torch.Tensor
    normalized_weights: torch.Tensor
    advantages: torch.Tensor
    metrics: dict[str, float]


def _validate_finite_non_negative(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"FiRe-OPD {name} must be finite and non-negative; got {value}.")
    return value


def compute_fire_opd_batch(
    old_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    teacher_entropy: torch.Tensor,
    student_entropy: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    genuine_trajectory_mask: torch.Tensor | None = None,
    filter_ratio: float = FIRE_OPD_DEFAULT_FILTER_RATIO,
    alpha: float = FIRE_OPD_DEFAULT_ALPHA,
    beta: float = FIRE_OPD_DEFAULT_BETA,
) -> FiReOPDBatchResult:
    """Compute the official FiRe-OPD filter and token-weighted advantage.

    ``old_log_probs`` and ``student_entropy`` come from the same frozen
    pre-update Student pass.  Entropy maxima are computed only over valid
    tokens in trajectories retained by the Teacher-score filter.
    """

    tensors = {
        "old_log_probs": old_log_probs,
        "teacher_log_probs": teacher_log_probs,
        "teacher_entropy": teacher_entropy,
        "student_entropy": student_entropy,
        "response_mask": response_mask,
    }
    if any(tensor.ndim != 2 for tensor in tensors.values()):
        raise ValueError("FiRe-OPD token tensors must all be two-dimensional.")
    shapes = {tensor.shape for tensor in tensors.values()}
    if len(shapes) != 1:
        details = ", ".join(f"{name}={tuple(value.shape)}" for name, value in tensors.items())
        raise ValueError(f"FiRe-OPD token tensors must have identical shapes; got {details}.")

    filter_ratio = float(filter_ratio)
    if not math.isfinite(filter_ratio) or not 0.0 <= filter_ratio < 1.0:
        raise ValueError(
            "FiRe-OPD filter_ratio must be finite and lie in [0, 1); "
            f"got {filter_ratio}."
        )
    alpha = _validate_finite_non_negative("alpha", alpha)
    beta = _validate_finite_non_negative("beta", beta)

    response_mask = response_mask.bool()
    batch_size = response_mask.shape[0]
    if genuine_trajectory_mask is None:
        genuine_trajectory_mask = torch.ones(
            batch_size, dtype=torch.bool, device=response_mask.device
        )
    else:
        if genuine_trajectory_mask.ndim != 1 or genuine_trajectory_mask.shape[0] != batch_size:
            raise ValueError(
                "FiRe-OPD genuine_trajectory_mask must have one entry per trajectory."
            )
        genuine_trajectory_mask = genuine_trajectory_mask.to(
            device=response_mask.device, dtype=torch.bool
        )

    genuine_count = int(genuine_trajectory_mask.sum().item())
    if genuine_count == 0:
        raise ValueError("FiRe-OPD cannot operate on a batch with no genuine trajectories.")
    token_counts = response_mask.sum(dim=-1)
    empty_genuine = genuine_trajectory_mask & token_counts.eq(0)
    if bool(empty_genuine.any()):
        rows = empty_genuine.nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(
            "FiRe-OPD cannot score a trajectory with no valid response tokens; "
            f"empty rows: {rows[:5]}."
        )

    genuine_tokens = response_mask & genuine_trajectory_mask.unsqueeze(-1)
    for name, tensor in (
        ("old Student log-probabilities", old_log_probs),
        ("Teacher log-probabilities", teacher_log_probs),
        ("Teacher entropy", teacher_entropy),
        ("Student entropy", student_entropy),
    ):
        if not bool(torch.isfinite(tensor[genuine_tokens]).all()):
            raise ValueError(f"FiRe-OPD {name} must be finite on valid tokens.")
    if bool((teacher_entropy[genuine_tokens] < -1.0e-6).any()):
        raise ValueError("FiRe-OPD Teacher entropy must be non-negative.")
    if bool((student_entropy[genuine_tokens] < -1.0e-6).any()):
        raise ValueError("FiRe-OPD Student entropy must be non-negative.")

    with torch.no_grad():
        mask_float = response_mask.to(torch.float32)
        trajectory_scores = (
            teacher_log_probs.float() * mask_float
        ).sum(dim=-1) / token_counts.clamp_min(1).float()
        trajectory_scores = trajectory_scores.masked_fill(
            ~genuine_trajectory_mask, float("inf")
        )

        genuine_indices = genuine_trajectory_mask.nonzero(
            as_tuple=False
        ).flatten()
        # Use a stable ranking so tied Teacher scores have deterministic
        # behavior.  floor(N*p) is the largest discard count that never
        # exceeds the configured fraction.
        drop_count = min(
            genuine_count - 1,
            math.floor(genuine_count * filter_ratio),
        )
        keep_mask = genuine_trajectory_mask.clone()
        if drop_count:
            ranking = torch.argsort(
                trajectory_scores[genuine_indices], stable=True
            )
            dropped_indices = genuine_indices[ranking[:drop_count]]
            keep_mask[dropped_indices] = False

        token_mask = response_mask & keep_mask.unsqueeze(-1)
        retained_token_count = int(token_mask.sum().item())
        if retained_token_count == 0:
            raise RuntimeError("FiRe-OPD trajectory filtering retained no valid tokens.")

        teacher_entropy_fp32 = teacher_entropy.float().clamp_min(0.0)
        student_entropy_fp32 = student_entropy.float().clamp_min(0.0)
        max_teacher_entropy = teacher_entropy_fp32[token_mask].max()
        max_student_entropy = student_entropy_fp32[token_mask].max()
        eps = torch.finfo(torch.float32).eps
        teacher_confidence = 1.0 - teacher_entropy_fp32 / max_teacher_entropy.clamp_min(eps)
        student_confusion = student_entropy_fp32 / max_student_entropy.clamp_min(eps)
        teacher_confidence = teacher_confidence.clamp(0.0, 1.0)
        student_confusion = student_confusion.clamp(0.0, 1.0)

        raw_weights = (1.0 + alpha * teacher_confidence) * (
            1.0 + beta * student_confusion
        )
        raw_weights = raw_weights * token_mask
        per_trajectory_mean = raw_weights.sum(dim=-1) / token_mask.sum(
            dim=-1
        ).clamp_min(1)
        normalized_weights = torch.zeros_like(raw_weights)
        normalized_weights[keep_mask] = (
            raw_weights[keep_mask]
            / per_trajectory_mean[keep_mask].unsqueeze(-1).clamp_min(eps)
        )

        base_advantages = teacher_log_probs.float() - old_log_probs.float()
        advantages = normalized_weights * base_advantages
        advantages = advantages * token_mask

    retained_count = int(keep_mask.sum().item())
    valid_weights = normalized_weights[token_mask]
    valid_base_advantages = base_advantages[token_mask]
    valid_advantages = advantages[token_mask]
    metrics = {
        "fire-opd/train/generated_trajectory_count": float(genuine_count),
        "fire-opd/train/filtered_trajectory_count": float(drop_count),
        "fire-opd/train/retained_trajectory_count": float(retained_count),
        "fire-opd/train/retained_trajectory_ratio": retained_count / genuine_count,
        "fire-opd/train/retained_token_count": float(retained_token_count),
        "fire-opd/train/teacher_score_mean": float(
            trajectory_scores[genuine_trajectory_mask].mean().item()
        ),
        "fire-opd/train/teacher_entropy_max": float(max_teacher_entropy.item()),
        "fire-opd/train/student_entropy_max": float(max_student_entropy.item()),
        "fire-opd/train/teacher_confidence_mean": float(
            teacher_confidence[token_mask].mean().item()
        ),
        "fire-opd/train/student_confusion_mean": float(
            student_confusion[token_mask].mean().item()
        ),
        "fire-opd/train/token_weight_mean": float(valid_weights.mean().item()),
        "fire-opd/train/token_weight_min": float(valid_weights.min().item()),
        "fire-opd/train/token_weight_max": float(valid_weights.max().item()),
        "fire-opd/train/base_advantage_abs_mean": float(
            valid_base_advantages.abs().mean().item()
        ),
        "fire-opd/train/advantage_abs_mean": float(
            valid_advantages.abs().mean().item()
        ),
    }
    return FiReOPDBatchResult(
        trajectory_scores=trajectory_scores,
        keep_mask=keep_mask,
        token_mask=token_mask,
        normalized_weights=normalized_weights,
        advantages=advantages,
        metrics=metrics,
    )


def validate_fire_opd_config(config) -> None:
    """Validate the FiRe-OPD objective and official hyperparameters."""

    validate_opd_runtime_config(config)
    if str(config.algorithm.name) != FIRE_OPD_VARIANT:
        raise ValueError(f"FiRe-OPD requires algorithm.name={FIRE_OPD_VARIANT}.")
    if str(config.get("group_name", "")) != FIRE_OPD_WANDB_GROUP:
        raise ValueError(
            f"FiRe-OPD requires group_name={FIRE_OPD_WANDB_GROUP!r} so its "
            "W&B runs remain separate from other OPD variants."
        )

    loss = config.distillation.distillation_loss
    if str(loss.loss_mode) != FIRE_OPD_LOSS_MODE:
        raise ValueError(f"FiRe-OPD requires loss_mode={FIRE_OPD_LOSS_MODE}.")
    if str(loss.policy_loss_mode) != "reinforce":
        raise ValueError("FiRe-OPD requires policy_loss_mode=reinforce.")
    if int(loss.topk) <= 0:
        raise ValueError(
            "FiRe-OPD requires a positive entropy-transfer topk implementation setting."
        )

    settings = config.algorithm.fire_opd
    _validate_finite_non_negative("alpha", settings.alpha)
    _validate_finite_non_negative("beta", settings.beta)
    filter_ratio = float(settings.filter_ratio)
    if not math.isfinite(filter_ratio) or not 0.0 <= filter_ratio < 1.0:
        raise ValueError("FiRe-OPD filter_ratio must be finite and lie in [0, 1).")

    rollout_correction = config.algorithm.get("rollout_correction", {}) or {}
    if bool(rollout_correction.get("bypass_mode", False)):
        raise ValueError(
            "FiRe-OPD requires the pre-update Student entropy pass; rollout "
            "correction bypass_mode is unsupported."
        )
    if str(config.actor_rollout_ref.actor.loss_agg_mode) not in {
        "token-mean",
        "seq-mean-token-mean",
        "seq-mean-token-sum",
        "seq-mean-token-sum-norm",
    }:
        raise ValueError("FiRe-OPD received an unsupported actor loss_agg_mode.")

    teacher = config.distillation.teacher_models.teacher_model.inference
    if str(teacher.name) != "vllm":
        raise ValueError("FiRe-OPD full-vocabulary Teacher entropy currently requires vLLM.")
    vllm_kwargs = teacher.engine_kwargs.get("vllm", {})
    max_logprobs = vllm_kwargs.get("max_logprobs")
    if max_logprobs is None or int(max_logprobs) < int(loss.topk) + 1:
        raise ValueError(
            "FiRe-OPD requires vLLM max_logprobs >= topk + 1 for its bounded "
            "exact-entropy transfer."
        )
    entropy_topk = vllm_kwargs.get(
        "full_vocab_entropy_topk", vllm_kwargs.get("eopd_entropy_topk")
    )
    if entropy_topk is None or int(entropy_topk) != int(loss.topk):
        raise ValueError(
            "FiRe-OPD requires full_vocab_entropy_topk to match distillation topk."
        )


class FiReOPDTrainer(BaseOPDTrainer):
    """Controller-side FiRe materialization plus the shared OPD optimizer."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        validate_fire_opd_config(config)
        super().__init__(*args, **kwargs)

    def _compute_old_log_prob(self, batch, metrics):
        batch = super()._compute_old_log_prob(batch, metrics)
        fields = [
            "prompts",
            "responses",
            "response_mask",
            "teacher_logprobs",
            "teacher_entropy",
            "old_log_probs",
            "entropy",
        ]
        data = verl_sync.tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=fields,
        )
        response_mask_nested = data["response_mask"]
        if not response_mask_nested.is_nested:
            raise RuntimeError(
                "FiRe-OPD controller expects jagged response masks from TransferQueue."
            )
        response_mask = response_mask_nested.to_padded_tensor(False).bool()

        old_log_probs = data["old_log_probs"]
        if old_log_probs.is_nested:
            old_log_probs = old_log_probs.to_padded_tensor(0.0)
        student_entropy = data["entropy"]
        if student_entropy.is_nested:
            student_entropy = student_entropy.to_padded_tensor(0.0)
        teacher_log_probs = no_padding_2_padding(
            data["teacher_logprobs"], data
        ).squeeze(-1)
        teacher_entropy = no_padding_2_padding(
            data["teacher_entropy"], data
        ).squeeze(-1)
        genuine_mask = torch.tensor(
            [not bool(tag.get("is_padding", False)) for tag in batch.tags],
            dtype=torch.bool,
            device=response_mask.device,
        )

        settings = self.config.algorithm.fire_opd
        result = compute_fire_opd_batch(
            old_log_probs=old_log_probs,
            teacher_log_probs=teacher_log_probs,
            teacher_entropy=teacher_entropy,
            student_entropy=student_entropy,
            response_mask=response_mask,
            genuine_trajectory_mask=genuine_mask,
            filter_ratio=float(settings.filter_ratio),
            alpha=float(settings.alpha),
            beta=float(settings.beta),
        )

        loss_agg_mode = str(self.config.actor_rollout_ref.actor.loss_agg_mode)
        if loss_agg_mode == "token-mean":
            denominator = int(result.token_mask.sum().item())
            numerator = int(response_mask.sum().item())
        else:
            denominator = int(result.keep_mask.sum().item())
            numerator = len(batch)
        if denominator <= 0:
            raise RuntimeError("FiRe-OPD loss normalization has an empty denominator.")
        loss_normalization = float(numerator) / denominator

        output = TensorDict(
            {
                "fire_opd_advantages": response_to_nested(
                    result.advantages, response_mask_nested
                ),
                "fire_opd_token_mask": response_to_nested(
                    result.token_mask, response_mask_nested
                ),
            },
            batch_size=len(batch),
        )
        batch = verl_sync.tq.kv_batch_put(
            keys=batch.keys,
            partition_id=batch.partition_id,
            fields=output,
        )
        batch.extra_info["fire_opd_loss_normalization"] = loss_normalization
        metrics["fire-opd/train/loss_normalization"] = loss_normalization
        metrics.update(result.metrics)
        return batch

    def algorithm_metric_aliases(self) -> dict[str, str]:
        return {
            "actor/distillation/loss": "fire-opd/train/policy_loss",
        }


__all__ = [
    "FIRE_OPD_DEFAULT_ALPHA",
    "FIRE_OPD_DEFAULT_BETA",
    "FIRE_OPD_DEFAULT_FILTER_RATIO",
    "FIRE_OPD_VARIANT",
    "FiReOPDBatchResult",
    "FiReOPDTrainer",
    "compute_fire_opd_batch",
    "validate_fire_opd_config",
]
