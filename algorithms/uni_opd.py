"""Unified trajectory-level on-policy distillation (Uni-OPD).

Uni-OPD first averages the sampled-token OPD signal over each Student
trajectory, balances correct and incorrect trajectories across the complete
rollout batch, applies a per-prompt spread margin, and finally broadcasts the
calibrated trajectory return back to every sampled response token.

The balancing and margin operations run on the controller rather than inside
an actor micro-batch.  This is essential: a worker micro-batch does not contain
all 16 rollouts for every prompt and therefore cannot reproduce the algorithm's
global correctness ratio or group-level means.
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


UNI_OPD_VARIANT = "uni_opd"
UNI_OPD_LOSS_MODE = "uni_opd"
UNI_OPD_WANDB_GROUP = "UniOPD"
UNI_OPD_PUBLICATION_WANDB_GROUP = "PUB_Uni_OPD_Thinking"
UNI_OPD_ROLLOUTS_PER_PROMPT = 16
UNI_OPD_TARGET_CORRECT_RATIO = 0.5
UNI_OPD_DEFAULT_MARGIN = 0.4


@dataclass(frozen=True)
class UniOPDBatchResult:
    """Materialized Uni-OPD signal for one complete rollout batch."""

    raw_returns: torch.Tensor
    calibrated_returns: torch.Tensor
    keep_mask: torch.Tensor
    token_advantages: torch.Tensor
    metrics: dict[str, float]


def _transfer_queue_prompt_group_ids(values) -> list[str]:
    """Normalize TransferQueue's list-backed non-tensor UID field.

    TensorDict materializes non-tensor columns as ``LinkedList`` objects.  They
    intentionally support normal Python iteration but do not expose NumPy's
    or Torch's ``tolist`` method.
    """

    return [str(value) for value in values]


def _ordered_groups(prompt_group_ids: Sequence[str]) -> list[str]:
    """Return group identifiers in stable first-occurrence order."""

    return list(dict.fromkeys(str(group_id) for group_id in prompt_group_ids))


def _uniform_group_subsample(
    candidate_mask: torch.Tensor,
    prompt_group_ids: Sequence[str],
    keep_count: int,
    *,
    seed: int,
) -> torch.Tensor:
    """Randomly retain candidates with quotas spread evenly across groups.

    A round-robin draw gives every prompt group one retained majority sample
    before any group receives a second one, subject to the samples available in
    that group.  Both group tie-breaking and within-group choices are seeded.
    """

    if candidate_mask.ndim != 1:
        raise ValueError("Uni-OPD candidate_mask must be one-dimensional.")
    candidate_count = int(candidate_mask.sum().item())
    if not 0 <= keep_count <= candidate_count:
        raise ValueError(
            "Uni-OPD keep_count must lie between zero and the candidate count; "
            f"got keep_count={keep_count}, candidate_count={candidate_count}."
        )

    selected = torch.zeros_like(candidate_mask, dtype=torch.bool)
    if keep_count == 0:
        return selected
    if keep_count == candidate_count:
        return candidate_mask.bool().clone()

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    groups = _ordered_groups(prompt_group_ids)
    group_order = torch.randperm(len(groups), generator=generator).tolist()

    pools: dict[str, list[int]] = {}
    candidate_cpu = candidate_mask.detach().bool().cpu()
    for group_id in groups:
        indices = [
            index
            for index, row_group_id in enumerate(prompt_group_ids)
            if str(row_group_id) == group_id and bool(candidate_cpu[index])
        ]
        if indices:
            permutation = torch.randperm(
                len(indices), generator=generator
            ).tolist()
            pools[group_id] = [indices[offset] for offset in permutation]

    selected_indices: list[int] = []
    pool_offsets = {group_id: 0 for group_id in pools}
    while len(selected_indices) < keep_count:
        made_progress = False
        for group_offset in group_order:
            group_id = groups[group_offset]
            pool = pools.get(group_id, [])
            offset = pool_offsets.get(group_id, 0)
            if offset >= len(pool):
                continue
            selected_indices.append(pool[offset])
            pool_offsets[group_id] = offset + 1
            made_progress = True
            if len(selected_indices) == keep_count:
                break
        if not made_progress:
            raise RuntimeError(
                "Uni-OPD group-balanced subsampling exhausted candidates before "
                "filling the requested quota."
            )

    selected[
        torch.tensor(selected_indices, dtype=torch.long, device=selected.device)
    ] = True
    return selected


def select_correctness_balanced_rollouts(
    correctness: torch.Tensor,
    prompt_group_ids: Sequence[str],
    *,
    target_correct_ratio: float = UNI_OPD_TARGET_CORRECT_RATIO,
    seed: int = 0,
) -> torch.Tensor:
    """Select the 1:1 Uni-OPD training subset without replacement.

    The minority side is always retained in full.  The majority side is
    subsampled uniformly across prompt groups.  If the complete batch contains
    only one class, balancing is skipped and every trajectory is retained.
    """

    if correctness.ndim != 1:
        raise ValueError("Uni-OPD correctness must be one-dimensional.")
    if correctness.numel() == 0:
        raise ValueError("Uni-OPD cannot balance an empty rollout batch.")
    if len(prompt_group_ids) != correctness.numel():
        raise ValueError(
            "Uni-OPD prompt_group_ids must have one entry per trajectory."
        )
    if not math.isclose(
        float(target_correct_ratio),
        UNI_OPD_TARGET_CORRECT_RATIO,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "This Uni-OPD implementation requires target_correct_ratio=0.5; "
            f"got {target_correct_ratio}."
        )

    correctness = correctness.bool()
    correct_count = int(correctness.sum().item())
    incorrect_count = int(correctness.numel() - correct_count)
    if correct_count == 0 or incorrect_count == 0 or correct_count == incorrect_count:
        return torch.ones_like(correctness, dtype=torch.bool)

    majority_is_correct = correct_count > incorrect_count
    majority_mask = correctness if majority_is_correct else ~correctness
    minority_mask = ~majority_mask
    retained_majority = _uniform_group_subsample(
        majority_mask,
        prompt_group_ids,
        keep_count=int(minority_mask.sum().item()),
        seed=seed,
    )
    return minority_mask | retained_majority


def compute_uni_opd_batch(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    verifier_rewards: torch.Tensor,
    prompt_group_ids: Sequence[str],
    *,
    margin_delta: float = UNI_OPD_DEFAULT_MARGIN,
    target_correct_ratio: float = UNI_OPD_TARGET_CORRECT_RATIO,
    seed: int = 0,
) -> UniOPDBatchResult:
    """Compute balancing, group margin shift, and token broadcast for Uni-OPD."""

    if student_log_probs.ndim != 2 or teacher_log_probs.ndim != 2:
        raise ValueError("Uni-OPD Student and Teacher log-probs must be 2-D.")
    if (
        student_log_probs.shape != teacher_log_probs.shape
        or student_log_probs.shape != response_mask.shape
    ):
        raise ValueError(
            "Uni-OPD Student log-probs, Teacher log-probs, and response mask "
            "must have identical shapes."
        )
    if verifier_rewards.ndim == 2:
        if verifier_rewards.shape[0] != student_log_probs.shape[0]:
            raise ValueError(
                "Uni-OPD verifier reward batch size must match the trajectories."
            )
        sequence_rewards = verifier_rewards.float().sum(dim=-1)
    elif verifier_rewards.ndim == 1:
        sequence_rewards = verifier_rewards.float()
    else:
        raise ValueError("Uni-OPD verifier_rewards must be one- or two-dimensional.")
    if sequence_rewards.shape[0] != student_log_probs.shape[0]:
        raise ValueError(
            "Uni-OPD verifier reward batch size must match the trajectories."
        )
    if len(prompt_group_ids) != student_log_probs.shape[0]:
        raise ValueError(
            "Uni-OPD prompt_group_ids must have one entry per trajectory."
        )

    margin_delta = float(margin_delta)
    if not math.isfinite(margin_delta) or margin_delta < 0.0:
        raise ValueError(
            "Uni-OPD margin_delta must be finite and non-negative; "
            f"got {margin_delta}."
        )
    if not torch.isfinite(sequence_rewards).all():
        raise ValueError("Uni-OPD verifier rewards must be finite.")
    binary_rewards = sequence_rewards.eq(0.0) | sequence_rewards.eq(1.0)
    if not bool(binary_rewards.all()):
        invalid = sequence_rewards[~binary_rewards].detach().cpu().tolist()
        raise ValueError(
            "Uni-OPD requires binary verifier rewards in {0, 1}; "
            f"got invalid values {invalid[:5]}."
        )

    response_mask = response_mask.bool()
    token_counts = response_mask.sum(dim=-1)
    if bool(token_counts.eq(0).any()):
        empty_rows = token_counts.eq(0).nonzero(as_tuple=False).flatten().tolist()
        raise ValueError(
            "Uni-OPD cannot average a trajectory with no sampled response tokens; "
            f"empty rows: {empty_rows[:5]}."
        )
    valid_student = student_log_probs[response_mask]
    valid_teacher = teacher_log_probs[response_mask]
    if not torch.isfinite(valid_student).all() or not torch.isfinite(valid_teacher).all():
        raise ValueError("Uni-OPD sampled-token log-probabilities must be finite.")

    with torch.no_grad():
        token_opd = teacher_log_probs.float() - student_log_probs.float()
        raw_returns = (token_opd * response_mask).sum(dim=-1) / token_counts.float()
        correctness = sequence_rewards.eq(1.0)
        keep_mask = select_correctness_balanced_rollouts(
            correctness,
            prompt_group_ids,
            target_correct_ratio=target_correct_ratio,
            seed=seed,
        )

        calibrated_returns = raw_returns.clone()
        margins_before: list[float] = []
        margins_after: list[float] = []
        shifted_group_count = 0
        for group_id in _ordered_groups(prompt_group_ids):
            group_indices = torch.tensor(
                [
                    index
                    for index, row_group_id in enumerate(prompt_group_ids)
                    if str(row_group_id) == group_id
                ],
                dtype=torch.long,
                device=raw_returns.device,
            )
            group_selected = keep_mask[group_indices]
            selected_indices = group_indices[group_selected]
            positive_indices = selected_indices[correctness[selected_indices]]
            negative_indices = selected_indices[~correctness[selected_indices]]
            if positive_indices.numel() == 0 or negative_indices.numel() == 0:
                continue

            margin_before = (
                raw_returns[positive_indices].mean()
                - raw_returns[negative_indices].mean()
            )
            margins_before.append(float(margin_before.item()))
            if float(margin_before.item()) < margin_delta:
                margin_shift = margin_delta - margin_before
                calibrated_returns[positive_indices] += margin_shift / 2.0
                calibrated_returns[negative_indices] -= margin_shift / 2.0
                shifted_group_count += 1
            margin_after = (
                calibrated_returns[positive_indices].mean()
                - calibrated_returns[negative_indices].mean()
            )
            margins_after.append(float(margin_after.item()))

        token_advantages = calibrated_returns.unsqueeze(-1).expand_as(
            token_opd
        ).clone()
        token_advantages *= response_mask
        token_advantages[~keep_mask] = 0.0

    generated_count = int(correctness.numel())
    selected_count = int(keep_mask.sum().item())
    correct_count = int(correctness.sum().item())
    selected_correct_count = int((correctness & keep_mask).sum().item())
    selected_incorrect_count = selected_count - selected_correct_count
    selected_returns = calibrated_returns[keep_mask]
    metrics = {
        "uni-opd/train/generated_trajectory_count": float(generated_count),
        "uni-opd/train/selected_trajectory_count": float(selected_count),
        "uni-opd/train/selected_trajectory_ratio": selected_count / generated_count,
        "uni-opd/train/correct_count_before_balance": float(correct_count),
        "uni-opd/train/incorrect_count_before_balance": float(
            generated_count - correct_count
        ),
        "uni-opd/train/correct_ratio_before_balance": correct_count
        / generated_count,
        "uni-opd/train/correct_count_after_balance": float(
            selected_correct_count
        ),
        "uni-opd/train/incorrect_count_after_balance": float(
            selected_incorrect_count
        ),
        "uni-opd/train/correct_ratio_after_balance": selected_correct_count
        / selected_count,
        "uni-opd/train/raw_return_mean": float(raw_returns[keep_mask].mean().item()),
        "uni-opd/train/calibrated_return_mean": float(
            selected_returns.mean().item()
        ),
        "uni-opd/train/mixed_group_count": float(len(margins_before)),
        "uni-opd/train/shifted_group_count": float(shifted_group_count),
        "uni-opd/train/margin_before_mean": (
            float(sum(margins_before) / len(margins_before))
            if margins_before
            else 0.0
        ),
        "uni-opd/train/margin_after_mean": (
            float(sum(margins_after) / len(margins_after))
            if margins_after
            else 0.0
        ),
    }
    return UniOPDBatchResult(
        raw_returns=raw_returns,
        calibrated_returns=calibrated_returns,
        keep_mask=keep_mask,
        token_advantages=token_advantages,
        metrics=metrics,
    )


def validate_uni_opd_config(config) -> None:
    """Validate the fixed official Uni-OPD training contract."""

    validate_opd_runtime_config(config)
    if str(config.algorithm.name) != UNI_OPD_VARIANT:
        raise ValueError(
            f"Uni-OPD requires algorithm.name={UNI_OPD_VARIANT}."
        )
    group_name = str(config.get("group_name", ""))
    if group_name not in {UNI_OPD_WANDB_GROUP, UNI_OPD_PUBLICATION_WANDB_GROUP}:
        raise ValueError(
            "Uni-OPD requires group_name to identify either its exploratory or "
            "publication experiment family; got "
            f"{group_name!r}, expected one of "
            f"{UNI_OPD_WANDB_GROUP!r}, {UNI_OPD_PUBLICATION_WANDB_GROUP!r}."
        )

    loss = config.distillation.distillation_loss
    if str(loss.loss_mode) != UNI_OPD_LOSS_MODE:
        raise ValueError(f"Uni-OPD requires loss_mode={UNI_OPD_LOSS_MODE}.")
    if str(loss.policy_loss_mode) != "reinforce":
        raise ValueError("Uni-OPD requires policy_loss_mode=reinforce.")
    if loss.loss_max_clamp is not None:
        raise ValueError(
            "Uni-OPD requires loss_max_clamp=null so the calibrated return is "
            "not clipped."
        )
    if str(config.actor_rollout_ref.actor.loss_agg_mode) != "token-mean":
        raise ValueError(
            "Uni-OPD requires loss_agg_mode=token-mean to implement the stated "
            "mean over retained response tokens."
        )

    settings = config.algorithm.uni_opd
    if not math.isclose(
        float(settings.target_correct_ratio),
        UNI_OPD_TARGET_CORRECT_RATIO,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("Uni-OPD requires target_correct_ratio=0.5.")
    margin_delta = float(settings.margin_delta)
    if not math.isfinite(margin_delta) or margin_delta < 0.0:
        raise ValueError(
            "Uni-OPD margin_delta must be finite and non-negative; "
            f"got {margin_delta}."
        )
    expected_modes = {
        "margin_scope": "group",
        "trajectory_reduce": "mean",
        "margin_direction": "spread",
    }
    for key, expected in expected_modes.items():
        actual = str(settings.get(key, "")).lower()
        if actual != expected:
            raise ValueError(
                f"Uni-OPD requires {key}={expected!r}; got {actual!r}."
            )

    rollout_n = int(config.actor_rollout_ref.rollout.n)
    question_count = int(config.data.train_batch_size)
    if rollout_n != UNI_OPD_ROLLOUTS_PER_PROMPT:
        raise ValueError(
            "Uni-OPD requires exactly 16 Student rollouts per training prompt."
        )
    if question_count != 16:
        raise ValueError(
            "Uni-OPD requires exactly 16 training questions per step."
        )
    if question_count * rollout_n != 256:
        raise ValueError("Uni-OPD requires exactly 256 generated rollouts per step.")


class UniOPDTrainer(BaseOPDTrainer):
    """Controller-side batch calibration plus detached Uni-OPD PG loss."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        validate_uni_opd_config(config)
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
                "Uni-OPD requires 256 genuine rollouts before subsampling; got "
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
                "Uni-OPD requires 16 prompt groups with 16 rollouts each; "
                f"group_count={len(counts)}, mismatched_counts={bad_counts}."
            )

    def _compute_old_log_prob(self, batch, metrics):
        batch = super()._compute_old_log_prob(batch, metrics)
        fields = [
            "uid",
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
        prompt_group_ids = _transfer_queue_prompt_group_ids(data["uid"])
        self._validate_complete_rollout_batch(batch, prompt_group_ids)

        response_mask_nested = data["response_mask"]
        if not response_mask_nested.is_nested:
            raise RuntimeError(
                "Uni-OPD controller expects jagged response masks from TransferQueue."
            )
        response_mask = response_mask_nested.to_padded_tensor(False).bool()
        old_log_probs = data["old_log_probs"]
        if old_log_probs.is_nested:
            old_log_probs = old_log_probs.to_padded_tensor(0.0)
        teacher_log_probs = no_padding_2_padding(
            data["teacher_logprobs"], data
        ).squeeze(-1)
        rewards = data["rm_scores"]
        if rewards.is_nested:
            rewards = rewards.to_padded_tensor(0.0)

        settings = self.config.algorithm.uni_opd
        result = compute_uni_opd_batch(
            student_log_probs=old_log_probs,
            teacher_log_probs=teacher_log_probs,
            response_mask=response_mask,
            verifier_rewards=rewards,
            prompt_group_ids=prompt_group_ids,
            margin_delta=float(settings.margin_delta),
            target_correct_ratio=float(settings.target_correct_ratio),
            seed=int(self.config.trainer.seed) + int(self.global_steps),
        )

        # Keep all 256 generated tensors for stable worker shapes.  Removed
        # trajectories already carry zero advantage.  Scaling the retained
        # numerator by all_tokens / selected_tokens makes the shared token-mean
        # reducer exactly equivalent to physically deleting removed rows,
        # without creating an all-masked worker micro-batch.
        selected_response_mask = response_mask & result.keep_mask.unsqueeze(-1)
        selected_token_count = int(selected_response_mask.sum().item())
        if selected_token_count <= 0:
            raise RuntimeError("Uni-OPD balancing produced no selected tokens.")
        loss_normalization = float(response_mask.sum().item()) / selected_token_count
        output = TensorDict(
            {
                "uni_opd_advantages": response_to_nested(
                    result.token_advantages, response_mask_nested
                ),
                # Store this as a batch-aligned tensor instead of KVBatchMeta
                # extra_info.  Subsequent controller stages write new
                # TransferQueue metadata objects and may replace extra_info,
                # whereas tensor fields survive through actor micro-batching.
                "uni_opd_loss_normalization": torch.full(
                    (len(batch),),
                    loss_normalization,
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
        metrics["uni-opd/train/selected_token_count"] = float(
            selected_token_count
        )
        metrics["uni-opd/train/loss_normalization"] = loss_normalization
        metrics.update(result.metrics)
        return batch

    def algorithm_metric_aliases(self) -> dict[str, str]:
        return {
            "actor/distillation/loss": "uni-opd/train/policy_loss",
        }


__all__ = [
    "UNI_OPD_DEFAULT_MARGIN",
    "UNI_OPD_TARGET_CORRECT_RATIO",
    "UNI_OPD_VARIANT",
    "UniOPDBatchResult",
    "UniOPDTrainer",
    "compute_uni_opd_batch",
    "select_correctness_balanced_rollouts",
    "validate_uni_opd_config",
]
