"""Calibrated policy-gradient on-policy distillation.

Cal-OPD teacher-forces the same student rollout with a base teacher prompt and
two feedback-conditioned prompts.  It subtracts only the teacher's largest
self-deviation toward the student from the signed teacher-student gap.  The
remaining advantage keeps its original sign and is clipped at zero magnitude.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any

import torch

from utils.opd_runtime import BaseOPDTrainer, validate_opd_runtime_config
from utils.prompts import (
    get_cal_privileged_feedback_prompt_names,
    get_prompt_template,
)
from verl.trainer import main_ppo_sync as verl_sync
from verl.trainer.distillation.losses import compute_calibrated_opd_advantage
from verl.workers.utils.padding import no_padding_2_padding


CAL_OPD_VARIANT = "cal_opd"
CAL_OPD_WANDB_NAMESPACE = "Cal-OPD"
CAL_OPD_TOKEN_TABLE_KEY = f"{CAL_OPD_WANDB_NAMESPACE}/token_changes"
CAL_OPD_TOKEN_TOP_K = 64
CAL_OPD_TOKEN_CANDIDATE_MULTIPLIER = 8
CAL_OPD_TOKEN_TABLE_COLUMNS = (
    "step",
    "category",
    "rank",
    "token",
    "total_change",
    "occurrence_count",
    "mean_change",
    "max_change",
)
CAL_OPD_TOKEN_CATEGORIES = (
    ("absolute", "Absolute"),
    ("positive", "Positive"),
    ("negative", "Negative"),
)


def normalize_cal_opd_token(decoded_token: str) -> tuple[str, str] | None:
    """Return a merge key and display label for a decoded token."""

    normalized = str(decoded_token).strip().lower()
    if not normalized:
        return None
    return normalized, normalized[:1].upper() + normalized[1:]


def _aggregate_token_id_changes(
    token_ids: torch.Tensor,
    changes: torch.Tensor,
    selection: torch.Tensor,
    *,
    candidate_limit: int,
) -> list[dict[str, int | float]]:
    """Aggregate changes by token id and retain only significant candidates."""

    selected_ids = token_ids[selection].long()
    selected_changes = changes[selection].float()
    nonzero = selected_changes > 0
    selected_ids = selected_ids[nonzero]
    selected_changes = selected_changes[nonzero]
    if selected_ids.numel() == 0:
        return []

    unique_ids, inverse = torch.unique(selected_ids, sorted=False, return_inverse=True)
    total_changes = torch.zeros(
        unique_ids.numel(), dtype=torch.float32, device=selected_changes.device
    )
    total_changes.scatter_add_(0, inverse, selected_changes)
    occurrence_counts = torch.zeros(
        unique_ids.numel(), dtype=torch.int64, device=selected_changes.device
    )
    occurrence_counts.scatter_add_(
        0, inverse, torch.ones_like(inverse, dtype=torch.int64)
    )
    max_changes = torch.zeros(
        unique_ids.numel(), dtype=torch.float32, device=selected_changes.device
    )
    max_changes.scatter_reduce_(
        0, inverse, selected_changes, reduce="amax", include_self=True
    )

    candidate_count = min(int(candidate_limit), unique_ids.numel())
    candidate_indices = torch.topk(
        total_changes, k=candidate_count, largest=True, sorted=True
    ).indices
    return [
        {
            "token_id": int(unique_ids[index].item()),
            "total_change": float(total_changes[index].item()),
            "occurrence_count": int(occurrence_counts[index].item()),
            "max_change": float(max_changes[index].item()),
        }
        for index in candidate_indices
    ]


def build_cal_opd_token_change_rows(
    *,
    response_token_ids: torch.Tensor,
    response_mask: torch.Tensor,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    positive_teacher_log_probs: torch.Tensor,
    negative_teacher_log_probs: torch.Tensor,
    tokenizer,
    step: int,
    top_k: int = CAL_OPD_TOKEN_TOP_K,
    candidate_multiplier: int = CAL_OPD_TOKEN_CANDIDATE_MULTIPLIER,
) -> list[list[Any]]:
    """Build the three normalized top-token tables for one training step.

    Ranking uses ``|A_OPD| - |A_Cal|``.  Changes are first aggregated by token
    id, and only a small significant candidate pool is decoded.  Decoded pieces
    are then stripped, lower-cased, merged, and displayed with an initial
    capital letter.
    """

    tensors = (
        response_token_ids,
        response_mask,
        student_log_probs,
        teacher_log_probs,
        positive_teacher_log_probs,
        negative_teacher_log_probs,
    )
    if any(tensor.ndim != 2 for tensor in tensors):
        raise ValueError("Cal-OPD token-table tensors must all be two-dimensional.")
    if len({tensor.shape for tensor in tensors}) != 1:
        raise ValueError("Cal-OPD token-table tensors must have identical shapes.")
    if top_k <= 0 or candidate_multiplier <= 0:
        raise ValueError("Cal-OPD token-table limits must be positive.")

    with torch.no_grad():
        calibrated_advantage, _ = compute_calibrated_opd_advantage(
            student_log_probs,
            teacher_log_probs,
            positive_teacher_log_probs,
            negative_teacher_log_probs,
        )
        base_advantage = teacher_log_probs - student_log_probs
        changes = base_advantage.abs() - calibrated_advantage.abs()
        valid = response_mask.bool()
        selections = {
            "absolute": valid,
            "positive": valid & base_advantage.gt(0),
            "negative": valid & base_advantage.lt(0),
        }
        candidate_limit = top_k * candidate_multiplier
        candidates = {
            category: _aggregate_token_id_changes(
                response_token_ids,
                changes,
                selections[category],
                candidate_limit=candidate_limit,
            )
            for category, _ in CAL_OPD_TOKEN_CATEGORIES
        }

    candidate_ids = sorted(
        {
            int(candidate["token_id"])
            for category_candidates in candidates.values()
            for candidate in category_candidates
        }
    )
    if not candidate_ids:
        return []
    decode_inputs = [[token_id] for token_id in candidate_ids]
    try:
        decoded_tokens = tokenizer.batch_decode(
            decode_inputs,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        decoded_tokens = tokenizer.batch_decode(
            decode_inputs,
            skip_special_tokens=True,
        )
    decoded_by_id = dict(zip(candidate_ids, decoded_tokens, strict=True))

    rows: list[list[Any]] = []
    for category, category_label in CAL_OPD_TOKEN_CATEGORIES:
        merged: dict[str, dict[str, int | float | str]] = defaultdict(
            lambda: {
                "display": "",
                "total_change": 0.0,
                "occurrence_count": 0,
                "max_change": 0.0,
            }
        )
        for candidate in candidates[category]:
            normalized_token = normalize_cal_opd_token(
                decoded_by_id[int(candidate["token_id"])]
            )
            if normalized_token is None:
                continue
            merge_key, display = normalized_token
            aggregate = merged[merge_key]
            aggregate["display"] = display
            aggregate["total_change"] = float(aggregate["total_change"]) + float(
                candidate["total_change"]
            )
            aggregate["occurrence_count"] = int(aggregate["occurrence_count"]) + int(
                candidate["occurrence_count"]
            )
            aggregate["max_change"] = max(
                float(aggregate["max_change"]), float(candidate["max_change"])
            )

        ranked = sorted(
            merged.values(),
            key=lambda item: (
                -float(item["total_change"]),
                -float(item["max_change"]),
                str(item["display"]),
            ),
        )[:top_k]
        for rank, item in enumerate(ranked, start=1):
            count = int(item["occurrence_count"])
            total_change = float(item["total_change"])
            rows.append(
                [
                    int(step),
                    category_label,
                    rank,
                    str(item["display"]),
                    total_change,
                    count,
                    total_change / count,
                    float(item["max_change"]),
                ]
            )
    return rows


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

    student_prompt = str(config.student_prompt)
    teacher_prompt = str(config.teacher_prompt)
    if student_prompt != teacher_prompt:
        raise ValueError(
            "Cal-OPD requires student and teacher to use the same thinking mode; "
            f"got student_prompt={student_prompt!r} and teacher_prompt={teacher_prompt!r}."
        )
    positive_prompt, negative_prompt = get_cal_privileged_feedback_prompt_names(
        teacher_prompt
    )
    get_prompt_template(positive_prompt)
    get_prompt_template(negative_prompt)


class CalOPDTrainer(BaseOPDTrainer):
    """Trainer using the signed, non-flipping calibrated advantage."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        validate_cal_opd_config(config)
        super().__init__(*args, **kwargs)

    def algorithm_metric_aliases(self) -> dict[str, str]:
        prefix = f"{CAL_OPD_WANDB_NAMESPACE}/train"
        return {
            "actor/distillation/cal_advantage_mean": f"{prefix}/advantage_mean",
            "actor/distillation/cal_advantage_abs_mean": f"{prefix}/advantage_abs_mean",
            "actor/distillation/cal_zero_token_ratio": f"{prefix}/zero_token_ratio",
            "actor/distillation/cal_teacher_self_deviation_mean": (
                f"{prefix}/teacher_self_deviation_mean"
            ),
            "actor/distillation/cal_retained_magnitude_ratio": (
                f"{prefix}/retained_magnitude_ratio"
            ),
            "actor/distillation/cal_positive_retained_magnitude_ratio": (
                f"{prefix}/positive_retained_magnitude_ratio"
            ),
            "actor/distillation/cal_negative_retained_magnitude_ratio": (
                f"{prefix}/negative_retained_magnitude_ratio"
            ),
            "actor/distillation/loss": f"{prefix}/policy_loss",
        }

    def _add_opd_training_metrics(self, metrics) -> None:
        super()._add_opd_training_metrics(metrics)
        # Cal-specific source names are implementation details.  Keep a single
        # public W&B namespace instead of creating duplicate actor panels.
        for source in self.algorithm_metric_aliases():
            metrics.pop(source, None)

    def _wandb_backend_for_step(self, step: int):
        tracking = getattr(self, "logger", None)
        backends = getattr(tracking, "logger", {})
        wandb = backends.get("wandb")
        if wandb is None or getattr(wandb, "run", None) is None:
            return None
        skip_until = os.environ.get("RLVR_WANDB_SKIP_UNTIL_STEP")
        if skip_until is not None and int(step) <= int(skip_until):
            return None
        return wandb

    @staticmethod
    def _to_padded(tensor: torch.Tensor, padding_value: int | float) -> torch.Tensor:
        if tensor.is_nested:
            return tensor.to_padded_tensor(padding_value)
        return tensor

    def _build_token_change_rows(self, batch, step: int) -> list[list[Any]]:
        fields = [
            "prompts",
            "responses",
            "response_mask",
            "old_log_probs",
            "teacher_logprobs",
            "cal_positive_teacher_logprobs",
            "cal_negative_teacher_logprobs",
        ]
        data = verl_sync.tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=fields,
        )
        response_token_ids = self._to_padded(
            data["responses"], self.tokenizer.pad_token_id
        )
        response_mask = self._to_padded(data["response_mask"], 0).bool()
        student_log_probs = self._to_padded(data["old_log_probs"], 0.0)
        teacher_log_probs = no_padding_2_padding(
            data["teacher_logprobs"], data
        ).squeeze(-1)
        positive_teacher_log_probs = no_padding_2_padding(
            data["cal_positive_teacher_logprobs"], data
        ).squeeze(-1)
        negative_teacher_log_probs = no_padding_2_padding(
            data["cal_negative_teacher_logprobs"], data
        ).squeeze(-1)

        non_padding_rows = torch.tensor(
            [not tag.get("is_padding", False) for tag in batch.tags],
            dtype=torch.bool,
            device=response_mask.device,
        )
        response_mask &= non_padding_rows.unsqueeze(-1)
        return build_cal_opd_token_change_rows(
            response_token_ids=response_token_ids.detach().cpu(),
            response_mask=response_mask.detach().cpu(),
            student_log_probs=student_log_probs.detach().float().cpu(),
            teacher_log_probs=teacher_log_probs.detach().float().cpu(),
            positive_teacher_log_probs=positive_teacher_log_probs.detach().float().cpu(),
            negative_teacher_log_probs=negative_teacher_log_probs.detach().float().cpu(),
            tokenizer=self.tokenizer,
            step=step,
        )

    def _log_token_change_rows(self, rows: list[list[Any]], step: int, wandb) -> None:
        previous_table = getattr(self, "_cal_opd_token_change_table", None)
        previous_rows = previous_table.data if previous_table is not None else None
        table = wandb.Table(
            columns=list(CAL_OPD_TOKEN_TABLE_COLUMNS),
            data=previous_rows,
        )
        for row in rows:
            table.add_data(*row)
        wandb.log({CAL_OPD_TOKEN_TABLE_KEY: table}, step=step)
        self._cal_opd_token_change_table = table

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)
        wandb = self._wandb_backend_for_step(global_steps)
        if wandb is None:
            return
        rows = self._build_token_change_rows(batch, global_steps)
        self._log_token_change_rows(rows, global_steps, wandb)
