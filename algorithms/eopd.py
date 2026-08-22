"""Entropy-gated on-policy distillation (EOPD)."""

from __future__ import annotations

import math

from utils.opd_runtime import BaseOPDTrainer, validate_opd_runtime_config


EOPD_VARIANT = "eopd"
EOPD_LOSS_MODE = "eopd"
EOPD_WANDB_GROUP = "EOPD"


def validate_eopd_config(config) -> None:
    """Validate EOPD's objective, entropy source, and training backend."""

    validate_opd_runtime_config(config)
    if str(config.algorithm.name) != EOPD_VARIANT:
        raise ValueError(f"EOPD requires algorithm.name={EOPD_VARIANT}.")

    loss = config.distillation.distillation_loss
    if str(loss.loss_mode) != EOPD_LOSS_MODE:
        raise ValueError(f"EOPD requires loss_mode={EOPD_LOSS_MODE}.")
    if str(loss.policy_loss_mode) != "reinforce":
        raise ValueError("EOPD requires policy_loss_mode=reinforce for standard OPD.")
    if str(config.get("group_name", "")) != EOPD_WANDB_GROUP:
        raise ValueError(
            f"EOPD requires group_name={EOPD_WANDB_GROUP!r} so W&B runs stay "
            "separate from other OPD variants."
        )
    if int(loss.topk) <= 0:
        raise ValueError(f"EOPD topk must be positive, got {loss.topk}.")
    entropy_threshold = float(loss.eopd_entropy_threshold)
    if not math.isfinite(entropy_threshold) or entropy_threshold < 0:
        raise ValueError(
            "EOPD entropy threshold must be finite and non-negative, got "
            f"{entropy_threshold}."
        )
    alpha = float(loss.eopd_alpha)
    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError(f"EOPD alpha must be finite and non-negative, got {alpha}.")

    if str(config.actor_rollout_ref.actor.strategy) != "fsdp":
        raise ValueError("EOPD's combined OPD + FKL path currently requires FSDP.")
    if float(config.actor_rollout_ref.actor.entropy_coeff) != 0.0:
        raise ValueError(
            "EOPD's chunked Student projection does not compute Student entropy; "
            "set actor.entropy_coeff=0."
        )
    teacher = config.distillation.teacher_models.teacher_model.inference
    if str(teacher.name) != "vllm":
        raise ValueError("EOPD full-vocabulary Teacher entropy currently requires vLLM.")
    max_logprobs = teacher.engine_kwargs.get("vllm", {}).get("max_logprobs")
    if max_logprobs is None or int(max_logprobs) < int(loss.topk) + 1:
        raise ValueError(
            "EOPD requires vLLM max_logprobs >= topk + 1 for its bounded "
            "Top-k plus entropy output."
        )
    eopd_entropy_topk = teacher.engine_kwargs.get("vllm", {}).get(
        "eopd_entropy_topk"
    )
    if eopd_entropy_topk is None or int(eopd_entropy_topk) != int(loss.topk):
        raise ValueError(
            "EOPD requires vLLM eopd_entropy_topk to match distillation topk."
        )


class EOPDTrainer(BaseOPDTrainer):
    """Trainer implementing standard OPD plus entropy-gated Top-k FKL."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        validate_eopd_config(config)
        super().__init__(*args, **kwargs)

    def algorithm_metric_aliases(self) -> dict[str, str]:
        prefix = "eopd/train"
        return {
            "actor/distillation/reverse_kl_estimate": (
                f"{prefix}/reverse_kl_estimate"
            ),
            "actor/distillation/eopd_teacher_entropy": (
                f"{prefix}/teacher_entropy"
            ),
            "actor/distillation/eopd_high_entropy_token_ratio": (
                f"{prefix}/high_entropy_token_ratio"
            ),
            "actor/distillation/eopd_forward_kl_per_token": (
                f"{prefix}/forward_kl_per_token"
            ),
            "actor/distillation/eopd_forward_kl_loss": (
                f"{prefix}/forward_kl_loss"
            ),
            "actor/distillation/eopd_scaled_forward_kl_loss": (
                f"{prefix}/scaled_forward_kl_loss"
            ),
            "actor/distillation/loss": f"{prefix}/total_loss",
        }
