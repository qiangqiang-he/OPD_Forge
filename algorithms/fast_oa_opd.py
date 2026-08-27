"""Single-forward, branch-masked outcome-aware on-policy distillation."""

from __future__ import annotations

import math

from utils.opd_runtime import BaseOPDTrainer, validate_opd_runtime_config


FAST_OA_OPD_VARIANT = "fast_oa_opd"
FAST_OA_OPD_LOSS_MODE = "oa_opd"
FAST_OA_OPD_WANDB_GROUP = "Fast-OA-OPD"


def validate_fast_oa_opd_config(config) -> None:
    """Validate Fast OA-OPD's masked-probe and shared loss contract."""

    validate_opd_runtime_config(config)
    if str(config.algorithm.name) != FAST_OA_OPD_VARIANT:
        raise ValueError(
            f"Fast OA-OPD requires algorithm.name={FAST_OA_OPD_VARIANT}."
        )
    if str(config.get("student_prompt", "")) != "qwen3_no_thinking_prompt":
        raise ValueError(
            "Fast OA-OPD experiments require the qwen3_no_thinking_prompt Student template."
        )
    if str(config.get("teacher_prompt", "")) != "qwen3_no_thinking_prompt":
        raise ValueError(
            "Fast OA-OPD experiments require the qwen3_no_thinking_prompt Teacher template."
        )
    if str(config.get("student_prompt")) != str(config.get("teacher_prompt")):
        raise ValueError(
            "Fast OA-OPD boundary probes require identical Student and Teacher prompts."
        )
    if str(config.get("group_name", "")) != FAST_OA_OPD_WANDB_GROUP:
        raise ValueError(
            f"Fast OA-OPD requires group_name={FAST_OA_OPD_WANDB_GROUP!r}."
        )

    loss = config.distillation.distillation_loss
    if str(loss.loss_mode) != FAST_OA_OPD_LOSS_MODE:
        raise ValueError(
            f"Fast OA-OPD requires loss_mode={FAST_OA_OPD_LOSS_MODE}."
        )
    if str(loss.policy_loss_mode) != "reinforce":
        raise ValueError("Fast OA-OPD requires policy_loss_mode=reinforce.")
    if float(loss.selection_ratio) != 1.0:
        raise ValueError(
            "Fast OA-OPD uses every token through step weights; set selection_ratio=1.0."
        )
    tau = float(loss.oa_opd_tau)
    beta = float(loss.oa_opd_beta)
    if not math.isfinite(tau):
        raise ValueError(f"Fast OA-OPD tau must be finite, got {tau}.")
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError(
            f"Fast OA-OPD beta must be finite and positive, got {beta}."
        )
    if str(config.actor_rollout_ref.actor.loss_agg_mode) != "seq-mean-token-mean":
        raise ValueError(
            "Fast OA-OPD requires loss_agg_mode=seq-mean-token-mean so inactive "
            "tokens remain in each response's loss denominator."
        )

    inference = config.distillation.teacher_models.teacher_model.inference
    if str(inference.name) != "vllm":
        raise ValueError("Fast OA-OPD masked probes require the vLLM Teacher backend.")
    if not bool(inference.enforce_eager):
        raise ValueError(
            "Fast OA-OPD requires Teacher inference.enforce_eager=true so its "
            "explicit masked forward and the vLLM reference use the same eager "
            "numerical path."
        )
    engine_kwargs = inference.get("engine_kwargs", {}).get("vllm", {}) or {}
    if str(engine_kwargs.get("attention_backend", "")).upper() != "FLASH_ATTN":
        raise ValueError(
            "Fast OA-OPD requires Teacher attention_backend=FLASH_ATTN."
        )
    if not bool(engine_kwargs.get("fast_oa_opd_batch_invariant", False)):
        raise ValueError(
            "Fast OA-OPD requires fast_oa_opd_batch_invariant=true so scores "
            "do not depend on vLLM's dynamic request batching."
        )
    if (
        int(inference.tensor_model_parallel_size) != 1
        or int(inference.data_parallel_size) != 1
        or int(inference.pipeline_model_parallel_size) != 1
    ):
        raise ValueError("Fast OA-OPD masked probes require each Teacher replica to use TP=DP=PP=1.")


class FastOAOPDTrainer(BaseOPDTrainer):
    """Trainer exposing the same 12 OA-OPD intervention metrics."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        validate_fast_oa_opd_config(config)
        super().__init__(*args, **kwargs)

    def _add_opd_training_metrics(self, metrics) -> None:
        super()._add_opd_training_metrics(metrics)
        prefix = "actor/distillation/oa_opd_stats/"
        if not any(key.startswith(prefix) for key in metrics):
            return

        def raw(name: str) -> float:
            return float(metrics.get(f"{prefix}{name}", 0.0))

        for suffix in ("", "correct", "wrong"):
            raw_prefix = f"{suffix}_" if suffix else ""
            metric_suffix = f"_{suffix}" if suffix else ""
            step_count = raw(f"{raw_prefix}step_count")
            token_count = raw(f"{raw_prefix}token_count")
            opd_mass = raw(f"{raw_prefix}opd_adv_mass")
            metrics[f"oa_opd/active_step_ratio{metric_suffix}"] = self._safe_ratio(
                raw(f"{raw_prefix}active_step_count"), step_count
            )
            metrics[f"oa_opd/active_token_ratio{metric_suffix}"] = self._safe_ratio(
                raw(f"{raw_prefix}active_token_count"), token_count
            )
            metrics[f"oa_opd/mean_weight{metric_suffix}"] = self._safe_ratio(
                raw(f"{raw_prefix}weight_sum"), token_count
            )
            metrics[f"oa_opd/adv_retention_ratio{metric_suffix}"] = (
                raw(f"{raw_prefix}oa_adv_mass") / (opd_mass + 1.0e-12)
            )

        for key in [key for key in metrics if key.startswith(prefix)]:
            metrics.pop(key)


__all__ = ["FastOAOPDTrainer", "validate_fast_oa_opd_config"]
