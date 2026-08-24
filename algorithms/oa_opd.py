"""Outcome-aware on-policy distillation for semantic reasoning steps."""

from __future__ import annotations

import math

from utils.opd_runtime import BaseOPDTrainer, validate_opd_runtime_config


OA_OPD_VARIANT = "oa_opd"
OA_OPD_LOSS_MODE = "oa_opd"
OA_OPD_WANDB_GROUP = "OA-OPD"


def validate_oa_opd_config(config) -> None:
    """Validate OA-OPD's no-thinking probe and loss contract."""

    validate_opd_runtime_config(config)
    if str(config.algorithm.name) != OA_OPD_VARIANT:
        raise ValueError(f"OA-OPD requires algorithm.name={OA_OPD_VARIANT}.")
    if str(config.get("student_prompt", "")) != "qwen3_no_thinking_prompt":
        raise ValueError("OA-OPD experiments require the qwen3_no_thinking_prompt Student template.")
    if str(config.get("teacher_prompt", "")) != "qwen3_no_thinking_prompt":
        raise ValueError("OA-OPD experiments require the qwen3_no_thinking_prompt Teacher template.")
    if str(config.get("student_prompt")) != str(config.get("teacher_prompt")):
        raise ValueError("OA-OPD boundary probes require identical Student and Teacher prompts.")
    if str(config.get("group_name", "")) != OA_OPD_WANDB_GROUP:
        raise ValueError(
            f"OA-OPD requires group_name={OA_OPD_WANDB_GROUP!r} so runs use a stable namespace."
        )

    loss = config.distillation.distillation_loss
    if str(loss.loss_mode) != OA_OPD_LOSS_MODE:
        raise ValueError(f"OA-OPD requires loss_mode={OA_OPD_LOSS_MODE}.")
    if str(loss.policy_loss_mode) != "reinforce":
        raise ValueError("OA-OPD requires policy_loss_mode=reinforce.")
    if float(loss.selection_ratio) != 1.0:
        raise ValueError("OA-OPD uses every token through step weights; set selection_ratio=1.0.")
    tau = float(loss.oa_opd_tau)
    beta = float(loss.oa_opd_beta)
    probe_batch_size = int(loss.oa_opd_probe_batch_size)
    if not math.isfinite(tau):
        raise ValueError(f"OA-OPD tau must be finite, got {tau}.")
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError(f"OA-OPD beta must be finite and positive, got {beta}.")
    if not 1 <= probe_batch_size <= 8:
        raise ValueError(
            f"OA-OPD probe batch size must lie in [1, 8], got {probe_batch_size}."
        )
    if str(config.actor_rollout_ref.actor.loss_agg_mode) != "seq-mean-token-mean":
        raise ValueError(
            "OA-OPD requires loss_agg_mode=seq-mean-token-mean so inactive tokens "
            "remain in each response's loss denominator."
        )
    if str(config.distillation.teacher_models.teacher_model.inference.name) != "vllm":
        raise ValueError("OA-OPD answer probes currently require the vLLM Teacher backend.")


class OAOPDTrainer(BaseOPDTrainer):
    """Trainer exposing globally aggregated OA-OPD intervention metrics."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        validate_oa_opd_config(config)
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

        # Sufficient statistics (including probe failure counts) are internal;
        # the public OA-OPD namespace intentionally contains exactly 12 metrics.
        for key in [key for key in metrics if key.startswith(prefix)]:
            metrics.pop(key)


__all__ = ["OAOPDTrainer", "validate_oa_opd_config"]
