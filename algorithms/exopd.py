"""ExOPD with a frozen initial-Student reference on every Student GPU.

For tokens sampled by the current Student, ExOPD uses

    A = lambda * (log pi_T - log pi_ref) - (log pi_S - log pi_ref)
    L = -E[stop_gradient(A) * log pi_S].

The reference policy is VERL's colocated forward-only reference worker.  The
configuration contract below requires one-element FSDP shard groups and no
parameter offload, so every Student data-parallel rank owns a complete current
Student and a complete frozen initial-Student reference on the same GPU.
"""

from __future__ import annotations

import math

from utils.opd_runtime import BaseOPDTrainer, validate_opd_runtime_config


EXOPD_VARIANT = "exopd"
EXOPD_LOSS_MODE = "exopd_reverse_kl"
EXOPD_DEFAULT_LAMBDA = 1.25
EXOPD_WANDB_GROUP = "ExOPD"
EXOPD_PUBLICATION_WANDB_GROUP = "PUB_ExOPD_Thinking"


def _model_lora_rank(model) -> int:
    rank = int(model.get("lora", {}).get("rank", 0))
    return rank if rank > 0 else int(model.get("lora_rank", 0))


def validate_exopd_config(config) -> None:
    """Validate the objective and colocated full-replica layout for ExOPD."""

    validate_opd_runtime_config(config)
    if str(config.algorithm.name) != EXOPD_VARIANT:
        raise ValueError(f"ExOPD requires algorithm.name={EXOPD_VARIANT}.")

    loss = config.distillation.distillation_loss
    if str(loss.loss_mode) != EXOPD_LOSS_MODE:
        raise ValueError(f"ExOPD requires loss_mode={EXOPD_LOSS_MODE}.")
    if str(loss.policy_loss_mode) != "reinforce":
        raise ValueError("ExOPD requires policy_loss_mode=reinforce.")
    if str(config.actor_rollout_ref.actor.loss_agg_mode) != "token-mean":
        raise ValueError(
            "ExOPD requires loss_agg_mode=token-mean so response tokens retain "
            "their relative weighting without per-sequence length normalization."
        )
    if loss.loss_max_clamp is not None:
        raise ValueError(
            "ExOPD requires loss_max_clamp=null to preserve the exact token-level "
            "advantage."
        )
    exopd_lambda = float(loss.get("exopd_lambda", EXOPD_DEFAULT_LAMBDA))
    if not math.isfinite(exopd_lambda) or exopd_lambda < 0:
        raise ValueError(
            "ExOPD requires a finite, non-negative exopd_lambda; got "
            f"{exopd_lambda}."
        )

    group_name = str(config.get("group_name", ""))
    if group_name not in {EXOPD_WANDB_GROUP, EXOPD_PUBLICATION_WANDB_GROUP}:
        raise ValueError(
            "ExOPD requires group_name to identify either its exploratory or "
            "publication experiment family; got "
            f"{group_name!r}, expected one of "
            f"{EXOPD_WANDB_GROUP!r}, {EXOPD_PUBLICATION_WANDB_GROUP!r}."
        )
    if bool(config.algorithm.get("use_kl_in_reward", False)):
        raise ValueError("ExOPD uses its reference only in A_ExOPD; disable KL reward.")
    if bool(config.actor_rollout_ref.actor.use_kl_loss):
        raise ValueError(
            "ExOPD uses its reference only in A_ExOPD; disable actor KL loss."
        )

    model = config.actor_rollout_ref.model
    if _model_lora_rank(model) > 0 or model.get("lora_adapter_path") is not None:
        raise ValueError(
            "ExOPD requires a separate full initial-Student reference; LoRA's "
            "in-actor base reference is not supported."
        )

    actor = config.actor_rollout_ref.actor
    reference = config.actor_rollout_ref.ref
    if str(actor.strategy) != "fsdp" or str(reference.strategy) != "fsdp":
        raise ValueError("ExOPD full per-GPU replicas currently require FSDP.")
    if int(actor.fsdp_config.fsdp_size) != 1:
        raise ValueError(
            "ExOPD requires actor.fsdp_config.fsdp_size=1 so every Student GPU "
            "holds the complete current Student."
        )
    if int(reference.fsdp_config.fsdp_size) != 1:
        raise ValueError(
            "ExOPD requires ref.fsdp_config.fsdp_size=1 so every Student GPU "
            "holds the complete initial-Student reference."
        )
    if bool(actor.fsdp_config.param_offload):
        raise ValueError(
            "ExOPD requires the current Student parameters to remain on GPU."
        )
    if bool(reference.fsdp_config.param_offload):
        raise ValueError("ExOPD requires the Reference parameters to remain on GPU.")
    if not bool(reference.fsdp_config.forward_only):
        raise ValueError("ExOPD Reference must be forward-only and frozen.")
    if not bool(reference.fsdp_config.forward_only_keep_on_device):
        raise ValueError("ExOPD Reference must remain resident on its Student GPU.")


class ExOPDTrainer(BaseOPDTrainer):
    """Trainer implementing detached-advantage ExOPD."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        validate_exopd_config(config)
        super().__init__(*args, **kwargs)

    def algorithm_metric_aliases(self) -> dict[str, str]:
        prefix = "ExOPD/train"
        return {
            "actor/distillation/exopd_advantage_mean": f"{prefix}/advantage_mean",
            "actor/distillation/exopd_advantage_abs_mean": (
                f"{prefix}/advantage_abs_mean"
            ),
            "actor/distillation/exopd_reference_sampled_token_prob": (
                f"{prefix}/reference_sampled_token_prob_mean"
            ),
            "actor/distillation/loss": f"{prefix}/policy_loss",
        }
