"""Outcome-filtered GRPO/REINFORCE on the Student policy.

Correct-RL keeps only verifier-correct rollouts.  A valid response token in a
correct rollout receives ``(1 - p_t) ** gamma`` where ``p_t`` is the Student's
original token probability and ``gamma`` is fixed to 0.5; incorrect rollouts
receive no policy gradient.  The standard ``seq-mean-token-mean`` reducer then
averages tokens within each trajectory and, with ``global_batch_size`` set to
the global number of correct trajectories, averages the remaining trajectories
equally.

Unlike the OPD algorithms this variant does not use a Teacher or a reference
policy.  It reuses VERL's normal actor/rollout path and its PPO clipped policy
loss (configured with a 0.2 lower and 0.27 upper clipping ratio).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from tensordict import TensorDict

from verl.trainer import main_ppo_sync as verl_sync
from verl.workers.utils.padding import response_to_nested


CORRECT_RL_VARIANT = "correct_rl"
CORRECT_RL_CLIP_RATIO = 0.2
CORRECT_RL_CLIP_RATIO_LOW = 0.2
CORRECT_RL_CLIP_RATIO_HIGH = 0.27
CORRECT_RL_GAMMA = 0.5


@dataclass(frozen=True)
class CorrectRLBatchResult:
    """Controller-computed outcome gate for one complete rollout batch."""

    correct_rollout_mask: torch.Tensor
    token_mask: torch.Tensor
    advantages: torch.Tensor
    trajectory_lengths: torch.Tensor
    correct_count: int
    genuine_count: int
    loss_normalization: float
    metrics: dict[str, float]


def compute_correct_rl_batch(
    rm_scores: torch.Tensor,
    response_mask: torch.Tensor,
    old_log_probs: torch.Tensor,
    *,
    genuine_trajectory_mask: torch.Tensor | None = None,
    gamma: float = CORRECT_RL_GAMMA,
) -> CorrectRLBatchResult:
    """Build the probability-weighted Correct-RL signal.

    A trajectory is correct when its token-level verifier score sums to more
    than ``0.5``.  For each valid token in a correct trajectory,
    ``advantage = (1 - exp(old_log_prob)) ** gamma``; all other positions
    receive zero.  The actor's ``seq-mean-token-mean`` reducer performs the
    per-trajectory length normalization, while the trainer supplies the global
    correct count as its sequence denominator.
    """

    if rm_scores.ndim != 2 or response_mask.ndim != 2 or old_log_probs.ndim != 2:
        raise ValueError(
            "Correct-RL rm_scores, response_mask, and old_log_probs must be 2-D tensors."
        )
    if rm_scores.shape != response_mask.shape or old_log_probs.shape != response_mask.shape:
        raise ValueError(
            "Correct-RL rm_scores, response_mask, and old_log_probs must have identical "
            f"shapes; got {tuple(rm_scores.shape)}, {tuple(response_mask.shape)}, and "
            f"{tuple(old_log_probs.shape)}."
        )
    gamma_value = float(gamma)
    if not math.isfinite(gamma_value) or gamma_value < 0:
        raise ValueError(f"Correct-RL gamma must be finite and non-negative, got {gamma}.")

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
            "Correct-RL genuine_trajectory_mask must have one entry per trajectory."
        )
    else:
        genuine_trajectory_mask = genuine_trajectory_mask.to(
            device=response_mask.device, dtype=torch.bool
        )

    genuine_count = int(genuine_trajectory_mask.sum().item())
    if genuine_count == 0:
        raise ValueError("Correct-RL cannot operate on a batch with no genuine trajectories.")

    genuine_tokens = response_mask & genuine_trajectory_mask.unsqueeze(-1)
    rm_scores_float = rm_scores.float()
    if not bool(torch.isfinite(rm_scores_float[genuine_tokens]).all()):
        raise ValueError("Correct-RL verifier scores must be finite on valid tokens.")
    old_log_probs_float = old_log_probs.float()
    valid_old_log_probs = old_log_probs_float[genuine_tokens]
    if bool(torch.isnan(valid_old_log_probs).any()) or bool(torch.isposinf(valid_old_log_probs).any()):
        raise ValueError("Correct-RL old log probabilities must not contain NaN or +inf.")

    trajectory_lengths = response_mask.sum(dim=-1).to(torch.float32)
    # Verifier scores outside the valid response span (e.g. padding) must not
    # affect correctness.
    safe_rm_scores = torch.where(genuine_tokens, rm_scores_float, torch.zeros_like(rm_scores_float))
    rollout_correct = safe_rm_scores.sum(dim=-1) > 0.5
    correct_rollout_mask = genuine_trajectory_mask & rollout_correct
    token_mask = response_mask & correct_rollout_mask.unsqueeze(-1)
    with torch.no_grad():
        safe_old_log_probs = torch.where(
            genuine_tokens, old_log_probs_float, torch.zeros_like(old_log_probs_float)
        )
        token_probabilities = safe_old_log_probs.exp().clamp_(min=0.0, max=1.0)
        advantages = (
            (1.0 - token_probabilities).clamp_min(0.0).pow(gamma_value) * token_mask
        ).to(dtype=rm_scores.dtype)

    correct_count = int(correct_rollout_mask.sum().item())
    incorrect_count = genuine_count - correct_count
    correct_token_count = int(token_mask.sum().item())
    accuracy = float(correct_count) / float(genuine_count)
    loss_normalization = 1.0 / float(correct_count) if correct_count > 0 else 0.0
    correct_lengths = trajectory_lengths[correct_rollout_mask]
    correct_response_length_mean = (
        float(correct_lengths.mean().item()) if correct_lengths.numel() else 0.0
    )
    correct_token_probability_mean = (
        float(token_probabilities[token_mask].mean().item()) if correct_token_count else 0.0
    )
    correct_advantage_mean = (
        float(advantages[token_mask].float().mean().item()) if correct_token_count else 0.0
    )

    metrics = {
        "correct-rl/train/generated_trajectory_count": float(genuine_count),
        "correct-rl/train/correct_trajectory_count": float(correct_count),
        "correct-rl/train/incorrect_trajectory_count": float(incorrect_count),
        "correct-rl/train/accuracy": accuracy,
        "correct-rl/train/correct_token_count": float(correct_token_count),
        "correct-rl/train/correct_response_length_mean": correct_response_length_mean,
        "correct-rl/train/loss_normalization": loss_normalization,
        "correct-rl/train/token_probability_mean": correct_token_probability_mean,
        "correct-rl/train/advantage_mean": correct_advantage_mean,
        "correct-rl/train/advantage_gamma": gamma_value,
    }
    return CorrectRLBatchResult(
        correct_rollout_mask=correct_rollout_mask,
        token_mask=token_mask,
        advantages=advantages,
        trajectory_lengths=trajectory_lengths,
        correct_count=correct_count,
        genuine_count=genuine_count,
        loss_normalization=loss_normalization,
        metrics=metrics,
    )


def validate_correct_rl_config(config) -> None:
    """Validate the no-Teacher, no-reference Correct-RL contract."""

    if str(config.algorithm.name) != CORRECT_RL_VARIANT:
        raise ValueError(f"Correct-RL requires algorithm.name={CORRECT_RL_VARIANT}.")
    if str(config.get("student_prompt", "")) != "qwen3_no_thinking_prompt":
        raise ValueError("Correct-RL requires the No-Thinking Student prompt.")
    if str(config.get("teacher_prompt", "")) != "qwen3_no_thinking_prompt":
        raise ValueError("Correct-RL requires the No-Thinking Teacher prompt.")

    distillation = config.get("distillation", None)
    if distillation is not None and bool(distillation.get("enabled", False)):
        raise ValueError("Correct-RL must disable distillation; no Teacher is required.")
    if bool(config.algorithm.get("use_kl_in_reward", False)):
        raise ValueError("Correct-RL does not use a reference-policy KL reward.")
    actor = config.actor_rollout_ref.actor
    if bool(actor.get("use_kl_loss", False)):
        raise ValueError("Correct-RL does not use a reference-policy KL loss.")
    if str(config.algorithm.get("adv_estimator", "")) != "grpo":
        raise ValueError("Correct-RL requires algorithm.adv_estimator=grpo.")
    if str(actor.get("loss_agg_mode", "")) != "seq-mean-token-mean":
        raise ValueError(
            "Correct-RL requires actor.loss_agg_mode=seq-mean-token-mean for "
            "trajectory-length normalization."
        )
    clip_requirements = {
        "clip_ratio": CORRECT_RL_CLIP_RATIO,
        "clip_ratio_low": CORRECT_RL_CLIP_RATIO_LOW,
        "clip_ratio_high": CORRECT_RL_CLIP_RATIO_HIGH,
    }
    for field, expected in clip_requirements.items():
        value = float(actor.get(field, expected))
        if abs(value - expected) > 1.0e-12:
            raise ValueError(f"Correct-RL requires actor.{field}={expected}, got {value}.")
    gamma = float(config.algorithm.get("correct_rl_gamma", CORRECT_RL_GAMMA))
    if abs(gamma - CORRECT_RL_GAMMA) > 1.0e-12:
        raise ValueError(
            f"Correct-RL requires algorithm.correct_rl_gamma={CORRECT_RL_GAMMA}, got {gamma}."
        )
    policy_loss = actor.get("policy_loss", {})
    if str(policy_loss.get("loss_mode", "vanilla")) != "vanilla":
        raise ValueError("Correct-RL requires the standard PPO clipped policy loss.")


class CorrectRLTrainer(verl_sync.PPOTrainer):
    """PPO trainer with a correctness gate and two-level loss normalization."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        validate_correct_rl_config(config)
        super().__init__(*args, **kwargs)

    def _compute_advantage(self, batch, metrics):
        """Materialize probability-weighted advantages and the loss mask."""

        data = verl_sync.tq.kv_batch_get(
            keys=batch.keys,
            partition_id=batch.partition_id,
            select_fields=["response_mask", "rm_scores", "old_log_probs"],
        )
        response_mask_nested = data["response_mask"]
        if not response_mask_nested.is_nested:
            raise RuntimeError("Correct-RL expects jagged response masks from TransferQueue.")
        response_mask = response_mask_nested.to_padded_tensor(False).bool()
        rm_scores = data["rm_scores"]
        if rm_scores.is_nested:
            rm_scores = rm_scores.to_padded_tensor(0.0)
        old_log_probs = data["old_log_probs"]
        if old_log_probs.is_nested:
            old_log_probs = old_log_probs.to_padded_tensor(0.0)

        genuine_mask = torch.tensor(
            [not bool(tag.get("is_padding", False)) for tag in batch.tags],
            dtype=torch.bool,
            device=response_mask.device,
        )
        result = compute_correct_rl_batch(
            rm_scores=rm_scores,
            response_mask=response_mask,
            old_log_probs=old_log_probs,
            genuine_trajectory_mask=genuine_mask,
            gamma=float(self.config.algorithm.get("correct_rl_gamma", CORRECT_RL_GAMMA)),
        )

        output = TensorDict(
            {
                "advantages": response_to_nested(result.advantages, response_mask_nested),
                "returns": response_to_nested(result.advantages, response_mask_nested),
                "correct_rl_loss_mask": response_to_nested(
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
        # This metadata is copied to every actor micro-batch.  It must be the
        # complete post-padding batch count, not a local micro-batch count.
        batch.extra_info["correct_rl_correct_count"] = result.correct_count
        metrics.update(result.metrics)
        return batch

    def _update_actor(self, batch, metrics):
        correct_count = int(batch.extra_info.get("correct_rl_correct_count", 0))
        if correct_count <= 0:
            # Avoid invoking the actor reducer with global_batch_size=0.  The
            # step is an intentional no-op and still records stable metrics.
            metrics["actor/grad_norm"] = 0.0
            metrics["actor/pg_loss"] = 0.0
            return batch
        batch.extra_info["correct_rl_correct_count"] = correct_count
        return super()._update_actor(batch, metrics)

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        """Add Correct-RL's stable per-batch W&B metric namespace."""

        super()._compute_metrics(
            batch,
            metrics,
            timing_raw,
            global_steps=global_steps,
            epoch=epoch,
        )
        for source, destination in self.algorithm_metric_aliases().items():
            if source in metrics:
                metrics[destination] = metrics[source]

    def algorithm_metric_aliases(self) -> dict[str, str]:
        return {
            "actor/entropy": "correct-rl/train/entropy",
            "actor/grad_norm": "correct-rl/train/grad_norm",
            "actor/pg_loss": "correct-rl/train/policy_loss",
            "actor/pg_clipfrac": "correct-rl/train/clipfrac",
            "response_length/mean": "correct-rl/train/response_length_mean",
            "response_length/max": "correct-rl/train/response_length_max",
            "response_length/clip_ratio": "correct-rl/train/response_truncated_ratio",
        }


__all__ = [
    "CORRECT_RL_CLIP_RATIO",
    "CORRECT_RL_CLIP_RATIO_HIGH",
    "CORRECT_RL_CLIP_RATIO_LOW",
    "CORRECT_RL_GAMMA",
    "CORRECT_RL_VARIANT",
    "CorrectRLBatchResult",
    "CorrectRLTrainer",
    "compute_correct_rl_batch",
    "validate_correct_rl_config",
]
