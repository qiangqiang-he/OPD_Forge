# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from uuid import uuid4
from typing import Any, Callable, Optional

import torch
from tensordict import TensorDict

from verl.base_config import BaseConfig
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.workers.config import ActorConfig, DistillationConfig, DistillationLossConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import no_padding_2_padding

DistillationLossFn = Callable[
    [
        ActorConfig,  # actor_config
        DistillationConfig,  # distillation_config
        dict,  # model_output
        TensorDict,  # micro batch input
    ],
    tuple[torch.Tensor, dict[str, Any]],
]


def is_distillation_enabled(config: Optional[DistillationConfig]) -> bool:
    """Check if distillation is enabled based on the provided configuration."""
    if config is None:
        return False
    return config.enabled


@dataclass
class DistillationLossSettings(BaseConfig):
    """
    Settings for a distillation loss function to be registered.

    Args:
        names (str | list[str]): Name(s) to register the distillation loss function under.
        use_topk (bool): Whether the loss function uses top-k log probabilities.
        use_estimator (bool): Whether the loss function uses single-sample KL estimators.
    """

    names: str | list[str] = field(default_factory=list)
    use_topk: bool = False
    use_estimator: bool = False

    _mutable_fields = {"names"}

    def __post_init__(self):
        self.names = [self.names] if isinstance(self.names, str) else self.names
        if sum([self.use_topk, self.use_estimator]) != 1:
            raise ValueError(
                f"Expected only one of use_estimator, use_topk, but got {self.use_estimator=}, {self.use_topk=}."
            )


DISTILLATION_LOSS_REGISTRY: dict[str, DistillationLossFn] = {}
DISTILLATION_SETTINGS_REGISTRY: dict[str, DistillationLossSettings] = {}


def register_distillation_loss(
    loss_settings: DistillationLossSettings,
) -> Callable[[DistillationLossFn], DistillationLossFn]:
    """Register a distillation loss function with the given name."""

    def decorator(func: DistillationLossFn) -> DistillationLossFn:
        for name in loss_settings.names:
            if name in DISTILLATION_LOSS_REGISTRY:
                raise ValueError(f"Distillation loss function with name '{name}' is already registered.")
            DISTILLATION_LOSS_REGISTRY[name] = func
            DISTILLATION_SETTINGS_REGISTRY[name] = loss_settings
        return func

    return decorator


def get_distillation_loss_fn(loss_name: str) -> DistillationLossFn:
    """Get the distillation loss function with a given name."""
    if loss_name not in DISTILLATION_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_LOSS_REGISTRY.keys())}"
        )
    return DISTILLATION_LOSS_REGISTRY[loss_name]


def get_distillation_loss_settings(loss_name: str) -> DistillationLossSettings:
    """Get the distillation loss settings with a given name."""
    if loss_name not in DISTILLATION_SETTINGS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(DISTILLATION_SETTINGS_REGISTRY.keys())}"
        )
    return DISTILLATION_SETTINGS_REGISTRY[loss_name]


def compute_distillation_loss_range(
    distillation_losses: torch.Tensor, response_mask: torch.Tensor
) -> dict[str, Metric]:
    """Compute min and max distillation loss over valid response tokens."""
    if response_mask.is_nested:
        distillation_losses_response = distillation_losses[response_mask.bool().to_padded_tensor(False)]
    else:
        distillation_losses_response = distillation_losses[response_mask.bool()]
    return {
        "distillation/loss_min": Metric(AggregationType.MIN, distillation_losses_response.min()),
        "distillation/loss_max": Metric(AggregationType.MAX, distillation_losses_response.max()),
    }


def compute_topk_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    data: TensorDict,
    student_logits: torch.Tensor,
    data_format: str,
) -> torch.Tensor:
    """Compute the topk loss in logit processor.

    Returns:
    - distillation_losses: (bsz, seqlen/cp_size)
    - student_mass: (bsz, seqlen/cp_size)
    - teacher_mass: (bsz, seqlen/cp_size)
    """
    match config.strategy:
        # VeOmni uses FSDP2 internally, so its loss computation is identical to FSDP.
        case "fsdp" | "veomni":
            import verl.trainer.distillation.fsdp.losses as fsdp_losses

            distillation_loss_fn = fsdp_losses.compute_forward_kl_topk
        case "megatron":
            import verl.trainer.distillation.megatron.losses as megatron_losses

            distillation_loss_fn = megatron_losses.compute_forward_kl_topk
        case _:
            raise NotImplementedError(f"Unsupported strategy: {config.strategy=}")

    outputs = distillation_loss_fn(
        student_logits=student_logits,
        teacher_topk_log_probs=data["teacher_logprobs"],
        teacher_topk_ids=data["teacher_ids"],
        config=distillation_config,
        data_format=data_format,
    )

    expected_shape = student_logits.shape[:2]
    for k, v in outputs.items():
        assert v.shape == expected_shape, f"Expected shape {expected_shape}, but got {v.shape} for {k=}."

    return outputs


def distillation_ppo_loss(
    config: ActorConfig,
    distillation_config: Optional[DistillationConfig],
    model_output: dict = None,
    data: TensorDict = None,
    dp_group=None,
    student_logits: torch.Tensor = None,
    data_format: str = "thd",
):
    """Loss function used both for logit processor and final policy loss.
    - student_logits is not None, compute the topk loss in logit processor.
    - student_logits is None, compute final policy loss.

    [split sequence across sp/cp groups]
                   |
    [model forward and output logits: (bsz, seqlen/cp_size, vocab_size/tp_size)]
                   |
    [logits processor compute topk loss: (bsz, seqlen/cp_size)]
                   |
    [all gather topk loss across sp/cp groups: (bsz, seqlen)]
                   |
    [combine topk loss with policy loss]

    Args:
        config: Actor configuration.
        distillation_config: Distillation configuration.
        model_output: Model output, including log_probs, entropy.
        data: Micro input batch, contains
          - teacher_logprobs: (bsz, seqlen, topk)
          - teacher_ids: (bsz, seqlen, topk)
        student_logits: (bsz, seqlen/cp_size, vocab_size/tp_size).
        data_format: "thd" or "bshd", models not support THD format, e.g GPT-OSS, Qwen3.5

    Returns:
    - student_logits is not None, return the topk loss tensor (bsz, seqlen/cp_size).
    - student_logits is None, return the final policy loss scalar and metrics.
    """

    # Called as logits processor
    if student_logits is not None:
        return compute_topk_loss(config, distillation_config, data, student_logits, data_format)

    # Called as final policy loss
    distillation_loss_config = distillation_config.distillation_loss
    distill_loss, distill_metrics = distillation_loss(config, distillation_config, model_output, data)
    if distillation_loss_config.use_task_rewards:
        policy_loss, policy_metrics = ppo_loss(config, model_output, data, dp_group)
    else:
        # Direct OPD does not consume sampled-action PPO log-probabilities.
        # Do not build a second, unused full-vocabulary autograd branch only to
        # discard its scalar loss immediately afterwards.
        policy_loss = 0.0
        policy_metrics = {}

    # Combine distillation with policy loss
    policy_metrics.update(distill_metrics)
    distillation_loss_coef = (
        distillation_loss_config.distillation_loss_coef if distillation_loss_config.use_task_rewards else 1.0
    )
    policy_loss += distill_loss * distillation_loss_coef
    policy_metrics["distillation/loss"] = Metric(value=distill_loss, aggregation=AggregationType.SUM)

    return policy_loss, policy_metrics


def distillation_loss(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics.

    Returns:
    - distillation_loss: Aggregated distillation loss scalar.
    - distillation_metrics: Dictionary of metrics.
    """
    assert distillation_config is not None
    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_loss_fn = get_distillation_loss_fn(loss_config.loss_mode)
    distillation_losses, distillation_metrics = distillation_loss_fn(
        config=config,
        distillation_config=distillation_config,
        model_output=model_output,
        data=data,
    )
    response_mask = data["response_mask"]
    loss_agg_mode = config.loss_agg_mode

    distillation_metrics.update(
        compute_distillation_loss_range(distillation_losses=distillation_losses, response_mask=response_mask)
    )
    if loss_config.loss_max_clamp is not None:
        # clamping min is for k1 loss which can be negative
        distillation_losses = distillation_losses.clamp(min=-loss_config.loss_max_clamp, max=loss_config.loss_max_clamp)

    if loss_config.use_policy_gradient:
        # Use negative distillation loss as reward, as done by https://thinkingmachines.ai/blog/on-policy-distillation/.
        policy_loss_fn = get_policy_loss_fn(loss_config.policy_loss_mode)
        for k, v in config.global_batch_info.items():
            loss_config.global_batch_info[k] = v
        log_prob = no_padding_2_padding(model_output["log_probs"], data)
        old_log_prob = data["old_log_probs"]
        if old_log_prob.is_nested:
            old_log_prob = data["old_log_probs"].to_padded_tensor(0.0)
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        rollout_is_weights = data.get("rollout_is_weights", None)
        distillation_loss, pg_metrics = policy_loss_fn(
            old_log_prob=old_log_prob,
            log_prob=log_prob,
            advantages=-distillation_losses.detach(),
            response_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            config=loss_config,
            rollout_is_weights=rollout_is_weights,
        )
        pg_metrics = {f"distillation/{k[len('actor/') :]}": v for k, v in pg_metrics.items()}
        distillation_metrics.update(pg_metrics)
    else:
        # Directly backpropagate distillation loss as a supervised loss, as in https://arxiv.org/abs/2306.13649.
        if response_mask.is_nested:
            response_mask = response_mask.to_padded_tensor(False)
        distillation_loss = agg_loss(
            loss_mat=distillation_losses,
            loss_mask=response_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )

    return distillation_loss, distillation_metrics


@register_distillation_loss(DistillationLossSettings(names=["forward_kl_topk"], use_topk=True))  # type: ignore[arg-type]
def compute_forward_kl_topk(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute forward KL distillation loss and related metrics using top-k log probabilities.

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    # topk loss has been computed in logits processor
    distillation_losses = no_padding_2_padding(model_output["distillation_losses"], data)
    student_mass = no_padding_2_padding(model_output["student_mass"], data)
    teacher_mass = no_padding_2_padding(model_output["teacher_mass"], data)
    overlap_count = model_output.get("overlap_count")
    overlap_token_advantage = model_output.get("overlap_token_advantage")
    if overlap_count is not None and overlap_token_advantage is not None:
        overlap_count = no_padding_2_padding(overlap_count, data)
        overlap_token_advantage = no_padding_2_padding(overlap_token_advantage, data)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert distillation_losses.shape == student_mass.shape == teacher_mass.shape == response_mask_bool.shape

    overlap_metrics = {}
    if overlap_count is not None and overlap_token_advantage is not None:
        assert overlap_count.shape == overlap_token_advantage.shape == response_mask_bool.shape
        valid_overlap_count = overlap_count[response_mask_bool]
        k = distillation_config.distillation_loss.topk
        assert k is not None
        # Diagnostics for tracking teacher/student top-k overlap in OPD, following
        # "Rethinking On-Policy Distillation of Large Language Models" (arXiv:2604.13016):
        # overlap ratio and average teacher-token KL contribution on overlapped tokens.
        overlap_metrics["distillation/overlap_ratio"] = (valid_overlap_count.float().mean() / k).item()
        overlap_position_mask = response_mask_bool & (overlap_count > 0)
        if overlap_position_mask.any():
            overlap_metrics["distillation/overlap_token_advantage"] = (
                overlap_token_advantage[overlap_position_mask].mean().item()
            )
        else:
            overlap_metrics["distillation/overlap_token_advantage"] = 0.0

    # Log amount of mass in the top-k log probabilities for both student and teacher.
    student_mass = student_mass[response_mask_bool]
    teacher_mass = teacher_mass[response_mask_bool]
    distillation_metrics = {
        "distillation/student_mass": student_mass.mean().item(),
        "distillation/student_mass_min": Metric(AggregationType.MIN, student_mass.min()),
        "distillation/student_mass_max": Metric(AggregationType.MAX, student_mass.max()),
        "distillation/teacher_mass": teacher_mass.mean().item(),
        "distillation/teacher_mass_min": Metric(AggregationType.MIN, teacher_mass.min()),
        "distillation/teacher_mass_max": Metric(AggregationType.MAX, teacher_mass.max()),
        **overlap_metrics,
    }

    # Due to use of top-k, student and teacher distributions don't sum to 1 -> divergences can be negative.
    distillation_losses = distillation_losses.clamp_min(0.0)

    return distillation_losses, distillation_metrics


@register_distillation_loss(
    DistillationLossSettings(names=["kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3"], use_estimator=True)
)  # type: ignore[arg-type]
def compute_distillation_loss_reverse_kl_estimator(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the distillation loss and related metrics using single-sample KL estimators.

    Uses the kl_penalty function from core_algos which supports various KL divergence
    estimators: "kl", "k1", "abs", "mse", "k2", "low_var_kl", "k3".

    Returns:
    - distillation_losses: (bsz, resp_len)
    - distillation_metrics: Dictionary of metrics.
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape

    loss_config: DistillationLossConfig = distillation_config.distillation_loss
    distillation_losses = kl_penalty(
        logprob=student_log_probs, ref_logprob=teacher_log_probs, kl_penalty=loss_config.loss_mode
    )
    # Since k1 can be negative, log the mean absolute loss.
    metrics = {
        "distillation/abs_loss": Metric(AggregationType.MEAN, distillation_losses[response_mask_bool].abs().mean()),
    }
    return distillation_losses, metrics


def compute_opd_outcome_statistics(
    advantage: torch.Tensor,
    response_mask: torch.Tensor,
    rm_scores: torch.Tensor,
    statistics_threshold: float = 1.0e-4,
) -> dict[str, Metric]:
    """Return sufficient statistics for standard OPD signal by outcome.

    Ratios are intentionally deferred until after micro-batch and data-parallel
    aggregation. This keeps variable-length trajectories weighted per token.
    """

    if advantage.ndim != 2 or response_mask.ndim != 2:
        raise ValueError("OPD outcome statistics expect 2-D advantage and response mask tensors.")
    if advantage.shape != response_mask.shape:
        raise ValueError("OPD outcome advantage and response mask must have identical shapes.")
    if statistics_threshold < 0:
        raise ValueError(f"statistics_threshold must be non-negative, got {statistics_threshold}.")

    reward_tensor = rm_scores.to_padded_tensor(0.0) if rm_scores.is_nested else rm_scores
    if reward_tensor.ndim != 2 or reward_tensor.shape[0] != advantage.shape[0]:
        raise ValueError("OPD outcome rm_scores must be a 2-D tensor with matching batch size.")

    values = advantage.float()
    valid = response_mask.bool()
    rollout_correct = reward_tensor.float().sum(dim=-1) > 0.5
    correct = valid & rollout_correct.unsqueeze(-1)
    wrong = valid & ~rollout_correct.unsqueeze(-1)
    positive = values > float(statistics_threshold)
    negative = values < -float(statistics_threshold)
    positive_mass = values.clamp_min(0.0)
    negative_mass = (-values).clamp_min(0.0)

    prefix = "distillation/opd_outcome_stats/"

    def sum_metric(value: torch.Tensor) -> Metric:
        return Metric(AggregationType.SUM, value)

    return {
        f"{prefix}correct_token_count": sum_metric(correct.float().sum()),
        f"{prefix}wrong_token_count": sum_metric(wrong.float().sum()),
        f"{prefix}correct_advantage_sum": sum_metric(values[correct].sum()),
        f"{prefix}wrong_advantage_sum": sum_metric(values[wrong].sum()),
        f"{prefix}correct_abs_advantage_sum": sum_metric(values[correct].abs().sum()),
        f"{prefix}wrong_abs_advantage_sum": sum_metric(values[wrong].abs().sum()),
        f"{prefix}correct_positive_token_count": sum_metric((correct & positive).float().sum()),
        f"{prefix}wrong_positive_token_count": sum_metric((wrong & positive).float().sum()),
        f"{prefix}correct_negative_token_count": sum_metric((correct & negative).float().sum()),
        f"{prefix}wrong_negative_token_count": sum_metric((wrong & negative).float().sum()),
        f"{prefix}correct_positive_mass_sum": sum_metric(positive_mass[correct].sum()),
        f"{prefix}wrong_positive_mass_sum": sum_metric(positive_mass[wrong].sum()),
        f"{prefix}correct_negative_mass_sum": sum_metric(negative_mass[correct].sum()),
        f"{prefix}wrong_negative_mass_sum": sum_metric(negative_mass[wrong].sum()),
    }


@register_distillation_loss(DistillationLossSettings(names=["reverse_kl"], use_estimator=True))  # type: ignore[arg-type]
def compute_sampled_token_reverse_kl(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Estimate KL(student || teacher) on tokens sampled by the student.

    The returned per-token value is ``log p_student(a) - log p_teacher(a)``.
    Standard OPD consumes it as a detached policy-gradient advantage; directly
    differentiating this sampled value would not produce the reverse-KL
    gradient.  No teacher or student top-k distribution is materialized.
    """
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape

    sampled_reverse_kl = student_log_probs - teacher_log_probs
    valid_student_log_probs = student_log_probs[response_mask_bool].float()
    valid_teacher_log_probs = teacher_log_probs[response_mask_bool].float()
    metrics = {
        "distillation/reverse_kl_estimate": Metric(
            AggregationType.MEAN, sampled_reverse_kl[response_mask_bool].mean()
        ),
        "distillation/student_sampled_token_prob": Metric(
            AggregationType.MEAN, valid_student_log_probs.exp().mean()
        ),
        "distillation/teacher_sampled_token_prob": Metric(
            AggregationType.MEAN, valid_teacher_log_probs.exp().mean()
        ),
    }
    track_outcome_metrics = (
        bool(tu.get_non_tensor_data(data, "track_opd_outcome_metrics", default=False))
        if isinstance(data, TensorDict)
        else bool(data.get("track_opd_outcome_metrics", False))
    )
    if track_outcome_metrics and "rm_scores" in data:
        metrics.update(
            compute_opd_outcome_statistics(
                advantage=-sampled_reverse_kl,
                response_mask=response_mask_bool,
                rm_scores=data["rm_scores"],
                statistics_threshold=float(
                    getattr(
                        distillation_config.distillation_loss,
                        "opd_statistics_threshold",
                        1.0e-4,
                    )
                ),
            )
        )
    return sampled_reverse_kl, metrics


def compute_calibrated_opd_advantage(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    positive_teacher_log_probs: torch.Tensor,
    negative_teacher_log_probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Cal-OPD advantage and the selected teacher self-deviation.

    Both feedback prompts are treated as interventions rather than assumed
    directions: their maximum positive and minimum negative likelihood shifts
    are computed first.  Only the extreme shift toward the student is removed
    from the original signed gap, and ``relu`` prevents a sign reversal.
    """

    shapes = {
        student_log_probs.shape,
        teacher_log_probs.shape,
        positive_teacher_log_probs.shape,
        negative_teacher_log_probs.shape,
    }
    if len(shapes) != 1:
        raise ValueError("All Cal-OPD log-probability tensors must have identical shapes.")

    base_advantage = teacher_log_probs - student_log_probs
    positive_shift = positive_teacher_log_probs - teacher_log_probs
    negative_shift = negative_teacher_log_probs - teacher_log_probs
    zero = torch.zeros_like(base_advantage)
    upward_deviation = torch.maximum(zero, torch.maximum(positive_shift, negative_shift))
    downward_deviation = torch.minimum(zero, torch.minimum(positive_shift, negative_shift))
    teacher_self_deviation = torch.where(
        base_advantage > 0,
        -downward_deviation,
        torch.where(base_advantage < 0, upward_deviation, zero),
    )
    calibrated_magnitude = torch.relu(base_advantage.abs() - teacher_self_deviation)
    calibrated_advantage = base_advantage.sign() * calibrated_magnitude
    return calibrated_advantage, teacher_self_deviation


@register_distillation_loss(DistillationLossSettings(names=["cal_reverse_kl"], use_estimator=True))  # type: ignore[arg-type]
def compute_calibrated_sampled_token_reverse_kl(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Return ``-A_cal`` for the shared detached REINFORCE loss path."""

    del config, distillation_config
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    positive_teacher_log_probs = no_padding_2_padding(
        data["cal_positive_teacher_logprobs"], data
    ).squeeze(-1)
    negative_teacher_log_probs = no_padding_2_padding(
        data["cal_negative_teacher_logprobs"], data
    ).squeeze(-1)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert (
        student_log_probs.shape
        == teacher_log_probs.shape
        == positive_teacher_log_probs.shape
        == negative_teacher_log_probs.shape
        == response_mask_bool.shape
    )

    # The calibrated advantage is a policy-gradient weight and must remain
    # stop-gradient.
    with torch.no_grad():
        calibrated_advantage, teacher_self_deviation = compute_calibrated_opd_advantage(
            student_log_probs,
            teacher_log_probs,
            positive_teacher_log_probs,
            negative_teacher_log_probs,
        )
        base_advantage = teacher_log_probs - student_log_probs
        valid = response_mask_bool
        valid_base = base_advantage[valid].float()
        valid_calibrated = calibrated_advantage[valid].float()
        valid_deviation = teacher_self_deviation[valid].float()

        positive = valid_base > 0
        negative = valid_base < 0

        def retained_magnitude_ratio(selection: torch.Tensor) -> torch.Tensor:
            base = valid_base[selection]
            calibrated = valid_calibrated[selection]
            return torch.linalg.vector_norm(calibrated) / torch.linalg.vector_norm(base).clamp_min(1e-12)

        metrics = {
            "distillation/reverse_kl_estimate": Metric(
                AggregationType.MEAN, -valid_base.mean()
            ),
            "distillation/student_sampled_token_prob": Metric(
                AggregationType.MEAN, student_log_probs[valid].float().exp().mean()
            ),
            "distillation/teacher_sampled_token_prob": Metric(
                AggregationType.MEAN, teacher_log_probs[valid].float().exp().mean()
            ),
            "distillation/cal_advantage_mean": Metric(
                AggregationType.MEAN, valid_calibrated.mean()
            ),
            "distillation/cal_advantage_abs_mean": Metric(
                AggregationType.MEAN, valid_calibrated.abs().mean()
            ),
            "distillation/cal_zero_token_ratio": Metric(
                AggregationType.MEAN, valid_calibrated.eq(0).float().mean()
            ),
            "distillation/cal_teacher_self_deviation_mean": Metric(
                AggregationType.MEAN, valid_deviation.mean()
            ),
            "distillation/cal_retained_magnitude_ratio": Metric(
                AggregationType.MEAN,
                retained_magnitude_ratio(torch.ones_like(positive, dtype=torch.bool)),
            ),
            "distillation/cal_positive_retained_magnitude_ratio": Metric(
                AggregationType.MEAN, retained_magnitude_ratio(positive)
            ),
            "distillation/cal_negative_retained_magnitude_ratio": Metric(
                AggregationType.MEAN, retained_magnitude_ratio(negative)
            ),
        }
    # The shared PG path consumes advantages=-distillation_loss.detach().
    return -calibrated_advantage, metrics


@register_distillation_loss(DistillationLossSettings(names=["ps_reverse_kl"], use_estimator=True))  # type: ignore[arg-type]
def compute_privilege_sensitive_reverse_kl(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Downweight sampled reverse-KL tokens that are sensitive to false feedback."""
    del config
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    privileged_log_probs = no_padding_2_padding(data["privileged_teacher_logprobs"], data).squeeze(-1)
    teacher_max_log_probs = None
    if data.get("teacher_max_logprobs", None) is not None:
        teacher_max_log_probs = no_padding_2_padding(data["teacher_max_logprobs"], data).squeeze(-1)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert (
        teacher_log_probs.shape
        == privileged_log_probs.shape
        == student_log_probs.shape
        == response_mask_bool.shape
    )

    loss_config = distillation_config.distillation_loss
    base_reverse_kl = student_log_probs - teacher_log_probs
    # Weight the same stabilized signal that standard OPD would consume.
    if loss_config.loss_max_clamp is not None:
        base_reverse_kl = base_reverse_kl.clamp(
            min=-loss_config.loss_max_clamp,
            max=loss_config.loss_max_clamp,
        )
    sensitivity = (privileged_log_probs - teacher_log_probs).abs()

    stats_dir = loss_config.sensitivity_stats_dir
    if stats_dir:
        if teacher_max_log_probs is None:
            raise RuntimeError("Sensitivity diagnostics require teacher_max_logprobs.")
        stats_path = Path(str(stats_dir))
        stats_path.mkdir(parents=True, exist_ok=True)
        records = []
        for row in range(response_mask_bool.shape[0]):
            row_valid = response_mask_bool[row]
            records.append(
                {
                    "student_logprob": student_log_probs[row][row_valid].detach().float().cpu(),
                    "teacher_logprob": teacher_log_probs[row][row_valid].detach().float().cpu(),
                    "teacher_max_logprob": teacher_max_log_probs[row][row_valid].detach().float().cpu(),
                    "sensitivity": sensitivity[row][row_valid].detach().float().cpu(),
                }
            )
        torch.save(
            records,
            stats_path / f"tokens_pid{os.getpid()}_{uuid4().hex}.pt",
        )
    sensitive_mask = sensitivity > float(loss_config.sensitivity_threshold)
    weights = torch.where(
        sensitive_mask,
        torch.full_like(base_reverse_kl, float(loss_config.w_sens)),
        torch.full_like(base_reverse_kl, float(loss_config.w_stable)),
    )
    weighted_reverse_kl = base_reverse_kl * weights

    valid = response_mask_bool
    base_signal = base_reverse_kl[valid].float()
    weighted_signal = weighted_reverse_kl[valid].float()
    signal_relative_change = torch.linalg.vector_norm(weighted_signal - base_signal) / torch.linalg.vector_norm(
        base_signal
    ).clamp_min(1e-12)
    metrics = {
        "distillation/reverse_kl_estimate": Metric(
            AggregationType.MEAN, base_signal.mean()
        ),
        "distillation/student_sampled_token_prob": Metric(
            AggregationType.MEAN, student_log_probs[valid].float().exp().mean()
        ),
        "distillation/teacher_sampled_token_prob": Metric(
            AggregationType.MEAN, teacher_log_probs[valid].float().exp().mean()
        ),
        "distillation/ps_sensitive_token_ratio": Metric(
            AggregationType.MEAN, sensitive_mask[valid].float().mean()
        ),
        "distillation/ps_sensitivity_mean": Metric(
            AggregationType.MEAN, sensitivity[valid].float().mean()
        ),
        "distillation/ps_gradient_signal_relative_change": Metric(
            AggregationType.MEAN, signal_relative_change
        ),
    }
    return weighted_reverse_kl, metrics


@register_distillation_loss(DistillationLossSettings(names=["random_reverse_kl"], use_estimator=True))  # type: ignore[arg-type]
def compute_random_sampled_token_reverse_kl(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply sampled-token reverse KL to an independently random token subset.

    Every valid response token is retained with probability
    ``random_token_ratio``.  The mask is resampled for each loss invocation,
    so this remains standard sampled-token OPD on the retained tokens while
    contributing exactly zero policy-gradient signal on the others.
    """
    del config
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape

    loss_config = distillation_config.distillation_loss
    base_reverse_kl = student_log_probs - teacher_log_probs
    if loss_config.loss_max_clamp is not None:
        base_reverse_kl = base_reverse_kl.clamp(
            min=-loss_config.loss_max_clamp,
            max=loss_config.loss_max_clamp,
        )

    selected_mask = torch.rand_like(base_reverse_kl, dtype=torch.float32) < float(
        loss_config.random_token_ratio
    )
    selected_mask &= response_mask_bool
    selected_reverse_kl = base_reverse_kl * selected_mask.to(base_reverse_kl.dtype)

    valid_base_signal = base_reverse_kl[response_mask_bool].float()
    valid_selected_signal = selected_reverse_kl[response_mask_bool].float()
    metrics = {
        "distillation/reverse_kl_estimate": Metric(
            AggregationType.MEAN, valid_base_signal.mean()
        ),
        "distillation/student_sampled_token_prob": Metric(
            AggregationType.MEAN, student_log_probs[response_mask_bool].float().exp().mean()
        ),
        "distillation/teacher_sampled_token_prob": Metric(
            AggregationType.MEAN, teacher_log_probs[response_mask_bool].float().exp().mean()
        ),
        "distillation/random_selected_token_ratio": Metric(
            AggregationType.MEAN, selected_mask[response_mask_bool].float().mean()
        ),
        "distillation/random_gradient_signal_relative_change": Metric(
            AggregationType.MEAN,
            torch.linalg.vector_norm(valid_selected_signal - valid_base_signal)
            / torch.linalg.vector_norm(valid_base_signal).clamp_min(1e-12),
        ),
    }
    return selected_reverse_kl, metrics


@register_distillation_loss(DistillationLossSettings(names=["topgap_reverse_kl"], use_estimator=True))  # type: ignore[arg-type]
def compute_topgap_sampled_token_reverse_kl(
    config: ActorConfig,
    distillation_config: DistillationConfig,
    model_output: dict,
    data: TensorDict,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Retain a per-response fraction ranked by ``|log T - log S|``."""
    del config
    student_log_probs = no_padding_2_padding(model_output["log_probs"], data)
    teacher_log_probs = no_padding_2_padding(data["teacher_logprobs"], data).squeeze(-1)
    if data["response_mask"].is_nested:
        response_mask_bool = data["response_mask"].bool().to_padded_tensor(False)
    else:
        response_mask_bool = data["response_mask"].bool()
    assert teacher_log_probs.shape == student_log_probs.shape == response_mask_bool.shape

    loss_config = distillation_config.distillation_loss
    base_reverse_kl = student_log_probs - teacher_log_probs
    if loss_config.loss_max_clamp is not None:
        base_reverse_kl = base_reverse_kl.clamp(
            min=-loss_config.loss_max_clamp,
            max=loss_config.loss_max_clamp,
        )

    gap = (teacher_log_probs - student_log_probs).abs()
    selected_mask = torch.zeros_like(response_mask_bool)
    ratio = float(loss_config.topgap_token_ratio)
    largest = str(loss_config.topgap_selection) == "top"
    if ratio > 0.0:
        for row in range(response_mask_bool.shape[0]):
            valid_indices = response_mask_bool[row].nonzero(as_tuple=False).squeeze(-1)
            valid_count = int(valid_indices.numel())
            if valid_count == 0:
                continue
            selected_count = min(valid_count, max(1, math.ceil(valid_count * ratio)))
            ranked_indices = torch.topk(
                gap[row, valid_indices], k=selected_count, largest=largest, sorted=False
            ).indices
            selected_mask[row, valid_indices[ranked_indices]] = True

    selected_reverse_kl = base_reverse_kl * selected_mask.to(base_reverse_kl.dtype)
    base_signal = base_reverse_kl[response_mask_bool].float()
    selected_signal = selected_reverse_kl[response_mask_bool].float()
    valid_gap = gap[response_mask_bool].float()
    selected_gap = gap[selected_mask].float()
    selected_student_prob = student_log_probs[selected_mask].float().exp()
    selected_teacher_prob = teacher_log_probs[selected_mask].float().exp()
    selected_min_prob = torch.minimum(selected_student_prob, selected_teacher_prob)
    selected_max_prob = torch.maximum(selected_student_prob, selected_teacher_prob)
    gap_quantiles = torch.quantile(
        valid_gap, valid_gap.new_tensor([0.50, 0.75, 0.90, 0.95, 0.99])
    )
    metrics = {
        "distillation/reverse_kl_estimate": Metric(AggregationType.MEAN, base_signal.mean()),
        "distillation/student_sampled_token_prob": Metric(
            AggregationType.MEAN, student_log_probs[response_mask_bool].float().exp().mean()
        ),
        "distillation/teacher_sampled_token_prob": Metric(
            AggregationType.MEAN, teacher_log_probs[response_mask_bool].float().exp().mean()
        ),
        "distillation/topgap_selected_token_ratio": Metric(
            AggregationType.MEAN, selected_mask[response_mask_bool].float().mean()
        ),
        "distillation/topgap_gap_mean": Metric(AggregationType.MEAN, valid_gap.mean()),
        "distillation/topgap_selected_gap_mean": Metric(
            AggregationType.MEAN,
            selected_gap.mean() if selected_gap.numel() else valid_gap.new_zeros(()),
        ),
        "distillation/topgap_selected_student_prob_mean": Metric(
            AggregationType.MEAN, selected_student_prob.mean()
        ),
        "distillation/topgap_selected_teacher_prob_mean": Metric(
            AggregationType.MEAN, selected_teacher_prob.mean()
        ),
        "distillation/topgap_selected_student_gt_teacher_ratio": Metric(
            AggregationType.MEAN, (selected_student_prob > selected_teacher_prob).float().mean()
        ),
        "distillation/topgap_selected_both_high_ratio": Metric(
            AggregationType.MEAN, (selected_min_prob >= 0.1).float().mean()
        ),
        "distillation/topgap_selected_mixed_ratio": Metric(
            AggregationType.MEAN,
            ((selected_max_prob >= 0.1) & (selected_min_prob < 0.1)).float().mean(),
        ),
        "distillation/topgap_selected_both_low_ratio": Metric(
            AggregationType.MEAN, (selected_max_prob < 0.1).float().mean()
        ),
        "distillation/topgap_selected_any_very_low_ratio": Metric(
            AggregationType.MEAN, (selected_min_prob < 0.01).float().mean()
        ),
        "distillation/topgap_gap_p50": Metric(AggregationType.MEAN, gap_quantiles[0]),
        "distillation/topgap_gap_p75": Metric(AggregationType.MEAN, gap_quantiles[1]),
        "distillation/topgap_gap_p90": Metric(AggregationType.MEAN, gap_quantiles[2]),
        "distillation/topgap_gap_p95": Metric(AggregationType.MEAN, gap_quantiles[3]),
        "distillation/topgap_gap_p99": Metric(AggregationType.MEAN, gap_quantiles[4]),
        "distillation/topgap_gradient_signal_relative_change": Metric(
            AggregationType.MEAN,
            torch.linalg.vector_norm(selected_signal - base_signal)
            / torch.linalg.vector_norm(base_signal).clamp_min(1e-12),
        ),
    }
    return selected_reverse_kl, metrics
