"""Shared runtime infrastructure for OPD-family algorithms.

This module contains no algorithm-specific objective.  Concrete algorithms
own their complete configuration contracts and import only this runtime base,
never another algorithm module.
"""

from __future__ import annotations

import math
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from verl.trainer import main_ppo_sync as verl_sync


SUPPORTED_LOSS_MODES = {
    "reverse_kl",
    "cal_reverse_kl",
    "eopd",
    "exopd_reverse_kl",
    "ps_reverse_kl",
    "uni_opd",
    "fire_opd",
    "oa_opd",
}

TOKEN_SELECTION_METHODS = {"random", "topgap", "bottomgap"}


def validation_namespace(data_source: str) -> str:
    """Convert a dataset filename or stem into a stable W&B namespace."""

    name = Path(str(data_source)).stem.lower()
    name = re.sub(r"20(\d{2})$", r"\1", name)
    name = re.sub(r"[^a-z0-9]+", "", name)
    if not name:
        raise ValueError(
            f"Cannot derive a validation name from data source {data_source!r}."
        )
    return f"val-{name}"


def compute_pass_avg_metrics(
    data_sources: list[str],
    sample_uids: list[str],
    accuracies: list[float],
    *,
    expected_questions: int | None,
    expected_rollouts: int,
) -> dict[str, float]:
    """Compute empirical Pass@1, Pass@K and Avg@K per data source."""

    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for data_source, uid, accuracy in zip(
        data_sources, sample_uids, accuracies, strict=True
    ):
        grouped[str(data_source)][str(uid)].append(float(accuracy))

    question_count = sum(len(uid_to_acc) for uid_to_acc in grouped.values())
    if expected_questions is not None and question_count != expected_questions:
        raise RuntimeError(
            f"Validation expected {expected_questions} questions, but received "
            f"{question_count}."
        )

    metrics: dict[str, float] = {}
    for data_source, uid_to_acc in grouped.items():
        bad_counts = {
            uid: len(values)
            for uid, values in uid_to_acc.items()
            if len(values) != expected_rollouts
        }
        if bad_counts:
            raise RuntimeError(
                f"Validation expected {expected_rollouts} rollouts per question; "
                f"mismatched counts (first 5): {list(bad_counts.items())[:5]}"
            )

        per_question = list(uid_to_acc.values())
        prefix = f"{validation_namespace(data_source)}-core"
        metrics[f"{prefix}/Pass@1"] = float(
            np.mean([values[0] for values in per_question])
        )
        metrics[f"{prefix}/Pass@{expected_rollouts}"] = float(
            np.mean([max(values) for values in per_question])
        )
        metrics[f"{prefix}/Avg@{expected_rollouts}"] = float(
            np.mean([np.mean(values) for values in per_question])
        )
    return metrics


def compute_response_metrics(
    data_sources: list[str],
    response_lengths: list[float],
    response_truncated: list[float],
    accuracies: list[float],
) -> dict[str, float]:
    """Aggregate validation response lengths by data source and outcome."""

    grouped_lengths: dict[str, list[float]] = defaultdict(list)
    grouped_truncated: dict[str, list[float]] = defaultdict(list)
    grouped_correct_lengths: dict[str, list[float]] = defaultdict(list)
    grouped_incorrect_lengths: dict[str, list[float]] = defaultdict(list)
    for data_source, length, truncated, accuracy in zip(
        data_sources, response_lengths, response_truncated, accuracies, strict=True
    ):
        source = str(data_source)
        response_length = float(length)
        grouped_lengths[source].append(response_length)
        grouped_truncated[source].append(float(truncated))
        outcome_lengths = (
            grouped_correct_lengths
            if float(accuracy) > 0.5
            else grouped_incorrect_lengths
        )
        outcome_lengths[source].append(response_length)

    metrics: dict[str, float] = {}
    for data_source, lengths in grouped_lengths.items():
        prefix = f"{validation_namespace(data_source)}-core/response_length"
        metrics[f"{prefix}/mean"] = float(np.mean(lengths))
        metrics[f"{prefix}/clip_ratio"] = float(np.mean(grouped_truncated[data_source]))
        correct_lengths = grouped_correct_lengths[data_source]
        incorrect_lengths = grouped_incorrect_lengths[data_source]
        metrics[f"{prefix}/correct_count"] = float(len(correct_lengths))
        metrics[f"{prefix}/incorrect_count"] = float(len(incorrect_lengths))
        if correct_lengths:
            metrics[f"{prefix}/correct_mean"] = float(np.mean(correct_lengths))
        if incorrect_lengths:
            metrics[f"{prefix}/incorrect_mean"] = float(np.mean(incorrect_lengths))
    return metrics


def configure_opd_defaults(config) -> tuple[str, str]:
    """Install project defaults, including stable run and W&B metadata."""

    project_root = Path(__file__).resolve().parents[1]
    from utils.prompts import get_prompt_template

    student_prompt = str(config.get("student_prompt", "")).strip()
    teacher_prompt = str(config.get("teacher_prompt", "")).strip()
    get_prompt_template(student_prompt)
    get_prompt_template(teacher_prompt)

    OmegaConf.update(
        config,
        "data.custom_cls",
        {
            "path": str(project_root / "utils" / "custom_dataset.py"),
            "name": "CustomDataset",
        },
        force_add=True,
    )
    OmegaConf.update(config, "data.student_prompt", student_prompt, force_add=True)
    OmegaConf.update(config, "data.teacher_prompt", teacher_prompt, force_add=True)
    configure_run_metadata(config)
    return student_prompt, teacher_prompt


def configure_run_metadata(config) -> tuple[str, str]:
    """Resolve explicit or automatic run names and a non-empty W&B group."""

    run_name = str(config.get("run_name", "") or "").strip()
    if not run_name:
        prefix = str(config.get("run_name_prefix", "") or "").strip()
        if not prefix:
            prefix = str(config.algorithm.name).strip()
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_name = f"{prefix}_{timestamp}_p{os.getpid()}"

    group_name = str(config.get("group_name", "") or "").strip() or "temp"
    OmegaConf.update(config, "run_name", run_name, force_add=True)
    OmegaConf.update(config, "group_name", group_name, force_add=True)
    OmegaConf.update(config, "trainer.experiment_name", run_name, force_add=True)
    return run_name, group_name


def configure_opd_data_parallel_batch(config) -> int:
    """Round the question batch up to a whole number per student DP rank."""

    student_gpus = int(config.trainer.n_gpus_per_node) * int(config.trainer.nnodes)
    rollout = config.actor_rollout_ref.rollout
    if int(rollout.tensor_model_parallel_size) != 1:
        raise ValueError(
            "OPD uses pure data parallelism; student rollout TP must be 1."
        )
    if int(rollout.data_parallel_size) != student_gpus:
        raise ValueError(
            "OPD student rollout DP must equal the student GPU count, got "
            f"DP={rollout.data_parallel_size} and student_gpus={student_gpus}."
        )
    if int(config.actor_rollout_ref.actor.ulysses_sequence_parallel_size) != 1:
        raise ValueError(
            "OPD uses pure data parallelism; Ulysses sequence parallel size must be 1."
        )

    configured_questions = int(config.data.train_batch_size)
    adjusted_questions = math.ceil(configured_questions / student_gpus) * student_gpus
    OmegaConf.update(config, "data.train_batch_size", adjusted_questions)
    OmegaConf.update(
        config, "actor_rollout_ref.actor.ppo_mini_batch_size", adjusted_questions
    )
    return adjusted_questions


def validate_token_selection_config(loss) -> None:
    """Validate the token-subset settings shared by GKD-OPD and PG-OPD."""

    ratio = float(loss.selection_ratio)
    if not 0.0 <= ratio <= 1.0:
        raise ValueError(
            f"OPD selection_ratio must lie in [0, 1], got {ratio}."
        )

    # Full-token training does not perform selection, so the configured method
    # is deliberately irrelevant in this case.
    if ratio < 1.0:
        method = str(loss.selection_method)
        if method not in TOKEN_SELECTION_METHODS:
            expected = ", ".join(sorted(TOKEN_SELECTION_METHODS))
            raise ValueError(
                f"OPD selection_method must be one of {expected}; got {method!r}."
            )


def validate_opd_runtime_config(config) -> None:
    """Validate invariants shared by every retained OPD-family algorithm."""

    if not bool(config.distillation.enabled):
        raise ValueError("OPD requires distillation.enabled=true.")

    loss = config.distillation.distillation_loss
    if int(loss.diagnostic_topk) != 16:
        raise ValueError(
            "OPD currently standardizes convergence diagnostics at diagnostic_topk=16."
        )
    loss_mode = str(loss.loss_mode)
    if loss_mode not in SUPPORTED_LOSS_MODES:
        raise ValueError(f"Unsupported OPD-family loss_mode={loss_mode!r}.")
    if loss_mode in {"eopd", "fire_opd"}:
        if loss.topk is None or int(loss.topk) <= 0:
            raise ValueError(
                f"{loss_mode} requires a positive distillation topk for "
                "full-vocabulary Teacher entropy transfer."
            )
    elif loss.topk is not None:
        raise ValueError("Reverse-KL OPD requires distillation topk=null.")
    if not bool(loss.use_policy_gradient):
        raise ValueError(
            "Reverse-KL OPD signals must be consumed with use_policy_gradient=true."
        )
    if bool(loss.use_task_rewards):
        raise ValueError("OPD is distillation-only; set use_task_rewards=false.")

    max_prompt_tokens = int(config.data.max_prompt_length)
    train_response_tokens = int(config.rlvr_generation.train_max_new_tokens)
    val_response_tokens = int(config.rlvr_generation.val_max_new_tokens)
    if min(max_prompt_tokens, train_response_tokens, val_response_tokens) <= 0:
        raise ValueError("OPD prompt and generation token limits must be positive.")
    if int(config.data.max_response_length) < max(
        train_response_tokens, val_response_tokens
    ):
        raise ValueError(
            "data.max_response_length must cover both OPD generation limits; "
            f"got data.max_response_length={config.data.max_response_length}, "
            f"train_max_new_tokens={train_response_tokens}, "
            f"val_max_new_tokens={val_response_tokens}."
        )

    required_val_context = max_prompt_tokens + val_response_tokens
    rollout = config.actor_rollout_ref.rollout
    if int(rollout.max_model_len) < required_val_context:
        raise ValueError(
            "Student rollout max_model_len must cover max_prompt_length + "
            f"val_max_new_tokens ({required_val_context}), got "
            f"{rollout.max_model_len}."
        )

    required_train_tokens = max_prompt_tokens + train_response_tokens + 1
    token_limits = {
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": (
            config.actor_rollout_ref.actor.ppo_max_token_len_per_gpu
        ),
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": (
            config.actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu
        ),
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu": (
            config.actor_rollout_ref.ref.log_prob_max_token_len_per_gpu
        ),
    }
    undersized_limits = {
        name: int(value)
        for name, value in token_limits.items()
        if int(value) < required_train_tokens
    }
    if undersized_limits:
        details = ", ".join(
            f"{name}={value}" for name, value in undersized_limits.items()
        )
        raise ValueError(
            "OPD token limits must cover max_prompt_length + "
            f"train_max_new_tokens + 1 ({required_train_tokens}); "
            f"undersized settings: {details}."
        )

    teacher = config.distillation.teacher_models.teacher_model.inference
    if int(teacher.prompt_length) < max_prompt_tokens:
        raise ValueError(
            "Teacher prompt_length must cover data.max_prompt_length; got "
            f"{teacher.prompt_length} < {max_prompt_tokens}."
        )
    if int(teacher.response_length) < train_response_tokens:
        raise ValueError(
            "Teacher response_length must cover train_max_new_tokens; got "
            f"{teacher.response_length} < {train_response_tokens}."
        )
    if int(teacher.max_model_len) < required_train_tokens:
        raise ValueError(
            "Teacher max_model_len must cover the prompt, full training response, "
            f"and one prediction token ({required_train_tokens}); got "
            f"{teacher.max_model_len}."
        )

    student_gpus = int(config.trainer.n_gpus_per_node) * int(config.trainer.nnodes)
    teacher_gpus = int(config.distillation.n_gpus_per_node) * int(
        config.distillation.nnodes
    )
    if student_gpus <= 0 or teacher_gpus <= 0:
        raise ValueError("Both student and teacher must receive at least one GPU.")

    if (
        int(teacher.tensor_model_parallel_size) != 1
        or int(teacher.data_parallel_size) != 1
    ):
        raise ValueError(
            "Each OPD teacher replica must occupy exactly one GPU; set teacher "
            "TP=1 and per-replica DP=1."
        )


class BaseOPDTrainer(verl_sync.PPOTrainer):
    """Synchronous VERL trainer providing algorithm-neutral OPD facilities."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("config")
        if config is None and args:
            config = args[0]
        validate_opd_runtime_config(config)
        super().__init__(*args, **kwargs)

    def _val_metrics_update(
        self, data_sources, sample_uids, reward_extra_infos_dict, sample_turns
    ):
        metrics = super()._val_metrics_update(
            data_sources, sample_uids, reward_extra_infos_dict, sample_turns
        )
        accuracies = reward_extra_infos_dict.get("acc")
        if accuracies is None:
            raise RuntimeError("OPD validation reward must return an 'acc' field.")

        response_lengths = reward_extra_infos_dict.get("response_length")
        response_truncated = reward_extra_infos_dict.get("response_truncated")
        if response_lengths is None or response_truncated is None:
            raise RuntimeError(
                "OPD validation requires response length and truncation metrics."
            )

        expected_rollouts = int(self.config.actor_rollout_ref.rollout.val_kwargs.n)
        metrics.update(
            compute_pass_avg_metrics(
                data_sources,
                sample_uids,
                accuracies,
                expected_questions=(
                    int(self.config.data.get("val_max_samples", -1))
                    if int(self.config.data.get("val_max_samples", -1)) > 0
                    else None
                ),
                expected_rollouts=expected_rollouts,
            )
        )
        metrics.update(
            compute_response_metrics(
                data_sources, response_lengths, response_truncated, accuracies
            )
        )
        for key in [
            key for key in metrics if key.startswith(("val-core/", "val-aux/"))
        ]:
            metrics.pop(key)
        return metrics

    def _add_opd_training_metrics(self, metrics) -> None:
        """Expose shared diagnostics under a stable W&B namespace."""

        track_reward_metrics = verl_sync.should_track_opd_reward_metrics(
            getattr(self, "config", None)
        )
        aliases = {
            "actor/entropy": "opd/train/entropy",
            "actor/grad_norm": "opd/train/grad_norm",
            "response_length/mean": "opd/train/response_length_mean",
            "response_length/max": "opd/train/response_length_max",
            "response_length/clip_ratio": "opd/train/response_truncated_ratio",
            "perf/throughput": "opd/perf/tokens_per_second_per_gpu",
            "perf/time_per_step": "opd/perf/time_per_step",
            "opd/diagnostics/top16_overlap": "opd/train/top16-overlap",
            "opd/diagnostics/top16_mass_overlap": "opd/train/top16-mass-overlap",
            "actor/distillation/reverse_kl_estimate": ("opd/train/reverse_kl_estimate"),
            "actor/distillation/student_sampled_token_prob": (
                "opd/train/student_sampled_token_prob_mean"
            ),
            "actor/distillation/teacher_sampled_token_prob": (
                "opd/train/teacher_sampled_token_prob_mean"
            ),
        }
        if track_reward_metrics:
            aliases["rollout/task_reward/mean"] = "opd/train/reward"
        aliases.update(self.algorithm_metric_aliases())
        for source, destination in aliases.items():
            if source in metrics:
                metrics[destination] = metrics[source]

        excluded_fragments = (
            "reward",
            "critic/score",
            "critic/advantages",
            "critic/returns",
        )
        for key in [
            key
            for key in metrics
            if key != "opd/train/reward"
            and any(fragment in key.lower() for fragment in excluded_fragments)
        ]:
            metrics.pop(key)

        # These generic VERL bookkeeping namespaces add W&B panels without
        # contributing any OPD signal. The logging step is supplied separately
        # to Tracking.log(), so removing training/global_step is safe.
        excluded_prefixes = (
            "response/",
            "response_length_non_aborted/",
            "training/",
        )
        for key in [key for key in metrics if key.startswith(excluded_prefixes)]:
            metrics.pop(key)

        stats_prefix = "actor/distillation/opd_outcome_stats/"

        def raw(name: str, default=0.0):
            return metrics.get(f"{stats_prefix}{name}", default)

        if track_reward_metrics and any(key.startswith(stats_prefix) for key in metrics):
            correct_count = raw("correct_token_count")
            wrong_count = raw("wrong_token_count")
            correct_positive_mass = raw("correct_positive_mass_sum")
            wrong_positive_mass = raw("wrong_positive_mass_sum")
            correct_negative_mass = raw("correct_negative_mass_sum")
            wrong_negative_mass = raw("wrong_negative_mass_sum")
            metrics.update(
                {
                    "opd/train/correct_mean_advantage": self._safe_ratio(
                        raw("correct_advantage_sum"), correct_count
                    ),
                    "opd/train/wrong_mean_advantage": self._safe_ratio(
                        raw("wrong_advantage_sum"), wrong_count
                    ),
                    "opd/train/correct_mean_abs_advantage": self._safe_ratio(
                        raw("correct_abs_advantage_sum"), correct_count
                    ),
                    "opd/train/wrong_mean_abs_advantage": self._safe_ratio(
                        raw("wrong_abs_advantage_sum"), wrong_count
                    ),
                    "opd/train/correct_positive_advantage_ratio": self._safe_ratio(
                        raw("correct_positive_token_count"), correct_count
                    ),
                    "opd/train/wrong_positive_advantage_ratio": self._safe_ratio(
                        raw("wrong_positive_token_count"), wrong_count
                    ),
                    "opd/train/correct_negative_advantage_ratio": self._safe_ratio(
                        raw("correct_negative_token_count"), correct_count
                    ),
                    "opd/train/wrong_negative_advantage_ratio": self._safe_ratio(
                        raw("wrong_negative_token_count"), wrong_count
                    ),
                    "opd/train/wrong_positive_advantage_mass_ratio": self._safe_ratio(
                        wrong_positive_mass,
                        correct_positive_mass + wrong_positive_mass,
                    ),
                    "opd/train/correct_negative_advantage_mass_ratio": self._safe_ratio(
                        correct_negative_mass,
                        correct_negative_mass + wrong_negative_mass,
                    ),
                }
            )

        # Sufficient statistics are implementation details, not W&B API.
        for key in [key for key in metrics if key.startswith(stats_prefix)]:
            metrics.pop(key)

        if not track_reward_metrics:
            reward_related_keys = (
                "opd/train/reward",
                "rollout/task_reward/",
                "opd/train/correct_",
                "opd/train/wrong_",
                "response_length/correct_",
                "response_length/incorrect_",
            )
            for key in [key for key in metrics if key.startswith(reward_related_keys)]:
                metrics.pop(key)

    @staticmethod
    def _safe_ratio(numerator, denominator) -> float:
        numerator = float(numerator)
        denominator = float(denominator)
        return numerator / denominator if denominator > 0.0 else 0.0

    def algorithm_metric_aliases(self) -> dict[str, str]:
        """Return metric aliases owned by a concrete algorithm."""

        return {}

    def _compute_metrics(self, batch, metrics, timing_raw, global_steps, epoch):
        super()._compute_metrics(batch, metrics, timing_raw, global_steps, epoch)
        self._add_opd_training_metrics(metrics)
