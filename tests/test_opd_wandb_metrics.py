"""Regression tests for the compact OPD W&B metric schema."""

from collections import Counter

from utils.opd_runtime import BaseOPDTrainer, compute_pass_avg_metrics
from verl.trainer.main_ppo_sync import (
    select_validation_generation_demos,
    should_track_opd_reward_metrics,
)
from verl.utils.tracking import ValidationGenerationsLogger


def test_validation_metrics_do_not_emit_auxiliary_question_count():
    metrics = compute_pass_avg_metrics(
        data_sources=["AIME-2024.json"] * 4,
        sample_uids=["q1", "q1", "q2", "q2"],
        accuracies=[1.0, 0.0, 0.0, 0.0],
        expected_questions=2,
        expected_rollouts=2,
    )

    assert metrics == {
        "val-aime24-core/Pass@1": 0.5,
        "val-aime24-core/Pass@2": 0.5,
        "val-aime24-core/Avg@2": 0.25,
    }


def test_training_metrics_drop_unused_verl_namespaces():
    trainer = object.__new__(BaseOPDTrainer)
    trainer.config = {
        "student_prompt": "qwen3_no_thinking_prompt",
        "teacher_prompt": "qwen3_no_thinking_prompt",
    }
    metrics = {
        "rollout/task_reward/mean": 0.375,
        # The generic mean may differ because it excludes aborted responses.
        "critic/score/mean": 0.5,
        "response/aborted_ratio": 0.0,
        "response_length_non_aborted/mean": 100.0,
        "training/global_step": 3,
        "training/epoch": 0,
        "training/num_turns/mean": 2.0,
        "response_length/mean": 100.0,
    }

    trainer._add_opd_training_metrics(metrics)

    assert not any(
        key.startswith(("response/", "response_length_non_aborted/", "training/"))
        for key in metrics
    )
    assert "opd/train/response_aborted_ratio" not in metrics
    assert metrics["opd/train/reward"] == 0.375
    assert "rollout/task_reward/mean" not in metrics
    assert "critic/score/mean" not in metrics
    assert metrics["opd/train/response_length_mean"] == 100.0


def test_thinking_training_drops_reward_and_outcome_metrics():
    trainer = object.__new__(BaseOPDTrainer)
    trainer.config = {
        "student_prompt": "qwen3_thinking_prompt",
        "teacher_prompt": "qwen3_thinking_prompt",
    }
    metrics = {
        "rollout/task_reward/mean": 0.5,
        "opd/train/reward": 0.5,
        "opd/train/correct_mean_advantage": 1.0,
        "opd/train/wrong_positive_advantage_ratio": 0.25,
        "response_length/correct_count": 4,
        "response_length/incorrect_mean": 100.0,
    }

    trainer._add_opd_training_metrics(metrics)

    assert not metrics


def test_reward_metrics_require_both_prompts_to_be_no_thinking():
    no_thinking = "qwen3_no_thinking_prompt"
    thinking = "qwen3_thinking_prompt"

    assert should_track_opd_reward_metrics(
        {"student_prompt": no_thinking, "teacher_prompt": no_thinking}
    )
    assert not should_track_opd_reward_metrics(
        {"student_prompt": thinking, "teacher_prompt": thinking}
    )
    assert not should_track_opd_reward_metrics(
        {"student_prompt": no_thinking, "teacher_prompt": thinking}
    )


def test_validation_demos_select_ten_distinct_questions_per_dataset():
    inputs = []
    outputs = []
    scores = []
    data_sources = []
    sample_uids = []
    for dataset in ("AIME-2024", "AMC-2023"):
        for question_index in range(12):
            for rollout_index in range(2):
                inputs.append(f"{dataset} question {question_index}")
                outputs.append(f"response {question_index}-{rollout_index}")
                scores.append(float(rollout_index == 0))
                data_sources.append(dataset)
                sample_uids.append(f"{dataset}-{question_index}")

    samples, selected_sources = select_validation_generation_demos(
        inputs,
        outputs,
        scores,
        data_sources,
        sample_uids,
        demos_per_dataset=10,
    )

    assert Counter(selected_sources) == {"AIME-2024": 10, "AMC-2023": 10}
    assert len(samples) == 20
    assert all(response.endswith("-0") for _, response, _ in samples)


class _FakeTable:
    def __init__(self, columns, data=None):
        self.columns = columns
        self.data = list(data or [])

    def add_data(self, *row):
        assert len(row) == len(self.columns)
        self.data.append(list(row))


class _FakeWandb:
    Table = _FakeTable
    run = object()

    def __init__(self):
        self.logged = []

    def log(self, payload, step):
        self.logged.append((payload, step))


def test_wandb_validation_demo_table_is_long_and_accumulates_steps():
    logger = ValidationGenerationsLogger()
    wandb = _FakeWandb()

    logger._log_generations_to_wandb(
        [("q1", "r1", 1.0), ("q2", "r2", 0.0)],
        0,
        wandb,
        data_sources=["AIME-2024", "AMC-2023"],
    )
    logger._log_generations_to_wandb(
        [("q3", "r3", 1.0)],
        20,
        wandb,
        data_sources=["AIME-2024"],
    )

    assert logger.validation_table.columns == [
        "step",
        "dataset",
        "question",
        "response",
        "score",
    ]
    assert logger.validation_table.data == [
        [0, "AIME-2024", "q1", "r1", 1.0],
        [0, "AMC-2023", "q2", "r2", 0.0],
        [20, "AIME-2024", "q3", "r3", 1.0],
    ]
