"""Regression coverage for the R²OPL-base objective and its production config."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from tensordict import TensorDict
from omegaconf import OmegaConf

from algorithms import resolve_algorithm
from algorithms.r2opl_base import (
    R2OPL_BASE_DEFAULT_LAMBDA,
    R2OPLBaseTrainer,
    compute_r2opl_base_batch,
    validate_r2opl_base_config,
)
from utils.opd_runtime import configure_opd_data_parallel_batch
from verl.trainer import main_ppo_sync as verl_sync


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "configs" / "PUB_R2OPL_Base"
CONFIG_NAME = "pub_r2opl_base_qwen3_4b_instruct_2507_to_1p7b_no_thinking_len10k_lambda0p05_500steps"


def _loss_config(lambda_: float = R2OPL_BASE_DEFAULT_LAMBDA):
    return SimpleNamespace(
        loss_mode="r2opl_base",
        loss_max_clamp=20.0,
        use_policy_gradient=True,
        policy_loss_mode="reinforce",
        use_task_rewards=False,
        global_batch_info={},
        r2opl_lambda=lambda_,
    )


def _run_r2opl_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    result,
    *,
    lambda_: float,
    branch: str = "total",
):
    """Exercise the registered loss with padded mock Student/Teacher values."""

    from verl.trainer.distillation import losses

    actor_config = SimpleNamespace(
        loss_agg_mode="seq-mean-token-mean",
        loss_scale_factor=None,
        global_batch_info={},
    )
    data = {
        "teacher_logprobs": teacher_log_probs.unsqueeze(-1),
        "response_mask": response_mask,
        "r2opl_base_correct_mask": result.correct_mask,
        "r2opl_base_error_mask": result.error_mask,
        "r2opl_base_difficulty": result.difficulty.unsqueeze(-1).expand_as(
            response_mask
        ),
        "r2opl_base_lambda": torch.full(
            (student_log_probs.shape[0],), lambda_, dtype=torch.float32
        ),
        "r2opl_base_gradient_branch": branch,
        "old_log_probs": student_log_probs.detach().clone(),
        "dp_size": 1,
        "batch_num_tokens": int(response_mask.sum().item()),
        "global_batch_size": student_log_probs.shape[0],
    }
    with patch.object(losses, "no_padding_2_padding", lambda tensor, _data: tensor):
        return losses.distillation_loss(
            actor_config,
            SimpleNamespace(distillation_loss=_loss_config(lambda_)),
            {"log_probs": student_log_probs},
            data,
        )


def _expected_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    result,
    *,
    lambda_: float,
    branch: str,
) -> torch.Tensor:
    """Direct reference implementation of the per-trajectory R²OPL loss."""

    difficulty = result.difficulty.unsqueeze(-1)
    correct_advantage = difficulty.expand_as(student_log_probs)
    error_advantage = difficulty * lambda_ * (
        teacher_log_probs - student_log_probs
    ).detach()
    if branch == "correct":
        advantage = correct_advantage * result.correct_mask
    elif branch == "error":
        advantage = error_advantage * result.error_mask
    elif branch == "total":
        advantage = torch.where(
            result.correct_mask, correct_advantage, error_advantage
        ) * response_mask
    else:  # pragma: no cover - helper contract
        raise ValueError(branch)
    token_counts = response_mask.sum(dim=-1).clamp_min(1)
    return (
        (-(advantage.detach() * student_log_probs) * response_mask)
        .sum(dim=-1)
        .div(token_counts)
        .mean()
    )


def test_32_questions_times_8_rollouts_get_group_difficulty_and_masks():
    group_size, question_count, width = 8, 32, 5
    batch_size = group_size * question_count
    response_mask = torch.ones((batch_size, width), dtype=torch.bool)
    response_mask[:, -1] = torch.arange(batch_size).remainder(2).eq(0)
    old = torch.full((batch_size, width), -2.0)
    teacher = old + 0.4
    rewards = torch.zeros((batch_size, width))
    group_ids = []
    expected_difficulties = []
    for question in range(question_count):
        correct_count = question % (group_size + 1)
        start = question * group_size
        rewards[start : start + correct_count, 0] = 1.0
        group_ids.extend([f"question-{question}"] * group_size)
        expected_difficulties.extend([1.0 - correct_count / group_size] * group_size)

    result = compute_r2opl_base_batch(
        old, teacher, response_mask, rewards, group_ids, lambda_=0.05
    )

    assert result.correct_mask.shape == (batch_size, width)
    assert result.error_mask.shape == (batch_size, width)
    assert int((result.correct_mask & result.error_mask).sum()) == 0
    torch.testing.assert_close(
        result.difficulty,
        torch.tensor(expected_difficulties, dtype=torch.float32),
    )
    assert result.group_success.shape == (question_count,)
    assert result.metrics["r2opl_base/train/lambda"] == pytest.approx(0.05)


def test_advantage_metrics_use_token_mean_then_trajectory_mean():
    # Both trajectories are wrong, so the difficulty is exactly one.  Their
    # token means are 0.10 and 0.01; a flat token average would be 0.046.
    response_mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    old = torch.zeros((2, 3))
    teacher = torch.tensor([[2.0, 2.0, 0.0], [0.2, 0.2, 0.2]])
    result = compute_r2opl_base_batch(
        old,
        teacher,
        response_mask,
        torch.zeros(2),
        ["same-question", "same-question"],
        lambda_=0.05,
    )

    assert result.metrics["r2opl_base/train/error_opd_advantage_mean"] == pytest.approx(
        0.055
    )
    assert result.metrics[
        "r2opl_base/train/error_opd_abs_advantage_mean"
    ] == pytest.approx(0.055)
    assert result.metrics["r2opl_base/train/error_raw_opd_mean"] == pytest.approx(
        1.1
    )


def test_correct_and_error_branches_match_the_requested_objective():
    lambda_ = 0.05
    response_mask = torch.tensor(
        [[1, 1, 1], [1, 1, 0], [1, 1, 1], [1, 0, 0]], dtype=torch.bool
    )
    teacher = torch.tensor(
        [[-0.3, -0.8, -1.1], [-1.0, -1.4, 0.0], [-0.5, -1.2, -0.9], [-2.0, 0.0, 0.0]]
    )
    old = torch.tensor(
        [[-0.8, -1.0, -1.3], [-1.4, -1.6, 0.0], [-0.7, -1.0, -1.5], [-1.5, 0.0, 0.0]]
    )
    # The first prompt has one correct and one wrong rollout (difficulty 1/2);
    # the second is all wrong (difficulty 1).
    result = compute_r2opl_base_batch(
        old,
        teacher,
        response_mask,
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
        ["a", "a", "b", "b"],
        lambda_=lambda_,
    )
    assert result.metrics["r2opl_base/train/correct_raw_advantage_mean"] == 1.0
    assert result.metrics["r2opl_base/train/correct_advantage_mean"] == pytest.approx(
        0.5
    )

    for branch in ("correct", "error", "total"):
        student = old.clone().requires_grad_(True)
        actual, _ = _run_r2opl_loss(
            student, teacher, response_mask, result, lambda_=lambda_, branch=branch
        )
        expected_student = old.clone().requires_grad_(True)
        expected = _expected_loss(
            expected_student,
            teacher,
            response_mask,
            result,
            lambda_=lambda_,
            branch=branch,
        )
        torch.testing.assert_close(actual, expected)
        actual.backward()
        expected.backward()
        torch.testing.assert_close(student.grad, expected_student.grad)


def test_all_correct_and_all_error_groups_have_safe_extreme_difficulties():
    response_mask = torch.ones((8, 2), dtype=torch.bool)
    old = torch.zeros((8, 2))
    teacher = torch.ones((8, 2))

    all_correct = compute_r2opl_base_batch(
        old, teacher, response_mask, torch.ones(8), ["x"] * 8
    )
    torch.testing.assert_close(all_correct.difficulty, torch.zeros(8))
    assert all_correct.metrics["r2opl_base/train/correct_advantage_mean"] == 0.0

    all_error = compute_r2opl_base_batch(
        old, teacher, response_mask, torch.zeros(8), ["x"] * 8
    )
    torch.testing.assert_close(all_error.difficulty, torch.ones(8))
    assert all_error.metrics["r2opl_base/train/error_opd_advantage_mean"] == pytest.approx(
        R2OPL_BASE_DEFAULT_LAMBDA
    )


def run_r2opl_base_toy_measurement() -> dict[str, float]:
    """Return deterministic small-batch R²OPL gradient diagnostics for λ=1/20.

    This deliberately uses a tiny linear Student instead of a large language
    model so the test can run on one local GPU or CPU while still exercising
    the registered policy loss, branch masks, per-trajectory normalization,
    and detached OPD advantage.
    """

    lambda_ = 1.0 / 20.0
    group_size, group_count, width = 8, 2, 5
    batch_size = group_size * group_count
    rows = torch.arange(batch_size, dtype=torch.float32).view(-1, 1)
    tokens = torch.arange(width, dtype=torch.float32).view(1, -1)
    features = torch.stack(
        (
            torch.ones((batch_size, width)),
            (rows + 1.0).expand(-1, width) / batch_size,
            (tokens + 1.0).expand(batch_size, -1) / width,
        ),
        dim=-1,
    )
    lengths = torch.tensor([1, 2, 3, 4, 5, 3, 4, 2] * group_count)
    response_mask = torch.arange(width).view(1, -1) < lengths.view(-1, 1)
    rewards = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0] + [1.0, 1.0] + [0.0] * 6)
    group_ids = ["prompt-0"] * group_size + ["prompt-1"] * group_size

    model = torch.nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.15, -0.35, 0.20]]))
    with torch.no_grad():
        old = model(features).squeeze(-1)
        gaps = (((rows * 3.0 + tokens * 5.0).remainder(11.0) - 5.0) / 10.0).expand_as(old)
        teacher = old + gaps
    result = compute_r2opl_base_batch(
        old, teacher, response_mask, rewards, group_ids, lambda_=lambda_
    )

    def branch_gradient(branch: str) -> tuple[torch.Tensor, float]:
        model.zero_grad(set_to_none=True)
        student = model(features).squeeze(-1)
        loss, _ = _run_r2opl_loss(
            student,
            teacher,
            response_mask,
            result,
            lambda_=lambda_,
            branch=branch,
        )
        loss.backward()
        return model.weight.grad.detach().clone(), float(loss.detach())

    correct_gradient, correct_loss = branch_gradient("correct")
    error_gradient, error_loss = branch_gradient("error")
    total_gradient, total_loss = branch_gradient("total")
    torch.testing.assert_close(total_gradient, correct_gradient + error_gradient)

    return {
        "lambda": lambda_,
        "correct_grad_norm": float(torch.linalg.vector_norm(correct_gradient)),
        "error_grad_norm": float(torch.linalg.vector_norm(error_gradient)),
        "total_grad_norm": float(torch.linalg.vector_norm(total_gradient)),
        "correct_advantage_abs_mean": result.metrics[
            "r2opl_base/train/correct_advantage_mean"
        ],
        "error_advantage_abs_mean": result.metrics[
            "r2opl_base/train/error_opd_abs_advantage_mean"
        ],
        "correct_loss": correct_loss,
        "error_loss": error_loss,
        "total_loss": total_loss,
    }


def test_toy_branch_gradients_add_to_the_total_gradient():
    measurement = run_r2opl_base_toy_measurement()
    assert measurement["lambda"] == pytest.approx(0.05)
    assert measurement["correct_grad_norm"] > 0.0
    assert measurement["error_grad_norm"] > 0.0
    assert measurement["total_grad_norm"] > 0.0
    assert measurement["correct_advantage_abs_mean"] == pytest.approx(7.0 / 12.0)
    assert measurement["error_advantage_abs_mean"] > 0.0


def _compose_production_config():
    from hydra import compose, initialize_config_dir

    search_path = (
        f"hydra.searchpath=[file://{PROJECT_ROOT / 'configs'},"
        "pkg://verl.trainer.config]"
    )
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        return compose(config_name=CONFIG_NAME, overrides=[search_path])


def test_production_config_preserves_the_32_times_8_8_gpu_contract():
    config = _compose_production_config()
    OmegaConf.resolve(config)
    configure_opd_data_parallel_batch(config)
    validate_r2opl_base_config(config)
    verl_sync.validate_config(
        config,
        use_reference_policy=verl_sync.need_reference_policy(config),
        use_critic=verl_sync.need_critic(config),
    )

    assert resolve_algorithm(config).trainer_class is R2OPLBaseTrainer
    assert config.algorithm.name == "r2opl_base"
    assert config.algorithm.r2opl_base["lambda"] == pytest.approx(0.05)
    assert config.data.train_batch_size == 32
    assert config.actor_rollout_ref.actor.ppo_mini_batch_size == 32
    assert config.actor_rollout_ref.rollout.n == 8
    assert config.data.train_batch_size * config.actor_rollout_ref.rollout.n == 256
    assert config.actor_rollout_ref.actor.optim.lr == pytest.approx(1.0e-6)
    assert config.actor_rollout_ref.actor.optim.clip_grad == pytest.approx(1.0)
    assert config.trainer.total_training_steps == 500
    assert config.trainer.test_freq == 20
    assert config.trainer.save_freq == 100
    assert config.actor_rollout_ref.rollout.enable_sleep_mode is False
    assert config.distillation.teacher_models.teacher_model.inference.enable_sleep_mode is False
    assert config.trainer.n_gpus_per_node == 4
    assert config.distillation.n_gpus_per_node == 4
    assert config.actor_rollout_ref.actor.loss_agg_mode == "seq-mean-token-mean"
    assert config.distillation.distillation_loss.loss_mode == "r2opl_base"
    assert config.distillation.distillation_loss.loss_max_clamp == pytest.approx(20.0)


def test_controller_old_log_prob_handles_two_questions_with_eight_rollouts(monkeypatch):
    """Exercise the real TransferQueue alignment path on a small 2x8 batch.

    The production run uses 32 questions, but this intentionally smaller
    controller smoke test catches missing ``prompts``/``responses`` fields
    before a full distributed launch.  Teacher log-probs are stored in the
    same nested full-sequence layout returned by TransferQueue, so
    ``no_padding_2_padding`` performs its actual token-boundary conversion.
    """

    from verl.trainer import main_ppo_sync as sync

    question_count, rollouts_per_question = 2, 8
    batch_size, response_len, prompt_len = question_count * rollouts_per_question, 3, 2
    sequence_len = prompt_len + response_len

    class FakeBatch:
        keys = [f"sample-{index}" for index in range(batch_size)]
        partition_id = 0
        tags = [{} for _ in range(batch_size)]

        def __len__(self):
            return batch_size

    prompts = torch.nested.as_nested_tensor(
        [torch.arange(prompt_len) + index for index in range(batch_size)],
        layout=torch.jagged,
    )
    responses = torch.nested.as_nested_tensor(
        [torch.arange(response_len) + 10 + index for index in range(batch_size)],
        layout=torch.jagged,
    )
    response_mask = torch.nested.as_nested_tensor(
        [torch.ones(response_len, dtype=torch.bool) for _ in range(batch_size)],
        layout=torch.jagged,
    )
    teacher_full_sequence = torch.nested.as_nested_tensor(
        [torch.arange(sequence_len, dtype=torch.float32) + index for index in range(batch_size)],
        layout=torch.jagged,
    )
    fields = TensorDict(
        {
            "uid": [f"question-{index // rollouts_per_question}" for index in range(batch_size)],
            "prompts": prompts,
            "responses": responses,
            "response_mask": response_mask,
            "rm_scores": torch.tensor(
                [1.0] * rollouts_per_question + [0.0] * rollouts_per_question
            ),
            "teacher_logprobs": teacher_full_sequence,
            "old_log_probs": torch.zeros((batch_size, response_len)),
        },
        batch_size=[batch_size],
    )
    requested_fields = []
    stored_fields = {}

    def fake_get(*, select_fields, **_kwargs):
        requested_fields.extend(select_fields)
        return fields

    def fake_put(*, fields, **_kwargs):
        stored_fields.update({key: value for key, value in fields.items()})
        return FakeBatch()

    monkeypatch.setattr(sync.tq, "kv_batch_get", fake_get)
    monkeypatch.setattr(sync.tq, "kv_batch_put", fake_put)
    monkeypatch.setattr(
        sync.PPOTrainer,
        "_compute_old_log_prob",
        lambda _self, batch, _metrics: batch,
    )

    trainer = object.__new__(R2OPLBaseTrainer)
    trainer.config = OmegaConf.create(
        {
            "data": {"train_batch_size": question_count},
            "actor_rollout_ref": {"rollout": {"n": rollouts_per_question}},
            "algorithm": {"r2opl_base": {"lambda": 0.05}},
        }
    )
    result = trainer._compute_old_log_prob(FakeBatch(), metrics={})

    assert "prompts" in requested_fields
    assert "responses" in requested_fields
    assert "teacher_logprobs" in requested_fields
    assert len(result) == batch_size
    assert stored_fields["r2opl_base_correct_mask"].is_nested
    assert stored_fields["r2opl_base_error_mask"].is_nested
    assert stored_fields["r2opl_base_difficulty"].is_nested
    assert stored_fields["r2opl_base_lambda"].shape == (batch_size,)


def test_wandb_aliases_expose_all_requested_gradient_diagnostics():
    aliases = R2OPLBaseTrainer.algorithm_metric_aliases(object())
    assert aliases["actor/entropy"] == "r2opl_base/train/entropy"
    assert aliases["actor/grad_norm"] == "r2opl_base/train/total_grad_norm"
    assert aliases["response_length/mean"] == "r2opl_base/train/response_length_mean"
    assert aliases["response_length/max"] == "r2opl_base/train/response_length_max"
    assert (
        aliases["response_length/clip_ratio"]
        == "r2opl_base/train/response_truncated_ratio"
    )
    assert (
        aliases["actor/r2opl_base/correct_grad_norm"]
        == "r2opl_base/train/correct_grad_norm"
    )
    assert (
        aliases["actor/r2opl_base/error_grad_norm"]
        == "r2opl_base/train/error_grad_norm"
    )
