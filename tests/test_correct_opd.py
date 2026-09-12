"""Regression tests for Correct-OPD gating and zero-advantage updates."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from algorithms.correct_opd import compute_correct_opd_batch


def _loss_config():
    return SimpleNamespace(
        loss_mode="correct_reverse_kl",
        loss_max_clamp=20.0,
        selection_ratio=1.0,
        selection_method="random",
        use_policy_gradient=True,
        policy_loss_mode="reinforce",
        use_task_rewards=False,
        global_batch_info={},
        opd_statistics_threshold=1.0e-4,
    )


def _run_full_correct_loss(student_logprobs, teacher_logprobs, correct_mask, normalization):
    from verl.trainer.distillation import losses

    actor_config = SimpleNamespace(
        loss_agg_mode="seq-mean-token-mean",
        global_batch_info={},
    )
    distillation_config = SimpleNamespace(distillation_loss=_loss_config())
    response_mask = torch.ones_like(correct_mask, dtype=torch.bool)
    data = {
        "teacher_logprobs": teacher_logprobs.unsqueeze(-1),
        "response_mask": response_mask,
        "correct_opd_token_mask": correct_mask,
        "correct_opd_loss_normalization": torch.full(
            (student_logprobs.shape[0],), normalization
        ),
        "old_log_probs": student_logprobs.detach().clone(),
        "rm_scores": torch.where(
            correct_mask.any(dim=-1, keepdim=True),
            torch.ones_like(student_logprobs),
            torch.zeros_like(student_logprobs),
        ),
        "track_opd_outcome_metrics": True,
    }
    with patch.object(losses, "no_padding_2_padding", lambda tensor, _data: tensor):
        return losses.distillation_loss(
            actor_config,
            distillation_config,
            {"log_probs": student_logprobs},
            data,
        )


def test_mixed_batch_marks_only_correct_rollouts_and_excludes_padding():
    response_mask = torch.tensor(
        [[1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.bool
    )
    rm_scores = torch.tensor(
        [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    )
    genuine = torch.tensor([True, True, True, False])
    result = compute_correct_opd_batch(
        rm_scores,
        response_mask,
        genuine_trajectory_mask=genuine,
        loss_agg_mode="seq-mean-token-mean",
    )
    torch.testing.assert_close(
        result.correct_rollout_mask,
        torch.tensor([True, False, True, False]),
    )
    torch.testing.assert_close(
        result.token_mask,
        torch.tensor([[1, 1, 1], [0, 0, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.bool),
    )
    assert result.loss_normalization == pytest.approx(3 / 2)
    assert result.metrics["correct-opd/train/correct_trajectory_count"] == 2.0
    assert result.metrics["correct-opd/train/skipped_error_trajectory_count"] == 1.0


def test_token_mean_normalization_matches_removing_incorrect_tokens():
    response_mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    rm_scores = torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    result = compute_correct_opd_batch(
        rm_scores,
        response_mask,
        loss_agg_mode="token-mean",
    )
    assert result.loss_normalization == pytest.approx(5 / 3)


def test_incorrect_trajectory_has_zero_advantage_and_zero_gradient():
    student = torch.tensor(
        [[-1.0, -1.5], [-2.0, -2.5], [-3.0, -3.5]], requires_grad=True
    )
    teacher = torch.tensor([[-0.5, -1.0], [-1.0, -1.0], [-2.0, -2.0]])
    correct_mask = torch.tensor([[1, 1], [0, 0], [1, 1]], dtype=torch.bool)
    loss, metrics = _run_full_correct_loss(student, teacher, correct_mask, 3 / 2)

    physical_student = student.detach()[[0, 2]].clone().requires_grad_(True)
    physical_teacher = teacher[[0, 2]]
    physical_advantage = (physical_teacher - physical_student).detach()
    physical_loss = -(physical_advantage * physical_student).mean(dim=-1).mean()
    physical_loss.backward()
    loss.backward()

    torch.testing.assert_close(loss.detach(), physical_loss.detach())
    torch.testing.assert_close(student.grad[[0, 2]], physical_student.grad)
    torch.testing.assert_close(student.grad[1], torch.zeros_like(student.grad[1]))
    assert metrics["distillation/selected_token_ratio"].aggregate() == 1.0
    assert "distillation/opd_outcome_stats/correct_token_count" in metrics


def test_all_incorrect_batch_is_safe_zero_gradient_noop():
    student = torch.tensor([[-1.0, -1.5], [-2.0, -2.5]], requires_grad=True)
    teacher = torch.tensor([[-0.5, -1.0], [-1.0, -1.0]])
    correct_mask = torch.zeros_like(student, dtype=torch.bool)
    gate = compute_correct_opd_batch(
        torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        torch.ones_like(correct_mask),
    )
    assert gate.loss_normalization == 0.0

    loss, metrics = _run_full_correct_loss(student, teacher, correct_mask, 0.0)
    loss.backward()
    assert loss.detach().item() == 0.0
    torch.testing.assert_close(student.grad, torch.zeros_like(student.grad))
    assert metrics["distillation/loss_min"].aggregate() == 0.0
    assert metrics["distillation/loss_max"].aggregate() == 0.0


def test_correct_opd_wandb_aliases_cover_pg_diagnostics():
    from algorithms.correct_opd import CorrectOPDTrainer

    aliases = CorrectOPDTrainer.algorithm_metric_aliases(object())
    assert aliases["actor/distillation/loss"] == "correct-opd/train/policy_loss"
    assert (
        aliases["actor/distillation/reverse_kl_estimate"]
        == "correct-opd/train/reverse_kl_estimate"
    )
