"""Regression tests for signed policy-gradient OPD signals."""

from types import SimpleNamespace

import pytest
import torch

from utils.opd_runtime import BaseOPDTrainer
from verl.trainer.distillation import losses


def _config(**loss_overrides):
    loss = {
        "loss_max_clamp": None,
        "random_token_ratio": 1.0,
        "topgap_token_ratio": 1.0,
        "topgap_selection": "top",
    }
    loss.update(loss_overrides)
    return SimpleNamespace(distillation_loss=SimpleNamespace(**loss))


def _data(student, teacher):
    return (
        {"log_probs": torch.tensor([student], dtype=torch.float32)},
        {
            "teacher_logprobs": torch.tensor([teacher], dtype=torch.float32).unsqueeze(
                -1
            ),
            "response_mask": torch.ones((1, len(student)), dtype=torch.bool),
        },
    )


def _identity_unpadding(monkeypatch):
    monkeypatch.setattr(losses, "no_padding_2_padding", lambda tensor, _data: tensor)


def test_pg_opd_advantage_is_signed(monkeypatch):
    _identity_unpadding(monkeypatch)
    model_output, data = _data(student=[-0.2, -2.0], teacher=[-1.2, -1.0])

    reverse_kl, _ = losses.compute_sampled_token_reverse_kl(
        None, _config(), model_output, data
    )

    advantage = -reverse_kl
    torch.testing.assert_close(advantage, torch.tensor([[-1.0, 1.0]]))


def test_pg_opd_loss_emits_outcome_sufficient_statistics(monkeypatch):
    _identity_unpadding(monkeypatch)
    model_output, data = _data(student=[-2.0, -1.0], teacher=[-1.0, -3.0])
    data["rm_scores"] = torch.tensor([[0.0, 1.0]])
    data["track_opd_outcome_metrics"] = True

    _, metrics = losses.compute_sampled_token_reverse_kl(
        None,
        _config(opd_statistics_threshold=1.0e-4),
        model_output,
        data,
    )

    prefix = "distillation/opd_outcome_stats/"
    assert metrics[f"{prefix}correct_token_count"].aggregate() == 2
    assert metrics[f"{prefix}wrong_token_count"].aggregate() == 0
    # A = teacher - student = [1, -2].
    assert metrics[f"{prefix}correct_advantage_sum"].aggregate() == -1


def test_pg_opd_loss_skips_outcome_statistics_when_disabled(monkeypatch):
    _identity_unpadding(monkeypatch)
    model_output, data = _data(student=[-2.0], teacher=[-1.0])
    data["rm_scores"] = torch.tensor([[1.0]])
    data["track_opd_outcome_metrics"] = False

    _, metrics = losses.compute_sampled_token_reverse_kl(
        None,
        _config(opd_statistics_threshold=1.0e-4),
        model_output,
        data,
    )

    assert not any("opd_outcome_stats" in key for key in metrics)


def test_pg_opd_outcome_statistics_are_token_weighted_and_materialized():
    advantage = torch.tensor(
        [
            [2.0, -1.0, 0.00005],
            [-3.0, 4.0, 99.0],
        ]
    )
    response_mask = torch.tensor(
        [
            [True, True, True],
            [True, True, False],
        ]
    )
    # Reward is stored on one response token; row 0 is correct and row 1 wrong.
    rm_scores = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
        ]
    )

    sufficient = losses.compute_opd_outcome_statistics(
        advantage,
        response_mask,
        rm_scores,
        statistics_threshold=1.0e-4,
    )
    prefix = "distillation/opd_outcome_stats/"

    def value(name):
        return sufficient[f"{prefix}{name}"].aggregate()

    assert value("correct_token_count") == 3
    assert value("wrong_token_count") == 2
    assert value("correct_positive_token_count") == 1
    assert value("wrong_positive_token_count") == 1
    assert value("correct_negative_token_count") == 1
    assert value("wrong_negative_token_count") == 1

    metrics = {
        f"actor/{key}": metric.aggregate() for key, metric in sufficient.items()
    }
    trainer = object.__new__(BaseOPDTrainer)
    trainer.config = {
        "student_prompt": "qwen3_no_thinking_prompt",
        "teacher_prompt": "qwen3_no_thinking_prompt",
    }
    trainer._add_opd_training_metrics(metrics)

    assert metrics["opd/train/correct_mean_advantage"] == pytest.approx(1.00005 / 3)
    assert metrics["opd/train/wrong_mean_advantage"] == pytest.approx(0.5)
    assert metrics["opd/train/correct_mean_abs_advantage"] == pytest.approx(3.00005 / 3)
    assert metrics["opd/train/wrong_mean_abs_advantage"] == pytest.approx(3.5)
    assert metrics["opd/train/correct_positive_advantage_ratio"] == pytest.approx(1 / 3)
    assert metrics["opd/train/wrong_positive_advantage_ratio"] == pytest.approx(1 / 2)
    assert metrics["opd/train/correct_negative_advantage_ratio"] == pytest.approx(1 / 3)
    assert metrics["opd/train/wrong_negative_advantage_ratio"] == pytest.approx(1 / 2)
    assert metrics["opd/train/wrong_positive_advantage_mass_ratio"] == pytest.approx(
        4 / 6.00005
    )
    assert metrics["opd/train/correct_negative_advantage_mass_ratio"] == pytest.approx(1 / 4)
    assert not any(key.startswith(f"actor/{prefix}") for key in metrics)


def test_pg_opd_outcome_statistics_handle_missing_correct_group():
    sufficient = losses.compute_opd_outcome_statistics(
        advantage=torch.tensor([[-1.0, 2.0]]),
        response_mask=torch.ones((1, 2), dtype=torch.bool),
        rm_scores=torch.zeros((1, 2)),
    )
    metrics = {
        f"actor/{key}": metric.aggregate() for key, metric in sufficient.items()
    }
    trainer = object.__new__(BaseOPDTrainer)
    trainer.config = {
        "student_prompt": "qwen3_no_thinking_prompt",
        "teacher_prompt": "qwen3_no_thinking_prompt",
    }

    trainer._add_opd_training_metrics(metrics)

    assert metrics["opd/train/correct_mean_advantage"] == 0.0
    assert metrics["opd/train/correct_mean_abs_advantage"] == 0.0


def test_pg_opd_outcome_statistics_ignore_padding_only_microbatch():
    sufficient = losses.compute_opd_outcome_statistics(
        advantage=torch.tensor([[5.0]]),
        response_mask=torch.zeros((1, 1), dtype=torch.bool),
        rm_scores=torch.zeros((1, 1)),
    )

    assert all(metric.aggregate() == 0.0 for metric in sufficient.values())


def test_random_pg_opd_preserves_sign_on_selected_tokens(monkeypatch):
    _identity_unpadding(monkeypatch)
    model_output, data = _data(student=[-0.2, -2.0], teacher=[-1.2, -1.0])

    reverse_kl, _ = losses.compute_random_sampled_token_reverse_kl(
        None, _config(random_token_ratio=1.0), model_output, data
    )

    advantage = -reverse_kl
    torch.testing.assert_close(advantage, torch.tensor([[-1.0, 1.0]]))


def test_topgap_ranks_by_absolute_gap_but_preserves_selected_sign(monkeypatch):
    _identity_unpadding(monkeypatch)
    model_output, data = _data(student=[-0.1, -2.0], teacher=[-2.0, -1.0])

    reverse_kl, _ = losses.compute_topgap_sampled_token_reverse_kl(
        None,
        _config(topgap_token_ratio=0.5, topgap_selection="top"),
        model_output,
        data,
    )

    # Token 0 has the larger absolute gap (1.9), but its signed advantage is
    # negative because the student assigns it more probability than the teacher.
    advantage = -reverse_kl
    torch.testing.assert_close(advantage, torch.tensor([[-1.9, 0.0]]))
