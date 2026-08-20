"""Regression tests for Cal-OPD's signed, non-flipping calibration."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from algorithms import cal_opd as cal_opd_module
from algorithms.cal_opd import (
    CAL_OPD_TOKEN_TABLE_COLUMNS,
    CAL_OPD_TOKEN_TABLE_KEY,
    CalOPDTrainer,
    build_cal_opd_token_change_rows,
)
from verl.trainer.distillation import losses


def test_calibrated_advantage_uses_both_interventions_and_never_flips_sign():
    student = torch.tensor([[-2.0, -2.0, -1.0, -1.0, -1.0]])
    teacher = torch.tensor([[-1.0, -1.0, -2.0, -2.0, -1.0]])
    positive = torch.tensor([[-0.5, -1.2, -2.4, -0.5, -0.8]])
    negative = torch.tensor([[-1.4, -2.5, -1.6, -2.2, -1.3]])

    advantage, self_deviation = losses.compute_calibrated_opd_advantage(
        student, teacher, positive, negative
    )

    # The negative-labelled prompt supplies the upward shift on token 2, which
    # verifies that labels do not hard-code the likelihood-shift direction.
    torch.testing.assert_close(
        self_deviation, torch.tensor([[0.4, 1.5, 0.4, 1.5, 0.0]])
    )
    torch.testing.assert_close(
        advantage, torch.tensor([[0.6, 0.0, -0.6, 0.0, 0.0]])
    )
    base_advantage = teacher - student
    assert torch.all(advantage.sign() * base_advantage.sign() >= 0)
    assert torch.all(advantage.abs() <= base_advantage.abs())


def test_calibrated_loss_returns_negative_advantage_and_lightweight_metrics(monkeypatch):
    monkeypatch.setattr(losses, "no_padding_2_padding", lambda tensor, _data: tensor)
    model_output = {"log_probs": torch.tensor([[-2.0, -1.0]])}
    data = {
        "teacher_logprobs": torch.tensor([[[-1.0], [-2.0]]]),
        "cal_positive_teacher_logprobs": torch.tensor([[[-0.5], [-1.6]]]),
        "cal_negative_teacher_logprobs": torch.tensor([[[-1.4], [-2.5]]]),
        "response_mask": torch.ones((1, 2), dtype=torch.bool),
    }

    reverse_signal, metrics = losses.compute_calibrated_sampled_token_reverse_kl(
        None, None, model_output, data
    )

    # The shared policy-gradient path applies another minus sign, yielding A_cal.
    torch.testing.assert_close(reverse_signal, torch.tensor([[-0.6, 0.6]]))
    assert set(metrics) == {
        "distillation/reverse_kl_estimate",
        "distillation/student_sampled_token_prob",
        "distillation/teacher_sampled_token_prob",
        "distillation/cal_advantage_mean",
        "distillation/cal_advantage_abs_mean",
        "distillation/cal_zero_token_ratio",
        "distillation/cal_teacher_self_deviation_mean",
        "distillation/cal_retained_magnitude_ratio",
        "distillation/cal_positive_retained_magnitude_ratio",
        "distillation/cal_negative_retained_magnitude_ratio",
    }
    assert metrics["distillation/cal_advantage_mean"].aggregate() == 0.0
    assert metrics["distillation/cal_advantage_abs_mean"].aggregate() == pytest.approx(0.6)
    assert metrics["distillation/cal_zero_token_ratio"].aggregate() == 0.0
    assert metrics["distillation/cal_teacher_self_deviation_mean"].aggregate() == pytest.approx(0.4)
    assert metrics["distillation/cal_retained_magnitude_ratio"].aggregate() == pytest.approx(0.6)
    assert metrics["distillation/cal_positive_retained_magnitude_ratio"].aggregate() == pytest.approx(0.6)
    assert metrics["distillation/cal_negative_retained_magnitude_ratio"].aggregate() == pytest.approx(0.6)


class _FakeTokenizer:
    pad_token_id = 0

    pieces = {
        1: " but",
        2: "But",
        3: " AND",
        4: "or",
    }

    def batch_decode(self, token_ids, **_kwargs):
        return [self.pieces[token_id[0]] for token_id in token_ids]


def test_token_change_rows_merge_case_and_whitespace_before_ranking():
    response_token_ids = torch.tensor([[1, 2, 3, 4]])
    response_mask = torch.ones_like(response_token_ids, dtype=torch.bool)
    student = torch.zeros((1, 4))
    base_advantage = torch.tensor([[1.0, 1.0, -1.0, -1.0]])
    teacher = student + base_advantage
    positive_teacher = teacher + torch.tensor([[0.0, 0.0, 0.9, 0.2]])
    negative_teacher = teacher - torch.tensor([[0.6, 0.4, 0.0, 0.0]])

    rows = build_cal_opd_token_change_rows(
        response_token_ids=response_token_ids,
        response_mask=response_mask,
        student_log_probs=student,
        teacher_log_probs=teacher,
        positive_teacher_log_probs=positive_teacher,
        negative_teacher_log_probs=negative_teacher,
        tokenizer=_FakeTokenizer(),
        step=7,
        top_k=2,
    )

    assert [row[:4] for row in rows] == [
        [7, "Absolute", 1, "But"],
        [7, "Absolute", 2, "And"],
        [7, "Positive", 1, "But"],
        [7, "Negative", 1, "And"],
        [7, "Negative", 2, "Or"],
    ]
    torch.testing.assert_close(
        torch.tensor([row[4:] for row in rows]),
        torch.tensor(
            [
                [1.0, 2, 0.5, 0.6],
                [0.9, 1, 0.9, 0.9],
                [1.0, 2, 0.5, 0.6],
                [0.9, 1, 0.9, 0.9],
                [0.2, 1, 0.2, 0.2],
            ]
        ),
    )


def test_trainer_extracts_response_aligned_token_changes_from_nested_queue(monkeypatch):
    def nested(rows):
        return torch.nested.as_nested_tensor(rows, layout=torch.jagged)

    # Teacher fields retain the full prompt+response layout in TransferQueue;
    # old_log_probs and responses are already response-only jagged tensors.
    data = TensorDict(
        {
            "prompts": nested([torch.tensor([10, 11])]),
            "responses": nested([torch.tensor([1, 2, 3, 4])]),
            "response_mask": nested([torch.ones(4, dtype=torch.bool)]),
            "old_log_probs": nested([torch.zeros(4)]),
            "teacher_logprobs": nested(
                [torch.tensor([[0.0], [1.0], [1.0], [-1.0], [-1.0], [0.0]])]
            ),
            "cal_positive_teacher_logprobs": nested(
                [torch.tensor([[0.0], [1.0], [1.0], [-0.1], [-0.8], [0.0]])]
            ),
            "cal_negative_teacher_logprobs": nested(
                [torch.tensor([[0.0], [0.4], [0.6], [-1.0], [-1.0], [0.0]])]
            ),
        },
        batch_size=[1],
    )
    monkeypatch.setattr(
        cal_opd_module.verl_sync.tq,
        "kv_batch_get",
        lambda **_kwargs: data,
    )
    trainer = object.__new__(CalOPDTrainer)
    trainer.tokenizer = _FakeTokenizer()
    batch = SimpleNamespace(keys=["sample"], partition_id="train", tags=[{}])

    rows = trainer._build_token_change_rows(batch, step=9)

    assert rows[0][:4] == [9, "Absolute", 1, "But"]
    assert any(row[:4] == [9, "Positive", 1, "But"] for row in rows)
    assert any(row[:4] == [9, "Negative", 1, "And"] for row in rows)


def test_cal_metrics_use_one_capitalized_namespace_and_drop_sources():
    trainer = object.__new__(CalOPDTrainer)
    trainer.config = {
        "student_prompt": "qwen3_no_thinking_prompt",
        "teacher_prompt": "qwen3_no_thinking_prompt",
    }
    metrics = {
        "actor/distillation/cal_advantage_mean": 0.1,
        "actor/distillation/cal_advantage_abs_mean": 0.2,
        "actor/distillation/cal_zero_token_ratio": 0.3,
        "actor/distillation/cal_teacher_self_deviation_mean": 0.4,
        "actor/distillation/cal_retained_magnitude_ratio": 0.5,
        "actor/distillation/cal_positive_retained_magnitude_ratio": 0.6,
        "actor/distillation/cal_negative_retained_magnitude_ratio": 0.7,
        "actor/distillation/loss": 0.8,
    }

    trainer._add_opd_training_metrics(metrics)

    assert metrics == {
        "Cal-OPD/train/advantage_mean": 0.1,
        "Cal-OPD/train/advantage_abs_mean": 0.2,
        "Cal-OPD/train/zero_token_ratio": 0.3,
        "Cal-OPD/train/teacher_self_deviation_mean": 0.4,
        "Cal-OPD/train/retained_magnitude_ratio": 0.5,
        "Cal-OPD/train/positive_retained_magnitude_ratio": 0.6,
        "Cal-OPD/train/negative_retained_magnitude_ratio": 0.7,
        "Cal-OPD/train/policy_loss": 0.8,
    }


class _FakeTable:
    def __init__(self, columns, data=None):
        self.columns = columns
        self.data = list(data or [])

    def add_data(self, *row):
        self.data.append(list(row))


class _FakeWandb:
    Table = _FakeTable
    run = object()

    def __init__(self):
        self.logged = []

    def log(self, payload, step):
        self.logged.append((payload, step))


def test_cal_token_table_accumulates_steps_in_one_wandb_table():
    trainer = object.__new__(CalOPDTrainer)
    wandb = _FakeWandb()
    first_row = [1, "Absolute", 1, "But", 1.0, 2, 0.5, 0.6]
    second_row = [2, "Positive", 1, "And", 0.9, 1, 0.9, 0.9]

    trainer._log_token_change_rows([first_row], 1, wandb)
    trainer._log_token_change_rows([second_row], 2, wandb)

    table = trainer._cal_opd_token_change_table
    assert table.columns == list(CAL_OPD_TOKEN_TABLE_COLUMNS)
    assert table.data == [first_row, second_row]
    assert [step for _, step in wandb.logged] == [1, 2]
    assert all(CAL_OPD_TOKEN_TABLE_KEY in payload for payload, _ in wandb.logged)
