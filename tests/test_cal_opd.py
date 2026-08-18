"""Regression tests for Cal-OPD's signed, non-flipping calibration."""

from types import SimpleNamespace

import torch

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


def test_calibrated_loss_returns_negative_advantage_for_reinforce(monkeypatch):
    monkeypatch.setattr(losses, "no_padding_2_padding", lambda tensor, _data: tensor)
    model_output = {"log_probs": torch.tensor([[-2.0, -1.0]])}
    data = {
        "teacher_logprobs": torch.tensor([[[-1.0], [-2.0]]]),
        "cal_positive_teacher_logprobs": torch.tensor([[[-0.5], [-1.6]]]),
        "cal_negative_teacher_logprobs": torch.tensor([[[-1.4], [-2.5]]]),
        "response_mask": torch.ones((1, 2), dtype=torch.bool),
    }

    reverse_signal, _ = losses.compute_calibrated_sampled_token_reverse_kl(
        None, SimpleNamespace(), model_output, data
    )

    # The shared policy-gradient path applies another minus sign, yielding A_cal.
    torch.testing.assert_close(reverse_signal, torch.tensor([[-0.6, 0.6]]))
