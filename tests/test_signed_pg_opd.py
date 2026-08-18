"""Regression tests for signed policy-gradient OPD signals."""

from types import SimpleNamespace

import torch

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
