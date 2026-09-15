"""Small CPU regression for R²OPL's two-backward gradient instrumentation."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import pytest
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu
from verl.utils.metric import AggregationType, Metric
from verl.workers.engine.fsdp import transformer_impl as fsdp_transformer_impl
from verl.workers.engine.fsdp.transformer_impl import FSDPEngine


class _TinyR2OPLEngine(FSDPEngine):
    """Use FSDPEngine's batching logic with a tiny ordinary module on CPU."""

    def __init__(self) -> None:
        self.module = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.module.weight.copy_(torch.tensor([[0.2, -0.4]]))
        self.optimizer = torch.optim.SGD(self.module.parameters(), lr=0.1)
        self.ulysses_sequence_parallel_size = 1
        self.ulysses_device_mesh = None
        self.scaler = None
        self._qat_enabled = False
        self.optimizer_config = SimpleNamespace(clip_grad=1.0)

    def get_data_parallel_size(self):
        return 1

    def get_data_parallel_group(self):
        return None

    def forward_step(self, micro_batch: TensorDict, loss_function, forward_only):
        log_probs = self.module(micro_batch["features"]).squeeze(-1)
        loss, metrics = loss_function(
            model_output={"log_probs": log_probs}, data=micro_batch, dp_group=None
        )
        return loss, {"model_output": {}, "loss": float(loss.detach()), "metrics": metrics}


def _loss_function(model_output, data: TensorDict, dp_group=None):
    del dp_group
    branch = tu.get_non_tensor_data(data, "r2opl_base_gradient_branch", "total")
    log_probs = model_output["log_probs"]
    if branch == "correct":
        advantage = data["correct_advantage"]
    elif branch == "error":
        advantage = data["error_advantage"]
    else:
        advantage = data["correct_advantage"] + data["error_advantage"]
    per_sequence = (-(advantage.detach() * log_probs)).mean(dim=-1)
    global_batch_size = tu.get_non_tensor_data(data, "global_batch_size", 4)
    loss = per_sequence.sum() / float(global_batch_size)
    return loss, {"distillation/loss": Metric(AggregationType.SUM, loss)}


def _data() -> TensorDict:
    features = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 1.0], [2.0, 0.0]],
            [[0.0, 2.0], [1.0, 2.0]],
            [[2.0, 1.0], [3.0, 1.0]],
        ]
    )
    correct_advantage = torch.tensor(
        [[0.5, 0.5], [0.0, 0.0], [0.75, 0.75], [0.0, 0.0]]
    )
    error_advantage = torch.tensor(
        [[0.0, 0.0], [0.02, -0.01], [0.0, 0.0], [-0.03, 0.04]]
    )
    data = TensorDict(
        {
            "features": features,
            "loss_mask": torch.ones((4, 2), dtype=torch.bool),
            "r2opl_base_correct_mask": correct_advantage.ne(0),
            "r2opl_base_error_mask": error_advantage.ne(0),
            "correct_advantage": correct_advantage,
            "error_advantage": error_advantage,
        },
        batch_size=4,
    )
    tu.assign_non_tensor(data, use_dynamic_bsz=False, micro_batch_size_per_gpu=2)
    return data


def test_two_branch_backwards_preserve_the_combined_training_gradient(monkeypatch):
    # The actual server uses NCCL/FSDP collectives.  This CPU test checks the
    # branch replay, gradient snapshot, merge, and metric transport path with
    # a world size of one, so its all-reduce is an identity.
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(fsdp_transformer_impl, "get_device_id", lambda: "cpu")
    engine = _TinyR2OPLEngine()
    data = _data()

    output = engine.forward_backward_batch(data, _loss_function, forward_only=False)
    merged_gradient = engine.module.weight.grad.detach().clone()

    reference = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        reference.weight.copy_(engine.module.weight.detach())
    features = data["features"]
    total_advantage = data["correct_advantage"] + data["error_advantage"]
    reference_loss = (-(total_advantage * reference(features).squeeze(-1))).mean(dim=-1).mean()
    reference_loss.backward()
    torch.testing.assert_close(merged_gradient, reference.weight.grad)

    pre_step_weight = engine.module.weight.detach().clone()
    reported_total_norm = engine.optimizer_step()
    assert reported_total_norm == pytest.approx(float(torch.linalg.vector_norm(merged_gradient)))
    assert not torch.equal(engine.module.weight.detach(), pre_step_weight)

    metrics = output["metrics"]
    assert metrics["r2opl_base/correct_grad_norm"][0] > 0.0
    assert metrics["r2opl_base/error_grad_norm"][0] > 0.0
    # The output loss is the sum of correct and error branch losses, rather
    # than the error-only scalar from the final branch replay.
    assert sum(output["loss"]) == pytest.approx(float(reference_loss.detach()))


def test_two_questions_eight_rollouts_complete_one_tiny_training_step(monkeypatch):
    """Run a full 2-question x 8-rollout optimizer step on the tiny engine."""

    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(fsdp_transformer_impl, "get_device_id", lambda: "cpu")
    question_count, rollouts_per_question = 2, 8
    batch_size, sequence_length = question_count * rollouts_per_question, 3
    rows = torch.arange(batch_size, dtype=torch.float32).view(-1, 1)
    positions = torch.arange(sequence_length, dtype=torch.float32).view(1, -1)
    features = torch.stack(
        (
            torch.ones((batch_size, sequence_length)),
            (rows + positions) / batch_size,
        ),
        dim=-1,
    )
    response_mask = torch.ones((batch_size, sequence_length), dtype=torch.bool)
    correct = torch.zeros((batch_size, sequence_length))
    error = torch.zeros_like(correct)
    rewards = torch.tensor(
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
         1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    for index, reward in enumerate(rewards):
        if reward > 0:
            correct[index] = 1.0 - float(index // rollouts_per_question) * 0.25
        else:
            error[index] = (torch.arange(sequence_length, dtype=torch.float32) + 1.0) * 0.01
    data = TensorDict(
        {
            "features": features,
            "loss_mask": response_mask,
            "r2opl_base_correct_mask": correct.ne(0),
            "r2opl_base_error_mask": error.ne(0),
            "correct_advantage": correct,
            "error_advantage": error,
        },
        batch_size=batch_size,
    )
    tu.assign_non_tensor(
        data,
        use_dynamic_bsz=False,
        micro_batch_size_per_gpu=4,
        global_batch_size=batch_size,
    )
    engine = _TinyR2OPLEngine()
    initial_weight = engine.module.weight.detach().clone()

    output = engine.train_batch(data, _loss_function)

    assert torch.isfinite(engine.module.weight).all()
    assert not torch.equal(engine.module.weight.detach(), initial_weight)
    assert output["metrics"]["r2opl_base/correct_grad_norm"][0] > 0.0
    assert output["metrics"]["r2opl_base/error_grad_norm"][0] > 0.0
    assert output["metrics"]["grad_norm"] > 0.0
