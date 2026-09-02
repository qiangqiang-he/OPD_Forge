"""Regression tests for causal teacher-response and diagnostic alignment."""

import asyncio
import math
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from verl.experimental.agent_loop.agent_loop import _compute_topk_overlap_metrics
from verl.experimental.teacher_loop.teacher_manager import (
    AsyncTeacherLLMServerManager,
    _align_teacher_response_outputs,
    _build_causal_response_layout,
    _pad_teacher_outputs,
    _slice_response_prediction_outputs,
)
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding


def _nested(rows: list[torch.Tensor]) -> torch.Tensor:
    return torch.nested.as_nested_tensor(rows, layout=torch.jagged)


def _response_values_from_loss_layout(
    full_sequence_values: torch.Tensor,
    *,
    student_prompt_ids: list[int],
    response_ids: list[int],
) -> torch.Tensor:
    data = TensorDict(
        {
            "prompts": _nested([torch.tensor(student_prompt_ids)]),
            "responses": _nested([torch.tensor(response_ids)]),
        },
        batch_size=[1],
    )
    return no_padding_2_padding(_nested([full_sequence_values]), data)


def test_vllm_extractor_to_loss_preserves_every_response_token():
    from verl.workers.rollout.vllm_rollout.utils import extract_prompt_logprobs

    # Teacher prompt [10, 11, 12] and response [20, 21, 22]. The extractor
    # stores token i's distribution at row i-1 and appends a trailing dummy.
    sequence_ids = [10, 11, 12, 20, 21, 22]
    token_logprobs = [-1.1, -1.2, -2.0, -2.1, -2.2]
    prompt_logprobs = [None]
    for token_id, logprob in zip(sequence_ids[1:], token_logprobs, strict=True):
        prompt_logprobs.append({token_id: SimpleNamespace(logprob=logprob, rank=1)})
    output = SimpleNamespace(
        prompt_token_ids=sequence_ids,
        prompt_logprobs=prompt_logprobs,
    )
    extracted: dict[str, list] = {}
    extract_prompt_logprobs(output, num_prompt_logprobs=1, result_dict=extracted)

    raw_ids = torch.tensor(extracted["prompt_sampled_ids"], dtype=torch.int32)
    raw_logprobs = torch.tensor(extracted["prompt_sampled_logprobs"])
    torch.testing.assert_close(
        raw_ids.squeeze(-1),
        torch.tensor([11, 12, 20, 21, 22, 0], dtype=torch.int32),
    )

    # The student prompt deliberately has a different length from the teacher.
    aligned_ids, aligned_logprobs = _align_teacher_response_outputs(
        raw_ids,
        raw_logprobs,
        teacher_sequence_length=len(sequence_ids),
        student_prompt_length=2,
        response_length=3,
    )
    torch.testing.assert_close(
        aligned_ids.squeeze(-1),
        torch.tensor([0, 20, 21, 22, 0], dtype=torch.int32),
    )
    torch.testing.assert_close(
        aligned_logprobs.squeeze(-1),
        torch.tensor([0.0, -2.0, -2.1, -2.2, 0.0]),
    )

    loss_ids = _response_values_from_loss_layout(
        aligned_ids,
        student_prompt_ids=[100, 101],
        response_ids=[20, 21, 22],
    )
    loss_logprobs = _response_values_from_loss_layout(
        aligned_logprobs,
        student_prompt_ids=[100, 101],
        response_ids=[20, 21, 22],
    )
    torch.testing.assert_close(
        loss_ids.squeeze(0).squeeze(-1),
        torch.tensor([20, 21, 22], dtype=torch.int32),
    )
    torch.testing.assert_close(
        loss_logprobs.squeeze(0).squeeze(-1),
        torch.tensor([-2.0, -2.1, -2.2]),
    )


def test_single_token_response_keeps_its_only_teacher_prediction():
    raw_ids = torch.tensor([[11], [20], [0]], dtype=torch.int32)
    raw_logprobs = torch.tensor([[-1.1], [-2.0], [0.0]])

    aligned_ids, aligned_logprobs = _align_teacher_response_outputs(
        raw_ids,
        raw_logprobs,
        teacher_sequence_length=3,
        student_prompt_length=4,
        response_length=1,
    )

    torch.testing.assert_close(
        aligned_ids.squeeze(-1),
        torch.tensor([0, 0, 0, 20, 0], dtype=torch.int32),
    )
    loss_logprobs = _response_values_from_loss_layout(
        aligned_logprobs,
        student_prompt_ids=[100, 101, 102, 103],
        response_ids=[20],
    )
    assert loss_logprobs.item() == pytest.approx(-2.0)


def test_left_right_padded_batch_preserves_response_alignment(monkeypatch):
    # Keep this layout regression CPU-only; production uses equivalent
    # flash-attn helpers for these two mechanical indexing operations.
    def fake_unpad_input(values, attention_mask):
        indices = attention_mask.flatten().nonzero(as_tuple=False).flatten()
        unpadded = values.flatten(0, 1).index_select(0, indices)
        sequence_lengths = attention_mask.sum(dim=-1, dtype=torch.int32)
        cumulative_lengths = torch.nn.functional.pad(sequence_lengths.cumsum(dim=0), (1, 0))
        return unpadded, indices, cumulative_lengths, int(sequence_lengths.max())

    monkeypatch.setattr(
        "verl.workers.utils.padding.unpad_input",
        fake_unpad_input,
    )
    monkeypatch.setattr(
        "verl.workers.utils.padding.index_first_axis",
        lambda values, indices: values.index_select(0, indices),
    )

    raw_ids = torch.tensor([[11], [20], [21], [0]], dtype=torch.int32)
    raw_logprobs = torch.tensor([[-1.1], [-2.0], [-2.1], [0.0]])
    aligned_ids, aligned_logprobs = _align_teacher_response_outputs(
        raw_ids,
        raw_logprobs,
        teacher_sequence_length=4,
        student_prompt_length=2,
        response_length=2,
    )
    padded_ids, padded_logprobs = _pad_teacher_outputs(
        aligned_ids,
        aligned_logprobs,
        prompt_width=4,
        response_width=3,
        prompt_length=2,
        response_length=2,
        pad_token_id=0,
    )
    data = TensorDict(
        {
            "prompts": torch.tensor([[0, 0, 100, 101]]),
            "responses": torch.tensor([[20, 21, 0]]),
            "input_ids": torch.tensor([[0, 0, 100, 101, 20, 21, 0]]),
            "attention_mask": torch.tensor([[0, 0, 1, 1, 1, 1, 0]]),
            "position_ids": torch.tensor([[0, 0, 0, 1, 2, 3, 0]]),
            "response_mask": torch.tensor([[1, 1, 0]]),
            "teacher_ids": padded_ids,
            "teacher_logprobs": padded_logprobs,
        },
        batch_size=[1],
    )

    unpadded = left_right_2_no_padding(data)
    loss_ids = no_padding_2_padding(unpadded["teacher_ids"], unpadded)
    loss_logprobs = no_padding_2_padding(unpadded["teacher_logprobs"], unpadded)

    torch.testing.assert_close(
        loss_ids.squeeze(0).squeeze(-1),
        torch.tensor([20, 21, 0], dtype=torch.int32),
    )
    torch.testing.assert_close(
        loss_logprobs.squeeze(0).squeeze(-1),
        torch.tensor([-2.0, -2.1, 0.0]),
    )


def test_response_prediction_slice_requires_a_prompt_token():
    with pytest.raises(ValueError, match="at least one prompt token"):
        _slice_response_prediction_outputs(torch.zeros((2, 1)), response_length=2)


def test_sampled_teacher_runtime_invariant_rejects_semantically_wrong_ids():
    class FakeClient:
        async def generate(self, **_kwargs):
            return SimpleNamespace(
                extra_fields={
                    "prompt_ids": [[20, 19], [0, 0]],
                    "prompt_logprobs": [[-0.1, -1.0], [0.0, 0.0]],
                    # The first row should contain response token 20.
                    "prompt_sampled_ids": [[999], [0]],
                    "prompt_sampled_logprobs": [[-2.0], [0.0]],
                }
            )

    manager = object.__new__(AsyncTeacherLLMServerManager)
    manager.teacher_model_configs = {"teacher": SimpleNamespace(inference=SimpleNamespace(temperature=1.0))}
    manager._teacher_prompt_lengths = {"teacher": 2048}
    manager.teacher_client = {"teacher": FakeClient()}
    manager.distillation_loss_config = SimpleNamespace(
        topk=None,
        diagnostic_topk=2,
        loss_settings=SimpleNamespace(use_topk=False),
    )

    with pytest.raises(RuntimeError, match="alignment invariant failed"):
        asyncio.run(
            manager.compute_teacher_logprobs_single(
                sequence_ids=[10, 20],
                student_prompt_length=1,
                response_length=1,
            )
        )


def test_wandb_topk_diagnostics_exclude_dummy_row_and_include_first_response_token():
    # Full causal layouts for prompt length 2 and response length 2. The final
    # all-zero row is the extractor dummy and must never enter W&B diagnostics.
    student_ids = torch.tensor([[99, 98], [10, 11], [20, 21], [0, 0]])
    teacher_response_ids = torch.tensor([[10, 12], [20, 22]])
    teacher_ids = _build_causal_response_layout(
        teacher_response_ids,
        student_prompt_length=2,
    )
    student_logprobs = torch.tensor([[-3.0, -4.0], [-0.2, -1.0], [-0.3, -1.1], [0.0, 0.0]])
    teacher_response_logprobs = torch.tensor([[-0.1, -1.2], [-0.4, -1.3]])
    teacher_logprobs = _build_causal_response_layout(
        teacher_response_logprobs,
        student_prompt_length=2,
    )

    student_response_ids = _slice_response_prediction_outputs(student_ids, 2)
    teacher_response_ids = _slice_response_prediction_outputs(teacher_ids, 2)
    student_response_logprobs = _slice_response_prediction_outputs(student_logprobs, 2)
    teacher_response_logprobs = _slice_response_prediction_outputs(teacher_logprobs, 2)
    overlap, mass_overlap = _compute_topk_overlap_metrics(
        student_response_ids,
        student_response_logprobs,
        teacher_response_ids,
        teacher_response_logprobs,
    )

    torch.testing.assert_close(
        student_response_ids,
        torch.tensor([[10, 11], [20, 21]]),
    )
    torch.testing.assert_close(
        teacher_response_ids,
        torch.tensor([[10, 12], [20, 22]]),
    )
    assert overlap == pytest.approx(0.5)
    assert mass_overlap == pytest.approx((math.exp(-0.2) + math.exp(-0.4)) / 2)

    response_max_logprobs = teacher_response_logprobs.max(dim=-1, keepdim=True).values
    full_max_logprobs = _build_causal_response_layout(
        response_max_logprobs,
        student_prompt_length=2,
    )
    loss_max_logprobs = _response_values_from_loss_layout(
        full_max_logprobs,
        student_prompt_ids=[100, 101],
        response_ids=[20, 21],
    )
    torch.testing.assert_close(
        loss_max_logprobs.squeeze(0).squeeze(-1),
        torch.tensor([-0.1, -0.4]),
    )
