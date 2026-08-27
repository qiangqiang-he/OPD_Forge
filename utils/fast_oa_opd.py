"""Single-forward, attention-masked outcome-aware OPD probes.

The reference implementation in :mod:`utils.oa_opd` launches one Teacher
request for every semantic-step boundary.  This module preserves that file as
the numerical reference and packs all boundary probes after the untouched
rollout instead::

    [prompt, response, probe_0, probe_1, ..., probe_K]

Every probe is an independent attention branch.  Probe ``k`` can attend only
to the original rollout prefix ending at boundary ``k`` and to its own causal
probe prefix.  Its position IDs start at the logical length of that rollout
prefix, so it is equivalent to an independently evaluated
``[prompt, response[:boundary_k], probe_k]`` sequence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from utils.oa_opd import (
    AnswerProbe,
    OAOPDComputation,
    TokenStep,
    build_answer_probe,
    intervention_weight,
    map_response_steps_to_original_tokens,
    reasoning_steps_before_answer,
)


@dataclass(frozen=True)
class MaskedProbeBranch:
    """Physical and logical layout of one appended answer-probe branch."""

    index: int
    visible_prefix_end: int
    probe_start: int
    probe_end: int
    answer_token_positions: tuple[int, ...]


@dataclass(frozen=True)
class MaskedProbeLayout:
    """One packed sequence plus branch metadata for a masked Teacher forward."""

    sequence_ids: tuple[int, ...]
    position_ids: tuple[int, ...]
    original_sequence_length: int
    branches: tuple[MaskedProbeBranch, ...]


def build_masked_probe_layout(
    *,
    prompt_ids: list[int],
    response_ids: list[int],
    reasoning_steps: list[TokenStep],
    probe: AnswerProbe,
) -> MaskedProbeLayout:
    """Append every boundary probe after the untouched original rollout.

    The original prompt/response IDs retain their ordinary causal positions.
    Probe positions ignore both the response suffix beyond their boundary and
    all physically preceding probes.
    """

    prompt_ids = [int(token_id) for token_id in prompt_ids]
    response_ids = [int(token_id) for token_id in response_ids]
    if not prompt_ids:
        raise ValueError("Fast OA-OPD requires at least one prompt token.")
    if not probe.token_ids:
        raise ValueError("Fast OA-OPD requires a non-empty answer probe.")
    if not probe.answer_token_positions:
        raise ValueError("Fast OA-OPD requires answer-token positions in its probe.")
    if min(probe.answer_token_positions) <= 0:
        raise ValueError(
            "Fast OA-OPD answer tokens must have a preceding token inside their probe."
        )

    response_length = len(response_ids)
    previous_end = 0
    for expected_index, step in enumerate(reasoning_steps):
        if step.index != expected_index:
            raise ValueError(
                "Fast OA-OPD reasoning steps must retain consecutive source indices."
            )
        if not 0 <= step.token_start < step.token_end <= response_length:
            raise ValueError(f"Invalid reasoning token interval: {step}.")
        if step.token_start != previous_end:
            raise ValueError(
                "Fast OA-OPD reasoning steps must be contiguous from response token zero."
            )
        previous_end = step.token_end

    original_ids = [*prompt_ids, *response_ids]
    sequence_ids = list(original_ids)
    position_ids = list(range(len(original_ids)))
    branches: list[MaskedProbeBranch] = []
    response_boundary_ends = [0, *[step.token_end for step in reasoning_steps]]
    probe_length = len(probe.token_ids)

    for index, response_boundary_end in enumerate(response_boundary_ends):
        visible_prefix_end = len(prompt_ids) + response_boundary_end
        probe_start = len(sequence_ids)
        probe_end = probe_start + probe_length
        sequence_ids.extend(probe.token_ids)
        position_ids.extend(
            range(visible_prefix_end, visible_prefix_end + probe_length)
        )
        answer_positions = tuple(
            probe_start + int(local_position)
            for local_position in probe.answer_token_positions
        )
        if any(position - 1 < probe_start for position in answer_positions):
            raise ValueError(
                "Fast OA-OPD cannot score an answer token whose causal predictor "
                "lies outside its own probe branch."
            )
        branches.append(
            MaskedProbeBranch(
                index=index,
                visible_prefix_end=visible_prefix_end,
                probe_start=probe_start,
                probe_end=probe_end,
                answer_token_positions=answer_positions,
            )
        )

    layout = MaskedProbeLayout(
        sequence_ids=tuple(sequence_ids),
        position_ids=tuple(position_ids),
        original_sequence_length=len(original_ids),
        branches=tuple(branches),
    )
    validate_masked_probe_layout(layout)
    return layout


def validate_masked_probe_layout(layout: MaskedProbeLayout) -> None:
    """Validate the packed sequence before it crosses the Teacher RPC boundary."""

    sequence_length = len(layout.sequence_ids)
    if sequence_length != len(layout.position_ids):
        raise ValueError("Fast OA-OPD sequence IDs and position IDs must align.")
    if not 1 <= layout.original_sequence_length <= sequence_length:
        raise ValueError("Fast OA-OPD original sequence length is invalid.")
    if tuple(layout.position_ids[: layout.original_sequence_length]) != tuple(
        range(layout.original_sequence_length)
    ):
        raise ValueError(
            "Fast OA-OPD must preserve the original rollout position IDs exactly."
        )

    previous_probe_end = layout.original_sequence_length
    for expected_index, branch in enumerate(layout.branches):
        if branch.index != expected_index:
            raise ValueError("Fast OA-OPD branch indices must be consecutive.")
        if branch.probe_start != previous_probe_end:
            raise ValueError("Fast OA-OPD probe branches must be contiguous.")
        if not (
            1
            <= branch.visible_prefix_end
            <= layout.original_sequence_length
            <= branch.probe_start
            < branch.probe_end
            <= sequence_length
        ):
            raise ValueError(f"Invalid Fast OA-OPD probe branch: {branch}.")
        expected_positions = tuple(
            range(
                branch.visible_prefix_end,
                branch.visible_prefix_end + branch.probe_end - branch.probe_start,
            )
        )
        if tuple(layout.position_ids[branch.probe_start : branch.probe_end]) != expected_positions:
            raise ValueError(
                f"Fast OA-OPD branch {branch.index} has incorrect logical position IDs."
            )
        if not branch.answer_token_positions:
            raise ValueError(f"Fast OA-OPD branch {branch.index} has no answer tokens.")
        if any(
            position <= branch.probe_start or position >= branch.probe_end
            for position in branch.answer_token_positions
        ):
            raise ValueError(
                f"Fast OA-OPD branch {branch.index} answer positions are not "
                "causally scoreable inside the branch."
            )
        previous_probe_end = branch.probe_end
    if previous_probe_end != sequence_length:
        raise ValueError("Fast OA-OPD packed sequence has unassigned trailing tokens.")


def build_dense_attention_mask(
    layout: MaskedProbeLayout,
    *,
    device: Any = None,
):
    """Build the exact boolean query/key visibility matrix for validation/tests.

    ``True`` means that a query row may attend to a key column, matching
    ``torch.nn.functional.scaled_dot_product_attention`` semantics.
    """

    import torch

    validate_masked_probe_layout(layout)
    sequence_length = len(layout.sequence_ids)
    mask = torch.zeros(
        (sequence_length, sequence_length), dtype=torch.bool, device=device
    )
    original_length = layout.original_sequence_length
    mask[:original_length, :original_length] = torch.ones(
        (original_length, original_length), dtype=torch.bool, device=device
    ).tril()
    for branch in layout.branches:
        mask[
            branch.probe_start : branch.probe_end,
            : branch.visible_prefix_end,
        ] = True
        probe_length = branch.probe_end - branch.probe_start
        mask[
            branch.probe_start : branch.probe_end,
            branch.probe_start : branch.probe_end,
        ] = torch.ones(
            (probe_length, probe_length), dtype=torch.bool, device=device
        ).tril()
    return mask


def layout_rpc_payload(layout: MaskedProbeLayout) -> dict[str, Any]:
    """Convert a validated layout to primitive Ray-serializable values."""

    validate_masked_probe_layout(layout)
    return {
        "sequence_ids": list(layout.sequence_ids),
        "position_ids": list(layout.position_ids),
        "original_sequence_length": int(layout.original_sequence_length),
        "branches": [
            {
                "index": int(branch.index),
                "visible_prefix_end": int(branch.visible_prefix_end),
                "probe_start": int(branch.probe_start),
                "probe_end": int(branch.probe_end),
                "answer_token_positions": list(branch.answer_token_positions),
            }
            for branch in layout.branches
        ],
    }


async def compute_fast_oa_opd_for_rollout(
    *,
    tokenizer: Any,
    teacher_masked_probe: Callable[..., Awaitable[list[float]]],
    prompt_ids: list[int],
    response_ids: list[int],
    answer: str,
    tau: float,
    beta: float,
    routing_key: Optional[str] = None,
) -> OAOPDComputation:
    """Compute all semantic-boundary values with one masked Teacher forward."""

    if not math.isfinite(float(tau)):
        raise ValueError(f"Fast OA-OPD tau must be finite, got {tau}.")
    if not math.isfinite(float(beta)) or float(beta) <= 0.0:
        raise ValueError(f"Fast OA-OPD beta must be finite and positive, got {beta}.")

    response_ids = [int(token_id) for token_id in response_ids]
    token_weights = [0.0] * len(response_ids)
    _, all_steps = map_response_steps_to_original_tokens(tokenizer, response_ids)
    reasoning_steps = reasoning_steps_before_answer(all_steps, str(answer))
    if not reasoning_steps:
        return OAOPDComputation(
            token_weights=token_weights,
            num_steps=0,
            active_steps=0,
            probe_failures=0,
            values=(),
            deltas=(),
        )

    probe = build_answer_probe(tokenizer, str(answer))
    layout = build_masked_probe_layout(
        prompt_ids=prompt_ids,
        response_ids=response_ids,
        reasoning_steps=reasoning_steps,
        probe=probe,
    )
    raw_values = await teacher_masked_probe(
        **layout_rpc_payload(layout),
        routing_key=routing_key,
    )
    if len(raw_values) != len(layout.branches):
        raise ValueError(
            "Fast OA-OPD Teacher returned the wrong number of boundary values: "
            f"got {len(raw_values)}, expected {len(layout.branches)}."
        )

    values: list[Optional[float]] = []
    failures = 0
    for raw_value in raw_values:
        value = float(raw_value)
        if math.isfinite(value):
            values.append(value)
        else:
            values.append(None)
            failures += 1

    active_steps = 0
    deltas: list[Optional[float]] = []
    for index, step in enumerate(reasoning_steps):
        before, after = values[index], values[index + 1]
        if before is None or after is None:
            deltas.append(None)
            continue
        delta = after - before
        deltas.append(delta)
        weight = intervention_weight(delta, tau=float(tau), beta=float(beta))
        if weight > 0.0:
            active_steps += 1
        for token_index in range(step.token_start, step.token_end):
            token_weights[token_index] = weight

    return OAOPDComputation(
        token_weights=token_weights,
        num_steps=len(reasoning_steps),
        active_steps=active_steps,
        probe_failures=failures,
        values=tuple(values),
        deltas=tuple(deltas),
    )


__all__ = [
    "MaskedProbeBranch",
    "MaskedProbeLayout",
    "build_dense_attention_mask",
    "build_masked_probe_layout",
    "compute_fast_oa_opd_for_rollout",
    "layout_rpc_payload",
    "validate_masked_probe_layout",
]
