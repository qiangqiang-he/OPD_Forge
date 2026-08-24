"""Outcome-aware OPD step mapping and answer-probe construction.

The central invariant in this module is that rollout prefixes are always
spliced from the original prompt/response token IDs.  Text is used only to
discover semantic boundaries; it is never re-tokenized to recover an index in
the rollout.
"""

from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from utils.split_answer_probe_steps import Span, split_semantic_hybrid


PROBE_PREFIX = "\n\nTherefore, the answer is \\boxed{"
PROBE_SUFFIX = "}"

FINAL_ANSWER_HEADING_RE = re.compile(
    r"(?im)^\s*(?:#{1,6}\s*)?(?:[✅☑🎯]\s*)?(?:\*{1,2})?"
    r"(?:final\s+(?:correct\s+)?answer|final\s+result|conclusion|最终答案|答案|结论)"
    r"\s*(?:\*{1,2})?\s*[:：]?"
)
DIRECT_ANSWER_RE = re.compile(
    r"(?ix)(?:"
    r"\b(?:therefore|thus|hence|consequently|finally)\b[,\s:]*"
    r"(?:the\s+)?(?:final\s+)?(?:answer|result|value)\s*(?:is\b|equals\b|=|:)|"
    r"\b(?:the\s+)?final\s+(?:correct\s+)?answer\s*(?:is\b|equals\b|=|:)|"
    r"\bthe\s+answer\s*(?:is\b|equals\b|=|:)|"
    r"(?:因此|所以|故)[，,\s:：]*(?:最终)?(?:答案|结果|所求值)\s*(?:为|是|=|:|：)|"
    r"(?:最终答案|答案)\s*(?:为|是|=|:|：)"
    r")"
)


@dataclass(frozen=True)
class TokenStep:
    """One semantic response step aligned to original response token IDs."""

    index: int
    char_start: int
    char_end: int
    token_start: int
    token_end: int
    text: str


@dataclass(frozen=True)
class AnswerProbe:
    """A separately tokenized probe and its scored token positions."""

    text: str
    token_ids: list[int]
    answer_token_positions: list[int]
    answer_char_start: int
    answer_char_end: int


@dataclass(frozen=True)
class OAOPDComputation:
    """Per-rollout OA-OPD data transported to actor training."""

    token_weights: list[float]
    num_steps: int
    active_steps: int
    probe_failures: int
    values: tuple[Optional[float], ...]
    deltas: tuple[Optional[float], ...]


def _decode(
    tokenizer: Any,
    token_ids: list[int],
    *,
    skip_special_tokens: bool,
) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=False,
    )


def _find_original_token_boundary(
    tokenizer: Any,
    token_ids: list[int],
    decoded_text: str,
    char_target: int,
    *,
    minimum_token_end: int,
    minimum_char_end: int,
    skip_special_tokens: bool,
    cache: dict[int, str],
) -> tuple[int, int]:
    """Find the first original-token prefix covering a character boundary.

    Prefix decoding, including the bounded fallback scan, operates solely on
    ``token_ids``.  This deliberately avoids the invalid shortcut of
    tokenizing ``decoded_text[:char_target]``.
    """

    if not 0 <= char_target <= len(decoded_text):
        raise ValueError(
            f"Character boundary {char_target} is outside decoded text of length {len(decoded_text)}."
        )
    minimum_token_end = max(0, int(minimum_token_end))
    required_chars = max(int(char_target), int(minimum_char_end))

    def decode_prefix(end: int) -> str:
        if end not in cache:
            cache[end] = _decode(
                tokenizer,
                token_ids[:end],
                skip_special_tokens=skip_special_tokens,
            )
        return cache[end]

    # Length gives a very accurate binary-search approximation for BPE/SentencePiece
    # tokenizers.  Temporary byte-fallback replacement characters are filtered by
    # the exact prefix check below.
    low, high = minimum_token_end, len(token_ids)
    while low < high:
        middle = (low + high) // 2
        if len(decode_prefix(middle)) < required_chars:
            low = middle + 1
        else:
            high = middle
    approximate = low

    def candidates(start: int, end: int) -> list[tuple[int, int]]:
        found = []
        for token_end in range(max(minimum_token_end, start), min(len(token_ids), end) + 1):
            prefix = decode_prefix(token_end)
            if (
                len(prefix) >= required_chars
                and decoded_text.startswith(prefix)
            ):
                found.append((token_end, len(prefix)))
        return found

    found = candidates(approximate - 16, approximate + 16)
    if not found:
        found = candidates(approximate - 128, approximate + 128)
    if not found:
        # Rare slow-tokenizer/byte-fallback path.  It remains correct and still
        # never reconstructs an index by re-tokenizing response text.
        found = candidates(minimum_token_end, len(token_ids))
    if not found:
        raise ValueError(
            "No original-token prefix can cover semantic character boundary "
            f"{char_target} (required_chars={required_chars})."
        )
    return min(found, key=lambda item: (item[1], item[0]))


def map_response_steps_to_original_tokens(
    tokenizer: Any,
    response_ids: list[int],
) -> tuple[str, list[TokenStep]]:
    """Split a decoded response and align every step to original token IDs.

    The returned character and token intervals are monotone, non-overlapping,
    and contiguous over the visible response.  A tokenizer token that crosses
    a proposed semantic boundary is assigned wholly to the preceding step and
    the stored character boundary is moved to that token's actual decoded end.
    """

    response_ids = [int(token_id) for token_id in response_ids]
    response = _decode(tokenizer, response_ids, skip_special_tokens=True)
    semantic_spans = split_semantic_hybrid(response)
    if not semantic_spans:
        return response, []

    cache: dict[int, str] = {0: ""}
    token_steps: list[TokenStep] = []
    previous_token_end = 0
    previous_char_end = 0
    for semantic in semantic_spans:
        if semantic.end <= previous_char_end:
            # A tokenizer token may straddle a very short semantic fragment.
            # Such a fragment has no independent original-token interval.
            continue
        token_end, char_end = _find_original_token_boundary(
            tokenizer,
            response_ids,
            response,
            semantic.end,
            minimum_token_end=previous_token_end,
            minimum_char_end=previous_char_end + 1,
            skip_special_tokens=True,
            cache=cache,
        )
        if token_end <= previous_token_end or char_end <= previous_char_end:
            # A semantic fragment entirely swallowed by a boundary-crossing
            # token is an empty step and is intentionally skipped.
            continue
        token_steps.append(
            TokenStep(
                index=len(token_steps),
                char_start=previous_char_end,
                char_end=char_end,
                token_start=previous_token_end,
                token_end=token_end,
                text=response[previous_char_end:char_end],
            )
        )
        previous_token_end = token_end
        previous_char_end = char_end

    if previous_char_end != len(response):
        raise RuntimeError(
            "Semantic step mapping did not recover the complete visible response: "
            f"covered {previous_char_end} of {len(response)} characters."
        )
    if "".join(step.text for step in token_steps) != response:
        raise RuntimeError("Mapped OA-OPD steps do not exactly recover the decoded response.")
    return response, token_steps


def _compact_answer_text(value: str) -> str:
    value = value.replace("\\left", "").replace("\\right", "")
    value = value.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    value = value.replace("$", "").replace(r"\(", "").replace(r"\)", "")
    value = value.replace(r"\[", "").replace(r"\]", "")
    return re.sub(r"\s+", "", value).strip(".,;!?，。；！").lower()


def _direct_answer_contains_gold(step_text: str, answer: str) -> bool:
    compact_answer = _compact_answer_text(str(answer))
    if not compact_answer:
        return False
    for match in DIRECT_ANSWER_RE.finditer(step_text):
        followup = _compact_answer_text(step_text[match.end() : match.end() + 256])
        if compact_answer in followup:
            return True
    return False


def find_final_answer_step(steps: list[TokenStep], answer: str) -> Optional[int]:
    """Locate the final-answer disclosure step to exclude from OA-OPD.

    Explicit final-answer headings take precedence.  Otherwise the last boxed
    step is treated as the answer step, matching no-thinking math rollouts that
    may contain intermediate boxed identities before their final conclusion.
    """

    explicit = [
        step.index
        for step in steps
        if FINAL_ANSWER_HEADING_RE.search(step.text)
        or _direct_answer_contains_gold(step.text, answer)
    ]
    if explicit:
        return min(explicit)
    boxed = [step.index for step in steps if r"\boxed" in step.text]
    return max(boxed) if boxed else None


def reasoning_steps_before_answer(
    steps: list[TokenStep],
    answer: str,
) -> list[TokenStep]:
    answer_step = find_final_answer_step(steps, answer)
    if answer_step is None:
        return steps
    return [step for step in steps if step.index < answer_step]


def _probe_offsets_fallback(
    tokenizer: Any,
    probe_ids: list[int],
    probe_text: str,
) -> list[tuple[int, int]]:
    """Recover probe-token character extents from the already encoded IDs."""

    boundaries = [0]
    for token_end in range(1, len(probe_ids) + 1):
        decoded = _decode(
            tokenizer,
            probe_ids[:token_end],
            skip_special_tokens=False,
        )
        # Byte-fallback prefixes can temporarily contain a replacement
        # character.  Holding the preceding valid boundary assigns the full
        # recovered character span to the token that completes the bytes.
        boundaries.append(
            len(decoded) if probe_text.startswith(decoded) else boundaries[-1]
        )
    if boundaries[-1] != len(probe_text):
        raise ValueError("Cannot recover probe offsets from its original token IDs.")
    return list(zip(boundaries[:-1], boundaries[1:], strict=True))


def build_answer_probe(tokenizer: Any, answer: str) -> AnswerProbe:
    """Tokenize the complete appended probe once and mark answer-overlap tokens."""

    answer = str(answer)
    if not answer:
        raise ValueError("OA-OPD cannot construct an answer probe for an empty dataset answer.")
    probe_text = f"{PROBE_PREFIX}{answer}{PROBE_SUFFIX}"
    answer_char_start = len(PROBE_PREFIX)
    answer_char_end = answer_char_start + len(answer)

    encoded = tokenizer(
        probe_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    probe_ids = [int(token_id) for token_id in encoded["input_ids"]]
    offsets = encoded.get("offset_mapping")
    if offsets is None:
        offsets = _probe_offsets_fallback(tokenizer, probe_ids, probe_text)
    offsets = [(int(start), int(end)) for start, end in offsets]
    if len(offsets) != len(probe_ids):
        raise ValueError("Probe offset mapping and token IDs have different lengths.")
    if _decode(tokenizer, probe_ids, skip_special_tokens=False) != probe_text:
        raise ValueError("The complete OA-OPD answer probe does not round-trip through the tokenizer.")

    # Overlap, rather than containment, is intentional.  A token such as
    # ``{4`` overlaps answer character ``4`` and must be scored, whereas a
    # standalone closing brace begins at answer_char_end and is excluded.
    answer_positions = [
        index
        for index, (start, end) in enumerate(offsets)
        if end > answer_char_start and start < answer_char_end
    ]
    if not answer_positions:
        raise ValueError("OA-OPD probe tokenization produced no token overlapping the answer text.")
    return AnswerProbe(
        text=probe_text,
        token_ids=probe_ids,
        answer_token_positions=answer_positions,
        answer_char_start=answer_char_start,
        answer_char_end=answer_char_end,
    )


def intervention_weight(delta: float, *, tau: float, beta: float) -> float:
    """Compute ``1 - exp(-beta * [tau - delta]_+)`` stably."""

    margin = max(float(tau) - float(delta), 0.0)
    if margin <= 0.0:
        return 0.0

    weight = float(-math.expm1(-float(beta) * margin))
    # OA weights are transported as float32 tensors.  For a sufficiently large
    # margin, even the stable expm1 expression rounds to exactly 1.0; cap at
    # float32's predecessor of one so the implemented value preserves the
    # mathematical [0, 1) range after transport.
    max_float32_below_one = float.fromhex("0x1.fffffep-1")
    return min(weight, max_float32_below_one)


async def compute_oa_opd_for_rollout(
    *,
    tokenizer: Any,
    teacher_probe: Callable[..., Awaitable[float]],
    prompt_ids: list[int],
    response_ids: list[int],
    answer: str,
    tau: float,
    beta: float,
    probe_batch_size: int,
    routing_key: Optional[str] = None,
) -> OAOPDComputation:
    """Compute batched boundary probes and map step weights to response tokens."""

    if not math.isfinite(float(tau)):
        raise ValueError(f"OA-OPD tau must be finite, got {tau}.")
    if not math.isfinite(float(beta)) or float(beta) <= 0.0:
        raise ValueError(f"OA-OPD beta must be finite and positive, got {beta}.")
    if not 1 <= int(probe_batch_size) <= 8:
        raise ValueError(
            f"OA-OPD probe_batch_size must lie in [1, 8], got {probe_batch_size}."
        )

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
    prefix_token_ends = [0, *[step.token_end for step in reasoning_steps]]

    async def run_probe(response_token_end: int) -> float:
        prefix = [*prompt_ids, *response_ids[:response_token_end]]
        answer_positions = [
            len(prefix) + position for position in probe.answer_token_positions
        ]
        return await teacher_probe(
            sequence_ids=prefix + probe.token_ids,
            answer_token_positions=answer_positions,
            routing_key=routing_key,
        )

    values: list[Optional[float]] = []
    failures = 0
    for start in range(0, len(prefix_token_ends), int(probe_batch_size)):
        chunk = prefix_token_ends[start : start + int(probe_batch_size)]
        results = await asyncio.gather(
            *(run_probe(token_end) for token_end in chunk),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                failures += 1
                values.append(None)
                continue
            score = float(result)
            if not math.isfinite(score):
                failures += 1
                values.append(None)
            else:
                values.append(score)

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
    "AnswerProbe",
    "OAOPDComputation",
    "PROBE_PREFIX",
    "PROBE_SUFFIX",
    "TokenStep",
    "build_answer_probe",
    "compute_oa_opd_for_rollout",
    "find_final_answer_step",
    "intervention_weight",
    "map_response_steps_to_original_tokens",
    "reasoning_steps_before_answer",
]
