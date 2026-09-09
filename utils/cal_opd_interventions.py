"""Deterministic data construction for Cal-OPD interventions."""

from __future__ import annotations

import bisect
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any


def make_cal_opd_wrong_answer(
    correct_answer: Any,
    *,
    seed: int,
    record_index: int,
) -> str:
    """Randomly replace every digit while preserving answer structure.

    A per-record RNG makes the intervention reproducible regardless of dataset
    worker count or traversal order.  The first digit of every numeric run is
    kept non-zero so an ordinary integer retains its number of digits.  LaTeX
    punctuation and all other non-digit characters remain unchanged.
    """

    correct = str(correct_answer)
    rng = random.Random(f"cal-opd:{int(seed)}:{int(record_index)}:{correct}")
    output: list[str] = []
    previous_was_digit = False
    changed_digit = False
    for character in correct:
        if not character.isascii() or not character.isdigit():
            output.append(character)
            previous_was_digit = False
            continue

        choices = list("123456789" if not previous_was_digit else "0123456789")
        if character in choices:
            choices.remove(character)
        output.append(rng.choice(choices))
        previous_was_digit = True
        changed_digit = True

    if not changed_digit:
        wrong = rf"\left({correct}\right)+1"
    else:
        wrong = "".join(output)
    if wrong == correct:
        raise RuntimeError(f"Failed to construct a wrong Cal-OPD answer from {correct!r}.")
    return wrong


def compute_token_lengths(
    tokenizer,
    texts: Sequence[str],
    *,
    batch_size: int = 256,
) -> list[int]:
    """Return untruncated token lengths for complete text strings."""

    if batch_size <= 0:
        raise ValueError(f"Token-length batch_size must be positive, got {batch_size}.")
    lengths: list[int] = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            list(texts[start : start + batch_size]),
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )["input_ids"]
        lengths.extend(len(token_ids) for token_ids in encoded)
    if len(lengths) != len(texts):
        raise RuntimeError("Tokenizer returned the wrong number of Cal-OPD solution lengths.")
    return lengths


def choose_cal_opd_wrong_solution_indices(
    records: Sequence[Mapping[str, Any]],
    solution_token_lengths: Sequence[int],
) -> list[int]:
    """Choose a different problem's closest-length non-empty solution.

    Candidates with the same question or answer are excluded.  Distance is
    measured in complete tokenizer tokens, with source index as the stable
    tie-breaker.
    """

    if len(records) != len(solution_token_lengths):
        raise ValueError("Cal-OPD solution lengths must cover the complete dataset.")
    if not records:
        return []

    length_buckets: dict[int, list[int]] = defaultdict(list)
    for index, (record, raw_length) in enumerate(
        zip(records, solution_token_lengths, strict=True)
    ):
        if str(record.get("solution") or "").strip():
            length_buckets[int(raw_length)].append(index)
    available_lengths = sorted(length_buckets)
    if len(available_lengths) == 0:
        return [-1] * len(records)

    chosen: list[int] = []
    for target_index, (target_record, raw_target_length) in enumerate(
        zip(records, solution_token_lengths, strict=True)
    ):
        if not str(target_record.get("solution") or "").strip():
            chosen.append(-1)
            continue

        target_length = int(raw_target_length)
        insertion = bisect.bisect_left(available_lengths, target_length)
        left = insertion - 1
        right = insertion
        selected = -1
        while left >= 0 or right < len(available_lengths):
            left_distance = (
                target_length - available_lengths[left]
                if left >= 0
                else None
            )
            right_distance = (
                available_lengths[right] - target_length
                if right < len(available_lengths)
                else None
            )
            next_distance = min(
                distance
                for distance in (left_distance, right_distance)
                if distance is not None
            )
            candidate_lengths: list[int] = []
            if left_distance == next_distance:
                candidate_lengths.append(available_lengths[left])
                left -= 1
            if right_distance == next_distance:
                candidate_lengths.append(available_lengths[right])
                right += 1

            valid_candidates = [
                candidate_index
                for length in candidate_lengths
                for candidate_index in length_buckets[length]
                if candidate_index != target_index
                and str(records[candidate_index].get("question"))
                != str(target_record.get("question"))
                and str(records[candidate_index].get("answer"))
                != str(target_record.get("answer"))
            ]
            if valid_candidates:
                selected = min(valid_candidates)
                break

        if selected < 0:
            raise ValueError(
                f"No unrelated Cal-OPD solution candidate exists for record {target_index}."
            )
        chosen.append(selected)
    return chosen


__all__ = [
    "choose_cal_opd_wrong_solution_indices",
    "compute_token_lengths",
    "make_cal_opd_wrong_answer",
]
