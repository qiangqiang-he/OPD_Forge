"""Pure selection helpers for the OA-OPD step-level evaluation pool.

The helpers in this module deliberately have no CUDA, vLLM, or Transformers
dependency.  They define the auditable parts of the response-selection
contract: which wrong answers are unambiguous, how a response earns a
positive/negative strength, and how four disjoint quota groups are selected
under one shared symmetric log-probability margin.
"""

from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Any, Iterable


_INTEGER_RE = re.compile(r"^[+-]?\d+$")
_DECIMAL_RE = re.compile(r"^[+-]?(?:\d+\.\d*|\d*\.\d+)$")
_PLAIN_FRACTION_RE = re.compile(r"^([+-]?\d+)/(\d+)$")
_LATEX_FRACTION_RE = re.compile(r"^\\frac\{([+-]?\d+)\}\{(\d+)\}$")


def parse_unambiguous_numeric_answer(value: Any) -> Fraction | None:
    """Parse only scalar numeric forms whose inequality is unambiguous.

    Symbolic expressions, sets, tuples, radicals, percentages, units, and
    other answers that may need a semantic equivalence checker are rejected.
    This intentionally conservative parser is used only for the *incorrect*
    source-response pool, where false-negative grading would contaminate the
    experiment.
    """

    text = str(value).strip()
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("$", "").replace(r"\(", "").replace(r"\)", "")
    text = re.sub(r"\s+", "", text)
    while len(text) >= 2 and text[0] == "{" and text[-1] == "}":
        text = text[1:-1]
    if _INTEGER_RE.fullmatch(text):
        return Fraction(int(text), 1)
    if _DECIMAL_RE.fullmatch(text):
        return Fraction(text)
    match = _PLAIN_FRACTION_RE.fullmatch(text)
    if match is None:
        match = _LATEX_FRACTION_RE.fullmatch(text)
    if match is not None:
        denominator = int(match.group(2))
        if denominator == 0:
            return None
        return Fraction(int(match.group(1)), denominator)
    return None


def clean_correct_source(record: dict[str, Any]) -> bool:
    """Return whether a recorded correct rollout is clean enough to use."""

    return (
        int(record.get("correct", 0)) == 1
        and int(record.get("format_valid", 0)) == 1
        and str(record.get("finish_reason", "")) == "stop"
    )


def strict_incorrect_source(record: dict[str, Any]) -> bool:
    """Accept only non-truncated, parseable, provably unequal numeric answers."""

    if (
        int(record.get("correct", 1)) != 0
        or int(record.get("format_valid", 0)) != 1
        or str(record.get("finish_reason", "")) != "stop"
    ):
        return False
    gold = parse_unambiguous_numeric_answer(record.get("answer", ""))
    extracted = parse_unambiguous_numeric_answer(record.get("extracted_answer", ""))
    return gold is not None and extracted is not None and gold != extracted


def second_step_strengths(deltas: Iterable[float]) -> tuple[float | None, float | None]:
    """Return the second strongest positive and negative magnitudes.

    A response qualifies at margin ``gamma`` exactly when the corresponding
    returned strength is at least ``gamma``.  Using the second order statistic
    encodes the requirement that every selected response contain at least two
    significant steps.
    """

    finite = [float(value) for value in deltas if math.isfinite(float(value))]
    positive = sorted((value for value in finite if value > 0.0), reverse=True)
    negative = sorted((-value for value in finite if value < 0.0), reverse=True)
    return (
        positive[1] if len(positive) >= 2 else None,
        negative[1] if len(negative) >= 2 else None,
    )


def _qualifies(row: dict[str, Any], sign: str, margin: float) -> bool:
    value = row.get(f"{sign}_strength")
    return value is not None and float(value) >= float(margin)


def feasibility_counts(
    rows: Iterable[dict[str, Any]], margin: float
) -> dict[str, dict[str, int]]:
    """Count sign-qualified responses and their union for each outcome."""

    materialized = list(rows)
    result: dict[str, dict[str, int]] = {}
    for outcome in ("correct", "incorrect"):
        subset = [row for row in materialized if row["source_outcome"] == outcome]
        positive = {int(row["record_index"]) for row in subset if _qualifies(row, "positive", margin)}
        negative = {int(row["record_index"]) for row in subset if _qualifies(row, "negative", margin)}
        result[outcome] = {
            "positive": len(positive),
            "negative": len(negative),
            "union": len(positive | negative),
            "both": len(positive & negative),
        }
    return result


def largest_feasible_symmetric_margin(
    rows: Iterable[dict[str, Any]], *, per_group: int
) -> tuple[float, dict[str, dict[str, int]]]:
    """Find the largest shared ``|delta|`` margin supporting four quotas.

    Responses must be disjoint between the positive and negative group within
    each source outcome.  With two signs, the three conditions
    ``positive>=k``, ``negative>=k``, and ``union>=2k`` are necessary and
    sufficient for such an assignment.
    """

    if per_group <= 0:
        raise ValueError("per_group must be positive.")
    materialized = list(rows)
    candidates = sorted(
        {
            float(value)
            for row in materialized
            for value in (row.get("positive_strength"), row.get("negative_strength"))
            if value is not None and math.isfinite(float(value)) and float(value) > 0.0
        },
        reverse=True,
    )
    for margin in candidates:
        counts = feasibility_counts(materialized, margin)
        if all(
            group["positive"] >= per_group
            and group["negative"] >= per_group
            and group["union"] >= 2 * per_group
            for group in counts.values()
        ):
            return margin, counts
    raise ValueError(
        "No positive symmetric margin can provide disjoint C+/C-/I+/I- quotas "
        f"of {per_group} responses."
    )


def _select_one_outcome(
    rows: list[dict[str, Any]], *, margin: float, per_group: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    positive_only = [
        row
        for row in rows
        if _qualifies(row, "positive", margin) and not _qualifies(row, "negative", margin)
    ]
    negative_only = [
        row
        for row in rows
        if _qualifies(row, "negative", margin) and not _qualifies(row, "positive", margin)
    ]
    both = [
        row
        for row in rows
        if _qualifies(row, "positive", margin) and _qualifies(row, "negative", margin)
    ]
    key = lambda sign: lambda row: (-float(row[f"{sign}_strength"]), int(row["record_index"]))
    positive_only.sort(key=key("positive"))
    negative_only.sort(key=key("negative"))
    both.sort(
        key=lambda row: (
            -(float(row["positive_strength"]) - float(row["negative_strength"])),
            int(row["record_index"]),
        )
    )

    # Reserve the minimum number of overlap candidates required by each sign.
    positive_overlap_need = max(0, per_group - len(positive_only))
    negative_overlap_need = max(0, per_group - len(negative_only))
    if positive_overlap_need + negative_overlap_need > len(both):
        raise ValueError("Overlapping candidates cannot satisfy disjoint sign quotas.")
    reserved_positive = both[:positive_overlap_need]
    remaining_both = both[positive_overlap_need:]
    reserved_negative = (
        remaining_both[-negative_overlap_need:] if negative_overlap_need else []
    )
    reserved_negative_ids = {int(row["record_index"]) for row in reserved_negative}

    positive_candidates = positive_only + reserved_positive + [
        row for row in remaining_both if int(row["record_index"]) not in reserved_negative_ids
    ]
    positive = sorted(positive_candidates, key=key("positive"))[:per_group]
    positive_ids = {int(row["record_index"]) for row in positive}
    # ``reserved_negative`` is already contained in ``both``.  Adding it
    # separately would duplicate the same response and could fill two quota
    # slots with one record when it has a high negative strength.
    negative_candidates = negative_only + [
        row for row in both if int(row["record_index"]) not in positive_ids
    ]
    negative = sorted(negative_candidates, key=key("negative"))[:per_group]
    if len(positive) != per_group or len(negative) != per_group:
        raise ValueError("Failed to fill disjoint positive/negative quotas.")
    if positive_ids & {int(row["record_index"]) for row in negative}:
        raise AssertionError("Positive and negative response selections overlap.")
    return positive, negative


def select_disjoint_balanced_responses(
    rows: Iterable[dict[str, Any]], *, margin: float, per_group: int
) -> dict[str, list[dict[str, Any]]]:
    """Select deterministic, disjoint C+/C-/I+/I- response groups."""

    materialized = list(rows)
    selected: dict[str, list[dict[str, Any]]] = {}
    labels = {("correct", "positive"): "C+", ("correct", "negative"): "C-",
              ("incorrect", "positive"): "I+", ("incorrect", "negative"): "I-"}
    for outcome in ("correct", "incorrect"):
        subset = [row for row in materialized if row["source_outcome"] == outcome]
        positive, negative = _select_one_outcome(
            subset, margin=margin, per_group=per_group
        )
        for sign, group in (("positive", positive), ("negative", negative)):
            label = labels[(outcome, sign)]
            selected[label] = [
                {
                    **row,
                    "selection_group": label,
                    "selected_sign": sign,
                    "selection_strength": float(row[f"{sign}_strength"]),
                    "significance_margin": float(margin),
                    "group_rank": rank,
                }
                for rank, row in enumerate(group, start=1)
            ]
    all_ids = [
        int(row["record_index"])
        for label in ("C+", "C-", "I+", "I-")
        for row in selected[label]
    ]
    if len(all_ids) != 4 * per_group or len(set(all_ids)) != len(all_ids):
        raise AssertionError("The four response groups are not globally disjoint and complete.")
    return selected


__all__ = [
    "clean_correct_source",
    "feasibility_counts",
    "largest_feasible_symmetric_margin",
    "parse_unambiguous_numeric_answer",
    "second_step_strengths",
    "select_disjoint_balanced_responses",
    "strict_incorrect_source",
]
