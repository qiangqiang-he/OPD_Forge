#!/usr/bin/env python3
"""Outcome analysis (01) for the ERSR Monte-Carlo evaluator output.

``tests/run_ersr_mc.py`` writes ``ersr_results.json`` with one record per
selected (rollout, reasoning-step) case.  This script reads that file and
emits one self-contained JSON report (default ``ersr_analysis.json``) with
every statistic and an auto-generated conclusions block:

1.  Core contrasts ``E[A | R]`` and ``E[A_TI | R]`` for correct (``R=1``) and
    incorrect (``R=0``) trajectories with 95% CIs, answering whether the
    student action or the teacher action has the higher expected return.
2.  The teacher-over-student gain ``Delta = A_TI - A`` in both regimes with
    the expected-sign check (negative on correct, positive on incorrect).
3.  Reasoning-position breakdowns (quartiles of the step inside the trace,
    using one-based step ordinal / number of reasoning steps):
    ``E[A | R, pos]``, ``E[A_TI | R, pos]``, and ``E[Delta | R, pos]``.
4.  Pearson/Spearman correlations between ``A`` and ``A_TI`` plus ``E[A_TI]``
    and ``E[Delta]`` inside five ``A`` bands (very low ... very high).
5.  ``A < 0`` / ``A = 0`` / ``A > 0`` grouping with ``P(A_TI > A | group)``.
6.  Damage statistics on correct trajectories: ``P(A_TI < A)``,
    ``E[Delta | A_TI < A]``, ``P(A_TI < 0)``, ``E[A_TI | A_TI < 0]``.

Definitions (matching the evaluator's output):

*   ``V(arm) = correct_count / mc_k`` from the Student MC phase;
*   ``A  = V(keep)                - V(redecide)``  (student action advantage);
*   ``A_TI = V(replace:<teacher>) - V(redecide)``  (teacher intervention
    advantage, one per configured Teacher);
*   ``Delta = A_TI - A = V(replace) - V(keep)``;
*   ``R = rollout_correct`` of the original full trajectory.

``A``, ``A_TI``, and ``Delta`` are recomputed from the integer
``correct_count`` / ``mc_k`` samples, so every ordering and equality statement
below is exact rational arithmetic rather than a float comparison.  numpy is
optional and only accelerates the bootstrap; every statistic has a
pure-Python fallback.

Example:

    python tests/01_analyze_ersr_outcome.py \
        --results outputs/ersr_dapo17k_qwen3_1p7b_mc128/ersr_results.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Conclusions print human-readable Chinese text; force UTF-8 so hosts with a
# legacy console codepage (e.g. Windows GBK) never mangle the output.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

try:  # numpy only accelerates the bootstrap; everything else is pure Python.
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore[assignment]


SCHEMA_VERSION = 1
Z_95 = 1.959963984540054  # scipy.stats.norm.ppf(0.975)
HIST_BINS = 20
HIST_RANGE = (-1.0, 1.0)
QUANTILE_POINTS = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0)
POSITION_LABELS = ("Q1_(0,25%]", "Q2_(25%,50%]", "Q3_(50%,75%]", "Q4_(75%,100%]")
DEFAULT_A_EDGES = (-0.25, -0.05, 0.05, 0.25)
A_LABELS = ("A_very_low", "A_low", "A_near_zero", "A_high", "A_very_high")
R_LABELS = {"correct": 1, "incorrect": 0}


# --------------------------------------------------------------------------
# Small numeric helpers (pure Python, with an optional numpy fast path).
# --------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def r6(value: Any) -> Any:
    """Round float leaves of a report tree; keep None and pass ints through."""

    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return round(value, 6)
    return value


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values)


def _sample_sd(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = _mean(values)
    return math.sqrt(math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1))


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        raise ValueError("percentile of an empty sample.")
    data = sorted(float(value) for value in values)
    if len(data) == 1:
        return data[0]
    rank = (len(data) - 1) * (q / 100.0)
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return data[low]
    frac = rank - low
    return data[low] * (1.0 - frac) + data[high] * frac


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = _mean(xs)
    my = _mean(ys)
    cov = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys))
    var_x = math.fsum((x - mx) ** 2 for x in xs)
    var_y = math.fsum((y - my) ** 2 for y in ys)
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    return cov / math.sqrt(var_x * var_y)


def average_ranks(values: Sequence[float]) -> list[float]:
    n = len(values)
    order = sorted(range(n), key=lambda index: values[index])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return pearson(average_ranks(xs), average_ranks(ys))


def bootstrap_mean_ci(
    values: Sequence[float], rounds: int, seed: int
) -> list[float] | None:
    if not values or rounds <= 0:
        return None
    if np is not None:
        rng = np.random.default_rng(seed)
        arr = np.asarray(values, dtype=np.float64)
        n = arr.size
        chunk = max(1, 4_000_000 // max(n, 1))
        means = np.empty(rounds, dtype=np.float64)
        filled = 0
        while filled < rounds:
            take = min(chunk, rounds - filled)
            indices = rng.integers(0, n, size=(take, n))
            means[filled : filled + take] = arr[indices].mean(axis=1)
            filled += take
        return [
            round(float(np.percentile(means, 2.5)), 6),
            round(float(np.percentile(means, 97.5)), 6),
        ]
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(rounds):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    return [round(percentile(means, 2.5), 6), round(percentile(means, 97.5), 6)]


def mean_stats(
    values: Sequence[float],
    *,
    bootstrap_rounds: int = 0,
    seed: int = 0,
) -> dict[str, Any]:
    """Mean with a normal-approximation 95% CI and an optional bootstrap CI."""

    n = len(values)
    if n == 0:
        return {
            "n": 0,
            "mean": None,
            "sample_sd": None,
            "standard_error": None,
            "ci95_normal": None,
            "ci95_bootstrap": None,
        }
    mean = _mean(values)
    sd = _sample_sd(values)
    se = sd / math.sqrt(n)
    ci = [mean - Z_95 * se, mean + Z_95 * se]
    boot = (
        bootstrap_mean_ci(values, bootstrap_rounds, seed)
        if bootstrap_rounds > 0
        else None
    )
    return {
        "n": n,
        "mean": round(mean, 6),
        "sample_sd": round(sd, 6),
        "standard_error": round(se, 6),
        "ci95_normal": [round(ci[0], 6), round(ci[1], 6)],
        "ci95_bootstrap": boot,
    }


def probability(values: Sequence[bool]) -> float | None:
    if not values:
        return None
    return round(sum(1 for value in values if value) / len(values), 6)


def histogram(values: Sequence[float]) -> dict[str, Any]:
    lo, hi = HIST_RANGE
    width = (hi - lo) / HIST_BINS
    counts = [0] * HIST_BINS
    for value in values:
        index = int((min(max(value, lo), hi - 1e-12) - lo) / width)
        index = min(max(index, 0), HIST_BINS - 1)
        counts[index] += 1
    edges = [round(lo + i * width, 6) for i in range(HIST_BINS + 1)]
    return {"bin_edges": edges, "counts": counts, "range": [lo, hi]}


def quantile_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": round(_mean(values), 6),
        **{
            ("min" if q == 0.0 else "max" if q == 100.0 else f"p{int(q)}"): round(
                percentile(values, q), 6
            )
            for q in QUANTILE_POINTS
        },
    }


class SeedSequence:
    """Deterministic per-call seeds so bootstrap CIs are reproducible."""

    def __init__(self, base: int) -> None:
        self._state = base

    def next(self) -> int:
        self._state += 1
        return self._state


# --------------------------------------------------------------------------
# Loading and flattening ersr_results.json into per-case / per-pair rows.
# --------------------------------------------------------------------------


def position_bin(position_fraction: float) -> str:
    if position_fraction <= 0.25:
        return POSITION_LABELS[0]
    if position_fraction <= 0.5:
        return POSITION_LABELS[1]
    if position_fraction <= 0.75:
        return POSITION_LABELS[2]
    return POSITION_LABELS[3]


def step_position_fraction(record: dict[str, Any], case_id: str) -> float:
    """Return the requested one-based, step-count position in ``(0, 1]``.

    The evaluator also stores ``position_fraction``, but that field uses
    ``reasoning_step_index / (num_reasoning_steps - 1)``.  The analysis plan
    instead defines, for example, the third step of a ten-step response as
    ``3 / 10 = 0.3``.  Recompute it from the integer step fields so the
    analysis follows that definition exactly and never uses token position.
    """

    step_index = int(record["reasoning_step_index"])
    num_steps = int(record["num_reasoning_steps"])
    if num_steps <= 0:
        raise ValueError(
            f"Case {case_id!r} has invalid num_reasoning_steps={num_steps}."
        )
    if step_index < 0 or step_index >= num_steps:
        raise ValueError(
            f"Case {case_id!r} has reasoning_step_index={step_index} outside "
            f"[0, {num_steps})."
        )
    return (step_index + 1) / num_steps


def a_bin(value: float, edges: Sequence[float]) -> str:
    # A_very_low: a < e0 | A_low: [e0, e1) | A_near_zero: [e1, e2]
    # A_high: (e2, e3] | A_very_high: a > e3
    if value < edges[0]:
        return A_LABELS[0]
    if value < edges[1]:
        return A_LABELS[1]
    if value <= edges[2]:
        return A_LABELS[2]
    if value <= edges[3]:
        return A_LABELS[3]
    return A_LABELS[4]


def _required_sample(samples: dict[str, Any], case_id: str, arm: str) -> dict[str, Any]:
    sample = samples.get(arm)
    if not isinstance(sample, dict) or "correct_count" not in sample or "mc_k" not in sample:
        raise ValueError(f"Case {case_id!r} is missing MC samples for arm {arm!r}.")
    return sample


def extract_rows(
    results: Sequence[dict[str, Any]], *, exclude_truncated: bool = False
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str], int]:
    """Build one case row per case and one pair row per (case, teacher).

    ``A``, ``A_TI``, and ``delta`` are stored as exact fractions of mc_k, and
    the boolean orderings are computed from the integer correct counts, so
    group membership and comparisons never suffer float-tie problems.
    """

    cases: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    teachers_seen: set[str] = set()
    truncated_pairs = 0
    for record in results:
        case_id = str(record["case_id"])
        samples = record.get("samples") or {}
        redecide = _required_sample(samples, case_id, "redecide")
        keep = _required_sample(samples, case_id, "keep")
        mc_k = int(redecide["mc_k"])
        if int(keep["mc_k"]) != mc_k:
            raise ValueError(f"Case {case_id!r} has inconsistent mc_k across arms.")
        rd = int(redecide["correct_count"])
        kp = int(keep["correct_count"])
        case_teachers = sorted(
            key[len("replace:") :] for key in samples if key.startswith("replace:")
        )
        if not case_teachers:
            raise ValueError(f"Case {case_id!r} has no replace:<teacher> arm.")
        teachers_seen.update(case_teachers)
        position = step_position_fraction(record, case_id)
        case_row = {
            "case_id": case_id,
            "case_index": int(record["case_index"]),
            "response_idx": int(record["response_idx"]),
            "rollout_index": int(record["rollout_index"]),
            "R": int(record["rollout_correct"]),
            "bucket": int(record["bucket"]),
            "reasoning_step_index": int(record["reasoning_step_index"]),
            "num_reasoning_steps": int(record["num_reasoning_steps"]),
            "position_fraction": position,
            "position_bin": position_bin(position),
            "mc_k": mc_k,
            "A": (kp - rd) / mc_k,
        }
        cases.append(case_row)
        replacements = record.get("teacher_replacements") or {}
        for name in case_teachers:
            sample = _required_sample(samples, case_id, f"replace:{name}")
            if int(sample["mc_k"]) != mc_k:
                raise ValueError(
                    f"Case {case_id!r} teacher {name!r} has inconsistent mc_k."
                )
            rp = int(sample["correct_count"])
            truncated = int((replacements.get(name) or {}).get("truncated", 0))
            if truncated:
                truncated_pairs += 1
            if exclude_truncated and truncated:
                continue
            pairs.append(
                {
                    **case_row,
                    "teacher": name,
                    "A_TI": (rp - rd) / mc_k,
                    "delta": (rp - kp) / mc_k,
                    "A_TI_gt_A": rp > kp,
                    "A_TI_eq_A": rp == kp,
                    "A_negative": kp < rd,
                    "A_zero": kp == rd,
                    "A_positive": kp > rd,
                    "A_TI_negative": rp < rd,
                    "truncated": truncated,
                }
            )
    return cases, pairs, sorted(teachers_seen), truncated_pairs


def split_filter(split: str) -> Callable[[dict[str, Any]], bool] | None:
    if split == "all":
        return None
    r = R_LABELS[split]
    return lambda row: row["R"] == r


def build_scopes(pairs: list[dict[str, Any]], teachers: Sequence[str]) -> dict[str, list[dict[str, Any]]]:
    """pooled = every (case, teacher) pair; teacher:<name> = one Teacher."""

    scopes: dict[str, list[dict[str, Any]]] = {"pooled": pairs}
    for name in teachers:
        scopes[f"teacher:{name}"] = [row for row in pairs if row["teacher"] == name]
    return scopes


# --------------------------------------------------------------------------
# Report sections.
# --------------------------------------------------------------------------


def core_section(
    cases: list[dict[str, Any]],
    scopes: dict[str, list[dict[str, Any]]],
    seeds: SeedSequence,
    bootstrap_rounds: int,
) -> dict[str, Any]:
    section: dict[str, Any] = {"E_A_given_R": {}, "E_A_TI_given_R": {}, "E_delta_given_R": {}}
    for r in (1, 0):
        key = f"R={r}"
        section["E_A_given_R"][key] = mean_stats(
            [case["A"] for case in cases if case["R"] == r],
            bootstrap_rounds=bootstrap_rounds,
            seed=seeds.next(),
        )
        section["E_A_TI_given_R"][key] = {}
        section["E_delta_given_R"][key] = {}
        for scope, rows in scopes.items():
            subset = [row for row in rows if row["R"] == r]
            section["E_A_TI_given_R"][key][scope] = mean_stats(
                [row["A_TI"] for row in subset],
                bootstrap_rounds=bootstrap_rounds,
                seed=seeds.next(),
            )
            section["E_delta_given_R"][key][scope] = mean_stats(
                [row["delta"] for row in subset],
                bootstrap_rounds=bootstrap_rounds,
                seed=seeds.next(),
            )
    return section


def _ci_excludes_zero(stats: dict[str, Any]) -> bool:
    ci = stats.get("ci95_normal")
    if not ci or ci[0] is None:
        return False
    return ci[0] > 0.0 or ci[1] < 0.0


def _fmt_ci(stats: dict[str, Any]) -> str:
    mean = stats.get("mean")
    ci = stats.get("ci95_normal")
    if mean is None:
        return "n/a"
    boot = stats.get("ci95_bootstrap")
    ci_text = f"[{ci[0]:.4f}, {ci[1]:.4f}]" if ci else "n/a"
    boot_text = f", bootstrap {boot[0]:.4f}..{boot[1]:.4f}" if boot else ""
    return f"{mean:.4f} (95% CI {ci_text}{boot_text})"


def questions_section(
    core: dict[str, Any], cases: list[dict[str, Any]], pairs: list[dict[str, Any]]
) -> dict[str, Any]:
    answers: dict[str, Any] = {}
    statements: dict[str, str] = {}
    for label, r in (("correct", 1), ("incorrect", 0)):
        key = f"R={r}"
        e_a = core["E_A_given_R"][key]
        e_ti = core["E_A_TI_given_R"][key]["pooled"]
        delta = core["E_delta_given_R"][key]["pooled"]
        n_cases = sum(1 for case in cases if case["R"] == r)
        n_pairs = sum(1 for row in pairs if row["R"] == r)
        if e_a["mean"] is None or e_ti["mean"] is None:
            higher = None
        elif e_ti["mean"] > e_a["mean"]:
            higher = "teacher_action"
        elif e_a["mean"] > e_ti["mean"]:
            higher = "student_action"
        else:
            higher = "tie"
        significant = _ci_excludes_zero(delta)
        by_scope: dict[str, Any] = {}
        for scope, scope_e_ti in core["E_A_TI_given_R"][key].items():
            scope_delta = core["E_delta_given_R"][key][scope]
            if scope_e_ti["mean"] is None or e_a["mean"] is None:
                scope_higher = None
            elif scope_e_ti["mean"] > e_a["mean"]:
                scope_higher = "teacher_action"
            elif e_a["mean"] > scope_e_ti["mean"]:
                scope_higher = "student_action"
            else:
                scope_higher = "tie"
            by_scope[scope] = {
                "n_case_teacher_pairs": (
                    n_pairs
                    if scope == "pooled"
                    else sum(
                        1
                        for row in pairs
                        if row["R"] == r and scope == f"teacher:{row['teacher']}"
                    )
                ),
                "E_A": e_a["mean"],
                "E_A_TI": scope_e_ti["mean"],
                "higher_expected_return": scope_higher,
                "paired_E_delta": scope_delta["mean"],
                "paired_delta_ci95_normal": scope_delta["ci95_normal"],
                "paired_delta_ci95_bootstrap": scope_delta.get("ci95_bootstrap"),
                "difference_significant_at_95pct": _ci_excludes_zero(scope_delta),
            }

        answers[f"{label}_trajectories"] = {
            "n_cases": n_cases,
            "n_case_teacher_pairs": n_pairs,
            "E_A": e_a["mean"],
            "E_A_TI_pooled": e_ti["mean"],
            "higher_expected_return": higher,
            "paired_E_delta": delta["mean"],
            "paired_delta_ci95_normal": delta["ci95_normal"],
            "paired_delta_ci95_bootstrap": delta.get("ci95_bootstrap"),
            "difference_significant_at_95pct": significant,
            "by_scope": by_scope,
        }
        statements[f"{label}_trajectories"] = (
            f"{'正确' if r == 1 else '错误'}轨迹 (n_cases={n_cases}, n_pairs={n_pairs}): "
            f"E[A|R={r}]={_fmt_ci(e_a)}, E[A_TI|R={r}]={_fmt_ci(e_ti)} → "
            f"{'teacher action' if higher == 'teacher_action' else 'student action' if higher == 'student_action' else '平手'}"
            f" 的 expected return 更高; 配对差 Delta=A_TI-A: {_fmt_ci(delta)}"
            f"{'，差异显著' if significant else '，差异不显著'}。"
        )
    expected = {}
    expected_by_scope: dict[str, Any] = {}
    for label, r, expected_sign in (("correct", 1, "negative"), ("incorrect", 0, "positive")):
        key = f"R={r}"
        delta = core["E_delta_given_R"][key]["pooled"]
        mean = delta["mean"]
        holds = (
            mean < 0.0 if expected_sign == "negative" and mean is not None
            else mean > 0.0 if mean is not None
            else None
        )
        expected[f"E_delta_given_R={r}"] = {
            "value": mean,
            "ci95_normal": delta["ci95_normal"],
            "expected_sign": expected_sign,
            "matches_expectation": holds,
            "significant_at_95pct": _ci_excludes_zero(delta),
        }
        expected_by_scope[key] = {}
        for scope, scope_delta in core["E_delta_given_R"][key].items():
            scope_mean = scope_delta["mean"]
            scope_holds = (
                scope_mean < 0.0
                if expected_sign == "negative" and scope_mean is not None
                else scope_mean > 0.0
                if scope_mean is not None
                else None
            )
            expected_by_scope[key][scope] = {
                "value": scope_mean,
                "ci95_normal": scope_delta["ci95_normal"],
                "ci95_bootstrap": scope_delta.get("ci95_bootstrap"),
                "expected_sign": expected_sign,
                "matches_expectation": scope_holds,
                "significant_at_95pct": _ci_excludes_zero(scope_delta),
            }
    statements["expected_signs"] = (
        f"E[A_TI-A|R=1]={_fmt_ci(core['E_delta_given_R']['R=1']['pooled'])} "
        f"({'符合' if expected['E_delta_given_R=1']['matches_expectation'] else '不符合'}预期 <0); "
        f"E[A_TI-A|R=0]={_fmt_ci(core['E_delta_given_R']['R=0']['pooled'])} "
        f"({'符合' if expected['E_delta_given_R=0']['matches_expectation'] else '不符合'}预期 >0)。"
    )
    return {
        "answers": answers,
        "expected_sign_check": expected,
        "expected_sign_check_by_scope": expected_by_scope,
        "statements": statements,
    }


def position_section(
    cases: list[dict[str, Any]],
    scopes: dict[str, list[dict[str, Any]]],
    seeds: SeedSequence,
) -> dict[str, Any]:
    by_r: dict[str, Any] = {}
    for r in (1, 0):
        bins: dict[str, Any] = {}
        for label in POSITION_LABELS:
            subset_cases = [
                case for case in cases if case["R"] == r and case["position_bin"] == label
            ]
            entry: dict[str, Any] = {
                "n_cases": len(subset_cases),
                "E_A": mean_stats([case["A"] for case in subset_cases], seed=seeds.next()),
            }
            for scope, rows in scopes.items():
                subset = [
                    row for row in rows if row["R"] == r and row["position_bin"] == label
                ]
                entry[f"E_A_TI__{scope}"] = mean_stats(
                    [row["A_TI"] for row in subset], seed=seeds.next()
                )
                entry[f"E_delta__{scope}"] = mean_stats(
                    [row["delta"] for row in subset], seed=seeds.next()
                )
            bins[label] = entry
        by_r[f"R={r}"] = bins

    trends: dict[str, Any] = {}
    for r in (1, 0):
        bins = by_r[f"R={r}"]
        e_a = [bins[label]["E_A"]["mean"] for label in POSITION_LABELS]
        e_ti = [bins[label]["E_A_TI__pooled"]["mean"] for label in POSITION_LABELS]
        delta = [bins[label]["E_delta__pooled"]["mean"] for label in POSITION_LABELS]
        available = [label for label in POSITION_LABELS if bins[label]["n_cases"] > 0]
        most_negative_bin = min(
            available,
            key=lambda label: bins[label]["E_delta__pooled"]["mean"] or 0.0,
        ) if available else None
        most_positive_bin = max(
            available,
            key=lambda label: bins[label]["E_delta__pooled"]["mean"] or 0.0,
        ) if available else None
        last_bin = POSITION_LABELS[-1]
        late_delta = bins[last_bin]["E_delta__pooled"]["mean"]
        trends[f"R={r}"] = {
            "E_A_by_bin": e_a,
            "E_A_TI_pooled_by_bin": e_ti,
            "E_delta_pooled_by_bin": delta,
            "E_A_monotonic_increasing": all(
                a is not None and b is not None and b >= a
                for a, b in zip(e_a, e_a[1:])
            ),
            "delta_most_negative_bin": most_negative_bin,
            "delta_most_positive_bin": most_positive_bin,
            "late_bin_delta_negative": (late_delta is not None and late_delta < 0.0),
            "late_bin_delta_value": late_delta,
        }
    return {
        "definition": (
            "position_fraction = (reasoning_step_index + 1) / num_reasoning_steps; "
            "the numerator is the one-based semantic-step ordinal. Binning uses "
            "step position, never token count."
        ),
        "bins": list(POSITION_LABELS),
        "by_R_and_bin": by_r,
        "trends": trends,
    }


def correlation_section(
    scopes: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    section: dict[str, Any] = {}
    for split in ("all", "correct", "incorrect"):
        predicate = split_filter(split)
        entry: dict[str, Any] = {}
        for scope, rows in scopes.items():
            data = [row for row in rows if predicate(row)] if predicate else rows
            xs = [row["A"] for row in data]
            ys = [row["A_TI"] for row in data]
            entry[scope] = {
                "n_pairs": len(data),
                "pearson": r6(pearson(xs, ys)),
                "spearman": r6(spearman(xs, ys)),
            }
        section[split] = entry
    return section


def a_binning_section(
    scopes: dict[str, list[dict[str, Any]]],
    edges: Sequence[float],
) -> dict[str, Any]:
    section: dict[str, Any] = {
        "edges": {
            A_LABELS[0]: f"A < {edges[0]}",
            A_LABELS[1]: f"{edges[0]} <= A < {edges[1]}",
            A_LABELS[2]: f"{edges[1]} <= A <= {edges[2]}",
            A_LABELS[3]: f"{edges[2]} < A <= {edges[3]}",
            A_LABELS[4]: f"A > {edges[3]}",
        },
    }
    for split in ("all", "correct", "incorrect"):
        predicate = split_filter(split)
        split_entry: dict[str, Any] = {}
        for scope, rows in scopes.items():
            data = [row for row in rows if predicate(row)] if predicate else rows
            grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in data:
                grouped[a_bin(row["A"], edges)].append(row)
            scope_entry: dict[str, Any] = {}
            for label in A_LABELS:
                bucket = grouped.get(label, [])
                scope_entry[label] = {
                    "n_pairs": len(bucket),
                    "mean_A": r6(_mean([row["A"] for row in bucket])) if bucket else None,
                    "mean_A_TI": r6(_mean([row["A_TI"] for row in bucket])) if bucket else None,
                    "mean_delta": r6(_mean([row["delta"] for row in bucket])) if bucket else None,
                    "P_A_TI_gt_A": probability([row["A_TI_gt_A"] for row in bucket]),
                }
            split_entry[scope] = scope_entry
        deltas = [
            split_entry["pooled"][label]["mean_delta"] for label in A_LABELS
        ]
        available = [value for value in deltas if value is not None]
        split_entry["pooled_mean_delta_by_bin"] = deltas
        split_entry["mean_delta_monotonic_decreasing"] = all(
            a >= b for a, b in zip(available, available[1:])
        ) if len(available) == len(A_LABELS) else None
        section[split] = split_entry
    return section


def low_value_section(scopes: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    groups = (("A<0", "A_negative"), ("A=0", "A_zero"), ("A>0", "A_positive"))
    section: dict[str, Any] = {}
    for split in ("all", "correct", "incorrect"):
        predicate = split_filter(split)
        split_entry: dict[str, Any] = {}
        for scope, rows in scopes.items():
            data = [row for row in rows if predicate(row)] if predicate else rows
            scope_entry: dict[str, Any] = {}
            for label, field in groups:
                bucket = [row for row in data if row[field]]
                scope_entry[label] = {
                    "n_pairs": len(bucket),
                    "mean_A": r6(_mean([row["A"] for row in bucket])) if bucket else None,
                    "mean_A_TI": r6(_mean([row["A_TI"] for row in bucket])) if bucket else None,
                    "mean_delta": r6(_mean([row["delta"] for row in bucket])) if bucket else None,
                    "P_A_TI_gt_A": probability([row["A_TI_gt_A"] for row in bucket]),
                    "P_A_TI_eq_A": probability([row["A_TI_eq_A"] for row in bucket]),
                }
            split_entry[scope] = scope_entry
        section[split] = split_entry
    return section


def damage_section(scopes: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Teacher-intervention damage statistics on correct trajectories."""

    section: dict[str, Any] = {}
    for scope, rows in scopes.items():
        data = [row for row in rows if row["R"] == 1]
        harmed = [row for row in data if not row["A_TI_gt_A"] and not row["A_TI_eq_A"]]
        negative = [row for row in data if row["A_TI_negative"]]
        helped = [row for row in data if row["A_TI_gt_A"]]
        unchanged = [row for row in data if row["A_TI_eq_A"]]
        section[scope] = {
            "n_pairs": len(data),
            "P_A_TI_lt_0": probability([row["A_TI_negative"] for row in data]),
            "P_A_TI_lt_A": probability(
                [not row["A_TI_gt_A"] and not row["A_TI_eq_A"] for row in data]
            ),
            "P_A_TI_gt_A": probability([row["A_TI_gt_A"] for row in data]),
            "P_A_TI_eq_A": probability([row["A_TI_eq_A"] for row in data]),
            "n_A_TI_lt_A": len(harmed),
            "n_A_TI_lt_0": len(negative),
            "n_A_TI_gt_A": len(helped),
            "n_A_TI_eq_A": len(unchanged),
            "E_delta_given_A_TI_lt_A": r6(_mean([row["delta"] for row in harmed]))
            if harmed
            else None,
            "E_A_TI_given_A_TI_lt_0": r6(_mean([row["A_TI"] for row in negative]))
            if negative
            else None,
            "E_A_given_A_TI_lt_A": r6(_mean([row["A"] for row in harmed]))
            if harmed
            else None,
        }
    return section


def distribution_section(
    cases: list[dict[str, Any]], pairs: list[dict[str, Any]]
) -> dict[str, Any]:
    section: dict[str, Any] = {}
    for split in ("all", "correct", "incorrect"):
        predicate = split_filter(split)
        subset_cases = (
            [case for case in cases if predicate(case)] if predicate else cases
        )
        subset_pairs = [row for row in pairs if predicate(row)] if predicate else pairs
        section[split] = {
            "A": {
                "unit": "case",
                "quantiles": quantile_summary([case["A"] for case in subset_cases]),
                "histogram": histogram([case["A"] for case in subset_cases]),
            },
            "A_TI": {
                "unit": "case_teacher_pair",
                "quantiles": quantile_summary([row["A_TI"] for row in subset_pairs]),
                "histogram": histogram([row["A_TI"] for row in subset_pairs]),
            },
            "delta": {
                "unit": "case_teacher_pair",
                "quantiles": quantile_summary([row["delta"] for row in subset_pairs]),
                "histogram": histogram([row["delta"] for row in subset_pairs]),
            },
        }
    return section


def sensitivity_section(
    results: Sequence[dict[str, Any]],
    truncated_pairs: int,
    seeds: SeedSequence,
    bootstrap_rounds: int,
) -> dict[str, Any]:
    """Re-run the headline contrasts with truncated Teacher steps excluded."""

    if truncated_pairs <= 0:
        return {"truncated_pairs": 0, "note": "no truncated Teacher replacements present"}
    cases, pairs, _teachers, excluded = extract_rows(results, exclude_truncated=True)
    section: dict[str, Any] = {
        "truncated_pairs_total": truncated_pairs,
        "truncated_pairs_excluded": excluded,
        "E_A_TI_given_R": {},
        "E_delta_given_R": {},
    }
    for r in (1, 0):
        key = f"R={r}"
        section["E_A_TI_given_R"][key] = mean_stats(
            [row["A_TI"] for row in pairs if row["R"] == r],
            bootstrap_rounds=bootstrap_rounds,
            seed=seeds.next(),
        )
        section["E_delta_given_R"][key] = mean_stats(
            [row["delta"] for row in pairs if row["R"] == r],
            bootstrap_rounds=bootstrap_rounds,
            seed=seeds.next(),
        )
    section["note"] = (
        "sensitivity subset: headline contrasts recomputed after dropping every "
        "(case, teacher) pair whose Teacher replacement was flagged truncated"
    )
    return section


# --------------------------------------------------------------------------
# Conclusions.
# --------------------------------------------------------------------------


def build_conclusions(
    core: dict[str, Any],
    questions: dict[str, Any],
    position: dict[str, Any],
    correlation: dict[str, Any],
    a_binning: dict[str, Any],
    low_value: dict[str, Any],
    damage: dict[str, Any],
) -> dict[str, str]:
    conclusions: dict[str, str] = {}
    statements = questions["statements"]
    conclusions["1_core_contrasts"] = (
        f"{statements['correct_trajectories']} {statements['incorrect_trajectories']}"
    )
    conclusions["2_expected_signs"] = statements["expected_signs"]

    trend_correct = position["trends"]["R=1"]
    trend_incorrect = position["trends"]["R=0"]
    fmt_series = lambda values: "[" + ", ".join("n/a" if v is None else f"{v:.4f}" for v in values) + "]"  # noqa: E731
    conclusions["3_position"] = (
        f"E[A|R=1] 按位置 Q1..Q4: {fmt_series(trend_correct['E_A_by_bin'])}; "
        f"E[A_TI|R=1]: {fmt_series(trend_correct['E_A_TI_pooled_by_bin'])}; "
        f"E[Delta|R=1]: {fmt_series(trend_correct['E_delta_pooled_by_bin'])} "
        f"(最负 bin={trend_correct['delta_most_negative_bin']}, "
        f"后期 Q4 Delta<0: {'是' if trend_correct['late_bin_delta_negative'] else '否'}); "
        f"E[A|R=0]: {fmt_series(trend_incorrect['E_A_by_bin'])}; "
        f"E[A_TI|R=0]: {fmt_series(trend_incorrect['E_A_TI_pooled_by_bin'])}; "
        f"E[Delta|R=0]: {fmt_series(trend_incorrect['E_delta_pooled_by_bin'])} "
        f"(最正 bin={trend_incorrect['delta_most_positive_bin']})."
    )

    corr_all = correlation["all"]["pooled"]
    bin_all = a_binning["all"]["pooled_mean_delta_by_bin"]
    decreasing = a_binning["all"]["mean_delta_monotonic_decreasing"]
    conclusions["4_A_vs_A_TI_relation"] = (
        f"全量 Pearson={corr_all['pearson']}, Spearman={corr_all['spearman']} "
        f"(correct: Pearson={correlation['correct']['pooled']['pearson']}, "
        f"Spearman={correlation['correct']['pooled']['spearman']}; "
        f"incorrect: Pearson={correlation['incorrect']['pooled']['pearson']}, "
        f"Spearman={correlation['incorrect']['pooled']['spearman']}); "
        f"A 分箱 (很低→很高) 对应 E[Delta]: {fmt_series(bin_all)}"
        f"{'，单调递减 → student action 越有价值，teacher 相对增益越低' if decreasing else '，非单调递减'}。"
    )

    incorrect = low_value["incorrect"]["pooled"]
    groups = ("A<0", "A=0", "A>0")

    def _p(group: str) -> str:
        value = incorrect[group]["P_A_TI_gt_A"]
        return "n/a" if value is None else f"{value:.4f}"

    conclusions["5_low_value_focus"] = (
        f"错误轨迹分组 P(A_TI>A): A<0 → {_p('A<0')}, A=0 → {_p('A=0')}, A>0 → {_p('A>0')}; "
        f"E[Delta]: A<0 → {incorrect['A<0']['mean_delta']}, "
        f"A=0 → {incorrect['A=0']['mean_delta']}, A>0 → {incorrect['A>0']['mean_delta']} "
        f"(收益{'主要' if (incorrect['A<0']['P_A_TI_gt_A'] or 0) > (incorrect['A>0']['P_A_TI_gt_A'] or 0) else '并非主要'}"
        f"集中在 student 自己做得差的步骤上; 分组={list(groups)})。"
    )

    pooled_damage = damage["pooled"]
    conclusions["6_damage_to_correct"] = (
        f"正确轨迹: P(A_TI<A|R=1)={pooled_damage['P_A_TI_lt_A']}, "
        f"P(A_TI<0|R=1)={pooled_damage['P_A_TI_lt_0']}, "
        f"P(A_TI>A|R=1)={pooled_damage['P_A_TI_gt_A']}, "
        f"P(A_TI=A|R=1)={pooled_damage['P_A_TI_eq_A']}; "
        f"E[Delta|A_TI<A,R=1]={pooled_damage['E_delta_given_A_TI_lt_A']}, "
        f"E[A_TI|A_TI<0,R=1]={pooled_damage['E_A_TI_given_A_TI_lt_0']} "
        f"→ 正确轨迹中 teacher replacement "
        f"{'仍会' if (pooled_damage['P_A_TI_lt_A'] or 0) > 0 else '几乎不会'}降低当前 policy 的 expected return。"
    )
    return conclusions


# --------------------------------------------------------------------------
# Entry point.
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=REPO_ROOT / "outputs/ersr_dapo17k_qwen3_1p7b_mc128/ersr_results.json",
        help="ersr_results.json written by tests/run_ersr_mc.py "
        "(a directory is also accepted; ersr_results.json is looked up inside).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output JSON path (default: <results dir>/ersr_analysis.json).",
    )
    parser.add_argument(
        "--bootstrap-rounds",
        type=int,
        default=2000,
        help="bootstrap resamples for the six headline 95%% CIs (0 disables).",
    )
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument(
        "--a-bin-edges",
        type=str,
        default=",".join(str(value) for value in DEFAULT_A_EDGES),
        help="four comma-separated increasing edges for the five A bands; "
        "the middle band [e1, e2] must contain 0.",
    )
    parser.add_argument(
        "--exclude-truncated",
        action="store_true",
        help="drop (case, teacher) pairs whose Teacher replacement was flagged truncated.",
    )
    return parser.parse_args()


def parse_a_edges(text: str) -> tuple[float, float, float, float]:
    try:
        values = tuple(float(token.strip()) for token in text.split(","))
    except Exception as exc:
        raise ValueError(f"--a-bin-edges must be four numbers: {text!r}") from exc
    if len(values) != 4 or not all(math.isfinite(v) for v in values):
        raise ValueError(f"--a-bin-edges must be four finite numbers: {text!r}")
    if not (values[0] < values[1] < values[2] < values[3]):
        raise ValueError(f"--a-bin-edges must be strictly increasing: {values}")
    if not (values[1] <= 0.0 <= values[2]):
        raise ValueError(f"the A_near_zero band must contain 0: {values}")
    return values  # type: ignore[return-value]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=4)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    results_path = args.results
    if results_path.is_dir():
        results_path = results_path / "ersr_results.json"
    if not results_path.is_file():
        raise FileNotFoundError(results_path)
    output_path = args.output or results_path.parent / "ersr_analysis.json"
    a_edges = parse_a_edges(args.a_bin_edges)

    with results_path.open(encoding="utf-8") as handle:
        results = json.load(handle)
    if not isinstance(results, list) or not results:
        raise ValueError(f"Expected a non-empty JSON array in {results_path}.")

    cases, pairs, teachers, truncated_pairs = extract_rows(
        results, exclude_truncated=bool(args.exclude_truncated)
    )
    if not pairs:
        raise ValueError("No (case, teacher) pairs could be extracted from the results.")
    mc_k_counts = Counter(case["mc_k"] for case in cases)
    scopes = build_scopes(pairs, teachers)
    seeds = SeedSequence(args.seed)
    boot = max(0, int(args.bootstrap_rounds))

    core = core_section(cases, scopes, seeds, boot)
    questions = questions_section(core, cases, pairs)
    position = position_section(cases, scopes, seeds)
    correlation = correlation_section(scopes)
    a_binning = a_binning_section(scopes, a_edges)
    low_value = low_value_section(scopes)
    damage = damage_section(scopes)
    distributions = distribution_section(cases, pairs)
    sensitivity = (
        {"truncated_pairs": truncated_pairs, "note": "--exclude-truncated is active"}
        if args.exclude_truncated
        else sensitivity_section(results, truncated_pairs, seeds, boot)
    )
    conclusions = build_conclusions(
        core, questions, position, correlation, a_binning, low_value, damage
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "script": "tests/01_analyze_ersr_outcome.py",
        "input": {
            "results_path": str(results_path),
            "n_cases": len(cases),
            "n_trajectories": len(
                {(case["response_idx"], case["rollout_index"]) for case in cases}
            ),
            "n_case_teacher_pairs": len(pairs),
            "teachers": teachers,
            "mc_k_distribution": {str(k): v for k, v in sorted(mc_k_counts.items())},
            "cases_by_R": {
                "R=1_correct": sum(1 for case in cases if case["R"] == 1),
                "R=0_incorrect": sum(1 for case in cases if case["R"] == 0),
            },
            "cases_by_position_bin": dict(
                Counter(case["position_bin"] for case in cases)
            ),
            "truncated_teacher_pairs": truncated_pairs,
            "excluded_truncated": bool(args.exclude_truncated),
        },
        "definitions": {
            "A": "V(keep) - V(redecide) — student action advantage",
            "A_TI": "V(replace:<teacher>) - V(redecide) — teacher intervention advantage",
            "delta_TI_S": "A_TI - A = V(replace) - V(keep) — teacher gain over student action",
            "R": "rollout_correct — whether the original full trajectory reached the correct answer",
            "V_arm": "correct_count / mc_k from the Student MC phase",
            "exactness": "A / A_TI / delta are exact fractions of integer correct counts; all orderings and equalities are exact",
            "scopes": "pooled = every (case, teacher) pair; teacher:<name> = that Teacher's pairs only; E[A | R] is computed per case",
            "ci95": "mean ± 1.96 * sample_sd / sqrt(n); bootstrap percentile CI reported alongside for the six headline statistics",
            "position_bins": list(POSITION_LABELS),
            "a_bins": {
                A_LABELS[0]: f"A < {a_edges[0]}",
                A_LABELS[1]: f"{a_edges[0]} <= A < {a_edges[1]}",
                A_LABELS[2]: f"{a_edges[1]} <= A <= {a_edges[2]} (A ~ 0)",
                A_LABELS[3]: f"{a_edges[2]} < A <= {a_edges[3]}",
                A_LABELS[4]: f"A > {a_edges[3]}",
            },
        },
        "core": core,
        "questions": questions,
        "position_analysis": position,
        "correlation_A_vs_A_TI": correlation,
        "A_binning": a_binning,
        "low_value_groups": low_value,
        "damage_on_correct": damage,
        "distributions": distributions,
        "sensitivity_truncated_excluded": sensitivity,
        "conclusions": conclusions,
        "caveats": [
            "Every V is a mc_k-sample Monte-Carlo estimate, so A and A_TI carry sampling noise; "
            "the CIs quantify it but per-case point values should not be over-interpreted.",
            "E[A | R] is computed over cases while E[A_TI | R] and E[delta | R] are computed over "
            "(case, teacher) pairs, so pooled scopes weight Teachers by pair count.",
            "The normal-approximation CI is unreliable for very small or heavily skewed subsets; "
            "prefer the bootstrap CI for the six headline statistics in that case.",
            "Position uses one-based semantic-step ordinal / num_reasoning_steps, not the "
            "evaluator's zero-based position_fraction and not token fraction.",
            "Teacher replacements flagged truncated are kept by default; see "
            "sensitivity_truncated_excluded for the same headline contrasts without them.",
        ],
    }

    atomic_json(output_path, report)

    print(
        json.dumps(
            {
                "event": "analysis_complete",
                "results_path": str(results_path),
                "output_path": str(output_path),
                "n_cases": len(cases),
                "n_pairs": len(pairs),
                "teachers": teachers,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    for key, text in conclusions.items():
        print(f"[{key}] {text}", flush=True)


if __name__ == "__main__":
    main()
