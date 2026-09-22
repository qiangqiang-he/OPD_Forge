#!/usr/bin/env python3
"""Both-probe head-to-head analysis on correct and incorrect trajectories (05).

Analysis 03 evaluated every branch only with its "matched" probe: the Student
probe against ``A_student`` on correct trajectories, and each Teacher probe
against ``A_teacher`` on incorrect trajectories.  This supplement re-uses the
unchanged ``03_probe_step_scores.jsonl`` and runs the complete matrix:

* branch:  correct (reward=1) / incorrect (reward=0);
* signal:  deltaP_student / deltaP_teacher (pooled and per Teacher);
* utility: A_student / A_teacher;

plus two statistics 03 could not answer:

* paired head-to-head: both probes correlated against the same utility on
  identical (step, Teacher) rows, with one shared prompt-cluster bootstrap so
  the correlation difference is a paired statistic;
* horse-race regression ``utility ~ deltaP_student + deltaP_teacher + P_before
  + V_before + relative_position``, reporting each probe's incremental
  predictive power beyond the other.

All per-scope machinery is imported from analysis 03, so the numbers stay
directly comparable with the 03 report.  No new probes are collected.

Two-step workflow
-----------------
1. Server (where the heavy JSONL lives), CPU-only and fast::

       python tests/extract_ersr_probe_results_05.py

   writes the compact ``05_probe_step_data.csv`` plus a provenance manifest.
2. Local: copy that small CSV over and run::

       python tests/analyze_ersr_answer_probes_05.py --step-data 05_probe_step_data.csv

``--step-data`` also still accepts the original ``03_probe_step_scores.jsonl``
directly; the format is detected from the file suffix and both paths produce
bit-identical statistics for the same seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.analyze_ersr_answer_probes_03 import (
    STEP_DATA_FIELDS,
    _cluster_indices,
    _extreme_csv,
    _global_csv,
    _nan,
    _positive_finding,
    _quartile_csv,
    _regression_csv,
    _sign_csv,
    _within_csv,
    analyze_scope,
    atomic_csv,
    atomic_json,
    bootstrap_two_sided_p,
    correlation_result,
    flatten_step_teacher_rows,
    load_jsonl,
    pearson,
    percentile_ci,
    r6,
    spearman,
)

DEFAULT_STEP_SCORES = (
    REPO_ROOT
    / "outputs/ersr_dapo17k_qwen3_1p7b_mc128/probe_analysis_03/03_probe_step_scores.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs/ersr_dapo17k_qwen3_1p7b_mc128/probe_analysis_05"
)
SCHEMA_VERSION = 1
BRANCHES = (("correct", 1), ("incorrect", 0))
SIGNALS = ("student", "teacher")
UTILITIES = ("A_student", "A_teacher")
HORSE_RACE_FORMULA = (
    "utility ~ deltaP_student + deltaP_teacher + P_before + V_before + relative_position"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Step-data loading (JSONL on the server, compact CSV locally)
# ---------------------------------------------------------------------------

_INT_FIELDS = (
    "reward",
    "step_id",
    "step_ordinal",
    "source_step_index",
    "num_steps",
    "teacher_truncated",
    "mc_k",
)
_FLOAT_FIELDS = (
    "relative_position",
    "prompt_accuracy",
    "P_before",
    "P_student_after",
    "P_teacher_after",
    "deltaP_student",
    "deltaP_teacher",
    "mean_logprob_before",
    "mean_logprob_student_after",
    "mean_logprob_teacher_after",
    "delta_logP_student",
    "delta_logP_teacher",
    "V_before",
    "V_student_after",
    "V_teacher_after",
    "A_student",
    "A_teacher",
)


def load_step_data_csv(path: Path) -> list[dict[str, Any]]:
    """Load the compact extractor CSV back into typed long-form rows."""

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = set(STEP_DATA_FIELDS) - fieldnames
        if missing:
            raise ValueError(
                f"Step data CSV {path} is missing columns: {sorted(missing)}."
            )
        for line_number, raw in enumerate(reader, 2):
            try:
                row = dict(raw)
                for field in _INT_FIELDS:
                    row[field] = int(row[field])
                for field in _FLOAT_FIELDS:
                    row[field] = float(row[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Malformed step data row at {path}:{line_number}.") from exc
            if row["reward"] not in (0, 1):
                raise ValueError(f"Invalid reward at {path}:{line_number}: {row['reward']!r}.")
            key = (str(row["case_id"]), str(row["teacher"]))
            if key in seen:
                raise ValueError(f"Duplicate (case_id, teacher) row at {path}:{line_number}.")
            seen.add(key)
            rows.append(row)
    if not rows:
        raise ValueError(f"No rows found in {path}.")
    return rows


def load_long_rows(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load long-form rows from the JSONL or the compact extractor CSV."""

    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        raw_steps = load_jsonl(path)
        rows = flatten_step_teacher_rows(raw_steps)
        return rows, {
            "source_format": "03_probe_step_scores.jsonl",
            "step_score_records": len(raw_steps),
        }
    if suffix == ".csv":
        rows = load_step_data_csv(path)
        return rows, {
            "source_format": "05_probe_step_data.csv (extractor output)",
            "step_score_records": None,
        }
    raise ValueError(
        f"Unsupported step-data format {suffix!r} for {path}; expected .jsonl or .csv."
    )


# ---------------------------------------------------------------------------
# Scope-row construction
# ---------------------------------------------------------------------------

_STUDENT_SHARED_FIELDS = (
    "prompt_id",
    "trajectory_id",
    "P_before",
    "V_before",
    "relative_position",
    "deltaP_student",
    "A_student",
)


def case_unit_rows(
    long_rows: Sequence[dict[str, Any]], *, reward: int
) -> list[dict[str, Any]]:
    """One deduplicated row per case for the student probe vs A_student scope.

    Teacher copies must agree on every Student-side field, mirroring the 03
    deduplication contract.
    """

    by_case: dict[str, dict[str, Any]] = {}
    for row in long_rows:
        if int(row["reward"]) != reward:
            continue
        case_id = str(row["case_id"])
        if case_id in by_case:
            previous = by_case[case_id]
            for field in _STUDENT_SHARED_FIELDS:
                if previous[field] != row[field]:
                    raise ValueError(
                        f"Teacher copies disagree on Student field {field}/{case_id}."
                    )
            continue
        by_case[case_id] = {
            **row,
            "probe_gain": float(row["deltaP_student"]),
            "utility": float(row["A_student"]),
            "within_trajectory_id": str(row["trajectory_id"]),
        }
    if not by_case:
        raise ValueError(f"No rows with reward={reward}.")
    return list(by_case.values())


def paired_rows(
    long_rows: Sequence[dict[str, Any]],
    *,
    reward: int,
    signal: str,
    utility: str,
    teacher: str | None = None,
) -> list[dict[str, Any]]:
    """One row per (case, Teacher) for any signal/utility combination."""

    if signal not in SIGNALS:
        raise ValueError(f"Unknown signal {signal!r}.")
    if utility not in UTILITIES:
        raise ValueError(f"Unknown utility {utility!r}.")
    signal_field = "deltaP_student" if signal == "student" else "deltaP_teacher"
    rows: list[dict[str, Any]] = []
    for row in long_rows:
        if int(row["reward"]) != reward:
            continue
        if teacher is not None and str(row["teacher"]) != teacher:
            continue
        rows.append(
            {
                **row,
                "probe_gain": float(row[signal_field]),
                "utility": float(row[utility]),
                "within_trajectory_id": f"{row['trajectory_id']}|{row['teacher']}",
            }
        )
    if not rows:
        raise ValueError(f"No rows for reward={reward}, teacher={teacher!r}.")
    return rows


def build_scopes(
    long_rows: Sequence[dict[str, Any]],
    *,
    teachers: Sequence[str],
    bootstrap_rounds: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Run the 03 per-scope battery over the full branch x signal x utility matrix."""

    scopes: list[dict[str, Any]] = []
    scope_seed = int(seed)
    for branch_name, reward in BRANCHES:
        for signal in SIGNALS:
            for utility in UTILITIES:
                if signal == "student" and utility == "A_student":
                    units: list[tuple[str, str | None]] = [("case", None)]
                else:
                    units = [("pooled", None)] + [
                        (f"teacher:{teacher}", teacher) for teacher in teachers
                    ]
                for unit_name, teacher_filter in units:
                    if unit_name == "case":
                        rows = case_unit_rows(long_rows, reward=reward)
                    else:
                        rows = paired_rows(
                            long_rows,
                            reward=reward,
                            signal=signal,
                            utility=utility,
                            teacher=teacher_filter,
                        )
                    scope = analyze_scope(
                        rows,
                        branch=branch_name,
                        scope=f"{signal}_probe|{utility}|{unit_name}",
                        bootstrap_rounds=bootstrap_rounds,
                        seed=scope_seed,
                    )
                    # analyze_scope hard-codes the 03 branch labels; repair the
                    # labels while keeping every number untouched.
                    scope["signal"] = f"deltaP_{signal}"
                    scope["utility"] = utility
                    scope["within_trajectory_correlation"]["centering_unit"] = (
                        "trajectory_id"
                        if unit_name == "case"
                        else "trajectory_id + Teacher"
                    )
                    scopes.append(scope)
                    scope_seed += 1
    return scopes


# ---------------------------------------------------------------------------
# Paired head-to-head statistics
# ---------------------------------------------------------------------------


def _center_three(
    x_student: np.ndarray,
    x_teacher: np.ndarray,
    y: np.ndarray,
    units: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, unit in enumerate(units):
        groups[str(unit)].append(index)
    eligible = {key: indices for key, indices in groups.items() if len(indices) >= 2}
    selected = np.asarray(
        [index for indices in eligible.values() for index in indices], dtype=np.int64
    )
    centered_student = np.empty(len(selected), dtype=np.float64)
    centered_teacher = np.empty(len(selected), dtype=np.float64)
    centered_y = np.empty(len(selected), dtype=np.float64)
    offset = 0
    for indices in eligible.values():
        indices_array = np.asarray(indices, dtype=np.int64)
        count = len(indices)
        centered_student[offset : offset + count] = x_student[indices_array] - x_student[
            indices_array
        ].mean()
        centered_teacher[offset : offset + count] = x_teacher[indices_array] - x_teacher[
            indices_array
        ].mean()
        centered_y[offset : offset + count] = y[indices_array] - y[indices_array].mean()
        offset += count
    return selected, centered_student, centered_teacher, centered_y


def _ols_two_probes(
    y: np.ndarray,
    x_student: np.ndarray,
    x_teacher: np.ndarray,
    p_before: np.ndarray,
    v_before: np.ndarray,
    position: np.ndarray,
) -> np.ndarray | None:
    n = len(y)
    design = np.column_stack(
        [np.ones(n), x_student, x_teacher, p_before, v_before, position]
    ).astype(np.float64)
    if n <= design.shape[1] or np.linalg.matrix_rank(design) < design.shape[1]:
        return None
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return coefficients


def _difference_block(distribution: Sequence[float], point: float) -> dict[str, Any]:
    return {
        "difference": r6(point),
        "ci95_prompt_cluster_bootstrap": percentile_ci(distribution),
        "p_value": r6(bootstrap_two_sided_p(distribution)),
        "p_value_method": "two-sided prompt-cluster bootstrap sign test",
    }


def _coefficient_block(distribution: Sequence[float], point: float) -> dict[str, Any]:
    block = _difference_block(distribution, point)
    return {
        "coefficient": block["difference"],
        "ci95_prompt_cluster_bootstrap": block["ci95_prompt_cluster_bootstrap"],
        "p_value": block["p_value"],
        "p_value_method": block["p_value_method"],
    }


def _paired_difference(first: float | None, second: float | None) -> float:
    if first is None or second is None:
        return _nan()
    if not (math.isfinite(first) and math.isfinite(second)):
        return _nan()
    return first - second


def analyze_head_to_head(
    rows: Sequence[dict[str, Any]],
    *,
    branch: str,
    utility: str,
    bootstrap_rounds: int,
    seed: int,
) -> dict[str, Any]:
    """Compare both probes against the same utility on identical rows.

    Every (step, Teacher) row carries both ``deltaP_student`` and
    ``deltaP_teacher``, so one prompt-cluster bootstrap replicate yields a
    paired correlation difference.  Because each case contributes one identical
    (deltaP_student, A_student) point per Teacher, the Student-probe Pearson on
    this grid equals its case-level value exactly.
    """

    x_student = np.asarray([float(row["deltaP_student"]) for row in rows], dtype=np.float64)
    x_teacher = np.asarray([float(row["deltaP_teacher"]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row[utility]) for row in rows], dtype=np.float64)
    p_before = np.asarray([float(row["P_before"]) for row in rows], dtype=np.float64)
    v_before = np.asarray([float(row["V_before"]) for row in rows], dtype=np.float64)
    position = np.asarray(
        [float(row["relative_position"]) for row in rows], dtype=np.float64
    )
    prompts = np.asarray([str(row["prompt_id"]) for row in rows], dtype=object)
    units = np.asarray(
        [f"{row['trajectory_id']}|{row['teacher']}" for row in rows], dtype=object
    )
    if not all(np.all(np.isfinite(values)) for values in (x_student, x_teacher, y)):
        raise ValueError(f"Non-finite head-to-head value in {branch}/{utility}.")

    cluster_groups = _cluster_indices(prompts)
    selected, centered_student, centered_teacher, centered_y = _center_three(
        x_student, x_teacher, y, units
    )
    within_prompts = prompts[selected]
    within_cluster_groups = _cluster_indices(within_prompts) if len(selected) else []

    distribution_keys = (
        "global_pearson_student",
        "global_pearson_teacher",
        "global_pearson_difference",
        "global_spearman_student",
        "global_spearman_teacher",
        "global_spearman_difference",
        "within_pearson_student",
        "within_pearson_teacher",
        "within_pearson_difference",
        "within_spearman_student",
        "within_spearman_teacher",
        "within_spearman_difference",
        "regression_student",
        "regression_teacher",
        "regression_difference",
    )
    distributions: dict[str, list[float]] = {key: [] for key in distribution_keys}

    rng = np.random.default_rng(seed)
    for _ in range(bootstrap_rounds):
        chosen = rng.integers(0, len(cluster_groups), size=len(cluster_groups))
        indices = np.concatenate([cluster_groups[int(index)] for index in chosen])
        r_student = pearson(x_student[indices], y[indices])
        r_teacher = pearson(x_teacher[indices], y[indices])
        rho_student = spearman(x_student[indices], y[indices])
        rho_teacher = spearman(x_teacher[indices], y[indices])
        distributions["global_pearson_student"].append(
            r_student if r_student is not None else _nan()
        )
        distributions["global_pearson_teacher"].append(
            r_teacher if r_teacher is not None else _nan()
        )
        distributions["global_pearson_difference"].append(
            _paired_difference(r_student, r_teacher)
        )
        distributions["global_spearman_student"].append(
            rho_student if rho_student is not None else _nan()
        )
        distributions["global_spearman_teacher"].append(
            rho_teacher if rho_teacher is not None else _nan()
        )
        distributions["global_spearman_difference"].append(
            _paired_difference(rho_student, rho_teacher)
        )

        if within_cluster_groups:
            within_chosen = rng.integers(
                0, len(within_cluster_groups), size=len(within_cluster_groups)
            )
            within_indices = np.concatenate(
                [within_cluster_groups[int(index)] for index in within_chosen]
            )
            wr_student = pearson(centered_student[within_indices], centered_y[within_indices])
            wr_teacher = pearson(centered_teacher[within_indices], centered_y[within_indices])
            wrho_student = spearman(
                centered_student[within_indices], centered_y[within_indices]
            )
            wrho_teacher = spearman(
                centered_teacher[within_indices], centered_y[within_indices]
            )
            distributions["within_pearson_student"].append(
                wr_student if wr_student is not None else _nan()
            )
            distributions["within_pearson_teacher"].append(
                wr_teacher if wr_teacher is not None else _nan()
            )
            distributions["within_pearson_difference"].append(
                _paired_difference(wr_student, wr_teacher)
            )
            distributions["within_spearman_student"].append(
                wrho_student if wrho_student is not None else _nan()
            )
            distributions["within_spearman_teacher"].append(
                wrho_teacher if wrho_teacher is not None else _nan()
            )
            distributions["within_spearman_difference"].append(
                _paired_difference(wrho_student, wrho_teacher)
            )

        coefficients = _ols_two_probes(
            y[indices],
            x_student[indices],
            x_teacher[indices],
            p_before[indices],
            v_before[indices],
            position[indices],
        )
        if coefficients is None:
            distributions["regression_student"].append(_nan())
            distributions["regression_teacher"].append(_nan())
            distributions["regression_difference"].append(_nan())
        else:
            distributions["regression_student"].append(float(coefficients[1]))
            distributions["regression_teacher"].append(float(coefficients[2]))
            distributions["regression_difference"].append(
                float(coefficients[1] - coefficients[2])
            )

    point_r_student = pearson(x_student, y)
    point_r_teacher = pearson(x_teacher, y)
    point_rho_student = spearman(x_student, y)
    point_rho_teacher = spearman(x_teacher, y)

    global_grid = {
        "student_probe": correlation_result(
            x_student,
            y,
            distributions["global_pearson_student"],
            distributions["global_spearman_student"],
        ),
        "teacher_probe": correlation_result(
            x_teacher,
            y,
            distributions["global_pearson_teacher"],
            distributions["global_spearman_teacher"],
        ),
        "pearson_difference": _difference_block(
            distributions["global_pearson_difference"],
            _paired_difference(point_r_student, point_r_teacher),
        ),
        "spearman_difference": _difference_block(
            distributions["global_spearman_difference"],
            _paired_difference(point_rho_student, point_rho_teacher),
        ),
    }

    within_grid: dict[str, Any] = {
        "centering_unit": "trajectory_id + Teacher",
        "n": int(len(selected)),
    }
    if len(selected):
        point_within_r_student = pearson(centered_student, centered_y)
        point_within_r_teacher = pearson(centered_teacher, centered_y)
        point_within_rho_student = spearman(centered_student, centered_y)
        point_within_rho_teacher = spearman(centered_teacher, centered_y)
        within_grid.update(
            {
                "student_probe": correlation_result(
                    centered_student,
                    centered_y,
                    distributions["within_pearson_student"],
                    distributions["within_spearman_student"],
                ),
                "teacher_probe": correlation_result(
                    centered_teacher,
                    centered_y,
                    distributions["within_pearson_teacher"],
                    distributions["within_spearman_teacher"],
                ),
                "pearson_difference": _difference_block(
                    distributions["within_pearson_difference"],
                    _paired_difference(point_within_r_student, point_within_r_teacher),
                ),
                "spearman_difference": _difference_block(
                    distributions["within_spearman_difference"],
                    _paired_difference(point_within_rho_student, point_within_rho_teacher),
                ),
            }
        )

    horse_race: dict[str, Any] = {
        "formula": HORSE_RACE_FORMULA,
        "n": int(len(y)),
    }
    point_coefficients = _ols_two_probes(y, x_student, x_teacher, p_before, v_before, position)
    if point_coefficients is None:
        horse_race["deltaP_student_coefficient"] = None
        horse_race["deltaP_teacher_coefficient"] = None
        horse_race["student_minus_teacher"] = None
    else:
        horse_race["deltaP_student_coefficient"] = _coefficient_block(
            distributions["regression_student"], float(point_coefficients[1])
        )
        horse_race["deltaP_teacher_coefficient"] = _coefficient_block(
            distributions["regression_teacher"], float(point_coefficients[2])
        )
        horse_race["student_minus_teacher"] = _difference_block(
            distributions["regression_difference"],
            float(point_coefficients[1] - point_coefficients[2]),
        )

    return {
        "branch": branch,
        "utility": utility,
        "n": int(len(rows)),
        "row_unit": (
            "one row per (selected ERSR step, Teacher); both probes on identical rows"
        ),
        "global": global_grid,
        "within_trajectory": within_grid,
        "horse_race_regression": horse_race,
    }


def build_head_to_head(
    long_rows: Sequence[dict[str, Any]],
    *,
    bootstrap_rounds: int,
    seed: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    offset = 0
    for _branch_name, reward in BRANCHES:
        rows = [row for row in long_rows if int(row["reward"]) == reward]
        if not rows:
            raise ValueError(f"No reward={reward} rows for the head-to-head analysis.")
        for utility in UTILITIES:
            results.append(
                analyze_head_to_head(
                    rows,
                    branch=_branch_name,
                    utility=utility,
                    bootstrap_rounds=bootstrap_rounds,
                    seed=int(seed) + offset,
                )
            )
            offset += 1
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _head_to_head_csv(head_to_head: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def metric(prefix: str, block: dict[str, Any], value_key: str) -> dict[str, Any]:
        ci = block.get("ci95_prompt_cluster_bootstrap") or [None, None]
        return {
            f"{prefix}": block.get(value_key),
            f"{prefix}_ci95_low": ci[0],
            f"{prefix}_ci95_high": ci[1],
            f"{prefix}_p_value": block.get("p_value"),
        }

    for block in head_to_head:
        row: dict[str, Any] = {
            "branch": block["branch"],
            "utility": block["utility"],
            "n": block["n"],
            "within_n": block["within_trajectory"].get("n"),
        }
        for scope_name, grid in (
            ("global", block["global"]),
            ("within", block["within_trajectory"]),
        ):
            for probe in ("student_probe", "teacher_probe"):
                probe_block = grid.get(probe)
                if not probe_block:
                    continue
                for kind, key in (("pearson", "r"), ("spearman", "rho")):
                    row.update(
                        metric(f"{scope_name}_{kind}_{probe}", probe_block[kind], key)
                    )
            for kind in ("pearson_difference", "spearman_difference"):
                difference = grid.get(kind)
                if difference:
                    row.update(metric(f"{scope_name}_{kind}", difference, "difference"))
        horse = block["horse_race_regression"]
        for name, key, value_key in (
            (
                "regression_deltaP_student_coefficient",
                "deltaP_student_coefficient",
                "coefficient",
            ),
            (
                "regression_deltaP_teacher_coefficient",
                "deltaP_teacher_coefficient",
                "coefficient",
            ),
            ("regression_student_minus_teacher", "student_minus_teacher", "difference"),
        ):
            coefficient = horse.get(key)
            if coefficient:
                row.update(metric(name, coefficient, value_key))
        rows.append(row)
    return rows


def build_conclusions(
    scopes: Sequence[dict[str, Any]],
    head_to_head: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    by_key = {(scope["branch"], scope["scope"]): scope for scope in scopes}

    def scope_for(branch: str, signal: str, utility: str) -> dict[str, Any]:
        unit = "case" if (signal == "student" and utility == "A_student") else "pooled"
        return by_key[(branch, f"{signal}_probe|{utility}|{unit}")]

    def predicts(scope: dict[str, Any]) -> bool:
        global_block = scope["global_correlation"]
        within_block = scope["within_trajectory_correlation"]
        return all(
            _positive_finding(global_block[kind]) and _positive_finding(within_block[kind])
            for kind in ("pearson", "spearman")
        )

    branch_flags: dict[str, Any] = {}
    for branch, _reward in BRANCHES:
        branch_flags[branch] = {
            "student_probe_predicts_A_student": predicts(
                scope_for(branch, "student", "A_student")
            ),
            "teacher_probe_predicts_A_teacher": predicts(
                scope_for(branch, "teacher", "A_teacher")
            ),
            "student_probe_predicts_A_teacher": predicts(
                scope_for(branch, "student", "A_teacher")
            ),
            "teacher_probe_predicts_A_student": predicts(
                scope_for(branch, "teacher", "A_student")
            ),
        }

    head_to_head_summary: dict[str, Any] = {}
    for block in head_to_head:
        r_student = block["global"]["student_probe"]["pearson"]["r"]
        r_teacher = block["global"]["teacher_probe"]["pearson"]["r"]
        difference = block["global"]["pearson_difference"]
        ci = difference.get("ci95_prompt_cluster_bootstrap")
        if r_student is None or r_teacher is None:
            winner = None
        elif r_student > r_teacher:
            winner = "student_probe"
        elif r_teacher > r_student:
            winner = "teacher_probe"
        else:
            winner = "tie"
        head_to_head_summary[f"{block['branch']}|{block['utility']}"] = {
            "pearson_student": r_student,
            "pearson_teacher": r_teacher,
            "pearson_difference": difference["difference"],
            "ci95_prompt_cluster_bootstrap": ci,
            "p_value": difference["p_value"],
            "difference_significant": bool(
                ci is not None and (ci[0] > 0.0 or ci[1] < 0.0)
            ),
            "point_estimate_winner": winner,
        }

    horse_race_summary: dict[str, Any] = {}
    for block in head_to_head:
        horse = block["horse_race_regression"]
        entry: dict[str, Any] = {}
        for label, key in (
            ("student", "deltaP_student_coefficient"),
            ("teacher", "deltaP_teacher_coefficient"),
        ):
            coefficient = horse.get(key)
            if not coefficient:
                entry[f"{label}_coefficient"] = None
                entry[f"{label}_ci95"] = None
                entry[f"{label}_ci95_excludes_zero"] = None
                continue
            ci = coefficient.get("ci95_prompt_cluster_bootstrap")
            entry[f"{label}_coefficient"] = coefficient.get("coefficient")
            entry[f"{label}_ci95"] = ci
            entry[f"{label}_ci95_excludes_zero"] = bool(
                ci is not None and (ci[0] > 0.0 or ci[1] < 0.0)
            )
        horse_race_summary[f"{block['branch']}|{block['utility']}"] = entry

    return {
        "branch_flags": branch_flags,
        "head_to_head": head_to_head_summary,
        "horse_race": horse_race_summary,
        "note": (
            "Mechanical summaries of the observed estimates and prompt-cluster "
            "bootstrap intervals; the headline head-to-head numbers use the "
            "global Pearson comparison. They do not force any expected conclusion."
        ),
    }


def _write_report_csvs(
    output_dir: Path,
    long_rows: list[dict[str, Any]],
    scopes: list[dict[str, Any]],
    head_to_head: list[dict[str, Any]],
    *,
    source_path: Path | None = None,
) -> dict[str, str]:
    outputs: dict[str, str] = {}

    def write(name: str, rows: list[dict[str, Any]]) -> None:
        path = output_dir / name
        if not rows:
            raise RuntimeError(f"Refusing to create empty required output {path}.")
        atomic_csv(path, rows, list(rows[0]))
        outputs[name] = str(path)

    step_data_output = output_dir / "05_probe_step_data.csv"
    if source_path is None or step_data_output.resolve() != source_path.resolve():
        atomic_csv(step_data_output, long_rows, STEP_DATA_FIELDS)
    outputs[step_data_output.name] = str(step_data_output)
    write("05_scope_global_correlations.csv", _global_csv(scopes))
    write("05_scope_within_trajectory_correlations.csv", _within_csv(scopes))
    write("05_scope_quartile_stats.csv", _quartile_csv(scopes))
    write("05_scope_extreme_groups.csv", _extreme_csv(scopes))
    write("05_scope_sign_stats.csv", _sign_csv(scopes))
    write("05_scope_regression_robustness.csv", _regression_csv(scopes))
    write("05_head_to_head.csv", _head_to_head_csv(head_to_head))
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--step-data",
        "--step-scores",
        dest="step_data",
        type=Path,
        default=DEFAULT_STEP_SCORES,
        help="03_probe_step_scores.jsonl or the compact 05_probe_step_data.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-rounds", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260920)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_rounds < 0:
        raise ValueError("--bootstrap-rounds cannot be negative.")
    step_data_path = args.step_data.resolve()
    output_dir = args.output_dir.resolve()
    if not step_data_path.is_file():
        raise FileNotFoundError(step_data_path)
    long_rows, source_info = load_long_rows(step_data_path)
    teachers = sorted({str(row["teacher"]) for row in long_rows})
    correct_cases = {str(row["case_id"]) for row in long_rows if int(row["reward"]) == 1}
    incorrect_cases = {
        str(row["case_id"]) for row in long_rows if int(row["reward"]) == 0
    }
    if not correct_cases or not incorrect_cases:
        raise ValueError("Both correct and incorrect branches are required.")

    scopes = build_scopes(
        long_rows,
        teachers=teachers,
        bootstrap_rounds=int(args.bootstrap_rounds),
        seed=int(args.seed),
    )
    head_to_head = build_head_to_head(
        long_rows,
        bootstrap_rounds=int(args.bootstrap_rounds),
        seed=int(args.seed) + 100_000,
    )
    conclusions = build_conclusions(scopes, head_to_head)
    outputs = _write_report_csvs(
        output_dir, long_rows, scopes, head_to_head, source_path=step_data_path
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "script": "tests/analyze_ersr_answer_probes_05.py",
        "input": {
            "step_data": str(step_data_path),
            **source_info,
            "step_teacher_rows": len(long_rows),
            "teachers": teachers,
            "correct_cases": len(correct_cases),
            "incorrect_cases": len(incorrect_cases),
            "bootstrap_rounds": int(args.bootstrap_rounds),
            "seed": int(args.seed),
        },
        "definitions": {
            "P": "exp(mean gold-answer-token logprob) under the Student; geometric-mean token probability",
            "deltaP_student": "P_student_after - P_before (the Student's own step)",
            "deltaP_teacher": "P_teacher_after - P_before for one Teacher replacement",
            "A_student": "V_student_after - V_before = ERSR V(keep) - V(redecide)",
            "A_teacher": "V_teacher_after - V_before = ERSR V(replace:<teacher>) - V(redecide) = A^TI",
            "branch": "correct = reward 1 trajectory, incorrect = reward 0 trajectory",
            "scope_matrix": (
                "branch x signal x utility; student_probe|A_student uses one deduplicated "
                "row per case, every other scope uses one row per (case, Teacher), either "
                "pooled across Teachers or filtered to a single Teacher"
            ),
            "head_to_head": (
                "both probes correlated against the same utility on identical (case, Teacher) "
                "rows; one shared prompt-cluster bootstrap makes the correlation difference "
                "a paired statistic; Pearson is invariant to the per-case duplication of "
                "Student-side values, so the student-probe r equals its case-level value"
            ),
            "within_trajectory": (
                "both signals and the utility are centered within each (trajectory, Teacher) "
                "unit before correlation"
            ),
            "horse_race_regression": (
                "utility ~ deltaP_student + deltaP_teacher + P_before + V_before + "
                "relative_position; each probe coefficient measures incremental predictive "
                "power beyond the other probe"
            ),
            "primary_inference": (
                "prompt-level cluster bootstrap: a sampled prompt carries all trajectories, "
                "steps, and Teacher interventions"
            ),
        },
        "scopes": {f"{scope['branch']}|{scope['scope']}": scope for scope in scopes},
        "head_to_head": {
            f"{block['branch']}|{block['utility']}": block for block in head_to_head
        },
        "conclusions": conclusions,
        "outputs": outputs,
        "caveats": [
            "This supplement re-uses the unchanged 03 step scores; no new probes were collected.",
            "Only reasoning steps selected and evaluated by ERSR are available; within-trajectory analysis excludes single-step centering units.",
            "Prompt-cluster resampling keeps both Teachers and all same-prompt rows together; analytic iid p-values are reference diagnostics only.",
            "Teacher replacements flagged truncated are retained and marked in 05_probe_step_data.csv so sensitivity filtering remains possible offline.",
        ],
    }
    report_path = output_dir / "05_probe_analysis.json"
    atomic_json(report_path, report)
    outputs[report_path.name] = str(report_path)
    atomic_json(output_dir / "05_probe_output_manifest.json", {"outputs": outputs})
    print(
        json.dumps(
            {
                "event": "05_probe_analysis_complete",
                "step_records": source_info["step_score_records"],
                "step_teacher_rows": len(long_rows),
                "teachers": teachers,
                "scopes": len(scopes),
                "bootstrap_rounds": int(args.bootstrap_rounds),
                "output_dir": str(output_dir),
                "conclusions": conclusions,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
