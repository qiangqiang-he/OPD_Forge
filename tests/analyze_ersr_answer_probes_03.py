#!/usr/bin/env python3
"""Analyze ERSR answer probes for analysis 03 and emit numbered CSV/JSON files.

Primary inference uses a prompt-level cluster bootstrap.  Sampling a prompt
brings all of its trajectories, steps, and Teacher interventions into the
bootstrap replicate.  The script reports global and trajectory-centered
Pearson/Spearman correlations, probe quartiles, extreme 20% groups, probe-sign
groups, and a starting-state/position-controlled OLS coefficient.

The input is ``03_probe_step_scores.jsonl`` from
``tests/collect_ersr_answer_probes_03.py``.  For the two-Teacher ERSR run, the
CSV is long-form with one row per (step, Teacher); Student-branch statistics
deduplicate by ``case_id``, while the failed Teacher branch reports pooled and
per-Teacher scopes.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    REPO_ROOT
    / "outputs/ersr_dapo17k_qwen3_1p7b_mc128/probe_analysis_03/03_probe_step_scores.jsonl"
)
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT.parent
SCHEMA_VERSION = 1
QUARTILE_LABELS = ("Q1", "Q2", "Q3", "Q4")

try:
    from scipy import stats as _scipy_stats
except Exception:  # pragma: no cover - scipy is optional on the server
    _scipy_stats = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def r6(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return round(value, 6) if math.isfinite(value) else None
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def atomic_csv(path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected an object at {path}:{line_number}.")
            rows.append(value)
    if not rows:
        raise ValueError(f"No records found in {path}.")
    return rows


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    if len(values) == 0:
        return ranks
    sorted_values = values[order]
    starts = np.flatnonzero(
        np.concatenate(([True], sorted_values[1:] != sorted_values[:-1]))
    )
    ends = np.concatenate((starts[1:], [len(values)]))
    group_ranks = (starts + 1 + ends) / 2.0
    ranks[order] = np.repeat(group_ranks, ends - starts)
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) != len(y) or len(x) < 2:
        return None
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = float(np.sqrt(np.dot(x_centered, x_centered) * np.dot(y_centered, y_centered)))
    if denominator <= 0.0:
        return None
    return float(np.dot(x_centered, y_centered) / denominator)


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2:
        return None
    return pearson(average_ranks(x), average_ranks(y))


def analytic_correlation_p(value: float | None, n: int) -> float | None:
    if value is None or n < 3:
        return None
    if abs(value) >= 1.0:
        return 0.0
    t_value = value * math.sqrt((n - 2) / max(1e-300, 1.0 - value * value))
    if _scipy_stats is not None:
        return float(2.0 * _scipy_stats.t.sf(abs(t_value), n - 2))
    return float(math.erfc(abs(t_value) / math.sqrt(2.0)))


def percentile_ci(values: Sequence[float]) -> list[float] | None:
    finite = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    if len(finite) < 2:
        return None
    result = np.quantile(finite, [0.025, 0.975])
    return [r6(result[0]), r6(result[1])]


def bootstrap_two_sided_p(values: Sequence[float], null: float = 0.0) -> float | None:
    finite = np.asarray([value for value in values if math.isfinite(float(value))], dtype=np.float64)
    if len(finite) == 0:
        return None
    below = (float(np.count_nonzero(finite <= null)) + 1.0) / (len(finite) + 1.0)
    above = (float(np.count_nonzero(finite >= null)) + 1.0) / (len(finite) + 1.0)
    return min(1.0, 2.0 * min(below, above))


def mean_ci_from_bootstrap(point: float | None, distribution: Sequence[float]) -> dict[str, Any]:
    return {
        "mean": r6(point),
        "ci95_prompt_cluster_bootstrap": percentile_ci(distribution),
        "bootstrap_valid_rounds": sum(math.isfinite(float(value)) for value in distribution),
    }


def correlation_result(
    x: np.ndarray,
    y: np.ndarray,
    pearson_boot: Sequence[float],
    spearman_boot: Sequence[float],
) -> dict[str, Any]:
    pearson_value = pearson(x, y)
    spearman_value = spearman(x, y)
    return {
        "n": len(x),
        "pearson": {
            "r": r6(pearson_value),
            "ci95_prompt_cluster_bootstrap": percentile_ci(pearson_boot),
            "p_value": r6(bootstrap_two_sided_p(pearson_boot)),
            "p_value_method": "two-sided prompt-cluster bootstrap sign test",
            "p_value_analytic_iid_reference": r6(analytic_correlation_p(pearson_value, len(x))),
            "bootstrap_valid_rounds": sum(math.isfinite(float(value)) for value in pearson_boot),
        },
        "spearman": {
            "rho": r6(spearman_value),
            "ci95_prompt_cluster_bootstrap": percentile_ci(spearman_boot),
            "p_value": r6(bootstrap_two_sided_p(spearman_boot)),
            "p_value_method": "two-sided prompt-cluster bootstrap sign test",
            "p_value_analytic_iid_reference": r6(analytic_correlation_p(spearman_value, len(x))),
            "bootstrap_valid_rounds": sum(math.isfinite(float(value)) for value in spearman_boot),
        },
    }


def quartile_membership(x: np.ndarray) -> tuple[np.ndarray, list[float]]:
    thresholds = np.quantile(x, [0.25, 0.5, 0.75])
    membership = np.digitize(x, thresholds, right=True)
    return membership, [float(value) for value in thresholds]


def extreme_masks(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    low, high = np.quantile(x, [0.2, 0.8])
    return x <= low, x >= high, float(low), float(high)


def _center_within(
    x: np.ndarray, y: np.ndarray, trajectory_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, trajectory_id in enumerate(trajectory_ids):
        groups[str(trajectory_id)].append(index)
    eligible = {key: indices for key, indices in groups.items() if len(indices) >= 2}
    selected = np.asarray(
        [index for indices in eligible.values() for index in indices], dtype=np.int64
    )
    centered_x = np.empty(len(selected), dtype=np.float64)
    centered_y = np.empty(len(selected), dtype=np.float64)
    offset = 0
    for indices in eligible.values():
        indices_array = np.asarray(indices, dtype=np.int64)
        count = len(indices)
        centered_x[offset : offset + count] = x[indices_array] - x[indices_array].mean()
        centered_y[offset : offset + count] = y[indices_array] - y[indices_array].mean()
        offset += count
    return selected, centered_x, centered_y


def _ols_probe_coefficient(
    y: np.ndarray,
    probe_gain: np.ndarray,
    p_before: np.ndarray,
    v_before: np.ndarray,
    position: np.ndarray,
) -> dict[str, Any]:
    n = len(y)
    design = np.column_stack(
        [np.ones(n), probe_gain, p_before, v_before, position]
    ).astype(np.float64)
    empty = {
        "n": n,
        "coefficient": None,
        "standard_error_iid_reference": None,
        "p_value_analytic_iid_reference": None,
        "rank": int(np.linalg.matrix_rank(design)) if n else 0,
    }
    if n <= design.shape[1] or empty["rank"] < design.shape[1]:
        return empty
    coefficients, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coefficients
    dof = n - design.shape[1]
    sigma2 = float(np.dot(residuals, residuals) / dof)
    covariance = sigma2 * np.linalg.pinv(design.T @ design)
    variance = max(0.0, float(covariance[1, 1]))
    standard_error = math.sqrt(variance)
    coefficient = float(coefficients[1])
    if standard_error == 0.0:
        p_value = 0.0 if coefficient != 0.0 else 1.0
    else:
        t_value = coefficient / standard_error
        if _scipy_stats is not None:
            p_value = float(2.0 * _scipy_stats.t.sf(abs(t_value), dof))
        else:
            p_value = float(math.erfc(abs(t_value) / math.sqrt(2.0)))
    return {
        "n": n,
        "coefficient": coefficient,
        "standard_error_iid_reference": standard_error,
        "p_value_analytic_iid_reference": p_value,
        "rank": int(empty["rank"]),
    }


def _cluster_indices(prompt_ids: np.ndarray) -> list[np.ndarray]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, prompt_id in enumerate(prompt_ids):
        groups[str(prompt_id)].append(index)
    return [np.asarray(indices, dtype=np.int64) for indices in groups.values()]


def _nan() -> float:
    return float("nan")


def analyze_scope(
    rows: Sequence[dict[str, Any]],
    *,
    branch: str,
    scope: str,
    bootstrap_rounds: int,
    seed: int,
) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"No rows for {branch}/{scope}.")
    x = np.asarray([float(row["probe_gain"]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row["utility"]) for row in rows], dtype=np.float64)
    p_before = np.asarray([float(row["P_before"]) for row in rows], dtype=np.float64)
    v_before = np.asarray([float(row["V_before"]) for row in rows], dtype=np.float64)
    position = np.asarray([float(row["relative_position"]) for row in rows], dtype=np.float64)
    prompts = np.asarray([str(row["prompt_id"]) for row in rows], dtype=object)
    trajectories = np.asarray([str(row["within_trajectory_id"]) for row in rows], dtype=object)
    case_ids = np.asarray([str(row["case_id"]) for row in rows], dtype=object)
    if not all(
        np.all(np.isfinite(values)) for values in (x, y, p_before, v_before, position)
    ):
        raise ValueError(f"Non-finite analysis value in {branch}/{scope}.")

    cluster_groups = _cluster_indices(prompts)
    within_selected, centered_x, centered_y = _center_within(x, y, trajectories)
    within_prompts = prompts[within_selected]
    within_cluster_groups = _cluster_indices(within_prompts) if len(within_selected) else []
    quartile_bins, quartile_thresholds = quartile_membership(x)
    bottom_mask, top_mask, bottom_threshold, top_threshold = extreme_masks(x)
    regression_point = _ols_probe_coefficient(y, x, p_before, v_before, position)

    distributions: dict[str, Any] = {
        "global_pearson": [],
        "global_spearman": [],
        "within_pearson": [],
        "within_spearman": [],
        "quartile_utility": [[] for _ in range(4)],
        "extreme_bottom_mean": [],
        "extreme_top_mean": [],
        "extreme_difference": [],
        "sign_positive_mean": [],
        "sign_nonpositive_mean": [],
        "sign_difference": [],
        "sign_positive_probability_utility_gt_zero": [],
        "sign_nonpositive_probability_utility_gt_zero": [],
        "regression_coefficient": [],
    }
    rng = np.random.default_rng(seed)
    for _ in range(bootstrap_rounds):
        chosen = rng.integers(0, len(cluster_groups), size=len(cluster_groups))
        indices = np.concatenate([cluster_groups[int(index)] for index in chosen])
        bx, by = x[indices], y[indices]
        global_pearson_value = pearson(bx, by)
        global_spearman_value = spearman(bx, by)
        distributions["global_pearson"].append(
            global_pearson_value if global_pearson_value is not None else _nan()
        )
        distributions["global_spearman"].append(
            global_spearman_value if global_spearman_value is not None else _nan()
        )

        if within_cluster_groups:
            within_chosen = rng.integers(
                0, len(within_cluster_groups), size=len(within_cluster_groups)
            )
            within_indices = np.concatenate(
                [within_cluster_groups[int(index)] for index in within_chosen]
            )
            within_pearson_value = pearson(
                centered_x[within_indices], centered_y[within_indices]
            )
            within_spearman_value = spearman(
                centered_x[within_indices], centered_y[within_indices]
            )
            distributions["within_pearson"].append(
                within_pearson_value if within_pearson_value is not None else _nan()
            )
            distributions["within_spearman"].append(
                within_spearman_value if within_spearman_value is not None else _nan()
            )

        b_quartiles, _ = quartile_membership(bx)
        for quartile in range(4):
            mask = b_quartiles == quartile
            distributions["quartile_utility"][quartile].append(
                float(by[mask].mean()) if np.any(mask) else _nan()
            )
        b_bottom, b_top, _, _ = extreme_masks(bx)
        bottom_mean = float(by[b_bottom].mean()) if np.any(b_bottom) else _nan()
        top_mean = float(by[b_top].mean()) if np.any(b_top) else _nan()
        distributions["extreme_bottom_mean"].append(bottom_mean)
        distributions["extreme_top_mean"].append(top_mean)
        distributions["extreme_difference"].append(top_mean - bottom_mean)

        b_positive = bx > 0.0
        b_nonpositive = ~b_positive
        positive_mean = float(by[b_positive].mean()) if np.any(b_positive) else _nan()
        nonpositive_mean = (
            float(by[b_nonpositive].mean()) if np.any(b_nonpositive) else _nan()
        )
        distributions["sign_positive_mean"].append(positive_mean)
        distributions["sign_nonpositive_mean"].append(nonpositive_mean)
        distributions["sign_difference"].append(positive_mean - nonpositive_mean)
        distributions["sign_positive_probability_utility_gt_zero"].append(
            float(np.mean(by[b_positive] > 0.0)) if np.any(b_positive) else _nan()
        )
        distributions["sign_nonpositive_probability_utility_gt_zero"].append(
            float(np.mean(by[b_nonpositive] > 0.0)) if np.any(b_nonpositive) else _nan()
        )
        regression = _ols_probe_coefficient(
            by,
            bx,
            p_before[indices],
            v_before[indices],
            position[indices],
        )
        distributions["regression_coefficient"].append(
            float(regression["coefficient"])
            if regression["coefficient"] is not None
            else _nan()
        )

    global_result = correlation_result(
        x,
        y,
        distributions["global_pearson"],
        distributions["global_spearman"],
    )
    global_result.update(
        {
            "observation_count": len(rows),
            "unique_step_count": len(set(case_ids)),
            "trajectory_count": len({str(row["trajectory_id"]) for row in rows}),
            "prompt_count": len(cluster_groups),
        }
    )

    within_result = correlation_result(
        centered_x,
        centered_y,
        distributions["within_pearson"],
        distributions["within_spearman"],
    )
    eligible_centering_units = set(str(value) for value in trajectories[within_selected])
    eligible_case_ids = {str(case_ids[index]) for index in within_selected}
    eligible_trajectories = {
        str(row["trajectory_id"])
        for row in rows
        if str(row["case_id"]) in eligible_case_ids
    }
    within_result.update(
        {
            "centered_step_observation_count": len(centered_x),
            "eligible_trajectory_count": len(eligible_trajectories),
            "eligible_centering_unit_count": len(eligible_centering_units),
            "excluded_single_step_trajectory_count": len(
                {str(row["trajectory_id"]) for row in rows}
            )
            - len(eligible_trajectories),
            "excluded_single_step_centering_unit_count": len(set(trajectories))
            - len(eligible_centering_units),
            "prompt_count": len(set(str(value) for value in within_prompts)),
            "centering_unit": (
                "trajectory_id" if branch == "correct_student" else "trajectory_id + Teacher"
            ),
        }
    )

    quartiles: list[dict[str, Any]] = []
    for quartile, label in enumerate(QUARTILE_LABELS):
        mask = quartile_bins == quartile
        subset_rows = [row for row, include in zip(rows, mask, strict=True) if include]
        quartiles.append(
            {
                "probe_bin": label,
                "step_count": int(np.count_nonzero(mask)),
                "unique_step_count": len({str(row["case_id"]) for row in subset_rows}),
                "trajectory_count": len(
                    {str(row["trajectory_id"]) for row in subset_rows}
                ),
                "prompt_count": len({str(row["prompt_id"]) for row in subset_rows}),
                "mean_probe_gain": r6(float(x[mask].mean())) if np.any(mask) else None,
                "mean_utility": r6(float(y[mask].mean())) if np.any(mask) else None,
                "utility_ci95_prompt_cluster_bootstrap": percentile_ci(
                    distributions["quartile_utility"][quartile]
                ),
                "bootstrap_valid_rounds": sum(
                    math.isfinite(float(value))
                    for value in distributions["quartile_utility"][quartile]
                ),
            }
        )

    bottom_rows = [row for row, include in zip(rows, bottom_mask, strict=True) if include]
    top_rows = [row for row, include in zip(rows, top_mask, strict=True) if include]
    bottom_mean = float(y[bottom_mask].mean())
    top_mean = float(y[top_mask].mean())
    extremes = {
        "bottom_20": {
            "threshold_probe_gain_le": r6(bottom_threshold),
            "step_count": int(np.count_nonzero(bottom_mask)),
            "unique_step_count": len({str(row["case_id"]) for row in bottom_rows}),
            "trajectory_count": len(
                {str(row["trajectory_id"]) for row in bottom_rows}
            ),
            "mean_utility": r6(bottom_mean),
            "utility_ci95_prompt_cluster_bootstrap": percentile_ci(
                distributions["extreme_bottom_mean"]
            ),
        },
        "top_20": {
            "threshold_probe_gain_ge": r6(top_threshold),
            "step_count": int(np.count_nonzero(top_mask)),
            "unique_step_count": len({str(row["case_id"]) for row in top_rows}),
            "trajectory_count": len(
                {str(row["trajectory_id"]) for row in top_rows}
            ),
            "mean_utility": r6(top_mean),
            "utility_ci95_prompt_cluster_bootstrap": percentile_ci(
                distributions["extreme_top_mean"]
            ),
        },
        "top_minus_bottom": {
            "difference": r6(top_mean - bottom_mean),
            "ci95_prompt_cluster_bootstrap": percentile_ci(
                distributions["extreme_difference"]
            ),
            "p_value": r6(bootstrap_two_sided_p(distributions["extreme_difference"])),
            "p_value_method": "two-sided prompt-cluster bootstrap sign test",
        },
        "groups_overlap_due_to_ties": bool(bottom_threshold >= top_threshold),
    }

    sign_groups: dict[str, Any] = {}
    for label, mask, mean_key, probability_key in (
        (
            "probe_gain_positive",
            x > 0.0,
            "sign_positive_mean",
            "sign_positive_probability_utility_gt_zero",
        ),
        (
            "probe_gain_nonpositive",
            x <= 0.0,
            "sign_nonpositive_mean",
            "sign_nonpositive_probability_utility_gt_zero",
        ),
    ):
        subset_rows = [row for row, include in zip(rows, mask, strict=True) if include]
        sign_groups[label] = {
            "step_count": int(np.count_nonzero(mask)),
            "unique_step_count": len({str(row["case_id"]) for row in subset_rows}),
            "trajectory_count": len(
                {str(row["trajectory_id"]) for row in subset_rows}
            ),
            "mean_utility": r6(float(y[mask].mean())) if np.any(mask) else None,
            "utility_ci95_prompt_cluster_bootstrap": percentile_ci(distributions[mean_key]),
            "P_utility_gt_zero": r6(float(np.mean(y[mask] > 0.0)))
            if np.any(mask)
            else None,
            "P_utility_gt_zero_ci95_prompt_cluster_bootstrap": percentile_ci(
                distributions[probability_key]
            ),
        }
    sign_groups["positive_minus_nonpositive_mean_utility"] = {
        "difference": r6(
            float(y[x > 0.0].mean() - y[x <= 0.0].mean())
            if np.any(x > 0.0) and np.any(x <= 0.0)
            else None
        ),
        "ci95_prompt_cluster_bootstrap": percentile_ci(distributions["sign_difference"]),
        "p_value": r6(bootstrap_two_sided_p(distributions["sign_difference"])),
    }

    regression = {
        "formula": "utility ~ probe_gain + P_before + V_before + relative_position",
        "probe_gain_coefficient": r6(regression_point["coefficient"]),
        "probe_gain_coefficient_ci95_prompt_cluster_bootstrap": percentile_ci(
            distributions["regression_coefficient"]
        ),
        "probe_gain_coefficient_p_value": r6(
            bootstrap_two_sided_p(distributions["regression_coefficient"])
        ),
        "p_value_method": "two-sided prompt-cluster bootstrap sign test",
        "standard_error_iid_reference": r6(
            regression_point["standard_error_iid_reference"]
        ),
        "p_value_analytic_iid_reference": r6(
            regression_point["p_value_analytic_iid_reference"]
        ),
        "design_rank": regression_point["rank"],
        "n": regression_point["n"],
        "bootstrap_valid_rounds": sum(
            math.isfinite(float(value))
            for value in distributions["regression_coefficient"]
        ),
    }

    means = [entry["mean_utility"] for entry in quartiles]
    quartile_monotonic = all(
        right >= left for left, right in zip(means, means[1:])
    ) if all(value is not None for value in means) else None
    return {
        "branch": branch,
        "scope": scope,
        "signal": "deltaP_student" if branch == "correct_student" else "deltaP_teacher",
        "utility": "A_student" if branch == "correct_student" else "A_teacher",
        "bootstrap_rounds": bootstrap_rounds,
        "global_correlation": global_result,
        "within_trajectory_correlation": within_result,
        "quartiles": {
            "thresholds": [r6(value) for value in quartile_thresholds],
            "groups": quartiles,
            "mean_utility_monotonic_nondecreasing": quartile_monotonic,
        },
        "extreme_groups": extremes,
        "probe_sign": sign_groups,
        "regression_robustness": regression,
    }


def flatten_step_teacher_rows(raw_steps: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_cases: set[str] = set()
    teacher_set: set[str] | None = None
    for step in raw_steps:
        case_id = str(step["case_id"])
        if case_id in seen_cases:
            raise ValueError(f"Duplicate step score case_id {case_id!r}.")
        seen_cases.add(case_id)
        teachers = step.get("teachers")
        if not isinstance(teachers, dict) or not teachers:
            raise ValueError(f"Step {case_id!r} has no Teacher scores.")
        current_teacher_set = set(str(value) for value in teachers)
        if teacher_set is None:
            teacher_set = current_teacher_set
        elif teacher_set != current_teacher_set:
            raise ValueError(f"Step {case_id!r} has an inconsistent Teacher set.")
        reward = int(step["reward"])
        if reward not in (0, 1):
            raise ValueError(f"Step {case_id!r} has reward={reward}, expected 0/1.")
        for teacher, teacher_data in sorted(teachers.items()):
            rows.append(
                {
                    "case_id": case_id,
                    "prompt_id": str(step["prompt_id"]),
                    "trajectory_id": str(step["trajectory_id"]),
                    "reward": reward,
                    "step_id": int(step["step_id"]),
                    "step_ordinal": int(step["step_ordinal"]),
                    "source_step_index": int(step["source_step_index"]),
                    "num_steps": int(step["num_steps"]),
                    "relative_position": float(step["relative_position"]),
                    "prompt_accuracy": float(step["prompt_accuracy"]),
                    "teacher": str(teacher),
                    "teacher_truncated": int(teacher_data["teacher_truncated"]),
                    "P_before": float(step["P_before"]),
                    "P_student_after": float(step["P_student_after"]),
                    "P_teacher_after": float(teacher_data["P_teacher_after"]),
                    "deltaP_student": float(step["deltaP_student"]),
                    "deltaP_teacher": float(teacher_data["deltaP_teacher"]),
                    "mean_logprob_before": float(step["probe_before"]["mean_logprob"]),
                    "mean_logprob_student_after": float(
                        step["probe_student_after"]["mean_logprob"]
                    ),
                    "mean_logprob_teacher_after": float(
                        teacher_data["probe"]["mean_logprob"]
                    ),
                    "delta_logP_student": float(step["delta_logP_student"]),
                    "delta_logP_teacher": float(teacher_data["delta_logP_teacher"]),
                    "V_before": float(step["V_before"]),
                    "V_student_after": float(step["V_student_after"]),
                    "V_teacher_after": float(teacher_data["V_teacher_after"]),
                    "A_student": float(step["A_student"]),
                    "A_teacher": float(teacher_data["A_teacher"]),
                    "mc_k": int(step["mc_k"]),
                }
            )
    return rows


STEP_DATA_FIELDS = (
    "case_id",
    "prompt_id",
    "trajectory_id",
    "reward",
    "step_id",
    "step_ordinal",
    "source_step_index",
    "num_steps",
    "relative_position",
    "prompt_accuracy",
    "teacher",
    "teacher_truncated",
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
    "mc_k",
)


def student_scope_rows(long_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case: dict[str, dict[str, Any]] = {}
    for row in long_rows:
        if int(row["reward"]) != 1:
            continue
        case_id = str(row["case_id"])
        if case_id in by_case:
            previous = by_case[case_id]
            for field in (
                "prompt_id",
                "trajectory_id",
                "P_before",
                "deltaP_student",
                "V_before",
                "A_student",
                "relative_position",
            ):
                if previous[field] != row[field]:
                    raise ValueError(f"Teacher copies disagree on Student field {field}/{case_id}.")
            continue
        by_case[case_id] = {
            **row,
            "probe_gain": float(row["deltaP_student"]),
            "utility": float(row["A_student"]),
            "within_trajectory_id": str(row["trajectory_id"]),
        }
    return list(by_case.values())


def teacher_scope_rows(
    long_rows: Sequence[dict[str, Any]], teacher: str | None
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in long_rows:
        if int(row["reward"]) != 0:
            continue
        if teacher is not None and str(row["teacher"]) != teacher:
            continue
        rows.append(
            {
                **row,
                "probe_gain": float(row["deltaP_teacher"]),
                "utility": float(row["A_teacher"]),
                "within_trajectory_id": f"{row['trajectory_id']}|{row['teacher']}",
            }
        )
    return rows


def _global_csv(scopes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope in scopes:
        data = scope["global_correlation"]
        for correlation_type, coefficient_name in (("pearson", "r"), ("spearman", "rho")):
            metric = data[correlation_type]
            ci = metric["ci95_prompt_cluster_bootstrap"] or [None, None]
            rows.append(
                {
                    "branch": scope["branch"],
                    "scope": scope["scope"],
                    "signal": scope["signal"],
                    "utility": scope["utility"],
                    "observation_count": data["observation_count"],
                    "unique_step_count": data["unique_step_count"],
                    "trajectory_count": data["trajectory_count"],
                    "prompt_count": data["prompt_count"],
                    "correlation_type": correlation_type,
                    "coefficient": metric[coefficient_name],
                    "ci95_low": ci[0],
                    "ci95_high": ci[1],
                    "p_value": metric["p_value"],
                    "p_value_method": metric["p_value_method"],
                    "p_value_analytic_iid_reference": metric[
                        "p_value_analytic_iid_reference"
                    ],
                    "bootstrap_valid_rounds": metric["bootstrap_valid_rounds"],
                }
            )
    return rows


def _within_csv(scopes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope in scopes:
        data = scope["within_trajectory_correlation"]
        for correlation_type, coefficient_name in (("pearson", "r"), ("spearman", "rho")):
            metric = data[correlation_type]
            ci = metric["ci95_prompt_cluster_bootstrap"] or [None, None]
            rows.append(
                {
                    "branch": scope["branch"],
                    "scope": scope["scope"],
                    "signal": scope["signal"],
                    "utility": scope["utility"],
                    "centered_step_observation_count": data[
                        "centered_step_observation_count"
                    ],
                    "eligible_trajectory_count": data["eligible_trajectory_count"],
                    "eligible_centering_unit_count": data[
                        "eligible_centering_unit_count"
                    ],
                    "excluded_single_step_trajectory_count": data[
                        "excluded_single_step_trajectory_count"
                    ],
                    "excluded_single_step_centering_unit_count": data[
                        "excluded_single_step_centering_unit_count"
                    ],
                    "prompt_count": data["prompt_count"],
                    "centering_unit": data["centering_unit"],
                    "correlation_type": correlation_type,
                    "coefficient": metric[coefficient_name],
                    "ci95_low": ci[0],
                    "ci95_high": ci[1],
                    "p_value": metric["p_value"],
                    "p_value_method": metric["p_value_method"],
                    "p_value_analytic_iid_reference": metric[
                        "p_value_analytic_iid_reference"
                    ],
                    "bootstrap_valid_rounds": metric["bootstrap_valid_rounds"],
                }
            )
    return rows


def _quartile_csv(scopes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope in scopes:
        for group in scope["quartiles"]["groups"]:
            ci = group["utility_ci95_prompt_cluster_bootstrap"] or [None, None]
            rows.append(
                {
                    "branch": scope["branch"],
                    "scope": scope["scope"],
                    "signal": scope["signal"],
                    "utility": scope["utility"],
                    "probe_bin": group["probe_bin"],
                    "step_count": group["step_count"],
                    "unique_step_count": group["unique_step_count"],
                    "trajectory_count": group["trajectory_count"],
                    "prompt_count": group["prompt_count"],
                    "mean_probe_gain": group["mean_probe_gain"],
                    "mean_utility": group["mean_utility"],
                    "utility_ci95_low": ci[0],
                    "utility_ci95_high": ci[1],
                    "bootstrap_valid_rounds": group["bootstrap_valid_rounds"],
                }
            )
    return rows


def _extreme_csv(scopes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope in scopes:
        data = scope["extreme_groups"]
        bottom = data["bottom_20"]
        top = data["top_20"]
        difference = data["top_minus_bottom"]
        bottom_ci = bottom["utility_ci95_prompt_cluster_bootstrap"] or [None, None]
        top_ci = top["utility_ci95_prompt_cluster_bootstrap"] or [None, None]
        difference_ci = difference["ci95_prompt_cluster_bootstrap"] or [None, None]
        rows.append(
            {
                "branch": scope["branch"],
                "scope": scope["scope"],
                "signal": scope["signal"],
                "utility": scope["utility"],
                "bottom_threshold": bottom["threshold_probe_gain_le"],
                "bottom_step_count": bottom["step_count"],
                "bottom_unique_step_count": bottom["unique_step_count"],
                "bottom_trajectory_count": bottom["trajectory_count"],
                "bottom_mean_utility": bottom["mean_utility"],
                "bottom_ci95_low": bottom_ci[0],
                "bottom_ci95_high": bottom_ci[1],
                "top_threshold": top["threshold_probe_gain_ge"],
                "top_step_count": top["step_count"],
                "top_unique_step_count": top["unique_step_count"],
                "top_trajectory_count": top["trajectory_count"],
                "top_mean_utility": top["mean_utility"],
                "top_ci95_low": top_ci[0],
                "top_ci95_high": top_ci[1],
                "top_minus_bottom": difference["difference"],
                "difference_ci95_low": difference_ci[0],
                "difference_ci95_high": difference_ci[1],
                "difference_p_value": difference["p_value"],
                "groups_overlap_due_to_ties": data["groups_overlap_due_to_ties"],
            }
        )
    return rows


def _sign_csv(scopes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope in scopes:
        data = scope["probe_sign"]
        difference = data["positive_minus_nonpositive_mean_utility"]
        difference_ci = difference["ci95_prompt_cluster_bootstrap"] or [None, None]
        for key in ("probe_gain_positive", "probe_gain_nonpositive"):
            group = data[key]
            utility_ci = group["utility_ci95_prompt_cluster_bootstrap"] or [None, None]
            probability_ci = group[
                "P_utility_gt_zero_ci95_prompt_cluster_bootstrap"
            ] or [None, None]
            rows.append(
                {
                    "branch": scope["branch"],
                    "scope": scope["scope"],
                    "signal": scope["signal"],
                    "utility": scope["utility"],
                    "probe_sign_group": key,
                    "step_count": group["step_count"],
                    "unique_step_count": group["unique_step_count"],
                    "trajectory_count": group["trajectory_count"],
                    "mean_utility": group["mean_utility"],
                    "utility_ci95_low": utility_ci[0],
                    "utility_ci95_high": utility_ci[1],
                    "P_utility_gt_zero": group["P_utility_gt_zero"],
                    "P_utility_gt_zero_ci95_low": probability_ci[0],
                    "P_utility_gt_zero_ci95_high": probability_ci[1],
                    "positive_minus_nonpositive_mean_utility": difference["difference"],
                    "difference_ci95_low": difference_ci[0],
                    "difference_ci95_high": difference_ci[1],
                    "difference_p_value": difference["p_value"],
                }
            )
    return rows


def _regression_csv(scopes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope in scopes:
        data = scope["regression_robustness"]
        ci = data["probe_gain_coefficient_ci95_prompt_cluster_bootstrap"] or [None, None]
        rows.append(
            {
                "branch": scope["branch"],
                "scope": scope["scope"],
                "signal": scope["signal"],
                "utility": scope["utility"],
                "formula": data["formula"],
                "n": data["n"],
                "design_rank": data["design_rank"],
                "probe_gain_coefficient": data["probe_gain_coefficient"],
                "coefficient_ci95_low": ci[0],
                "coefficient_ci95_high": ci[1],
                "coefficient_p_value": data["probe_gain_coefficient_p_value"],
                "p_value_method": data["p_value_method"],
                "standard_error_iid_reference": data["standard_error_iid_reference"],
                "p_value_analytic_iid_reference": data[
                    "p_value_analytic_iid_reference"
                ],
                "bootstrap_valid_rounds": data["bootstrap_valid_rounds"],
            }
        )
    return rows


def _positive_finding(correlation: dict[str, Any]) -> bool:
    ci = correlation["ci95_prompt_cluster_bootstrap"]
    return (
        correlation.get("r", correlation.get("rho")) is not None
        and correlation.get("r", correlation.get("rho")) > 0.0
        and ci is not None
        and ci[0] > 0.0
    )


def build_conclusions(scopes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_key = {(scope["branch"], scope["scope"]): scope for scope in scopes}
    correct = by_key[("correct_student", "student")]
    incorrect = by_key[("incorrect_teacher", "pooled")]

    def summary(scope: dict[str, Any]) -> dict[str, Any]:
        global_pearson = scope["global_correlation"]["pearson"]
        global_spearman = scope["global_correlation"]["spearman"]
        within_pearson = scope["within_trajectory_correlation"]["pearson"]
        within_spearman = scope["within_trajectory_correlation"]["spearman"]
        extreme = scope["extreme_groups"]["top_minus_bottom"]
        regression = scope["regression_robustness"]
        return {
            "global_positive_with_ci_above_zero": _positive_finding(global_pearson)
            and _positive_finding(global_spearman),
            "within_positive_with_ci_above_zero": _positive_finding(within_pearson)
            and _positive_finding(within_spearman),
            "quartile_mean_utility_monotonic_nondecreasing": scope["quartiles"][
                "mean_utility_monotonic_nondecreasing"
            ],
            "top_minus_bottom_positive_with_ci_above_zero": (
                extreme["difference"] is not None
                and extreme["difference"] > 0.0
                and extreme["ci95_prompt_cluster_bootstrap"] is not None
                and extreme["ci95_prompt_cluster_bootstrap"][0] > 0.0
            ),
            "controlled_probe_coefficient_positive_with_ci_above_zero": (
                regression["probe_gain_coefficient"] is not None
                and regression["probe_gain_coefficient"] > 0.0
                and regression[
                    "probe_gain_coefficient_ci95_prompt_cluster_bootstrap"
                ]
                is not None
                and regression[
                    "probe_gain_coefficient_ci95_prompt_cluster_bootstrap"
                ][0]
                > 0.0
            ),
        }

    correct_summary = summary(correct)
    incorrect_summary = summary(incorrect)
    return {
        "correct_student": correct_summary,
        "incorrect_teacher_pooled": incorrect_summary,
        "finding_1_correct_supported": correct_summary[
            "global_positive_with_ci_above_zero"
        ]
        and correct_summary["within_positive_with_ci_above_zero"],
        "finding_2_incorrect_supported": incorrect_summary[
            "global_positive_with_ci_above_zero"
        ]
        and incorrect_summary["within_positive_with_ci_above_zero"],
        "finding_3_step_modulation_supported": (
            correct_summary["quartile_mean_utility_monotonic_nondecreasing"]
            and correct_summary["top_minus_bottom_positive_with_ci_above_zero"]
            and incorrect_summary["quartile_mean_utility_monotonic_nondecreasing"]
            and incorrect_summary["top_minus_bottom_positive_with_ci_above_zero"]
        ),
        "note": (
            "These flags are mechanical summaries of the observed estimates and "
            "prompt-cluster bootstrap intervals; they do not force the expected conclusion."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-scores", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-rounds", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260919)
    return parser.parse_args()


def _write_report_csvs(output_dir: Path, long_rows: list[dict[str, Any]], scopes: list[dict[str, Any]]) -> dict[str, str]:
    outputs: dict[str, str] = {}

    def write(name: str, rows: list[dict[str, Any]]) -> None:
        path = output_dir / name
        if not rows:
            raise RuntimeError(f"Refusing to create empty required output {path}.")
        atomic_csv(path, rows, list(rows[0]))
        outputs[name] = str(path)

    path = output_dir / "03_probe_step_data.csv"
    atomic_csv(path, long_rows, STEP_DATA_FIELDS)
    outputs[path.name] = str(path)
    write("03_probe_global_correlations.csv", _global_csv(scopes))
    write("03_probe_within_trajectory_correlations.csv", _within_csv(scopes))
    write("03_probe_quartile_stats.csv", _quartile_csv(scopes))
    write("03_probe_extreme_groups.csv", _extreme_csv(scopes))
    write("03_probe_sign_stats.csv", _sign_csv(scopes))
    write("03_probe_regression_robustness.csv", _regression_csv(scopes))
    return outputs


def main() -> None:
    args = parse_args()
    if args.bootstrap_rounds < 0:
        raise ValueError("--bootstrap-rounds cannot be negative.")
    step_scores_path = args.step_scores.resolve()
    output_dir = args.output_dir.resolve()
    if not step_scores_path.is_file():
        raise FileNotFoundError(step_scores_path)
    raw_steps = load_jsonl(step_scores_path)
    long_rows = flatten_step_teacher_rows(raw_steps)
    teachers = sorted({str(row["teacher"]) for row in long_rows})
    correct_rows = student_scope_rows(long_rows)
    if not correct_rows:
        raise ValueError("No correct-trajectory Student rows were found.")
    pooled_incorrect = teacher_scope_rows(long_rows, None)
    if not pooled_incorrect:
        raise ValueError("No incorrect-trajectory Teacher rows were found.")

    scopes: list[dict[str, Any]] = []
    scope_seed = int(args.seed)
    scopes.append(
        analyze_scope(
            correct_rows,
            branch="correct_student",
            scope="student",
            bootstrap_rounds=int(args.bootstrap_rounds),
            seed=scope_seed,
        )
    )
    scope_seed += 1
    scopes.append(
        analyze_scope(
            pooled_incorrect,
            branch="incorrect_teacher",
            scope="pooled",
            bootstrap_rounds=int(args.bootstrap_rounds),
            seed=scope_seed,
        )
    )
    for teacher in teachers:
        scope_seed += 1
        scopes.append(
            analyze_scope(
                teacher_scope_rows(long_rows, teacher),
                branch="incorrect_teacher",
                scope=f"teacher:{teacher}",
                bootstrap_rounds=int(args.bootstrap_rounds),
                seed=scope_seed,
            )
        )

    outputs = _write_report_csvs(output_dir, long_rows, scopes)
    conclusions = build_conclusions(scopes)
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "script": "tests/analyze_ersr_answer_probes_03.py",
        "input": {
            "step_scores": str(step_scores_path),
            "step_score_records": len(raw_steps),
            "step_teacher_rows": len(long_rows),
            "correct_student_steps": len(correct_rows),
            "incorrect_teacher_interventions": len(pooled_incorrect),
            "teachers": teachers,
            "bootstrap_rounds": int(args.bootstrap_rounds),
            "seed": int(args.seed),
            "reward_counts_by_unique_step": dict(
                Counter(str(int(row["reward"])) for row in raw_steps)
            ),
        },
        "definitions": {
            "P": "exp(mean gold-answer-token logprob) under the Student; geometric-mean token probability",
            "deltaP_student": "P_student_after - P_before",
            "deltaP_teacher": "P_teacher_after - P_before for one Teacher replacement",
            "V_before": "ERSR V(redecide), recomputed from correct_count / mc_k",
            "V_student_after": "ERSR V(keep)",
            "V_teacher_after": "ERSR V(replace:<teacher>)",
            "A_student": "V_student_after - V_before",
            "A_teacher": "V_teacher_after - V_before",
            "step_csv_unit": "one row per (selected ERSR step, Teacher); Student analyses deduplicate by case_id",
            "pooled_teacher_scope": "all (step, Teacher) interventions; prompt bootstrap keeps all Teachers together",
            "relative_position": "(reasoning_step_index + 1) / num_reasoning_steps",
            "primary_inference": (
                "prompt-level cluster bootstrap: a sampled prompt carries all trajectories, "
                "steps, and Teacher interventions"
            ),
            "within_trajectory": (
                "subtract trajectory means before correlation; Teacher pooled scope centers "
                "within each (trajectory, Teacher) unit"
            ),
        },
        "scopes": {
            f"{scope['branch']}|{scope['scope']}": scope for scope in scopes
        },
        "conclusions": conclusions,
        "outputs": outputs,
        "caveats": [
            "Only reasoning steps selected and evaluated by ERSR are available; within-trajectory analysis excludes trajectories represented by a single selected step.",
            "Teacher-pooled observations are not treated as iid for primary inference: prompt-cluster resampling keeps both Teachers and all same-prompt rows together.",
            "Analytic iid p-values are reference diagnostics only; the primary p-values and confidence intervals are prompt-cluster bootstrap results.",
            "Teacher replacements flagged truncated are retained and explicitly marked in 03_probe_step_data.csv so sensitivity filtering remains possible offline.",
        ],
    }
    report_path = output_dir / "03_probe_analysis.json"
    atomic_json(report_path, report)
    outputs[report_path.name] = str(report_path)
    atomic_json(output_dir / "03_probe_output_manifest.json", {"outputs": outputs})
    print(
        json.dumps(
            {
                "event": "03_probe_analysis_complete",
                "step_records": len(raw_steps),
                "step_teacher_rows": len(long_rows),
                "teachers": teachers,
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
