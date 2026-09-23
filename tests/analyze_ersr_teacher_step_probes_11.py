#!/usr/bin/env python3
"""Analyze Teacher-model answer probes around the Student's own step (11).

Input is ``10_teacher_student_step_probes.jsonl`` from
``tests/collect_ersr_teacher_student_step_probes_10.py``.  Every signal in
this pipeline is a function of the Student's own trajectory only: the
Teacher models act as scorers over the Student's step, and no teacher
replacement step exists in any signal, matching the training-time
constraint.

For every branch (correct / incorrect), utility (A_student / A_teacher),
and Teacher scope (pooled / per Teacher) the script reports:

* global and trajectory-centered Pearson/Spearman correlations of
  ``deltaP_T_student_step`` with the utility, with prompt-cluster bootstrap
  intervals;
* ascending quintiles (G1 = lowest probe delta ... G5 = highest) with
  group-mean utilities, adjacent-step differences, the G5-G1 span, and the
  per-group-index trend slope.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.analyze_ersr_answer_probes_03 import (
    _center_within,
    _cluster_indices,
    _nan,
    atomic_csv,
    atomic_json,
    bootstrap_two_sided_p,
    correlation_result,
    pearson,
    spearman,
)

DEFAULT_STEP_PROBES = (
    REPO_ROOT
    / "outputs/ersr_dapo17k_qwen3_1p7b_mc128/teacher_student_step_probes_10/10_teacher_student_step_probes.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs/ersr_dapo17k_qwen3_1p7b_mc128/teacher_student_step_probes_10/analysis_11"
)
SCHEMA_VERSION = 1
NUM_GROUPS = 5
BRANCHES = (("correct", 1), ("incorrect", 0))
UTILITIES = ("A_student", "A_teacher")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_long_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            teachers = record.get("teachers")
            if not isinstance(teachers, dict) or not teachers:
                raise ValueError(f"No teacher block at {path}:{line_number}.")
            trajectory_id = str(record["trajectory_id"])
            for name in sorted(teachers):
                block = teachers[name]
                rows.append(
                    {
                        "case_id": str(record["case_id"]),
                        "prompt_id": str(record["response_idx"]),
                        "trajectory_id": trajectory_id,
                        "teacher": name,
                        "reward": int(record["reward"]),
                        "deltaP_T_student_step": float(block["deltaP_T_student_step"]),
                        "A_student": float(record["student"]["A_student"]),
                        "A_teacher": float(block["A_teacher"]),
                        "centering_unit": f"{trajectory_id}|{name}",
                    }
                )
    if not rows:
        raise ValueError(f"No rows found in {path}.")
    return rows


def analyze_correlation(
    rows: Sequence[dict[str, Any]],
    *,
    signal: str,
    utility: str,
    bootstrap_rounds: int,
    seed: int,
) -> dict[str, Any]:
    x = np.asarray([float(row[signal]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row[utility]) for row in rows], dtype=np.float64)
    clusters = np.asarray([str(row["prompt_id"]) for row in rows], dtype=object)
    units = np.asarray([str(row["centering_unit"]) for row in rows], dtype=object)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError(f"Non-finite value in {signal}/{utility}.")

    cluster_groups = _cluster_indices(clusters)
    selected, centered_x, centered_y = _center_within(x, y, units)
    within_clusters = _cluster_indices(clusters[selected]) if len(selected) else []

    pearson_boot: list[float] = []
    spearman_boot: list[float] = []
    within_pearson_boot: list[float] = []
    within_spearman_boot: list[float] = []
    rng = np.random.default_rng(seed)
    for _ in range(bootstrap_rounds):
        chosen = rng.integers(0, len(cluster_groups), size=len(cluster_groups))
        indices = np.concatenate([cluster_groups[int(index)] for index in chosen])
        r_value = pearson(x[indices], y[indices])
        rho_value = spearman(x[indices], y[indices])
        pearson_boot.append(r_value if r_value is not None else _nan())
        spearman_boot.append(rho_value if rho_value is not None else _nan())
        if within_clusters:
            within_chosen = rng.integers(
                0, len(within_clusters), size=len(within_clusters)
            )
            within_indices = np.concatenate(
                [within_clusters[int(index)] for index in within_chosen]
            )
            wr = pearson(centered_x[within_indices], centered_y[within_indices])
            wrho = spearman(centered_x[within_indices], centered_y[within_indices])
            within_pearson_boot.append(wr if wr is not None else _nan())
            within_spearman_boot.append(wrho if wrho is not None else _nan())

    global_result = correlation_result(x, y, pearson_boot, spearman_boot)
    within_result = correlation_result(
        centered_x, centered_y, within_pearson_boot, within_spearman_boot
    )
    within_result.update(
        {
            "centered_step_observation_count": int(len(selected)),
            "centering_unit": "trajectory_id + Teacher",
        }
    )
    return {
        "signal": signal,
        "utility": utility,
        "n": int(len(rows)),
        "global_correlation": global_result,
        "within_trajectory_correlation": within_result,
    }


def _groups_ascending(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, list[float]]:
    order = np.argsort(x, kind="mergesort")
    membership = np.empty(len(x), dtype=np.int64)
    for group_index, indices in enumerate(np.array_split(order, NUM_GROUPS)):
        membership[indices] = group_index
    means = [float(y[membership == group].mean()) for group in range(NUM_GROUPS)]
    return membership, means


def _trend_slope(group_means: Sequence[float]) -> float:
    index = np.arange(len(group_means), dtype=np.float64)
    slope, _intercept = np.polyfit(index, np.asarray(group_means, dtype=np.float64), 1)
    return float(slope)


def _quantile_ci(values: Sequence[float]) -> list[float] | None:
    finite = np.asarray(
        [value for value in values if np.isfinite(value)], dtype=np.float64
    )
    if len(finite) < 2:
        return None
    return [round(float(value), 6) for value in np.quantile(finite, [0.025, 0.975])]


def analyze_quintiles(
    rows: Sequence[dict[str, Any]],
    *,
    signal: str,
    utility: str,
    bootstrap_rounds: int,
    seed: int,
) -> dict[str, Any]:
    x = np.asarray([float(row[signal]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row[utility]) for row in rows], dtype=np.float64)
    clusters = np.asarray([str(row["prompt_id"]) for row in rows], dtype=object)
    cluster_groups = _cluster_indices(clusters)
    membership, point_means = _groups_ascending(x, y)

    distributions: dict[str, list[float]] = {
        **{f"group_mean_{index}": [] for index in range(NUM_GROUPS)},
        **{f"adjacent_{index}": [] for index in range(NUM_GROUPS - 1)},
        "span": [],
        "slope": [],
    }
    rng = np.random.default_rng(seed)
    for _ in range(bootstrap_rounds):
        chosen = rng.integers(0, len(cluster_groups), size=len(cluster_groups))
        indices = np.concatenate([cluster_groups[int(index)] for index in chosen])
        _, means = _groups_ascending(x[indices], y[indices])
        for group in range(NUM_GROUPS):
            distributions[f"group_mean_{group}"].append(means[group])
        for group in range(NUM_GROUPS - 1):
            distributions[f"adjacent_{group}"].append(means[group + 1] - means[group])
        distributions["span"].append(means[-1] - means[0])
        distributions["slope"].append(_trend_slope(means))

    groups: list[dict[str, Any]] = []
    for group in range(NUM_GROUPS):
        mask = membership == group
        groups.append(
            {
                "group": f"G{group + 1}",
                "note": (
                    "lowest probe delta"
                    if group == 0
                    else "highest probe delta" if group == NUM_GROUPS - 1 else ""
                ),
                "step_count": int(np.count_nonzero(mask)),
                "signal_min": round(float(x[mask].min()), 6),
                "signal_max": round(float(x[mask].max()), 6),
                "mean_signal": round(float(x[mask].mean()), 6),
                "mean_utility": round(point_means[group], 6),
                "utility_ci95_prompt_cluster_bootstrap": _quantile_ci(
                    distributions[f"group_mean_{group}"]
                ),
            }
        )
    adjacent = []
    for group in range(NUM_GROUPS - 1):
        point = point_means[group + 1] - point_means[group]
        adjacent.append(
            {
                "pair": f"G{group + 2}-G{group + 1}",
                "difference": round(point, 6),
                "ci95_prompt_cluster_bootstrap": _quantile_ci(
                    distributions[f"adjacent_{group}"]
                ),
                "p_value": round(
                    bootstrap_two_sided_p(distributions[f"adjacent_{group}"]), 6
                ),
                "significantly_nonzero": bool(
                    _quantile_ci(distributions[f"adjacent_{group}"])[0] > 0.0
                    or _quantile_ci(distributions[f"adjacent_{group}"])[1] < 0.0
                ),
            }
        )
    return {
        "signal": signal,
        "utility": utility,
        "n": int(len(rows)),
        "order": "ascending probe delta; G1 lowest, G5 highest",
        "groups": groups,
        "adjacent_differences": adjacent,
        "span_G5_minus_G1": {
            "difference": round(point_means[-1] - point_means[0], 6),
            "ci95_prompt_cluster_bootstrap": _quantile_ci(distributions["span"]),
            "p_value": round(bootstrap_two_sided_p(distributions["span"]), 6),
        },
        "trend_slope_per_group_index": {
            "slope": round(_trend_slope(point_means), 6),
            "ci95_prompt_cluster_bootstrap": _quantile_ci(distributions["slope"]),
        },
        "monotonic_nondecreasing_ascending": all(
            right >= left for left, right in zip(point_means, point_means[1:])
        ),
    }


def build_scopes(
    long_rows: Sequence[dict[str, Any]],
    *,
    teachers: Sequence[str],
    bootstrap_rounds: int,
    seed: int,
) -> list[dict[str, Any]]:
    scopes: list[dict[str, Any]] = []
    scope_seed = int(seed)
    for branch, reward in BRANCHES:
        branch_rows = [row for row in long_rows if int(row["reward"]) == reward]
        if not branch_rows:
            raise ValueError(f"No rows for the {branch} branch.")
        units: list[tuple[str, str | None]] = [("pooled", None)] + [
            (f"teacher:{teacher}", teacher) for teacher in teachers
        ]
        for utility in UTILITIES:
            for unit_name, teacher_filter in units:
                rows = [
                    row
                    for row in branch_rows
                    if teacher_filter is None or row["teacher"] == teacher_filter
                ]
                if not rows:
                    raise ValueError(f"No rows for {branch}/{utility}/{unit_name}.")
                correlation = analyze_correlation(
                    rows,
                    signal="deltaP_T_student_step",
                    utility=utility,
                    bootstrap_rounds=bootstrap_rounds,
                    seed=scope_seed,
                )
                scope_seed += 1
                quintiles = analyze_quintiles(
                    rows,
                    signal="deltaP_T_student_step",
                    utility=utility,
                    bootstrap_rounds=bootstrap_rounds,
                    seed=scope_seed,
                )
                scope_seed += 1
                scopes.append(
                    {
                        "branch": branch,
                        "unit": unit_name,
                        "correlation": correlation,
                        "quintiles": quintiles,
                    }
                )
    return scopes


def _csv_rows(scopes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scope in scopes:
        correlation = scope["correlation"]
        for level, block_name in (
            ("global", "global_correlation"),
            ("within", "within_trajectory_correlation"),
        ):
            block = correlation[block_name]
            for kind, name in (("pearson", "r"), ("spearman", "rho")):
                metric = block[kind]
                ci = metric["ci95_prompt_cluster_bootstrap"] or [None, None]
                rows.append(
                    {
                        "branch": scope["branch"],
                        "unit": scope["unit"],
                        "record": "correlation",
                        "detail": f"{level}_{kind}",
                        "value": metric[name],
                        "ci95_low": ci[0],
                        "ci95_high": ci[1],
                        "p_value": metric["p_value"],
                    }
                )
        for group in scope["quintiles"]["groups"]:
            ci = group["utility_ci95_prompt_cluster_bootstrap"] or [None, None]
            rows.append(
                {
                    "branch": scope["branch"],
                    "unit": scope["unit"],
                    "record": "quintile",
                    "detail": group["group"],
                    "value": group["mean_utility"],
                    "ci95_low": ci[0],
                    "ci95_high": ci[1],
                    "p_value": None,
                }
            )
        span = scope["quintiles"]["span_G5_minus_G1"]
        rows.append(
            {
                "branch": scope["branch"],
                "unit": scope["unit"],
                "record": "span_G5_minus_G1",
                "detail": "ascending",
                "value": span["difference"],
                "ci95_low": (span["ci95_prompt_cluster_bootstrap"] or [None, None])[0],
                "ci95_high": (span["ci95_prompt_cluster_bootstrap"] or [None, None])[1],
                "p_value": span["p_value"],
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-probes", type=Path, default=DEFAULT_STEP_PROBES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--bootstrap-rounds", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260925)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.bootstrap_rounds < 0:
        raise ValueError("--bootstrap-rounds cannot be negative.")
    step_probes_path = args.step_probes.resolve()
    output_dir = args.output_dir.resolve()
    if not step_probes_path.is_file():
        raise FileNotFoundError(step_probes_path)
    long_rows = load_long_rows(step_probes_path)
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
    headline: dict[str, Any] = {}
    for scope in scopes:
        metric = scope["correlation"]["global_correlation"]["pearson"]
        headline[f"{scope['branch']}|{scope['correlation']['utility']}|{scope['unit']}"] = {
            "pearson_r": metric["r"],
            "ci95": metric["ci95_prompt_cluster_bootstrap"],
            "p_value": metric["p_value"],
            "quintile_span_G5_minus_G1": scope["quintiles"]["span_G5_minus_G1"],
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "11_teacher_step_probe_stats.csv"
    csv_rows = _csv_rows(scopes)
    atomic_csv(csv_path, csv_rows, list(csv_rows[0]))
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "script": "tests/analyze_ersr_teacher_step_probes_11.py",
        "input": {
            "step_probes": str(step_probes_path),
            "step_teacher_rows": len(long_rows),
            "teachers": teachers,
            "correct_cases": len(correct_cases),
            "incorrect_cases": len(incorrect_cases),
            "bootstrap_rounds": int(args.bootstrap_rounds),
            "seed": int(args.seed),
        },
        "definitions": {
            "deltaP_T_student_step": (
                "P_T(gold answer | prompt + pre-step + the Student's own step) minus "
                "P_T(gold answer | prompt + pre-step); the Teacher model scores the "
                "Student's step; no teacher replacement step is involved anywhere"
            ),
            "quintile_order": "ascending probe delta: G1 lowest, G5 highest",
            "utility": "A_student = V(keep)-V(redecide); A_teacher = V(replace)-V(redecide)",
            "cluster": "prompt-level bootstrap keyed by response_idx (the question)",
        },
        "scopes": {
            f"{scope['branch']}|{scope['correlation']['utility']}|{scope['unit']}": scope
            for scope in scopes
        },
        "headline": headline,
        "outputs": {csv_path.name: str(csv_path)},
    }
    report_path = output_dir / "11_teacher_step_probe_analysis.json"
    atomic_json(report_path, report)
    print(
        json.dumps(
            {
                "event": "11_teacher_step_probe_analysis_complete",
                "step_teacher_rows": len(long_rows),
                "scopes": len(scopes),
                "bootstrap_rounds": int(args.bootstrap_rounds),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
