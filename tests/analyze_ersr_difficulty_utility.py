#!/usr/bin/env python3
"""Difficulty-utility analyzer for the ERSR Monte-Carlo evaluator output.

Reads ``ersr_results.json`` written by ``tests/run_ersr_mc.py`` and emits one
self-contained JSON report (default ``difficulty_utility_analysis.json``) with
every number required by the paper's "problem difficulty vs. action utility"
section:

*   Prompt accuracy ``R_x = bucket / G`` with ``G = rollouts_per_prompt``
    (G = 5 for the DAPO-17k Avg5 run, so the levels are ``0/5 ... 5/5``; the
    analysis plan written for G = 8 maps one-to-one onto these levels).
    Difficulty is ``D_x = 1 - R_x``.
*   Trajectory outcome ``R_i = rollout_correct`` of the rollout that
    contributed the step.
*   Student action utility ``A = V(keep) - V(redecide)``.
*   Teacher intervention utility ``A_TI:<teacher> =
    V(replace:<teacher>) - V(redecide)``, reported separately for every
    configured Teacher and as ``A_TI:pooled`` (the per-step arithmetic mean
    over configured Teachers).

Report sections:

  A. ``A_basic_stats_by_accuracy`` — prompts / trajectories / correct /
      incorrect trajectories / steps per accuracy level.
  B. ``B_all_trajectories`` — per-level mean A and per-Teacher A_TI with
      95% CIs, Pearson/Spearman correlations against accuracy, and OLS
      slope + 95% CI + p-value.
  C. ``C_correct_trajectories`` — the same statistics restricted to
      R_i = 1 trajectories (the RL-branch curve ``E[A | R=1]``).
  D. ``D_incorrect_trajectories`` — the same statistics restricted to
      R_i = 0 trajectories (the OPD-branch curve ``E[A_TI | R=0]``).
  E. ``matrix_outcome_x_difficulty`` — the compact 2.4 data matrix with
      mean + CI per (outcome, level, metric) for direct plotting.
  F. ``sensitivity_truncated_excluded`` — every A_TI curve re-estimated after
      dropping steps whose Teacher replacement was flagged truncated.
  G. ``conclusions`` — mechanically generated monotonicity / slope / sign
      statements for the headline curves.

Statistics conventions:

*   ``A`` and ``A_TI`` are recomputed exactly from the integer
    ``correct_count`` / ``mc_k`` values in ``samples`` (mirroring
    ``tests/analyze_ersr_mc.py``), not read back from ``advantages`` floats.
*   Primary CI: step-level ``mean ± t_{0.975,n-1} * sd / sqrt(n)``.
    Robustness CI: per-trajectory step means first, then the same interval
    over trajectories (guards against within-trajectory correlation).
*   scipy is optional: with it, exact t-distribution tails are used for CIs
    and p-values; without it a normal approximation is used and flagged in
    ``input.p_value_method``.

Structurally empty cells: level ``0/G`` has no correct rollouts and level
``G/G`` has no incorrect rollouts, so ``E[A | R=1, R_x=0]`` and
``E[A_TI | R=0, R_x=1]`` are reported as ``null`` with an explanatory note.

Example:

    python tests/analyze_ersr_difficulty_utility.py \
        --results outputs/ersr_dapo17k_qwen3_1p7b_mc128/ersr_results.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Conclusions print human-readable Chinese text; force UTF-8 so hosts with a
# legacy console codepage (e.g. Windows GBK) never mangle the output.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

try:  # Exact t tails when available; a normal approximation otherwise.
    from scipy import stats as _scipy_stats
except Exception:  # pragma: no cover
    _scipy_stats = None


SCHEMA_VERSION = 1
Z_95 = 1.959963984540054  # scipy.stats.norm.ppf(0.975)
DEFAULT_RESULTS = REPO_ROOT / "outputs/ersr_dapo17k_qwen3_1p7b_mc128/ersr_results.json"
SUBSETS: tuple[str, ...] = ("all", "correct", "incorrect")
METRIC_A = "A"
METRIC_A_TI_POOLED = "A_TI:pooled"
TOL = 1e-9


# ---------------------------------------------------------------------------
# Small numeric helpers (pure Python).
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def r6(value: Any) -> Any:
    """Round float leaves of a report tree; keep None and pass ints/bools."""

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
    return math.sqrt(
        math.fsum((value - mean) ** 2 for value in values) / (len(values) - 1)
    )


def _t_critical(df: int) -> float:
    if _scipy_stats is not None:
        return float(_scipy_stats.t.ppf(0.975, df))
    return Z_95  # normal approximation; flagged in the report metadata


def _two_sided_p(t_stat: float, df: int) -> float:
    if _scipy_stats is not None:
        return float(_scipy_stats.t.sf(abs(t_stat), df) * 2.0)
    return math.erfc(abs(t_stat) / math.sqrt(2.0))


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


def _corr_p_value(r: float | None, n: int) -> float | None:
    if r is None or n < 3 or abs(r) >= 1.0:
        return None
    t_stat = r * math.sqrt((n - 2) / (1.0 - r * r))
    return _two_sided_p(t_stat, n - 2)


def mean_ci_stats(values: Sequence[float]) -> dict[str, Any]:
    """Step-level mean with a t (or normal) 95% CI."""

    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "sample_sd": None, "standard_error": None, "ci95": None}
    mean = _mean(values)
    if n == 1:
        return {
            "n": 1,
            "mean": r6(mean),
            "sample_sd": None,
            "standard_error": None,
            "ci95": None,
        }
    sd = _sample_sd(values)
    se = sd / math.sqrt(n)
    critical = _t_critical(n - 1)
    ci = [mean - critical * se, mean + critical * se]
    return {
        "n": n,
        "mean": r6(mean),
        "sample_sd": r6(sd),
        "standard_error": r6(se),
        "ci95": [r6(ci[0]), r6(ci[1])],
    }


def correlation_stats(xs: Sequence[float], ys: Sequence[float]) -> dict[str, Any]:
    r = pearson(xs, ys)
    rho = spearman(xs, ys)
    return {
        "n": len(xs),
        "pearson_r": r6(r),
        "pearson_p": r6(_corr_p_value(r, len(xs))),
        "spearman_rho": r6(rho),
        "spearman_p": r6(_corr_p_value(rho, len(xs))),
    }


def ols_stats(xs: Sequence[float], ys: Sequence[float]) -> dict[str, Any]:
    """OLS y = intercept + slope * x with slope SE / 95% CI / two-sided p."""

    n = len(xs)
    empty = {
        "n": n,
        "slope": None,
        "intercept": None,
        "slope_se": None,
        "slope_ci95": None,
        "slope_p_value": None,
        "r_squared": None,
    }
    if n < 3 or len(ys) != n:
        return empty
    mx = _mean(xs)
    my = _mean(ys)
    sxx = math.fsum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return empty
    sxy = math.fsum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx
    intercept = my - slope * mx
    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    dof = n - 2
    sse = math.fsum(value * value for value in residuals)
    mse = sse / dof
    if mse <= 0.0:
        se = 0.0
        p_value = 0.0 if slope != 0.0 else 1.0
        ci = [slope, slope]
    else:
        se = math.sqrt(mse / sxx)
        t_stat = slope / se
        p_value = _two_sided_p(t_stat, dof)
        critical = _t_critical(dof)
        ci = [slope - critical * se, slope + critical * se]
    sst = math.fsum((y - my) ** 2 for y in ys)
    r_squared = 1.0 - sse / sst if sst > 0.0 else None
    return {
        "n": n,
        "slope": r6(slope),
        "intercept": r6(intercept),
        "slope_se": r6(se),
        "slope_ci95": [r6(ci[0]), r6(ci[1])],
        "slope_p_value": r6(p_value),
        "r_squared": r6(r_squared),
    }


# ---------------------------------------------------------------------------
# Loading ersr_results.json into per-step rows.
# ---------------------------------------------------------------------------


def _required_sample(samples: dict[str, Any], case_id: str, arm: str) -> dict[str, Any]:
    sample = samples.get(arm)
    if not isinstance(sample, dict) or "correct_count" not in sample or "mc_k" not in sample:
        raise ValueError(f"Case {case_id!r} is missing MC samples for arm {arm!r}.")
    return sample


def metric_names(teachers: Sequence[str]) -> list[str]:
    return [METRIC_A, METRIC_A_TI_POOLED] + [f"A_TI:{name}" for name in teachers]


def metric_value(row: dict[str, Any], metric: str) -> float:
    if metric == METRIC_A:
        return row["A"]
    if metric == METRIC_A_TI_POOLED:
        # Every case has the same configured Teacher set.  Averaging within a
        # step gives one A_TI observation per selected step, avoids pretending
        # the two Teacher values are independent, and has the same grand mean
        # as equal-weight pooling of all (step, Teacher) pairs.
        return _mean(list(row["A_TI"].values()))
    return row["A_TI"][metric[len("A_TI:") :]]


def extract_rows(
    results: Sequence[dict[str, Any]], rollouts_per_prompt: int
) -> tuple[list[dict[str, Any]], list[str]]:
    """Flatten every record into one row keyed by trajectory and bucket.

    ``A`` and every ``A_TI`` are exact fractions of ``mc_k`` recomputed from
    the integer ``correct_count`` values, mirroring analyze_ersr_mc.py.
    """

    teachers: list[str] | None = None
    rows: list[dict[str, Any]] = []
    for record_index, record in enumerate(results):
        case_id = str(record["case_id"])
        samples = record.get("samples") or {}
        redecide = _required_sample(samples, case_id, "redecide")
        keep = _required_sample(samples, case_id, "keep")
        mc_k = int(redecide["mc_k"])
        if int(keep["mc_k"]) != mc_k:
            raise ValueError(f"Case {case_id!r} has inconsistent mc_k across arms.")
        case_teachers = sorted(
            key[len("replace:") :] for key in samples if key.startswith("replace:")
        )
        if not case_teachers:
            raise ValueError(f"Case {case_id!r} has no replace:<teacher> arm.")
        if teachers is None:
            teachers = case_teachers
        elif set(teachers) != set(case_teachers):
            raise ValueError(
                f"Case {case_id!r} teacher set {case_teachers} differs from {teachers}."
            )
        bucket = int(record["bucket"])
        outcome = int(record["rollout_correct"])
        if not 0 <= bucket <= rollouts_per_prompt:
            raise ValueError(
                f"Case {case_id!r} bucket={bucket} outside 0..{rollouts_per_prompt}; "
                "pass --rollouts-per_prompt accordingly."
            )
        if outcome not in (0, 1):
            raise ValueError(f"Case {case_id!r} rollout_correct={outcome} is not 0/1.")
        rd = int(redecide["correct_count"])
        kp = int(keep["correct_count"])
        replacements = record.get("teacher_replacements") or {}
        a_ti: dict[str, float] = {}
        truncated: dict[str, int] = {}
        for name in case_teachers:
            sample = _required_sample(samples, case_id, f"replace:{name}")
            if int(sample["mc_k"]) != mc_k:
                raise ValueError(
                    f"Case {case_id!r} teacher {name!r} has inconsistent mc_k."
                )
            a_ti[name] = (int(sample["correct_count"]) - rd) / mc_k
            truncated[name] = int((replacements.get(name) or {}).get("truncated", 0))
        rows.append(
            {
                "case_id": case_id,
                "case_index": int(record["case_index"]),
                "record_index": record_index,
                "response_idx": int(record["response_idx"]),
                "rollout_index": int(record["rollout_index"]),
                "trajectory": (int(record["response_idx"]), int(record["rollout_index"])),
                "bucket": bucket,
                "R": outcome,
                "mc_k": mc_k,
                "A": (kp - rd) / mc_k,
                "A_TI": a_ti,
                "truncated": truncated,
            }
        )
    if teachers is None:
        raise ValueError("No records found.")

    # A prompt and a trajectory must not change difficulty/outcome merely
    # because more than one selected step came from it.
    prompt_buckets: dict[int, int] = {}
    trajectory_metadata: dict[tuple[int, int], tuple[int, int]] = {}
    case_ids: set[str] = set()
    for row in rows:
        if row["case_id"] in case_ids:
            raise ValueError(f"Duplicate case_id {row['case_id']!r} in results.")
        case_ids.add(row["case_id"])
        previous_bucket = prompt_buckets.setdefault(row["response_idx"], row["bucket"])
        if previous_bucket != row["bucket"]:
            raise ValueError(
                f"Prompt {row['response_idx']} appears in buckets "
                f"{previous_bucket} and {row['bucket']}."
            )
        metadata = (row["bucket"], row["R"])
        previous_metadata = trajectory_metadata.setdefault(row["trajectory"], metadata)
        if previous_metadata != metadata:
            raise ValueError(
                f"Trajectory {row['trajectory']} has inconsistent bucket/outcome: "
                f"{previous_metadata} vs {metadata}."
            )
    return rows, teachers


def subset_predicate(subset: str) -> Callable[[dict[str, Any]], bool] | None:
    if subset == "all":
        return None
    target = 1 if subset == "correct" else 0
    return lambda row: row["R"] == target


def level_key(bucket: int, rollouts_per_prompt: int) -> str:
    return f"{bucket}/{rollouts_per_prompt}"


# ---------------------------------------------------------------------------
# Report sections.
# ---------------------------------------------------------------------------


def group_metric_stats(rows: Sequence[dict[str, Any]], metric: str) -> dict[str, Any]:
    """Step-level CI plus the trajectory-aggregated robustness CI."""

    values = [metric_value(row, metric) for row in rows]
    step = mean_ci_stats(values)
    by_trajectory: dict[tuple[int, int], list[float]] = defaultdict(list)
    for row, value in zip(rows, values):
        by_trajectory[row["trajectory"]].append(value)
    trajectory_values = [_mean(group) for group in by_trajectory.values()]
    trajectory = mean_ci_stats(trajectory_values)
    return {
        "n_steps": step["n"],
        "mean": step["mean"],
        "ci95": step["ci95"],
        "sample_sd": step["sample_sd"],
        "standard_error": step["standard_error"],
        "ci95_step_level": step["ci95"],
        "n_trajectories": trajectory["n"],
        "mean_trajectory_level": trajectory["mean"],
        "ci95_trajectory_level": trajectory["ci95"],
    }


def subset_section(
    rows: Sequence[dict[str, Any]],
    subset: str,
    metrics: Sequence[str],
    rollouts_per_prompt: int,
) -> dict[str, Any]:
    predicate = subset_predicate(subset)
    subset_rows = [row for row in rows if predicate(row)] if predicate else list(rows)
    groups: dict[str, Any] = {}
    for bucket in range(rollouts_per_prompt + 1):
        bucket_rows = [row for row in subset_rows if row["bucket"] == bucket]
        entry: dict[str, Any] = {
            "accuracy": r6(bucket / rollouts_per_prompt),
            "difficulty": r6(1.0 - bucket / rollouts_per_prompt),
            "bucket": bucket,
            "prompt_count": len({row["response_idx"] for row in bucket_rows}),
            "trajectory_count": len({row["trajectory"] for row in bucket_rows}),
            "response_count": len({row["trajectory"] for row in bucket_rows}),
            "step_count": len(bucket_rows),
        }
        for metric in metrics:
            entry[metric] = group_metric_stats(bucket_rows, metric)
        groups[level_key(bucket, rollouts_per_prompt)] = entry
    correlations: dict[str, Any] = {}
    regressions: dict[str, Any] = {}
    xs = [row["bucket"] / rollouts_per_prompt for row in subset_rows]
    for metric in metrics:
        ys = [metric_value(row, metric) for row in subset_rows]
        correlations[metric] = correlation_stats(xs, ys)
        regressions[metric] = ols_stats(xs, ys)
    return {
        "subset": subset,
        "subset_definition": (
            "every selected step"
            if subset == "all"
            else f"steps from trajectories with rollout_correct={1 if subset == 'correct' else 0}"
        ),
        "n_steps": len(subset_rows),
        "n_trajectories": len({row["trajectory"] for row in subset_rows}),
        "n_responses": len({row["trajectory"] for row in subset_rows}),
        "n_prompts": len({row["response_idx"] for row in subset_rows}),
        "groups": groups,
        "correlations": correlations,
        "regressions": regressions,
    }


def basic_stats_section(
    rows: Sequence[dict[str, Any]], rollouts_per_prompt: int
) -> dict[str, Any]:
    section: dict[str, Any] = {}
    for bucket in range(rollouts_per_prompt + 1):
        bucket_rows = [row for row in rows if row["bucket"] == bucket]
        outcome_by_trajectory: dict[tuple[int, int], int] = {}
        for row in bucket_rows:
            outcome_by_trajectory.setdefault(row["trajectory"], row["R"])
        correct = sum(1 for value in outcome_by_trajectory.values() if value == 1)
        section[level_key(bucket, rollouts_per_prompt)] = {
            "accuracy": r6(bucket / rollouts_per_prompt),
            "difficulty": r6(1.0 - bucket / rollouts_per_prompt),
            "bucket": bucket,
            "prompt_count": len({row["response_idx"] for row in bucket_rows}),
            "trajectory_count": len(outcome_by_trajectory),
            "response_count": len(outcome_by_trajectory),
            "correct_trajectory_count": correct,
            "correct_response_count": correct,
            "incorrect_trajectory_count": len(outcome_by_trajectory) - correct,
            "incorrect_response_count": len(outcome_by_trajectory) - correct,
            "step_count": len(bucket_rows),
        }
    return section


def matrix_section(
    sections: dict[str, dict[str, Any]],
    metrics: Sequence[str],
    rollouts_per_prompt: int,
) -> dict[str, Any]:
    """Compact (outcome x level x metric) matrix of means with plotting CIs."""

    levels = [
        {
            "key": level_key(bucket, rollouts_per_prompt),
            "bucket": bucket,
            "accuracy": r6(bucket / rollouts_per_prompt),
            "difficulty": r6(1.0 - bucket / rollouts_per_prompt),
        }
        for bucket in range(rollouts_per_prompt + 1)
    ]
    matrix: dict[str, Any] = {"levels": levels}
    for subset in ("correct", "incorrect"):
        cells: dict[str, Any] = {}
        for level in levels:
            group = sections[subset]["groups"][level["key"]]
            cells[level["key"]] = {
                metric: {
                    "mean": group[metric]["mean"],
                    "ci95": group[metric]["ci95"],
                    "ci95_step_level": group[metric]["ci95_step_level"],
                    "n_steps": group[metric]["n_steps"],
                }
                for metric in metrics
            }
        matrix[subset] = cells
    return matrix


def sensitivity_section(
    rows: Sequence[dict[str, Any]],
    teachers: Sequence[str],
    rollouts_per_prompt: int,
) -> dict[str, Any]:
    """Per-Teacher A_TI re-estimated without truncated replacements."""

    per_teacher: dict[str, Any] = {}
    for name in teachers:
        metric = f"A_TI:{name}"
        kept = [row for row in rows if row["truncated"].get(name, 0) == 0]
        per_teacher[name] = {
            "metric": metric,
            "excluded_steps": len(rows) - len(kept),
            "kept_steps": len(kept),
            "all": subset_section(kept, "all", [metric], rollouts_per_prompt),
            "correct": subset_section(kept, "correct", [metric], rollouts_per_prompt),
            "incorrect": subset_section(kept, "incorrect", [metric], rollouts_per_prompt),
        }
    return {
        "definition": (
            "for every Teacher, every A_TI statistic re-estimated after dropping "
            "steps whose teacher_replacements.<name>.truncated flag is 1; A is "
            "unaffected by Teacher truncation and is not repeated here"
        ),
        "teachers": per_teacher,
    }


# ---------------------------------------------------------------------------
# Conclusions.
# ---------------------------------------------------------------------------


def _fmt(value: Any, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _fmt_p(value: Any) -> str:
    if value is None:
        return "n/a"
    return "<1e-4" if value < 1e-4 else f"{value:.3g}"


def curve_summary(
    section: dict[str, Any], metric: str, rollouts_per_prompt: int
) -> dict[str, Any]:
    groups = section["groups"]
    levels = [level_key(bucket, rollouts_per_prompt) for bucket in range(rollouts_per_prompt + 1)]
    means = [groups[level][metric]["mean"] for level in levels]
    present = [value for value in means if value is not None]
    regression = section["regressions"][metric]
    correlation = section["correlations"][metric]
    slope = regression["slope"]
    p_value = regression["slope_p_value"]
    return {
        "group_means": dict(zip(levels, means)),
        "n_nonempty_groups": len(present),
        "monotonic_nonincreasing_mean": (
            all(b <= a + TOL for a, b in zip(present, present[1:]))
            if len(present) >= 2
            else None
        ),
        "monotonic_nondecreasing_mean": (
            all(b >= a - TOL for a, b in zip(present, present[1:]))
            if len(present) >= 2
            else None
        ),
        "mean_change_last_minus_first": r6(present[-1] - present[0]) if len(present) >= 2 else None,
        "ols_slope": slope,
        "ols_slope_ci95": regression["slope_ci95"],
        "ols_slope_p_value": p_value,
        "slope_negative": slope is not None and slope < 0.0,
        "significant_negative_slope_at_95pct": (
            slope is not None and slope < 0.0 and p_value is not None and p_value < 0.05
        ),
        "pearson_r": correlation["pearson_r"],
        "spearman_rho": correlation["spearman_rho"],
    }


def build_conclusions(
    sections: dict[str, dict[str, Any]],
    teachers: Sequence[str],
    rollouts_per_prompt: int,
) -> dict[str, Any]:
    curves: dict[str, Any] = {}
    key_curves: list[tuple[str, str]] = [
        ("all", METRIC_A),
        ("all", METRIC_A_TI_POOLED),
    ]
    key_curves += [("all", f"A_TI:{name}") for name in teachers]
    key_curves.append(("correct", METRIC_A))
    key_curves.append(("incorrect", METRIC_A_TI_POOLED))
    key_curves += [("incorrect", f"A_TI:{name}") for name in teachers]
    for subset, metric in key_curves:
        curves[f"{subset}|{metric}"] = curve_summary(
            sections[subset], metric, rollouts_per_prompt
        )

    findings: list[str] = []

    def _trend_text(summary: dict[str, Any]) -> str:
        verdict = (
            "随 accuracy 提高显著下降"
            if summary["significant_negative_slope_at_95pct"]
            else "斜率为负但不显著"
            if summary["slope_negative"]
            else "未见下降"
        )
        ci = summary["ols_slope_ci95"]
        ci_text = "n/a" if ci is None else f"[{_fmt(ci[0])}, {_fmt(ci[1])}]"
        return (
            f"slope={_fmt(summary['ols_slope'])} (95% CI {ci_text}, "
            f"p={_fmt_p(summary['ols_slope_p_value'])}), Pearson={_fmt(summary['pearson_r'])}, "
            f"Spearman={_fmt(summary['spearman_rho'])} → {verdict}"
        )

    layer_one = [
        f"E[A]={_trend_text(curves['all|A'])}",
        f"E[A_TI(pooled)]={_trend_text(curves['all|A_TI:pooled'])}",
    ]
    for name in teachers:
        layer_one.append(f"E[A_TI({name})]={_trend_text(curves[f'all|A_TI:{name}'])}")
    findings.append(
        "第一层（全部轨迹）: " + "; ".join(layer_one)
        + "。即随问题变容易，student/teacher 动作效用是否整体下降。"
    )
    findings.append(
        "第二层（RL branch）: 正确轨迹 E[A|R=1]: "
        + _trend_text(curves["correct|A"])
        + "。困难 prompt 上成功轨迹的 self-reinforcement 价值是否更高。"
    )
    layer_opd = [
        f"E[A_TI(pooled)|R=0]: {_trend_text(curves['incorrect|A_TI:pooled'])}"
    ] + [
        f"E[A_TI({name})|R=0]: {_trend_text(curves[f'incorrect|A_TI:{name}'])}"
        for name in teachers
    ]
    findings.append(
        "第二层（OPD branch）: 错误轨迹 " + "; ".join(layer_opd)
        + "。困难 prompt 上失败轨迹的 teacher-intervention 收益是否更高。"
    )
    notes = [
        "格子 0/G × R=1 与 G/G × R=0 结构性为空（bucket=0 的 G 条 rollout 全错、"
        "bucket=G 的全对），对应 mean 为 null。",
        "分组均值的 95% CI 同时给出 step 层（主口径）与轨迹层（稳健口径）两个版本；"
        "同一轨迹的多个 step 相关，step 层 CI 偏窄。",
        "每个 step 的 A / A_TI 本身是 mc_k 次采样的蒙特卡洛估计，单点噪声约 "
        "±2·sqrt(2/mc_k)，分组均值会平均掉该噪声。",
    ]
    return {"curves": curves, "key_findings": findings, "notes": notes}


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS,
        help="ersr_results.json written by tests/run_ersr_mc.py "
        "(a directory is also accepted; ersr_results.json is looked up inside).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output JSON path (default: <results dir>/difficulty_utility_analysis.json).",
    )
    parser.add_argument(
        "--rollouts-per-prompt",
        type=int,
        default=5,
        help="G: rollouts per prompt in the evaluator selection (5 for the Avg5 run).",
    )
    return parser.parse_args()


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
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def _print_table(
    sections: dict[str, dict[str, Any]],
    teachers: Sequence[str],
    rollouts_per_prompt: int,
) -> None:
    columns = [
        ("correct", METRIC_A, "E[A|R=1]"),
        ("incorrect", METRIC_A_TI_POOLED, "E[A_TI:pooled|R=0]"),
    ]
    columns += [("incorrect", f"A_TI:{name}", f"E[A_TI:{name}|R=0]") for name in teachers]
    header = ["accuracy"] + [label for _, _, label in columns]
    lines = ["  ".join(header)]
    for bucket in range(rollouts_per_prompt + 1):
        key = level_key(bucket, rollouts_per_prompt)
        cells = [key]
        for subset, metric, _ in columns:
            stats = sections[subset]["groups"][key][metric]
            cells.append("n/a" if stats["mean"] is None else f"{stats['mean']:.4f} (n={stats['n_steps']})")
        lines.append("  ".join(cells))
    print("[headline table: step-level means]", flush=True)
    for line in lines:
        print(line, flush=True)


def main() -> None:
    args = parse_args()
    rollouts_per_prompt = int(args.rollouts_per_prompt)
    if rollouts_per_prompt <= 0:
        raise ValueError("--rollouts-per-prompt must be positive.")
    results_path = args.results
    if results_path.is_dir():
        results_path = results_path / "ersr_results.json"
    if not results_path.is_file():
        raise FileNotFoundError(results_path)
    output_path = args.output or results_path.parent / "difficulty_utility_analysis.json"

    with results_path.open(encoding="utf-8") as handle:
        results = json.load(handle)
    if not isinstance(results, list) or not results:
        raise ValueError(f"Expected a non-empty JSON array in {results_path}.")

    rows, teachers = extract_rows(results, rollouts_per_prompt)
    metrics = metric_names(teachers)

    sections = {
        "all": subset_section(rows, "all", metrics, rollouts_per_prompt),
        "correct": subset_section(rows, "correct", metrics, rollouts_per_prompt),
        "incorrect": subset_section(rows, "incorrect", metrics, rollouts_per_prompt),
    }
    basic = basic_stats_section(rows, rollouts_per_prompt)
    matrix = matrix_section(sections, metrics, rollouts_per_prompt)
    sensitivity = sensitivity_section(rows, teachers, rollouts_per_prompt)
    conclusions = build_conclusions(sections, teachers, rollouts_per_prompt)

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "script": "tests/analyze_ersr_difficulty_utility.py",
        "input": {
            "results_path": str(results_path),
            "n_steps": len(rows),
            "n_trajectories": len({row["trajectory"] for row in rows}),
            "n_responses": len({row["trajectory"] for row in rows}),
            "n_prompts": len({row["response_idx"] for row in rows}),
            "teachers": teachers,
            "metrics": metrics,
            "rollouts_per_prompt_G": rollouts_per_prompt,
            "accuracy_levels": [
                level_key(bucket, rollouts_per_prompt)
                for bucket in range(rollouts_per_prompt + 1)
            ],
            "steps_by_bucket": {
                level_key(bucket, rollouts_per_prompt): sum(
                    1 for row in rows if row["bucket"] == bucket
                )
                for bucket in range(rollouts_per_prompt + 1)
            },
            "mc_k_distribution": {
                str(key): value
                for key, value in sorted(Counter(row["mc_k"] for row in rows).items())
            },
            "truncated_replacements_by_teacher": {
                name: sum(1 for row in rows if row["truncated"].get(name, 0))
                for name in teachers
            },
            "p_value_method": (
                "scipy t distribution (exact)"
                if _scipy_stats is not None
                else "normal approximation (scipy unavailable)"
            ),
        },
        "definitions": {
            "R_x_bar": "bucket / G — mean accuracy of the prompt's G rollouts; difficulty D_x = 1 - R_x_bar",
            "R_i": "rollout_correct of the trajectory that contributed the step",
            "A": "V(keep) - V(redecide) — student action utility, recomputed exactly from integer correct counts / mc_k",
            "A_TI": "V(replace:<teacher>) - V(redecide) — teacher intervention utility, one metric per Teacher",
            "A_TI_pooled": "per-step arithmetic mean of A_TI over configured Teachers; because every step has every Teacher, its grand mean equals equal-weight pooling of all (step, Teacher) pairs",
            "V_arm": "correct_count / mc_k from the Student MC phase",
            "unit": "one row = one selected reasoning step (case); response_count and trajectory_count are aliases for unique (response_idx, rollout_index), and one trajectory may contribute several steps",
            "ci95_step_level": "mean ± t_{0.975,n-1} * sample_sd / sqrt(n) over steps (primary)",
            "ci95_trajectory_level": "per-trajectory step means first, then the same interval over trajectories (robustness; steps of one trajectory are correlated)",
            "correlations": "Pearson and Spearman between R_x_bar and the utility over steps of the subset",
            "regressions": "OLS utility = intercept + slope * R_x_bar; slope CI/p from t (or normal) tails",
            "empty_cells": "level 0/G has no correct rollouts and level G/G has no incorrect rollouts by construction; those cells are null",
        },
        "A_basic_stats_by_accuracy": basic,
        "B_all_trajectories": sections["all"],
        "C_correct_trajectories": sections["correct"],
        "D_incorrect_trajectories": sections["incorrect"],
        "matrix_outcome_x_difficulty": matrix,
        "sensitivity_truncated_excluded": sensitivity,
        "conclusions": conclusions,
        "caveats": [
            "Every V is a mc_k-sample Monte-Carlo estimate, so A and A_TI carry sampling noise; "
            "the CIs quantify it but per-step point values should not be over-interpreted.",
            "E[A] and every A_TI metric are per selected step. A_TI:pooled first averages "
            "the configured Teachers within the same step; per-Teacher metrics are also retained.",
            "Step counts per level follow the evaluator's selection quotas "
            "(correct:wrong ≈ bucket:(G-bucket)), so cell sizes are designed, not random.",
            "The normal-approximation tails (when scipy is absent) are indistinguishable from "
            "exact t tails at these sample sizes but are flagged in input.p_value_method.",
            "Teacher replacements flagged truncated stay in the main analysis; see "
            "sensitivity_truncated_excluded for the same A_TI curves without them.",
            "The analysis plan was written for G=8 rollouts per prompt; this run used G=5, "
            "so levels are 0/5..5/5 and every statement maps one-to-one.",
        ],
    }

    atomic_json(output_path, report)

    print(
        json.dumps(
            {
                "event": "analysis_complete",
                "results_path": str(results_path),
                "output_path": str(output_path),
                "n_steps": len(rows),
                "n_prompts": len({row["response_idx"] for row in rows}),
                "teachers": teachers,
                "p_value_method": report["input"]["p_value_method"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    for finding in conclusions["key_findings"]:
        print(f"[finding] {finding}", flush=True)
    for note in conclusions["notes"]:
        print(f"[note] {note}", flush=True)
    _print_table(sections, teachers, rollouts_per_prompt)


if __name__ == "__main__":
    main()
