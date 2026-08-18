#!/usr/bin/env python3
"""Analyze Teacher flat/peaked sensitivity from PS-OPD raw token dumps."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch


def rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, matching the standard Spearman definition."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def summarize(rows: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, float | int]:
    sensitivity = rows["sensitivity"][mask]
    result: dict[str, float | int] = {"count": int(mask.sum())}
    if not sensitivity.size:
        return result
    for name in ("teacher_prob", "teacher_max_prob", "sensitivity", "gap"):
        values = rows[name][mask]
        result[f"{name}_mean"] = float(values.mean())
        result[f"{name}_median"] = float(np.median(values))
    result["sensitivity_p90"] = float(np.quantile(sensitivity, 0.9))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stats_dir", type=Path)
    parser.add_argument("--large-gap-fraction", type=float, default=0.25)
    parser.add_argument("--teacher-low-threshold", type=float, default=0.1)
    parser.add_argument("--flat-peak-threshold", type=float, default=0.5)
    args = parser.parse_args()

    files = sorted(args.stats_dir.glob("tokens_*.pt"))
    if not files:
        raise SystemExit(f"No token dumps found in {args.stats_dir}")

    sequences = []
    for path in files:
        sequences.extend(torch.load(path, map_location="cpu", weights_only=True))

    columns = {name: [] for name in ("student_logprob", "teacher_logprob", "teacher_max_logprob", "sensitivity")}
    large_masks = []
    for record in sequences:
        length = int(record["sensitivity"].numel())
        gap = (record["student_logprob"] - record["teacher_logprob"]).abs()
        k = max(1, math.ceil(length * args.large_gap_fraction))
        large = torch.zeros(length, dtype=torch.bool)
        large[torch.topk(gap, k=k, sorted=False).indices] = True
        large_masks.append(large.numpy())
        for name in columns:
            columns[name].append(record[name].float().numpy())

    raw = {name: np.concatenate(parts) for name, parts in columns.items()}
    rows = {
        **raw,
        "student_prob": np.exp(raw["student_logprob"]),
        "teacher_prob": np.exp(raw["teacher_logprob"]),
        "teacher_max_prob": np.exp(raw["teacher_max_logprob"]),
        "gap": raw["student_logprob"] - raw["teacher_logprob"],
    }
    large = np.concatenate(large_masks)
    student_gt_teacher = rows["gap"] > 0
    selected = large & student_gt_teacher
    teacher_low = rows["teacher_prob"] < args.teacher_low_threshold
    core = selected & teacher_low
    flat = rows["teacher_max_prob"] < args.flat_peak_threshold
    peaked = ~flat

    result: dict[str, object] = {
        "definitions": {
            "rollouts": len(sequences),
            "large_gap": f"per-rollout top {args.large_gap_fraction:.0%} by abs(logS-logT)",
            "direction": "student_prob > teacher_prob",
            "teacher_low": f"teacher_prob < {args.teacher_low_threshold}",
            "teacher_flat": f"teacher_max_prob < {args.flat_peak_threshold}",
            "teacher_peaked": f"teacher_max_prob >= {args.flat_peak_threshold}",
        },
        "all_valid_tokens": int(rows["gap"].size),
        "large_gap_student_gt_teacher": summarize(rows, selected),
        "core_teacher_low": summarize(rows, core),
    }
    selected_count = int(selected.sum())
    selected_group_stats = {}
    for name, mask in (("teacher_flat", selected & flat), ("teacher_peaked", selected & peaked)):
        stats = summarize(rows, mask)
        stats["share_of_selected"] = float(mask.sum() / selected_count) if selected_count else float("nan")
        selected_group_stats[name] = stats
    result["flat_vs_peaked_within_all_selected"] = selected_group_stats

    core_count = int(core.sum())
    group_stats = {}
    for name, mask in (("teacher_flat", core & flat), ("teacher_peaked", core & peaked)):
        stats = summarize(rows, mask)
        stats["share_of_core"] = float(mask.sum() / core_count) if core_count else float("nan")
        group_stats[name] = stats
    result["flat_vs_peaked_within_core"] = group_stats

    bins = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.000001]
    bucket_stats = []
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        mask = core & (rows["teacher_max_prob"] >= lower) & (rows["teacher_max_prob"] < upper)
        stats = summarize(rows, mask)
        stats["teacher_max_prob_bucket"] = f"[{lower:.1f}, {min(upper, 1.0):.1f})"
        bucket_stats.append(stats)
    result["teacher_max_probability_buckets"] = bucket_stats

    if core_count >= 2:
        x_ranks = rankdata(rows["teacher_max_prob"][core])
        y_ranks = rankdata(rows["sensitivity"][core])
        rho = np.corrcoef(x_ranks, y_ranks)[0, 1]
        result["core_spearman_teacher_max_vs_sensitivity"] = {
            "rho": float(rho),
        }

    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
