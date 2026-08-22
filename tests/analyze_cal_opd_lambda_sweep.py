"""Aggregate the six local Cal-OPD lambda-sweep JSONL artifacts."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
SUMMARY_DIR = OUTPUT_ROOT / "cal_opd_lambda_sweep_summary"
MODES = ("no_thinking", "thinking")
LAMBDAS = (1, 2, 3)
TRANSITIONS = ("base_to_lambda1", "lambda1_to_lambda2", "lambda2_to_lambda3")


def _run_name(mode: str, cal_lambda: int) -> str:
    return f"temp_cal_opd_qwen3_4b_to_1p7b_{mode}_lambda{cal_lambda}_5steps"


def _load_records(mode: str, cal_lambda: int) -> list[dict[str, Any]]:
    path = OUTPUT_ROOT / _run_name(mode, cal_lambda) / "cal_opd_lambda_stats.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(records) != 5:
        raise RuntimeError(f"Expected five records in {path}, got {len(records)}")
    if any(float(record["configured_lambda"]) != cal_lambda for record in records):
        raise RuntimeError(f"Configured lambda mismatch in {path}")
    return records


def _summarize_lambda_rows(
    records: list[dict[str, Any]], lambda_key: str
) -> dict[str, float]:
    rows = [record["per_lambda"][lambda_key] for record in records]
    total_base_nonzero = sum(record["base_nonzero_token_count"] for record in records)
    total_newly_zero = sum(row["newly_zero_from_base_count"] for row in rows)
    return {
        "steps": len(records),
        "mean_valid_token_count": mean(
            record["valid_token_count"] for record in records
        ),
        "mean_base_nonzero_token_count": mean(
            record["base_nonzero_token_count"] for record in records
        ),
        "mean_newly_zero_token_count": mean(
            row["newly_zero_from_base_count"] for row in rows
        ),
        "mean_newly_zero_token_ratio": mean(
            row["newly_zero_from_base_ratio"] for row in rows
        ),
        "pooled_newly_zero_token_ratio": (
            total_newly_zero / total_base_nonzero if total_base_nonzero else 0.0
        ),
        "mean_total_zero_token_ratio": mean(row["zero_token_ratio"] for row in rows),
        "mean_retained_l1_magnitude_ratio": mean(
            row["retained_l1_magnitude_ratio"] for row in rows
        ),
        "mean_positive_newly_zero_count": mean(
            row["positive_newly_zero_count"] for row in rows
        ),
        "mean_negative_newly_zero_count": mean(
            row["negative_newly_zero_count"] for row in rows
        ),
    }


def _display_token(token: str) -> str:
    if token == "":
        return "<empty>"
    if token.isspace():
        return f"<whitespace:{token.encode('unicode_escape').decode()}>"
    return token.replace("|", "\\|").replace("\n", "\\n").replace("\r", "\\r")


def _aggregate_transition_tokens(
    records: list[dict[str, Any]], transition: str, top_k: int = 30
) -> list[dict[str, Any]]:
    aggregates: dict[str, dict[str, float | int | str]] = defaultdict(
        lambda: {
            "count": 0,
            "weighted_abs_base_sum": 0.0,
            "weighted_deviation_sum": 0.0,
            "token_ids": set(),
        }
    )
    for record in records:
        for token in record["transitions"][transition]["tokens"]:
            key = str(token["normalized_token"])
            count = int(token["count"])
            aggregate = aggregates[key]
            aggregate["count"] = int(aggregate["count"]) + count
            aggregate["weighted_abs_base_sum"] = float(
                aggregate["weighted_abs_base_sum"]
            ) + count * float(token["mean_abs_base_advantage"])
            aggregate["weighted_deviation_sum"] = float(
                aggregate["weighted_deviation_sum"]
            ) + count * float(token["mean_teacher_self_deviation"])
            aggregate["token_ids"].add(int(token["token_id"]))

    rows = []
    for token, aggregate in aggregates.items():
        count = int(aggregate["count"])
        rows.append(
            {
                "token": _display_token(token),
                "count": count,
                "token_id_count": len(aggregate["token_ids"]),
                "mean_abs_base_advantage": float(
                    aggregate["weighted_abs_base_sum"]
                )
                / count,
                "mean_teacher_self_deviation": float(
                    aggregate["weighted_deviation_sum"]
                )
                / count,
            }
        )
    return sorted(
        rows,
        key=lambda row: (-int(row["count"]), -float(row["mean_abs_base_advantage"])),
    )[:top_k]


def _summarize_transition_counts(
    records: list[dict[str, Any]], transition: str
) -> dict[str, float]:
    counts = [int(record["transitions"][transition]["count"]) for record in records]
    total_base_nonzero = sum(
        int(record["base_nonzero_token_count"]) for record in records
    )
    total_count = sum(counts)
    return {
        "steps": len(records),
        "mean_newly_zero_token_count": mean(counts),
        "total_newly_zero_token_count": total_count,
        "pooled_fraction_of_base_nonzero": (
            total_count / total_base_nonzero if total_base_nonzero else 0.0
        ),
    }


def build_summary() -> dict[str, Any]:
    run_records = {
        (mode, cal_lambda): _load_records(mode, cal_lambda)
        for mode in MODES
        for cal_lambda in LAMBDAS
    }
    actual_runs = {
        mode: {
            str(cal_lambda): _summarize_lambda_rows(
                run_records[(mode, cal_lambda)], str(cal_lambda)
            )
            for cal_lambda in LAMBDAS
        }
        for mode in MODES
    }
    matched_counterfactual = {}
    transition_counts = {}
    transition_tokens = {}
    for mode in MODES:
        all_mode_records = [
            record
            for cal_lambda in LAMBDAS
            for record in run_records[(mode, cal_lambda)]
        ]
        matched_counterfactual[mode] = {
            str(cal_lambda): _summarize_lambda_rows(
                all_mode_records, str(cal_lambda)
            )
            for cal_lambda in LAMBDAS
        }
        transition_counts[mode] = {
            transition: _summarize_transition_counts(all_mode_records, transition)
            for transition in TRANSITIONS
        }
        transition_tokens[mode] = {
            transition: _aggregate_transition_tokens(all_mode_records, transition)
            for transition in TRANSITIONS
        }
    return {
        "actual_runs": actual_runs,
        "matched_counterfactual_15_batches_per_mode": matched_counterfactual,
        "transition_counts_15_batches_per_mode": transition_counts,
        "transition_tokens_15_batches_per_mode": transition_tokens,
    }


def _write_csv(summary: dict[str, Any]) -> None:
    path = SUMMARY_DIR / "actual_run_summary.csv"
    rows = []
    for mode in MODES:
        for cal_lambda in LAMBDAS:
            rows.append(
                {
                    "mode": mode,
                    "lambda": cal_lambda,
                    **summary["actual_runs"][mode][str(cal_lambda)],
                }
            )
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(summary: dict[str, Any]) -> None:
    lines = [
        "# Cal-OPD lambda sweep",
        "",
        "## Actual five-step runs",
        "",
        "| Mode | Lambda | Mean newly-zero tokens/step | Newly-zero ratio | Retained magnitude (L1) |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in MODES:
        for cal_lambda in LAMBDAS:
            row = summary["actual_runs"][mode][str(cal_lambda)]
            lines.append(
                f"| {mode} | {cal_lambda} | {row['mean_newly_zero_token_count']:.1f} "
                f"| {row['pooled_newly_zero_token_ratio']:.2%} "
                f"| {row['mean_retained_l1_magnitude_ratio']:.2%} |"
            )

    lines.extend(["", "## Matched counterfactuals", ""])
    for mode in MODES:
        lines.extend(
            [
                f"### {mode}",
                "",
                "| Lambda | Mean newly-zero tokens/step | Newly-zero ratio | Retained magnitude (L1) |",
                "|---:|---:|---:|---:|",
            ]
        )
        for cal_lambda in LAMBDAS:
            row = summary["matched_counterfactual_15_batches_per_mode"][mode][
                str(cal_lambda)
            ]
            lines.append(
                f"| {cal_lambda} | {row['mean_newly_zero_token_count']:.1f} "
                f"| {row['pooled_newly_zero_token_ratio']:.2%} "
                f"| {row['mean_retained_l1_magnitude_ratio']:.2%} |"
            )

        lines.extend(
            [
                "",
                "Incremental zero transitions on the same realized batches:",
                "",
                "| Transition | Mean additional zero tokens/step | Fraction of original nonzero tokens |",
                "|---|---:|---:|",
            ]
        )
        for transition in ("lambda1_to_lambda2", "lambda2_to_lambda3"):
            row = summary["transition_counts_15_batches_per_mode"][mode][
                transition
            ]
            lines.append(
                f"| {transition} | {row['mean_newly_zero_token_count']:.1f} "
                f"| {row['pooled_fraction_of_base_nonzero']:.2%} |"
            )

        for transition in ("lambda1_to_lambda2", "lambda2_to_lambda3"):
            lines.extend(
                [
                    "",
                    f"Top tokens for `{transition}`:",
                    "",
                    "| Token | Count | Mean abs base advantage | Mean teacher deviation |",
                    "|---|---:|---:|---:|",
                ]
            )
            for token in summary["transition_tokens_15_batches_per_mode"][mode][
                transition
            ][:20]:
                lines.append(
                    f"| {token['token']} | {token['count']} "
                    f"| {token['mean_abs_base_advantage']:.6f} "
                    f"| {token['mean_teacher_self_deviation']:.6f} |"
                )
    (SUMMARY_DIR / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    summary = build_summary()
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    (SUMMARY_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_csv(summary)
    _write_markdown(summary)
    print(SUMMARY_DIR / "summary.md")


if __name__ == "__main__":
    main()
