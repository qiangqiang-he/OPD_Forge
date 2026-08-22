"""Summarize the matched 2048-question Cal-OPD lambda coverage runs."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
SUMMARY_DIR = OUTPUT_ROOT / "cal_opd_lambda_coverage_2048_summary"
MODES = ("no_thinking", "thinking")
LAMBDAS = (1, 2, 3)
TRANSITIONS = ("lambda1_to_lambda2", "lambda2_to_lambda3")
EXPECTED_SAMPLES = 2048


def _run_name(mode: str) -> str:
    return f"temp_cal_opd_qwen3_4b_to_1p7b_{mode}_lambda_coverage_2048"


def _load_record(mode: str) -> dict[str, Any]:
    path = OUTPUT_ROOT / _run_name(mode) / "cal_opd_lambda_stats.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    if len(records) != 1:
        raise RuntimeError(f"Expected one record in {path}, got {len(records)}")
    record = records[0]
    if int(record["sample_count"]) != EXPECTED_SAMPLES:
        raise RuntimeError(
            f"Expected {EXPECTED_SAMPLES} samples in {path}, got {record['sample_count']}"
        )
    return record


def _display_token(token: str) -> str:
    if token == "":
        return "<empty>"
    if token.isspace():
        escaped = token.encode("unicode_escape").decode()
        return f"<whitespace:{escaped}>"
    return token.replace("|", "\\|").replace("\n", "\\n").replace("\r", "\\r")


def _aggregate_transition_tokens(
    record: dict[str, Any], transition: str, top_k: int = 30
) -> list[dict[str, Any]]:
    aggregates: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "weighted_abs_base_sum": 0.0,
            "weighted_deviation_sum": 0.0,
            "token_ids": set(),
        }
    )
    transition_count = int(record["transitions"][transition]["count"])
    for token in record["transitions"][transition]["tokens"]:
        key = str(token["normalized_token"])
        count = int(token["count"])
        aggregate = aggregates[key]
        aggregate["count"] += count
        aggregate["weighted_abs_base_sum"] += count * float(
            token["mean_abs_base_advantage"]
        )
        aggregate["weighted_deviation_sum"] += count * float(
            token["mean_teacher_self_deviation"]
        )
        aggregate["token_ids"].add(int(token["token_id"]))

    rows = []
    for token, aggregate in aggregates.items():
        count = int(aggregate["count"])
        rows.append(
            {
                "token": _display_token(token),
                "count": count,
                "fraction_of_transition": (
                    count / transition_count if transition_count else 0.0
                ),
                "token_id_count": len(aggregate["token_ids"]),
                "mean_abs_base_advantage": aggregate["weighted_abs_base_sum"]
                / count,
                "mean_teacher_self_deviation": aggregate[
                    "weighted_deviation_sum"
                ]
                / count,
            }
        )
    return sorted(
        rows,
        key=lambda row: (-int(row["count"]), -float(row["mean_abs_base_advantage"])),
    )[:top_k]


def build_summary() -> dict[str, Any]:
    records = {mode: _load_record(mode) for mode in MODES}
    summary: dict[str, Any] = {"modes": {}}
    for mode, record in records.items():
        base_nonzero = int(record["base_nonzero_token_count"])
        mode_summary: dict[str, Any] = {
            "sample_count": int(record["sample_count"]),
            "valid_token_count": int(record["valid_token_count"]),
            "base_nonzero_token_count": base_nonzero,
            "base_zero_token_count": int(record["valid_token_count"]) - base_nonzero,
            "mean_valid_tokens_per_response": float(
                record["mean_valid_tokens_per_response"]
            ),
            "per_lambda": {
                cal_lambda: {
                    key: value
                    for key, value in metrics.items()
                    if key != "retained_l2_magnitude_ratio"
                }
                for cal_lambda, metrics in record["per_lambda"].items()
            },
            "transitions": {},
        }
        for transition in TRANSITIONS:
            transition_data = record["transitions"][transition]
            count = int(transition_data["count"])
            prior_lambda = 1 if transition == "lambda1_to_lambda2" else 2
            prior_zero = int(
                record["per_lambda"][str(prior_lambda)][
                    "newly_zero_from_base_count"
                ]
            )
            prior_surviving = base_nonzero - prior_zero
            mode_summary["transitions"][transition] = {
                "count": count,
                "mean_count_per_response": float(
                    transition_data["mean_count_per_response"]
                ),
                "fraction_of_base_nonzero": count / base_nonzero,
                "fraction_of_prior_surviving": (
                    count / prior_surviving if prior_surviving else 0.0
                ),
                "top_tokens": _aggregate_transition_tokens(record, transition),
            }
        summary["modes"][mode] = mode_summary
    return summary


def _write_csv(summary: dict[str, Any]) -> None:
    rows = []
    for mode in MODES:
        common = summary["modes"][mode]
        for cal_lambda in LAMBDAS:
            row = common["per_lambda"][str(cal_lambda)]
            rows.append(
                {
                    "mode": mode,
                    "lambda": cal_lambda,
                    "sample_count": common["sample_count"],
                    "valid_token_count": common["valid_token_count"],
                    "base_nonzero_token_count": common["base_nonzero_token_count"],
                    **row,
                }
            )
    path = SUMMARY_DIR / "lambda_coverage.csv"
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(summary: dict[str, Any]) -> None:
    lines = [
        "# Cal-OPD matched lambda coverage (2048 questions × 1 rollout)",
        "",
        "`New zero` excludes tokens whose base advantage was already exactly zero. ",
        "`Total zero` includes both base-zero and calibration-zero tokens. Retained magnitude is the ",
        "L1 ratio of summed absolute advantage, matching the production metric.",
        "",
    ]
    for mode in MODES:
        data = summary["modes"][mode]
        lines.extend(
            [
                f"## {mode}",
                "",
                f"Samples: {data['sample_count']}; valid response tokens: "
                f"{data['valid_token_count']:,}; mean tokens/response: "
                f"{data['mean_valid_tokens_per_response']:.2f}; base-nonzero tokens: "
                f"{data['base_nonzero_token_count']:,}.",
                "",
                "| Lambda | New-zero count | New zero/response | New-zero / base-nonzero | Total-zero / valid | Retained magnitude (L1) |",
                "|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for cal_lambda in LAMBDAS:
            row = data["per_lambda"][str(cal_lambda)]
            lines.append(
                f"| {cal_lambda} | {row['newly_zero_from_base_count']:,} | "
                f"{row['mean_newly_zero_tokens_per_response']:.2f} | "
                f"{row['newly_zero_from_base_ratio']:.2%} | "
                f"{row['zero_token_ratio']:.2%} | "
                f"{row['retained_l1_magnitude_ratio']:.2%} |"
            )

        lines.extend(
            [
                "",
                "| Transition | Additional zero count | Additional zero/response | / base-nonzero | / previously surviving |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for transition in TRANSITIONS:
            row = data["transitions"][transition]
            lines.append(
                f"| {transition} | {row['count']:,} | "
                f"{row['mean_count_per_response']:.2f} | "
                f"{row['fraction_of_base_nonzero']:.2%} | "
                f"{row['fraction_of_prior_surviving']:.2%} |"
            )

        for transition in TRANSITIONS:
            lines.extend(
                [
                    "",
                    f"Top tokens for `{transition}`:",
                    "",
                    "| Token | Count | Share of transition | Mean abs base A | Mean teacher deviation |",
                    "|---|---:|---:|---:|---:|",
                ]
            )
            for token in data["transitions"][transition]["top_tokens"]:
                lines.append(
                    f"| {token['token']} | {token['count']:,} | "
                    f"{token['fraction_of_transition']:.2%} | "
                    f"{token['mean_abs_base_advantage']:.5f} | "
                    f"{token['mean_teacher_self_deviation']:.5f} |"
                )
        lines.append("")

    (SUMMARY_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    summary = build_summary()
    (SUMMARY_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(summary)
    _write_markdown(summary)
    print(SUMMARY_DIR / "summary.md")


if __name__ == "__main__":
    main()
