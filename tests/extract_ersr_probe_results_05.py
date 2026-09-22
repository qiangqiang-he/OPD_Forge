#!/usr/bin/env python3
"""Extract the compact probe-result table for analysis 05 (server side).

Run this once on the machine that holds the heavy
``03_probe_step_scores.jsonl`` (token audits and full probe payloads).  It
writes only the per-(step, Teacher) analysis columns to
``05_probe_step_data.csv`` plus a provenance manifest.  Copy the small CSV to
a local machine and run::

    python tests/analyze_ersr_answer_probes_05.py --step-data 05_probe_step_data.csv

to execute the complete both-probe analysis locally on CPU.  The extractor
performs the same flattening and validation as analysis 03, so the CSV is a
lossless projection of the analysis-relevant fields.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.analyze_ersr_answer_probes_03 import (
    STEP_DATA_FIELDS,
    atomic_csv,
    atomic_json,
    flatten_step_teacher_rows,
    load_jsonl,
)

DEFAULT_STEP_SCORES = (
    REPO_ROOT
    / "outputs/ersr_dapo17k_qwen3_1p7b_mc128/probe_analysis_03/03_probe_step_scores.jsonl"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "outputs/ersr_dapo17k_qwen3_1p7b_mc128/probe_analysis_05/05_probe_step_data.csv"
)
SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-scores", type=Path, default=DEFAULT_STEP_SCORES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    step_scores_path = args.step_scores.resolve()
    output_path = args.output.resolve()
    if not step_scores_path.is_file():
        raise FileNotFoundError(step_scores_path)
    raw_steps = load_jsonl(step_scores_path)
    long_rows = flatten_step_teacher_rows(raw_steps)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_csv(output_path, long_rows, STEP_DATA_FIELDS)
    case_reward = {
        str(row["case_id"]): int(row["reward"]) for row in long_rows
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "script": "tests/extract_ersr_probe_results_05.py",
        "source": str(step_scores_path),
        "source_sha256": sha256_file(step_scores_path),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "columns": list(STEP_DATA_FIELDS),
        "step_score_records": len(raw_steps),
        "rows": len(long_rows),
        "unique_cases": len(case_reward),
        "teachers": sorted({str(row["teacher"]) for row in long_rows}),
        "rows_reward_counts": dict(
            Counter(str(int(row["reward"])) for row in long_rows)
        ),
        "cases_reward_counts": dict(
            Counter(str(value) for value in case_reward.values())
        ),
        "next_step_local": (
            "python tests/analyze_ersr_answer_probes_05.py "
            f"--step-data {output_path.name}"
        ),
    }
    atomic_json(output_path.with_name("05_probe_extraction_manifest.json"), manifest)
    print(
        json.dumps(
            {
                "event": "05_probe_extraction_complete",
                "rows": len(long_rows),
                "unique_cases": len(case_reward),
                "teachers": manifest["teachers"],
                "output": str(output_path),
                "output_sha256": manifest["output_sha256"],
                "next_step_local": manifest["next_step_local"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
