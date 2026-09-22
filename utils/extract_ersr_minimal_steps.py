#!/usr/bin/env python3
"""Extract a lightweight per-step table from a (multi-GB) ersr_results.json.

``tests/run_ersr_mc.py`` writes ``ersr_results.json`` as one JSON array whose
records carry full token IDs, rewards, and texts.  This script streams that
array without loading it into memory and emits one compact JSONL file with
only the fields needed by every downstream difficulty/utility analysis:

    line 1 (header):  {"record_type": "header", "teachers": [...], ...}
    line k (data):    [case_id, bucket, rollout_correct, position_fraction,
                       mc_k, A, A_TI_teacher1, A_TI_teacher2, ...]

``A`` and every ``A_TI`` are exact fractions of ``mc_k`` recomputed from the
integer ``correct_count`` values (never rounded — 1/128 multiples are exact
in binary floating point, so 0.01-threshold banding stays exact).  Teacher
columns follow the sorted teacher names recorded in the header.

Example:

    python utils/extract_ersr_minimal_steps.py \
        --results outputs/ersr_dapo17k_qwen3_1p7b_mc128/ersr_results.json \
        --output eser_analysis/a5_minimal_steps.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

SCHEMA_VERSION = 1
DEFAULT_RESULTS = REPO_ROOT / "outputs/ersr_dapo17k_qwen3_1p7b_mc128/ersr_results.json"
WHITESPACE = " \t\r\n"


def iter_json_array(path: Path, chunk_size: int = 1 << 20) -> Iterator[Any]:
    """Stream the elements of a top-level JSON array without loading it all.

    Uses ``json.JSONDecoder.raw_decode`` over a sliding text buffer, so memory
    is bounded by the largest single record plus one chunk.
    """

    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        started = False
        eof = False
        while True:
            if not eof:
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer = buffer[position:] + chunk if position else buffer + chunk
                    position = 0
                else:
                    eof = True
            progressed = True
            while progressed:
                progressed = False
                while position < len(buffer) and buffer[position] in WHITESPACE:
                    position += 1
                    progressed = True
                if position >= len(buffer):
                    if eof:
                        if started:
                            raise ValueError("Unexpected end of file inside the JSON array.")
                        raise ValueError(f"No JSON array found in {path}.")
                    break
                char = buffer[position]
                if not started:
                    if char != "[":
                        raise ValueError(f"Expected '[' to start the array, got {char!r}.")
                    started = True
                    position += 1
                    progressed = True
                    continue
                if char == "]":
                    return
                if char == ",":
                    position += 1
                    progressed = True
                    continue
                try:
                    record, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError:
                    if eof:
                        raise
                    break
                position = end
                yield record
            if eof:
                raise ValueError("Unexpected end of file inside the JSON array.")


def _required_sample(samples: dict[str, Any], case_id: str, arm: str) -> dict[str, Any]:
    sample = samples.get(arm)
    if not isinstance(sample, dict) or "correct_count" not in sample or "mc_k" not in sample:
        raise ValueError(f"Case {case_id!r} is missing MC samples for arm {arm!r}.")
    return sample


def extract_row(record: dict[str, Any]) -> dict[str, Any]:
    case_id = str(record["case_id"])
    samples = record.get("samples") or {}
    redecide = _required_sample(samples, case_id, "redecide")
    keep = _required_sample(samples, case_id, "keep")
    mc_k = int(redecide["mc_k"])
    if int(keep["mc_k"]) != mc_k:
        raise ValueError(f"Case {case_id!r} has inconsistent mc_k across arms.")
    rd = int(redecide["correct_count"])
    arms = sorted(key for key in samples if key.startswith("replace:"))
    if not arms:
        raise ValueError(f"Case {case_id!r} has no replace:<teacher> arm.")
    a_ti = {arm[len("replace:") :]: (int(samples[arm]["correct_count"]) - rd) / mc_k for arm in arms}
    replacements = record.get("teacher_replacements") or {}
    truncated = {
        name: int((replacements.get(name) or {}).get("truncated", 0)) for name in a_ti
    }
    position = record.get("position_fraction")
    return {
        "case_id": case_id,
        "bucket": int(record["bucket"]),
        "rollout_correct": int(record["rollout_correct"]),
        "position_fraction": None if position is None else float(position),
        "mc_k": mc_k,
        "A": (int(keep["correct_count"]) - rd) / mc_k,
        "A_TI": a_ti,
        "truncated": truncated,
    }


def main() -> None:
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
        help="output JSONL path (default: <results dir>/minimal_steps.jsonl).",
    )
    parser.add_argument("--progress-every", type=int, default=2000)
    args = parser.parse_args()

    results_path = args.results
    if results_path.is_dir():
        results_path = results_path / "ersr_results.json"
    if not results_path.is_file():
        raise FileNotFoundError(results_path)
    output_path = args.output or results_path.parent / "minimal_steps.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + f".tmp.{os.getpid()}")

    teachers: Sequence[str] | None = None
    by_bucket: Counter[int] = Counter()
    by_outcome: Counter[int] = Counter()
    truncated_total: Counter[str] = Counter()
    rows = 0
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(
                json.dumps(
                    {
                        "record_type": "header",
                        "schema_version": SCHEMA_VERSION,
                        "results_path": str(results_path),
                        "column_order": [
                            "case_id",
                            "bucket",
                            "rollout_correct",
                            "position_fraction",
                            "mc_k",
                            "A",
                            "A_TI:<teacher> (sorted teacher names below)",
                        ],
                        "teachers": None,  # patched after the first record
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            for index, record in enumerate(iter_json_array(results_path)):
                row = extract_row(record)
                if teachers is None:
                    teachers = sorted(row["A_TI"])
                    header = {
                        "record_type": "header",
                        "schema_version": SCHEMA_VERSION,
                        "results_path": str(results_path),
                        "column_order": [
                            "case_id",
                            "bucket",
                            "rollout_correct",
                            "position_fraction",
                            "mc_k",
                            "A",
                            *(f"A_TI:{name}" for name in teachers),
                        ],
                        "teachers": teachers,
                    }
                    # Rewrite the header with the discovered teacher columns.
                    handle.flush()
                    handle.seek(0)
                    handle.truncate()
                    handle.write(json.dumps(header, ensure_ascii=False) + "\n")
                elif sorted(row["A_TI"]) != list(teachers):
                    raise ValueError(
                        f"Case {row['case_id']!r} teacher set differs from {teachers}."
                    )
                handle.write(
                    json.dumps(
                        [
                            row["case_id"],
                            row["bucket"],
                            row["rollout_correct"],
                            row["position_fraction"],
                            row["mc_k"],
                            row["A"],
                            *(row["A_TI"][name] for name in teachers),
                        ],
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                rows += 1
                by_bucket[row["bucket"]] += 1
                by_outcome[row["rollout_correct"]] += 1
                for name, flag in row["truncated"].items():
                    truncated_total[name] += flag
                if args.progress_every and rows % args.progress_every == 0:
                    print(
                        json.dumps({"event": "progress", "records": rows}),
                        flush=True,
                    )
            if rows == 0:
                raise ValueError(f"No records found in {results_path}.")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()

    print(
        json.dumps(
            {
                "event": "extract_complete",
                "results_path": str(results_path),
                "output_path": str(output_path),
                "rows": rows,
                "teachers": teachers,
                "by_bucket": {str(key): value for key, value in sorted(by_bucket.items())},
                "by_rollout_correct": {
                    str(key): value for key, value in sorted(by_outcome.items())
                },
                "truncated_replacements": dict(sorted(truncated_total.items())),
                "output_bytes": output_path.stat().st_size,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
