"""Combine AIME24 vLLM result shards and print Avg@16."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payloads = []
    for path in args.results:
        with path.open() as handle:
            payloads.append(json.load(handle))
    records = [record for payload in payloads for record in payload["records"]]
    model_names = {payload["model_name"] for payload in payloads}
    if len(model_names) != 1:
        raise ValueError(f"Cannot combine different models: {sorted(model_names)}")
    question_indices = {record["question_index"] for record in records}
    per_question_counts = {
        index: sum(record["question_index"] == index for record in records) for index in question_indices
    }
    expected_questions = sum(payload["num_questions"] for payload in payloads)
    if question_indices != set(range(expected_questions)) or set(per_question_counts.values()) != {16}:
        raise ValueError(f"Expected {expected_questions} questions x 16 samples, got {per_question_counts}")
    summary = {
        "model_name": next(iter(model_names)),
        "num_questions": expected_questions,
        "num_samples": len(records),
        "avg_at_16": sum(record["correct"] for record in records) / len(records),
        "format_valid_rate": sum(record["format_valid"] for record in records) / len(records),
        "truncated_rate": sum(record["finish_reason"] == "length" for record in records) / len(records),
        "mean_response_tokens": sum(record["num_tokens"] for record in records) / len(records),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as handle:
            json.dump({**summary, "records": records}, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
