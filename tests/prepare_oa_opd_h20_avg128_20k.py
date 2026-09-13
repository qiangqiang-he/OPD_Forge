#!/usr/bin/env python3
"""Build the locked 20k-step OA-OPD Avg@128 H20 evaluation cohort.

This CPU-only preparation utility intentionally has no command-line
configuration.  All paths and scientific settings are constants below, so
the two emitted JSON datasets can be copied to the server unchanged.

The source is the immutable Qwen3-1.7B no-thinking DAPO-17k rollout set and
its canonical reasoning-step partition.  We select 2,000 clean-correct and
2,000 strictly-incorrect trajectories.  Every selected trajectory contributes
one SHA256-random eligible step from each of five response-relative position
bins, for exactly 10,000 steps per outcome and 20,000 steps overall.

Canonical prompt and response token IDs are stored once per trajectory.  Step
prefixes are always recovered as exact slices of those IDs; the output never
re-tokenizes concatenated text to reconstruct a boundary.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests import prepare_oa_opd_keep_delete_replace_avg16 as base
from utils.oa_opd_step_selection import clean_correct_source, strict_incorrect_source


SCHEMA_VERSION = 1
EXPERIMENT = "oa_opd_h20_keep_delete_replace_avg128_20k_steps"
OUTCOMES = ("correct", "incorrect")
POSITION_BINS = ("p1", "p2", "p3", "p4", "p5")

# Fixed local inputs and outputs.  Run this file from the WSL2 repository copy.
SOURCE_PREPARED_DIR = (
    REPO_ROOT / "output" / "oa_opd_step_eval_selection" / "full_prepared"
)
SOURCE_ROLLOUT_DIR = (
    REPO_ROOT / "data" / "DAPO-17k-Qwen3-1.7B-NoThinking-Rollouts-Max8192"
)
OUTPUT_DIR = (
    REPO_ROOT / "tests" / "artifacts" / "oa_opd_h20_avg128_20k" / "prepared"
)
CORRECT_OUTPUT_PATH = OUTPUT_DIR / "correct_2000_trajectories_10000_steps.json"
INCORRECT_OUTPUT_PATH = OUTPUT_DIR / "incorrect_2000_trajectories_10000_steps.json"
SUMMARY_OUTPUT_PATH = OUTPUT_DIR / "selection_summary.json"
MANIFEST_OUTPUT_PATH = OUTPUT_DIR / "selection_manifest.json"

# Locked experimental settings.
SELECTION_SEED = 20260911
GENERATION_SEED = 20260911
RESPONSES_PER_OUTCOME = 2000
STEPS_PER_RESPONSE = 5
SAMPLES_PER_ARM = 128
MAX_RESPONSE_TOKENS = 10240
TEACHER_PROPOSAL_MAX_NEW_TOKENS = 1024
MINIMUM_CONTINUATION_TOKENS = 256
MAXIMUM_STUDENT_STEP_TOKENS = 512
STUDENT_MODEL_BASENAME = "Qwen3-1.7B"
TEACHER_MODEL_BASENAME = "Qwen3-4B-Instruct-2507"
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = -1


def position_fraction(step_offset: int, num_steps: int) -> float:
    if num_steps < STEPS_PER_RESPONSE:
        raise ValueError(
            f"Five-bin sampling requires at least {STEPS_PER_RESPONSE} steps."
        )
    if not 0 <= step_offset < num_steps:
        raise ValueError("Step offset is outside its reasoning-step sequence.")
    return step_offset / (num_steps - 1)


def position_bin_index(step_offset: int, num_steps: int) -> int:
    """Map a canonical reasoning step into one of five relative-position bins."""

    return min(
        len(POSITION_BINS) - 1,
        int(len(POSITION_BINS) * position_fraction(step_offset, num_steps)),
    )


def position_third(position: float) -> str:
    if not 0.0 <= position <= 1.0:
        raise ValueError(f"Invalid relative position: {position}.")
    if position < 1.0 / 3.0:
        return "early"
    if position < 2.0 / 3.0:
        return "middle"
    return "late"


def eligible_steps_by_position_bin(
    record: dict[str, Any],
    *,
    max_response_tokens: int,
    teacher_proposal_max_new_tokens: int,
    minimum_continuation_tokens: int,
    maximum_student_step_tokens: int,
) -> dict[str, list[dict[str, Any]]]:
    """Return generation-safe steps grouped by response-relative quintile."""

    grouped = {name: [] for name in POSITION_BINS}
    steps = record["steps"]
    if len(steps) < STEPS_PER_RESPONSE:
        return grouped

    maximum_keep_boundary = max_response_tokens - minimum_continuation_tokens
    maximum_pre_boundary = (
        max_response_tokens
        - teacher_proposal_max_new_tokens
        - minimum_continuation_tokens
    )
    for offset, step in enumerate(steps):
        token_start = int(step["token_start"])
        token_end = int(step["token_end"])
        if token_end - token_start > maximum_student_step_tokens:
            continue
        if token_end > maximum_keep_boundary or token_start > maximum_pre_boundary:
            continue
        position = position_fraction(offset, len(steps))
        bin_index = position_bin_index(offset, len(steps))
        grouped[POSITION_BINS[bin_index]].append(
            {
                "offset": offset,
                "position_fraction": position,
                "position_bin_index": bin_index,
                "step": step,
            }
        )
    return grouped


def _source_matches_outcome(record: dict[str, Any], outcome: str) -> bool:
    if outcome == "correct":
        return clean_correct_source(record)
    if outcome == "incorrect":
        return strict_incorrect_source(record)
    raise ValueError(f"Unknown outcome: {outcome}.")


def select_responses_and_steps(
    records: dict[int, dict[str, Any]],
    *,
    responses_per_outcome: int,
    selection_seed: int,
    max_response_tokens: int,
    teacher_proposal_max_new_tokens: int,
    minimum_continuation_tokens: int,
    maximum_student_step_tokens: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Lock outcome cohorts first, then choose one random step per position bin."""

    selected: dict[str, list[dict[str, Any]]] = {}
    pool_summary: dict[str, Any] = {}
    for outcome in OUTCOMES:
        outcome_records = [
            record
            for record in records.values()
            if str(record["source_outcome"]) == outcome
            and _source_matches_outcome(record, outcome)
        ]
        eligibility: dict[int, dict[str, list[dict[str, Any]]]] = {}
        eligible_step_counts: Counter[str] = Counter()
        source_num_steps: list[int] = []
        eligible_num_steps: list[int] = []
        exclusion_counts: Counter[str] = Counter()

        for record in outcome_records:
            source_num_steps.append(len(record["steps"]))
            grouped = eligible_steps_by_position_bin(
                record,
                max_response_tokens=max_response_tokens,
                teacher_proposal_max_new_tokens=teacher_proposal_max_new_tokens,
                minimum_continuation_tokens=minimum_continuation_tokens,
                maximum_student_step_tokens=maximum_student_step_tokens,
            )
            missing = [name for name in POSITION_BINS if not grouped[name]]
            if missing:
                exclusion_counts["missing_one_or_more_position_bins"] += 1
                for name in missing:
                    exclusion_counts[f"missing_{name}"] += 1
                continue
            record_index = int(record["record_index"])
            eligibility[record_index] = grouped
            eligible_count = sum(len(grouped[name]) for name in POSITION_BINS)
            eligible_num_steps.append(eligible_count)
            for name in POSITION_BINS:
                eligible_step_counts[name] += len(grouped[name])

        if len(eligibility) < responses_per_outcome:
            raise ValueError(
                f"Only {len(eligibility)} eligible {outcome} responses; "
                f"need {responses_per_outcome}."
            )

        response_order = sorted(
            eligibility,
            key=lambda record_index: (
                base.stable_digest(
                    selection_seed, "response", outcome, record_index
                ),
                record_index,
            ),
        )
        selected_indices = response_order[:responses_per_outcome]
        outcome_selections: list[dict[str, Any]] = []
        for response_rank, record_index in enumerate(selected_indices, start=1):
            record = records[record_index]
            grouped = eligibility[record_index]
            picked_steps: list[dict[str, Any]] = []
            for name in POSITION_BINS:
                picked = min(
                    grouped[name],
                    key=lambda candidate: (
                        base.stable_digest(
                            selection_seed,
                            "step",
                            outcome,
                            record_index,
                            name,
                            candidate["step"]["step_index"],
                        ),
                        int(candidate["step"]["step_index"]),
                    ),
                )
                picked_steps.append({"position_bin": name, **picked})
            outcome_selections.append(
                {
                    "outcome": outcome,
                    "outcome_response_rank": response_rank,
                    "record": record,
                    "eligible_counts_by_position_bin": {
                        name: len(grouped[name]) for name in POSITION_BINS
                    },
                    "picked_steps": picked_steps,
                }
            )

        selected[outcome] = outcome_selections
        pool_summary[outcome] = {
            "source_responses": len(outcome_records),
            "eligible_responses": len(eligibility),
            "excluded_responses": len(outcome_records) - len(eligibility),
            "exclusion_counts": dict(sorted(exclusion_counts.items())),
            "eligible_steps_by_position_bin": {
                name: eligible_step_counts[name] for name in POSITION_BINS
            },
            "selected_responses": len(outcome_selections),
            "selected_steps": len(outcome_selections) * STEPS_PER_RESPONSE,
            "source_response_num_steps": base.distribution_summary(source_num_steps),
            "eligible_response_num_steps": base.distribution_summary(
                eligible_num_steps
            ),
        }
    return selected, pool_summary


def _selected_step_row(
    *,
    selection: dict[str, Any],
    picked: dict[str, Any],
    cohort_case_index: int,
    global_case_index: int,
    response_ids: Sequence[int],
    max_response_tokens: int,
    teacher_proposal_max_new_tokens: int,
    minimum_continuation_tokens: int,
    maximum_student_step_tokens: int,
    samples_per_arm: int,
    generation_seed: int,
) -> dict[str, Any]:
    record = selection["record"]
    step = picked["step"]
    offset = int(picked["offset"])
    token_start = int(step["token_start"])
    token_end = int(step["token_end"])
    step_ids = [int(value) for value in response_ids[token_start:token_end]]
    position = float(picked["position_fraction"])
    return {
        "cohort_case_index": cohort_case_index,
        "global_case_index": global_case_index,
        "case_id": f"{int(record['record_index'])}:{int(step['step_index'])}",
        "within_response_step_slot": int(picked["position_bin_index"]),
        "response_position_bin": str(picked["position_bin"]),
        "response_position_bin_index": int(picked["position_bin_index"]),
        "response_position_fraction": position,
        "response_position_third": position_third(position),
        "step_offset": offset,
        "step_index": int(step["step_index"]),
        "reasoning_step_index": int(step["reasoning_step_index"]),
        "step_char_start": int(step["char_start"]),
        "step_char_end": int(step["char_end"]),
        "step_token_start": token_start,
        "step_token_end": token_end,
        "step_num_tokens": len(step_ids),
        "step_text_sha256": base.text_sha256(str(step["text"])),
        "step_token_ids_sha256": base.token_ids_sha256(step_ids),
        "keep_continuation_max_new_tokens": max_response_tokens - token_end,
        "delete_continuation_max_new_tokens": max_response_tokens - token_start,
        "replace_continuation_max_new_tokens_if_full_replacement": (
            max_response_tokens
            - token_start
            - teacher_proposal_max_new_tokens
        ),
        "minimum_continuation_tokens": minimum_continuation_tokens,
        "maximum_student_step_tokens": maximum_student_step_tokens,
        "samples_per_arm": samples_per_arm,
        "generation_seed": generation_seed,
    }


def build_response_rows(
    selections: dict[str, list[dict[str, Any]]],
    *,
    response_ids_by_record: dict[int, list[int]],
    array_bindings: dict[int, dict[str, Any]],
    prepared_bindings: dict[int, dict[str, Any]],
    selection_seed: int,
    max_response_tokens: int,
    teacher_proposal_max_new_tokens: int,
    minimum_continuation_tokens: int,
    maximum_student_step_tokens: int,
    samples_per_arm: int,
    generation_seed: int,
) -> dict[str, list[dict[str, Any]]]:
    rows_by_outcome: dict[str, list[dict[str, Any]]] = {}
    global_case_index = 0
    for outcome in OUTCOMES:
        rows: list[dict[str, Any]] = []
        cohort_case_index = 0
        for selection in selections[outcome]:
            record = selection["record"]
            record_index = int(record["record_index"])
            response_ids = response_ids_by_record[record_index]
            prompt_ids = [int(value) for value in record["prompt_token_ids"]]
            selected_steps: list[dict[str, Any]] = []
            for picked in selection["picked_steps"]:
                selected_steps.append(
                    _selected_step_row(
                        selection=selection,
                        picked=picked,
                        cohort_case_index=cohort_case_index,
                        global_case_index=global_case_index,
                        response_ids=response_ids,
                        max_response_tokens=max_response_tokens,
                        teacher_proposal_max_new_tokens=(
                            teacher_proposal_max_new_tokens
                        ),
                        minimum_continuation_tokens=minimum_continuation_tokens,
                        maximum_student_step_tokens=maximum_student_step_tokens,
                        samples_per_arm=samples_per_arm,
                        generation_seed=generation_seed,
                    )
                )
                cohort_case_index += 1
                global_case_index += 1

            steps = [dict(step) for step in record["steps"]]
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "experiment": EXPERIMENT,
                    "trajectory_id": f"{outcome}:{record_index}",
                    "record_index": record_index,
                    "source_outcome": outcome,
                    "outcome_response_rank": int(
                        selection["outcome_response_rank"]
                    ),
                    "question": str(record["question"]),
                    "answer": str(record["answer"]),
                    "source_response": str(record["response"]),
                    "source_extracted_answer": str(record["extracted_answer"]),
                    "source_correct": int(record["correct"]),
                    "source_format_valid": int(record["format_valid"]),
                    "source_finish_reason": str(record["finish_reason"]),
                    "source_shard_index": int(record["source_shard_index"]),
                    "source_array_row": int(record["source_array_row"]),
                    "source_response_text_sha256": base.text_sha256(
                        str(record["response"])
                    ),
                    "prompt_token_ids": prompt_ids,
                    "prompt_num_tokens": len(prompt_ids),
                    "prompt_token_ids_sha256": base.token_ids_sha256(prompt_ids),
                    "source_response_token_ids": response_ids,
                    "source_response_num_tokens": len(response_ids),
                    "source_response_token_ids_sha256": base.token_ids_sha256(
                        response_ids
                    ),
                    "probe_text": str(record["probe_text"]),
                    "probe_token_ids": [
                        int(value) for value in record["probe_token_ids"]
                    ],
                    "probe_answer_token_positions": [
                        int(value)
                        for value in record["probe_answer_token_positions"]
                    ],
                    "step_partition": {
                        "policy": "canonical_reasoning_steps_before_final_answer",
                        "num_reasoning_steps": len(steps),
                        "steps": steps,
                        "boundary_probes": [
                            dict(probe) for probe in record["boundary_probes"]
                        ],
                    },
                    "eligible_counts_by_position_bin": dict(
                        selection["eligible_counts_by_position_bin"]
                    ),
                    "selected_steps": selected_steps,
                    "selection_seed": selection_seed,
                    "selection_policy": (
                        "outcome_balanced_response_sha256_random_then_one_"
                        "sha256_random_eligible_step_per_response_relative_quintile"
                    ),
                    "delta_used_for_selection": False,
                    "teacher_score_used_for_selection": False,
                    "prior_continuation_reward_used_for_selection": False,
                    "source_prepared_record_sha256": base.canonical_sha256(record),
                    **prepared_bindings[record_index],
                    **array_bindings[record_index],
                }
            )
        rows_by_outcome[outcome] = rows
    return rows_by_outcome


def validate_response_rows(
    rows_by_outcome: dict[str, list[dict[str, Any]]],
    *,
    responses_per_outcome: int,
    steps_per_response: int,
    max_response_tokens: int,
    teacher_proposal_max_new_tokens: int,
    minimum_continuation_tokens: int,
    maximum_student_step_tokens: int,
) -> dict[str, Any]:
    """Audit counts, source labels, text spans, token slices, and positions."""

    all_record_indices: set[int] = set()
    all_case_ids: set[str] = set()
    all_global_indices: set[int] = set()
    summary: dict[str, Any] = {}
    for outcome in OUTCOMES:
        rows = rows_by_outcome[outcome]
        if len(rows) != responses_per_outcome:
            raise ValueError(f"Unexpected {outcome} response count: {len(rows)}.")
        position_counts: Counter[str] = Counter()
        third_counts: Counter[str] = Counter()
        selected_lengths: list[int] = []
        response_lengths: list[int] = []
        response_step_counts: list[int] = []

        for row in rows:
            record_index = int(row["record_index"])
            if record_index in all_record_indices:
                raise ValueError(f"Duplicate selected response {record_index}.")
            all_record_indices.add(record_index)
            source_view = {
                "correct": row["source_correct"],
                "format_valid": row["source_format_valid"],
                "finish_reason": row["source_finish_reason"],
                "answer": row["answer"],
                "extracted_answer": row["source_extracted_answer"],
            }
            if not _source_matches_outcome(source_view, outcome):
                raise ValueError(f"Source-outcome audit failed for {record_index}.")

            prompt_ids = [int(value) for value in row["prompt_token_ids"]]
            response_ids = [
                int(value) for value in row["source_response_token_ids"]
            ]
            if len(prompt_ids) != int(row["prompt_num_tokens"]):
                raise ValueError(f"Prompt length mismatch for {record_index}.")
            if len(response_ids) != int(row["source_response_num_tokens"]):
                raise ValueError(f"Response length mismatch for {record_index}.")
            if base.token_ids_sha256(prompt_ids) != row["prompt_token_ids_sha256"]:
                raise ValueError(f"Prompt hash mismatch for {record_index}.")
            if (
                base.token_ids_sha256(response_ids)
                != row["source_response_token_ids_sha256"]
            ):
                raise ValueError(f"Response hash mismatch for {record_index}.")

            partition = row["step_partition"]
            steps = partition["steps"]
            if int(partition["num_reasoning_steps"]) != len(steps):
                raise ValueError(f"Step count mismatch for {record_index}.")
            if len(partition["boundary_probes"]) != len(steps) + 1:
                raise ValueError(f"Boundary count mismatch for {record_index}.")
            previous_token_end = 0
            previous_char_end = 0
            for expected_offset, step in enumerate(steps):
                token_start = int(step["token_start"])
                token_end = int(step["token_end"])
                char_start = int(step["char_start"])
                char_end = int(step["char_end"])
                if token_start != previous_token_end or char_start != previous_char_end:
                    raise ValueError(f"Non-contiguous step partition for {record_index}.")
                if int(step["reasoning_step_index"]) != expected_offset:
                    raise ValueError(f"Step offset mismatch for {record_index}.")
                if token_end - token_start != int(step["num_tokens"]):
                    raise ValueError(f"Step token length mismatch for {record_index}.")
                if row["source_response"][char_start:char_end] != str(step["text"]):
                    raise ValueError(f"Step text span mismatch for {record_index}.")
                previous_token_end = token_end
                previous_char_end = char_end

            selected_steps = row["selected_steps"]
            if len(selected_steps) != steps_per_response:
                raise ValueError(f"Selected-step count mismatch for {record_index}.")
            if {step["response_position_bin"] for step in selected_steps} != set(
                POSITION_BINS
            ):
                raise ValueError(f"Position-bin coverage mismatch for {record_index}.")
            for selected in selected_steps:
                case_id = str(selected["case_id"])
                global_index = int(selected["global_case_index"])
                if case_id in all_case_ids or global_index in all_global_indices:
                    raise ValueError(f"Duplicate selected case {case_id}.")
                all_case_ids.add(case_id)
                all_global_indices.add(global_index)
                offset = int(selected["step_offset"])
                canonical = steps[offset]
                if int(selected["step_index"]) != int(canonical["step_index"]):
                    raise ValueError(f"Selected-step identity mismatch for {case_id}.")
                token_start = int(selected["step_token_start"])
                token_end = int(selected["step_token_end"])
                if (token_start, token_end) != (
                    int(canonical["token_start"]),
                    int(canonical["token_end"]),
                ):
                    raise ValueError(f"Selected-step token span mismatch for {case_id}.")
                step_ids = response_ids[token_start:token_end]
                if len(step_ids) > maximum_student_step_tokens:
                    raise ValueError(f"Selected step exceeds cap for {case_id}.")
                if base.token_ids_sha256(step_ids) != selected["step_token_ids_sha256"]:
                    raise ValueError(f"Selected-step token hash mismatch for {case_id}.")
                expected_position = position_fraction(offset, len(steps))
                expected_bin_index = position_bin_index(offset, len(steps))
                expected_bin = POSITION_BINS[expected_bin_index]
                if selected["response_position_bin"] != expected_bin:
                    raise ValueError(f"Selected-step position mismatch for {case_id}.")
                if abs(float(selected["response_position_fraction"]) - expected_position) > 1e-12:
                    raise ValueError(f"Selected-step position fraction mismatch for {case_id}.")
                if max_response_tokens - token_end < minimum_continuation_tokens:
                    raise ValueError(f"Insufficient Keep budget for {case_id}.")
                if (
                    max_response_tokens
                    - token_start
                    - teacher_proposal_max_new_tokens
                    < minimum_continuation_tokens
                ):
                    raise ValueError(f"Insufficient Replace budget for {case_id}.")
                position_counts[expected_bin] += 1
                third_counts[str(selected["response_position_third"])] += 1
                selected_lengths.append(len(step_ids))

            response_lengths.append(len(response_ids))
            response_step_counts.append(len(steps))

        expected_per_bin = responses_per_outcome
        observed_bins = {name: position_counts[name] for name in POSITION_BINS}
        if any(value != expected_per_bin for value in observed_bins.values()):
            raise ValueError(f"Unbalanced {outcome} positions: {observed_bins}.")
        summary[outcome] = {
            "responses": len(rows),
            "selected_steps": len(rows) * steps_per_response,
            "selected_steps_by_position_bin": observed_bins,
            "selected_steps_by_position_third": dict(sorted(third_counts.items())),
            "source_response_token_lengths": base.distribution_summary(
                response_lengths
            ),
            "source_response_reasoning_step_counts": base.distribution_summary(
                response_step_counts
            ),
            "selected_step_token_lengths": base.distribution_summary(
                selected_lengths
            ),
        }

    expected_total = responses_per_outcome * steps_per_response * len(OUTCOMES)
    if all_global_indices != set(range(expected_total)):
        raise ValueError("Global case indices are not one contiguous 20k range.")
    return summary


def _dataset_metadata(
    *,
    outcome: str,
    response_count: int,
    source_signature: dict[str, Any],
    selection_seed: int,
    generation_seed: int,
    samples_per_arm: int,
    max_response_tokens: int,
    teacher_proposal_max_new_tokens: int,
    source_max_response_tokens: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "source_outcome": outcome,
        "response_count": response_count,
        "selected_step_count": response_count * STEPS_PER_RESPONSE,
        "steps_per_response": STEPS_PER_RESPONSE,
        "position_bins": list(POSITION_BINS),
        "selection_seed": selection_seed,
        "generation_seed": generation_seed,
        "selection_is_delta_blind": True,
        "selection_is_teacher_score_blind": True,
        "selection_is_prior_reward_blind": True,
        "student_model": source_signature.get("model"),
        "teacher_model_basename": TEACHER_MODEL_BASENAME,
        "prompt_name": source_signature.get("prompt_name"),
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "max_response_tokens": max_response_tokens,
        "teacher_proposal_max_new_tokens": teacher_proposal_max_new_tokens,
        "source_max_response_tokens": source_max_response_tokens,
        "samples_per_arm": samples_per_arm,
        "arms": ["keep", "delete", "replace"],
        "student_continuations": response_count
        * STEPS_PER_RESPONSE
        * samples_per_arm
        * 3,
        "token_storage": (
            "canonical prompt/response token IDs once per trajectory; prefixes and "
            "student-step IDs are exact slices"
        ),
    }


def atomic_dataset_json(
    path: Path, *, metadata: dict[str, Any], responses: Iterable[dict[str, Any]]
) -> None:
    """Write valid compact JSON while keeping one response per physical line."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write('{"metadata":')
            json.dump(
                metadata,
                handle,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.write(',"responses":[\n')
            for index, response in enumerate(responses):
                if index:
                    handle.write(",\n")
                json.dump(
                    response,
                    handle,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            handle.write("\n]}\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_dataset(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or not isinstance(value.get("responses"), list):
        raise ValueError(f"Invalid prepared dataset: {path}.")
    return value


def prepare(
    *,
    source_prepared_dir: Path = SOURCE_PREPARED_DIR,
    rollout_dir: Path = SOURCE_ROLLOUT_DIR,
    output_dir: Path = OUTPUT_DIR,
    responses_per_outcome: int = RESPONSES_PER_OUTCOME,
    selection_seed: int = SELECTION_SEED,
    generation_seed: int = GENERATION_SEED,
    samples_per_arm: int = SAMPLES_PER_ARM,
    max_response_tokens: int = MAX_RESPONSE_TOKENS,
    teacher_proposal_max_new_tokens: int = TEACHER_PROPOSAL_MAX_NEW_TOKENS,
    minimum_continuation_tokens: int = MINIMUM_CONTINUATION_TOKENS,
    maximum_student_step_tokens: int = MAXIMUM_STUDENT_STEP_TOKENS,
) -> tuple[dict[str, Path], dict[str, Any], dict[str, Any]]:
    if STEPS_PER_RESPONSE != len(POSITION_BINS):
        raise AssertionError("The locked design requires one step per position bin.")
    if responses_per_outcome <= 0 or samples_per_arm <= 0:
        raise ValueError("Response and rollout counts must be positive.")
    if teacher_proposal_max_new_tokens + minimum_continuation_tokens >= max_response_tokens:
        raise ValueError("Teacher proposal and continuation reserve exhaust the cap.")

    source_prepared_dir = source_prepared_dir.resolve()
    rollout_dir = rollout_dir.resolve()
    output_dir = output_dir.resolve()
    prepared_source = base.load_prepared_source(source_prepared_dir)
    # The checked-in source rollout was generated with an 8192-token cap.  A
    # stricter source cap remains valid when the downstream evaluation budget
    # is expanded, so validate against the cap recorded by that source
    # manifest while using the requested (larger) cap for selection/output.
    source_manifest = base.load_json(rollout_dir / "manifest.json")
    source_signature = source_manifest.get("signature", {})
    source_max_response_tokens = int(source_signature.get("max_new_tokens", -1))
    if source_max_response_tokens <= 0:
        raise ValueError("Source rollout manifest has no positive max_new_tokens.")
    if source_max_response_tokens > max_response_tokens:
        raise ValueError(
            "Source rollout response cap exceeds the requested evaluation cap."
        )
    rollout_manifest, rollout_manifest_path, rollout_manifest_hash = (
        base.validate_rollout_source(
            rollout_dir,
            prepared_source,
            max_response_tokens=source_max_response_tokens,
        )
    )
    source_signature = rollout_manifest.get("signature", {})
    source_signature = rollout_manifest.get("signature", {})
    if Path(str(source_signature.get("model", ""))).name != STUDENT_MODEL_BASENAME:
        raise ValueError(f"Source model is not {STUDENT_MODEL_BASENAME}.")

    selections, pool_summary = select_responses_and_steps(
        prepared_source["records"],
        responses_per_outcome=responses_per_outcome,
        selection_seed=selection_seed,
        max_response_tokens=max_response_tokens,
        teacher_proposal_max_new_tokens=teacher_proposal_max_new_tokens,
        minimum_continuation_tokens=minimum_continuation_tokens,
        maximum_student_step_tokens=maximum_student_step_tokens,
    )
    flat_selections = [
        selection for outcome in OUTCOMES for selection in selections[outcome]
    ]
    response_ids, array_bindings = base.load_selected_response_token_ids(
        rollout_dir, rollout_manifest, flat_selections
    )
    rows_by_outcome = build_response_rows(
        selections,
        response_ids_by_record=response_ids,
        array_bindings=array_bindings,
        prepared_bindings=prepared_source["bindings"],
        selection_seed=selection_seed,
        max_response_tokens=max_response_tokens,
        teacher_proposal_max_new_tokens=teacher_proposal_max_new_tokens,
        minimum_continuation_tokens=minimum_continuation_tokens,
        maximum_student_step_tokens=maximum_student_step_tokens,
        samples_per_arm=samples_per_arm,
        generation_seed=generation_seed,
    )
    validation = validate_response_rows(
        rows_by_outcome,
        responses_per_outcome=responses_per_outcome,
        steps_per_response=STEPS_PER_RESPONSE,
        max_response_tokens=max_response_tokens,
        teacher_proposal_max_new_tokens=teacher_proposal_max_new_tokens,
        minimum_continuation_tokens=minimum_continuation_tokens,
        maximum_student_step_tokens=maximum_student_step_tokens,
    )

    output_paths = {
        "correct": output_dir / CORRECT_OUTPUT_PATH.name,
        "incorrect": output_dir / INCORRECT_OUTPUT_PATH.name,
    }
    for outcome in OUTCOMES:
        atomic_dataset_json(
            output_paths[outcome],
            metadata=_dataset_metadata(
                outcome=outcome,
                response_count=responses_per_outcome,
                source_signature=source_signature,
                selection_seed=selection_seed,
                generation_seed=generation_seed,
                samples_per_arm=samples_per_arm,
                max_response_tokens=max_response_tokens,
                teacher_proposal_max_new_tokens=teacher_proposal_max_new_tokens,
                source_max_response_tokens=source_max_response_tokens,
            ),
            responses=rows_by_outcome[outcome],
        )

    # Round-trip the exact files before publishing their hashes.
    for outcome in OUTCOMES:
        loaded = load_dataset(output_paths[outcome])
        if len(loaded["responses"]) != responses_per_outcome:
            raise ValueError(f"Round-trip response count failed for {outcome}.")
        if int(loaded["metadata"]["selected_step_count"]) != (
            responses_per_outcome * STEPS_PER_RESPONSE
        ):
            raise ValueError(f"Round-trip selected-step count failed for {outcome}.")

    total_steps = responses_per_outcome * STEPS_PER_RESPONSE * len(OUTCOMES)
    total_rollouts = total_steps * samples_per_arm * 3
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "responses": responses_per_outcome * len(OUTCOMES),
        "selected_steps": total_steps,
        "responses_per_outcome": responses_per_outcome,
        "steps_per_response": STEPS_PER_RESPONSE,
        "samples_per_arm": samples_per_arm,
        "student_continuations": total_rollouts,
        "teacher_proposals": total_steps,
        "validation": validation,
        "pool": pool_summary,
        "outputs": {
            outcome: {
                "path": str(output_paths[outcome]),
                "bytes": output_paths[outcome].stat().st_size,
                "sha256": base.sha256_file(output_paths[outcome]),
                "responses": responses_per_outcome,
                "selected_steps": responses_per_outcome * STEPS_PER_RESPONSE,
            }
            for outcome in OUTCOMES
        },
    }
    summary_path = output_dir / SUMMARY_OUTPUT_PATH.name
    base.atomic_json(summary_path, summary)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "status": "prepared_and_validated",
        "source": {
            "rollout_dir": str(rollout_dir),
            "rollout_manifest": str(rollout_manifest_path),
            "rollout_manifest_sha256": rollout_manifest_hash,
            "source_max_response_tokens": source_max_response_tokens,
            "source_prepared_dir": str(source_prepared_dir),
            "source_preparation_manifest": str(prepared_source["manifest_path"]),
            "source_preparation_manifest_sha256": prepared_source["manifest_sha256"],
            "source_prepared_shards": prepared_source["prepared_shards"],
            "source_prepared_shard_collection_sha256": prepared_source[
                "prepared_shard_collection_sha256"
            ],
            "source_prepared_records": len(prepared_source["records"]),
            "source_prepared_reasoning_steps": prepared_source["reasoning_steps"],
            "source_prepared_boundary_probes": prepared_source["boundary_probes"],
        },
        "source_outcomes": {
            "correct": "correct=1, format_valid=1, finish_reason=stop",
            "incorrect": (
                "correct=0, format_valid=1, finish_reason=stop, and exact "
                "inequality between parseable scalar integer/decimal/rational "
                "gold and extracted answers"
            ),
        },
        "selection": {
            "selection_seed": selection_seed,
            "responses_per_outcome": responses_per_outcome,
            "steps_per_response": STEPS_PER_RESPONSE,
            "position_bins": list(POSITION_BINS),
            "relative_position_formula": "step_offset / (num_reasoning_steps - 1)",
            "position_bin_formula": "min(4, floor(5 * relative_position))",
            "response_randomization": (
                "ascending SHA256(seed|response|outcome|record_index) among "
                "five-bin-eligible responses, then take quota"
            ),
            "step_randomization": (
                "minimum SHA256(seed|step|outcome|record_index|position_bin|"
                "step_index) within each bin"
            ),
            "maximum_student_step_tokens": maximum_student_step_tokens,
            "teacher_proposal_max_new_tokens": teacher_proposal_max_new_tokens,
            "minimum_continuation_tokens": minimum_continuation_tokens,
            "delta_blind": True,
            "teacher_score_blind": True,
            "prior_reward_blind": True,
            "pool": pool_summary,
        },
        "generation": {
            "generation_seed": generation_seed,
            "student_model": source_signature.get("model"),
            "teacher_model_basename": TEACHER_MODEL_BASENAME,
            "prompt_name": source_signature.get("prompt_name"),
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "max_response_tokens": max_response_tokens,
            "source_max_response_tokens": source_max_response_tokens,
            "samples_per_arm": samples_per_arm,
            "arms": ["keep", "delete", "replace"],
            "value_definition": (
                f"mean of {samples_per_arm} binary correctness rewards per arm"
            ),
            "teacher_proposals": total_steps,
            "student_continuations_per_arm": total_steps * samples_per_arm,
            "total_student_continuations": total_rollouts,
            "server_allocation": {
                "correct": "one independent 8xH20 job",
                "incorrect": "one independent 8xH20 job",
            },
        },
        "token_policy": (
            "canonical prompt and response token IDs are stored once per trajectory; "
            "all intervention prefixes and step spans must be exact slices, never "
            "reconstructed by tokenizing concatenated text"
        ),
        "outputs": summary["outputs"],
        "summary": {
            "path": str(summary_path),
            "sha256": base.sha256_file(summary_path),
        },
    }
    manifest_path = output_dir / MANIFEST_OUTPUT_PATH.name
    base.atomic_json(manifest_path, manifest)
    return output_paths, summary, manifest


def main() -> None:
    output_paths, summary, _ = prepare()
    print(
        json.dumps(
            {
                "status": "prepared_and_validated",
                "correct_json": str(output_paths["correct"]),
                "correct_bytes": summary["outputs"]["correct"]["bytes"],
                "incorrect_json": str(output_paths["incorrect"]),
                "incorrect_bytes": summary["outputs"]["incorrect"]["bytes"],
                "responses": summary["responses"],
                "selected_steps": summary["selected_steps"],
                "student_continuations": summary["student_continuations"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
