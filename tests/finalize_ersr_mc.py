#!/usr/bin/env python3
"""Safely finalize an interrupted/resumed ERSR run from persisted shards.

The main ERSR evaluator writes Teacher and Student shards atomically, but its
resume path can regenerate all four Student arms for a partially completed
case.  The resulting duplicate ``(case_id, arm)`` records make the evaluator's
strict final merge abort even when all expensive GPU generation is complete.

This CPU-only finalizer never modifies or deletes source shards.  It:

1. reads ``run_config.json`` and ``selection_manifest.json``;
2. streams every Teacher and Student shard;
3. validates record counts, case IDs, arms, rewards and MC values;
4. removes only duplicates whose result-bearing fields are identical;
5. blocks on conflicting duplicates or missing/unexpected records;
6. writes ``ersr_results.json``, ``ersr_summary.json`` and a detailed
   ``finalization_audit.json`` using atomic replacement.

Student continuation ``texts`` are deliberately excluded from duplicate
equivalence because they are not copied into ``ersr_results.json``.  Their
hashes are still compared and any differences are reported in the audit.

Example:

    python tests/finalize_ersr_mc.py \
        --output-dir outputs/ersr_dapo17k_qwen3_1p7b_mc128

Use ``--audit-only`` first when desired.  Existing final outputs are never
overwritten unless ``--force`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests import run_ersr_mc as base  # noqa: E402


SCHEMA_VERSION = 1
MAX_AUDIT_EXAMPLES = 200
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/ersr_dapo17k_qwen3_1p7b_mc128"


class FinalizationError(RuntimeError):
    """Raised after a blocked audit has been persisted."""


def _int(value: Any, label: str) -> int:
    try:
        return int(value)
    except Exception as exc:
        raise ValueError(f"{label} must be an integer, got {value!r}.") from exc


def _sample(values: Iterable[Any], limit: int = MAX_AUDIT_EXAMPLES) -> list[Any]:
    result: list[Any] = []
    for value in values:
        if len(result) >= limit:
            break
        result.append(value)
    return result


def _relative(path: Path, output_dir: Path) -> str:
    with contextlib.suppress(ValueError):
        return path.relative_to(output_dir).as_posix()
    return str(path)


def _teacher_names(config: dict[str, Any]) -> list[str]:
    teachers = config.get("teachers")
    if not isinstance(teachers, list) or not teachers:
        raise ValueError("run_config.json must contain a non-empty teachers list.")
    names = [str(row.get("name", "")).strip() for row in teachers if isinstance(row, dict)]
    if len(names) != len(teachers) or any(not name for name in names):
        raise ValueError("Every configured Teacher must have a non-empty name.")
    if len(set(names)) != len(names):
        raise ValueError(f"Teacher names are not unique: {names}.")
    return names


def _selection_index(selection: Any) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if not isinstance(selection, list) or not selection:
        raise ValueError("selection_manifest.json must be a non-empty JSON array.")
    rows: list[dict[str, Any]] = []
    by_case: dict[str, dict[str, Any]] = {}
    case_indices: set[int] = set()
    for offset, raw in enumerate(selection):
        if not isinstance(raw, dict):
            raise ValueError(f"Selection row {offset} is not an object.")
        case_id = str(raw.get("case_id", ""))
        case_index = _int(raw.get("case_index"), f"selection[{offset}].case_index")
        if not case_id:
            raise ValueError(f"Selection row {offset} has an empty case_id.")
        if case_id in by_case:
            raise ValueError(f"Duplicate selection case_id {case_id!r}.")
        if case_index in case_indices:
            raise ValueError(f"Duplicate selection case_index {case_index}.")
        rows.append(raw)
        by_case[case_id] = raw
        case_indices.add(case_index)
    expected_indices = set(range(len(rows)))
    if case_indices != expected_indices:
        missing = sorted(expected_indices - case_indices)
        extra = sorted(case_indices - expected_indices)
        raise ValueError(
            "Selection case_index values must be exactly 0..n-1; "
            f"missing={missing[:10]}, extra={extra[:10]}."
        )
    return rows, by_case


def _load_shard(path: Path, role: str) -> list[dict[str, Any]]:
    payload = base.load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Shard {path} is not a JSON object.")
    if str(payload.get("type", "")) != "batch":
        raise ValueError(f"Shard {path} does not have type='batch'.")
    if str(payload.get("role", "")) != role:
        raise ValueError(
            f"Shard {path} has role={payload.get('role')!r}, expected {role!r}."
        )
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Shard {path} does not contain a records array.")
    if _int(payload.get("completed_items", len(records)), f"{path}.completed_items") != len(records):
        raise ValueError(
            f"Shard {path} completed_items does not equal len(records)={len(records)}."
        )
    if any(not isinstance(row, dict) for row in records):
        raise ValueError(f"Shard {path} contains a non-object record.")
    return records


def _normalize_student(
    raw: dict[str, Any],
    *,
    source: str,
    selected: dict[str, dict[str, Any]],
    expected_arms: set[str],
) -> tuple[tuple[str, str], dict[str, Any], str | None]:
    if not bool(raw.get("ok", False)):
        raise ValueError(f"Student record in {source} is not marked ok=true.")
    case_id = str(raw.get("case_id", ""))
    arm = str(raw.get("arm", ""))
    if case_id not in selected:
        raise ValueError(f"Student record in {source} has unexpected case_id {case_id!r}.")
    if arm not in expected_arms:
        raise ValueError(f"Student record {case_id!r} in {source} has unexpected arm {arm!r}.")
    case_index = _int(raw.get("case_index"), f"{source}:{case_id}/{arm}.case_index")
    expected_index = _int(selected[case_id]["case_index"], f"selection:{case_id}.case_index")
    if case_index != expected_index:
        raise ValueError(
            f"Student record {case_id}/{arm} in {source} has case_index={case_index}, "
            f"expected {expected_index}."
        )
    mc_k = _int(raw.get("mc_k"), f"{source}:{case_id}/{arm}.mc_k")
    correct_count = _int(
        raw.get("correct_count"), f"{source}:{case_id}/{arm}.correct_count"
    )
    if mc_k <= 0 or not 0 <= correct_count <= mc_k:
        raise ValueError(
            f"Student record {case_id}/{arm} in {source} has invalid "
            f"mc_k={mc_k}, correct_count={correct_count}."
        )
    rewards = raw.get("rewards")
    if not isinstance(rewards, list) or len(rewards) != mc_k:
        raise ValueError(
            f"Student record {case_id}/{arm} in {source} must contain {mc_k} rewards."
        )
    normalized_rewards = [_int(value, f"{source}:{case_id}/{arm}.reward") for value in rewards]
    if any(value not in (0, 1) for value in normalized_rewards):
        raise ValueError(f"Student record {case_id}/{arm} in {source} has non-binary rewards.")
    if sum(normalized_rewards) != correct_count:
        raise ValueError(
            f"Student record {case_id}/{arm} in {source} has correct_count={correct_count} "
            f"but sum(rewards)={sum(normalized_rewards)}."
        )
    expected_value = correct_count / mc_k
    try:
        stored_value = float(raw.get("value"))
    except Exception as exc:
        raise ValueError(f"Student record {case_id}/{arm} in {source} has invalid value.") from exc
    if not math.isfinite(stored_value) or not math.isclose(
        stored_value, expected_value, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            f"Student record {case_id}/{arm} in {source} has value={stored_value}, "
            f"expected {expected_value}."
        )
    truncated_count = _int(
        raw.get("truncated_count"), f"{source}:{case_id}/{arm}.truncated_count"
    )
    format_invalid_count = _int(
        raw.get("format_invalid_count"),
        f"{source}:{case_id}/{arm}.format_invalid_count",
    )
    if not 0 <= truncated_count <= mc_k or not 0 <= format_invalid_count <= mc_k:
        raise ValueError(
            f"Student record {case_id}/{arm} in {source} has invalid diagnostic counts."
        )
    texts = raw.get("texts")
    if texts is not None and (not isinstance(texts, list) or len(texts) != mc_k):
        raise ValueError(
            f"Student record {case_id}/{arm} in {source} has invalid texts payload."
        )
    normalized = {
        "ok": True,
        "case_id": case_id,
        "case_index": case_index,
        "arm": arm,
        "mc_k": mc_k,
        "correct_count": correct_count,
        "value": expected_value,
        "truncated_count": truncated_count,
        "format_invalid_count": format_invalid_count,
        "rewards": normalized_rewards,
    }
    text_hash = base.canonical_sha256(texts) if texts is not None else None
    return (case_id, arm), normalized, text_hash


def _normalize_teacher(
    raw: dict[str, Any],
    *,
    source: str,
    teacher_name: str,
    selected: dict[str, dict[str, Any]],
) -> tuple[str, dict[str, Any]]:
    if not bool(raw.get("ok", False)):
        raise ValueError(f"Teacher record in {source} is not marked ok=true.")
    case_id = str(raw.get("case_id", ""))
    if case_id not in selected:
        raise ValueError(f"Teacher record in {source} has unexpected case_id {case_id!r}.")
    actual_teacher = str(raw.get("teacher_name", ""))
    if actual_teacher != teacher_name:
        raise ValueError(
            f"Teacher record {case_id!r} in {source} has teacher_name={actual_teacher!r}, "
            f"expected {teacher_name!r}."
        )
    case_index = _int(raw.get("case_index"), f"{source}:{case_id}.case_index")
    expected_index = _int(selected[case_id]["case_index"], f"selection:{case_id}.case_index")
    if case_index != expected_index:
        raise ValueError(
            f"Teacher record {teacher_name}/{case_id} in {source} has "
            f"case_index={case_index}, expected {expected_index}."
        )
    replacement_text = raw.get("replacement_text")
    proposal_text = raw.get("proposal_text")
    if not isinstance(replacement_text, str) or not replacement_text:
        raise ValueError(f"Teacher record {teacher_name}/{case_id} has empty replacement_text.")
    if proposal_text is not None and not isinstance(proposal_text, str):
        raise ValueError(f"Teacher record {teacher_name}/{case_id} has invalid proposal_text.")
    proposal_num_tokens = _int(
        raw.get("proposal_num_tokens"), f"{source}:{case_id}.proposal_num_tokens"
    )
    replacement_num_tokens = _int(
        raw.get("replacement_num_tokens"),
        f"{source}:{case_id}.replacement_num_tokens",
    )
    truncated = _int(raw.get("truncated", 0), f"{source}:{case_id}.truncated")
    if proposal_num_tokens <= 0 or replacement_num_tokens <= 0:
        raise ValueError(f"Teacher record {teacher_name}/{case_id} has invalid token counts.")
    if replacement_num_tokens > proposal_num_tokens:
        raise ValueError(
            f"Teacher record {teacher_name}/{case_id} replacement tokens exceed proposal tokens."
        )
    if truncated not in (0, 1):
        raise ValueError(f"Teacher record {teacher_name}/{case_id} has truncated={truncated}.")
    normalized = {
        "ok": True,
        "case_id": case_id,
        "case_index": case_index,
        "teacher_name": teacher_name,
        "replacement_text": replacement_text,
        "proposal_text": proposal_text,
        "proposal_num_tokens": proposal_num_tokens,
        "replacement_num_tokens": replacement_num_tokens,
        "finish_reason": raw.get("finish_reason"),
        "truncated": truncated,
    }
    return case_id, normalized


def _differing_fields(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    return [key for key in sorted(set(left) | set(right)) if left.get(key) != right.get(key)]


def _scan_student(
    output_dir: Path,
    selected: dict[str, dict[str, Any]],
    expected_arms: set[str],
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, Any]]:
    paths = sorted((output_dir / "student" / "shards").glob("*.json"))
    if not paths:
        raise FileNotFoundError(output_dir / "student" / "shards")
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    metadata: dict[tuple[str, str], dict[str, Any]] = {}
    raw_records = 0
    duplicate_records = 0
    duplicate_keys: set[tuple[str, str]] = set()
    conflicting_keys: set[tuple[str, str]] = set()
    text_difference_keys: set[tuple[str, str]] = set()
    duplicate_examples: list[dict[str, Any]] = []
    conflict_examples: list[dict[str, Any]] = []
    text_difference_examples: list[dict[str, Any]] = []

    for path in paths:
        source = _relative(path, output_dir)
        for raw in _load_shard(path, "student"):
            raw_records += 1
            key, normalized, text_hash = _normalize_student(
                raw,
                source=source,
                selected=selected,
                expected_arms=expected_arms,
            )
            if key not in unique:
                unique[key] = normalized
                metadata[key] = {"source": source, "text_hash": text_hash}
                continue
            duplicate_records += 1
            duplicate_keys.add(key)
            first = unique[key]
            first_meta = metadata[key]
            core_identical = first == normalized
            texts_identical = first_meta["text_hash"] == text_hash
            example = {
                "case_id": key[0],
                "arm": key[1],
                "first_source": first_meta["source"],
                "duplicate_source": source,
                "result_fields_identical": core_identical,
                "texts_identical": texts_identical,
            }
            if len(duplicate_examples) < MAX_AUDIT_EXAMPLES:
                duplicate_examples.append(example)
            if not core_identical:
                conflicting_keys.add(key)
                if len(conflict_examples) < MAX_AUDIT_EXAMPLES:
                    conflict_examples.append(
                        {**example, "differing_fields": _differing_fields(first, normalized)}
                    )
            elif not texts_identical:
                text_difference_keys.add(key)
                if len(text_difference_examples) < MAX_AUDIT_EXAMPLES:
                    text_difference_examples.append(example)

    merged: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for (case_id, arm), row in unique.items():
        merged[case_id][arm] = row
    expected_keys = {(case_id, arm) for case_id in selected for arm in expected_arms}
    actual_keys = set(unique)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    audit = {
        "shards": len(paths),
        "raw_records": raw_records,
        "unique_case_arm_records": len(unique),
        "duplicate_records_removed": duplicate_records,
        "duplicate_case_arm_groups": len(duplicate_keys),
        "conflicting_duplicate_groups": len(conflicting_keys),
        "duplicate_groups_with_different_texts_only": len(text_difference_keys),
        "missing_case_arm_records": len(missing),
        "unexpected_case_arm_records": len(unexpected),
        "duplicate_examples": duplicate_examples,
        "conflict_examples": conflict_examples,
        "text_difference_examples": text_difference_examples,
        "missing_examples": [
            {"case_id": case_id, "arm": arm} for case_id, arm in missing[:MAX_AUDIT_EXAMPLES]
        ],
        "unexpected_examples": [
            {"case_id": case_id, "arm": arm}
            for case_id, arm in unexpected[:MAX_AUDIT_EXAMPLES]
        ],
    }
    return dict(merged), audit


def _scan_teacher(
    output_dir: Path,
    teacher_name: str,
    selected: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    paths = sorted(
        (output_dir / "teachers" / teacher_name / "shards").glob("*.json")
    )
    if not paths:
        raise FileNotFoundError(output_dir / "teachers" / teacher_name / "shards")
    unique: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}
    raw_records = 0
    duplicate_records = 0
    duplicate_cases: set[str] = set()
    conflicting_cases: set[str] = set()
    duplicate_examples: list[dict[str, Any]] = []
    conflict_examples: list[dict[str, Any]] = []
    for path in paths:
        source = _relative(path, output_dir)
        for raw in _load_shard(path, "teacher"):
            raw_records += 1
            case_id, normalized = _normalize_teacher(
                raw,
                source=source,
                teacher_name=teacher_name,
                selected=selected,
            )
            if case_id not in unique:
                unique[case_id] = normalized
                sources[case_id] = source
                continue
            duplicate_records += 1
            duplicate_cases.add(case_id)
            identical = unique[case_id] == normalized
            example = {
                "case_id": case_id,
                "first_source": sources[case_id],
                "duplicate_source": source,
                "result_fields_identical": identical,
            }
            if len(duplicate_examples) < MAX_AUDIT_EXAMPLES:
                duplicate_examples.append(example)
            if not identical:
                conflicting_cases.add(case_id)
                if len(conflict_examples) < MAX_AUDIT_EXAMPLES:
                    conflict_examples.append(
                        {
                            **example,
                            "differing_fields": _differing_fields(
                                unique[case_id], normalized
                            ),
                        }
                    )
    expected_cases = set(selected)
    actual_cases = set(unique)
    missing = sorted(expected_cases - actual_cases)
    unexpected = sorted(actual_cases - expected_cases)
    audit = {
        "shards": len(paths),
        "raw_records": raw_records,
        "unique_case_records": len(unique),
        "duplicate_records_removed": duplicate_records,
        "duplicate_case_groups": len(duplicate_cases),
        "conflicting_duplicate_groups": len(conflicting_cases),
        "missing_case_records": len(missing),
        "unexpected_case_records": len(unexpected),
        "duplicate_examples": duplicate_examples,
        "conflict_examples": conflict_examples,
        "missing_examples": missing[:MAX_AUDIT_EXAMPLES],
        "unexpected_examples": unexpected[:MAX_AUDIT_EXAMPLES],
    }
    return unique, audit


def _audit_problems(audit: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    student = audit["observed"]["student"]
    for field in (
        "conflicting_duplicate_groups",
        "missing_case_arm_records",
        "unexpected_case_arm_records",
    ):
        if int(student[field]):
            problems.append(f"student.{field}={student[field]}")
    for teacher, entry in audit["observed"]["teachers"].items():
        for field in (
            "conflicting_duplicate_groups",
            "missing_case_records",
            "unexpected_case_records",
        ):
            if int(entry[field]):
                problems.append(f"teachers.{teacher}.{field}={entry[field]}")
    return problems


def finalize(
    output_dir: Path,
    *,
    audit_path: Path | None = None,
    audit_only: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Audit shards and, when safe, create the evaluator's final JSON files."""

    output_dir = output_dir.resolve()
    audit_path = (audit_path or output_dir / "finalization_audit.json").resolve()
    audit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": base.utc_now(),
        "status": "started",
        "output_dir": str(output_dir),
        "policy": {
            "source_shards_modified": False,
            "duplicate_result_fields": (
                "Student duplicates are removed only when case/arm, MC counts, "
                "diagnostic counts and rewards are identical; Teacher duplicates "
                "must be fully identical in all final-result fields."
            ),
            "student_texts": (
                "Continuation texts are not copied to ersr_results.json; differing "
                "text hashes with identical result fields are reported but do not block."
            ),
        },
    }
    try:
        if not output_dir.is_dir():
            raise FileNotFoundError(output_dir)
        config_path = output_dir / "run_config.json"
        selection_path = output_dir / "selection_manifest.json"
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        if not selection_path.is_file():
            raise FileNotFoundError(selection_path)
        config = base.load_json(config_path)
        if not isinstance(config, dict):
            raise ValueError("run_config.json must be a JSON object.")
        teachers = _teacher_names(config)
        selection, selected = _selection_index(base.load_json(selection_path))
        expected_arms = set(base.MC_ARMS) | {f"replace:{name}" for name in teachers}
        audit["expected"] = {
            "cases": len(selection),
            "teachers": teachers,
            "arms_per_case": sorted(expected_arms),
            "teacher_case_records_per_teacher": len(selection),
            "student_case_arm_records": len(selection) * len(expected_arms),
        }
        teacher_results: dict[str, dict[str, dict[str, Any]]] = {}
        teacher_audits: dict[str, Any] = {}
        for teacher_name in teachers:
            records, teacher_audit = _scan_teacher(
                output_dir, teacher_name, selected
            )
            teacher_results[teacher_name] = records
            teacher_audits[teacher_name] = teacher_audit
        student_results, student_audit = _scan_student(
            output_dir, selected, expected_arms
        )
        audit["observed"] = {
            "teachers": teacher_audits,
            "student": student_audit,
        }
        problems = _audit_problems(audit)
        audit["problems"] = problems
        if problems:
            raise FinalizationError(
                "Shard audit blocked finalization: " + "; ".join(problems)
            )
        audit["status"] = "validated"
        audit["validated_at"] = base.utc_now()
        audit["outputs"] = {
            "ersr_results": str(output_dir / "ersr_results.json"),
            "ersr_summary": str(output_dir / "ersr_summary.json"),
            "audit": str(audit_path),
        }
        base.atomic_json(audit_path, audit)
        if audit_only:
            audit["status"] = "audit_complete"
            audit["completed_at"] = base.utc_now()
            base.atomic_json(audit_path, audit)
            return audit

        results_path = output_dir / "ersr_results.json"
        summary_path = output_dir / "ersr_summary.json"
        existing = [path for path in (results_path, summary_path) if path.exists()]
        if existing and not force:
            raise FinalizationError(
                "Refusing to overwrite existing final outputs without --force: "
                + ", ".join(str(path) for path in existing)
            )
        results = base.build_results(
            selection,
            list(config["teachers"]),
            teacher_results,
            student_results,
        )
        elapsed_seconds = base.persisted_phase_elapsed(
            output_dir, list(config["teachers"])
        )
        summary = base.summarize_results(results, elapsed_seconds)
        base.atomic_json(results_path, results)
        base.atomic_json(summary_path, summary)
        audit["status"] = "complete"
        audit["completed_at"] = base.utc_now()
        audit["final"] = {
            "cases": len(results),
            "student_duplicate_records_removed": student_audit[
                "duplicate_records_removed"
            ],
            "teacher_duplicate_records_removed": sum(
                int(entry["duplicate_records_removed"])
                for entry in teacher_audits.values()
            ),
            "elapsed_seconds_note": (
                "Recovered from the latest persisted per-phase progress files; "
                "this may exclude elapsed time from earlier interrupted invocations."
            ),
        }
        base.atomic_json(audit_path, audit)
        return audit
    except BaseException as exc:
        audit["status"] = "blocked"
        audit["blocked_at"] = base.utc_now()
        audit["error"] = {"type": type(exc).__name__, "message": str(exc)}
        with contextlib.suppress(Exception):
            base.atomic_json(audit_path, audit)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="ERSR output directory containing run_config, selection and shards.",
    )
    parser.add_argument(
        "--audit-output",
        type=Path,
        default=None,
        help="audit JSON path (default: <output-dir>/finalization_audit.json).",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="validate and write the audit without creating final result files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="atomically replace existing ersr_results/ersr_summary files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        audit = finalize(
            args.output_dir,
            audit_path=args.audit_output,
            audit_only=bool(args.audit_only),
            force=bool(args.force),
        )
    except Exception as exc:
        print(
            f"ERSR finalization blocked: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(1) from exc
    student = audit["observed"]["student"]
    print(
        json.dumps(
            {
                "event": "ersr_finalization_complete",
                "status": audit["status"],
                "cases": audit["expected"]["cases"],
                "unique_student_records": student["unique_case_arm_records"],
                "student_duplicates_removed": student["duplicate_records_removed"],
                "audit": audit["outputs"]["audit"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
