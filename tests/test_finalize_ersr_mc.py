from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from tests import finalize_ersr_mc as finalizer
from tests import run_ersr_mc as base


REPO_ROOT = Path(__file__).resolve().parents[1]
TEACHERS = ("teacher_a", "teacher_b")
MC_K = 4


def _selection_row(case_index: int) -> dict:
    return {
        "case_index": case_index,
        "case_id": f"case-{case_index}",
        "record_index": case_index,
        "response_idx": 100 + case_index,
        "rollout_index": case_index,
        "bucket": 4 if case_index == 0 else 1,
        "rollout_correct": 1 if case_index == 0 else 0,
        "step_index": 2,
        "reasoning_step_index": 2,
        "num_steps": 5,
        "num_reasoning_steps": 4,
        "step_char_start": 10,
        "step_char_end": 20,
        "step_token_start": 4,
        "step_token_end": 8,
        "position_fraction": 2 / 3,
        "question": f"question {case_index}",
        "answer": "42",
        "response": f"response {case_index}",
        "step_text": "reasoning step",
        "prompt_token_ids": [1, 2, 3],
        "response_token_ids": [4, 5, 6, 7, 8],
    }


def _teacher_record(case_index: int, teacher: str) -> dict:
    return {
        "ok": True,
        "case_id": f"case-{case_index}",
        "case_index": case_index,
        "teacher_name": teacher,
        "replacement_text": f"replacement {teacher} {case_index}",
        "proposal_text": f"proposal {teacher} {case_index}",
        "proposal_num_tokens": 4,
        "replacement_num_tokens": 2,
        "finish_reason": "stop",
        "truncated": 0,
    }


def _student_record(
    case_index: int,
    arm: str,
    correct_count: int,
    *,
    text_tag: str = "first",
) -> dict:
    rewards = [1] * correct_count + [0] * (MC_K - correct_count)
    return {
        "ok": True,
        "case_id": f"case-{case_index}",
        "case_index": case_index,
        "arm": arm,
        "mc_k": MC_K,
        "correct_count": correct_count,
        "value": correct_count / MC_K,
        "truncated_count": 0,
        "format_invalid_count": 0,
        "rewards": rewards,
        "texts": [f"{text_tag}-{arm}-{index}" for index in range(MC_K)],
    }


def _shard(role: str, records: list[dict], batch_index: int = 0) -> dict:
    return {
        "type": "batch",
        "role": role,
        "rank": 0,
        "gpu": 0,
        "batch_index": batch_index,
        "records": records,
        "completed_items": len(records),
        "generated_rollouts": len(records) * (MC_K if role == "student" else 1),
        "elapsed_seconds": 1.0,
        "memory": {"total": 100, "used": 50, "free": 50},
    }


def _make_run(root: Path, *, with_duplicate: bool = True) -> None:
    config = {
        "teachers": [{"name": name} for name in TEACHERS],
        "generation": {"mc_k": MC_K},
    }
    selection = [_selection_row(0), _selection_row(1)]
    base.atomic_json(root / "run_config.json", config)
    base.atomic_json(root / "selection_manifest.json", selection)
    for teacher in TEACHERS:
        records = [_teacher_record(index, teacher) for index in range(2)]
        base.atomic_json(
            root
            / "teachers"
            / teacher
            / "shards"
            / "gpu_0_batch_000000.json",
            _shard("teacher", records),
        )
    counts = {
        "redecide": (1, 2),
        "keep": (2, 1),
        "replace:teacher_a": (3, 3),
        "replace:teacher_b": (1, 2),
    }
    records = [
        _student_record(case_index, arm, per_case[case_index])
        for case_index in range(2)
        for arm, per_case in counts.items()
    ]
    base.atomic_json(
        root / "student" / "shards" / "gpu_0_batch_000000.json",
        _shard("student", records),
    )
    if with_duplicate:
        # Same result-bearing fields, intentionally different raw texts.  This
        # must be safe because texts are not copied into ersr_results.json.
        duplicate = _student_record(0, "redecide", 1, text_tag="resumed")
        base.atomic_json(
            root / "student" / "shards" / "gpu_0_batch_000001.json",
            _shard("student", [duplicate], batch_index=1),
        )


class FinalizerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = REPO_ROOT / "tmp" / self.id().replace(".", "_")
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


class TestSuccessfulFinalization(FinalizerTestCase):
    def test_exact_duplicate_is_removed_and_final_outputs_are_built(self) -> None:
        _make_run(self.root)
        source_shards_before = sorted(
            path.relative_to(self.root)
            for path in self.root.glob("**/shards/*.json")
        )

        audit = finalizer.finalize(self.root)

        self.assertEqual(audit["status"], "complete")
        student = audit["observed"]["student"]
        self.assertEqual(student["raw_records"], 9)
        self.assertEqual(student["unique_case_arm_records"], 8)
        self.assertEqual(student["duplicate_records_removed"], 1)
        self.assertEqual(student["conflicting_duplicate_groups"], 0)
        self.assertEqual(student["duplicate_groups_with_different_texts_only"], 1)
        results = base.load_json(self.root / "ersr_results.json")
        summary = base.load_json(self.root / "ersr_summary.json")
        persisted_audit = base.load_json(self.root / "finalization_audit.json")
        self.assertEqual(len(results), 2)
        self.assertEqual(summary["status"], "complete")
        self.assertEqual(summary["cases"], 2)
        self.assertEqual(persisted_audit["status"], "complete")
        self.assertEqual(results[0]["values"]["redecide"], 0.25)
        self.assertEqual(results[0]["values"]["keep"], 0.5)
        self.assertEqual(results[0]["advantages"]["A_k"], 0.25)
        source_shards_after = sorted(
            path.relative_to(self.root)
            for path in self.root.glob("**/shards/*.json")
        )
        self.assertEqual(source_shards_after, source_shards_before)

    def test_audit_only_writes_no_final_results(self) -> None:
        _make_run(self.root)
        audit = finalizer.finalize(self.root, audit_only=True)
        self.assertEqual(audit["status"], "audit_complete")
        self.assertFalse((self.root / "ersr_results.json").exists())
        self.assertFalse((self.root / "ersr_summary.json").exists())
        self.assertTrue((self.root / "finalization_audit.json").is_file())

    def test_existing_results_require_force(self) -> None:
        _make_run(self.root)
        base.atomic_json(self.root / "ersr_results.json", {"sentinel": True})
        with self.assertRaises(finalizer.FinalizationError):
            finalizer.finalize(self.root)
        sentinel = base.load_json(self.root / "ersr_results.json")
        self.assertEqual(sentinel, {"sentinel": True})


class TestBlockedFinalization(FinalizerTestCase):
    def test_conflicting_duplicate_blocks_and_reports_fields(self) -> None:
        _make_run(self.root, with_duplicate=False)
        conflicting = _student_record(0, "redecide", 2, text_tag="conflict")
        base.atomic_json(
            self.root / "student" / "shards" / "gpu_0_batch_000001.json",
            _shard("student", [conflicting], batch_index=1),
        )

        with self.assertRaises(finalizer.FinalizationError):
            finalizer.finalize(self.root)

        audit = base.load_json(self.root / "finalization_audit.json")
        self.assertEqual(audit["status"], "blocked")
        self.assertEqual(
            audit["observed"]["student"]["conflicting_duplicate_groups"], 1
        )
        differing = audit["observed"]["student"]["conflict_examples"][0][
            "differing_fields"
        ]
        self.assertIn("correct_count", differing)
        self.assertIn("rewards", differing)
        self.assertFalse((self.root / "ersr_results.json").exists())

    def test_missing_arm_blocks_finalization(self) -> None:
        _make_run(self.root, with_duplicate=False)
        shard_path = self.root / "student" / "shards" / "gpu_0_batch_000000.json"
        shard = base.load_json(shard_path)
        shard["records"] = shard["records"][:-1]
        shard["completed_items"] = len(shard["records"])
        base.atomic_json(shard_path, shard)

        with self.assertRaises(finalizer.FinalizationError):
            finalizer.finalize(self.root)

        audit = json.loads(
            (self.root / "finalization_audit.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            audit["observed"]["student"]["missing_case_arm_records"], 1
        )
        self.assertFalse((self.root / "ersr_results.json").exists())


if __name__ == "__main__":
    unittest.main()
