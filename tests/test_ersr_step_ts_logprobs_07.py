from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace

from tests import collect_ersr_step_ts_logprobs_07 as collector


REPO_ROOT = Path(__file__).resolve().parents[1]


def ersr_record(case_index: int, *, start: int, end: int, rollout: int = 0) -> dict:
    return {
        "case_id": f"7:{rollout}:{case_index}",
        "case_index": case_index,
        "response_idx": 7,
        "rollout_index": rollout,
        "rollout_correct": rollout,
        "bucket": 3,
        "step_index": start // 2,
        "reasoning_step_index": 1,
        "num_reasoning_steps": 4,
        "step_token_start": start,
        "step_token_end": end,
        "prompt_token_ids": [1, 2, 3],
        "response_token_ids": [10, 11, 12, 13, 14, 15, 16],
        "teacher_replacements": {"t1": {"text": "x"}, "t2": {"text": "y"}},
        "values": {
            "redecide": 0.25,
            "keep": 0.375,
            "replace:t1": 0.5,
            "replace:t2": 0.125,
        },
        "advantages": {
            "A_k": 0.125,
            "A_k_TR:t1": 0.25,
            "A_k_TR:t2": -0.125,
        },
    }


class TestTaskConstruction(unittest.TestCase):
    def test_deduplicates_shared_trajectory(self) -> None:
        records = [
            ersr_record(0, start=2, end=4),
            ersr_record(1, start=4, end=6),
        ]
        tasks = collector.build_trajectory_tasks(records)
        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["trajectory_id"], "7:0")
        self.assertEqual(
            [case["case_id"] for case in tasks[0]["cases"]], ["7:0:0", "7:0:1"]
        )

    def test_rejects_inconsistent_trajectory_ids(self) -> None:
        first = ersr_record(0, start=2, end=4, rollout=0)
        second = ersr_record(1, start=4, end=6, rollout=0)
        second["response_token_ids"] = [10, 11, 12, 13, 14, 15, 99]
        with self.assertRaises(ValueError):
            collector.build_trajectory_tasks([first, second])

    def test_rejects_empty_step_window(self) -> None:
        with self.assertRaises(ValueError):
            collector.build_trajectory_tasks([ersr_record(0, start=4, end=4)])


class TestExtraction(unittest.TestCase):
    def test_window_uses_prompt_offset(self) -> None:
        prompt_ids = [1, 2, 3]
        response_ids = [10, 11, 12, 13, 14]
        full_ids = prompt_ids + response_ids
        prompt_logprobs = [
            None if position == 0
            else {full_ids[position]: SimpleNamespace(logprob=-0.5 * position)}
            for position in range(len(full_ids))
        ]
        values = collector.extract_step_logprob_values(
            full_ids,
            prompt_logprobs,
            prompt_len=len(prompt_ids),
            start=1,
            end=4,
        )
        self.assertEqual(values, [-0.5 * 4, -0.5 * 5, -0.5 * 6])

    def test_missing_token_raises(self) -> None:
        full_ids = [1, 2, 3]
        prompt_logprobs = [None, {1: SimpleNamespace(logprob=0.0)}, {}]
        with self.assertRaises(RuntimeError):
            collector.extract_step_logprob_values(
                full_ids, prompt_logprobs, prompt_len=1, start=1, end=2
            )

    def test_summarize_step(self) -> None:
        summary = collector.summarize_step([-1.0, -3.0])
        self.assertEqual(summary["num_tokens"], 2)
        self.assertEqual(summary["sum_logprob"], -4.0)
        self.assertEqual(summary["mean_logprob"], -2.0)


class TestMerge(unittest.TestCase):
    def setUp(self) -> None:
        self.root = REPO_ROOT / "tmp" / self.id().replace(".", "_")
        shutil.rmtree(self.root, ignore_errors=True)
        self.output_dir = self.root / "step_ts_07"
        self.shard_dir = self.output_dir / "shards"
        self.shard_dir.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_shard(self, name: str, role: str, rows: list[dict]) -> None:
        payload = {
            "type": "batch",
            "role": role,
            "records": rows,
            "trajectory_ids": sorted({row["trajectory_id"] for row in rows}),
        }
        (self.shard_dir / name).write_text(json.dumps(payload), encoding="utf-8")

    def test_merge_produces_d_ts_and_ersr_join(self) -> None:
        records = [ersr_record(0, start=2, end=4), ersr_record(1, start=4, end=6)]
        student_rows = [
            {
                "case_id": "7:0:0",
                "trajectory_id": "7:0",
                "mean_logprob": -2.0,
                "sum_logprob": -4.0,
                "num_tokens": 2,
            },
            {
                "case_id": "7:0:1",
                "trajectory_id": "7:0",
                "mean_logprob": -1.0,
                "sum_logprob": -2.0,
                "num_tokens": 2,
            },
        ]
        teacher_rows = [
            {
                "case_id": "7:0:0",
                "trajectory_id": "7:0",
                "mean_logprob": -1.5,
                "sum_logprob": -3.0,
                "num_tokens": 2,
            },
            {
                "case_id": "7:0:1",
                "trajectory_id": "7:0",
                "mean_logprob": -2.5,
                "sum_logprob": -5.0,
                "num_tokens": 2,
            },
        ]
        self._write_shard("gpu_0_student_batch_000000.json", "student", student_rows)
        self._write_shard("gpu_0_t1_batch_000000.json", "t1", teacher_rows)
        rows = collector.merge_all(
            output_dir=self.output_dir,
            ersr_records=records,
            student_role_tag="student",
            teacher_role_tags={"t1": "t1"},
        )
        self.assertEqual([row["case_index"] for row in rows], [0, 1])
        first = rows[0]
        self.assertEqual(first["teachers"]["t1"]["D_TS_mean"], 0.5)
        self.assertEqual(first["teachers"]["t1"]["D_TS_sum"], 1.0)
        self.assertEqual(first["ersr"]["A_student"], 0.125)
        self.assertEqual(first["ersr"]["A_teacher"]["t1"], 0.25)
        self.assertEqual(first["ersr"]["V_replace"]["t1"], 0.5)
        self.assertAlmostEqual(first["relative_position"], 0.5)
        self.assertEqual(first["step_num_tokens"], 2)

    def test_merge_rejects_missing_teacher(self) -> None:
        records = [ersr_record(0, start=2, end=4)]
        self._write_shard(
            "gpu_0_student_batch_000000.json",
            "student",
            [
                {
                    "case_id": "7:0:0",
                    "trajectory_id": "7:0",
                    "mean_logprob": -2.0,
                    "sum_logprob": -4.0,
                    "num_tokens": 2,
                }
            ],
        )
        with self.assertRaises(ValueError):
            collector.merge_all(
                output_dir=self.output_dir,
                ersr_records=records,
                student_role_tag="student",
                teacher_role_tags={"t1": "t1"},
            )


class TestShardResume(unittest.TestCase):
    def test_next_shard_index_and_completed_trajectories(self) -> None:
        root = REPO_ROOT / "tmp" / self.id().replace(".", "_")
        shutil.rmtree(root, ignore_errors=True)
        shard_dir = root / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        try:
            (shard_dir / "gpu_0_student_batch_000000.json").write_text(
                json.dumps({"trajectory_ids": ["7:0"]}), encoding="utf-8"
            )
            (shard_dir / "gpu_0_student_batch_000003.json").write_text(
                json.dumps({"trajectory_ids": ["8:0"]}), encoding="utf-8"
            )
            self.assertEqual(collector.next_shard_index(shard_dir, 0, "student"), 4)
            self.assertEqual(
                collector.completed_trajectories(shard_dir, "student"), {"7:0", "8:0"}
            )
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
