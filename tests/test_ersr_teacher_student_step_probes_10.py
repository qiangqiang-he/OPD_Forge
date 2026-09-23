from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from tests import collect_ersr_teacher_student_step_probes_10 as collector
from tests.collect_ersr_answer_probes_03 import prepare_task
from tests.test_ersr_answer_probe_03 import FakeTokenizer, ersr_record, fake_probe_builder


REPO_ROOT = Path(__file__).resolve().parents[1]


def prepared_task(case_index: int) -> dict:
    return prepare_task(
        ersr_record(case_index), FakeTokenizer(), probe_builder=fake_probe_builder
    )


class TestProbeSequences(unittest.TestCase):
    def test_two_states_around_the_student_step(self) -> None:
        task = prepared_task(1)
        sequences = collector.build_probe_sequences(task)
        self.assertEqual([entry["kind"] for entry in sequences], ["before", "student_after"])
        prompt = task["prompt_token_ids"]
        pre = task["pre_step_response_token_ids"]
        step = task["student_step_token_ids"]
        probe = task["probe_token_ids"]
        self.assertEqual(sequences[0]["input_ids"], prompt + pre + probe)
        self.assertEqual(sequences[1]["input_ids"], prompt + pre + step + probe)
        base = len(prompt) + len(pre)
        self.assertEqual(
            sequences[0]["answer_positions"],
            [base + position for position in task["probe_answer_token_positions"]],
        )
        self.assertEqual(
            sequences[1]["answer_positions"],
            [base + len(step) + position for position in task["probe_answer_token_positions"]],
        )


class TestRecordAndMerge(unittest.TestCase):
    def setUp(self) -> None:
        self.root = REPO_ROOT / "tmp" / self.id().replace(".", "_")
        shutil.rmtree(self.root, ignore_errors=True)
        self.output_dir = self.root / "probes_10"
        (self.output_dir / "shards").mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def _write_shard(self, name: str, role: str, rows: list[dict]) -> None:
        payload = {
            "type": "batch",
            "role": role,
            "records": rows,
            "case_ids": [row["case_id"] for row in rows],
        }
        (self.output_dir / "shards" / name).write_text(json.dumps(payload), encoding="utf-8")

    def test_merge_joins_teachers_and_ersr_values(self) -> None:
        task = prepared_task(0)
        row = {
            "schema_version": 1,
            "role": "t1",
            "case_id": task["case_id"],
            "case_index": task["case_index"],
            "trajectory_id": task["trajectory_id"],
            "probe": {},
            "P_before": 0.2,
            "P_student_after": 0.35,
            "deltaP_T_student_step": 0.15000000000000002,
            "delta_logP_T_student_step": 0.5,
        }
        self._write_shard("gpu_0_t1_batch_000000.json", "t1", [row])
        rows = collector.merge_all(
            output_dir=self.output_dir,
            tasks=[task],
            teacher_role_tags={"t1": "t1"},
        )
        merged = rows[0]
        self.assertAlmostEqual(
            merged["teachers"]["t1"]["deltaP_T_student_step"], 0.15, places=10
        )
        self.assertEqual(merged["teachers"]["t1"]["P_student_after"], 0.35)
        self.assertAlmostEqual(merged["teachers"]["t1"]["A_teacher"], 0.2)
        self.assertAlmostEqual(merged["student"]["A_student"], 0.1)
        self.assertEqual(merged["reward"], task["reward"])

    def test_merge_rejects_missing_teacher(self) -> None:
        task = prepared_task(0)
        with self.assertRaises(ValueError):
            collector.merge_all(
                output_dir=self.output_dir,
                tasks=[task],
                teacher_role_tags={"t1": "t1"},
            )


class TestShardResume(unittest.TestCase):
    def test_next_shard_index_and_completed_cases(self) -> None:
        root = REPO_ROOT / "tmp" / self.id().replace(".", "_")
        shutil.rmtree(root, ignore_errors=True)
        shard_dir = root / "shards"
        shard_dir.mkdir(parents=True, exist_ok=True)
        try:
            (shard_dir / "gpu_0_t1_batch_000000.json").write_text(
                json.dumps({"case_ids": ["a", "b"]}), encoding="utf-8"
            )
            (shard_dir / "gpu_0_t1_batch_000002.json").write_text(
                json.dumps({"case_ids": ["c"]}), encoding="utf-8"
            )
            self.assertEqual(collector.next_shard_index(shard_dir, 0, "t1"), 3)
            self.assertEqual(collector.completed_cases(shard_dir, "t1"), {"a", "b", "c"})
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
