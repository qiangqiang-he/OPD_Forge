from __future__ import annotations

import json
import math
import os
import shutil
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tests import analyze_ersr_answer_probes_03 as analysis
from tests import collect_ersr_answer_probes_03 as collector


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [1000 + ord(character) for character in text]

    def decode(
        self,
        token_ids: list[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(value - 1000) for value in token_ids)


def fake_probe_builder(tokenizer, answer: str):
    del tokenizer
    text = f"probe:{answer}"
    return SimpleNamespace(
        text=text,
        token_ids=[10, 11, 12],
        answer_token_positions=[1],
        answer_char_start=6,
        answer_char_end=len(text),
    )


def ersr_record(case_index: int) -> dict:
    mc_k = 10
    rd = 4
    return {
        "case_id": f"case-{case_index}",
        "case_index": case_index,
        "response_idx": case_index // 2,
        "rollout_index": case_index % 2,
        "bucket": 3,
        "rollout_correct": case_index % 2,
        "answer": "42",
        "step_index": 2,
        "reasoning_step_index": 1,
        "num_reasoning_steps": 4,
        "step_token_start": 2,
        "step_token_end": 4,
        "prompt_token_ids": [1, 2, 3],
        "response_token_ids": [20, 21, 22, 23, 24],
        "samples": {
            "redecide": {"mc_k": mc_k, "correct_count": rd},
            "keep": {"mc_k": mc_k, "correct_count": rd + 1},
            "replace:t1": {"mc_k": mc_k, "correct_count": rd + 2},
            "replace:t2": {"mc_k": mc_k, "correct_count": rd - 1},
        },
        "teacher_replacements": {
            "t1": {"text": "teacher one", "truncated": 0},
            "t2": {"text": "teacher two", "truncated": 1},
        },
    }


def probe_score(probability: float) -> dict:
    mean = math.log(probability)
    return {
        "answer_token_logprobs": [mean],
        "answer_token_ranks": [1],
        "sum_logprob": mean,
        "mean_logprob": mean,
        "geometric_mean_probability": probability,
    }


def synthetic_step_scores() -> list[dict]:
    rows: list[dict] = []
    case_index = 0
    for prompt in range(8):
        for reward in (1, 0):
            trajectory_id = f"p{prompt}:r{reward}"
            for step in range(4):
                before = 0.24 + prompt * 0.012 + step * 0.003
                nonlinear = ((prompt + 2 * step) % 5 - 2) * 0.002
                student_gain = -0.07 + step * 0.045 + nonlinear
                a_student = 0.015 + 1.35 * student_gain + prompt * 0.001
                student_after = before + student_gain
                v_before = 0.36 + prompt * 0.009 + (step % 2) * 0.012
                teacher_rows = {}
                for teacher_index, teacher in enumerate(("t1", "t2")):
                    teacher_gain = (
                        -0.08
                        + step * 0.052
                        + nonlinear
                        + teacher_index * 0.008
                    )
                    a_teacher = 0.01 + 1.55 * teacher_gain + prompt * 0.0015
                    teacher_probability = before + teacher_gain
                    teacher_rows[teacher] = {
                        "replacement_num_tokens": 3,
                        "replacement_token_ids_sha256": "x",
                        "teacher_truncated": 0,
                        "V_teacher_after": v_before + a_teacher,
                        "A_teacher": a_teacher,
                        "probe": probe_score(teacher_probability),
                        "P_teacher_after": teacher_probability,
                        "deltaP_teacher": teacher_gain,
                        "delta_logP_teacher": math.log(teacher_probability)
                        - math.log(before),
                    }
                rows.append(
                    {
                        "schema_version": 1,
                        "case_id": f"case-{case_index}",
                        "case_index": case_index,
                        "prompt_id": prompt,
                        "trajectory_id": trajectory_id,
                        "response_idx": prompt,
                        "rollout_index": reward,
                        "reward": reward,
                        "bucket": 3,
                        "prompt_accuracy": 0.6,
                        "step_id": step,
                        "step_ordinal": step + 1,
                        "source_step_index": step,
                        "num_steps": 4,
                        "relative_position": (step + 1) / 4,
                        "mc_k": 128,
                        "probe_text": "probe:42",
                        "V_before": v_before,
                        "V_student_after": v_before + a_student,
                        "A_student": a_student,
                        "probe_before": probe_score(before),
                        "probe_student_after": probe_score(student_after),
                        "P_before": before,
                        "P_student_after": student_after,
                        "deltaP_student": student_gain,
                        "delta_logP_student": math.log(student_after)
                        - math.log(before),
                        "teachers": teacher_rows,
                    }
                )
                case_index += 1
    return rows


class ScratchTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = REPO_ROOT / "tmp" / self.id().replace(".", "_")
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


class TestCollectorPreparation(ScratchTestCase):
    def test_server_default_has_no_reserved_free_memory(self) -> None:
        with mock.patch.object(sys, "argv", ["collect_ersr_answer_probes_03.py"]):
            args = collector.parse_args()
        self.assertEqual(args.min_free_gpu_gib, 0.0)

    def test_exact_prefix_and_ersr_values(self) -> None:
        task = collector.prepare_task(
            ersr_record(1), FakeTokenizer(), probe_builder=fake_probe_builder
        )
        self.assertEqual(task["pre_step_response_token_ids"], [20, 21])
        self.assertEqual(task["student_step_token_ids"], [22, 23])
        self.assertEqual(task["relative_position"], 0.5)
        self.assertEqual(task["prompt_accuracy"], 0.6)
        self.assertAlmostEqual(task["V_before"], 0.4)
        self.assertAlmostEqual(task["A_student"], 0.1)
        self.assertAlmostEqual(task["teachers"]["t1"]["A_teacher"], 0.2)
        self.assertAlmostEqual(task["teachers"]["t2"]["A_teacher"], -0.1)
        expected_ids = FakeTokenizer().encode("teacher one")
        self.assertEqual(task["teachers"]["t1"]["replacement_token_ids"], expected_ids)

    def test_preparation_shards_every_requested_gpu(self) -> None:
        results_path = self.root / "ersr_results.json"
        results_path.write_text(
            json.dumps([ersr_record(index) for index in range(8)]), encoding="utf-8"
        )
        manifest = collector.prepare_task_shards(
            results_path=results_path,
            model_path=self.root / "not-needed-with-injected-tokenizer",
            output_dir=self.root / "output",
            gpu_ids=list(range(8)),
            tokenizer=FakeTokenizer(),
            probe_builder=fake_probe_builder,
        )
        self.assertEqual(manifest["gpu_ids"], list(range(8)))
        self.assertEqual(manifest["selected_cases"], 8)
        self.assertEqual(len(manifest["task_shards"]), 8)
        self.assertTrue(all(value == 1 for value in manifest["task_shard_case_counts"].values()))
        self.assertIn("no WORLD_SIZE", manifest["gpu_policy"])

    def test_auto_gpu_discovery_ignores_world_size(self) -> None:
        completed = mock.Mock(stdout="0\n1\n2\n3\n4\n5\n6\n7\n")
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "999"}), mock.patch.object(
            collector.subprocess, "run", return_value=completed
        ) as run:
            self.assertEqual(collector.discover_local_gpus("auto"), list(range(8)))
        command = run.call_args.args[0]
        self.assertEqual(command[0], "nvidia-smi")


class TestAnalysis(ScratchTestCase):
    def test_end_to_end_outputs_and_positive_relationships(self) -> None:
        scores_path = self.root / "03_probe_step_scores.jsonl"
        with scores_path.open("w", encoding="utf-8") as handle:
            for row in synthetic_step_scores():
                handle.write(json.dumps(row) + "\n")
        output_dir = self.root / "analysis"
        argv = [
            "analyze_ersr_answer_probes_03.py",
            "--step-scores",
            str(scores_path),
            "--output-dir",
            str(output_dir),
            "--bootstrap-rounds",
            "50",
            "--seed",
            "17",
        ]
        with mock.patch.object(sys, "argv", argv):
            analysis.main()

        required = {
            "03_probe_step_data.csv",
            "03_probe_global_correlations.csv",
            "03_probe_within_trajectory_correlations.csv",
            "03_probe_quartile_stats.csv",
            "03_probe_extreme_groups.csv",
            "03_probe_sign_stats.csv",
            "03_probe_regression_robustness.csv",
            "03_probe_analysis.json",
            "03_probe_output_manifest.json",
        }
        self.assertTrue(required.issubset({path.name for path in output_dir.iterdir()}))
        report = json.loads((output_dir / "03_probe_analysis.json").read_text())
        correct = report["scopes"]["correct_student|student"]
        incorrect = report["scopes"]["incorrect_teacher|pooled"]
        self.assertGreater(correct["global_correlation"]["pearson"]["r"], 0.9)
        self.assertGreater(
            correct["within_trajectory_correlation"]["pearson"]["r"], 0.9
        )
        self.assertGreater(incorrect["global_correlation"]["pearson"]["r"], 0.9)
        self.assertGreater(
            incorrect["within_trajectory_correlation"]["pearson"]["r"], 0.9
        )
        self.assertTrue(
            correct["quartiles"]["mean_utility_monotonic_nondecreasing"]
        )
        self.assertGreater(
            correct["extreme_groups"]["top_minus_bottom"]["difference"], 0.0
        )
        self.assertEqual(report["input"]["correct_student_steps"], 8 * 4)
        self.assertEqual(report["input"]["incorrect_teacher_interventions"], 8 * 4 * 2)

        with (output_dir / "03_probe_step_data.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            csv_rows = list(__import__("csv").DictReader(handle))
        self.assertEqual(len(csv_rows), len(synthetic_step_scores()) * 2)
        self.assertTrue(set(analysis.STEP_DATA_FIELDS).issubset(csv_rows[0]))


if __name__ == "__main__":
    unittest.main()
