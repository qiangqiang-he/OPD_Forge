from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from tests import analyze_ersr_low_budget_mc_04 as analysis


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_K = 16
TEACHERS = ("t1", "t2")


def rewards(correct: int) -> list[int]:
    # Interleave outcomes so the fixture resembles ordinary rollout records.
    values = [1] * correct + [0] * (REFERENCE_K - correct)
    return [values[(index * 5) % REFERENCE_K] for index in range(REFERENCE_K)]


def sample(correct: int) -> dict:
    values = rewards(correct)
    return {
        "mc_k": REFERENCE_K,
        "correct_count": sum(values),
        "truncated_count": 0,
        "format_invalid_count": 0,
        "rewards": values,
    }


def fixture_results() -> list[dict]:
    rows: list[dict] = []
    case_index = 0
    for prompt in range(6):
        bucket = 2 + (prompt % 2)
        for rollout in range(5):
            reward = int(rollout < bucket)
            for step in range(2):
                before_count = 4 + ((prompt + rollout + step) % 5)
                if reward:
                    keep_count = min(REFERENCE_K, before_count + 3)
                    teacher_counts = {"t1": before_count + 1, "t2": before_count}
                else:
                    keep_count = max(0, before_count - 1)
                    teacher_counts = {"t1": before_count + 4, "t2": before_count + 2}
                teacher_counts = {
                    name: min(REFERENCE_K, max(0, value))
                    for name, value in teacher_counts.items()
                }
                samples = {
                    "redecide": sample(before_count),
                    "keep": sample(keep_count),
                    **{
                        f"replace:{name}": sample(value)
                        for name, value in teacher_counts.items()
                    },
                }
                before_value = before_count / REFERENCE_K
                rows.append(
                    {
                        "schema_version": 1,
                        "case_id": f"{prompt}:{rollout}:{step}",
                        "case_index": case_index,
                        "response_idx": prompt,
                        "rollout_index": rollout,
                        "bucket": bucket,
                        "rollout_correct": reward,
                        "reasoning_step_index": step,
                        "num_reasoning_steps": 2,
                        "samples": samples,
                        "advantages": {
                            "A_k": keep_count / REFERENCE_K - before_value,
                            **{
                                f"A_k_TR:{name}": value / REFERENCE_K - before_value
                                for name, value in teacher_counts.items()
                            },
                        },
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


class TestValidationAndSampling(ScratchTestCase):
    def test_missing_individual_rewards_is_rejected(self) -> None:
        results = fixture_results()[:1]
        del results[0]["samples"]["keep"]["rewards"]
        path = self.root / "ersr_results.json"
        path.write_text(json.dumps(results), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "individual rollout rewards"):
            analysis.load_step_dataset(path, reference_k=REFERENCE_K)

    def test_reference_k_draw_is_exact(self) -> None:
        counts = np.asarray([0, 3, 9, REFERENCE_K])
        drawn = analysis.draw_low_budget_counts(
            counts,
            reference_k=REFERENCE_K,
            k=REFERENCE_K,
            rng=np.random.default_rng(3),
        )
        np.testing.assert_array_equal(drawn, counts)

    def test_prompt_cluster_keeps_whole_clusters(self) -> None:
        prompts = np.asarray(["a", "a", "a", "b", "b", "c", "c"], dtype=object)
        order, actual = analysis.nested_sampling_plan(
            population_size=len(prompts),
            prompt_ids=prompts,
            m_values=[2, 4],
            setting="prompt_cluster",
            rng=np.random.default_rng(8),
        )
        self.assertGreaterEqual(actual[2], 2)
        self.assertGreaterEqual(actual[4], 4)
        for count in actual.values():
            selected = prompts[order[:count]]
            for prompt in set(selected):
                self.assertEqual(np.count_nonzero(selected == prompt), np.count_nonzero(prompts == prompt))


class TestEndToEnd(ScratchTestCase):
    def test_all_04_outputs_and_zero_error_at_reference_k(self) -> None:
        results_path = self.root / "ersr_results.json"
        results_path.write_text(json.dumps(fixture_results()), encoding="utf-8")
        output_dir = self.root / "analysis"
        argv = [
            "analyze_ersr_low_budget_mc_04.py",
            "--results",
            str(results_path),
            "--output-dir",
            str(output_dir),
            "--reference-k",
            str(REFERENCE_K),
            "--k-values",
            f"2,4,{REFERENCE_K}",
            "--m-values",
            "5,10,20",
            "--repetitions",
            "20",
            "--individual-repetitions",
            "10",
            "--sampling-settings",
            "prompt_cluster,step_random",
            "--seed",
            "9",
        ]
        with mock.patch.object(sys, "argv", argv):
            analysis.main()

        required = {
            "04_mc_step_rollouts.jsonl",
            "04_mc_individual_step_accuracy.csv",
            "04_mc_population_mean_grid.csv",
            "04_mc_outcome_recovery_grid.csv",
            "04_mc_analysis.json",
            "04_mc_output_manifest.json",
        }
        self.assertEqual(required, {path.name for path in output_dir.iterdir()})
        report = json.loads((output_dir / "04_mc_analysis.json").read_text())
        self.assertEqual(report["input"]["steps"], len(fixture_results()))
        self.assertEqual(report["input"]["teachers"], list(TEACHERS))
        self.assertEqual(set(report["configuration"]["sampling_settings"]), {"prompt_cluster", "step_random"})

        with (output_dir / "04_mc_population_mean_grid.csv").open(encoding="utf-8") as handle:
            population_rows = list(__import__("csv").DictReader(handle))
        self.assertEqual(
            {row["reference_type"] for row in population_rows},
            {"same_step_mc128", "full_population_mc128"},
        )
        self.assertTrue({"correct_A", "correct_ATR", "incorrect_A", "incorrect_ATR"}.issubset({row["target"] for row in population_rows}))
        exact_rows = [
            row
            for row in population_rows
            if int(row["K"]) == REFERENCE_K
            and row["reference_type"] == "same_step_mc128"
        ]
        self.assertTrue(exact_rows)
        self.assertTrue(all(float(row["MAE"]) < 1e-12 for row in exact_rows))

        first_raw = json.loads((output_dir / "04_mc_step_rollouts.jsonl").read_text().splitlines()[0])
        self.assertEqual(len(first_raw["before_rollouts"]), REFERENCE_K)
        self.assertEqual(set(first_raw["teacher_rollouts"]), set(TEACHERS))


if __name__ == "__main__":
    unittest.main()
