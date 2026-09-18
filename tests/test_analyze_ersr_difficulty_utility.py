from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

from tests import analyze_ersr_difficulty_utility as analysis


TEACHERS = ("t1", "t2")
MC_K = 128


def _clamp_count(value: float) -> int:
    return max(0, min(MC_K, int(round(value))))


def _synthetic_results() -> list[dict]:
    """Utility decreases linearly with bucket so every trend check is stable.

    Layout: 6 buckets x 3 prompts x 5 rollouts (rollout < bucket is correct)
    x 1-2 steps.  Teacher t1 carries truncated flags on every 17th case.
    """

    records: list[dict] = []
    case_index = 0
    for bucket in range(6):
        for prompt in range(3):
            response_idx = bucket * 100 + prompt
            for rollout in range(5):
                correct = 1 if rollout < bucket else 0
                n_steps = 1 + (rollout % 2)
                for step in range(n_steps):
                    noise = ((case_index * 37) % 11 - 5) / (MC_K * 4)
                    a_value = 0.30 - 0.24 * (bucket / 5) + noise
                    ti1_value = a_value + 0.06 - 0.10 * (bucket / 5) + 0.5 * noise
                    ti2_value = a_value + 0.02 - 0.08 * (bucket / 5) - 0.25 * noise
                    rd = 40 + (case_index % 5)
                    kp = _clamp_count(rd + a_value * MC_K)
                    rp1 = _clamp_count(rd + ti1_value * MC_K)
                    rp2 = _clamp_count(rd + ti2_value * MC_K)
                    truncated1 = 1 if case_index % 17 == 0 else 0
                    records.append(
                        {
                            "schema_version": 1,
                            "case_id": f"{response_idx}:{rollout}:{step}",
                            "case_index": case_index,
                            "response_idx": response_idx,
                            "rollout_index": rollout,
                            "bucket": bucket,
                            "rollout_correct": correct,
                            "question": f"q{response_idx}",
                            "answer": "1",
                            "step_index": step,
                            "num_reasoning_steps": 3,
                            "position_fraction": 0.5,
                            "samples": {
                                "redecide": {"mc_k": MC_K, "correct_count": rd},
                                "keep": {"mc_k": MC_K, "correct_count": kp},
                                "replace:t1": {"mc_k": MC_K, "correct_count": rp1},
                                "replace:t2": {"mc_k": MC_K, "correct_count": rp2},
                            },
                            "teacher_replacements": {
                                "t1": {"truncated": truncated1},
                                "t2": {"truncated": 0},
                            },
                            "advantages": {
                                "A_k": (kp - rd) / MC_K,
                                "A_k_TR:t1": (rp1 - rd) / MC_K,
                                "A_k_TR:t2": (rp2 - rd) / MC_K,
                            },
                        }
                    )
                    case_index += 1
    return records


def _extract():
    return analysis.extract_rows(_synthetic_results(), 5)


class TestExtractRows(unittest.TestCase):
    def test_counts_and_teachers(self) -> None:
        rows, teachers = _extract()
        self.assertEqual(teachers, list(TEACHERS))
        # 6 buckets x 3 prompts x (1+2+1+2+1) steps = 126.
        self.assertEqual(len(rows), 126)
        self.assertEqual(len({row["trajectory"] for row in rows}), 6 * 3 * 5)
        self.assertEqual(len({row["response_idx"] for row in rows}), 18)
        # Correct steps per prompt per bucket b: sum(steps(r) for r < b) with
        # steps = (1, 2, 1, 2, 1) for rollouts 0..4 -> 0,1,3,4,6,7.
        self.assertEqual(
            sum(row["R"] for row in rows), 3 * sum((0, 1, 3, 4, 6, 7))
        )

    def test_a_is_exact_fraction(self) -> None:
        rows, _ = _extract()
        records = _synthetic_results()
        for row, record in zip(rows, records):
            samples = record["samples"]
            expected = (
                samples["keep"]["correct_count"] - samples["redecide"]["correct_count"]
            ) / MC_K
            self.assertEqual(row["A"], expected)
            self.assertEqual(row["A_TI"]["t1"], (samples["replace:t1"]["correct_count"] - samples["redecide"]["correct_count"]) / MC_K)

    def test_bucket_outside_range_raises(self) -> None:
        records = _synthetic_results()
        records[0]["bucket"] = 6
        with self.assertRaises(ValueError):
            analysis.extract_rows(records, 5)


class TestSections(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.rows, cls.teachers = _extract()
        cls.metrics = analysis.metric_names(cls.teachers)
        cls.sections = {
            subset: analysis.subset_section(cls.rows, subset, cls.metrics, 5)
            for subset in analysis.SUBSETS
        }

    def test_structural_empty_cells(self) -> None:
        self.assertEqual(
            self.sections["correct"]["groups"]["0/5"]["A"]["mean"], None
        )
        self.assertEqual(
            self.sections["incorrect"]["groups"]["5/5"]["A_TI:t1"]["mean"], None
        )
        # Every non-structural cell is populated.
        for subset in analysis.SUBSETS:
            for key, group in self.sections[subset]["groups"].items():
                for metric in self.metrics:
                    if (subset, key) in {("correct", "0/5"), ("incorrect", "5/5")}:
                        continue
                    self.assertIsNotNone(group[metric]["mean"], (subset, key, metric))

    def test_basic_stats(self) -> None:
        basic = analysis.basic_stats_section(self.rows, 5)
        self.assertEqual(basic["0/5"]["correct_trajectory_count"], 0)
        self.assertEqual(basic["0/5"]["incorrect_trajectory_count"], 15)
        self.assertEqual(
            basic["0/5"]["response_count"], basic["0/5"]["trajectory_count"]
        )
        self.assertEqual(basic["5/5"]["incorrect_trajectory_count"], 0)
        self.assertEqual(basic["5/5"]["correct_trajectory_count"], 15)
        self.assertEqual(basic["3/5"]["step_count"], 21)
        self.assertEqual(basic["3/5"]["difficulty"], 0.4)

    def test_trajectory_level_ci_present(self) -> None:
        group = self.sections["all"]["groups"]["2/5"]["A"]
        self.assertEqual(group["n_steps"], 21)
        self.assertEqual(group["n_trajectories"], 15)
        self.assertIsNotNone(group["ci95_step_level"])
        self.assertIsNotNone(group["ci95_trajectory_level"])

    def test_headline_slopes_negative_and_significant(self) -> None:
        for subset, metric in (
            ("all", "A"),
            ("all", "A_TI:pooled"),
            ("all", "A_TI:t1"),
            ("all", "A_TI:t2"),
            ("correct", "A"),
            ("incorrect", "A_TI:pooled"),
            ("incorrect", "A_TI:t1"),
            ("incorrect", "A_TI:t2"),
        ):
            regression = self.sections[subset]["regressions"][metric]
            self.assertLess(regression["slope"], 0.0, (subset, metric))
            self.assertLess(regression["slope_p_value"], 1e-4, (subset, metric))
            self.assertLess(regression["slope_ci95"][1], 0.0, (subset, metric))
            correlation = self.sections[subset]["correlations"][metric]
            self.assertLess(correlation["pearson_r"], 0.0, (subset, metric))
            self.assertLess(correlation["spearman_rho"], 0.0, (subset, metric))

    def test_group_means_monotonic_nonincreasing(self) -> None:
        summary = analysis.curve_summary(self.sections["all"], "A", 5)
        self.assertTrue(summary["monotonic_nonincreasing_mean"])
        self.assertTrue(summary["significant_negative_slope_at_95pct"])
        self.assertLess(summary["mean_change_last_minus_first"], 0.0)

    def test_matrix_matches_sections(self) -> None:
        matrix = analysis.matrix_section(self.sections, self.metrics, 5)
        self.assertIsNone(matrix["correct"]["0/5"]["A"]["mean"])
        self.assertIsNone(matrix["incorrect"]["5/5"]["A_TI:t2"]["mean"])
        for key in ("1/5", "2/5", "3/5", "4/5", "5/5"):
            self.assertEqual(
                matrix["correct"][key]["A"]["mean"],
                self.sections["correct"]["groups"][key]["A"]["mean"],
            )

    def test_pooled_teacher_metric_is_per_step_teacher_mean(self) -> None:
        row = self.rows[0]
        expected = (row["A_TI"]["t1"] + row["A_TI"]["t2"]) / 2
        self.assertEqual(
            analysis.metric_value(row, analysis.METRIC_A_TI_POOLED), expected
        )
        group = self.sections["all"]["groups"]["0/5"]["A_TI:pooled"]
        self.assertEqual(group["ci95"], group["ci95_step_level"])


class TestSensitivity(unittest.TestCase):
    def test_excludes_truncated_t1_steps(self) -> None:
        rows, teachers = _extract()
        records = _synthetic_results()
        truncated = sum(1 for record in records if record["teacher_replacements"]["t1"]["truncated"])
        section = analysis.sensitivity_section(rows, teachers, 5)
        entry = section["teachers"]["t1"]
        self.assertEqual(entry["metric"], "A_TI:t1")
        self.assertEqual(entry["excluded_steps"], truncated)
        self.assertGreater(entry["excluded_steps"], 0)
        self.assertEqual(entry["kept_steps"], len(rows) - truncated)
        regression = entry["incorrect"]["regressions"]["A_TI:t1"]
        self.assertLess(regression["slope"], 0.0)
        # Teacher t2 has no truncated replacements: identical sample size.
        self.assertEqual(section["teachers"]["t2"]["excluded_steps"], 0)
        self.assertEqual(section["teachers"]["t2"]["all"]["n_steps"], len(rows))


class TestStatsEdges(unittest.TestCase):
    def test_empty_and_degenerate(self) -> None:
        empty = analysis.mean_ci_stats([])
        self.assertIsNone(empty["mean"])
        single = analysis.mean_ci_stats([0.3])
        self.assertEqual(single["mean"], 0.3)
        self.assertIsNone(single["ci95"])
        correlation = analysis.correlation_stats([0.2] * 5, [0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertIsNone(correlation["pearson_r"])
        regression = analysis.ols_stats([0.2] * 5, [0.1, 0.2, 0.3, 0.4, 0.5])
        self.assertIsNone(regression["slope"])

    def test_known_ols(self) -> None:
        xs = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ys = [0.5 - 0.25 * x for x in xs]
        regression = analysis.ols_stats(xs, ys)
        self.assertAlmostEqual(regression["slope"], -0.25, places=9)
        self.assertAlmostEqual(regression["intercept"], 0.5, places=9)
        self.assertAlmostEqual(regression["slope_p_value"], 0.0, places=9)
        self.assertAlmostEqual(regression["r_squared"], 1.0, places=9)


class TestMainEndToEnd(unittest.TestCase):
    def test_report_written(self) -> None:
        records = _synthetic_results()
        # A plain scratch directory (not tempfile.mkdtemp): the Windows host's
        # restricted mkdtemp DACL can defeat sandbox write grants, while an
        # ordinary mkdir works everywhere including the server.
        tmp_path = analysis.REPO_ROOT / "tmp" / "ersr_difficulty_utility_test"
        shutil.rmtree(tmp_path, ignore_errors=True)
        tmp_path.mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, tmp_path, ignore_errors=True)
        results_path = tmp_path / "ersr_results.json"
        output_path = tmp_path / "difficulty_utility_analysis.json"
        results_path.write_text(
            json.dumps(records, ensure_ascii=False), encoding="utf-8"
        )
        argv = [
            "analyze_ersr_difficulty_utility.py",
            "--results",
            str(results_path),
            "--output",
            str(output_path),
        ]
        with mock.patch.object(sys, "argv", argv):
            analysis.main()
        self.assertTrue(output_path.is_file())
        with output_path.open(encoding="utf-8") as handle:
            report = json.load(handle)
        expected_keys = {
            "schema_version",
            "generated_at",
            "script",
            "input",
            "definitions",
            "A_basic_stats_by_accuracy",
            "B_all_trajectories",
            "C_correct_trajectories",
            "D_incorrect_trajectories",
            "matrix_outcome_x_difficulty",
            "sensitivity_truncated_excluded",
            "conclusions",
            "caveats",
        }
        self.assertEqual(set(report), expected_keys)
        self.assertEqual(report["input"]["n_steps"], 126)
        self.assertEqual(report["input"]["teachers"], list(TEACHERS))
        self.assertIn(report["input"]["p_value_method"], (
            "scipy t distribution (exact)",
            "normal approximation (scipy unavailable)",
        ))
        curves = report["conclusions"]["curves"]
        for key in (
            "all|A",
            "all|A_TI:pooled",
            "all|A_TI:t1",
            "correct|A",
            "incorrect|A_TI:pooled",
            "incorrect|A_TI:t2",
        ):
            self.assertTrue(curves[key]["significant_negative_slope_at_95pct"], key)
        self.assertEqual(len(report["conclusions"]["key_findings"]), 3)
        # The atomic writer must not leave temporary files behind.
        leftovers = [p.name for p in tmp_path.iterdir() if ".tmp." in p.name]
        self.assertEqual(leftovers, [])


if __name__ == "__main__":
    unittest.main()
