from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "tests" / "01_analyze_ersr_outcome.py"
SPEC = importlib.util.spec_from_file_location("analyze_ersr_outcome_01", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"Cannot import {SCRIPT_PATH}")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


MC_K = 128
TEACHERS = ("teacher_a", "teacher_b")


def _record(case_index: int, r: int, reasoning_step_index: int) -> dict:
    """Build a compact evaluator-shaped record with two Teacher arms."""

    rd = 64
    position_slot = (0, 2, 5, 8).index(reasoning_step_index)
    keep = (48, 64, 80, 96)[position_slot]
    if r:
        replace_a = (40, 60, 76, 84)[position_slot]
        replace_b = (44, 68, 72, 88)[position_slot]
        bucket = 4
    else:
        replace_a = (72, 76, 84, 92)[position_slot]
        replace_b = (68, 80, 88, 100)[position_slot]
        bucket = 1
    samples = {
        "redecide": {"mc_k": MC_K, "correct_count": rd},
        "keep": {"mc_k": MC_K, "correct_count": keep},
        "replace:teacher_a": {"mc_k": MC_K, "correct_count": replace_a},
        "replace:teacher_b": {"mc_k": MC_K, "correct_count": replace_b},
    }
    return {
        "schema_version": 1,
        "case_id": f"{r}:{case_index}:{reasoning_step_index}",
        "case_index": case_index,
        "response_idx": r * 100 + case_index,
        "rollout_index": case_index % 5,
        "bucket": bucket,
        "rollout_correct": r,
        "reasoning_step_index": reasoning_step_index,
        "num_reasoning_steps": 10,
        # Deliberately inconsistent: the analyzer must recompute the requested
        # one-based step fraction rather than trust this evaluator field.
        "position_fraction": 0.0,
        "samples": samples,
        "teacher_replacements": {
            "teacher_a": {"truncated": 0},
            "teacher_b": {"truncated": 0},
        },
    }


def synthetic_results() -> list[dict]:
    records: list[dict] = []
    case_index = 0
    for r in (1, 0):
        for step_index in (0, 2, 5, 8):
            records.append(_record(case_index, r, step_index))
            case_index += 1
    return records


class TestPositionDefinition(unittest.TestCase):
    def test_third_of_ten_is_point_three_and_q2(self) -> None:
        record = _record(0, 1, 2)
        fraction = analysis.step_position_fraction(record, record["case_id"])
        self.assertEqual(fraction, 0.3)
        self.assertEqual(analysis.position_bin(fraction), "Q2_(25%,50%]")

    def test_quartile_boundaries_are_complete(self) -> None:
        labels = [analysis.position_bin(value) for value in (0.25, 0.5, 0.75, 1.0)]
        self.assertEqual(labels, list(analysis.POSITION_LABELS))

    def test_invalid_step_index_is_rejected(self) -> None:
        record = _record(0, 1, 2)
        record["reasoning_step_index"] = 10
        with self.assertRaises(ValueError):
            analysis.step_position_fraction(record, record["case_id"])


class TestAnalysisSections(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.results = synthetic_results()
        cls.cases, cls.pairs, cls.teachers, cls.truncated = analysis.extract_rows(
            cls.results
        )
        cls.scopes = analysis.build_scopes(cls.pairs, cls.teachers)

    def test_extracts_cases_and_teacher_pairs(self) -> None:
        self.assertEqual(len(self.cases), 8)
        self.assertEqual(len(self.pairs), 16)
        self.assertEqual(self.teachers, list(TEACHERS))
        self.assertEqual(self.truncated, 0)
        self.assertEqual(
            [case["position_fraction"] for case in self.cases[:4]],
            [0.1, 0.3, 0.6, 0.9],
        )

    def test_core_contains_all_six_requested_statistics(self) -> None:
        core = analysis.core_section(
            self.cases, self.scopes, analysis.SeedSequence(7), bootstrap_rounds=20
        )
        for r in ("R=1", "R=0"):
            self.assertIsNotNone(core["E_A_given_R"][r]["mean"])
            self.assertIsNotNone(core["E_A_given_R"][r]["ci95_normal"])
            for metric in ("E_A_TI_given_R", "E_delta_given_R"):
                for scope in ("pooled", "teacher:teacher_a", "teacher:teacher_b"):
                    self.assertIsNotNone(core[metric][r][scope]["mean"])
                    self.assertIsNotNone(core[metric][r][scope]["ci95_normal"])
                    self.assertIsNotNone(core[metric][r][scope]["ci95_bootstrap"])

        questions = analysis.questions_section(core, self.cases, self.pairs)
        correct = questions["answers"]["correct_trajectories"]
        self.assertEqual(
            set(correct["by_scope"]),
            {"pooled", "teacher:teacher_a", "teacher:teacher_b"},
        )
        self.assertEqual(
            set(questions["expected_sign_check_by_scope"]), {"R=1", "R=0"}
        )

    def test_position_correlation_low_value_and_damage_sections(self) -> None:
        position = analysis.position_section(
            self.cases, self.scopes, analysis.SeedSequence(11)
        )
        for r in ("R=1", "R=0"):
            self.assertEqual(
                [
                    position["by_R_and_bin"][r][label]["n_cases"]
                    for label in analysis.POSITION_LABELS
                ],
                [1, 1, 1, 1],
            )

        correlations = analysis.correlation_section(self.scopes)
        for split in ("all", "correct", "incorrect"):
            self.assertIn("pearson", correlations[split]["pooled"])
            self.assertIn("spearman", correlations[split]["pooled"])

        low_value = analysis.low_value_section(self.scopes)
        self.assertIn("A<0", low_value["incorrect"]["pooled"])
        self.assertIn("P_A_TI_gt_A", low_value["incorrect"]["pooled"]["A<0"])

        damage = analysis.damage_section(self.scopes)["pooled"]
        partition = (
            damage["P_A_TI_lt_A"]
            + damage["P_A_TI_eq_A"]
            + damage["P_A_TI_gt_A"]
        )
        self.assertAlmostEqual(partition, 1.0, places=6)


class TestEndToEnd(unittest.TestCase):
    def test_writes_self_contained_json_report(self) -> None:
        tmp_dir = REPO_ROOT / "tmp" / "test_analyze_ersr_outcome"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, tmp_dir, ignore_errors=True)
        results_path = tmp_dir / "ersr_results.json"
        output_path = tmp_dir / "ersr_analysis.json"
        results_path.write_text(
            json.dumps(synthetic_results(), ensure_ascii=False), encoding="utf-8"
        )

        argv = [
            "01_analyze_ersr_outcome.py",
            "--results",
            str(results_path),
            "--output",
            str(output_path),
            "--bootstrap-rounds",
            "20",
        ]
        with mock.patch.object(sys, "argv", argv):
            analysis.main()

        report = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(report["input"]["n_cases"], 8)
        self.assertEqual(report["input"]["n_case_teacher_pairs"], 16)
        self.assertEqual(report["input"]["teachers"], list(TEACHERS))
        self.assertIn("core", report)
        self.assertIn("position_analysis", report)
        self.assertIn("correlation_A_vs_A_TI", report)
        self.assertIn("A_binning", report)
        self.assertIn("low_value_groups", report)
        self.assertIn("damage_on_correct", report)
        self.assertIn("conclusions", report)


if __name__ == "__main__":
    unittest.main()
