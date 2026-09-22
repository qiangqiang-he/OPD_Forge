from __future__ import annotations

import csv
import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

from tests import analyze_ersr_answer_probes_05 as analysis
from tests import extract_ersr_probe_results_05 as extractor
from tests.analyze_ersr_answer_probes_03 import (
    STEP_DATA_FIELDS,
    flatten_step_teacher_rows,
)
from tests.test_ersr_answer_probe_03 import synthetic_step_scores


REPO_ROOT = Path(__file__).resolve().parents[1]


def write_step_scores(path: Path) -> int:
    rows = synthetic_step_scores()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return len(rows)


class ScratchTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.root = REPO_ROOT / "tmp" / self.id().replace(".", "_")
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


class TestExtraction(ScratchTestCase):
    def test_extraction_writes_compact_csv_and_manifest(self) -> None:
        scores_path = self.root / "03_probe_step_scores.jsonl"
        record_count = write_step_scores(scores_path)
        output_path = self.root / "compact" / "05_probe_step_data.csv"
        argv = [
            "extract_ersr_probe_results_05.py",
            "--step-scores",
            str(scores_path),
            "--output",
            str(output_path),
        ]
        with mock.patch.object(sys, "argv", argv):
            extractor.main()

        manifest = json.loads(
            (output_path.parent / "05_probe_extraction_manifest.json").read_text()
        )
        self.assertEqual(manifest["step_score_records"], record_count)
        self.assertEqual(manifest["rows"], record_count * 2)
        self.assertEqual(manifest["unique_cases"], record_count)
        self.assertEqual(manifest["columns"], list(STEP_DATA_FIELDS))
        self.assertEqual(manifest["teachers"], ["t1", "t2"])
        self.assertEqual(manifest["cases_reward_counts"], {"1": 32, "0": 32})

        with output_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), record_count * 2)
        self.assertEqual(list(rows[0]), list(STEP_DATA_FIELDS))


class TestCsvLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.long_rows = flatten_step_teacher_rows(synthetic_step_scores())

    def _write_csv(self, path: Path, columns: list[str]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in self.long_rows:
                writer.writerow({field: row[field] for field in columns})

    def test_loader_round_trips_types_exactly(self) -> None:
        path = Path(self.id().replace(".", "_") + ".csv")
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        self._write_csv(path, list(STEP_DATA_FIELDS))
        loaded = analysis.load_step_data_csv(path)
        self.assertEqual(len(loaded), len(self.long_rows))
        for original, parsed in zip(self.long_rows, loaded):
            for field in STEP_DATA_FIELDS:
                self.assertEqual(parsed[field], original[field], field)

    def test_loader_rejects_missing_column(self) -> None:
        path = Path(self.id().replace(".", "_") + ".csv")
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        columns = [field for field in STEP_DATA_FIELDS if field != "A_teacher"]
        self._write_csv(path, columns)
        with self.assertRaises(ValueError):
            analysis.load_step_data_csv(path)

    def test_loader_rejects_invalid_reward(self) -> None:
        path = Path(self.id().replace(".", "_") + ".csv")
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        self._write_csv(path, list(STEP_DATA_FIELDS))
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["reward"] = "2"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(STEP_DATA_FIELDS))
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaises(ValueError):
            analysis.load_step_data_csv(path)


class TestEndToEnd(ScratchTestCase):
    def test_csv_input_reproduces_jsonl_analysis_exactly(self) -> None:
        scores_path = self.root / "03_probe_step_scores.jsonl"
        write_step_scores(scores_path)
        compact_path = self.root / "compact" / "05_probe_step_data.csv"
        with mock.patch.object(
            sys,
            "argv",
            [
                "extract_ersr_probe_results_05.py",
                "--step-scores",
                str(scores_path),
                "--output",
                str(compact_path),
            ],
        ):
            extractor.main()

        from_jsonl = self.root / "from_jsonl"
        from_csv = self.root / "from_csv"
        for output_dir, step_data in ((from_jsonl, scores_path), (from_csv, compact_path)):
            argv = [
                "analyze_ersr_answer_probes_05.py",
                "--step-data",
                str(step_data),
                "--output-dir",
                str(output_dir),
                "--bootstrap-rounds",
                "30",
                "--seed",
                "17",
            ]
            with mock.patch.object(sys, "argv", argv):
                analysis.main()

        report_jsonl = json.loads((from_jsonl / "05_probe_analysis.json").read_text())
        report_csv = json.loads((from_csv / "05_probe_analysis.json").read_text())

        # The compact CSV round trip is lossless: identical seeds must produce
        # bit-identical statistics through either input path.
        self.assertEqual(report_jsonl["scopes"], report_csv["scopes"])
        self.assertEqual(report_jsonl["head_to_head"], report_csv["head_to_head"])
        self.assertEqual(report_jsonl["conclusions"], report_csv["conclusions"])

        required = {
            "05_probe_step_data.csv",
            "05_scope_global_correlations.csv",
            "05_scope_within_trajectory_correlations.csv",
            "05_scope_quartile_stats.csv",
            "05_scope_extreme_groups.csv",
            "05_scope_sign_stats.csv",
            "05_scope_regression_robustness.csv",
            "05_head_to_head.csv",
            "05_probe_analysis.json",
            "05_probe_output_manifest.json",
        }
        self.assertTrue(required.issubset({path.name for path in from_csv.iterdir()}))

        # Full matrix: 2 branches x (1 case-unit scope + 3 paired scopes per
        # signal/utility combination except student|A_student).
        self.assertEqual(len(report_csv["scopes"]), 20)
        expected_keys = {
            "correct|student_probe|A_student|case",
            "correct|student_probe|A_teacher|pooled",
            "correct|teacher_probe|A_student|pooled",
            "correct|teacher_probe|A_teacher|pooled",
            "correct|teacher_probe|A_teacher|teacher:t1",
            "incorrect|student_probe|A_student|case",
            "incorrect|student_probe|A_teacher|pooled",
            "incorrect|teacher_probe|A_student|pooled",
            "incorrect|teacher_probe|A_teacher|pooled",
            "incorrect|student_probe|A_teacher|teacher:t2",
        }
        self.assertTrue(expected_keys.issubset(report_csv["scopes"]))
        self.assertEqual(
            report_csv["scopes"]["correct|student_probe|A_student|case"]["signal"],
            "deltaP_student",
        )
        self.assertEqual(
            report_csv["scopes"]["incorrect|teacher_probe|A_teacher|pooled"]["utility"],
            "A_teacher",
        )

        # Matched probe/utility pairs are near-deterministic in the synthetic
        # generator; the cross pairs stay strongly positive as well.
        for key in (
            "correct|student_probe|A_student|case",
            "incorrect|teacher_probe|A_teacher|pooled",
            "correct|teacher_probe|A_teacher|pooled",
            "incorrect|student_probe|A_teacher|pooled",
        ):
            self.assertGreater(
                report_csv["scopes"][key]["global_correlation"]["pearson"]["r"], 0.7, key
            )

        # Head-to-head wiring: the matched signal must beat the mismatched one.
        for key in ("correct|A_student", "incorrect|A_student"):
            grid = report_csv["head_to_head"][key]["global"]
            self.assertGreater(
                grid["student_probe"]["pearson"]["r"],
                grid["teacher_probe"]["pearson"]["r"],
                key,
            )
        for key in ("correct|A_teacher", "incorrect|A_teacher"):
            grid = report_csv["head_to_head"][key]["global"]
            self.assertGreater(
                grid["teacher_probe"]["pearson"]["r"],
                grid["student_probe"]["pearson"]["r"],
                key,
            )

        for branch in ("correct", "incorrect"):
            flags = report_csv["conclusions"]["branch_flags"][branch]
            self.assertTrue(flags["student_probe_predicts_A_student"], branch)
            self.assertTrue(flags["teacher_probe_predicts_A_teacher"], branch)

        with (from_csv / "05_probe_step_data.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            step_rows = list(csv.DictReader(handle))
        self.assertEqual(len(step_rows), len(synthetic_step_scores()) * 2)


if __name__ == "__main__":
    unittest.main()
