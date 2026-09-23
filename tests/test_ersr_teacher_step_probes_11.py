from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path
from unittest import mock

from tests import analyze_ersr_teacher_step_probes_11 as analysis


REPO_ROOT = Path(__file__).resolve().parents[1]


def synthetic_probes(cases: int = 24, teachers: tuple[str, ...] = ("t1", "t2")) -> list[dict]:
    rows = []
    for index in range(cases):
        prompt = index % 6
        reward = index % 2
        trajectory = f"p{prompt}:r{reward}"
        base = -0.10 + 0.012 * (index % 8)
        teacher_blocks = {}
        for teacher_index, name in enumerate(teachers):
            delta = base + 0.004 * teacher_index + 0.002 * (index % 3)
            noise = 0.001 * ((index * 7 + teacher_index) % 5 - 2)
            if reward == 1:
                a_student = 0.02 + 0.8 * delta + noise
                a_teacher = 0.01 - 0.1 * delta
            else:
                a_student = -0.01 - 0.2 * delta
                a_teacher = 0.04 - 1.2 * delta + noise
            p_before = 0.2
            p_after = p_before + delta
            teacher_blocks[name] = {
                "P_before": p_before,
                "P_student_after": p_after,
                "deltaP_T_student_step": delta,
                "delta_logP_T_student_step": delta,
                "A_teacher": a_teacher,
                "V_replace": 0.3 + a_teacher,
            }
        rows.append(
            {
                "schema_version": 1,
                "case_id": f"case-{index}",
                "case_index": index,
                "trajectory_id": trajectory,
                "response_idx": prompt,
                "rollout_index": reward,
                "reward": reward,
                "bucket": 3,
                "step_id": 1,
                "num_steps": 4,
                "relative_position": 0.5,
                "mc_k": 128,
                "probe_text": "probe:42",
                "student": {
                    "V_redecide": 0.3,
                    "V_keep": 0.3 + (0.02 + 0.8 * base if reward == 1 else -0.01),
                    "A_student": 0.02 + 0.8 * base if reward == 1 else -0.01 - 0.2 * base,
                },
                "teachers": teacher_blocks,
            }
        )
    return rows


class TestEndToEnd(unittest.TestCase):
    def setUp(self) -> None:
        self.root = REPO_ROOT / "tmp" / self.id().replace(".", "_")
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_full_run_signs_and_outputs(self) -> None:
        probes_path = self.root / "10_teacher_student_step_probes.jsonl"
        with probes_path.open("w", encoding="utf-8") as handle:
            for row in synthetic_probes():
                handle.write(json.dumps(row) + "\n")
        output_dir = self.root / "analysis_11"
        argv = [
            "analyze_ersr_teacher_step_probes_11.py",
            "--step-probes",
            str(probes_path),
            "--output-dir",
            str(output_dir),
            "--bootstrap-rounds",
            "40",
            "--seed",
            "5",
        ]
        with mock.patch.object(sys, "argv", argv):
            analysis.main()

        report = json.loads(
            (output_dir / "11_teacher_step_probe_analysis.json").read_text()
        )
        self.assertEqual(len(report["scopes"]), 12)

        # Correct branch: deltaP_T positively tracks A_student, negatively A_teacher.
        correct_student = report["scopes"]["correct|A_student|pooled"]["correlation"]
        self.assertGreater(correct_student["global_correlation"]["pearson"]["r"], 0.8)
        correct_teacher = report["scopes"]["correct|A_teacher|pooled"]["correlation"]
        self.assertLess(correct_teacher["global_correlation"]["pearson"]["r"], -0.5)

        # Incorrect branch: deltaP_T negatively tracks A_teacher (strong).
        incorrect_teacher = report["scopes"]["incorrect|A_teacher|pooled"]["correlation"]
        self.assertLess(incorrect_teacher["global_correlation"]["pearson"]["r"], -0.8)

        # Ascending quintiles: on the incorrect branch, A_teacher decreases from
        # G1 (lowest probe delta) to G5 for the planted negative relation.
        quintiles = report["scopes"]["incorrect|A_teacher|pooled"]["quintiles"]
        means = [group["mean_utility"] for group in quintiles["groups"]]
        self.assertGreater(means[0], means[-1])
        self.assertLess(
            quintiles["span_G5_minus_G1"]["difference"], 0.0
        )

        required = {"11_teacher_step_probe_analysis.json", "11_teacher_step_probe_stats.csv"}
        self.assertTrue(required.issubset({path.name for path in output_dir.iterdir()}))


if __name__ == "__main__":
    unittest.main()
