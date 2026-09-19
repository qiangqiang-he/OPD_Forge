#!/usr/bin/env python3
"""Analysis 04: low-budget Monte Carlo population-level ERSR estimation.

This script reads the finalized ``ersr_results.json`` produced by
``tests/run_ersr_mc.py``.  It requires every arm to retain all individual
binary rollout rewards.  MC@K is reconstructed by sampling K rewards without
replacement from the MC@128 population.  Because rewards are binary, a
vectorized hypergeometric draw is exactly equivalent to selecting explicit
reward indices and is substantially faster for the K x M x repetition grid.

The analysis separates:

* individual-step accuracy;
* population means against a same-step MC@128 reference;
* population means against the full-population MC@128 reference;
* correct/incorrect outcome-finding recovery;
* step-random and prompt-cluster sampling.

Every script output is numbered ``04``.  No model or GPU is required.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS = REPO_ROOT / "outputs/ersr_dapo17k_qwen3_1p7b_mc128/ersr_results.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/ersr_dapo17k_qwen3_1p7b_mc128/mc_analysis_04"
SCHEMA_VERSION = 1
GROUP_SIZE_G = 5
DEFAULT_K_VALUES = (2, 4, 8, 16, 32, 64)
DEFAULT_M_VALUES = (25, 50, 100, 200, 500, 1000, 2000, 5000, 10000)
DEFAULT_SAMPLING_SETTINGS = ("prompt_cluster", "step_random")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def atomic_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty required CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def parse_positive_ints(value: str, *, name: str) -> list[int]:
    try:
        parsed = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} must be comma-separated integers.") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError(f"{name} must contain positive integers.")
    return parsed


def parse_sampling_settings(value: str) -> list[str]:
    parsed = [item.strip() for item in value.split(",") if item.strip()]
    allowed = set(DEFAULT_SAMPLING_SETTINGS)
    if not parsed or len(parsed) != len(set(parsed)) or any(item not in allowed for item in parsed):
        raise argparse.ArgumentTypeError(
            "sampling settings must be unique values from prompt_cluster,step_random"
        )
    return parsed


def _binary_rewards(sample: Any, *, label: str, reference_k: int) -> np.ndarray:
    if not isinstance(sample, dict):
        raise ValueError(f"{label} is missing its sample object.")
    try:
        mc_k = int(sample["mc_k"])
        correct_count = int(sample["correct_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} has invalid mc_k/correct_count.") from exc
    rewards = sample.get("rewards")
    if mc_k != reference_k:
        raise ValueError(f"{label} has mc_k={mc_k}; expected reference_k={reference_k}.")
    if not isinstance(rewards, list) or len(rewards) != reference_k:
        raise ValueError(
            f"{label} must retain exactly {reference_k} individual rollout rewards; "
            "means/correct_count alone cannot reconstruct low-budget MC."
        )
    try:
        array = np.asarray(rewards, dtype=np.int8)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} contains invalid rewards.") from exc
    if array.ndim != 1 or np.any((array != 0) & (array != 1)):
        raise ValueError(f"{label} rewards must all be binary 0/1 values.")
    if int(array.sum()) != correct_count:
        raise ValueError(
            f"{label} has correct_count={correct_count}, sum(rewards)={int(array.sum())}."
        )
    return array.astype(np.uint8, copy=False)


@dataclass
class StepDataset:
    raw_rows: list[dict[str, Any]]
    case_ids: np.ndarray
    prompt_ids: np.ndarray
    trajectory_ids: np.ndarray
    rewards: np.ndarray
    step_ids: np.ndarray
    relative_positions: np.ndarray
    prompt_accuracies: np.ndarray
    before: np.ndarray
    student: np.ndarray
    teachers: dict[str, np.ndarray]
    reference_k: int

    @property
    def size(self) -> int:
        return len(self.case_ids)

    @property
    def teacher_names(self) -> list[str]:
        return sorted(self.teachers)


def load_step_dataset(results_path: Path, *, reference_k: int) -> StepDataset:
    with results_path.open(encoding="utf-8") as handle:
        results = json.load(handle)
    if not isinstance(results, list) or not results:
        raise ValueError(f"Expected a non-empty JSON array in {results_path}.")

    raw_rows: list[dict[str, Any]] = []
    case_ids: list[str] = []
    prompt_ids: list[str] = []
    trajectory_ids: list[str] = []
    trajectory_rewards: list[int] = []
    step_ids: list[int] = []
    relative_positions: list[float] = []
    prompt_accuracies: list[float] = []
    before_rows: list[np.ndarray] = []
    student_rows: list[np.ndarray] = []
    teacher_rows: dict[str, list[np.ndarray]] = defaultdict(list)
    expected_teachers: tuple[str, ...] | None = None
    seen_cases: set[str] = set()
    prompt_buckets: dict[str, int] = {}
    trajectory_metadata: dict[str, tuple[int, int]] = {}

    for record_index, record in enumerate(results):
        if not isinstance(record, dict):
            raise ValueError(f"Result index {record_index} is not an object.")
        case_id = str(record.get("case_id", ""))
        if not case_id or case_id in seen_cases:
            raise ValueError(f"Missing or duplicate case_id {case_id!r}.")
        seen_cases.add(case_id)
        samples = record.get("samples")
        if not isinstance(samples, dict):
            raise ValueError(f"{case_id} has no samples mapping.")
        teacher_names = tuple(sorted(key.removeprefix("replace:") for key in samples if key.startswith("replace:")))
        if not teacher_names:
            raise ValueError(f"{case_id} has no replace:<Teacher> arm.")
        if expected_teachers is None:
            expected_teachers = teacher_names
        elif teacher_names != expected_teachers:
            raise ValueError(
                f"{case_id} has Teacher set {teacher_names}, expected {expected_teachers}."
            )

        before = _binary_rewards(
            samples.get("redecide"), label=f"{case_id}/redecide", reference_k=reference_k
        )
        student = _binary_rewards(
            samples.get("keep"), label=f"{case_id}/keep", reference_k=reference_k
        )
        teacher_arrays = {
            name: _binary_rewards(
                samples.get(f"replace:{name}"),
                label=f"{case_id}/replace:{name}",
                reference_k=reference_k,
            )
            for name in teacher_names
        }

        reward = int(record["rollout_correct"])
        bucket = int(record["bucket"])
        reasoning_step_index = int(record["reasoning_step_index"])
        num_reasoning_steps = int(record["num_reasoning_steps"])
        if reward not in (0, 1):
            raise ValueError(f"{case_id} has reward={reward}; expected 0/1.")
        if not 0 <= bucket <= GROUP_SIZE_G:
            raise ValueError(f"{case_id} has bucket={bucket}; expected 0..{GROUP_SIZE_G}.")
        if num_reasoning_steps <= 0 or not 0 <= reasoning_step_index < num_reasoning_steps:
            raise ValueError(
                f"{case_id} has invalid reasoning step {reasoning_step_index}/{num_reasoning_steps}."
            )
        prompt_id = str(record["response_idx"])
        trajectory_id = f"{prompt_id}:{int(record['rollout_index'])}"
        relative_position = (reasoning_step_index + 1) / num_reasoning_steps
        prompt_accuracy = bucket / GROUP_SIZE_G
        previous_bucket = prompt_buckets.setdefault(prompt_id, bucket)
        if previous_bucket != bucket:
            raise ValueError(
                f"Prompt {prompt_id} appears in buckets {previous_bucket} and {bucket}."
            )
        metadata = (bucket, reward)
        previous_metadata = trajectory_metadata.setdefault(trajectory_id, metadata)
        if previous_metadata != metadata:
            raise ValueError(
                f"Trajectory {trajectory_id} has inconsistent bucket/reward: "
                f"{previous_metadata} vs {metadata}."
            )

        before_value = float(before.mean())
        student_value = float(student.mean())
        teacher_values = {name: float(array.mean()) for name, array in teacher_arrays.items()}
        stored_advantages = record.get("advantages") or {}
        if "A_k" in stored_advantages and not math.isclose(
            float(stored_advantages["A_k"]), student_value - before_value, abs_tol=1e-12
        ):
            raise ValueError(f"{case_id} stored A_k disagrees with individual rewards.")
        for name, value in teacher_values.items():
            key = f"A_k_TR:{name}"
            if key in stored_advantages and not math.isclose(
                float(stored_advantages[key]), value - before_value, abs_tol=1e-12
            ):
                raise ValueError(f"{case_id} stored {key} disagrees with individual rewards.")

        raw_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "case_id": case_id,
                "prompt_id": prompt_id,
                "trajectory_id": trajectory_id,
                "step_id": reasoning_step_index,
                "reward": reward,
                "num_steps": num_reasoning_steps,
                "relative_position": relative_position,
                "prompt_accuracy": prompt_accuracy,
                "reference_k": reference_k,
                "before_rollouts": before.tolist(),
                "student_rollouts": student.tolist(),
                "teacher_rollouts": {
                    name: array.tolist() for name, array in teacher_arrays.items()
                },
                "V_before_128": before_value,
                "V_student_after_128": student_value,
                "V_teacher_after_128": teacher_values,
                "A_student_128": student_value - before_value,
                "A_teacher_128": {
                    name: value - before_value for name, value in teacher_values.items()
                },
            }
        )
        case_ids.append(case_id)
        prompt_ids.append(prompt_id)
        trajectory_ids.append(trajectory_id)
        trajectory_rewards.append(reward)
        step_ids.append(reasoning_step_index)
        relative_positions.append(relative_position)
        prompt_accuracies.append(prompt_accuracy)
        before_rows.append(before)
        student_rows.append(student)
        for name, array in teacher_arrays.items():
            teacher_rows[name].append(array)

    del results
    assert expected_teachers is not None
    return StepDataset(
        raw_rows=raw_rows,
        case_ids=np.asarray(case_ids, dtype=object),
        prompt_ids=np.asarray(prompt_ids, dtype=object),
        trajectory_ids=np.asarray(trajectory_ids, dtype=object),
        rewards=np.asarray(trajectory_rewards, dtype=np.int8),
        step_ids=np.asarray(step_ids, dtype=np.int32),
        relative_positions=np.asarray(relative_positions, dtype=np.float64),
        prompt_accuracies=np.asarray(prompt_accuracies, dtype=np.float64),
        before=np.stack(before_rows).astype(np.uint8, copy=False),
        student=np.stack(student_rows).astype(np.uint8, copy=False),
        teachers={
            name: np.stack(teacher_rows[name]).astype(np.uint8, copy=False)
            for name in expected_teachers
        },
        reference_k=reference_k,
    )


def supported_m_values(requested: Sequence[int], population_size: int) -> list[int]:
    if population_size <= 0:
        raise ValueError("Population size must be positive.")
    values = [value for value in sorted(set(requested)) if value <= population_size]
    if max(requested) > population_size and population_size not in values:
        values.append(population_size)
    if not values:
        values = [population_size]
    return sorted(values)


def prompt_clusters(prompt_ids: np.ndarray) -> list[np.ndarray]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, prompt_id in enumerate(prompt_ids):
        grouped[str(prompt_id)].append(index)
    return [np.asarray(indices, dtype=np.int64) for indices in grouped.values()]


def nested_sampling_plan(
    *,
    population_size: int,
    prompt_ids: np.ndarray,
    m_values: Sequence[int],
    setting: str,
    rng: np.random.Generator,
    clusters: Sequence[np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[int, int]]:
    targets = sorted(m_values)
    if setting == "step_random":
        order = rng.permutation(population_size)[: max(targets)]
        return order.astype(np.int64, copy=False), {target: target for target in targets}
    if setting != "prompt_cluster":
        raise ValueError(f"Unknown sampling setting {setting!r}.")
    groups = list(clusters) if clusters is not None else prompt_clusters(prompt_ids)
    group_order = rng.permutation(len(groups))
    pieces: list[np.ndarray] = []
    actual: dict[int, int] = {}
    count = 0
    next_target = 0
    for group_index in group_order:
        group = groups[int(group_index)]
        pieces.append(group)
        count += len(group)
        while next_target < len(targets) and count >= targets[next_target]:
            actual[targets[next_target]] = count
            next_target += 1
        if next_target == len(targets):
            break
    if len(actual) != len(targets):
        raise RuntimeError("Prompt-cluster sampler could not satisfy all M targets.")
    return np.concatenate(pieces), actual


def draw_low_budget_counts(
    correct_counts: np.ndarray,
    *,
    reference_k: int,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    counts = np.asarray(correct_counts, dtype=np.int64)
    return rng.hypergeometric(counts, reference_k - counts, k).astype(np.float64)


@dataclass
class ErrorSamples:
    estimates: list[float]
    references: list[float]
    actual_m: list[int]

    @classmethod
    def empty(cls) -> "ErrorSamples":
        return cls([], [], [])

    def add(self, estimate: float, reference: float, actual_m: int) -> None:
        self.estimates.append(float(estimate))
        self.references.append(float(reference))
        self.actual_m.append(int(actual_m))

    def summary(self) -> dict[str, Any]:
        estimates = np.asarray(self.estimates, dtype=np.float64)
        references = np.asarray(self.references, dtype=np.float64)
        errors = estimates - references
        absolute = np.abs(errors)
        return {
            "num_repetitions": len(estimates),
            "mean_estimate": float(estimates.mean()),
            "estimate_std": float(estimates.std(ddof=1)) if len(estimates) > 1 else 0.0,
            "estimate_p2p5": float(np.quantile(estimates, 0.025)),
            "estimate_p97p5": float(np.quantile(estimates, 0.975)),
            "mean_reference": float(references.mean()),
            "bias": float(errors.mean()),
            "MAE": float(absolute.mean()),
            "RMSE": float(np.sqrt(np.mean(errors * errors))),
            "median_abs_error": float(np.median(absolute)),
            "p90_abs_error": float(np.quantile(absolute, 0.90)),
            "p95_abs_error": float(np.quantile(absolute, 0.95)),
            "mean_actual_M": float(np.mean(self.actual_m)),
            "min_actual_M": int(min(self.actual_m)),
            "max_actual_M": int(max(self.actual_m)),
        }


@dataclass
class PairMoments:
    count: int = 0
    sum_abs_error: float = 0.0
    sum_squared_error: float = 0.0
    sign_matches: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_xx: float = 0.0
    sum_yy: float = 0.0
    sum_xy: float = 0.0

    def add(self, estimate: np.ndarray, reference: np.ndarray) -> None:
        x = np.asarray(estimate, dtype=np.float64).ravel()
        y = np.asarray(reference, dtype=np.float64).ravel()
        if len(x) != len(y):
            raise ValueError("Estimate/reference length mismatch.")
        error = x - y
        self.count += len(x)
        self.sum_abs_error += float(np.abs(error).sum())
        self.sum_squared_error += float(np.dot(error, error))
        self.sign_matches += int(np.count_nonzero(np.sign(x) == np.sign(y)))
        self.sum_x += float(x.sum())
        self.sum_y += float(y.sum())
        self.sum_xx += float(np.dot(x, x))
        self.sum_yy += float(np.dot(y, y))
        self.sum_xy += float(np.dot(x, y))

    def summary(self) -> dict[str, Any]:
        if self.count == 0:
            return {"MAE": None, "RMSE": None, "corr": None, "sign_agreement": None}
        numerator = self.count * self.sum_xy - self.sum_x * self.sum_y
        denominator = math.sqrt(
            max(0.0, self.count * self.sum_xx - self.sum_x * self.sum_x)
            * max(0.0, self.count * self.sum_yy - self.sum_y * self.sum_y)
        )
        correlation = numerator / denominator if denominator > 0.0 else None
        return {
            "MAE": self.sum_abs_error / self.count,
            "RMSE": math.sqrt(self.sum_squared_error / self.count),
            "corr": correlation,
            "sign_agreement": self.sign_matches / self.count,
        }


def population_masks(rewards: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "all": np.ones(len(rewards), dtype=bool),
        "correct": rewards == 1,
        "incorrect": rewards == 0,
    }


def individual_step_accuracy(
    dataset: StepDataset,
    *,
    k_values: Sequence[int],
    repetitions: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    before_counts = dataset.before.sum(axis=1, dtype=np.int64)
    student_counts = dataset.student.sum(axis=1, dtype=np.int64)
    teacher_names = dataset.teacher_names
    teacher_counts = np.stack(
        [dataset.teachers[name].sum(axis=1, dtype=np.int64) for name in teacher_names]
    )
    ref_before = before_counts / dataset.reference_k
    ref_a = student_counts / dataset.reference_k - ref_before
    ref_atr = teacher_counts / dataset.reference_k - ref_before[None, :]
    masks = population_masks(dataset.rewards)
    rows: list[dict[str, Any]] = []

    for k in k_values:
        a_moments = {population: PairMoments() for population in masks}
        atr_moments = {
            (population, scope): PairMoments()
            for population in masks
            for scope in ["pooled", *[f"teacher:{name}" for name in teacher_names]]
        }
        for _ in range(repetitions):
            low_before = draw_low_budget_counts(
                before_counts, reference_k=dataset.reference_k, k=k, rng=rng
            ) / k
            low_student = draw_low_budget_counts(
                student_counts, reference_k=dataset.reference_k, k=k, rng=rng
            ) / k
            low_teacher = draw_low_budget_counts(
                teacher_counts, reference_k=dataset.reference_k, k=k, rng=rng
            ) / k
            low_a = low_student - low_before
            low_atr = low_teacher - low_before[None, :]
            for population, mask in masks.items():
                a_moments[population].add(low_a[mask], ref_a[mask])
                atr_moments[(population, "pooled")].add(
                    low_atr[:, mask], ref_atr[:, mask]
                )
                for teacher_index, name in enumerate(teacher_names):
                    atr_moments[(population, f"teacher:{name}")].add(
                        low_atr[teacher_index, mask], ref_atr[teacher_index, mask]
                    )
        for population, mask in masks.items():
            a_summary = a_moments[population].summary()
            for scope in ["pooled", *[f"teacher:{name}" for name in teacher_names]]:
                atr_summary = atr_moments[(population, scope)].summary()
                rows.append(
                    {
                        "K": k,
                        "reference_K": dataset.reference_k,
                        "population": population,
                        "teacher_scope": scope,
                        "num_steps": int(np.count_nonzero(mask)),
                        "num_repetitions": repetitions,
                        "A_sample_count": a_moments[population].count,
                        "A_MAE": a_summary["MAE"],
                        "A_RMSE": a_summary["RMSE"],
                        "A_corr_with_128": a_summary["corr"],
                        "A_sign_agreement": a_summary["sign_agreement"],
                        "ATR_sample_count": atr_moments[(population, scope)].count,
                        "ATR_MAE": atr_summary["MAE"],
                        "ATR_RMSE": atr_summary["RMSE"],
                        "ATR_corr_with_128": atr_summary["corr"],
                        "ATR_sign_agreement": atr_summary["sign_agreement"],
                    }
                )
        print(
            json.dumps(
                {
                    "event": "04_mc_individual_progress",
                    "K": k,
                    "repetitions": repetitions,
                }
            ),
            flush=True,
        )
    return rows


def _target_specs(
    branch: str,
    ref_a: np.ndarray,
    ref_atr: np.ndarray,
    teacher_names: Sequence[str],
) -> list[dict[str, Any]]:
    specs = [
        {
            "target": f"{branch}_A",
            "scope": "student",
            "reference": ref_a,
            "states_per_step": 2,
        },
        {
            "target": f"{branch}_ATR",
            "scope": "pooled",
            "reference": ref_atr,
            "states_per_step": 1 + len(teacher_names),
        },
    ]
    for index, name in enumerate(teacher_names):
        specs.append(
            {
                "target": f"{branch}_ATR",
                "scope": f"teacher:{name}",
                "reference": ref_atr[index],
                "states_per_step": 2,
                "teacher_index": index,
            }
        )
    return specs


def _metric_cost_fields(
    *, k: int, m: int, states_per_step: int, summary: dict[str, Any]
) -> dict[str, Any]:
    mean_actual_m = float(summary["mean_actual_M"])
    repetitions = int(summary["num_repetitions"])
    return {
        "K_times_M": k * m,
        "states_per_step": states_per_step,
        "nominal_rollouts_per_repetition": states_per_step * k * m,
        "mean_actual_rollouts_per_repetition": states_per_step * k * mean_actual_m,
        "mean_actual_total_rollouts_all_repetitions": (
            states_per_step * k * mean_actual_m * repetitions
        ),
    }


def population_grid(
    dataset: StepDataset,
    *,
    k_values: Sequence[int],
    requested_m_values: Sequence[int],
    repetitions: int,
    sampling_settings: Sequence[str],
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    teacher_names = dataset.teacher_names
    before_counts_all = dataset.before.sum(axis=1, dtype=np.int64)
    student_counts_all = dataset.student.sum(axis=1, dtype=np.int64)
    teacher_counts_all = np.stack(
        [dataset.teachers[name].sum(axis=1, dtype=np.int64) for name in teacher_names]
    )
    ref_before_all = before_counts_all / dataset.reference_k
    ref_a_all = student_counts_all / dataset.reference_k - ref_before_all
    ref_atr_all = teacher_counts_all / dataset.reference_k - ref_before_all[None, :]

    population_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    reference_summary: dict[str, Any] = {
        "reference_K": dataset.reference_k,
        "teachers": teacher_names,
        "branches": {},
    }

    for branch, branch_reward in (("correct", 1), ("incorrect", 0)):
        global_indices = np.flatnonzero(dataset.rewards == branch_reward)
        if len(global_indices) == 0:
            raise ValueError(f"No {branch} steps were found.")
        prompt_ids = dataset.prompt_ids[global_indices]
        before_counts = before_counts_all[global_indices]
        student_counts = student_counts_all[global_indices]
        teacher_counts = teacher_counts_all[:, global_indices]
        ref_before = ref_before_all[global_indices]
        ref_a = ref_a_all[global_indices]
        ref_atr = ref_atr_all[:, global_indices]
        m_values = supported_m_values(requested_m_values, len(global_indices))
        branch_prompt_clusters = prompt_clusters(prompt_ids)
        target_specs = _target_specs(branch, ref_a, ref_atr, teacher_names)
        outcome_direction = 1.0 if branch == "correct" else -1.0
        # direction * (A - ATR) gives A-ATR for correct and ATR-A for incorrect.
        full_outcome_pooled = float(outcome_direction * (ref_a[None, :] - ref_atr).mean())
        full_outcome_teachers = {
            name: float(outcome_direction * (ref_a - ref_atr[index]).mean())
            for index, name in enumerate(teacher_names)
        }
        reference_summary["branches"][branch] = {
            "step_count": len(global_indices),
            "prompt_count": len(set(str(value) for value in prompt_ids)),
            "supported_M": m_values,
            "mean_A": float(ref_a.mean()),
            "mean_ATR_pooled": float(ref_atr.mean()),
            "mean_ATR_by_teacher": {
                name: float(ref_atr[index].mean())
                for index, name in enumerate(teacher_names)
            },
            "outcome_difference_pooled": full_outcome_pooled,
            "outcome_difference_by_teacher": full_outcome_teachers,
        }

        for setting in sampling_settings:
            for k in k_values:
                metric_samples: dict[tuple[int, str, str, str], ErrorSamples] = {}
                recovery_samples: dict[tuple[int, str], dict[str, Any]] = {}
                for m in m_values:
                    for spec in target_specs:
                        for reference_type in ("same_step_mc128", "full_population_mc128"):
                            metric_samples[(m, spec["target"], spec["scope"], reference_type)] = ErrorSamples.empty()
                    for scope in ["pooled", *[f"teacher:{name}" for name in teacher_names]]:
                        recovery_samples[(m, scope)] = {
                            "estimated": [],
                            "same_step_reference": [],
                            "actual_m": [],
                        }

                full_target_references = {
                    (spec["target"], spec["scope"]): float(
                        np.asarray(spec["reference"], dtype=np.float64).mean()
                    )
                    for spec in target_specs
                }

                for _ in range(repetitions):
                    local_order, actual_m = nested_sampling_plan(
                        population_size=len(global_indices),
                        prompt_ids=prompt_ids,
                        m_values=m_values,
                        setting=setting,
                        rng=rng,
                        clusters=branch_prompt_clusters if setting == "prompt_cluster" else None,
                    )
                    selected_before = before_counts[local_order]
                    low_before = draw_low_budget_counts(
                        selected_before,
                        reference_k=dataset.reference_k,
                        k=k,
                        rng=rng,
                    ) / k
                    low_student = draw_low_budget_counts(
                        student_counts[local_order],
                        reference_k=dataset.reference_k,
                        k=k,
                        rng=rng,
                    ) / k
                    low_teacher = draw_low_budget_counts(
                        teacher_counts[:, local_order],
                        reference_k=dataset.reference_k,
                        k=k,
                        rng=rng,
                    ) / k
                    low_a = low_student - low_before
                    low_atr = low_teacher - low_before[None, :]
                    selected_ref_a = ref_a[local_order]
                    selected_ref_atr = ref_atr[:, local_order]

                    for m in m_values:
                        count = actual_m[m]
                        estimates = {
                            (f"{branch}_A", "student"): float(low_a[:count].mean()),
                            (f"{branch}_ATR", "pooled"): float(low_atr[:, :count].mean()),
                        }
                        same_step_references = {
                            (f"{branch}_A", "student"): float(selected_ref_a[:count].mean()),
                            (f"{branch}_ATR", "pooled"): float(
                                selected_ref_atr[:, :count].mean()
                            ),
                        }
                        for teacher_index, name in enumerate(teacher_names):
                            key = (f"{branch}_ATR", f"teacher:{name}")
                            estimates[key] = float(low_atr[teacher_index, :count].mean())
                            same_step_references[key] = float(
                                selected_ref_atr[teacher_index, :count].mean()
                            )
                        for spec in target_specs:
                            key = (spec["target"], spec["scope"])
                            metric_samples[(*((m,) + key), "same_step_mc128")].add(
                                estimates[key], same_step_references[key], count
                            )
                            metric_samples[(*((m,) + key), "full_population_mc128")].add(
                                estimates[key], full_target_references[key], count
                            )

                        low_difference = outcome_direction * (
                            low_a[None, :count] - low_atr[:, :count]
                        )
                        ref_difference = outcome_direction * (
                            selected_ref_a[None, :count]
                            - selected_ref_atr[:, :count]
                        )
                        recovery_samples[(m, "pooled")]["estimated"].append(
                            float(low_difference.mean())
                        )
                        recovery_samples[(m, "pooled")]["same_step_reference"].append(
                            float(ref_difference.mean())
                        )
                        recovery_samples[(m, "pooled")]["actual_m"].append(count)
                        for teacher_index, name in enumerate(teacher_names):
                            scope = f"teacher:{name}"
                            recovery_samples[(m, scope)]["estimated"].append(
                                float(low_difference[teacher_index].mean())
                            )
                            recovery_samples[(m, scope)]["same_step_reference"].append(
                                float(ref_difference[teacher_index].mean())
                            )
                            recovery_samples[(m, scope)]["actual_m"].append(count)

                for m in m_values:
                    for spec in target_specs:
                        for reference_type in ("same_step_mc128", "full_population_mc128"):
                            key = (m, spec["target"], spec["scope"], reference_type)
                            summary = metric_samples[key].summary()
                            row = {
                                "sampling_setting": setting,
                                "K": k,
                                "M": m,
                                "target": spec["target"],
                                "teacher_scope": spec["scope"],
                                "reference_type": reference_type,
                                **summary,
                                **_metric_cost_fields(
                                    k=k,
                                    m=m,
                                    states_per_step=int(spec["states_per_step"]),
                                    summary=summary,
                                ),
                            }
                            population_rows.append(row)

                    for scope in ["pooled", *[f"teacher:{name}" for name in teacher_names]]:
                        samples = recovery_samples[(m, scope)]
                        estimated = np.asarray(samples["estimated"], dtype=np.float64)
                        same_ref = np.asarray(samples["same_step_reference"], dtype=np.float64)
                        if scope == "pooled":
                            full_ref = full_outcome_pooled
                            states_per_step = 2 + len(teacher_names)
                        else:
                            name = scope.split(":", 1)[1]
                            full_ref = full_outcome_teachers[name]
                            states_per_step = 3
                        full_errors = estimated - full_ref
                        same_errors = estimated - same_ref
                        actual_array = np.asarray(samples["actual_m"], dtype=np.float64)
                        recovery_rows.append(
                            {
                                "sampling_setting": setting,
                                "K": k,
                                "M": m,
                                "branch": branch,
                                "teacher_scope": scope,
                                "num_repetitions": repetitions,
                                "reference_difference": full_ref,
                                "reference_sign": int(np.sign(full_ref)),
                                "mean_estimated_difference": float(estimated.mean()),
                                "difference_bias": float(full_errors.mean()),
                                "difference_MAE": float(np.abs(full_errors).mean()),
                                "difference_RMSE": float(np.sqrt(np.mean(full_errors**2))),
                                "median_abs_error": float(np.median(np.abs(full_errors))),
                                "p90_abs_error": float(np.quantile(np.abs(full_errors), 0.90)),
                                "p95_abs_error": float(np.quantile(np.abs(full_errors), 0.95)),
                                "sign_recovery_rate": float(
                                    np.mean(np.sign(estimated) == np.sign(full_ref))
                                ),
                                "same_step_difference_MAE": float(
                                    np.abs(same_errors).mean()
                                ),
                                "same_step_sign_recovery_rate": float(
                                    np.mean(np.sign(estimated) == np.sign(same_ref))
                                ),
                                "mean_actual_M": float(actual_array.mean()),
                                "min_actual_M": int(actual_array.min()),
                                "max_actual_M": int(actual_array.max()),
                                "K_times_M": k * m,
                                "states_per_step": states_per_step,
                                "nominal_rollouts_per_repetition": states_per_step * k * m,
                                "mean_actual_rollouts_per_repetition": float(
                                    states_per_step * k * actual_array.mean()
                                ),
                                "mean_actual_total_rollouts_all_repetitions": float(
                                    states_per_step * k * actual_array.mean() * repetitions
                                ),
                            }
                        )

                print(
                    json.dumps(
                        {
                            "event": "04_mc_population_progress",
                            "branch": branch,
                            "sampling_setting": setting,
                            "K": k,
                            "M_values": m_values,
                            "repetitions": repetitions,
                        }
                    ),
                    flush=True,
                )
    return population_rows, recovery_rows, reference_summary


def build_threshold_summary(
    population_rows: Sequence[dict[str, Any]],
    recovery_rows: Sequence[dict[str, Any]],
    *,
    mae_threshold: float,
    recovery_threshold: float,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "criteria": {
            "population_full_reference_MAE_at_most": mae_threshold,
            "outcome_sign_recovery_rate_at_least": recovery_threshold,
        },
        "population": {},
        "outcome_recovery": {},
    }
    headline_targets = (("correct_A", "student"), ("incorrect_ATR", "pooled"))
    for target, scope in headline_targets:
        key = f"prompt_cluster|{target}|{scope}"
        summary["population"][key] = {}
        k_values = sorted(
            {
                int(row["K"])
                for row in population_rows
                if row["sampling_setting"] == "prompt_cluster"
                and row["target"] == target
                and row["teacher_scope"] == scope
                and row["reference_type"] == "full_population_mc128"
            }
        )
        for k in k_values:
            candidates = sorted(
                (
                    row
                    for row in population_rows
                    if row["sampling_setting"] == "prompt_cluster"
                    and row["target"] == target
                    and row["teacher_scope"] == scope
                    and row["reference_type"] == "full_population_mc128"
                    and int(row["K"]) == k
                    and float(row["MAE"]) <= mae_threshold
                ),
                key=lambda row: int(row["M"]),
            )
            summary["population"][key][str(k)] = (
                {"minimum_M": int(candidates[0]["M"]), "MAE": candidates[0]["MAE"]}
                if candidates
                else None
            )

    for branch in ("correct", "incorrect"):
        key = f"prompt_cluster|{branch}|pooled"
        summary["outcome_recovery"][key] = {}
        k_values = sorted(
            {
                int(row["K"])
                for row in recovery_rows
                if row["sampling_setting"] == "prompt_cluster"
                and row["branch"] == branch
                and row["teacher_scope"] == "pooled"
            }
        )
        for k in k_values:
            candidates = sorted(
                (
                    row
                    for row in recovery_rows
                    if row["sampling_setting"] == "prompt_cluster"
                    and row["branch"] == branch
                    and row["teacher_scope"] == "pooled"
                    and int(row["K"]) == k
                    and float(row["sign_recovery_rate"]) >= recovery_threshold
                ),
                key=lambda row: int(row["M"]),
            )
            summary["outcome_recovery"][key][str(k)] = (
                {
                    "minimum_M": int(candidates[0]["M"]),
                    "sign_recovery_rate": candidates[0]["sign_recovery_rate"],
                }
                if candidates
                else None
            )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--reference-k", type=int, default=128)
    parser.add_argument(
        "--k-values",
        default=",".join(str(value) for value in DEFAULT_K_VALUES),
        help="comma-separated low per-step MC budgets",
    )
    parser.add_argument(
        "--m-values",
        default=",".join(str(value) for value in DEFAULT_M_VALUES),
        help="comma-separated target numbers of aggregated reasoning steps",
    )
    parser.add_argument("--repetitions", type=int, default=1000)
    parser.add_argument(
        "--individual-repetitions",
        type=int,
        default=100,
        help="individual-step subsampling repetitions; zero uses --repetitions",
    )
    parser.add_argument(
        "--sampling-settings",
        default=",".join(DEFAULT_SAMPLING_SETTINGS),
        help="prompt_cluster,step_random or both",
    )
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--mae-threshold", type=float, default=0.02)
    parser.add_argument("--recovery-threshold", type=float, default=0.95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = args.results.resolve()
    if results_path.is_dir():
        results_path = results_path / "ersr_results.json"
    if not results_path.is_file():
        raise FileNotFoundError(results_path)
    if args.reference_k <= 1 or args.repetitions <= 0 or args.individual_repetitions < 0:
        raise ValueError("reference K/repetition counts are invalid.")
    if args.mae_threshold < 0.0 or not 0.0 <= args.recovery_threshold <= 1.0:
        raise ValueError("Thresholds are out of range.")
    k_values = parse_positive_ints(str(args.k_values), name="K values")
    m_values = parse_positive_ints(str(args.m_values), name="M values")
    sampling_settings = parse_sampling_settings(str(args.sampling_settings))
    if any(k > args.reference_k for k in k_values):
        raise ValueError(f"Every K must be <= reference_k={args.reference_k}.")
    individual_repetitions = int(args.individual_repetitions or args.repetitions)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    dataset = load_step_dataset(results_path, reference_k=int(args.reference_k))
    step_path = output_dir / "04_mc_step_rollouts.jsonl"
    atomic_jsonl(step_path, dataset.raw_rows)
    dataset.raw_rows.clear()

    individual_rows = individual_step_accuracy(
        dataset,
        k_values=k_values,
        repetitions=individual_repetitions,
        rng=np.random.default_rng(int(args.seed) + 1),
    )
    population_rows, recovery_rows, reference_summary = population_grid(
        dataset,
        k_values=k_values,
        requested_m_values=m_values,
        repetitions=int(args.repetitions),
        sampling_settings=sampling_settings,
        rng=np.random.default_rng(int(args.seed) + 2),
    )
    individual_path = output_dir / "04_mc_individual_step_accuracy.csv"
    population_path = output_dir / "04_mc_population_mean_grid.csv"
    recovery_path = output_dir / "04_mc_outcome_recovery_grid.csv"
    atomic_csv(individual_path, individual_rows)
    atomic_csv(population_path, population_rows)
    atomic_csv(recovery_path, recovery_rows)

    threshold_summary = build_threshold_summary(
        population_rows,
        recovery_rows,
        mae_threshold=float(args.mae_threshold),
        recovery_threshold=float(args.recovery_threshold),
    )
    outputs = {
        path.name: {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in (step_path, individual_path, population_path, recovery_path)
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "script": "tests/analyze_ersr_low_budget_mc_04.py",
        "input": {
            "results": str(results_path),
            "results_sha256": sha256_file(results_path),
            "steps": dataset.size,
            "correct_steps": int(np.count_nonzero(dataset.rewards == 1)),
            "incorrect_steps": int(np.count_nonzero(dataset.rewards == 0)),
            "prompts": len(set(str(value) for value in dataset.prompt_ids)),
            "trajectories": len(set(str(value) for value in dataset.trajectory_ids)),
            "teachers": dataset.teacher_names,
            "reference_K": dataset.reference_k,
        },
        "configuration": {
            "K_values": k_values,
            "requested_M_values": m_values,
            "repetitions": int(args.repetitions),
            "individual_repetitions": individual_repetitions,
            "sampling_settings": sampling_settings,
            "seed": int(args.seed),
        },
        "definitions": {
            "low_budget_sampling": (
                "K rewards sampled without replacement from each state's reference rewards; "
                "binary-reward hypergeometric draws are exactly distribution-equivalent"
            ),
            "same_step_mc128": "the same sampled steps evaluated with all reference rewards",
            "full_population_mc128": "all available branch steps evaluated with all reference rewards",
            "step_random": "M steps sampled without replacement; nested M values reuse prefixes",
            "prompt_cluster": (
                "prompts sampled without replacement and all their branch steps retained until "
                "the target M is reached or slightly exceeded"
            ),
            "correct_outcome_difference": "mean(A - ATR | R=1)",
            "incorrect_outcome_difference": "mean(ATR - A | R=0)",
            "pooled_teacher": "all Teacher interventions for every sampled reasoning step",
            "cost": (
                "A uses 2 states; one-Teacher ATR uses 2; one-Teacher outcome comparison uses 3; "
                "pooled costs reflect all configured Teachers"
            ),
        },
        "reference_summary": reference_summary,
        "threshold_summary": threshold_summary,
        "row_counts": {
            "individual_step_accuracy": len(individual_rows),
            "population_mean_grid": len(population_rows),
            "outcome_recovery_grid": len(recovery_rows),
        },
        "optional_extensions": {
            "difficulty_recovery": "not emitted by the core implementation",
            "probe_correlation_recovery": "not emitted; requires joining Analysis 03 probe data",
        },
        "outputs": outputs,
        "elapsed_seconds": time.monotonic() - started,
        "interpretation_note": (
            "Threshold summaries are mechanical lookups from observed simulation results and do "
            "not force the expected low-K/high-M conclusion."
        ),
    }
    report_path = output_dir / "04_mc_analysis.json"
    atomic_json(report_path, report)
    outputs[report_path.name] = {
        "path": str(report_path),
        "sha256": sha256_file(report_path),
        "bytes": report_path.stat().st_size,
    }
    manifest_path = output_dir / "04_mc_output_manifest.json"
    atomic_json(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "generated_at": utc_now(),
            "outputs": outputs,
        },
    )
    print(
        json.dumps(
            {
                "event": "04_mc_analysis_complete",
                "steps": dataset.size,
                "teachers": dataset.teacher_names,
                "K_values": k_values,
                "requested_M_values": m_values,
                "repetitions": int(args.repetitions),
                "output_dir": str(output_dir),
                "threshold_summary": threshold_summary,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
