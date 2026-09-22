#!/usr/bin/env python3
"""Collect per-step Teacher-Student token logprobs for ERSR analysis 07.

Server-side GPU collector.  For every ERSR-selected step k (exactly the steps
that carry an ``A_k`` estimate in ``ersr_results.json``), it scores the
Student's own step tokens under both the Student model and every configured
Teacher model, with identical context splicing::

    y_1..y_L = response_token_ids[step_token_start:step_token_end]
    ctx_t    = prompt_token_ids + response_token_ids[:step_token_start]
               + y_1..y_{t-1}

    log-ratio(step k, Teacher T)
        = (1/L) * sum_t [ log p_T(y_t | ctx_t) - log p_S(y_t | ctx_t) ]

The collector performs no analysis.  It writes one JSON record per case
(``07_step_ts_logprob.jsonl``) containing the per-model step means/sums, the
derived ``D_TS_mean``/``D_TS_sum``, and the matching ERSR values, so the file
can be copied off the server and analyzed locally.

Cost control: whole trajectories are deduplicated (one forward pass per model
per trajectory; the 12,000 selected steps live on 2,400 trajectories), every
GPU writes atomic resumable shards, and the three model phases run
sequentially (Student first, then each Teacher) on every local GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.collect_ersr_answer_probes_03 import (
    assert_gpu_headroom,
    atomic_json,
    atomic_jsonl,
    discover_local_gpus,
    load_json,
    sha256_file,
    token_ids_sha256,
    utc_now,
)

DEFAULT_RESULTS = REPO_ROOT / "outputs/ersr_dapo17k_qwen3_1p7b_mc128/ersr_results.json"
DEFAULT_CONFIG = REPO_ROOT / "configs/ersr/ersr_dapo17k_qwen3_1p7b.yaml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/ersr_dapo17k_qwen3_1p7b_mc128/step_ts_07"
SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Task construction (pure, unit-testable)
# ---------------------------------------------------------------------------


def build_trajectory_tasks(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate ERSR cases into one scoring task per trajectory.

    Only the selected steps (the ones carrying A_k) keep their token windows.
    Repeated trajectories must agree bit-exactly on prompt and response ids.
    """

    tasks: dict[tuple[int, int], dict[str, Any]] = {}
    for record in records:
        response_idx = int(record["response_idx"])
        rollout_index = int(record["rollout_index"])
        prompt_ids = [int(value) for value in record["prompt_token_ids"]]
        response_ids = [int(value) for value in record["response_token_ids"]]
        prompt_hash = token_ids_sha256(prompt_ids)
        response_hash = token_ids_sha256(response_ids)
        key = (response_idx, rollout_index)
        case = {
            "case_id": str(record["case_id"]),
            "case_index": int(record["case_index"]),
            "step_token_start": int(record["step_token_start"]),
            "step_token_end": int(record["step_token_end"]),
        }
        if case["step_token_end"] <= case["step_token_start"]:
            raise ValueError(f"Empty selected step for {case['case_id']}.")
        if key not in tasks:
            tasks[key] = {
                "trajectory_id": f"{response_idx}:{rollout_index}",
                "response_idx": response_idx,
                "rollout_index": rollout_index,
                "prompt_token_ids": prompt_ids,
                "response_token_ids": response_ids,
                "prompt_ids_sha256": prompt_hash,
                "response_ids_sha256": response_hash,
                "cases": [case],
            }
            continue
        task = tasks[key]
        if (
            task["prompt_ids_sha256"] != prompt_hash
            or task["response_ids_sha256"] != response_hash
        ):
            raise ValueError(
                f"Trajectory {task['trajectory_id']} appears with different token ids."
            )
        task["cases"].append(case)
    for task in tasks.values():
        task["cases"].sort(key=lambda value: value["case_index"])
    ordered = sorted(tasks.values(), key=lambda value: value["cases"][0]["case_index"])
    if len({case["case_id"] for task in ordered for case in task["cases"]}) != sum(
        len(task["cases"]) for task in ordered
    ):
        raise ValueError("Duplicate case_id across trajectory tasks.")
    return ordered


def extract_step_logprob_values(
    full_ids: Sequence[int],
    prompt_logprobs: Sequence[Any],
    *,
    prompt_len: int,
    start: int,
    end: int,
) -> list[float]:
    """Logprobs of the student step tokens at ``[start, end)`` of the response."""

    values: list[float] = []
    for response_position in range(start, end):
        position = prompt_len + response_position
        if position >= len(full_ids):
            raise ValueError(
                f"Step window position {position} outside sequence of {len(full_ids)} tokens."
            )
        candidates = prompt_logprobs[position]
        token_id = int(full_ids[position])
        if candidates is None or token_id not in candidates:
            raise RuntimeError(
                f"Chosen step token {token_id} is absent at prompt position {position}."
            )
        values.append(float(candidates[token_id].logprob))
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError("Step window produced empty or non-finite token logprobs.")
    return values


def summarize_step(values: Sequence[float]) -> dict[str, Any]:
    total = math.fsum(values)
    return {
        "num_tokens": len(values),
        "sum_logprob": total,
        "mean_logprob": total / len(values),
    }


def required_model_len(tasks: Sequence[dict[str, Any]]) -> int:
    longest = max(
        len(task["prompt_token_ids"]) + len(task["response_token_ids"])
        for task in tasks
    )
    return max(8192, longest + 1)


# ---------------------------------------------------------------------------
# GPU worker and role orchestration
# ---------------------------------------------------------------------------


def _sanitize_role_tag(role: str) -> str:
    tag = re.sub(r"[^0-9A-Za-z_.-]+", "_", role)
    if not tag or tag in {".", ".."}:
        raise ValueError(f"Unusable role name for shard files: {role!r}.")
    return tag


def score_worker(
    *,
    gpu_id: int,
    rank: int,
    role: str,
    model_path: str,
    tasks: list[dict[str, Any]],
    output_dir: Path,
    max_model_len: int,
    batch_size: int,
    max_num_batched_tokens: int,
    gpu_memory_utilization: float,
    minimum_free_mib: int,
    store_token_logprobs: bool,
    result_queue: Any,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    llm: Any | None = None
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        assert_gpu_headroom(gpu_id, minimum_free_mib, stage=f"07-{role}-before-load")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True, local_files_only=True
        )
        vocabulary = len(tokenizer)
        for task in tasks:
            largest = max(
                max(task["prompt_token_ids"]), max(task["response_token_ids"])
            )
            if largest >= vocabulary:
                raise RuntimeError(
                    f"Token id {largest} outside {role} vocabulary of {vocabulary}."
                )
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=max_model_len,
            max_num_seqs=batch_size,
            max_num_batched_tokens=max_num_batched_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            enforce_eager=True,
            enable_sleep_mode=False,
        )
        memory = assert_gpu_headroom(gpu_id, minimum_free_mib, stage=f"07-{role}-after-load")
        result_queue.put(
            {"type": "ready", "role": role, "rank": rank, "gpu": gpu_id, "memory": memory}
        )
        params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
        for batch_index, offset in enumerate(range(0, len(tasks), batch_size)):
            batch = tasks[offset : offset + batch_size]
            prompts = [
                {
                    "prompt_token_ids": list(task["prompt_token_ids"])
                    + list(task["response_token_ids"])
                }
                for task in batch
            ]
            started = time.monotonic()
            outputs = llm.generate(prompts, sampling_params=params, use_tqdm=False)
            if len(outputs) != len(batch):
                raise RuntimeError(
                    f"{role} returned {len(outputs)} sequences, expected {len(batch)}."
                )
            records: list[dict[str, Any]] = []
            for task, output in zip(batch, outputs, strict=True):
                full_ids = [int(value) for value in output.prompt_token_ids]
                expected = list(task["prompt_token_ids"]) + list(task["response_token_ids"])
                if full_ids != expected:
                    raise RuntimeError(
                        f"vLLM changed the input token ids for {task['trajectory_id']}."
                    )
                prompt_logprobs = output.prompt_logprobs
                if prompt_logprobs is None or len(prompt_logprobs) != len(full_ids):
                    raise RuntimeError(
                        f"vLLM did not return complete prompt logprobs for "
                        f"{task['trajectory_id']}."
                    )
                prompt_len = len(task["prompt_token_ids"])
                for case in task["cases"]:
                    values = extract_step_logprob_values(
                        full_ids,
                        prompt_logprobs,
                        prompt_len=prompt_len,
                        start=case["step_token_start"],
                        end=case["step_token_end"],
                    )
                    summary = summarize_step(values)
                    record = {
                        "schema_version": SCHEMA_VERSION,
                        "role": role,
                        "model_path": model_path,
                        "trajectory_id": task["trajectory_id"],
                        "case_id": case["case_id"],
                        "case_index": case["case_index"],
                        "step_token_start": case["step_token_start"],
                        "step_token_end": case["step_token_end"],
                        **summary,
                    }
                    if store_token_logprobs:
                        record["token_logprobs"] = values
                    records.append(record)
            memory = assert_gpu_headroom(gpu_id, minimum_free_mib, stage=f"07-{role}-after-batch")
            result_queue.put(
                {
                    "type": "batch",
                    "role": role,
                    "rank": rank,
                    "gpu": gpu_id,
                    "batch_index": batch_index,
                    "trajectory_ids": [task["trajectory_id"] for task in batch],
                    "records": records,
                    "completed_trajectories": len(batch),
                    "completed_cases": len(records),
                    "elapsed_seconds": time.monotonic() - started,
                    "memory": memory,
                }
            )
        result_queue.put({"type": "done", "role": role, "rank": rank, "gpu": gpu_id})
    except BaseException as exc:
        result_queue.put(
            {
                "type": "error",
                "role": role,
                "rank": rank,
                "gpu": gpu_id,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if llm is not None:
            try:
                engine = getattr(llm, "llm_engine", None)
                core = getattr(engine, "engine_core", None)
                shutdown = getattr(core, "shutdown", None)
                if callable(shutdown):
                    shutdown()
                else:
                    fallback = getattr(engine, "shutdown", None)
                    if callable(fallback):
                        fallback()
            except Exception:
                pass
            del llm


def next_shard_index(shard_dir: Path, gpu_id: int, role_tag: str) -> int:
    prefix = f"gpu_{gpu_id}_{role_tag}_batch_"
    highest = -1
    for path in shard_dir.glob(f"{prefix}*.json"):
        name = path.name
        if not name.startswith(prefix) or not name.endswith(".json"):
            continue
        digits = name[len(prefix) : -len(".json")]
        if digits.isdigit():
            highest = max(highest, int(digits))
    return highest + 1


def completed_trajectories(shard_dir: Path, role_tag: str) -> set[str]:
    done: set[str] = set()
    for path in sorted(shard_dir.glob(f"gpu_*_{role_tag}_batch_*.json")):
        with open(path, encoding="utf-8") as handle:
            try:
                payload = json.load(handle)
            except Exception:
                continue
        for trajectory_id in payload.get("trajectory_ids", []):
            done.add(str(trajectory_id))
    return done


def run_role(
    *,
    role: str,
    model_path: str,
    tasks: list[dict[str, Any]],
    gpu_ids: Sequence[int],
    output_dir: Path,
    max_model_len: int,
    batch_size: int,
    max_num_batched_tokens: int,
    gpu_memory_utilization: float,
    minimum_free_mib: int,
    store_token_logprobs: bool,
) -> None:
    role_tag = _sanitize_role_tag(role)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    already = completed_trajectories(shard_dir, role_tag)
    pending = [task for task in tasks if task["trajectory_id"] not in already]
    role_root = output_dir / "roles" / role_tag
    role_root.mkdir(parents=True, exist_ok=True)
    total_items = len(tasks)
    completed_items = len(already)
    progress = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": utc_now(),
        "role": role,
        "model_path": model_path,
        "total_trajectories": total_items,
        "completed_trajectories": completed_items,
        "pending_trajectories": len(pending),
        "progress_fraction": completed_items / total_items if total_items else 1.0,
        "status": "running" if pending else "already-complete",
    }
    atomic_json(role_root / "progress.json", progress)
    print(json.dumps({"event": "07_role_started", **progress}, ensure_ascii=False), flush=True)
    if not pending:
        return

    context = mp.get_context("spawn")
    result_queue = context.Queue()
    processes: list[mp.Process] = []
    assignments = [pending[index :: len(gpu_ids)] for index in range(len(gpu_ids))]
    shard_offsets = {
        rank: next_shard_index(shard_dir, gpu_id, role_tag)
        for rank, gpu_id in enumerate(gpu_ids)
    }
    started = time.monotonic()
    ready: set[int] = set()
    done: set[int] = set()
    try:
        for rank, (gpu_id, assigned) in enumerate(zip(gpu_ids, assignments, strict=True)):
            process = context.Process(
                target=score_worker,
                kwargs={
                    "gpu_id": gpu_id,
                    "rank": rank,
                    "role": role,
                    "model_path": model_path,
                    "tasks": assigned,
                    "output_dir": output_dir,
                    "max_model_len": max_model_len,
                    "batch_size": batch_size,
                    "max_num_batched_tokens": max_num_batched_tokens,
                    "gpu_memory_utilization": gpu_memory_utilization,
                    "minimum_free_mib": minimum_free_mib,
                    "store_token_logprobs": store_token_logprobs,
                    "result_queue": result_queue,
                },
                daemon=False,
            )
            process.start()
            processes.append(process)
        while len(done) < len(processes):
            try:
                message = result_queue.get(timeout=5.0)
            except queue.Empty:
                failed = [
                    f"rank={rank}, exitcode={process.exitcode}"
                    for rank, process in enumerate(processes)
                    if process.exitcode not in (None, 0)
                ]
                if failed:
                    raise RuntimeError("Worker exited unexpectedly: " + "; ".join(failed))
                continue
            kind = message.get("type")
            if kind == "ready":
                ready.add(int(message["rank"]))
                print(
                    json.dumps({"event": "07_worker_ready", **message}, ensure_ascii=False),
                    flush=True,
                )
            elif kind == "error":
                raise RuntimeError(
                    f"07 role={message.get('role')} rank={message.get('rank')} failed: "
                    f"{message.get('error')}\n{message.get('traceback', '')}"
                )
            elif kind == "batch":
                rank = int(message["rank"])
                gpu_id = int(message["gpu"])
                shard_path = (
                    shard_dir
                    / f"gpu_{gpu_id}_{role_tag}_batch_{shard_offsets[rank] + int(message['batch_index']):06d}.json"
                )
                atomic_json(shard_path, message)
                completed_items += int(message["completed_trajectories"])
                elapsed = max(time.monotonic() - started, 1e-6)
                progress = {
                    "schema_version": SCHEMA_VERSION,
                    "updated_at": utc_now(),
                    "role": role,
                    "model_path": model_path,
                    "total_trajectories": total_items,
                    "completed_trajectories": completed_items,
                    "progress_fraction": completed_items / total_items,
                    "elapsed_seconds": elapsed,
                    "ready_workers": len(ready),
                    "done_workers": len(done),
                    "last_gpu_memory_mib": message.get("memory"),
                }
                atomic_json(role_root / "progress.json", progress)
                print(
                    json.dumps({"event": "07_role_progress", **progress}, ensure_ascii=False),
                    flush=True,
                )
            elif kind == "done":
                done.add(int(message["rank"]))
            else:
                raise RuntimeError(f"Unknown worker message: {message!r}")
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=30)
    failed = [
        f"rank={rank}, exitcode={process.exitcode}"
        for rank, process in enumerate(processes)
        if process.exitcode not in (0, None)
    ]
    if failed:
        raise RuntimeError("Worker processes failed: " + "; ".join(failed))
    if len(ready) != len(processes):
        raise RuntimeError(f"Only {len(ready)}/{len(processes)} workers became ready.")
    final = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": utc_now(),
        "role": role,
        "model_path": model_path,
        "total_trajectories": total_items,
        "completed_trajectories": completed_items,
        "progress_fraction": 1.0,
        "elapsed_seconds": max(time.monotonic() - started, 1e-6),
        "status": "complete",
    }
    atomic_json(role_root / "progress.json", final)
    print(json.dumps({"event": "07_role_complete", **final}, ensure_ascii=False), flush=True)


# ---------------------------------------------------------------------------
# Merge (pure, unit-testable)
# ---------------------------------------------------------------------------


def load_role_records(shard_dir: Path, role_tag: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(shard_dir.glob(f"gpu_*_{role_tag}_batch_*.json")):
        payload = load_json(path)
        for row in payload.get("records", []):
            case_id = str(row["case_id"])
            if case_id in records:
                raise ValueError(f"Duplicate {role_tag} score for {case_id}.")
            records[case_id] = row
    return records


def build_output_record(
    ersr_record: dict[str, Any],
    student_row: dict[str, Any],
    teacher_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    case_id = str(ersr_record["case_id"])
    if str(student_row["case_id"]) != case_id:
        raise ValueError(f"Student row mismatch for {case_id}.")
    reasoning_step_index = int(ersr_record["reasoning_step_index"])
    num_reasoning_steps = int(ersr_record["num_reasoning_steps"])
    student_mean = float(student_row["mean_logprob"])
    student_sum = float(student_row["sum_logprob"])
    teachers: dict[str, Any] = {}
    for name, row in teacher_rows.items():
        if str(row["case_id"]) != case_id:
            raise ValueError(f"Teacher row mismatch for {name}/{case_id}.")
        teachers[name] = {
            "mean_logprob": float(row["mean_logprob"]),
            "sum_logprob": float(row["sum_logprob"]),
            "D_TS_mean": float(row["mean_logprob"]) - student_mean,
            "D_TS_sum": float(row["sum_logprob"]) - student_sum,
        }
    values = ersr_record["values"]
    advantages = ersr_record["advantages"]
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "case_index": int(ersr_record["case_index"]),
        "trajectory_id": f"{int(ersr_record['response_idx'])}:{int(ersr_record['rollout_index'])}",
        "response_idx": int(ersr_record["response_idx"]),
        "rollout_index": int(ersr_record["rollout_index"]),
        "reward": int(ersr_record["rollout_correct"]),
        "bucket": int(ersr_record["bucket"]),
        "step_index": int(ersr_record["step_index"]),
        "reasoning_step_index": reasoning_step_index,
        "num_reasoning_steps": num_reasoning_steps,
        "relative_position": (reasoning_step_index + 1) / num_reasoning_steps,
        "step_token_start": int(ersr_record["step_token_start"]),
        "step_token_end": int(ersr_record["step_token_end"]),
        "step_num_tokens": int(student_row["num_tokens"]),
        "student": {
            "mean_logprob": student_mean,
            "sum_logprob": student_sum,
        },
        "teachers": teachers,
        "ersr": {
            "V_redecide": float(values["redecide"]),
            "V_keep": float(values["keep"]),
            "V_replace": {
                name: float(values[f"replace:{name}"]) for name in teacher_rows
            },
            "A_student": float(advantages["A_k"]),
            "A_teacher": {
                name: float(advantages[f"A_k_TR:{name}"]) for name in teacher_rows
            },
        },
    }


def merge_all(
    *,
    output_dir: Path,
    ersr_records: Sequence[dict[str, Any]],
    student_role_tag: str,
    teacher_role_tags: dict[str, str],
) -> list[dict[str, Any]]:
    shard_dir = output_dir / "shards"
    student_records = load_role_records(shard_dir, student_role_tag)
    teacher_records = {
        name: load_role_records(shard_dir, tag) for name, tag in teacher_role_tags.items()
    }
    rows: list[dict[str, Any]] = []
    for ersr_record in sorted(ersr_records, key=lambda value: int(value["case_index"])):
        case_id = str(ersr_record["case_id"])
        if case_id not in student_records:
            raise ValueError(f"Missing student score for {case_id}.")
        teacher_rows = {}
        for name, records in teacher_records.items():
            if case_id not in records:
                raise ValueError(f"Missing {name} score for {case_id}.")
            teacher_rows[name] = records[case_id]
        rows.append(
            build_output_record(ersr_record, student_records[case_id], teacher_rows)
        )
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_models_from_config(path: Path) -> tuple[str, list[dict[str, str]]]:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    student_path = REPO_ROOT / str(raw["student"]["model_path"])
    teachers = []
    for entry in raw["teachers"]:
        teachers.append(
            {
                "name": str(entry["name"]),
                "model_path": str((REPO_ROOT / str(entry["model_path"])).resolve()),
            }
        )
    names = [teacher["name"] for teacher in teachers]
    if len(set(names)) != len(names) or not names:
        raise ValueError(f"Teacher names must be non-empty and unique: {names}.")
    return str(student_path.resolve()), teachers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--gpus",
        default="auto",
        help="auto uses every GPU reported by nvidia-smi; WORLD_SIZE is ignored.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=8, help="trajectories per vLLM call"
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.78)
    parser.add_argument("--min-free-gpu-gib", type=float, default=0.0)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument(
        "--max-trajectories",
        type=int,
        default=0,
        help="smoke-only limit; zero processes every trajectory.",
    )
    parser.add_argument(
        "--store-token-logprobs",
        action="store_true",
        help="also persist every per-token logprob (large output)",
    )
    parser.add_argument("--merge-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise ValueError("--gpu-memory-utilization must lie in (0, 1).")
    if args.min_free_gpu_gib < 0.0:
        raise ValueError("--min-free-gpu-gib cannot be negative.")
    if args.max_num_batched_tokens <= 0 or args.max_trajectories < 0:
        raise ValueError("Token and trajectory limits must be positive/non-negative.")

    started = time.monotonic()
    results_path = args.results.resolve()
    output_dir = args.output_dir.resolve()
    if not results_path.is_file():
        raise FileNotFoundError(results_path)
    student_path, teachers = load_models_from_config(args.config.resolve())
    records = load_json(results_path)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a non-empty JSON array in {results_path}.")
    configured_names = {teacher["name"] for teacher in teachers}
    for record in records:
        if set(record.get("teacher_replacements", {})) != configured_names:
            raise ValueError(
                f"Teacher names in results do not match the config for "
                f"{record.get('case_id')}."
            )
    tasks = build_trajectory_tasks(records)
    if args.max_trajectories:
        tasks = tasks[: args.max_trajectories]
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu_ids = discover_local_gpus(str(args.gpus))
    minimum_free_mib = math.ceil(float(args.min_free_gpu_gib) * 1024)
    max_model_len = required_model_len(tasks)
    max_num_batched_tokens = max(int(args.max_num_batched_tokens), max_model_len)
    plan = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "script": "tests/collect_ersr_step_ts_logprobs_07.py",
        "results_path": str(results_path),
        "results_sha256": sha256_file(results_path),
        "output_dir": str(output_dir),
        "student_model": student_path,
        "teachers": teachers,
        "gpu_ids": gpu_ids,
        "trajectories": len(tasks),
        "selected_steps": sum(len(task["cases"]) for task in tasks),
        "max_model_len": max_model_len,
        "batch_size": int(args.batch_size),
        "store_token_logprobs": bool(args.store_token_logprobs),
        "definitions": {
            "window": "response_token_ids[step_token_start:step_token_end] of the selected Student step",
            "context": "prompt + full token prefix before each step token (matches ERSR splicing)",
            "D_TS_mean": "mean_t[log p_T(y_t) - log p_S(y_t)] over the step tokens",
            "roles": "one scoring pass per model per trajectory; Student first, then each Teacher",
        },
    }
    atomic_json(output_dir / "07_collection_plan.json", plan)
    print(
        json.dumps(
            {
                "event": "07_collection_plan",
                "trajectories": plan["trajectories"],
                "selected_steps": plan["selected_steps"],
                "student_model": student_path,
                "teachers": [teacher["name"] for teacher in teachers],
                "gpu_ids": gpu_ids,
                "max_model_len": max_model_len,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not args.merge_only:
        for role, model_path in [
            ("student", student_path),
            *[(teacher["name"], teacher["model_path"]) for teacher in teachers],
        ]:
            run_role(
                role=role,
                model_path=model_path,
                tasks=tasks,
                gpu_ids=gpu_ids,
                output_dir=output_dir,
                max_model_len=max_model_len,
                batch_size=int(args.batch_size),
                max_num_batched_tokens=max_num_batched_tokens,
                gpu_memory_utilization=float(args.gpu_memory_utilization),
                minimum_free_mib=minimum_free_mib,
                store_token_logprobs=bool(args.store_token_logprobs),
            )
    covered_cases = {case["case_id"] for task in tasks for case in task["cases"]}
    covered_records = [
        record for record in records if str(record["case_id"]) in covered_cases
    ]
    rows = merge_all(
        output_dir=output_dir,
        ersr_records=covered_records,
        student_role_tag=_sanitize_role_tag("student"),
        teacher_role_tags={
            teacher["name"]: _sanitize_role_tag(teacher["name"]) for teacher in teachers
        },
    )
    step_scores_path = output_dir / "07_step_ts_logprob.jsonl"
    atomic_jsonl(step_scores_path, rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "completed_at": utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "results_path": str(results_path),
        "student_model": student_path,
        "teachers": [teacher["name"] for teacher in teachers],
        "gpu_ids": gpu_ids,
        "trajectories": len(tasks),
        "cases": len(rows),
        "step_scores": str(step_scores_path),
        "step_scores_sha256": sha256_file(step_scores_path),
    }
    atomic_json(output_dir / "07_step_ts_logprob_summary.json", summary)
    print(
        json.dumps({"event": "07_step_ts_logprob_complete", **summary}, ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
