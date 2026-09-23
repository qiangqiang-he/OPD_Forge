#!/usr/bin/env python3
"""Teacher-model answer probes around the Student's own step (10).

Server-side GPU collector.  Every probe is built on the STUDENT's step, for
both the Student and the Teacher probe family alike.  For each ERSR-selected
step it scores, under every configured Teacher model, the gold-answer probe
at two states::

    before        prefix = prompt + response tokens before the step
    student_after prefix = prompt + tokens before the step + the Student's
                           own step tokens

    deltaP_T_student_step
        = P_T(answer | prompt + pre-step + student step)
        - P_T(answer | prompt + pre-step)

Positive values mean the Student's own step moved the TEACHER's belief
toward the gold answer.  The Teacher's replacement step is never touched.

The collector performs no analysis.  It writes one JSON record per case
(``10_teacher_student_step_probes.jsonl``) containing both states per
Teacher, the derived probability-space and log-space deltas, and the joined
ERSR utilities, so the file can be copied off the server and analyzed
locally.  Model phases run sequentially (one Teacher at a time), every GPU
writes atomic resumable shards, and the teardown grace window from analysis
07 is applied.
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
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.collect_ersr_answer_probes_03 import (
    _score_one,
    assert_gpu_headroom,
    atomic_json,
    atomic_jsonl,
    discover_local_gpus,
    load_json,
    prepare_task,
    sha256_file,
    utc_now,
)

DEFAULT_RESULTS = REPO_ROOT / "outputs/ersr_dapo17k_qwen3_1p7b_mc128/ersr_results.json"
DEFAULT_CONFIG = REPO_ROOT / "configs/ersr/ersr_dapo17k_qwen3_1p7b.yaml"
DEFAULT_TOKENIZER = REPO_ROOT / "models/Qwen3-1.7B"
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "outputs/ersr_dapo17k_qwen3_1p7b_mc128/teacher_student_step_probes_10"
)
SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Task construction (pure, unit-testable)
# ---------------------------------------------------------------------------


def build_probe_sequences(task: dict[str, Any]) -> list[dict[str, Any]]:
    """Two scoring sequences per case: before and after the Student's step."""

    prompt = list(task["prompt_token_ids"])
    pre = list(task["pre_step_response_token_ids"])
    step = list(task["student_step_token_ids"])
    probe = list(task["probe_token_ids"])
    local_positions = [int(value) for value in task["probe_answer_token_positions"]]

    def entry(kind: str, suffix: list[int]) -> dict[str, Any]:
        prefix = prompt + pre + suffix
        return {
            "kind": kind,
            "input_ids": prefix + probe,
            "answer_positions": [len(prefix) + position for position in local_positions],
        }

    return [entry("before", []), entry("student_after", step)]


def required_model_len(tasks: Sequence[dict[str, Any]]) -> int:
    longest = 0
    for task in tasks:
        base = (
            len(task["prompt_token_ids"])
            + len(task["pre_step_response_token_ids"])
            + len(task["student_step_token_ids"])
            + len(task["probe_token_ids"])
        )
        longest = max(longest, base)
    return max(8192, longest + 1)


def _sanitize_role_tag(role: str) -> str:
    tag = re.sub(r"[^0-9A-Za-z_.-]+", "_", role)
    if not tag or tag in {".", ".."}:
        raise ValueError(f"Unusable role name for shard files: {role!r}.")
    return tag


def _teacher_case_record(
    task: dict[str, Any],
    role: str,
    model_path: str,
    scores: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    before = scores["before"]
    after = scores["student_after"]
    p_before = float(before["geometric_mean_probability"])
    p_after = float(after["geometric_mean_probability"])
    return {
        "schema_version": SCHEMA_VERSION,
        "role": role,
        "model_path": model_path,
        "case_id": str(task["case_id"]),
        "case_index": int(task["case_index"]),
        "trajectory_id": str(task["trajectory_id"]),
        "probe": {"before": before, "student_after": after},
        "P_before": p_before,
        "P_student_after": p_after,
        "deltaP_T_student_step": p_after - p_before,
        "delta_logP_T_student_step": float(after["mean_logprob"])
        - float(before["mean_logprob"]),
    }


# ---------------------------------------------------------------------------
# GPU worker and role orchestration
# ---------------------------------------------------------------------------


def score_worker(
    *,
    gpu_id: int,
    rank: int,
    role: str,
    model_path: str,
    tasks: list[dict[str, Any]],
    max_model_len: int,
    batch_cases: int,
    max_num_batched_tokens: int,
    gpu_memory_utilization: float,
    minimum_free_mib: int,
    result_queue: Any,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    llm: Any | None = None
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        assert_gpu_headroom(gpu_id, minimum_free_mib, stage=f"10-{role}-before-load")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True, use_fast=True, local_files_only=True
        )
        vocabulary = len(tokenizer)
        for task in tasks:
            largest = max(
                max(task["prompt_token_ids"]),
                max(task["pre_step_response_token_ids"], default=0),
                max(task["student_step_token_ids"], default=0),
                max(task["probe_token_ids"]),
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
            max_num_seqs=batch_cases * 2,
            max_num_batched_tokens=max_num_batched_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=False,
            enable_chunked_prefill=True,
            enforce_eager=True,
            enable_sleep_mode=False,
        )
        memory = assert_gpu_headroom(gpu_id, minimum_free_mib, stage=f"10-{role}-after-load")
        result_queue.put(
            {"type": "ready", "role": role, "rank": rank, "gpu": gpu_id, "memory": memory}
        )
        params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
        for batch_index, offset in enumerate(range(0, len(tasks), batch_cases)):
            batch = tasks[offset : offset + batch_cases]
            sequences = [(task, entry) for task in batch for entry in build_probe_sequences(task)]
            prompts = [{"prompt_token_ids": entry["input_ids"]} for _task, entry in sequences]
            started = time.monotonic()
            outputs = llm.generate(prompts, sampling_params=params, use_tqdm=False)
            if len(outputs) != len(sequences):
                raise RuntimeError(
                    f"{role} returned {len(outputs)} sequences, expected {len(sequences)}."
                )
            scores_by_case: dict[str, dict[str, dict[str, Any]]] = {}
            for (task, entry), output in zip(sequences, outputs, strict=True):
                block = _score_one(output, list(entry["input_ids"]), entry["answer_positions"])
                scores_by_case.setdefault(str(task["case_id"]), {})[entry["kind"]] = block
            records = [
                _teacher_case_record(task, role, model_path, scores_by_case[str(task["case_id"])])
                for task in batch
            ]
            memory = assert_gpu_headroom(gpu_id, minimum_free_mib, stage=f"10-{role}-after-batch")
            result_queue.put(
                {
                    "type": "batch",
                    "role": role,
                    "rank": rank,
                    "gpu": gpu_id,
                    "batch_index": batch_index,
                    "case_ids": [str(task["case_id"]) for task in batch],
                    "records": records,
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
            import gc

            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:
                pass


def next_shard_index(shard_dir: Path, gpu_id: int, role_tag: str) -> int:
    prefix = f"gpu_{gpu_id}_{role_tag}_batch_"
    highest = -1
    for path in shard_dir.glob(f"{prefix}*.json"):
        digits = path.name[len(prefix) : -len(".json")]
        if path.name.startswith(prefix) and path.name.endswith(".json") and digits.isdigit():
            highest = max(highest, int(digits))
    return highest + 1


def completed_cases(shard_dir: Path, role_tag: str) -> set[str]:
    done: set[str] = set()
    for path in sorted(shard_dir.glob(f"gpu_*_{role_tag}_batch_*.json")):
        try:
            payload = load_json(path)
        except Exception:
            continue
        for case_id in payload.get("case_ids", []):
            done.add(str(case_id))
    return done


def run_role(
    *,
    role: str,
    model_path: str,
    tasks: list[dict[str, Any]],
    gpu_ids: Sequence[int],
    output_dir: Path,
    max_model_len: int,
    batch_cases: int,
    max_num_batched_tokens: int,
    gpu_memory_utilization: float,
    minimum_free_mib: int,
) -> None:
    role_tag = _sanitize_role_tag(role)
    shard_dir = output_dir / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    already = completed_cases(shard_dir, role_tag)
    pending = [task for task in tasks if str(task["case_id"]) not in already]
    role_root = output_dir / "roles" / role_tag
    role_root.mkdir(parents=True, exist_ok=True)
    total_items = len(tasks)
    completed_items = len(already)
    progress = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": utc_now(),
        "role": role,
        "model_path": model_path,
        "total_cases": total_items,
        "completed_cases": completed_items,
        "pending_cases": len(pending),
        "progress_fraction": completed_items / total_items if total_items else 1.0,
        "status": "running" if pending else "already-complete",
    }
    atomic_json(role_root / "progress.json", progress)
    print(json.dumps({"event": "10_role_started", **progress}, ensure_ascii=False), flush=True)
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
    completed_normally = False
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
                    "max_model_len": max_model_len,
                    "batch_cases": batch_cases,
                    "max_num_batched_tokens": max_num_batched_tokens,
                    "gpu_memory_utilization": gpu_memory_utilization,
                    "minimum_free_mib": minimum_free_mib,
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
                    json.dumps({"event": "10_worker_ready", **message}, ensure_ascii=False),
                    flush=True,
                )
            elif kind == "error":
                raise RuntimeError(
                    f"10 role={message.get('role')} rank={message.get('rank')} failed: "
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
                completed_items += int(message["completed_cases"])
                elapsed = max(time.monotonic() - started, 1e-6)
                progress = {
                    "schema_version": SCHEMA_VERSION,
                    "updated_at": utc_now(),
                    "role": role,
                    "model_path": model_path,
                    "total_cases": total_items,
                    "completed_cases": completed_items,
                    "progress_fraction": completed_items / total_items,
                    "elapsed_seconds": elapsed,
                    "ready_workers": len(ready),
                    "done_workers": len(done),
                    "last_gpu_memory_mib": message.get("memory"),
                }
                atomic_json(role_root / "progress.json", progress)
                print(
                    json.dumps({"event": "10_role_progress", **progress}, ensure_ascii=False),
                    flush=True,
                )
            elif kind == "done":
                done.add(int(message["rank"]))
            else:
                raise RuntimeError(f"Unknown worker message: {message!r}")
        completed_normally = True
    finally:
        deadline = time.monotonic() + (180.0 if completed_normally else 0.0)
        for process in processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        terminated: list[int] = []
        for rank_index, process in enumerate(processes):
            if process.is_alive():
                process.terminate()
                terminated.append(rank_index)
        for process in processes:
            process.join(timeout=30)
    if terminated:
        print(
            json.dumps(
                {
                    "event": "10_worker_shutdown_terminated",
                    "role": role,
                    "ranks": terminated,
                    "note": "workers already reported done; teardown terminated after grace window",
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    exit_report = [
        f"rank={rank}, exitcode={process.exitcode}"
        for rank, process in enumerate(processes)
        if process.exitcode not in (0, None)
    ]
    if exit_report:
        print(
            json.dumps(
                {"event": "10_worker_exit_codes_after_done", "role": role, "workers": exit_report},
                ensure_ascii=False,
            ),
            flush=True,
        )
    if len(ready) != len(processes):
        raise RuntimeError(f"Only {len(ready)}/{len(processes)} workers became ready.")
    final = {
        "schema_version": SCHEMA_VERSION,
        "updated_at": utc_now(),
        "role": role,
        "model_path": model_path,
        "total_cases": total_items,
        "completed_cases": completed_items,
        "progress_fraction": 1.0,
        "elapsed_seconds": max(time.monotonic() - started, 1e-6),
        "status": "complete",
    }
    atomic_json(role_root / "progress.json", final)
    print(json.dumps({"event": "10_role_complete", **final}, ensure_ascii=False), flush=True)


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


def build_output_record(task: dict[str, Any], teacher_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    case_id = str(task["case_id"])
    teachers: dict[str, Any] = {}
    for name, row in teacher_rows.items():
        if str(row["case_id"]) != case_id:
            raise ValueError(f"Teacher row mismatch for {name}/{case_id}.")
        base = task["teachers"][name]
        teachers[name] = {
            "P_before": float(row["P_before"]),
            "P_student_after": float(row["P_student_after"]),
            "deltaP_T_student_step": float(row["deltaP_T_student_step"]),
            "delta_logP_T_student_step": float(row["delta_logP_T_student_step"]),
            "A_teacher": float(base["A_teacher"]),
            "V_replace": float(base["V_teacher_after"]),
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "case_index": int(task["case_index"]),
        "trajectory_id": str(task["trajectory_id"]),
        "response_idx": int(task["response_idx"]),
        "rollout_index": int(task["rollout_index"]),
        "reward": int(task["reward"]),
        "bucket": int(task["bucket"]),
        "step_id": int(task["step_id"]),
        "num_steps": int(task["num_steps"]),
        "relative_position": float(task["relative_position"]),
        "mc_k": int(task["mc_k"]),
        "probe_text": str(task["probe_text"]),
        "student": {
            "V_redecide": float(task["V_before"]),
            "V_keep": float(task["V_student_after"]),
            "A_student": float(task["A_student"]),
        },
        "teachers": teachers,
    }


def merge_all(
    *,
    output_dir: Path,
    tasks: Sequence[dict[str, Any]],
    teacher_role_tags: dict[str, str],
) -> list[dict[str, Any]]:
    shard_dir = output_dir / "shards"
    role_records = {
        name: load_role_records(shard_dir, tag) for name, tag in teacher_role_tags.items()
    }
    rows: list[dict[str, Any]] = []
    for task in sorted(tasks, key=lambda value: int(value["case_index"])):
        case_id = str(task["case_id"])
        teacher_rows = {}
        for name, records in role_records.items():
            if case_id not in records:
                raise ValueError(f"Missing {name} score for {case_id}.")
            teacher_rows[name] = records[case_id]
        rows.append(build_output_record(task, teacher_rows))
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def load_teachers_from_config(path: Path) -> list[dict[str, str]]:
    import yaml

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    teachers = [
        {
            "name": str(entry["name"]),
            "model_path": str((REPO_ROOT / str(entry["model_path"])).resolve()),
        }
        for entry in raw["teachers"]
    ]
    names = [teacher["name"] for teacher in teachers]
    if len(set(names)) != len(names) or not names:
        raise ValueError(f"Teacher names must be non-empty and unique: {names}.")
    return teachers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--gpus",
        default="auto",
        help="auto uses every GPU reported by nvidia-smi; WORLD_SIZE is ignored.",
    )
    parser.add_argument("--batch-cases", type=int, default=16, help="cases per vLLM call")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.78)
    parser.add_argument("--min-free-gpu-gib", type=float, default=0.0)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="smoke-only limit; zero processes every case.",
    )
    parser.add_argument("--merge-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_cases <= 0:
        raise ValueError("--batch-cases must be positive.")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise ValueError("--gpu-memory-utilization must lie in (0, 1).")
    if args.min_free_gpu_gib < 0.0:
        raise ValueError("--min-free-gpu-gib cannot be negative.")
    if args.max_num_batched_tokens <= 0 or args.max_cases < 0:
        raise ValueError("Token and case limits must be positive/non-negative.")

    started = time.monotonic()
    results_path = args.results.resolve()
    output_dir = args.output_dir.resolve()
    if not results_path.is_file():
        raise FileNotFoundError(results_path)
    teachers = load_teachers_from_config(args.config.resolve())
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer.resolve()),
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )
    records = load_json(results_path)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a non-empty JSON array in {results_path}.")
    tasks = [prepare_task(record, tokenizer) for record in records]
    if args.max_cases:
        tasks = tasks[: args.max_cases]
    output_dir.mkdir(parents=True, exist_ok=True)
    gpu_ids = discover_local_gpus(str(args.gpus))
    minimum_free_mib = math.ceil(float(args.min_free_gpu_gib) * 1024)
    max_model_len = required_model_len(tasks)
    max_num_batched_tokens = max(int(args.max_num_batched_tokens), max_model_len)
    plan = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "script": "tests/collect_ersr_teacher_student_step_probes_10.py",
        "results_path": str(results_path),
        "results_sha256": sha256_file(results_path),
        "tokenizer": str(args.tokenizer.resolve()),
        "output_dir": str(output_dir),
        "teachers": teachers,
        "gpu_ids": gpu_ids,
        "cases": len(tasks),
        "states_per_case": 2,
        "max_model_len": max_model_len,
        "batch_cases": int(args.batch_cases),
        "definitions": {
            "before": "P_T(gold answer) with prefix prompt + tokens before the Student step",
            "student_after": "P_T(gold answer) with the Student's own step appended",
            "deltaP_T_student_step": "P_T(answer|student step) - P_T(answer|before); the Student's step as evaluated by the Teacher model; the Teacher replacement step is never used",
        },
    }
    atomic_json(output_dir / "10_collection_plan.json", plan)
    print(
        json.dumps(
            {
                "event": "10_collection_plan",
                "cases": plan["cases"],
                "teachers": [teacher["name"] for teacher in teachers],
                "gpu_ids": gpu_ids,
                "max_model_len": max_model_len,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if not args.merge_only:
        for teacher in teachers:
            run_role(
                role=teacher["name"],
                model_path=teacher["model_path"],
                tasks=tasks,
                gpu_ids=gpu_ids,
                output_dir=output_dir,
                max_model_len=max_model_len,
                batch_cases=int(args.batch_cases),
                max_num_batched_tokens=max_num_batched_tokens,
                gpu_memory_utilization=float(args.gpu_memory_utilization),
                minimum_free_mib=minimum_free_mib,
            )
    rows = merge_all(
        output_dir=output_dir,
        tasks=tasks,
        teacher_role_tags={
            teacher["name"]: _sanitize_role_tag(teacher["name"]) for teacher in teachers
        },
    )
    output_path = output_dir / "10_teacher_student_step_probes.jsonl"
    atomic_jsonl(output_path, rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "completed_at": utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "results_path": str(results_path),
        "teachers": [teacher["name"] for teacher in teachers],
        "gpu_ids": gpu_ids,
        "cases": len(rows),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    atomic_json(output_dir / "10_teacher_student_step_probes_summary.json", summary)
    print(
        json.dumps({"event": "10_teacher_student_step_probes_complete", **summary}, ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
