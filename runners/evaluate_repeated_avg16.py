#!/usr/bin/env python3
"""Run six reproducible thinking-mode Avg@16 evaluations per Qwen3 model.

The evaluator deliberately generates exactly 256 rollouts in every vLLM call:
16 question prompts x 16 samples.  Batch shards are written atomically, so an
interrupted run can resume without regenerating completed work.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODELS = (
    "Qwen3-0.6B",
    "Qwen3-1.7B",
    "Qwen3-4B",
    "Qwen3-8B",
    "Qwen3-30B-A3B",
)
DEFAULT_DATASETS = (
    ("AIME24", WORKSPACE_ROOT / "FrugalRL/data/eval/AIME-2024.json", 30),
    ("AIME25", WORKSPACE_ROOT / "FrugalRL/data/eval/AIME-2025.json", 30),
    ("AIME26", WORKSPACE_ROOT / "FrugalRL/data/eval/AIME-2026.json", 30),
    ("AMC23", WORKSPACE_ROOT / "FrugalRL/data/eval/AMC-2023.json", 40),
    ("MATH500", WORKSPACE_ROOT / "FrugalRL/data/eval/MATH500.json", 500),
)
DEFAULT_SEEDS = (42, 43, 44, 45, 46, 47)
SAMPLES_PER_QUESTION = 16
QUESTIONS_PER_BATCH = 16
ROLLOUTS_PER_BATCH = SAMPLES_PER_QUESTION * QUESTIONS_PER_BATCH


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def load_tasks() -> list[dict[str, Any]]:
    from utils.prompts import render_prompt

    tasks: list[dict[str, Any]] = []
    for dataset_name, dataset_path, expected_questions in DEFAULT_DATASETS:
        if not dataset_path.is_file():
            raise FileNotFoundError(dataset_path)
        with dataset_path.open(encoding="utf-8") as handle:
            examples = json.load(handle)
        if len(examples) != expected_questions:
            raise ValueError(
                f"{dataset_name}: expected {expected_questions} questions, found {len(examples)}"
            )
        for question_index, example in enumerate(examples):
            tasks.append(
                {
                    "dataset": dataset_name,
                    "question_index": question_index,
                    "answer": str(example["answer"]),
                    "prompt": render_prompt(
                        "qwen3_thinking_prompt", question=str(example["question"])
                    ),
                }
            )
    return tasks


def make_batches(tasks: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    batches = [
        tasks[start : start + QUESTIONS_PER_BATCH]
        for start in range(0, len(tasks), QUESTIONS_PER_BATCH)
    ]
    if not batches:
        raise ValueError("No evaluation tasks were loaded.")
    return batches


def batch_output_path(output_root: Path, model: str, seed: int, batch_id: int) -> Path:
    return output_root / slug(model) / f"seed_{seed}" / "batches" / f"batch_{batch_id:02d}.json"


def validate_batch_payload(
    payload: dict[str, Any], model: str, seed: int, batch_id: int, expected_real_questions: int
) -> None:
    expected_records = expected_real_questions * SAMPLES_PER_QUESTION
    checks = {
        "model": model,
        "seed": seed,
        "batch_id": batch_id,
        "generated_rollouts": ROLLOUTS_PER_BATCH,
        "kept_rollouts": expected_records,
    }
    for key, expected in checks.items():
        if payload.get(key) != expected:
            raise ValueError(
                f"Invalid completed shard {model}/seed={seed}/batch={batch_id}: "
                f"{key}={payload.get(key)!r}, expected {expected!r}"
            )
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != expected_records:
        raise ValueError(
            f"Invalid completed shard {model}/seed={seed}/batch={batch_id}: "
            f"expected {expected_records} records"
        )


def load_valid_batch(
    path: Path, model: str, seed: int, batch_id: int, expected_real_questions: int
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    validate_batch_payload(payload, model, seed, batch_id, expected_real_questions)
    return payload


def worker(
    gpu_id: int,
    model_name: str,
    model_path: str,
    job_queue: Any,
    result_queue: Any,
    max_new_tokens: int,
    gpu_memory_utilization: float,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        from utils.math_verifier import verify_response_answer
        from utils.prompts import render_prompt

        AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=max_new_tokens + 2048,
            max_num_seqs=ROLLOUTS_PER_BATCH,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=True,
            seed=0,
        )
        result_queue.put(
            {"kind": "ready", "ok": True, "gpu": gpu_id, "model": model_name, "time": utc_now()}
        )
    except BaseException as exc:
        result_queue.put(
            {
                "kind": "ready",
                "ok": False,
                "gpu": gpu_id,
                "model": model_name,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return

    while True:
        job = job_queue.get()
        if job is None:
            return
        try:
            real_tasks = job["tasks"]
            if not 1 <= len(real_tasks) <= QUESTIONS_PER_BATCH:
                raise ValueError(f"Invalid real batch size: {len(real_tasks)}")
            padded_tasks = list(real_tasks)
            padding_task = {
                "dataset": "__padding__",
                "question_index": -1,
                "answer": "2",
                "prompt": render_prompt("qwen3_thinking_prompt", question="What is 1+1?"),
            }
            while len(padded_tasks) < QUESTIONS_PER_BATCH:
                padded_tasks.append(padding_task)
            if len(padded_tasks) * SAMPLES_PER_QUESTION != ROLLOUTS_PER_BATCH:
                raise AssertionError("Every generation call must contain exactly 256 rollouts.")

            sampling = SamplingParams(
                n=SAMPLES_PER_QUESTION,
                temperature=0.6,
                top_p=0.95,
                top_k=-1,
                max_tokens=max_new_tokens,
                seed=int(job["seed"]),
            )
            started_at = utc_now()
            started_monotonic = time.monotonic()
            outputs = llm.generate(
                [task["prompt"] for task in padded_tasks],
                sampling,
                use_tqdm=False,
            )
            if len(outputs) != QUESTIONS_PER_BATCH:
                raise RuntimeError(
                    f"Expected {QUESTIONS_PER_BATCH} request outputs, received {len(outputs)}"
                )

            records: list[dict[str, Any]] = []
            for task, request_output in zip(real_tasks, outputs[: len(real_tasks)], strict=True):
                if len(request_output.outputs) != SAMPLES_PER_QUESTION:
                    raise RuntimeError(
                        f"Expected {SAMPLES_PER_QUESTION} samples for "
                        f"{task['dataset']}/{task['question_index']}, "
                        f"received {len(request_output.outputs)}"
                    )
                for sample_index, completion in enumerate(request_output.outputs):
                    response = completion.text
                    records.append(
                        {
                            "dataset": task["dataset"],
                            "question_index": task["question_index"],
                            "sample_index": sample_index,
                            "correct": int(verify_response_answer(response, task["answer"])),
                            "num_response_tokens": len(completion.token_ids),
                            "finish_reason": completion.finish_reason,
                            "response": response,
                        }
                    )
            payload = {
                "model": model_name,
                "model_path": model_path,
                "seed": int(job["seed"]),
                "batch_id": int(job["batch_id"]),
                "gpu": gpu_id,
                "thinking": True,
                "n": SAMPLES_PER_QUESTION,
                "real_questions": len(real_tasks),
                "padded_questions": QUESTIONS_PER_BATCH - len(real_tasks),
                "generated_rollouts": ROLLOUTS_PER_BATCH,
                "kept_rollouts": len(records),
                "started_at": started_at,
                "finished_at": utc_now(),
                "elapsed_seconds": time.monotonic() - started_monotonic,
                "records": records,
            }
            atomic_json(Path(job["output_path"]), payload)
            result_queue.put(
                {
                    "kind": "batch",
                    "ok": True,
                    "gpu": gpu_id,
                    "model": model_name,
                    "seed": int(job["seed"]),
                    "batch_id": int(job["batch_id"]),
                    "output_path": job["output_path"],
                    "elapsed_seconds": payload["elapsed_seconds"],
                }
            )
        except BaseException as exc:
            result_queue.put(
                {
                    "kind": "batch",
                    "ok": False,
                    "gpu": gpu_id,
                    "model": model_name,
                    "seed": job.get("seed"),
                    "batch_id": job.get("batch_id"),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            return


def worker_failure(processes: list[mp.Process]) -> str | None:
    failures = [
        f"pid={process.pid}, exitcode={process.exitcode}"
        for process in processes
        if process.exitcode not in (None, 0)
    ]
    return "; ".join(failures) if failures else None


def await_messages(
    result_queue: Any,
    processes: list[mp.Process],
    count: int,
    expected_kind: str,
    model_name: str,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    last_heartbeat = time.monotonic()
    while len(messages) < count:
        try:
            message = result_queue.get(timeout=15)
        except queue.Empty:
            failure = worker_failure(processes)
            if failure:
                raise RuntimeError(f"Worker exited unexpectedly while evaluating {model_name}: {failure}")
            if time.monotonic() - last_heartbeat >= 60:
                suffix = f", seed={seed}" if seed is not None else ""
                print(
                    f"HEARTBEAT {utc_now()} model={model_name}{suffix} "
                    f"waiting={count - len(messages)}",
                    flush=True,
                )
                last_heartbeat = time.monotonic()
            continue
        if message.get("kind") != expected_kind:
            raise RuntimeError(f"Unexpected worker message: {message}")
        if not message.get("ok"):
            raise RuntimeError(
                f"GPU {message.get('gpu')} failed for {model_name}: {message.get('error')}\n"
                f"{message.get('traceback', '')}"
            )
        messages.append(message)
        if expected_kind == "ready":
            print(f"READY model={model_name} gpu={message['gpu']}", flush=True)
        else:
            print(
                f"DONE model={model_name} seed={message['seed']} batch={message['batch_id']} "
                f"gpu={message['gpu']} elapsed={message['elapsed_seconds']:.1f}s "
                f"completed={len(messages)}/{count}",
                flush=True,
            )
    return messages


def summarize_seed(
    output_root: Path,
    model_name: str,
    seed: int,
    batches: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    shard_paths: list[str] = []
    for batch_id, batch in enumerate(batches):
        path = batch_output_path(output_root, model_name, seed, batch_id)
        payload = load_valid_batch(path, model_name, seed, batch_id, len(batch))
        if payload is None:
            raise RuntimeError(f"Missing completed shard: {path}")
        records.extend(payload["records"])
        shard_paths.append(str(path))

    results = []
    expected_keys = {
        (dataset_name, question_index, sample_index)
        for dataset_name, _, question_count in DEFAULT_DATASETS
        for question_index in range(question_count)
        for sample_index in range(SAMPLES_PER_QUESTION)
    }
    actual_keys = {
        (record["dataset"], record["question_index"], record["sample_index"])
        for record in records
    }
    if actual_keys != expected_keys or len(records) != len(expected_keys):
        raise RuntimeError(
            f"{model_name}/seed={seed}: incomplete or duplicate records "
            f"({len(records)} records, {len(actual_keys)} unique, expected {len(expected_keys)})"
        )
    for dataset_name, _, question_count in DEFAULT_DATASETS:
        selected = [record for record in records if record["dataset"] == dataset_name]
        results.append(
            {
                "dataset": dataset_name,
                "num_questions": question_count,
                "num_rollouts": len(selected),
                "Avg@16": sum(record["correct"] for record in selected) / len(selected),
                "mean_response_tokens": sum(
                    record["num_response_tokens"] for record in selected
                )
                / len(selected),
                "truncation_rate": sum(
                    record["finish_reason"] == "length" for record in selected
                )
                / len(selected),
            }
        )
    summary = {
        "model": model_name,
        "seed": seed,
        "thinking": True,
        "n": SAMPLES_PER_QUESTION,
        "completed_at": utc_now(),
        "results": results,
        "batch_shards": shard_paths,
    }
    summary_path = output_root / slug(model_name) / f"seed_{seed}" / "summary.json"
    atomic_json(summary_path, summary)
    return summary


def summarize_model(output_root: Path, model_name: str, seeds: list[int]) -> dict[str, Any]:
    seed_summaries = []
    for seed in seeds:
        path = output_root / slug(model_name) / f"seed_{seed}" / "summary.json"
        with path.open(encoding="utf-8") as handle:
            seed_summaries.append(json.load(handle))

    aggregates = []
    for dataset_name, _, _ in DEFAULT_DATASETS:
        rows = [
            next(row for row in summary["results"] if row["dataset"] == dataset_name)
            for summary in seed_summaries
        ]
        accuracies = [float(row["Avg@16"]) for row in rows]
        response_lengths = [float(row["mean_response_tokens"]) for row in rows]
        aggregates.append(
            {
                "dataset": dataset_name,
                "num_independent_evaluations": len(rows),
                "Avg@16": {
                    "min": min(accuracies),
                    "max": max(accuracies),
                    "mean": sum(accuracies) / len(accuracies),
                    "range_max_minus_min": max(accuracies) - min(accuracies),
                    "values_by_seed": {
                        str(seed): value for seed, value in zip(seeds, accuracies, strict=True)
                    },
                },
                "mean_response_tokens": {
                    "min": min(response_lengths),
                    "max": max(response_lengths),
                    "mean": sum(response_lengths) / len(response_lengths),
                    "values_by_seed": {
                        str(seed): value
                        for seed, value in zip(seeds, response_lengths, strict=True)
                    },
                },
            }
        )
    result = {
        "model": model_name,
        "seeds": seeds,
        "thinking": True,
        "n": SAMPLES_PER_QUESTION,
        "completed_at": utc_now(),
        "aggregates": aggregates,
    }
    atomic_json(output_root / slug(model_name) / "aggregate.json", result)
    return result


def write_global_summary(output_root: Path, models: list[str]) -> None:
    values = []
    for model_name in models:
        path = output_root / slug(model_name) / "aggregate.json"
        if path.exists():
            with path.open(encoding="utf-8") as handle:
                values.append(json.load(handle))
    atomic_json(
        output_root / "summary.json",
        {
            "completed_at": utc_now(),
            "models": values,
        },
    )


def stop_workers(job_queue: Any, processes: list[mp.Process]) -> None:
    for _ in processes:
        job_queue.put(None)
    for process in processes:
        process.join(timeout=30)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)


def evaluate_model(
    output_root: Path,
    model_name: str,
    batches: list[list[dict[str, Any]]],
    seeds: list[int],
    gpu_ids: list[int],
    max_new_tokens: int,
    gpu_memory_utilization: float,
) -> None:
    aggregate_path = output_root / slug(model_name) / "aggregate.json"
    if aggregate_path.exists():
        print(f"SKIP model={model_name}: aggregate already exists", flush=True)
        return
    model_path = WORKSPACE_ROOT / "FrugalRL/models" / model_name
    if not model_path.is_dir():
        raise FileNotFoundError(model_path)

    pending_exists = any(
        load_valid_batch(
            batch_output_path(output_root, model_name, seed, batch_id),
            model_name,
            seed,
            batch_id,
            len(batch),
        )
        is None
        for seed in seeds
        for batch_id, batch in enumerate(batches)
    )
    if not pending_exists:
        for seed in seeds:
            summarize_seed(output_root, model_name, seed, batches)
        summarize_model(output_root, model_name, seeds)
        return

    worker_count = min(len(gpu_ids), len(batches))
    selected_gpus = gpu_ids[:worker_count]
    context = mp.get_context("spawn")
    job_queue = context.Queue()
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=worker,
            args=(
                gpu_id,
                model_name,
                str(model_path.resolve()),
                job_queue,
                result_queue,
                max_new_tokens,
                gpu_memory_utilization,
            ),
            name=f"eval-{slug(model_name)}-gpu{gpu_id}",
        )
        for gpu_id in selected_gpus
    ]
    print(
        f"LOAD model={model_name} workers={worker_count} gpus={selected_gpus} "
        f"rollouts_per_call={ROLLOUTS_PER_BATCH}",
        flush=True,
    )
    for process in processes:
        process.start()
    try:
        await_messages(result_queue, processes, worker_count, "ready", model_name)
        for seed_index, seed in enumerate(seeds, start=1):
            pending = []
            for batch_id, batch in enumerate(batches):
                output_path = batch_output_path(output_root, model_name, seed, batch_id)
                if load_valid_batch(output_path, model_name, seed, batch_id, len(batch)) is None:
                    pending.append((batch_id, batch, output_path))
            print(
                f"SEED_START model={model_name} seed={seed} index={seed_index}/{len(seeds)} "
                f"pending_batches={len(pending)}",
                flush=True,
            )
            for batch_id, batch, output_path in pending:
                job_queue.put(
                    {
                        "seed": seed,
                        "batch_id": batch_id,
                        "tasks": batch,
                        "output_path": str(output_path),
                    }
                )
            if pending:
                await_messages(
                    result_queue,
                    processes,
                    len(pending),
                    "batch",
                    model_name,
                    seed,
                )
            seed_summary = summarize_seed(output_root, model_name, seed, batches)
            concise = {
                row["dataset"]: {
                    "Avg@16": row["Avg@16"],
                    "mean_response_tokens": row["mean_response_tokens"],
                }
                for row in seed_summary["results"]
            }
            print(
                f"SEED_DONE model={model_name} seed={seed} results="
                f"{json.dumps(concise, ensure_ascii=False)}",
                flush=True,
            )
        summarize_model(output_root, model_name, seeds)
        print(f"MODEL_DONE model={model_name} aggregate={aggregate_path}", flush=True)
    finally:
        stop_workers(job_queue, processes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS, default=list(DEFAULT_MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "outputs/repeated_avg16_thinking_5datasets",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs and print the execution plan without loading a model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(args.seeds) != 6 or len(set(args.seeds)) != 6:
        raise ValueError("Exactly six distinct seeds are required.")
    if len(set(args.gpus)) != len(args.gpus) or any(gpu < 0 for gpu in args.gpus):
        raise ValueError("GPU IDs must be distinct non-negative integers.")
    if not 0 < args.gpu_memory_utilization < 1:
        raise ValueError("--gpu-memory-utilization must be between 0 and 1.")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be positive.")

    tasks = load_tasks()
    batches = make_batches(tasks)
    if len(tasks) != 630 or len(batches) != 40:
        raise AssertionError(f"Expected 630 tasks in 40 batches, got {len(tasks)} in {len(batches)}")
    plan = {
        "models": args.models,
        "datasets": [name for name, _, _ in DEFAULT_DATASETS],
        "seeds": args.seeds,
        "gpus_available": args.gpus,
        "gpus_used_per_model": args.gpus[: min(len(args.gpus), len(batches))],
        "thinking": True,
        "samples_per_question": SAMPLES_PER_QUESTION,
        "questions_per_batch": QUESTIONS_PER_BATCH,
        "rollouts_per_gpu_call": ROLLOUTS_PER_BATCH,
        "batches_per_seed": len(batches),
        "max_new_tokens": args.max_new_tokens,
        "output_root": str(args.output_root.resolve()),
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.validate_only:
        return

    args.output_root.mkdir(parents=True, exist_ok=True)
    previous_failure = args.output_root / "FAILURE.json"
    if previous_failure.exists():
        archived_failure = args.output_root / f"FAILURE.previous.{int(time.time())}.json"
        os.replace(previous_failure, archived_failure)
        print(f"ARCHIVED_FAILURE source={previous_failure} target={archived_failure}", flush=True)
    atomic_json(args.output_root / "plan.json", plan)
    try:
        for model_index, model_name in enumerate(args.models, start=1):
            print(
                f"MODEL_START model={model_name} index={model_index}/{len(args.models)} time={utc_now()}",
                flush=True,
            )
            evaluate_model(
                args.output_root,
                model_name,
                batches,
                list(args.seeds),
                list(args.gpus),
                args.max_new_tokens,
                args.gpu_memory_utilization,
            )
            write_global_summary(args.output_root, list(args.models))
        success = {
            "completed_at": utc_now(),
            "models": args.models,
            "seeds": args.seeds,
            "summary": str((args.output_root / "summary.json").resolve()),
        }
        atomic_json(args.output_root / "SUCCESS.json", success)
        print(f"ALL_DONE {json.dumps(success, ensure_ascii=False)}", flush=True)
    except BaseException as exc:
        atomic_json(
            args.output_root / "FAILURE.json",
            {
                "failed_at": utc_now(),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
