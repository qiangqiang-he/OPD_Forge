#!/usr/bin/env python3
"""Collect Student answer probes for ERSR analysis 03 on every local GPU.

The collector reads the finalized ``ersr_results.json`` and scores, with the
Student model, the gold-answer probe at three states for every selected step:

* before the original Student step;
* after the original Student step;
* after each configured Teacher replacement.

Original Student prefixes are exact slices of the token IDs persisted by ERSR.
Teacher replacement text is encoded by the Student tokenizer exactly as in
``tests/run_ersr_mc.py``.  The scored probability is ``exp(mean_logprob)`` over
the gold-answer-overlap tokens selected by ``utils.oa_opd.build_answer_probe``.

GPU policy is deliberately local and non-distributed: ``--gpus auto`` queries
``nvidia-smi`` and starts one tensor-parallel-size-1 vLLM replica on every
locally visible physical GPU.  ``WORLD_SIZE`` and distributed environment
variables are neither read nor validated.

Outputs are numbered ``03`` and resumable at atomic per-GPU batch shards.  No
source result or shard is modified.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SCHEMA_VERSION = 1
GROUP_SIZE_G = 5
DEFAULT_RESULTS = REPO_ROOT / "outputs/ersr_dapo17k_qwen3_1p7b_mc128/ersr_results.json"
DEFAULT_MODEL = REPO_ROOT / "models/Qwen3-1.7B"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs/ersr_dapo17k_qwen3_1p7b_mc128/probe_analysis_03"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"Expected an object at {path}:{line_number}.")
            rows.append(value)
    return rows


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def token_ids_sha256(values: Iterable[int]) -> str:
    digest = hashlib.sha256()
    digest.update(b"[")
    first = True
    for value in values:
        if not first:
            digest.update(b",")
        digest.update(str(int(value)).encode("ascii"))
        first = False
    digest.update(b"]")
    return digest.hexdigest()


def _int_ids(values: Iterable[Any], *, label: str) -> list[int]:
    raw = list(values)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in raw):
        raise TypeError(f"{label} contains a non-integer token ID.")
    result = [int(value) for value in raw]
    if any(value < 0 for value in result):
        raise ValueError(f"{label} contains a negative token ID.")
    return result


def _required_sample(samples: dict[str, Any], case_id: str, arm: str) -> dict[str, Any]:
    value = samples.get(arm)
    if not isinstance(value, dict):
        raise ValueError(f"Case {case_id!r} is missing sample arm {arm!r}.")
    for field in ("mc_k", "correct_count"):
        if field not in value:
            raise ValueError(f"Case {case_id!r}/{arm!r} is missing {field}.")
    return value


def _sample_value(sample: dict[str, Any], *, case_id: str, arm: str, mc_k: int) -> float:
    if int(sample["mc_k"]) != mc_k:
        raise ValueError(f"Case {case_id!r}/{arm!r} has inconsistent mc_k.")
    count = int(sample["correct_count"])
    if not 0 <= count <= mc_k:
        raise ValueError(f"Case {case_id!r}/{arm!r} has invalid correct_count={count}.")
    return count / mc_k


def prepare_task(
    record: dict[str, Any],
    tokenizer: Any,
    *,
    probe_builder: Callable[[Any, str], Any] | None = None,
) -> dict[str, Any]:
    """Validate one ERSR row and preserve all exact prefix/token invariants."""

    case_id = str(record["case_id"])
    prompt_ids = _int_ids(record["prompt_token_ids"], label=f"{case_id}/prompt")
    response_ids = _int_ids(record["response_token_ids"], label=f"{case_id}/response")
    token_start = int(record["step_token_start"])
    token_end = int(record["step_token_end"])
    if not 0 <= token_start < token_end <= len(response_ids):
        raise ValueError(
            f"Case {case_id!r} has invalid step token span "
            f"[{token_start}, {token_end}) for response length {len(response_ids)}."
        )

    reasoning_index = int(record["reasoning_step_index"])
    num_steps = int(record["num_reasoning_steps"])
    if not 0 <= reasoning_index < num_steps:
        raise ValueError(
            f"Case {case_id!r} has invalid reasoning index "
            f"{reasoning_index}/{num_steps}."
        )
    bucket = int(record["bucket"])
    reward = int(record["rollout_correct"])
    if not 0 <= bucket <= GROUP_SIZE_G or reward not in (0, 1):
        raise ValueError(f"Case {case_id!r} has invalid bucket/reward {bucket}/{reward}.")

    if probe_builder is None:
        from utils.oa_opd import build_answer_probe

        probe_builder = build_answer_probe
    probe = probe_builder(tokenizer, str(record["answer"]))
    probe_ids = _int_ids(probe.token_ids, label=f"{case_id}/probe")
    answer_positions = [int(value) for value in probe.answer_token_positions]
    if not answer_positions or min(answer_positions) <= 0:
        raise ValueError(f"Case {case_id!r} has an invalid answer-probe position set.")

    samples = record.get("samples")
    if not isinstance(samples, dict):
        raise ValueError(f"Case {case_id!r} has no samples object.")
    redecide = _required_sample(samples, case_id, "redecide")
    keep = _required_sample(samples, case_id, "keep")
    mc_k = int(redecide["mc_k"])
    if mc_k <= 0:
        raise ValueError(f"Case {case_id!r} has non-positive mc_k={mc_k}.")
    v_before = _sample_value(redecide, case_id=case_id, arm="redecide", mc_k=mc_k)
    v_student = _sample_value(keep, case_id=case_id, arm="keep", mc_k=mc_k)

    replacement_metadata = record.get("teacher_replacements")
    if not isinstance(replacement_metadata, dict) or not replacement_metadata:
        raise ValueError(f"Case {case_id!r} has no Teacher replacements.")
    teacher_names = sorted(
        key[len("replace:") :] for key in samples if str(key).startswith("replace:")
    )
    if set(teacher_names) != set(str(key) for key in replacement_metadata):
        raise ValueError(f"Case {case_id!r} has inconsistent Teacher sets.")

    teachers: dict[str, dict[str, Any]] = {}
    for name in teacher_names:
        metadata = replacement_metadata[name]
        text = str(metadata["text"])
        replacement_ids = [
            int(value) for value in tokenizer.encode(text, add_special_tokens=False)
        ]
        decoded = tokenizer.decode(
            replacement_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded != text:
            raise ValueError(
                f"Student tokenizer cannot round-trip Teacher replacement "
                f"{name!r} for case {case_id!r}."
            )
        arm = f"replace:{name}"
        v_teacher = _sample_value(
            _required_sample(samples, case_id, arm),
            case_id=case_id,
            arm=arm,
            mc_k=mc_k,
        )
        teachers[name] = {
            "replacement_token_ids": replacement_ids,
            "replacement_token_ids_sha256": token_ids_sha256(replacement_ids),
            "replacement_num_tokens": len(replacement_ids),
            "teacher_truncated": int(metadata.get("truncated", 0)),
            "V_teacher_after": v_teacher,
            "A_teacher": v_teacher - v_before,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "case_index": int(record["case_index"]),
        "prompt_id": int(record["response_idx"]),
        "trajectory_id": f"{int(record['response_idx'])}:{int(record['rollout_index'])}",
        "response_idx": int(record["response_idx"]),
        "rollout_index": int(record["rollout_index"]),
        "reward": reward,
        "bucket": bucket,
        "prompt_accuracy": bucket / GROUP_SIZE_G,
        "step_id": reasoning_index,
        "step_ordinal": reasoning_index + 1,
        "source_step_index": int(record["step_index"]),
        "num_steps": num_steps,
        "relative_position": (reasoning_index + 1) / num_steps,
        "mc_k": mc_k,
        "prompt_token_ids": prompt_ids,
        "pre_step_response_token_ids": response_ids[:token_start],
        "student_step_token_ids": response_ids[token_start:token_end],
        "probe_text": str(probe.text),
        "probe_token_ids": probe_ids,
        "probe_answer_token_positions": answer_positions,
        "V_before": v_before,
        "V_student_after": v_student,
        "A_student": v_student - v_before,
        "teachers": teachers,
    }


def max_probe_input_tokens(task: dict[str, Any]) -> int:
    base = (
        len(task["prompt_token_ids"])
        + len(task["pre_step_response_token_ids"])
        + len(task["probe_token_ids"])
    )
    additions = [0, len(task["student_step_token_ids"])] + [
        len(value["replacement_token_ids"]) for value in task["teachers"].values()
    ]
    return base + max(additions)


def discover_local_gpus(specification: str) -> list[int]:
    if specification.strip().lower() != "auto":
        values = [int(value.strip()) for value in specification.split(",") if value.strip()]
        if not values or len(values) != len(set(values)) or any(value < 0 for value in values):
            raise ValueError("--gpus must be auto or a unique comma-separated GPU index list.")
        return values
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [int(line.strip()) for line in completed.stdout.splitlines() if line.strip()]
    if not values:
        raise RuntimeError("nvidia-smi reported no local GPUs.")
    if len(values) != len(set(values)):
        raise RuntimeError(f"nvidia-smi reported duplicate GPU indices: {values}.")
    return values


def gpu_free_mib(gpu_id: int) -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(completed.stdout.strip().splitlines()[0])


def assert_gpu_headroom(gpu_id: int, minimum_mib: int, *, stage: str) -> int:
    free = gpu_free_mib(gpu_id)
    if free < minimum_mib:
        raise RuntimeError(
            f"GPU {gpu_id} has {free} MiB free at {stage}; required >= {minimum_mib} MiB."
        )
    return free


def prepare_task_shards(
    *,
    results_path: Path,
    model_path: Path,
    output_dir: Path,
    gpu_ids: Sequence[int],
    max_cases: int = 0,
    tokenizer: Any | None = None,
    probe_builder: Callable[[Any, str], Any] | None = None,
) -> dict[str, Any]:
    results_path = results_path.resolve()
    model_path = model_path.resolve()
    output_dir = output_dir.resolve()
    if not results_path.is_file():
        raise FileNotFoundError(results_path)
    if tokenizer is None and not model_path.is_dir():
        raise FileNotFoundError(model_path)
    results = load_json(results_path)
    if not isinstance(results, list) or not results:
        raise ValueError(f"Expected a non-empty JSON array in {results_path}.")
    results = sorted(results, key=lambda value: int(value["case_index"]))
    if max_cases > 0:
        results = results[:max_cases]
    if len(results) < len(gpu_ids):
        raise ValueError(
            f"Selected {len(results)} cases for {len(gpu_ids)} GPUs; every local GPU "
            "must receive at least one case. Increase --max-cases or use the full run."
        )
    if tokenizer is None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_path), trust_remote_code=True, use_fast=True, local_files_only=True
        )

    tasks = [
        prepare_task(record, tokenizer, probe_builder=probe_builder) for record in results
    ]
    del results
    case_ids = [str(task["case_id"]) for task in tasks]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("ERSR results contain duplicate case IDs.")
    teacher_sets = {tuple(task["teachers"]) for task in tasks}
    if len(teacher_sets) != 1:
        raise ValueError(f"Cases have inconsistent Teacher sets: {teacher_sets}.")
    teacher_names = list(next(iter(teacher_sets)))

    task_dir = output_dir / "03_prepared_task_shards"
    task_paths: list[Path] = []
    task_hashes: dict[str, str] = {}
    counts: dict[str, int] = {}
    for replica_index, gpu_id in enumerate(gpu_ids):
        shard = tasks[replica_index:: len(gpu_ids)]
        path = task_dir / f"03_replica_{replica_index:02d}_gpu_{gpu_id}.jsonl"
        atomic_jsonl(path, shard)
        task_paths.append(path)
        task_hashes[path.name] = sha256_file(path)
        counts[path.name] = len(shard)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "results_path": str(results_path),
        "results_sha256": sha256_file(results_path),
        "student_model": str(model_path),
        "gpu_policy": "one independent Student replica on every nvidia-smi local GPU; no WORLD_SIZE",
        "gpu_ids": list(gpu_ids),
        "group_size_G": GROUP_SIZE_G,
        "selected_cases": len(tasks),
        "max_cases_override": int(max_cases),
        "teachers": teacher_names,
        "probes_per_case": 2 + len(teacher_names),
        "total_probes": len(tasks) * (2 + len(teacher_names)),
        "max_probe_input_tokens": max(max_probe_input_tokens(task) for task in tasks),
        "task_shards": [str(path) for path in task_paths],
        "task_shard_sha256": task_hashes,
        "task_shard_case_counts": counts,
        "token_policy": (
            "prompt and Student response prefixes are exact ERSR token-ID slices; "
            "Teacher replacement text is encoded once by the Student tokenizer exactly "
            "as in run_ersr_mc.py"
        ),
        "probe_definition": (
            "utils.oa_opd.build_answer_probe; P=exp(mean logprob over gold-answer-overlap tokens)"
        ),
    }
    atomic_json(output_dir / "03_probe_collection_manifest.json", manifest)
    return manifest


def _score_one(output: Any, expected_ids: list[int], answer_positions: list[int]) -> dict[str, Any]:
    returned_ids = [int(value) for value in output.prompt_token_ids]
    if returned_ids != expected_ids:
        raise RuntimeError("vLLM changed the exact answer-probe input token IDs.")
    prompt_logprobs = output.prompt_logprobs
    if prompt_logprobs is None or len(prompt_logprobs) != len(returned_ids):
        raise RuntimeError("vLLM did not return complete prompt logprobs.")
    token_logprobs: list[float] = []
    token_ranks: list[int | None] = []
    for position in answer_positions:
        candidates = prompt_logprobs[position]
        token_id = returned_ids[position]
        if candidates is None or token_id not in candidates:
            raise RuntimeError(
                f"Chosen answer token {token_id} is absent at prompt position {position}."
            )
        item = candidates[token_id]
        token_logprobs.append(float(item.logprob))
        token_ranks.append(int(item.rank) if item.rank is not None else None)
    if not token_logprobs or not all(math.isfinite(value) for value in token_logprobs):
        raise RuntimeError("Answer probe produced empty or non-finite token logprobs.")
    sum_logprob = math.fsum(token_logprobs)
    mean_logprob = sum_logprob / len(token_logprobs)
    probability = math.exp(mean_logprob)
    return {
        "answer_token_logprobs": token_logprobs,
        "answer_token_ranks": token_ranks,
        "sum_logprob": sum_logprob,
        "mean_logprob": mean_logprob,
        "geometric_mean_probability": probability,
    }


def _build_probe_inputs(task: dict[str, Any]) -> list[dict[str, Any]]:
    prompt = list(task["prompt_token_ids"])
    pre = list(task["pre_step_response_token_ids"])
    probe = list(task["probe_token_ids"])
    answer_local = [int(value) for value in task["probe_answer_token_positions"]]

    def entry(kind: str, suffix: list[int], teacher: str | None = None) -> dict[str, Any]:
        prefix = prompt + pre + suffix
        input_ids = prefix + probe
        return {
            "case_id": task["case_id"],
            "kind": kind,
            "teacher": teacher,
            "input_ids": input_ids,
            "answer_positions": [len(prefix) + position for position in answer_local],
        }

    rows = [
        entry("before", []),
        entry("student_after", list(task["student_step_token_ids"])),
    ]
    rows.extend(
        entry(
            "teacher_after",
            list(value["replacement_token_ids"]),
            teacher=name,
        )
        for name, value in task["teachers"].items()
    )
    return rows


def _valid_score_shard(
    path: Path,
    *,
    task_shard_sha256: str,
    expected_case_ids: Sequence[str],
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = load_json(path)
        records = payload["records"]
        return (
            int(payload.get("schema_version", -1)) == SCHEMA_VERSION
            and str(payload.get("task_shard_sha256")) == task_shard_sha256
            and [str(row["case_id"]) for row in records] == list(expected_case_ids)
            and all("before" in row and "student_after" in row for row in records)
        )
    except Exception:
        return False


def _shutdown_llm(llm: Any) -> None:
    with contextlib.suppress(Exception):
        engine = getattr(llm, "llm_engine", None)
        shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()
    del llm
    gc.collect()
    with contextlib.suppress(Exception):
        import torch

        torch.cuda.empty_cache()


def score_worker(
    *,
    gpu_id: int,
    replica_index: int,
    task_path: str,
    task_sha256: str,
    model_path: str,
    output_dir: str,
    max_model_len: int,
    max_num_batched_tokens: int,
    batch_size: int,
    gpu_memory_utilization: float,
    minimum_free_mib: int,
    result_queue: Any,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    llm: Any | None = None
    try:
        from vllm import LLM, SamplingParams

        tasks = load_jsonl(Path(task_path))
        probes_per_case = 2 + len(tasks[0]["teachers"])
        cases_per_batch = max(1, batch_size // probes_per_case)
        assert_gpu_headroom(gpu_id, minimum_free_mib, stage="before model load")
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=max_model_len,
            max_num_seqs=max(batch_size, probes_per_case),
            max_num_batched_tokens=max_num_batched_tokens,
            gpu_memory_utilization=gpu_memory_utilization,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            enforce_eager=True,
            seed=42,
        )
        free_after_load = assert_gpu_headroom(
            gpu_id, minimum_free_mib, stage="after model load"
        )
        result_queue.put(
            {
                "type": "ready",
                "gpu": gpu_id,
                "replica": replica_index,
                "cases": len(tasks),
                "free_mib": free_after_load,
            }
        )
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=0,
            detokenize=False,
        )
        score_root = Path(output_dir) / "03_score_shards"
        completed_cases = 0
        for batch_index, start in enumerate(range(0, len(tasks), cases_per_batch)):
            batch_cases = tasks[start : start + cases_per_batch]
            score_path = score_root / (
                f"03_replica_{replica_index:02d}_gpu_{gpu_id}_batch_{batch_index:06d}.json"
            )
            expected_case_ids = [str(task["case_id"]) for task in batch_cases]
            if _valid_score_shard(
                score_path,
                task_shard_sha256=task_sha256,
                expected_case_ids=expected_case_ids,
            ):
                completed_cases += len(batch_cases)
                result_queue.put(
                    {
                        "type": "progress",
                        "gpu": gpu_id,
                        "replica": replica_index,
                        "completed_cases": completed_cases,
                        "total_cases": len(tasks),
                        "resumed": True,
                    }
                )
                continue

            probe_tasks = [
                probe_task
                for case in batch_cases
                for probe_task in _build_probe_inputs(case)
            ]
            assert_gpu_headroom(gpu_id, minimum_free_mib, stage=f"before batch {batch_index}")
            outputs = llm.generate(
                [{"prompt_token_ids": row["input_ids"]} for row in probe_tasks],
                sampling,
                use_tqdm=False,
            )
            if len(outputs) != len(probe_tasks):
                raise RuntimeError(
                    f"vLLM returned {len(outputs)} outputs for {len(probe_tasks)} probes."
                )
            by_case: dict[str, dict[str, Any]] = {
                str(task["case_id"]): {
                    "case_id": str(task["case_id"]),
                    "teacher_after": {},
                }
                for task in batch_cases
            }
            for probe_task, output in zip(probe_tasks, outputs, strict=True):
                score = _score_one(
                    output,
                    list(probe_task["input_ids"]),
                    list(probe_task["answer_positions"]),
                )
                destination = by_case[str(probe_task["case_id"])]
                if probe_task["kind"] == "teacher_after":
                    destination["teacher_after"][str(probe_task["teacher"])] = score
                else:
                    destination[str(probe_task["kind"])] = score
            for task in batch_cases:
                row = by_case[str(task["case_id"])]
                if set(row["teacher_after"]) != set(task["teachers"]):
                    raise RuntimeError(f"Incomplete Teacher probe scores for {task['case_id']}.")
            atomic_json(
                score_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "gpu": gpu_id,
                    "replica": replica_index,
                    "batch_index": batch_index,
                    "task_shard": task_path,
                    "task_shard_sha256": task_sha256,
                    "records": [by_case[case_id] for case_id in expected_case_ids],
                },
            )
            # The completed batch is already durable before the safety guard
            # can stop a low-headroom run, so a safer retry can resume it.
            free = assert_gpu_headroom(
                gpu_id, minimum_free_mib, stage=f"after batch {batch_index}"
            )
            completed_cases += len(batch_cases)
            result_queue.put(
                {
                    "type": "progress",
                    "gpu": gpu_id,
                    "replica": replica_index,
                    "completed_cases": completed_cases,
                    "total_cases": len(tasks),
                    "resumed": False,
                    "free_mib": free,
                }
            )
        result_queue.put(
            {
                "type": "done",
                "gpu": gpu_id,
                "replica": replica_index,
                "completed_cases": completed_cases,
            }
        )
    except BaseException as exc:
        result_queue.put(
            {
                "type": "error",
                "gpu": gpu_id,
                "replica": replica_index,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if llm is not None:
            _shutdown_llm(llm)


def _join_workers(workers: Sequence[mp.Process], result_queue: Any) -> list[dict[str, Any]]:
    done: set[int] = set()
    messages: list[dict[str, Any]] = []
    while len(done) < len(workers):
        try:
            message = result_queue.get(timeout=2.0)
        except queue.Empty:
            dead_without_done = [
                index
                for index, worker in enumerate(workers)
                if not worker.is_alive() and index not in done
            ]
            if dead_without_done:
                raise RuntimeError(
                    f"Probe workers exited without completion messages: {dead_without_done}."
                )
            continue
        messages.append(message)
        print(json.dumps(message, ensure_ascii=False), flush=True)
        if message["type"] == "error":
            for worker in workers:
                if worker.is_alive():
                    worker.terminate()
            raise RuntimeError(
                f"Probe worker GPU {message['gpu']} failed: {message['error']}\n"
                f"{message['traceback']}"
            )
        if message["type"] == "done":
            done.add(int(message["replica"]))
    for worker in workers:
        worker.join()
    abnormal = [worker.exitcode for worker in workers if worker.exitcode != 0]
    if abnormal:
        raise RuntimeError(f"Probe workers had abnormal exit codes: {abnormal}.")
    return messages


def _merge_scores(manifest: dict[str, Any], output_dir: Path, cases_per_batch: int) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    gpu_ids = [int(value) for value in manifest["gpu_ids"]]
    task_hashes = manifest["task_shard_sha256"]
    for replica_index, (gpu_id, task_path_raw) in enumerate(
        zip(gpu_ids, manifest["task_shards"], strict=True)
    ):
        task_path = Path(task_path_raw)
        tasks = load_jsonl(task_path)
        scores: dict[str, dict[str, Any]] = {}
        for batch_index, start in enumerate(range(0, len(tasks), cases_per_batch)):
            batch = tasks[start : start + cases_per_batch]
            path = output_dir / "03_score_shards" / (
                f"03_replica_{replica_index:02d}_gpu_{gpu_id}_batch_{batch_index:06d}.json"
            )
            expected = [str(task["case_id"]) for task in batch]
            if not _valid_score_shard(
                path,
                task_shard_sha256=task_hashes[task_path.name],
                expected_case_ids=expected,
            ):
                raise RuntimeError(f"Missing or invalid completed score shard: {path}.")
            for row in load_json(path)["records"]:
                case_id = str(row["case_id"])
                if case_id in scores:
                    raise RuntimeError(f"Duplicate score for {case_id!r}.")
                scores[case_id] = row

        for task in tasks:
            case_id = str(task["case_id"])
            score = scores.get(case_id)
            if score is None:
                raise RuntimeError(f"Missing probe score for {case_id!r}.")
            if case_id in seen:
                raise RuntimeError(f"Duplicate merged task {case_id!r}.")
            seen.add(case_id)
            before = score["before"]
            student = score["student_after"]
            p_before = float(before["geometric_mean_probability"])
            p_student = float(student["geometric_mean_probability"])
            teachers: dict[str, Any] = {}
            for name, metadata in task["teachers"].items():
                teacher_score = score["teacher_after"][name]
                p_teacher = float(teacher_score["geometric_mean_probability"])
                compact_metadata = {
                    key: value
                    for key, value in metadata.items()
                    if key != "replacement_token_ids"
                }
                teachers[name] = {
                    **compact_metadata,
                    "probe": teacher_score,
                    "P_teacher_after": p_teacher,
                    "deltaP_teacher": p_teacher - p_before,
                    "delta_logP_teacher": float(teacher_score["mean_logprob"])
                    - float(before["mean_logprob"]),
                }
            merged.append(
                {
                    **{
                        key: value
                        for key, value in task.items()
                        if key
                        not in {
                            "prompt_token_ids",
                            "pre_step_response_token_ids",
                            "student_step_token_ids",
                            "probe_token_ids",
                            "probe_answer_token_positions",
                            "teachers",
                        }
                    },
                    "token_audit": {
                        "prompt_num_tokens": len(task["prompt_token_ids"]),
                        "pre_step_response_num_tokens": len(
                            task["pre_step_response_token_ids"]
                        ),
                        "student_step_num_tokens": len(task["student_step_token_ids"]),
                        "probe_num_tokens": len(task["probe_token_ids"]),
                        "probe_answer_token_positions": task[
                            "probe_answer_token_positions"
                        ],
                        "prompt_token_ids_sha256": token_ids_sha256(
                            task["prompt_token_ids"]
                        ),
                        "pre_step_response_token_ids_sha256": token_ids_sha256(
                            task["pre_step_response_token_ids"]
                        ),
                        "student_step_token_ids_sha256": token_ids_sha256(
                            task["student_step_token_ids"]
                        ),
                        "probe_token_ids_sha256": token_ids_sha256(
                            task["probe_token_ids"]
                        ),
                    },
                    "probe_before": before,
                    "probe_student_after": student,
                    "P_before": p_before,
                    "P_student_after": p_student,
                    "deltaP_student": p_student - p_before,
                    "delta_logP_student": float(student["mean_logprob"])
                    - float(before["mean_logprob"]),
                    "teachers": teachers,
                }
            )
    merged.sort(key=lambda value: int(value["case_index"]))
    if len(merged) != int(manifest["selected_cases"]):
        raise RuntimeError(
            f"Merged {len(merged)} cases, expected {manifest['selected_cases']}."
        )
    return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--student-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--gpus",
        default="auto",
        help="auto uses every GPU reported by nvidia-smi; WORLD_SIZE is ignored.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="probe sequences per vLLM call")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.78)
    parser.add_argument("--min-free-gpu-gib", type=float, default=5.0)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="smoke-only limit; zero processes every ERSR case.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise ValueError("--gpu-memory-utilization must lie in (0, 1).")
    if args.min_free_gpu_gib < 0.0:
        raise ValueError("--min-free-gpu-gib cannot be negative.")
    if args.max_num_batched_tokens <= 0 or args.max_cases < 0:
        raise ValueError("Token and case limits cannot be negative/zero.")

    started = time.monotonic()
    gpu_ids = discover_local_gpus(str(args.gpus))
    output_dir = args.output_dir.resolve()
    minimum_free_mib = math.ceil(float(args.min_free_gpu_gib) * 1024)
    preflight_memory = {
        str(gpu_id): assert_gpu_headroom(
            gpu_id, minimum_free_mib, stage="collector preflight"
        )
        for gpu_id in gpu_ids
    }
    print(
        json.dumps(
            {
                "event": "03_probe_gpu_plan",
                "gpu_source": "nvidia-smi local enumeration",
                "gpu_ids": gpu_ids,
                "free_memory_mib": preflight_memory,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    manifest = prepare_task_shards(
        results_path=args.results,
        model_path=args.student_model,
        output_dir=output_dir,
        gpu_ids=gpu_ids,
        max_cases=int(args.max_cases),
    )
    print(
        json.dumps(
            {
                "event": "03_probe_prepared",
                "cases": manifest["selected_cases"],
                "probes": manifest["total_probes"],
                "teachers": manifest["teachers"],
                "max_probe_input_tokens": manifest["max_probe_input_tokens"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.prepare_only:
        return

    max_model_len = max(
        8192,
        ((int(manifest["max_probe_input_tokens"]) + 1 + 255) // 256) * 256,
    )
    max_num_batched_tokens = max(int(args.max_num_batched_tokens), max_model_len)
    probes_per_case = int(manifest["probes_per_case"])
    cases_per_batch = max(1, int(args.batch_size) // probes_per_case)
    scoring_config = {
        "schema_version": SCHEMA_VERSION,
        "created_at": utc_now(),
        "student_model": str(args.student_model.resolve()),
        "gpu_ids": gpu_ids,
        "replica_policy": "one full tensor_parallel_size=1 Student model per local GPU",
        "batch_size_probe_sequences": int(args.batch_size),
        "cases_per_batch": cases_per_batch,
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "minimum_free_gpu_mib": minimum_free_mib,
        "max_model_len": max_model_len,
        "max_num_batched_tokens": max_num_batched_tokens,
        "dtype": "bfloat16",
        "prompt_logprobs": 0,
        "max_generated_tokens": 1,
    }
    atomic_json(output_dir / "03_probe_scoring_config.json", scoring_config)
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    workers: list[mp.Process] = []
    for replica_index, (gpu_id, task_path_raw) in enumerate(
        zip(gpu_ids, manifest["task_shards"], strict=True)
    ):
        task_path = Path(task_path_raw)
        worker = context.Process(
            target=score_worker,
            kwargs={
                "gpu_id": gpu_id,
                "replica_index": replica_index,
                "task_path": str(task_path),
                "task_sha256": manifest["task_shard_sha256"][task_path.name],
                "model_path": str(args.student_model.resolve()),
                "output_dir": str(output_dir),
                "max_model_len": max_model_len,
                "max_num_batched_tokens": max_num_batched_tokens,
                "batch_size": int(args.batch_size),
                "gpu_memory_utilization": float(args.gpu_memory_utilization),
                "minimum_free_mib": minimum_free_mib,
                "result_queue": result_queue,
            },
        )
        worker.start()
        workers.append(worker)
    messages = _join_workers(workers, result_queue)
    merged = _merge_scores(manifest, output_dir, cases_per_batch)
    step_scores_path = output_dir / "03_probe_step_scores.jsonl"
    atomic_jsonl(step_scores_path, merged)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "completed_at": utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "results_path": str(args.results.resolve()),
        "student_model": str(args.student_model.resolve()),
        "gpu_ids": gpu_ids,
        "gpu_count": len(gpu_ids),
        "gpu_selection": "all indices reported by nvidia-smi" if args.gpus == "auto" else "explicit",
        "world_size_used": False,
        "cases": len(merged),
        "teacher_names": manifest["teachers"],
        "step_teacher_pairs": sum(len(row["teachers"]) for row in merged),
        "reward_counts": dict(Counter(str(row["reward"]) for row in merged)),
        "step_scores": str(step_scores_path),
        "step_scores_sha256": sha256_file(step_scores_path),
        "worker_messages": [
            message for message in messages if message["type"] in {"ready", "done"}
        ],
    }
    atomic_json(output_dir / "03_probe_collection_summary.json", summary)
    print(json.dumps({"event": "03_probe_collection_complete", **summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
