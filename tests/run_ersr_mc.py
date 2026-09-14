#!/usr/bin/env python3
"""Run the ERSR Monte-Carlo experiment on the saved DAPO rollouts.

This is intentionally a tests-only offline evaluator.  It keeps the formal
8-GPU training entry points untouched.  The evaluator has two phases:

1. every configured Teacher samples one replacement semantic step for every
   selected non-answer Student step;
2. a Student replica samples the re-decide, keep, and one replace arm for each
   case and estimates the corresponding final-reward values.

Each worker process owns one physical ``nvidia-smi`` visible GPU and one vLLM
model replica.  A vLLM call always contains exactly 256 completions (padding
only the final call and discarding padded outputs).  The parent process writes
atomic shards, a progress file, console heartbeats, and WandB metrics.
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
import random
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.oa_opd import find_final_answer_step, map_response_steps_to_original_tokens
from utils.prompts import render_prompt
from utils.split_answer_probe_steps import split_semantic_hybrid


SCHEMA_VERSION = 1
BUCKETS = tuple(range(6))
ROLL_OUTS_PER_GPU_CALL = 256
MC_ARMS = ("redecide", "keep")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, value: Any, *, indent: int = 4) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=indent)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def deep_merge(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    result = dict(left)
    for key, value in right.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path) -> dict[str, Any]:
    import yaml

    path = path.resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"Configuration must be a mapping: {path}")
    base_name = raw.pop("base_config", None)
    if base_name:
        base_path = Path(str(base_name))
        if not base_path.is_absolute():
            base_path = (path.parent / base_path).resolve()
        return deep_merge(load_config(base_path), raw)
    return raw


def resolve_path(value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _int(value: object, name: str) -> int:
    try:
        result = int(value)
    except Exception as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from exc
    return result


def _float(value: object, name: str) -> float:
    try:
        result = float(value)
    except Exception as exc:
        raise ValueError(f"{name} must be numeric, got {value!r}.") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return result


def detect_nvidia_smi_gpus() -> list[int]:
    """Read physical GPU indices from nvidia-smi, never from world_size."""

    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    ids: list[int] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            ids.append(int(line.split(",", 1)[0].strip()))
    if not ids:
        raise RuntimeError("nvidia-smi reported no GPUs.")

    # Respect a numeric CUDA_VISIBLE_DEVICES restriction when the caller set
    # one, while still using nvidia-smi as the source of physical devices.
    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_env and visible_env.lower() not in {"all", "none"}:
        requested: list[int] = []
        for token in visible_env.split(","):
            token = token.strip()
            if token.isdigit():
                requested.append(int(token))
        if requested:
            ids = [gpu for gpu in ids if gpu in set(requested)]
    if not ids:
        raise RuntimeError("CUDA_VISIBLE_DEVICES hides every nvidia-smi GPU.")
    return ids


def configured_gpus(config: dict[str, Any]) -> list[int]:
    hardware = config.get("hardware", {}) or {}
    requested = hardware.get("gpu_ids", "auto")
    visible = detect_nvidia_smi_gpus()
    if requested != "auto":
        if not isinstance(requested, list) or not requested:
            raise ValueError("hardware.gpu_ids must be 'auto' or a non-empty list.")
        selected = [_int(value, "hardware.gpu_ids") for value in requested]
        missing = [gpu for gpu in selected if gpu not in visible]
        if missing:
            raise RuntimeError(
                f"Configured GPU IDs {missing} are not nvidia-smi visible; visible={visible}."
            )
        visible = selected
    expected = _int(hardware.get("expected_gpu_count", len(visible)), "hardware.expected_gpu_count")
    if expected <= 0:
        raise ValueError("hardware.expected_gpu_count must be positive.")
    if bool(hardware.get("require_expected_gpu_count", True)) and len(visible) != expected:
        raise RuntimeError(
            f"Expected {expected} nvidia-smi visible GPUs, found {len(visible)}: {visible}."
        )
    return visible


def query_gpu_memory_mib(gpu_id: int) -> dict[str, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    fields = result.stdout.strip().splitlines()[0].split(",")
    if len(fields) != 3:
        raise RuntimeError(f"Unexpected nvidia-smi memory output: {result.stdout!r}")
    total, used, free = (int(float(field.strip())) for field in fields)
    return {"total": total, "used": used, "free": free}


def assert_gpu_headroom(gpu_id: int, required_free_mib: int, *, stage: str) -> dict[str, int]:
    memory = query_gpu_memory_mib(gpu_id)
    if memory["free"] < required_free_mib:
        raise RuntimeError(
            f"GPU reserve violated during {stage}: free={memory['free']} MiB, "
            f"required={required_free_mib} MiB."
        )
    return memory


class GPUHeartbeat:
    """Check physical free memory while a synchronous vLLM call is running."""

    def __init__(
        self,
        *,
        gpu_id: int,
        required_free_mib: int,
        check_interval: float,
        report_interval: float,
        stage: str,
    ) -> None:
        self.gpu_id = gpu_id
        self.required_free_mib = required_free_mib
        self.check_interval = check_interval
        self.report_interval = report_interval
        self.stage = stage
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.violation: dict[str, int] | None = None

    def __enter__(self) -> "GPUHeartbeat":
        started = time.monotonic()

        def monitor() -> None:
            last_report = started
            while not self.stop_event.wait(self.check_interval):
                try:
                    memory = query_gpu_memory_mib(self.gpu_id)
                except Exception as exc:
                    print(
                        json.dumps(
                            {
                                "event": "gpu_heartbeat_query_failed",
                                "stage": self.stage,
                                "error": repr(exc),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    continue
                now = time.monotonic()
                violated = memory["free"] < self.required_free_mib
                if violated or now - last_report >= self.report_interval:
                    print(
                        json.dumps(
                            {
                                "event": "gpu_heartbeat",
                                "stage": self.stage,
                                "elapsed_seconds": round(now - started, 1),
                                "gpu_memory_mib": memory,
                                "required_free_mib": self.required_free_mib,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    last_report = now
                if violated:
                    self.violation = memory
                    os.kill(os.getpid(), signal.SIGINT)
                    return

        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(2.0, 2.0 * self.check_interval))
        if exc_type is None and self.violation is not None:
            raise RuntimeError(
                f"GPU reserve was violated during {self.stage}: "
                f"free={self.violation['free']} MiB, required={self.required_free_mib} MiB."
            )


def shutdown_llm(llm: Any) -> None:
    try:
        engine = getattr(llm, "llm_engine", None)
        core = getattr(engine, "engine_core", None)
        shutdown = getattr(core, "shutdown", None)
        if callable(shutdown):
            shutdown()
        else:
            shutdown = getattr(engine, "shutdown", None)
            if callable(shutdown):
                shutdown()
    except Exception:
        pass
    del llm
    gc.collect()
    with contextlib.suppress(Exception):
        import torch

        torch.cuda.empty_cache()


def normalized_reason(value: Any) -> str | None:
    if value is None:
        return None
    return str(getattr(value, "value", value))


def load_rollout_metadata(batches_dir: Path) -> dict[int, dict[str, Any]]:
    metadata: dict[int, dict[str, Any]] = {}
    for path in sorted(batches_dir.glob("batch_*.json")):
        payload = load_json(path)
        for item in payload.get("accepted", []):
            source_index = _int(item["source_index"], "source_index")
            if source_index in metadata:
                raise ValueError(f"Duplicate accepted source_index {source_index}.")
            responses = item.get("responses")
            if not isinstance(responses, list) or len(responses) != 5:
                raise ValueError(f"Accepted question {source_index} does not have five responses.")
            metadata[source_index] = {
                "question": str(item["question"]),
                "answer": str(item["answer"]),
                "bucket": _int(item["bucket"], "bucket"),
                "responses": [str(value) for value in responses],
            }
    if not metadata:
        raise FileNotFoundError(f"No accepted batch metadata found under {batches_dir}.")
    return metadata


def _last_boxed_step(steps: list[Any], answer: str) -> int | None:
    boxed = [step.index for step in steps if r"\boxed" in step.text]
    if boxed:
        # ERSR's explicit policy is to exclude the final answer-containing
        # step while retaining any earlier intermediate boxed identities.
        return max(boxed)
    return find_final_answer_step(steps, answer)


def build_candidates(config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    input_config = config["input"]
    rollouts_path = resolve_path(input_config["rollouts_with_steps"])
    batches_dir = resolve_path(input_config["batches_dir"])
    tokenizer_path = resolve_path(input_config["tokenizer_path"])
    prompt_name = str(input_config.get("prompt_name", ""))
    if prompt_name != "qwen3_no_thinking_prompt":
        raise ValueError("ERSR requires input.prompt_name=qwen3_no_thinking_prompt.")
    if not rollouts_path.is_file():
        raise FileNotFoundError(rollouts_path)
    if not batches_dir.is_dir():
        raise FileNotFoundError(batches_dir)
    records = load_json(rollouts_path)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a non-empty JSON array in {rollouts_path}.")
    metadata = load_rollout_metadata(batches_dir)

    from transformers import AutoTokenizer
    from utils.math_verifier import verify_response_answer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        use_fast=True,
        local_files_only=True,
    )
    prompt_ids_by_source: dict[int, list[int]] = {}
    occurrence_by_source: Counter[int] = Counter()
    candidates: list[dict[str, Any]] = []
    skipped = Counter()

    for record_index, raw in enumerate(records):
        if not isinstance(raw, dict) or not isinstance(raw.get("response"), str):
            raise ValueError(f"Invalid rollout record {record_index}.")
        source_index = _int(raw.get("response_idx"), "response_idx")
        if source_index not in metadata:
            raise ValueError(f"response_idx={source_index} has no accepted metadata.")
        meta = metadata[source_index]
        occurrence = occurrence_by_source[source_index]
        occurrence_by_source[source_index] += 1
        if occurrence >= len(meta["responses"]):
            raise ValueError(f"More than five responses for source_index={source_index}.")
        response = str(raw["response"])
        if response != meta["responses"][occurrence]:
            raise ValueError(
                f"Rollout order/text mismatch for source_index={source_index}, occurrence={occurrence}."
            )
        if source_index not in prompt_ids_by_source:
            prompt = render_prompt(prompt_name, question=meta["question"])
            prompt_ids_by_source[source_index] = [
                int(value) for value in tokenizer.encode(prompt, add_special_tokens=False)
            ]
        response_ids = [
            int(value) for value in tokenizer.encode(response, add_special_tokens=False)
        ]
        starts_stored = [int(value) for value in raw.get("step_token_starts", [])]
        decoded = tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if decoded != response:
            raise RuntimeError(f"Response {record_index} failed tokenizer round-trip.")
        spans = split_semantic_hybrid(response)
        if len(spans) != len(starts_stored):
            # This should never happen for the checked-in segmentation file,
            # but retain the production mapper as a correctness fallback when
            # a tokenizer or splitter version changes.
            _, mapped = map_response_steps_to_original_tokens(tokenizer, response_ids)
            starts_mapped = [int(step.token_start) for step in mapped]
            if starts_stored != starts_mapped:
                raise ValueError(
                    f"Saved step starts disagree with current tokenizer for record {record_index}."
                )
            step_rows = [
                {
                    "index": int(step.index),
                    "text": step.text,
                    "char_start": int(step.char_start),
                    "char_end": int(step.char_end),
                    "token_start": int(step.token_start),
                    "token_end": int(step.token_end),
                }
                for step in mapped
            ]
        else:
            step_ends = starts_stored[1:] + [len(response_ids)]
            step_rows = [
                {
                    "index": index,
                    "text": span.text,
                    "char_start": int(span.start),
                    "char_end": int(span.end),
                    "token_start": starts_stored[index],
                    "token_end": step_ends[index],
                }
                for index, span in enumerate(spans)
            ]
        answer_step = _last_boxed_step(
            [type("Step", (), row)() for row in step_rows], meta["answer"]
        )
        if answer_step is None:
            skipped["missing_answer_step"] += 1
            continue
        reasoning = [row for row in step_rows if int(row["index"]) < answer_step]
        if not reasoning:
            skipped["no_reasoning_steps"] += 1
            continue
        rollout_correct = bool(verify_response_answer(response, meta["answer"]))
        for reasoning_index, step in enumerate(reasoning):
            step_end = int(step["token_end"])
            if step_end <= int(step["token_start"]):
                skipped["empty_step"] += 1
                continue
            fraction = (
                reasoning_index / (len(reasoning) - 1) if len(reasoning) > 1 else 0.0
            )
            candidates.append(
                {
                    "record_index": record_index,
                    "response_idx": source_index,
                    "rollout_index": occurrence,
                    "bucket": int(meta["bucket"]),
                    "rollout_correct": int(rollout_correct),
                    "question": meta["question"],
                    "answer": meta["answer"],
                    "response": response,
                    "prompt_token_ids": list(prompt_ids_by_source[source_index]),
                    "response_token_ids": response_ids,
                    "step_index": int(step["index"]),
                    "reasoning_step_index": reasoning_index,
                    "num_steps": len(step_rows),
                    "num_reasoning_steps": len(reasoning),
                    "step_char_start": int(step["char_start"]),
                    "step_char_end": int(step["char_end"]),
                    "step_token_start": int(step["token_start"]),
                    "step_token_end": step_end,
                    "step_text": step["text"],
                    "position_fraction": fraction,
                }
            )
    if any(count != 5 for count in occurrence_by_source.values()):
        raise ValueError(f"Some questions do not have exactly five rollout records: {occurrence_by_source}")
    return candidates, {
        "rollout_records": len(records),
        "questions": len(occurrence_by_source),
        "candidate_steps": len(candidates),
        "skipped": dict(skipped),
        "tokenizer_path": str(tokenizer_path),
        "prompt_name": prompt_name,
    }


def allocate_bucket_targets(total_steps: int, per_bucket: int) -> dict[int, int]:
    if per_bucket > 0:
        return {bucket: per_bucket for bucket in BUCKETS}
    if total_steps <= 0:
        raise ValueError("selection.total_steps must be positive when steps_per_bucket is zero.")
    base, remainder = divmod(total_steps, len(BUCKETS))
    return {bucket: base + int(bucket < remainder) for bucket in BUCKETS}


def take_position_balanced(
    rows: list[dict[str, Any]], quota: int, *, bins: int, seed: int
) -> list[dict[str, Any]]:
    if quota == 0:
        return []
    if len(rows) < quota:
        raise RuntimeError(f"Only {len(rows)} candidates are available for quota {quota}.")
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        position = min(max(float(row["position_fraction"]), 0.0), 1.0)
        bin_index = min(bins - 1, int(position * bins))
        buckets[bin_index].append(row)
    rng = random.Random(seed)
    for values in buckets.values():
        rng.shuffle(values)
    selected: list[dict[str, Any]] = []
    selected_by_bin: Counter[int] = Counter()
    while len(selected) < quota:
        available = [index for index, values in buckets.items() if values]
        if not available:
            raise RuntimeError("Position-balanced sampling ran out of candidates.")
        min_count = min(selected_by_bin[index] for index in available)
        tied = [index for index in available if selected_by_bin[index] == min_count]
        chosen_bin = tied[rng.randrange(len(tied))]
        selected.append(buckets[chosen_bin].pop())
        selected_by_bin[chosen_bin] += 1
    return selected


def select_steps(
    candidates: list[dict[str, Any]], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selection = config["selection"]
    per_bucket = _int(selection.get("steps_per_bucket", 0), "selection.steps_per_bucket")
    total_steps = _int(selection.get("total_steps", 0), "selection.total_steps")
    targets = allocate_bucket_targets(total_steps, per_bucket)
    position_bins = _int(selection.get("position_bins", 20), "selection.position_bins")
    seed = _int(selection.get("seed", 0), "selection.seed")
    if position_bins <= 0:
        raise ValueError("selection.position_bins must be positive.")

    selected: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"targets": {str(k): v for k, v in targets.items()}, "by_bucket": {}}
    for bucket in BUCKETS:
        target = targets[bucket]
        pool = [row for row in candidates if int(row["bucket"]) == bucket]
        correct_quota = int(round(target * bucket / 5.0))
        wrong_quota = target - correct_quota
        bucket_selected: list[dict[str, Any]] = []
        for correct, quota in ((1, correct_quota), (0, wrong_quota)):
            group = [row for row in pool if int(row["rollout_correct"]) == correct]
            picked = take_position_balanced(
                group,
                quota,
                bins=position_bins,
                seed=seed + bucket * 1009 + correct * 100_003,
            )
            bucket_selected.extend(picked)
        selected.extend(bucket_selected)
        summary["by_bucket"][str(bucket)] = {
            "target": target,
            "selected": len(bucket_selected),
            "selected_correct": sum(int(row["rollout_correct"]) for row in bucket_selected),
            "selected_wrong": sum(1 - int(row["rollout_correct"]) for row in bucket_selected),
            "position_bins": dict(
                Counter(
                    min(position_bins - 1, int(float(row["position_fraction"]) * position_bins))
                    for row in bucket_selected
                )
            ),
            "available": len(pool),
            "available_correct": sum(int(row["rollout_correct"]) for row in pool),
            "available_wrong": sum(1 - int(row["rollout_correct"]) for row in pool),
        }

    order_rng = random.Random(seed + 7_919_357)
    order_rng.shuffle(selected)
    for case_index, row in enumerate(selected):
        row["case_index"] = case_index
        row["case_id"] = (
            f"{row['response_idx']}:{row['rollout_index']}:{row['step_index']}"
        )
    if len({str(row["case_id"]) for row in selected}) != len(selected):
        raise RuntimeError("Step selection produced duplicate case IDs.")
    summary["selected_steps"] = len(selected)
    summary["selected_rollout_accuracy"] = (
        sum(int(row["rollout_correct"]) for row in selected) / len(selected)
        if selected
        else 0.0
    )
    return selected, summary


def manifest_row(row: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "case_index",
        "case_id",
        "record_index",
        "response_idx",
        "rollout_index",
        "bucket",
        "rollout_correct",
        "step_index",
        "reasoning_step_index",
        "num_steps",
        "num_reasoning_steps",
        "step_char_start",
        "step_char_end",
        "step_token_start",
        "step_token_end",
        "position_fraction",
        # Keep the exact source context with the manifest.  Downstream ERSR
        # consumers can therefore splice original token IDs without rebuilding
        # indices by concatenating re-tokenized text.
        "question",
        "answer",
        "response",
        "step_text",
        "prompt_token_ids",
        "response_token_ids",
    )
    return {field: row[field] for field in fields}


def load_completed_ids(root: Path, *, key: str, require_ok: bool = True) -> set[str]:
    completed: set[str] = set()
    for path in sorted((root / "shards").glob("*.json")):
        with contextlib.suppress(Exception):
            payload = load_json(path)
            for row in payload.get("records", []):
                if (not require_ok or bool(row.get("ok", True))) and row.get(key) is not None:
                    completed.add(str(row[key]))
    return completed


def assignments(items: list[dict[str, Any]], gpu_ids: list[int]) -> list[list[dict[str, Any]]]:
    return [items[index:: len(gpu_ids)] for index in range(len(gpu_ids))]


def worker_model_kwargs(model_path: str, generation: dict[str, Any], max_model_len: int) -> dict[str, Any]:
    return {
        "model": model_path,
        "tensor_parallel_size": 1,
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "max_model_len": max_model_len,
        "max_num_seqs": _int(generation.get("max_num_seqs", 256), "generation.max_num_seqs"),
        "max_num_batched_tokens": _int(
            generation.get("max_num_batched_tokens", 32768),
            "generation.max_num_batched_tokens",
        ),
        "gpu_memory_utilization": _float(
            generation.get("gpu_memory_utilization", 0.78),
            "generation.gpu_memory_utilization",
        ),
        "enable_prefix_caching": bool(generation.get("enable_prefix_caching", True)),
        "enable_chunked_prefill": bool(generation.get("enable_chunked_prefill", True)),
        "enforce_eager": bool(generation.get("enforce_eager", True)),
        # Keep vLLM sleep mode explicitly disabled for every Student and
        # Teacher replica.  This is intentionally code-level behavior rather
        # than a per-run config option required by the server launcher.
        "enable_sleep_mode": False,
        "seed": _int(generation.get("seed", 0), "generation.seed"),
    }


def _worker_setup(gpu_id: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def _padding_specs(specs: list[dict[str, Any]], params: list[Any], slots: int) -> tuple[list[dict[str, Any]], list[Any]]:
    if not specs:
        raise ValueError("Cannot generate an empty batch.")
    if len(specs) > slots:
        raise ValueError(f"Batch has {len(specs)} prompts but only {slots} slots.")
    if len(specs) == slots:
        return specs, params
    padding = slots - len(specs)
    return specs + [specs[-1]] * padding, params + [params[-1]] * padding


def teacher_worker(
    *,
    gpu_id: int,
    rank: int,
    tasks: list[dict[str, Any]],
    teacher: dict[str, Any],
    generation: dict[str, Any],
    max_model_len: int,
    result_queue: Any,
) -> None:
    _worker_setup(gpu_id)
    llm: Any | None = None
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from utils.oa_opd import map_response_steps_to_original_tokens

        tokenizer = AutoTokenizer.from_pretrained(
            str(teacher["model_path"]),
            trust_remote_code=True,
            use_fast=True,
            local_files_only=True,
        )
        required_free = math.ceil(
            _float(generation.get("min_free_gpu_gib", 5.0), "generation.min_free_gpu_gib") * 1024
        )
        assert_gpu_headroom(gpu_id, required_free, stage="teacher-before-load")
        llm = LLM(**worker_model_kwargs(str(teacher["model_path"]), generation, max_model_len))
        memory = assert_gpu_headroom(gpu_id, required_free, stage="teacher-after-load")
        result_queue.put({"type": "ready", "role": "teacher", "rank": rank, "gpu": gpu_id, "memory": memory})
        slots = ROLL_OUTS_PER_GPU_CALL
        max_total = _int(generation["max_total_response_tokens"], "generation.max_total_response_tokens")
        max_step = _int(teacher["max_step_new_tokens"], "teacher.max_step_new_tokens")
        check_interval = _float(generation.get("gpu_check_interval_seconds", 5.0), "generation.gpu_check_interval_seconds")
        report_interval = _float(generation.get("gpu_report_interval_seconds", 60.0), "generation.gpu_report_interval_seconds")
        reject_truncated = bool(generation.get("reject_truncated_teacher_step", True))

        for batch_index, start in enumerate(range(0, len(tasks), slots)):
            batch = tasks[start : start + slots]
            prompts: list[dict[str, Any]] = []
            params: list[Any] = []
            for row in batch:
                pre = row["response_token_ids"][: int(row["step_token_start"])]
                input_ids = list(row["prompt_token_ids"]) + list(pre)
                budget = min(max_step, max_total - len(pre))
                if budget <= 0:
                    raise RuntimeError(f"No Teacher budget remains for {row['case_id']}.")
                prompts.append({"prompt_token_ids": input_ids})
                params.append(
                    SamplingParams(
                        n=1,
                        temperature=_float(teacher.get("temperature", 0.6), "teacher.temperature"),
                        top_p=_float(teacher.get("top_p", 1.0), "teacher.top_p"),
                        top_k=-1,
                        max_tokens=budget,
                        seed=_int(generation.get("seed", 0), "generation.seed") + int(row["case_index"]),
                    )
                )
            prompts, params = _padding_specs(prompts, params, slots)
            started = time.monotonic()
            with GPUHeartbeat(
                gpu_id=gpu_id,
                required_free_mib=required_free,
                check_interval=check_interval,
                report_interval=report_interval,
                stage=f"teacher/{teacher['name']}/batch_{batch_index}",
            ):
                outputs = llm.generate(prompts, sampling_params=params, use_tqdm=False)
            if len(outputs) != slots:
                raise RuntimeError(f"Teacher returned {len(outputs)} requests, expected {slots}.")
            records: list[dict[str, Any]] = []
            for row, output in zip(batch, outputs[: len(batch)], strict=True):
                if list(output.prompt_token_ids) != list(prompts[batch.index(row)]["prompt_token_ids"]):
                    raise RuntimeError(f"Teacher changed input IDs for {row['case_id']}.")
                if len(output.outputs) != 1:
                    raise RuntimeError(f"Teacher did not return one sample for {row['case_id']}.")
                completion = output.outputs[0]
                proposal_ids = [int(value) for value in completion.token_ids]
                if not proposal_ids:
                    raise RuntimeError(f"Teacher returned an empty proposal for {row['case_id']}.")
                visible, mapped = map_response_steps_to_original_tokens(tokenizer, proposal_ids)
                if not mapped:
                    raise RuntimeError(f"Teacher proposal has no semantic step for {row['case_id']}.")
                first = mapped[0]
                finish_reason = normalized_reason(completion.finish_reason)
                maybe_truncated = finish_reason == "length" and int(first.token_end) == len(proposal_ids)
                if reject_truncated and maybe_truncated:
                    raise RuntimeError(f"Teacher first step was truncated for {row['case_id']}.")
                records.append(
                    {
                        "ok": True,
                        "case_id": str(row["case_id"]),
                        "case_index": int(row["case_index"]),
                        "teacher_name": str(teacher["name"]),
                        "replacement_text": str(first.text),
                        "proposal_text": str(visible),
                        "proposal_num_tokens": len(proposal_ids),
                        "replacement_num_tokens": int(first.token_end),
                        "finish_reason": finish_reason,
                        "truncated": int(maybe_truncated),
                    }
                )
            memory = assert_gpu_headroom(gpu_id, required_free, stage="teacher-after-batch")
            result_queue.put(
                {
                    "type": "batch",
                    "role": "teacher",
                    "rank": rank,
                    "gpu": gpu_id,
                    "batch_index": batch_index,
                    "records": records,
                    "completed_items": len(records),
                    "generated_rollouts": len(records),
                    "elapsed_seconds": time.monotonic() - started,
                    "memory": memory,
                }
            )
        result_queue.put({"type": "done", "role": "teacher", "rank": rank, "gpu": gpu_id})
    except BaseException as exc:
        result_queue.put(
            {
                "type": "error",
                "role": "teacher",
                "rank": rank,
                "gpu": gpu_id,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if llm is not None:
            shutdown_llm(llm)


def student_worker(
    *,
    gpu_id: int,
    rank: int,
    tasks: list[dict[str, Any]],
    teacher_replacements: dict[str, dict[str, str]],
    teachers: list[dict[str, Any]],
    generation: dict[str, Any],
    max_model_len: int,
    result_queue: Any,
) -> None:
    _worker_setup(gpu_id)
    llm: Any | None = None
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
        from utils.math_verifier import extract_final_answer, verify_response_answer

        tokenizer = AutoTokenizer.from_pretrained(
            str(generation["student_tokenizer_path"]),
            trust_remote_code=True,
            use_fast=True,
            local_files_only=True,
        )
        required_free = math.ceil(
            _float(generation.get("min_free_gpu_gib", 5.0), "generation.min_free_gpu_gib") * 1024
        )
        assert_gpu_headroom(gpu_id, required_free, stage="student-before-load")
        llm = LLM(**worker_model_kwargs(str(generation["student_model_path"]), generation, max_model_len))
        memory = assert_gpu_headroom(gpu_id, required_free, stage="student-after-load")
        result_queue.put({"type": "ready", "role": "student", "rank": rank, "gpu": gpu_id, "memory": memory})
        mc_k = _int(generation["mc_k"], "generation.mc_k")
        slots = ROLL_OUTS_PER_GPU_CALL // mc_k
        max_total = _int(generation["max_total_response_tokens"], "generation.max_total_response_tokens")
        min_continuation = _int(generation.get("min_continuation_tokens", 1), "generation.min_continuation_tokens")
        check_interval = _float(generation.get("gpu_check_interval_seconds", 5.0), "generation.gpu_check_interval_seconds")
        report_interval = _float(generation.get("gpu_report_interval_seconds", 60.0), "generation.gpu_report_interval_seconds")
        save_texts = bool(generation.get("save_continuation_texts", False))

        branches: list[dict[str, Any]] = []
        for row in tasks:
            response_ids = list(row["response_token_ids"])
            start = int(row["step_token_start"])
            end = int(row["step_token_end"])
            pre = response_ids[:start]
            keep = response_ids[:end]
            replacements = teacher_replacements[str(row["case_id"])]
            prefixes: dict[str, list[int]] = {"redecide": pre, "keep": keep}
            for teacher in teachers:
                name = str(teacher["name"])
                text = str(replacements[name]["replacement_text"])
                replacement_ids = [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]
                if tokenizer.decode(replacement_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != text:
                    raise RuntimeError(f"Student tokenizer cannot round-trip Teacher step for {row['case_id']}.")
                prefixes[f"replace:{name}"] = pre + replacement_ids
            paired_seed = _int(generation.get("seed", 0), "generation.seed") + 10_000_000 + int(row["case_index"])
            for arm, prefix in prefixes.items():
                budget = max_total - len(prefix)
                if budget < min_continuation:
                    raise RuntimeError(f"{arm} has no continuation budget for {row['case_id']}.")
                branches.append(
                    {
                        "case_id": str(row["case_id"]),
                        "case_index": int(row["case_index"]),
                        "arm": arm,
                        "input_ids": list(row["prompt_token_ids"]) + prefix,
                        "response_prefix_ids": prefix,
                        "answer": str(row["answer"]),
                        "seed": paired_seed,
                        "max_tokens": budget,
                    }
                )

        for batch_index, start in enumerate(range(0, len(branches), slots)):
            batch = branches[start : start + slots]
            prompts: list[dict[str, Any]] = []
            params: list[Any] = []
            for branch in batch:
                prompts.append({"prompt_token_ids": branch["input_ids"]})
                params.append(
                    SamplingParams(
                        n=mc_k,
                        temperature=_float(generation.get("temperature", 0.6), "generation.temperature"),
                        top_p=_float(generation.get("top_p", 1.0), "generation.top_p"),
                        top_k=-1,
                        max_tokens=int(branch["max_tokens"]),
                        seed=int(branch["seed"]),
                    )
                )
            prompts, params = _padding_specs(prompts, params, slots)
            started = time.monotonic()
            with GPUHeartbeat(
                gpu_id=gpu_id,
                required_free_mib=required_free,
                check_interval=check_interval,
                report_interval=report_interval,
                stage=f"student/batch_{batch_index}",
            ):
                outputs = llm.generate(prompts, sampling_params=params, use_tqdm=False)
            if len(outputs) != slots:
                raise RuntimeError(f"Student returned {len(outputs)} requests, expected {slots}.")
            records: list[dict[str, Any]] = []
            for branch, output in zip(batch, outputs[: len(batch)], strict=True):
                if list(output.prompt_token_ids) != list(branch["input_ids"]):
                    raise RuntimeError(f"Student changed input IDs for {branch['case_id']}/{branch['arm']}.")
                if len(output.outputs) != mc_k:
                    raise RuntimeError(
                        f"Student returned {len(output.outputs)} samples for {branch['case_id']}/{branch['arm']}, expected {mc_k}."
                    )
                rewards: list[int] = []
                texts: list[str] = []
                truncated_count = 0
                format_invalid_count = 0
                for completion in output.outputs:
                    continuation_ids = [int(value) for value in completion.token_ids]
                    finish_reason = normalized_reason(completion.finish_reason)
                    truncated_count += int(finish_reason == "length")
                    full_ids = list(branch["response_prefix_ids"]) + continuation_ids
                    full_text = tokenizer.decode(
                        full_ids,
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                    extracted, format_valid = extract_final_answer(full_text)
                    del extracted
                    if not format_valid:
                        format_invalid_count += 1
                    reward = int(format_valid and verify_response_answer(full_text, branch["answer"]))
                    rewards.append(reward)
                    if save_texts:
                        texts.append(full_text)
                records.append(
                    {
                        "ok": True,
                        "case_id": branch["case_id"],
                        "case_index": branch["case_index"],
                        "arm": branch["arm"],
                        "mc_k": mc_k,
                        "correct_count": sum(rewards),
                        "value": sum(rewards) / mc_k,
                        "truncated_count": truncated_count,
                        "format_invalid_count": format_invalid_count,
                        "rewards": rewards,
                        "texts": texts if save_texts else None,
                    }
                )
            memory = assert_gpu_headroom(gpu_id, required_free, stage="student-after-batch")
            result_queue.put(
                {
                    "type": "batch",
                    "role": "student",
                    "rank": rank,
                    "gpu": gpu_id,
                    "batch_index": batch_index,
                    "records": records,
                    "completed_items": len(records),
                    "generated_rollouts": len(records) * mc_k,
                    "elapsed_seconds": time.monotonic() - started,
                    "memory": memory,
                }
            )
        result_queue.put({"type": "done", "role": "student", "rank": rank, "gpu": gpu_id})
    except BaseException as exc:
        result_queue.put(
            {
                "type": "error",
                "role": "student",
                "rank": rank,
                "gpu": gpu_id,
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        if llm is not None:
            shutdown_llm(llm)


def run_workers(
    *,
    role: str,
    assignments_by_rank: list[list[dict[str, Any]]],
    gpu_ids: list[int],
    worker_kwargs_by_rank: dict[str, Any],
    output_root: Path,
    total_items: int,
    total_rollouts: int,
    wandb_run: Any,
    phase_label: str,
) -> list[dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    processes: list[mp.Process] = []
    for rank, (gpu_id, assigned) in enumerate(zip(gpu_ids, assignments_by_rank, strict=True)):
        target = teacher_worker if role == "teacher" else student_worker
        kwargs = {
            "gpu_id": gpu_id,
            "rank": rank,
            "tasks": assigned,
            "result_queue": result_queue,
            **worker_kwargs_by_rank,
        }
        process = context.Process(target=target, kwargs=kwargs, daemon=False)
        process.start()
        processes.append(process)

    started = time.monotonic()
    ready: set[int] = set()
    done: set[int] = set()
    completed_items = 0
    generated_rollouts = 0
    log_step = 0
    all_records: list[dict[str, Any]] = []
    last_memory: dict[str, int] | None = None
    try:
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
            message_type = message.get("type")
            if message_type == "ready":
                ready.add(int(message["rank"]))
                print(json.dumps({"event": "worker_ready", "phase": phase_label, **message}, ensure_ascii=False), flush=True)
            elif message_type == "error":
                raise RuntimeError(
                    f"{phase_label} worker rank={message.get('rank')} failed: {message.get('error')}\n"
                    f"{message.get('traceback', '')}"
                )
            elif message_type == "batch":
                rank = int(message["rank"])
                shard_path = output_root / "shards" / f"gpu_{gpu_ids[rank]}_batch_{int(message['batch_index']):06d}.json"
                atomic_json(shard_path, message)
                all_records.extend(message.get("records", []))
                completed_items += int(message.get("completed_items", 0))
                generated_rollouts += int(message.get("generated_rollouts", 0))
                last_memory = message.get("memory") or last_memory
                elapsed = max(time.monotonic() - started, 1e-6)
                rate = completed_items / elapsed
                remaining = max(total_items - completed_items, 0)
                eta = remaining / rate if rate > 0 else None
                progress = {
                    "schema_version": SCHEMA_VERSION,
                    "updated_at": utc_now(),
                    "phase": phase_label,
                    "total_items": total_items,
                    "completed_items": completed_items,
                    "total_rollouts": total_rollouts,
                    "generated_rollouts": generated_rollouts,
                    "progress_fraction": completed_items / total_items if total_items else 1.0,
                    "elapsed_seconds": elapsed,
                    "eta_seconds": eta,
                    "ready_workers": len(ready),
                    "done_workers": len(done),
                    "last_gpu_memory_mib": message.get("memory"),
                }
                atomic_json(output_root / "progress.json", progress)
                print(json.dumps({"event": "progress", **progress}, ensure_ascii=False), flush=True)
                if wandb_run is not None:
                    metrics = {
                        f"ersr/{phase_label}/completed_items": completed_items,
                        f"ersr/{phase_label}/total_items": total_items,
                        f"ersr/{phase_label}/progress": progress["progress_fraction"],
                        f"ersr/{phase_label}/generated_rollouts": generated_rollouts,
                        f"ersr/{phase_label}/elapsed_seconds": elapsed,
                        f"ersr/{phase_label}/eta_seconds": eta or 0.0,
                    }
                    memory = message.get("memory") or {}
                    if "free" in memory:
                        metrics[f"ersr/{phase_label}/gpu_free_mib"] = memory["free"]
                    wandb_run.log(metrics, step=log_step)
                    log_step += 1
            elif message_type == "done":
                done.add(int(message["rank"]))
            else:
                raise RuntimeError(f"Unknown worker message: {message!r}")
        # Persist an explicit terminal heartbeat after every worker reports
        # done.  This prevents a completed phase from appearing stuck at
        # ``done_workers=0`` in the last batch progress file.
        elapsed = max(time.monotonic() - started, 1e-6)
        final_progress = {
            "schema_version": SCHEMA_VERSION,
            "updated_at": utc_now(),
            "phase": phase_label,
            "total_items": total_items,
            "completed_items": completed_items,
            "total_rollouts": total_rollouts,
            "generated_rollouts": generated_rollouts,
            "progress_fraction": completed_items / total_items if total_items else 1.0,
            "elapsed_seconds": elapsed,
            "eta_seconds": 0.0,
            "ready_workers": len(ready),
            "done_workers": len(done),
            "last_gpu_memory_mib": last_memory,
        }
        atomic_json(output_root / "progress.json", final_progress)
        print(json.dumps({"event": "progress", **final_progress}, ensure_ascii=False), flush=True)
        if wandb_run is not None:
            final_metrics = {
                f"ersr/{phase_label}/completed_items": completed_items,
                f"ersr/{phase_label}/total_items": total_items,
                f"ersr/{phase_label}/progress": final_progress["progress_fraction"],
                f"ersr/{phase_label}/generated_rollouts": generated_rollouts,
                f"ersr/{phase_label}/elapsed_seconds": elapsed,
                f"ersr/{phase_label}/eta_seconds": 0.0,
            }
            if last_memory is not None and "free" in last_memory:
                final_metrics[f"ersr/{phase_label}/gpu_free_mib"] = last_memory["free"]
            wandb_run.log(final_metrics, step=log_step)
    except BaseException:
        for process in processes:
            if process.is_alive():
                process.terminate()
        raise
    finally:
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
    return all_records


def estimate_max_model_len(selected: list[dict[str, Any]], generation: dict[str, Any]) -> int:
    configured = _int(generation.get("max_model_len", 0), "generation.max_model_len")
    max_prompt = max(len(row["prompt_token_ids"]) for row in selected)
    required = max_prompt + _int(generation["max_total_response_tokens"], "generation.max_total_response_tokens")
    if configured and configured < required:
        raise ValueError(f"generation.max_model_len={configured} is smaller than required {required}.")
    return configured or required


def init_wandb(config: dict[str, Any], output_dir: Path) -> Any:
    settings = config.get("wandb", {}) or {}
    mode = str(settings.get("mode", "online"))
    if mode in {"disabled", "off", "false"}:
        return None
    try:
        import wandb

        init_kwargs: dict[str, Any] = {
            "project": str(settings.get("project", "OPD_Forge")),
            "name": str(settings.get("run_name", output_dir.name)),
            "mode": mode,
            "config": {
                "experiment": "ersr_mc",
                "output_dir": str(output_dir),
                "config_sha256": canonical_sha256(config),
            },
        }
        if settings.get("group"):
            init_kwargs["group"] = str(settings["group"])
        if settings.get("entity"):
            init_kwargs["entity"] = str(settings["entity"])
        return wandb.init(**init_kwargs)
    except Exception:
        if bool(settings.get("required", True)):
            raise
        print(json.dumps({"event": "wandb_disabled_after_init_failure"}, ensure_ascii=False), flush=True)
        return None


def merge_teacher_records(records: list[dict[str, Any]], teacher_name: str) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in records:
        if not bool(row.get("ok", False)):
            continue
        case_id = str(row["case_id"])
        if case_id in merged:
            raise ValueError(f"Duplicate Teacher result for {teacher_name}/{case_id}.")
        merged[case_id] = row
    return merged


def merge_student_records(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    merged: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in records:
        if not bool(row.get("ok", False)):
            continue
        case_id = str(row["case_id"])
        arm = str(row["arm"])
        if arm in merged[case_id]:
            raise ValueError(f"Duplicate Student result for {case_id}/{arm}.")
        merged[case_id][arm] = row
    return merged


def build_results(
    selected: list[dict[str, Any]],
    teachers: list[dict[str, Any]],
    teacher_results: dict[str, dict[str, dict[str, Any]]],
    student_results: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for row in sorted(selected, key=lambda value: int(value["case_index"])):
        case_id = str(row["case_id"])
        arms = student_results.get(case_id, {})
        required_arms = set(MC_ARMS) | {f"replace:{teacher['name']}" for teacher in teachers}
        missing = sorted(required_arms - set(arms))
        if missing:
            raise RuntimeError(f"Missing Student arms for {case_id}: {missing}.")
        replacements: dict[str, Any] = {}
        values: dict[str, float] = {
            "redecide": float(arms["redecide"]["value"]),
            "keep": float(arms["keep"]["value"]),
        }
        samples: dict[str, Any] = {
            arm: {
                "mc_k": int(arms[arm]["mc_k"]),
                "correct_count": int(arms[arm]["correct_count"]),
                "truncated_count": int(arms[arm]["truncated_count"]),
                "format_invalid_count": int(arms[arm]["format_invalid_count"]),
                "rewards": arms[arm]["rewards"],
            }
            for arm in MC_ARMS
        }
        for teacher in teachers:
            name = str(teacher["name"])
            replacement = teacher_results[name].get(case_id)
            if replacement is None:
                raise RuntimeError(f"Missing Teacher replacement for {name}/{case_id}.")
            arm_name = f"replace:{name}"
            values[arm_name] = float(arms[arm_name]["value"])
            samples[arm_name] = {
                "mc_k": int(arms[arm_name]["mc_k"]),
                "correct_count": int(arms[arm_name]["correct_count"]),
                "truncated_count": int(arms[arm_name]["truncated_count"]),
                "format_invalid_count": int(arms[arm_name]["format_invalid_count"]),
                "rewards": arms[arm_name]["rewards"],
            }
            replacements[name] = {
                "text": replacement["replacement_text"],
                "proposal_text": replacement.get("proposal_text"),
                "finish_reason": replacement["finish_reason"],
                "proposal_num_tokens": int(replacement["proposal_num_tokens"]),
                "replacement_num_tokens": int(replacement["replacement_num_tokens"]),
                "truncated": int(replacement.get("truncated", 0)),
            }
        advantages = {
            "A_k": values["keep"] - values["redecide"],
            **{
                f"A_k_TR:{teacher['name']}": values[f"replace:{teacher['name']}"] - values["redecide"]
                for teacher in teachers
            },
        }
        named_values = {
            "V_redecide": values["redecide"],
            "V_keep": values["keep"],
            **{
                f"V_replace:{teacher['name']}": values[f"replace:{teacher['name']}"]
                for teacher in teachers
            },
        }
        results.append(
            {
                "schema_version": SCHEMA_VERSION,
                "case_id": case_id,
                "case_index": int(row["case_index"]),
                "response_idx": int(row["response_idx"]),
                "rollout_index": int(row["rollout_index"]),
                "bucket": int(row["bucket"]),
                "rollout_correct": int(row["rollout_correct"]),
                "question": str(row["question"]),
                "answer": str(row["answer"]),
                "response": str(row["response"]),
                "step_index": int(row["step_index"]),
                "reasoning_step_index": int(row["reasoning_step_index"]),
                "num_reasoning_steps": int(row["num_reasoning_steps"]),
                "step_token_start": int(row["step_token_start"]),
                "step_token_end": int(row["step_token_end"]),
                "step_text": str(row["step_text"]),
                "prompt_token_ids": [int(value) for value in row["prompt_token_ids"]],
                "response_token_ids": [int(value) for value in row["response_token_ids"]],
                "position_fraction": float(row["position_fraction"]),
                "teacher_replacements": replacements,
                "values": values,
                "V": named_values,
                "advantages": advantages,
                "samples": samples,
            }
        )
    return results


def summarize_results(results: list[dict[str, Any]], elapsed_seconds: float) -> dict[str, Any]:
    by_bucket: dict[str, dict[str, Any]] = {}
    for bucket in BUCKETS:
        rows = [row for row in results if int(row["bucket"]) == bucket]
        teacher_truncated = sum(
            int(value.get("truncated", 0))
            for row in rows
            for value in row["teacher_replacements"].values()
        )
        mc_truncated = sum(
            int(value.get("truncated_count", 0))
            for row in rows
            for value in row["samples"].values()
        )
        mc_format_invalid = sum(
            int(value.get("format_invalid_count", 0))
            for row in rows
            for value in row["samples"].values()
        )
        by_bucket[str(bucket)] = {
            "cases": len(rows),
            "teacher_truncated_replacements": teacher_truncated,
            "mc_truncated_samples": mc_truncated,
            "mc_format_invalid_samples": mc_format_invalid,
            "mean_A_k": sum(float(row["advantages"]["A_k"]) for row in rows) / len(rows) if rows else 0.0,
            "mean_A_k_TR": {
                name: sum(float(row["advantages"][name]) for row in rows) / len(rows)
                if rows
                else 0.0
                for name in (rows[0]["advantages"] if rows else {})
                if name.startswith("A_k_TR:")
            },
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "cases": len(results),
        "elapsed_seconds": elapsed_seconds,
        "teacher_truncated_replacements": sum(
            int(value.get("truncated", 0))
            for row in results
            for value in row["teacher_replacements"].values()
        ),
        "mc_truncated_samples": sum(
            int(value.get("truncated_count", 0))
            for row in results
            for value in row["samples"].values()
        ),
        "mc_format_invalid_samples": sum(
            int(value.get("format_invalid_count", 0))
            for row in results
            for value in row["samples"].values()
        ),
        "by_accuracy_bucket": by_bucket,
        "serialization": {"format": "json", "encoding": "utf-8", "indent": 4, "ensure_ascii": False},
    }


def persisted_phase_elapsed(output_dir: Path, teachers: list[dict[str, Any]]) -> float:
    """Recover elapsed generation time when a completed run is resumed.

    A resume should not replace the original wall-clock duration with the few
    milliseconds needed to re-read completed shards and rebuild the summary.
    """

    phase_roots = [
        output_dir / "teachers" / str(teacher["name"])
        for teacher in teachers
    ] + [output_dir / "student"]
    elapsed = 0.0
    for root in phase_roots:
        progress_path = root / "progress.json"
        if not progress_path.is_file():
            continue
        with contextlib.suppress(Exception):
            elapsed += max(0.0, float(load_json(progress_path).get("elapsed_seconds", 0.0)))
    return elapsed


def validate_config(config: dict[str, Any]) -> None:
    if str(config.get("input", {}).get("prompt_name", "")) != "qwen3_no_thinking_prompt":
        raise ValueError("input.prompt_name must be qwen3_no_thinking_prompt.")
    teachers = config.get("teachers")
    if not isinstance(teachers, list) or not teachers:
        raise ValueError("At least one Teacher must be configured.")
    names: set[str] = set()
    for teacher in teachers:
        if not isinstance(teacher, dict):
            raise ValueError("Each Teacher entry must be a mapping.")
        name = str(teacher.get("name", "")).strip()
        if not name or name in names:
            raise ValueError(f"Teacher names must be non-empty and unique: {name!r}.")
        names.add(name)
        if not resolve_path(teacher["model_path"]).is_dir():
            raise FileNotFoundError(resolve_path(teacher["model_path"]))
        if _int(teacher.get("max_step_new_tokens", 0), "teacher.max_step_new_tokens") <= 0:
            raise ValueError(f"Teacher {name} has an invalid max_step_new_tokens.")
    generation = config.get("generation", {}) or {}
    per_gpu = _int(generation.get("per_gpu_rollouts", 0), "generation.per_gpu_rollouts")
    mc_k = _int(generation.get("mc_k", 0), "generation.mc_k")
    if per_gpu != ROLL_OUTS_PER_GPU_CALL:
        raise ValueError("generation.per_gpu_rollouts must be exactly 256.")
    if mc_k <= 0 or per_gpu % mc_k:
        raise ValueError("generation.mc_k must be a positive divisor of 256.")
    if _int(generation.get("max_total_response_tokens", 0), "generation.max_total_response_tokens") <= 0:
        raise ValueError("generation.max_total_response_tokens must be positive.")
    if not 0.0 < _float(generation.get("gpu_memory_utilization", 0.78), "generation.gpu_memory_utilization") < 1.0:
        raise ValueError("generation.gpu_memory_utilization must lie in (0, 1).")
    if _float(generation.get("min_free_gpu_gib", 5.0), "generation.min_free_gpu_gib") < 5.0:
        raise ValueError("At least 5 GiB of free GPU memory must be preserved.")
    student_path = resolve_path(config["student"]["model_path"])
    if not student_path.is_dir():
        raise FileNotFoundError(student_path)
    if resolve_path(config["input"]["tokenizer_path"]).is_dir() is False:
        raise FileNotFoundError(resolve_path(config["input"]["tokenizer_path"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    validate_config(config)
    gpu_ids = configured_gpus(config)
    output_dir = resolve_path(config["output"]["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(output_dir / "run_config.json", config)
    candidates, candidate_summary = build_candidates(config)
    selected, selection_summary = select_steps(candidates, config)
    atomic_json(output_dir / "selection_manifest.json", [manifest_row(row) for row in selected])
    atomic_json(output_dir / "selection_summary.json", {**candidate_summary, **selection_summary})
    print(
        json.dumps(
            {
                "event": "selection_ready",
                "gpu_ids": gpu_ids,
                "output_dir": str(output_dir),
                "candidate_summary": candidate_summary,
                "selection_summary": selection_summary,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    if args.validate_only:
        print("VALIDATION_OK", flush=True)
        return

    generation = dict(config["generation"])
    generation["student_model_path"] = str(resolve_path(config["student"]["model_path"]))
    generation["student_tokenizer_path"] = str(resolve_path(config["input"]["tokenizer_path"]))
    max_model_len = estimate_max_model_len(selected, generation)
    wandb_run = init_wandb(config, output_dir)
    started = time.monotonic()
    teacher_results: dict[str, dict[str, dict[str, Any]]] = {}
    try:
        for teacher in config["teachers"]:
            name = str(teacher["name"])
            teacher_root = output_dir / "teachers" / name
            completed = load_completed_ids(teacher_root, key="case_id")
            pending = [row for row in selected if str(row["case_id"]) not in completed]
            task_assignments = assignments(pending, gpu_ids) if pending else [[] for _ in gpu_ids]
            teacher_model_len = max_model_len
            run_workers(
                role="teacher",
                assignments_by_rank=task_assignments,
                gpu_ids=gpu_ids,
                worker_kwargs_by_rank={
                    "teacher": {
                        **teacher,
                        "model_path": str(resolve_path(teacher["model_path"])),
                    },
                    "generation": generation,
                    "max_model_len": teacher_model_len,
                },
                output_root=teacher_root,
                total_items=len(selected),
                total_rollouts=len(selected),
                wandb_run=wandb_run,
                phase_label=f"teacher_{name}",
            ) if pending else []
            records = [
                row
                for path in sorted((teacher_root / "shards").glob("*.json"))
                for row in load_json(path).get("records", [])
            ]
            teacher_results[name] = merge_teacher_records(records, name)
            if len(teacher_results[name]) != len(selected):
                raise RuntimeError(
                    f"Teacher {name} produced {len(teacher_results[name])}/{len(selected)} replacements."
                )

        # Student re-decide/keep are shared across Teachers; all configured
        # replacement arms are sampled in this single Student phase.
        student_root = output_dir / "student"
        # A case is resumable only when all arms are present.  Read the shards
        # to build an exact case/arm completion set.
        completed_arms: dict[str, set[str]] = defaultdict(set)
        for path in sorted((student_root / "shards").glob("*.json")):
            with contextlib.suppress(Exception):
                for row in load_json(path).get("records", []):
                    if bool(row.get("ok", False)):
                        completed_arms[str(row["case_id"])].add(str(row["arm"]))
        expected_arms = set(MC_ARMS) | {f"replace:{teacher['name']}" for teacher in config["teachers"]}
        pending_student = [
            row for row in selected if completed_arms.get(str(row["case_id"]), set()) < expected_arms
        ]
        student_assignments = assignments(pending_student, gpu_ids) if pending_student else [[] for _ in gpu_ids]
        run_workers(
            role="student",
            assignments_by_rank=student_assignments,
            gpu_ids=gpu_ids,
            worker_kwargs_by_rank={
                "teacher_replacements": {
                    case_id: {
                        teacher_name: {"replacement_text": row["replacement_text"]}
                        for teacher_name, mapping in teacher_results.items()
                        for case_id2, row in mapping.items()
                        if case_id2 == case_id
                    }
                    for case_id in (str(row["case_id"]) for row in pending_student)
                },
                "teachers": [
                    {**teacher, "model_path": str(resolve_path(teacher["model_path"]))}
                    for teacher in config["teachers"]
                ],
                "generation": generation,
                "max_model_len": max_model_len,
            },
            output_root=student_root,
            total_items=len(selected) * len(expected_arms),
            total_rollouts=len(selected) * len(expected_arms) * _int(generation["mc_k"], "generation.mc_k"),
            wandb_run=wandb_run,
            phase_label="student_values",
        ) if pending_student else []
        student_records = [
            row
            for path in sorted((student_root / "shards").glob("*.json"))
            for row in load_json(path).get("records", [])
        ]
        merged_student = merge_student_records(student_records)
        results = build_results(selected, config["teachers"], teacher_results, merged_student)
        elapsed_seconds = max(
            time.monotonic() - started,
            persisted_phase_elapsed(output_dir, config["teachers"]),
        )
        summary = summarize_results(results, elapsed_seconds)
        atomic_json(output_dir / "ersr_results.json", results)
        atomic_json(output_dir / "ersr_summary.json", summary)
        print(json.dumps({"event": "complete", **summary}, ensure_ascii=False), flush=True)
        if wandb_run is not None:
            wandb_run.summary.update(
                {
                    "ersr/cases": len(results),
                    "ersr/elapsed_seconds": summary["elapsed_seconds"],
                    "ersr/mean_A_k": sum(float(row["advantages"]["A_k"]) for row in results) / len(results),
                }
            )
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
