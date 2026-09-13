#!/usr/bin/env python3
"""Run the 20k-step OA--OPD Avg@128 collection on independent GPUs.

This file is intentionally a tests-only experiment runner.  It consumes the
response-level JSON files produced by ``prepare_oa_opd_h20_avg128_20k.py`` and
does not change the production training path.  Every worker owns a disjoint
set of global shards and loads exactly one model at a time.  The
``TEACHER_MODELS`` dictionary below is the user-editable source of truth: each
named Teacher generates one 1024-token proposal for every selected step, and a
single Student instance generates a separate Replace arm (128 continuations)
for each Teacher.  Keep and Delete are shared Student baselines.

``--phase all`` (the default used by the companion shell script) runs the two
phases sequentially in one process, releasing each Teacher before loading the
Student.  A generation micro-batch contains two prompts, hence exactly
``2 * 128 = 256`` expanded continuations.  This is deliberately different
from passing 256 prompts to vLLM, which would create 32,768 active sequences.

All input prefixes are exact slices of the saved token IDs.  Shards are
atomic and resumable.  Continuation shards are gzip-compressed JSON to keep
the large process log practical; exact continuation IDs, complete response
text, verifier results, and provenance are retained, while duplicate decoded
strings are omitted.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import gzip
import hashlib
import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.oa_opd_step_selection import strict_incorrect_source

try:  # The local analysis helper is available in the development checkout.
    from analyze import run_oa_opd_step_eval_vllm as base
except ModuleNotFoundError:  # Keep this tests-only runner portable to a clean clone.
    class _Runtime:
        MIB_PER_GIB = 1024

        @staticmethod
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

        @staticmethod
        def text_sha256(value: str) -> str:
            return hashlib.sha256(value.encode("utf-8")).hexdigest()

        @staticmethod
        def normalized_reason(value: Any) -> str | int | None:
            if value is None or isinstance(value, (str, int)):
                return value
            enum_value = getattr(value, "value", None)
            return str(enum_value if enum_value is not None else value)

        @staticmethod
        def decode_ids(tokenizer: Any, token_ids: Iterable[int], *, skip_special_tokens: bool) -> str:
            return tokenizer.decode([int(v) for v in token_ids],
                                    skip_special_tokens=skip_special_tokens,
                                    clean_up_tokenization_spaces=False)

        @staticmethod
        def query_gpu_memory_mib(gpu_id: int) -> dict[str, int]:
            result = subprocess.run(
                ["nvidia-smi", f"--id={gpu_id}",
                 "--query-gpu=memory.total,memory.used,memory.free",
                 "--format=csv,noheader,nounits"],
                check=True, capture_output=True, text=True,
            )
            fields = result.stdout.strip().splitlines()[0].split(",")
            if len(fields) != 3:
                raise RuntimeError(f"unexpected nvidia-smi output: {result.stdout!r}")
            total, used, free = (int(float(v.strip())) for v in fields)
            return {"total": total, "used": used, "free": free}

        @classmethod
        def assert_gpu_headroom(cls, gpu_id: int, minimum_free_mib: int, *, stage: str) -> dict[str, int]:
            memory = cls.query_gpu_memory_mib(gpu_id)
            if memory["free"] < minimum_free_mib:
                raise RuntimeError(f"GPU reserve violated at {stage}: {memory['free']} < {minimum_free_mib} MiB")
            return memory

        class GPUHeartbeat:
            def __init__(self, *, gpu_id: int, minimum_free_mib: int,
                         check_interval: float, report_interval: float, stage: str) -> None:
                self.gpu_id = gpu_id
                self.minimum_free_mib = minimum_free_mib
                self.check_interval = check_interval
                self.report_interval = report_interval
                self.stage = stage
                self.stop_event = threading.Event()
                self.thread: threading.Thread | None = None
                self.violation: dict[str, int] | None = None

            def __enter__(self) -> "_Runtime.GPUHeartbeat":
                started = time.monotonic()

                def watch() -> None:
                    last_report = started
                    while not self.stop_event.wait(self.check_interval):
                        try:
                            memory = _Runtime.query_gpu_memory_mib(self.gpu_id)
                        except Exception as exc:
                            print(json.dumps({"event": "gpu_heartbeat_query_failed",
                                              "stage": self.stage, "error": repr(exc)}), flush=True)
                            continue
                        now = time.monotonic()
                        if memory["free"] < self.minimum_free_mib or now - last_report >= self.report_interval:
                            print(json.dumps({"event": "gpu_heartbeat", "stage": self.stage,
                                              "elapsed_seconds": round(now - started, 1),
                                              "gpu_memory_mib": memory,
                                              "required_free_mib": self.minimum_free_mib}), flush=True)
                            last_report = now
                        if memory["free"] < self.minimum_free_mib:
                            self.violation = memory
                            os.kill(os.getpid(), signal.SIGINT)
                            return

                self.thread = threading.Thread(target=watch, daemon=True)
                self.thread.start()
                return self

            def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
                self.stop_event.set()
                if self.thread is not None:
                    self.thread.join(timeout=max(2.0, self.check_interval * 2.0))
                if exc_type is None and self.violation is not None:
                    raise RuntimeError(f"GPU reserve violated during {self.stage}")

        @staticmethod
        def shutdown_vllm(llm: Any) -> None:
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

        @staticmethod
        def completion_result(*, completion: Any, tokenizer: Any,
                              response_prefix_ids: list[int], max_response_tokens: int,
                              ground_truth: Any, sample_index: int) -> dict[str, Any]:
            from utils.math_verifier import extract_final_answer, verify_response_answer
            continuation_ids = [int(v) for v in completion.token_ids]
            full_ids = response_prefix_ids + continuation_ids
            if len(full_ids) > max_response_tokens:
                raise RuntimeError("vLLM exceeded the response budget")
            full_text = _Runtime.decode_ids(tokenizer, full_ids, skip_special_tokens=False)
            extracted, format_valid = extract_final_answer(full_text)
            finish = _Runtime.normalized_reason(completion.finish_reason)
            budget = max_response_tokens - len(response_prefix_ids)
            return {
                "sample_index": sample_index,
                "continuation_token_ids": continuation_ids,
                "continuation_num_tokens": len(continuation_ids),
                "continuation_token_ids_sha256": _Runtime.token_ids_sha256(continuation_ids),
                "continuation_text": _Runtime.decode_ids(tokenizer, continuation_ids, skip_special_tokens=False),
                "vllm_completion_text": getattr(completion, "text", ""),
                "full_response_num_tokens": len(full_ids),
                "full_response_token_ids_sha256": _Runtime.token_ids_sha256(full_ids),
                "full_response_text": full_text,
                "full_response_text_sha256": _Runtime.text_sha256(full_text),
                "finish_reason": finish,
                "stop_reason": _Runtime.normalized_reason(getattr(completion, "stop_reason", None)),
                "truncated_by_response_cap": bool(finish == "length" and len(continuation_ids) >= budget),
                "extracted_answer": extracted,
                "format_valid": int(format_valid),
                "correct": int(verify_response_answer(full_text, str(ground_truth))),
            }

    base = _Runtime()


# Version 2 records the compact completion payload and run-ID-tagged progress
# introduced for this large collection. The input preparation files remain
# schema version 1 and are validated independently below.
SCHEMA_VERSION = 2
EXPERIMENT = "oa_opd_h20_keep_delete_replace_avg128_20k_steps"
OUTCOMES = {"correct", "incorrect"}
ARMS = ("keep", "delete", "replace")
POSITION_BINS = ("p1", "p2", "p3", "p4", "p5")

# Scientific settings are intentionally fixed here rather than in YAML.
SAMPLES_PER_ARM = 128
PROMPTS_PER_GENERATE = 2
EXPANDED_ROLLOUTS_PER_GENERATE = SAMPLES_PER_ARM * PROMPTS_PER_GENERATE
SHARD_CASES = 32
MAX_RESPONSE_TOKENS = 10240
TEACHER_PROPOSAL_MAX_NEW_TOKENS = 1024
TEACHER_PROPOSAL_ATTEMPTS = 3
TEMPERATURE = 0.6
TOP_P = 0.95
TOP_K = -1
SEED = 20260911
MIN_FREE_GPU_GIB = 5.0
GPU_SAFETY_MARGIN_MIB = 512
MAX_NUM_SEQS = EXPANDED_ROLLOUTS_PER_GENERATE
MAX_NUM_BATCHED_TOKENS = 32768
STUDENT_BASENAME = "Qwen3-1.7B"
TEACHER_BASENAME = "Qwen3-4B-Instruct-2507"

# Configure all Teachers for a multi-Teacher run here.  The dictionary key is
# a stable, filesystem-safe identifier used in output paths; the value is the
# absolute local model directory.  Replace these two WSL paths with the
# corresponding absolute paths on the server.  Editing this mapping is the
# recommended way to add or remove Teachers.  A legacy ``--teacher-model`` CLI
# value, when supplied, intentionally overrides this mapping with a single
# Teacher so old launch commands remain valid.
TEACHER_MODELS: dict[str, Path | str] = {
    "qwen3_4b_instruct_2507": "/models/Qwen3-4B-Instruct-2507",
    "qwen3_30b_a3b_instruct_2507_fp8": "/models/Qwen3-30B-A3B-Instruct-2507-FP8",
}

# Keep the exact generated token IDs and the decoded complete response for
# offline auditing, but omit two redundant per-sample strings (the decoded
# continuation and vLLM's duplicate completion text). The source case stores
# the exact prefix IDs, so the full response can be reconstructed as
# ``prefix_ids + continuation_token_ids`` when needed. This keeps the
# compressed collection practical at millions of rollouts without losing
# reward, verifier, or token-level information.
COMPLETION_FIELDS_TO_SAVE = (
    "sample_index",
    "continuation_token_ids",
    "continuation_num_tokens",
    "continuation_token_ids_sha256",
    "full_response_num_tokens",
    "full_response_token_ids_sha256",
    "full_response_text",
    "full_response_text_sha256",
    "finish_reason",
    "stop_reason",
    "truncated_by_response_cap",
    "extracted_answer",
    "format_valid",
    "correct",
)

DEFAULT_DATASET = {
    "correct": REPO_ROOT / "data" / "correct_2000_trajectories_10000_steps.json",
    "incorrect": REPO_ROOT / "data" / "incorrect_2000_trajectories_10000_steps.json",
}
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "oa_opd_h20_avg128_20k"
DEFAULT_STUDENT_MODEL = REPO_ROOT / "models" / STUDENT_BASENAME
DEFAULT_TEACHER_MODEL = REPO_ROOT / "models" / TEACHER_BASENAME

KEY_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _validate_teacher_key(key: str) -> str:
    """Return *key* after enforcing the safe output-directory alphabet."""

    value = str(key)
    if not KEY_RE.fullmatch(value):
        raise ValueError(
            f"invalid Teacher key {value!r}; use lowercase letters, digits, '.', '_' or '-'."
        )
    return value


def configured_teacher_models() -> dict[str, Path]:
    """Normalize and validate the module-level Teacher mapping.

    A fresh mapping is returned so callers can safely modify it while
    constructing a run plan without mutating the user-editable constant.
    """

    if not isinstance(TEACHER_MODELS, dict) or not TEACHER_MODELS:
        raise ValueError("TEACHER_MODELS must be a non-empty dict of key -> model path")
    result: dict[str, Path] = {}
    for raw_key, raw_path in TEACHER_MODELS.items():
        key = _validate_teacher_key(str(raw_key))
        if key in result:
            raise ValueError(f"duplicate Teacher key: {key}")
        if not isinstance(raw_path, (str, os.PathLike, Path)):
            raise TypeError(f"Teacher model path for {key!r} must be a path or string")
        result[key] = Path(raw_path)
    return result


def resolve_teacher_models(args: argparse.Namespace) -> dict[str, Path]:
    """Resolve the Teacher plan for an invocation.

    The normal path is the top-level :data:`TEACHER_MODELS` dictionary.  For
    backwards compatibility, an explicitly supplied ``--teacher-model`` (or
    an object carrying a non-``None`` ``teacher_model`` attribute) selects a
    single model.  Repeated ``--teacher-key`` values can select a subset of
    the configured mapping; with the legacy override one key may be supplied
    to name that model.
    """

    cli_model = getattr(args, "teacher_model", None)
    raw_keys = getattr(args, "teacher_key", None)
    if raw_keys is None:
        selected_keys: list[str] = []
    elif isinstance(raw_keys, (list, tuple)):
        selected_keys = [_validate_teacher_key(str(value)) for value in raw_keys]
    else:
        selected_keys = [_validate_teacher_key(str(raw_keys))]
    if len(selected_keys) != len(set(selected_keys)):
        raise ValueError("duplicate Teacher key selection")
    if cli_model is not None:
        model = Path(cli_model)
        if len(selected_keys) > 1:
            raise ValueError("legacy --teacher-model accepts at most one --teacher-key")
        key = selected_keys[0] if selected_keys else None
        if not key:
            raw_key = re.sub(r"[^a-z0-9._-]+", "_", model.name.lower()).strip("._-")
            key = _validate_teacher_key(raw_key or "teacher")
        return {key: model}
    configured = configured_teacher_models()
    if not selected_keys:
        return configured
    missing = [key for key in selected_keys if key not in configured]
    if missing:
        raise KeyError(
            "unknown Teacher key(s): " + ", ".join(missing)
            + "; configured keys are " + ", ".join(configured)
        )
    return {key: configured[key] for key in selected_keys}


def resolve_teacher_model_paths(models: dict[str, Path | str]) -> dict[str, Path]:
    """Resolve mapping paths relative to the repository, not the shell CWD.

    A model mapping lives in this Python file, so a relative value such as
    ``models/Qwen3-4B`` should mean a path under the checkout even when the
    launcher is invoked from another directory.  Absolute paths are preserved.
    """

    resolved: dict[str, Path] = {}
    for raw_key, raw_path in models.items():
        key = _validate_teacher_key(raw_key)
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = REPO_ROOT / path
        resolved[key] = path.resolve()
    return resolved


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


def atomic_gzip_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    try:
        with gzip.open(tmp, "wt", encoding="utf-8", newline="\n", compresslevel=5) as handle:
            json.dump(value, handle, ensure_ascii=False, separators=(",", ":"))
            handle.write("\n")
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_json_maybe_gzip(path: Path) -> Any:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def ints(values: Any, field: str) -> list[int]:
    if not isinstance(values, list) or any(type(v) is not int for v in values):
        raise ValueError(f"{field} must be a list of integer token IDs")
    return [int(v) for v in values]


def token_hash(values: Iterable[int]) -> str:
    return base.token_ids_sha256([int(v) for v in values])


def text_hash(value: str) -> str:
    return base.text_sha256(value)


def proposal_seed(case_index: int, attempt: int) -> int:
    # Stable across workers and independent of shard order.
    return SEED + 10_000_000 + int(case_index) * TEACHER_PROPOSAL_ATTEMPTS + (attempt - 1)


def continuation_seed(case_index: int) -> int:
    return SEED + 20_000_000 + int(case_index)


def _source_is_clean(row: dict[str, Any], outcome: str) -> bool:
    if outcome == "correct":
        return (
            int(row.get("source_correct", 0)) == 1
            and int(row.get("source_format_valid", 0)) == 1
            and str(row.get("source_finish_reason", "")) == "stop"
        )
    return strict_incorrect_source(
        {
            "correct": row.get("source_correct", 1),
            "format_valid": row.get("source_format_valid", 0),
            "finish_reason": row.get("source_finish_reason", ""),
            "answer": row.get("answer", ""),
            "extracted_answer": row.get("source_extracted_answer", ""),
        }
    )


def _case_from_selected(
    response: dict[str, Any], selected: dict[str, Any], metadata: dict[str, Any],
    *, prompt_ids: list[int] | None = None, response_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Expand one nested selected step into an exact-token flat case."""

    prompt = prompt_ids if prompt_ids is not None else ints(response["prompt_token_ids"], "prompt_token_ids")
    response_ids = response_ids if response_ids is not None else ints(response["source_response_token_ids"], "source_response_token_ids")
    if len(prompt) != int(response["prompt_num_tokens"]):
        raise ValueError(f"prompt length mismatch for {response.get('trajectory_id')}")
    if len(response_ids) != int(response["source_response_num_tokens"]):
        raise ValueError(f"response length mismatch for {response.get('trajectory_id')}")
    if token_hash(prompt) != response["prompt_token_ids_sha256"]:
        raise ValueError(f"prompt hash mismatch for {response.get('trajectory_id')}")
    if token_hash(response_ids) != response["source_response_token_ids_sha256"]:
        raise ValueError(f"response hash mismatch for {response.get('trajectory_id')}")
    partition = response["step_partition"]
    steps = partition["steps"]
    offset = int(selected["step_offset"])
    if not 0 <= offset < len(steps):
        raise ValueError(f"invalid selected step offset in {response.get('trajectory_id')}")
    canonical = steps[offset]
    for field in ("step_index", "step_token_start", "step_token_end", "step_char_start", "step_char_end"):
        selected_field = {
            "step_index": "step_index",
            "step_token_start": "token_start",
            "step_token_end": "token_end",
            "step_char_start": "char_start",
            "step_char_end": "char_end",
        }[field]
        if int(selected[field]) != int(canonical[selected_field]):
            raise ValueError(f"selected/canonical {field} mismatch")
    start = int(canonical["token_start"])
    end = int(canonical["token_end"])
    if not 0 <= start < end <= len(response_ids):
        raise ValueError(f"invalid token span for {response.get('trajectory_id')}")
    step_ids = response_ids[start:end]
    if token_hash(step_ids) != selected["step_token_ids_sha256"]:
        raise ValueError(f"step token hash mismatch for {response.get('trajectory_id')}")
    step_text = str(canonical["text"])
    if text_hash(step_text) != selected["step_text_sha256"]:
        raise ValueError(f"step text hash mismatch for {response.get('trajectory_id')}")
    source_text = str(response["source_response"])
    if text_hash(source_text) != response["source_response_text_sha256"]:
        raise ValueError(f"source response text hash mismatch for {response.get('trajectory_id')}")
    char_start, char_end = int(canonical["char_start"]), int(canonical["char_end"])
    if source_text[char_start:char_end] != step_text:
        raise ValueError(f"step character span mismatch for {response.get('trajectory_id')}")

    pre = response_ids[:start]
    keep = response_ids[:end]
    # Prepared cohorts may have been built with an older, stricter cap (the
    # checked-in 20k cohort was prepared at 8192/512).  Such a cohort is safe
    # to evaluate under a larger runtime budget, provided its recorded limits
    # are truthful and do not exceed the current experiment limits.  We keep
    # both values in the case for provenance, while all generated arms use the
    # current runner constants below.
    dataset_max_response = int(metadata.get("max_response_tokens", MAX_RESPONSE_TOKENS))
    dataset_proposal_cap = int(
        metadata.get("teacher_proposal_max_new_tokens", TEACHER_PROPOSAL_MAX_NEW_TOKENS)
    )
    if dataset_max_response <= 0 or dataset_proposal_cap <= 0:
        raise ValueError("dataset generation caps must be positive")
    if dataset_max_response > MAX_RESPONSE_TOKENS:
        raise ValueError(
            "dataset max_response_tokens exceeds the runner response cap"
        )
    if dataset_proposal_cap > TEACHER_PROPOSAL_MAX_NEW_TOKENS:
        raise ValueError(
            "dataset teacher proposal cap exceeds the runner proposal cap"
        )
    if len(response_ids) > dataset_max_response:
        raise ValueError(
            f"source response exceeds its recorded cap for {response.get('trajectory_id')}"
        )
    if len(pre) + TEACHER_PROPOSAL_MAX_NEW_TOKENS >= MAX_RESPONSE_TOKENS:
        raise ValueError(f"no replacement budget remains for {response.get('trajectory_id')}")
    if len(keep) >= MAX_RESPONSE_TOKENS:
        raise ValueError(f"no keep budget remains for {response.get('trajectory_id')}")

    outcome = str(response["source_outcome"])
    if outcome not in OUTCOMES or not _source_is_clean(response, outcome):
        raise ValueError(f"source outcome audit failed for {response.get('trajectory_id')}")
    case_index = int(selected["global_case_index"])
    return {
        "schema_version": SCHEMA_VERSION,
        "case_index": case_index,
        "global_case_index": case_index,
        "cohort_case_index": int(selected["cohort_case_index"]),
        "case_id": str(selected["case_id"]),
        "trajectory_id": str(response.get("trajectory_id", f"{outcome}:{response['record_index']}")),
        "record_index": int(response["record_index"]),
        "source_outcome": outcome,
        "outcome_response_rank": int(response["outcome_response_rank"]),
        "question": str(response["question"]),
        "answer": str(response["answer"]),
        "source_response": source_text,
        "source_extracted_answer": str(response["source_extracted_answer"]),
        "source_correct": int(response["source_correct"]),
        "source_format_valid": int(response["source_format_valid"]),
        "source_finish_reason": str(response["source_finish_reason"]),
        "source_response_text_sha256": str(response["source_response_text_sha256"]),
        "prompt_token_ids": prompt,
        "source_response_token_ids": response_ids,
        "pre_step_response_token_ids": pre,
        "student_step_token_ids": step_ids,
        "keep_response_prefix_token_ids": keep,
        "pre_step_input_token_ids": prompt + pre,
        "keep_input_token_ids": prompt + keep,
        "prompt_num_tokens": len(prompt),
        "source_response_num_tokens": len(response_ids),
        "pre_step_response_num_tokens": len(pre),
        "student_step_num_tokens": len(step_ids),
        "keep_response_prefix_num_tokens": len(keep),
        "prompt_token_ids_sha256": token_hash(prompt),
        "source_response_token_ids_sha256": token_hash(response_ids),
        "pre_step_response_token_ids_sha256": token_hash(pre),
        "student_step_token_ids_sha256": token_hash(step_ids),
        "keep_response_prefix_token_ids_sha256": token_hash(keep),
        "pre_step_input_token_ids_sha256": token_hash(prompt + pre),
        "keep_input_token_ids_sha256": token_hash(prompt + keep),
        "step_index": int(canonical["step_index"]),
        "reasoning_step_index": int(canonical["reasoning_step_index"]),
        "step_offset": offset,
        "step_char_start": char_start,
        "step_char_end": char_end,
        "step_token_start": start,
        "step_token_end": end,
        "step_text": step_text,
        "step_text_sha256": text_hash(step_text),
        "step_token_ids_sha256": token_hash(step_ids),
        "response_position_bin": str(selected["response_position_bin"]),
        "response_position_bin_index": int(selected["response_position_bin_index"]),
        "response_position_fraction": float(selected["response_position_fraction"]),
        "response_position_third": str(selected["response_position_third"]),
        "max_response_tokens": MAX_RESPONSE_TOKENS,
        "teacher_replacement_max_new_tokens": TEACHER_PROPOSAL_MAX_NEW_TOKENS,
        "prepared_dataset_max_response_tokens": dataset_max_response,
        "prepared_dataset_teacher_proposal_max_new_tokens": dataset_proposal_cap,
        "keep_continuation_max_new_tokens": MAX_RESPONSE_TOKENS - len(keep),
        "delete_continuation_max_new_tokens": MAX_RESPONSE_TOKENS - len(pre),
        "replace_continuation_max_new_tokens_if_full_replacement": (
            MAX_RESPONSE_TOKENS - len(pre) - TEACHER_PROPOSAL_MAX_NEW_TOKENS
        ),
        "prepared_dataset_record_sha256": canonical_sha(response),
    }


def load_dataset(path: Path, outcome: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if outcome not in OUTCOMES:
        raise ValueError(f"outcome must be one of {sorted(OUTCOMES)}")
    payload = load_json(path)
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError(f"invalid nested dataset: {path}")
    metadata = payload["metadata"]
    responses = payload.get("responses")
    if not isinstance(responses, list) or not responses:
        raise ValueError(f"dataset has no responses: {path}")
    if str(metadata.get("source_outcome")) != outcome:
        raise ValueError(f"dataset outcome mismatch: expected {outcome}")
    if int(metadata.get("samples_per_arm", SAMPLES_PER_ARM)) != SAMPLES_PER_ARM:
        raise ValueError("dataset samples_per_arm does not equal 128")
    if int(metadata.get("steps_per_response", 5)) != 5:
        raise ValueError("dataset must contain five selected steps per response")
    if str(metadata.get("prompt_name", "qwen3_no_thinking_prompt")) != "qwen3_no_thinking_prompt":
        raise ValueError("dataset is not bound to the no-thinking prompt")
    if not math.isclose(float(metadata.get("temperature", TEMPERATURE)), TEMPERATURE):
        raise ValueError("dataset temperature does not match 0.6")
    if not math.isclose(float(metadata.get("top_p", TOP_P)), TOP_P) or int(metadata.get("top_k", TOP_K)) != TOP_K:
        raise ValueError("dataset sampling parameters do not match top_p=0.95/top_k=-1")
    cases: list[dict[str, Any]] = []
    for response in responses:
        if not isinstance(response, dict):
            raise ValueError("response row is not an object")
        # Parse and hash the canonical arrays once per trajectory.  The five
        # selected cases intentionally share these immutable lists in memory;
        # only their exact prefix/step slices are newly allocated.
        prompt_ids = ints(response["prompt_token_ids"], "prompt_token_ids")
        response_ids = ints(response["source_response_token_ids"], "source_response_token_ids")
        selected = response.get("selected_steps")
        if not isinstance(selected, list) or len(selected) != 5:
            raise ValueError(f"response {response.get('record_index')} does not have five steps")
        if {str(item.get("response_position_bin")) for item in selected} != set(POSITION_BINS):
            raise ValueError(f"response {response.get('record_index')} does not cover p1--p5")
        for item in selected:
            cases.append(_case_from_selected(response, item, metadata,
                                             prompt_ids=prompt_ids,
                                             response_ids=response_ids))
    cases.sort(key=lambda row: int(row["case_index"]))
    # The preparation script deliberately reserves disjoint index ranges for
    # the two cohorts (correct: 0--9999, incorrect: 10000--19999).  Keep the
    # original values so paired seeds remain stable when the cohorts are run
    # in separate jobs.
    first_case_index = min(int(c["case_index"]) for c in cases)
    expected = list(range(first_case_index, first_case_index + len(cases)))
    observed = [int(row["case_index"]) for row in cases]
    if observed != expected:
        raise ValueError("global_case_index must be consecutive from zero within each outcome")
    if len({str(row["case_id"]) for row in cases}) != len(cases):
        raise ValueError("duplicate selected case_id")
    if len({int(row["record_index"]) for row in cases}) != len(responses):
        raise ValueError("response rows are not unique")
    expected_count = int(metadata.get("selected_step_count", len(cases)))
    if expected_count != len(cases):
        raise ValueError("metadata selected_step_count does not match rows")
    return metadata, cases


def chunks(values: Sequence[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    for start in range(0, len(values), size):
        yield list(values[start : start + size])


def shard_plan(cases: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return list(chunks(cases, SHARD_CASES))


def worker_shards(total: int, worker_id: int, world_size: int) -> list[int]:
    return [i for i in range(total) if i % world_size == worker_id]


def model_id(model: Path) -> dict[str, Any]:
    """Small reproducibility fingerprint (does not hash multi-GB weights)."""
    files = {
        name: (sha256_file(model / name) if (model / name).is_file() else None)
        for name in (
            "config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.json",
            "merges.txt",
            "generation_config.json",
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
        )
    }
    weight_files = []
    if model.is_dir():
        for path in sorted(model.iterdir(), key=lambda item: item.name):
            if path.is_file() and path.suffix.lower() in {".safetensors", ".bin", ".pt", ".pth"}:
                stat = path.stat()
                weight_files.append({
                    "name": path.name,
                    "size": int(stat.st_size),
                    "mtime_ns": int(stat.st_mtime_ns),
                })
    return {
        "basename": model.name,
        "resolved_path": str(model.resolve()),
        **{f"{name.replace('.', '_')}_sha256": digest for name, digest in files.items()},
        # File names/sizes/timestamps catch an in-place checkpoint swap while
        # avoiding a multi-gigabyte weight hash at every worker startup.
        "weight_files": weight_files,
    }


def run_signature(dataset_sha: str, outcome: str, student: Path,
                  teacher: Path | None = None, world_size: int = 8, *,
                  max_model_len: int = 0,
                  teacher_key: str | None = None,
                  teacher_models: dict[str, Path] | None = None) -> str:
    student_info = model_id(student)
    # ``teacher`` is retained as the positional single-Teacher API.  A
    # multi-Teacher signature binds the complete ordered key -> model mapping
    # so shards from one Teacher set can never be resumed with another set.
    if teacher_models is not None:
        if not isinstance(teacher_models, dict) or not teacher_models:
            raise ValueError("teacher_models must be a non-empty dict")
        normalized_teachers = {
            _validate_teacher_key(key): model_id(Path(path))
            for key, path in teacher_models.items()
        }
        if len(normalized_teachers) != len(teacher_models):
            raise ValueError("teacher_models contains duplicate/invalid keys")
        teacher_info: Any = normalized_teachers
    else:
        if teacher is None:
            raise ValueError("run_signature requires teacher or teacher_models")
        teacher_info = model_id(Path(teacher))
    return canonical_sha(
        {
            "experiment": EXPERIMENT,
            "dataset_sha256": dataset_sha,
            "outcome": outcome,
            # Include both the resolved path and lightweight model-file
            # fingerprints. A same-named model copied from another checkpoint
            # must never silently reuse old continuation shards.
            "student_model": student_info,
            "teacher_model": teacher_info,
            **({"teacher_key": _validate_teacher_key(teacher_key)}
               if teacher_key is not None else {}),
            "samples_per_arm": SAMPLES_PER_ARM,
            "prompts_per_generate": PROMPTS_PER_GENERATE,
            "shard_cases": SHARD_CASES,
            "max_response_tokens": MAX_RESPONSE_TOKENS,
            "proposal_max_new_tokens": TEACHER_PROPOSAL_MAX_NEW_TOKENS,
            "max_model_len": int(max_model_len),
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "seed": SEED,
            "runner_sha256": sha256_file(Path(__file__).resolve()),
            "world_size": int(world_size),
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("worker", "monitor"), default="worker")
    parser.add_argument("--outcome", choices=sorted(OUTCOMES), required=True)
    parser.add_argument("--dataset", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--student-model", type=Path, default=DEFAULT_STUDENT_MODEL)
    parser.add_argument(
        "--teacher-model",
        type=Path,
        default=None,
        help=(
            "Legacy single-Teacher override. Normally edit TEACHER_MODELS in this "
            "Python file and omit this option."
        ),
    )
    parser.add_argument(
        "--teacher-key",
        action="append",
        default=None,
        help=(
            "Select one or more keys from TEACHER_MODELS; with legacy "
            "--teacher-model, at most one key is allowed. Repeat the option "
            "to select multiple configured Teachers."
        ),
    )
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--phase", choices=("all", "proposals", "continuations"), default="all")
    parser.add_argument("--arm", choices=("all", *ARMS), default="all")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--heartbeat-seconds", type=float, default=30.0)
    parser.add_argument("--memory-check-seconds", type=float, default=2.0)
    parser.add_argument("--min-free-gpu-gib", type=float, default=MIN_FREE_GPU_GIB)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument(
        "--run-id",
        default="",
        help="Invocation identifier used to ignore stale progress files (the shell sets it automatically).",
    )
    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
    teacher_models: dict[str, Path] | None = None,
) -> None:
    if args.world_size < 1 or not 0 <= args.worker_id < args.world_size:
        raise ValueError("worker-id must lie in [0, world-size)")
    if args.gpu_id < 0:
        raise ValueError("gpu-id must be nonnegative")
    if args.min_free_gpu_gib < 0.0:
        raise ValueError("min-free-gpu-gib must be nonnegative")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise ValueError("gpu-memory-utilization must lie in (0,1)")
    if args.heartbeat_seconds < args.memory_check_seconds or args.memory_check_seconds <= 0:
        raise ValueError("invalid heartbeat interval")
    if args.mode == "monitor" and args.phase != "all":
        # Monitor observes the complete shell run; a phase-specific monitor is
        # still useful, so this is intentionally permissive below.
        pass
    if not args.student_model.is_dir() and not args.validate_only:
        raise FileNotFoundError(args.student_model)
    # ``resolve_teacher_models`` validates the mapping and gives the user one
    # clear missing-directory error per configured Teacher.  ``main`` passes
    # its already-resolved mapping so relative values are checked against the
    # repository root consistently with the paths used by the workers.
    models_to_check = (
        teacher_models
        if teacher_models is not None
        else resolve_teacher_model_paths(resolve_teacher_models(args))
    )
    for key, model in models_to_check.items():
        if not model.is_dir() and not args.validate_only:
            raise FileNotFoundError(f"Teacher {key!r}: {model}")


def _set_cuda_visibility(gpu_id: int) -> None:
    # Set before importing vLLM.  nvidia-smi uses the physical ID passed to
    # GPUHeartbeat, while CUDA sees only this one device.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # A pre-existing ``fork`` setting is unsafe after CUDA initialization;
    # each worker is deliberately isolated, so force the safe start method.
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


def _load_tokenizer(model: Path) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=True)


def derive_max_model_len(cases: list[dict[str, Any]], override: int = 0) -> int:
    maximum = max(len(row["prompt_token_ids"]) + MAX_RESPONSE_TOKENS for row in cases)
    derived = int(math.ceil(maximum / 256.0) * 256)
    selected = int(override or derived)
    if selected < maximum:
        raise ValueError(f"max_model_len={selected} is below required context {maximum}")
    return selected


def _new_llm(model: Path, args: argparse.Namespace, max_model_len: int) -> Any:
    from vllm import LLM

    # The utilization is capped by the explicit physical reserve.  The guard
    # below remains active during generation in case other processes consume
    # memory after initialization.
    memory = base.query_gpu_memory_mib(args.gpu_id)
    required_free = int(math.ceil(args.min_free_gpu_gib * base.MIB_PER_GIB))
    reserve = required_free + GPU_SAFETY_MARGIN_MIB
    if memory["free"] < reserve:
        raise RuntimeError(
            f"GPU {args.gpu_id} has only {memory['free']} MiB free before "
            f"initialization; need at least {reserve} MiB to preserve the "
            f"{args.min_free_gpu_gib:g} GiB reserve plus safety margin."
        )
    # Base the cap on *currently free* memory, not just card capacity.  This
    # prevents a second process or an unrelated CUDA allocation from consuming
    # the explicit reserve before vLLM starts its allocator.
    available_fraction = max(0.05, (memory["free"] - reserve) / max(1, memory["total"]))
    effective = min(float(args.gpu_memory_utilization), available_fraction)
    print(json.dumps({"event": "gpu_budget", "gpu": args.gpu_id, "memory_mib": memory,
                      "requested_utilization": args.gpu_memory_utilization,
                      "effective_utilization": round(effective, 5),
                       "required_free_mib": required_free},
                      ensure_ascii=False), flush=True)
    return LLM(
        model=str(model),
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=max_model_len,
        max_num_seqs=MAX_NUM_SEQS,
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        gpu_memory_utilization=effective,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        enforce_eager=True,
        seed=SEED,
    )


def _sampling(n: int, max_tokens: int, seed: int) -> Any:
    from vllm import SamplingParams

    return SamplingParams(n=n, temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
                          max_tokens=max_tokens, seed=int(seed))


def _check_headroom(args: argparse.Namespace, stage: str) -> dict[str, int]:
    minimum = int(math.ceil(args.min_free_gpu_gib * base.MIB_PER_GIB))
    return base.assert_gpu_headroom(args.gpu_id, minimum, stage=stage)


def _generate(llm: Any, prompts: list[list[int]], params: list[Any], args: argparse.Namespace,
              stage: str) -> list[Any]:
    minimum = int(math.ceil(args.min_free_gpu_gib * base.MIB_PER_GIB))
    request = [{"prompt_token_ids": ids} for ids in prompts]
    try:
        with base.GPUHeartbeat(gpu_id=args.gpu_id, minimum_free_mib=minimum,
                               check_interval=args.memory_check_seconds,
                               report_interval=args.heartbeat_seconds, stage=stage):
            outputs = llm.generate(request, sampling_params=params, use_tqdm=False)
    except Exception as exc:
        error_text = repr(exc).lower()
        if ("out of memory" in error_text or "outofmemory" in error_text
                or "cuda oom" in error_text
                or "reserve violated" in error_text
                or "gpu reserve" in error_text):
            # A smaller retry is not a safe response to a physical memory
            # violation: it can race the heartbeat and obscure the failure.
            raise
        # A pair of very long prompts can occasionally exceed a scheduler
        # token limit even when a single prompt is safe.  Retry that microbatch
        # as two requests; seeds and prefixes stay unchanged, so the retry is
        # scientifically paired and resumable.
        if len(prompts) <= 1:
            raise
        print(json.dumps({"event": "microbatch_fallback", "stage": stage,
                          "prompt_cases": len(prompts), "error": repr(exc)},
                         ensure_ascii=False), flush=True)
        outputs = []
        for index, (one_prompt, one_param) in enumerate(zip(prompts, params)):
            with base.GPUHeartbeat(gpu_id=args.gpu_id, minimum_free_mib=minimum,
                                   check_interval=args.memory_check_seconds,
                                   report_interval=args.heartbeat_seconds,
                                   stage=f"{stage}/fallback_{index}"):
                one = llm.generate([{"prompt_token_ids": one_prompt}],
                                   sampling_params=[one_param], use_tqdm=False)
            if len(one) != 1:
                raise RuntimeError(f"{stage}/fallback returned {len(one)} requests")
            outputs.extend(one)
    if len(outputs) != len(prompts):
        raise RuntimeError(f"{stage}: vLLM returned {len(outputs)} requests for {len(prompts)} prompts")
    return list(outputs)


def _proposal_record(case: dict[str, Any], output: Any, tokenizer: Any,
                     request_seed: int, attempt: int, model: Path,
                     teacher_key: str | None = None) -> dict[str, Any]:
    from utils.oa_opd import map_response_steps_to_original_tokens

    input_ids = list(case["pre_step_input_token_ids"])
    if [int(v) for v in output.prompt_token_ids] != input_ids:
        raise RuntimeError("vLLM changed exact Teacher proposal input IDs")
    if len(output.outputs) != 1:
        raise RuntimeError("Teacher proposal did not return n=1")
    completion = output.outputs[0]
    proposal_ids = [int(v) for v in completion.token_ids]
    if not proposal_ids:
        raise RuntimeError("Teacher returned an empty proposal")
    visible, mapped = map_response_steps_to_original_tokens(tokenizer, proposal_ids)
    if not mapped:
        raise RuntimeError("Teacher proposal contains no extractable semantic step")
    first = mapped[0]
    replacement_ids = proposal_ids[: int(first.token_end)]
    if not replacement_ids:
        raise RuntimeError("Teacher semantic step is empty")
    prefix_len = len(case["pre_step_response_token_ids"]) + len(replacement_ids)
    if prefix_len >= MAX_RESPONSE_TOKENS:
        raise RuntimeError("Teacher proposal leaves no Student continuation budget")
    finish = base.normalized_reason(completion.finish_reason)
    model = Path(model)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "phase": "proposals",
        "case_id": case["case_id"],
        "case_index": int(case["case_index"]),
        "record_index": int(case["record_index"]),
        "source_outcome": case["source_outcome"],
        "answer": case["answer"],
        "question": case["question"],
        "source_response_text_sha256": case["source_response_text_sha256"],
        "teacher_model": str(model),
        "teacher_model_basename": model.name,
        **({"teacher_key": _validate_teacher_key(teacher_key)}
           if teacher_key is not None else {}),
        "request_seed": int(request_seed),
        "accepted_attempt": int(attempt),
        "input_token_ids": input_ids,
        "input_num_tokens": len(input_ids),
        "input_token_ids_sha256": token_hash(input_ids),
        "proposal_max_new_tokens": TEACHER_PROPOSAL_MAX_NEW_TOKENS,
        "proposal_token_ids": proposal_ids,
        "proposal_num_tokens": len(proposal_ids),
        "proposal_token_ids_sha256": token_hash(proposal_ids),
        "proposal_text": base.decode_ids(tokenizer, proposal_ids, skip_special_tokens=False),
        "proposal_visible_text": visible,
        "proposal_finish_reason": finish,
        "proposal_stop_reason": base.normalized_reason(completion.stop_reason),
        "proposal_was_length_truncated": finish == "length",
        "mapped_semantic_steps": len(mapped),
        "replacement_step_token_ids": replacement_ids,
        "replacement_step_num_tokens": len(replacement_ids),
        "replacement_step_token_ids_sha256": token_hash(replacement_ids),
        "replacement_step_text": str(first.text),
        "replacement_step_decoded_text": base.decode_ids(tokenizer, replacement_ids, skip_special_tokens=False),
        "replacement_step_token_end_in_proposal": int(first.token_end),
        "replace_response_prefix_num_tokens": prefix_len,
        "replace_continuation_max_new_tokens": MAX_RESPONSE_TOKENS - prefix_len,
        "saved_at": utc_now(),
    }


def _completion_item(completion: Any, tokenizer: Any, prefix: list[int], answer: str,
                     sample_index: int) -> dict[str, Any]:
    full_item = base.completion_result(
        completion=completion,
        tokenizer=tokenizer,
        response_prefix_ids=prefix,
        max_response_tokens=MAX_RESPONSE_TOKENS,
        ground_truth=answer,
        sample_index=sample_index,
    )
    # ``full_response_text`` and the exact continuation IDs are the two useful
    # payloads for later auditing. ``continuation_text`` and
    # ``vllm_completion_text`` are byte-for-byte redundant in this pipeline and
    # would multiply storage by several gigabytes at 7.68M samples.
    item = {key: full_item[key] for key in COMPLETION_FIELDS_TO_SAVE}
    item["reward"] = int(item["correct"])
    return item


def _continuation_record(case: dict[str, Any], arm: str, proposal: dict[str, Any] | None,
                         output: Any, tokenizer: Any, model: Path, request_seed: int,
                         student_tokenizer: Any | None = None,
                         teacher_key: str | None = None) -> dict[str, Any]:
    if arm == "keep":
        prefix = list(case["keep_response_prefix_token_ids"])
    elif arm == "delete":
        prefix = list(case["pre_step_response_token_ids"])
    elif arm == "replace" and proposal is not None:
        prefix = list(case["pre_step_response_token_ids"]) + [int(v) for v in proposal["replacement_step_token_ids"]]
    else:
        raise ValueError("replace requires a valid Teacher proposal")
    input_ids = list(case["prompt_token_ids"]) + prefix
    if [int(v) for v in output.prompt_token_ids] != input_ids:
        raise RuntimeError(f"vLLM changed exact {arm} input IDs for {case['case_id']}")
    if len(output.outputs) != SAMPLES_PER_ARM:
        raise RuntimeError(f"{arm} returned {len(output.outputs)} outputs, expected {SAMPLES_PER_ARM}")
    results = []
    for i, completion in enumerate(output.outputs):
        observed_index = int(getattr(completion, "index", i))
        if observed_index != i:
            raise RuntimeError(
                f"{case['case_id']} {arm} returned completion index {observed_index}, expected {i}"
            )
        results.append(_completion_item(completion=completion, tokenizer=tokenizer,
                                        prefix=prefix, answer=str(case["answer"]),
                                        sample_index=i))
    model = Path(model)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "phase": "continuations",
        "arm": arm,
        "case_id": case["case_id"],
        "case_index": int(case["case_index"]),
        "record_index": int(case["record_index"]),
        "source_outcome": case["source_outcome"],
        "answer": case["answer"],
        "question": case["question"],
        "source_response_text_sha256": case["source_response_text_sha256"],
        "student_model": str(model),
        "student_model_basename": model.name,
        **({"teacher_key": _validate_teacher_key(teacher_key)}
           if teacher_key is not None else {}),
        "request_seed": int(request_seed),
        "samples_per_arm": SAMPLES_PER_ARM,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "response_prefix_token_ids": prefix,
        "response_prefix_num_tokens": len(prefix),
        "response_prefix_token_ids_sha256": token_hash(prefix),
        "input_token_ids_sha256": token_hash(input_ids),
        "max_new_tokens": MAX_RESPONSE_TOKENS - len(prefix),
        "student_step_token_ids_sha256": case["student_step_token_ids_sha256"],
        "step_offset": int(case["step_offset"]),
        "response_position_fraction": float(case["response_position_fraction"]),
        "response_position_bin": case["response_position_bin"],
        "outputs": results,
        "rewards": [int(item["reward"]) for item in results],
        "reward_sum": sum(int(item["reward"]) for item in results),
        "accuracy_avg128": sum(int(item["reward"]) for item in results) / SAMPLES_PER_ARM,
        "value_name": {"keep": "V_keep", "delete": "V_delete", "replace": "V_replace"}[arm],
        # ``V_replace`` is additionally keyed by the Teacher in the enclosing
        # directory/manifest.  Keep the historical value name for downstream
        # readers, and expose an unambiguous label for multi-Teacher analysis.
        "value_name_with_teacher": (
            f"V_replace[{_validate_teacher_key(teacher_key)}]"
            if arm == "replace" and teacher_key is not None else
            {"keep": "V_keep", "delete": "V_delete", "replace": "V_replace"}[arm]
        ),
        "truncated_by_response_cap_count": sum(bool(item["truncated_by_response_cap"]) for item in results),
        **({
            "teacher_proposal_case_index": int(proposal["case_index"]),
            "replacement_step_token_ids": list(proposal["replacement_step_token_ids"]),
            "replacement_step_token_ids_sha256": proposal["replacement_step_token_ids_sha256"],
            "replacement_step_decoded_text": proposal["replacement_step_decoded_text"],
            "replacement_teacher_model": proposal["teacher_model"],
            **({"replacement_teacher_key": proposal["teacher_key"]}
               if proposal.get("teacher_key") is not None else {}),
            "replacement_request_seed": int(proposal["request_seed"]),
        } if proposal is not None else {}),
        "saved_at": utc_now(),
    }


def _proposal_paths(root: Path, worker_id: int, shard_index: int,
                    teacher_key: str | None = None) -> Path:
    """Return a proposal shard path, namespaced for multi-Teacher runs.

    With no key this preserves the original single-Teacher layout, which is
    important for resuming existing collections and for downstream scripts.
    """

    directory = root / f"worker_{worker_id:02d}" / "teacher_proposals"
    if teacher_key is not None:
        directory = directory / _validate_teacher_key(teacher_key)
    return directory / f"part_{shard_index:05d}.json"


def _arm_paths(root: Path, worker_id: int, arm: str, shard_index: int,
               teacher_key: str | None = None) -> Path:
    """Return a continuation shard path.

    Keep/Delete are shared across Teachers.  Replace is namespaced when a
    key is supplied; omitting it retains the historical single-Teacher path.
    """

    directory = root / f"worker_{worker_id:02d}" / "student" / arm
    if arm == "replace" and teacher_key is not None:
        directory = directory / _validate_teacher_key(teacher_key)
    return directory / f"part_{shard_index:05d}.json.gz"


def clear_worker_shards(root: Path, worker_id: int, shard_ids: Iterable[int], *,
                        clear_proposals: bool, clear_arms: Iterable[str],
                        teacher_key: str | None = None) -> None:
    """Remove only this worker's exact shard files for an explicit overwrite.

    Without this cleanup, an interrupted overwrite could leave an old valid
    shard that a later ordinary resume would incorrectly reuse alongside newly
    generated shards.  No directory or unrelated file is removed.
    """

    for shard_id in shard_ids:
        if clear_proposals:
            _proposal_paths(root, worker_id, shard_id, teacher_key).unlink(missing_ok=True)
        for arm in clear_arms:
            _arm_paths(root, worker_id, arm, shard_id, teacher_key).unlink(missing_ok=True)


def _proposal_map_for_worker(
    root: Path,
    worker_id: int,
    shard_ids: list[int],
    batches: list[list[dict[str, Any]]],
    signature: str | None = None,
    teacher_key: str | None = None,
    teacher_model: Path | str | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    proposals: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    for shard_id in shard_ids:
        path = _proposal_paths(root, worker_id, shard_id, teacher_key)
        if not path.is_file():
            continue
        # Never hydrate a partially written or stale shard into the resume
        # map.  Atomic writes make a complete file visible at once, but a
        # process can still be interrupted between attempts; accepting those
        # records would let an old proposal leak into a new run.
        if not _valid_proposal_shard(
            path,
            batches[shard_id],
            signature,
            teacher_key=teacher_key,
            teacher_model=teacher_model,
        ):
            continue
        payload = load_json(path)
        for row in payload.get("records", []):
            proposals[str(row["case_id"])] = row
        for row in payload.get("failures", []):
            failures[str(row["case_id"])] = str(row.get("error", "proposal_failed"))
    return proposals, failures


def _valid_proposal_shard(
    path: Path,
    expected: list[dict[str, Any]],
    signature: str | None,
    *,
    teacher_key: str | None = None,
    teacher_model: Path | str | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = load_json(path)
        records = payload.get("records", [])
        failures = payload.get("failures", [])
        if not isinstance(records, list) or not isinstance(failures, list):
            return False
        if (signature is not None and payload.get("config_signature") != signature
                or payload.get("phase") != "proposals"
                or int(payload.get("target_cases", -1)) != len(expected)
                or int(payload.get("generated_outputs", -1)) != len(records)):
            return False
        expected_teacher_key = (
            _validate_teacher_key(teacher_key) if teacher_key is not None else None
        )
        # Keep keyed and legacy proposal namespaces strictly separate.  The
        # run signature is an additional guard, but checking the field here
        # prevents a copied shard from being silently reused in the wrong
        # layout.
        if expected_teacher_key is None:
            if "teacher_key" in payload:
                return False
        elif payload.get("teacher_key") != expected_teacher_key:
            return False
        if teacher_model is not None and str(payload.get("teacher_model")) != str(Path(teacher_model)):
            return False
        expected_by_id = {str(c["case_id"]): c for c in expected}
        if len(expected_by_id) != len(expected):
            return False
        seen: set[str] = set()
        for row in records:
            if not isinstance(row, dict):
                return False
            case_id = str(row.get("case_id"))
            reference = expected_by_id.get(case_id)
            if reference is None or case_id in seen:
                return False
            if expected_teacher_key is None:
                if "teacher_key" in row:
                    return False
            elif row.get("teacher_key") != expected_teacher_key:
                return False
            if teacher_model is not None and str(row.get("teacher_model")) != str(Path(teacher_model)):
                return False
            if (int(row.get("case_index", -1)) != int(reference["case_index"])
                    or int(row.get("record_index", -1)) != int(reference["record_index"])):
                return False
            token_ids = row.get("replacement_step_token_ids")
            if (not isinstance(token_ids, list) or not token_ids
                    or any(type(value) is not int for value in token_ids)):
                return False
            if row.get("input_token_ids_sha256") != reference.get("pre_step_input_token_ids_sha256"):
                return False
            proposal_ids = row.get("proposal_token_ids")
            accepted_attempt = int(row.get("accepted_attempt", 0))
            if (not isinstance(proposal_ids, list) or not proposal_ids
                    or any(type(value) is not int for value in proposal_ids)
                    or row.get("proposal_token_ids_sha256") != token_hash(proposal_ids)
                    or int(row.get("proposal_num_tokens", -1)) != len(proposal_ids)
                    or int(row.get("replacement_step_num_tokens", -1)) != len(token_ids)
                    or row.get("replacement_step_token_ids_sha256") != token_hash(token_ids)
                    or proposal_ids[:len(token_ids)] != token_ids
                    or accepted_attempt < 1
                    or accepted_attempt > TEACHER_PROPOSAL_ATTEMPTS
                    or int(row.get("request_seed", -1))
                    != proposal_seed(int(reference["case_index"]), accepted_attempt)
                    or int(row.get("proposal_max_new_tokens", -1))
                    != TEACHER_PROPOSAL_MAX_NEW_TOKENS):
                return False
            seen.add(case_id)
        failed: set[str] = set()
        for row in failures:
            if not isinstance(row, dict):
                return False
            case_id = str(row.get("case_id"))
            reference = expected_by_id.get(case_id)
            if reference is None or case_id in seen or case_id in failed:
                return False
            if (int(row.get("case_index", -1)) != int(reference["case_index"])
                    or int(row.get("record_index", -1)) != int(reference["record_index"])):
                return False
            failed.add(case_id)
        return seen | failed == set(expected_by_id) and not (seen & failed)
    except Exception:
        return False


def _valid_arm_shard(
    path: Path,
    expected: list[dict[str, Any]],
    signature: str,
    arm: str,
    proposals: dict[str, dict[str, Any]] | None = None,
    *,
    teacher_key: str | None = None,
    proposal_failures: dict[str, str] | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        payload = load_json_maybe_gzip(path)
        rows = payload.get("records", [])
        if (payload.get("config_signature") != signature
                or payload.get("phase") != "continuations"
                or payload.get("arm") != arm
                or int(payload.get("target_cases", -1)) != len(expected)):
            return False
        expected_teacher_key = (
            _validate_teacher_key(teacher_key) if teacher_key is not None else None
        )
        if expected_teacher_key is None:
            if "teacher_key" in payload:
                return False
        elif payload.get("teacher_key") != expected_teacher_key:
            return False
        if arm != "replace" and expected_teacher_key is not None:
            return False
        if not isinstance(rows, list):
            return False
        # Replace shards may intentionally contain only proposal-success cases.
        expected_by_id = {str(c["case_id"]): c for c in expected}
        if len(expected_by_id) != len(expected):
            return False
        expected_ids = set(expected_by_id)
        skipped_rows = payload.get("skipped", [])
        if not isinstance(skipped_rows, list):
            return False
        observed: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                return False
            case_id = str(row.get("case_id"))
            reference = expected_by_id.get(case_id)
            if reference is None or case_id in observed:
                return False
            if (int(row.get("case_index", -1)) != int(reference["case_index"])
                    or int(row.get("record_index", -1)) != int(reference["record_index"])):
                return False
            if (row.get("source_response_text_sha256")
                    != reference.get("source_response_text_sha256")
                    or row.get("student_step_token_ids_sha256")
                    != reference.get("student_step_token_ids_sha256")):
                return False
            prefix = row.get("response_prefix_token_ids")
            if (not isinstance(prefix, list)
                    or any(type(value) is not int for value in prefix)
                    or int(row.get("response_prefix_num_tokens", -1)) != len(prefix)
                    or row.get("response_prefix_token_ids_sha256") != token_hash(prefix)
                    or row.get("input_token_ids_sha256")
                    != token_hash(list(reference["prompt_token_ids"]) + prefix)
                    or int(row.get("max_new_tokens", 0)) <= 0
                    or len(prefix) + int(row.get("max_new_tokens", 0)) > MAX_RESPONSE_TOKENS):
                return False
            if arm == "keep" and prefix != list(reference["keep_response_prefix_token_ids"]):
                return False
            if arm == "delete" and prefix != list(reference["pre_step_response_token_ids"]):
                return False
            if arm == "replace":
                pre = list(reference["pre_step_response_token_ids"])
                if len(prefix) <= len(pre) or prefix[:len(pre)] != pre:
                    return False
                # In production a Replace shard is valid only when its case
                # has a successful proposal for this exact Teacher.  Keep the
                # ``proposals=None`` form for legacy callers that only want a
                # structural check and have no proposal map available.
                if proposals is not None and case_id not in proposals:
                    return False
                if proposals is not None and case_id in proposals:
                    replacement_ids = proposals[case_id].get("replacement_step_token_ids")
                    if (not isinstance(replacement_ids, list)
                            or prefix[len(pre):] != [int(value) for value in replacement_ids]):
                        return False
                    # Bind a continuation shard to the exact proposal source;
                    # this prevents a Replace result from another Teacher
                    # being silently associated with the current key.
                    proposal_model = proposals[case_id].get("teacher_model")
                    if (proposal_model is not None
                            and row.get("replacement_teacher_model") != proposal_model):
                        return False
                    if (expected_teacher_key is not None
                        and proposals[case_id].get("teacher_key") != expected_teacher_key):
                        return False
            expected_budget = MAX_RESPONSE_TOKENS - len(prefix)
            if (int(row.get("max_new_tokens", -1)) != expected_budget
                    or int(row.get("samples_per_arm", -1)) != SAMPLES_PER_ARM
                    or not math.isclose(float(row.get("temperature", -1.0)), TEMPERATURE)
                    or not math.isclose(float(row.get("top_p", -1.0)), TOP_P)
                    or int(row.get("top_k", 0)) != TOP_K
                    or int(row.get("request_seed", -1))
                    != continuation_seed(int(reference["case_index"]))):
                return False
            outputs = row.get("outputs")
            if (not isinstance(outputs, list)
                    or len(outputs) != SAMPLES_PER_ARM
                    or any(not isinstance(item.get("continuation_token_ids"), list)
                           or any(type(value) is not int
                                  for value in item.get("continuation_token_ids", []))
                           or int(item.get("continuation_num_tokens", -1))
                           != len(item.get("continuation_token_ids", []))
                           or len(item.get("continuation_token_ids", [])) > expected_budget
                           or int(item.get("correct", -1)) not in (0, 1)
                           or int(item.get("reward", -1)) != int(item.get("correct", -1))
                           for item in outputs)):
                return False
            observed.add(case_id)
        skipped: set[str] = set()
        for row in skipped_rows:
            if not isinstance(row, dict):
                return False
            case_id = str(row.get("case_id"))
            reference = expected_by_id.get(case_id)
            if reference is None or case_id in skipped:
                return False
            if int(row.get("case_index", -1)) != int(reference["case_index"]):
                return False
            if arm == "replace" and proposals is not None:
                # A skipped case must be explained by this Teacher's
                # proposal failure.  Otherwise an old skip could be paired
                # with a newly generated proposal map.
                # A tokenizer compatibility check can turn a previously
                # successful proposal into a failure in memory without
                # rewriting the proposal shard.  In that narrow case the
                # maps overlap, and the explicit failure entry is sufficient
                # evidence for the skip (including validate-only runs).
                if proposal_failures is not None:
                    if case_id not in proposal_failures:
                        # ``--validate-only`` deliberately does not load the
                        # Student tokenizer.  Preserve the one skip reason
                        # that can therefore be absent from the in-memory
                        # failure map: a proposal whose Teacher IDs failed
                        # the decode-equivalence check during generation.
                        if not (
                            case_id in proposals
                            and str(row.get("reason", ""))
                            == "student_teacher_tokenizer_decode_mismatch"
                        ):
                            return False
                elif case_id in proposals:
                    return False
            skipped.add(case_id)
        if observed | skipped != expected_ids or (observed & skipped):
            return False
        for row in rows:
            if row.get("arm") != arm or len(row.get("outputs", [])) != SAMPLES_PER_ARM:
                return False
            if expected_teacher_key is None:
                if "teacher_key" in row:
                    return False
            elif row.get("teacher_key") != expected_teacher_key:
                return False
            if not isinstance(row.get("case_index"), int):
                return False
            outputs = row.get("outputs")
            if (not isinstance(outputs, list)
                    or any(not isinstance(item, dict) for item in outputs)
                    or [int(item.get("sample_index", -1)) for item in outputs]
                    != list(range(SAMPLES_PER_ARM))):
                return False
            rewards = row.get("rewards")
            if (not isinstance(rewards, list) or len(rewards) != SAMPLES_PER_ARM
                    or any(int(value) not in (0, 1) for value in rewards)):
                return False
            if int(row.get("reward_sum", -1)) != sum(int(value) for value in rewards):
                return False
        observed_indices = {int(row["case_index"]) for row in rows}
        skipped_indices = {
            int(row["case_index"])
            for row in skipped_rows
            if isinstance(row.get("case_index"), int)
        }
        expected_indices = {int(c["case_index"]) for c in expected}
        if observed_indices | skipped_indices != expected_indices:
            return False
        if int(payload.get("generated_outputs", -1)) != len(rows) * SAMPLES_PER_ARM:
            return False
        return True
    except Exception:
        return False


def _progress_path(root: Path, worker_id: int) -> Path:
    return root / "progress" / f"worker_{worker_id:02d}.json"


def write_progress(root: Path, worker_id: int, signature: str, state: dict[str, Any]) -> None:
    payload = {"schema_version": SCHEMA_VERSION, "experiment": EXPERIMENT,
               "worker_id": worker_id, "config_signature": signature,
               "updated_at": utc_now(), **state}
    atomic_json(_progress_path(root, worker_id), payload)


def save_progress(args: argparse.Namespace, root: Path, signature: str,
                  state: dict[str, Any]) -> None:
    """Write progress tagged with this invocation's run ID.

    A resumed run keeps completed shards but receives a fresh run ID, so the
    monitor cannot mistake progress left by an older process for a live worker.
    ``getattr`` keeps the helper convenient for the small unit-test namespaces.
    """

    save_state = dict(state)
    save_state["run_id"] = str(getattr(args, "run_id", ""))
    write_progress(root, args.worker_id, signature, save_state)


def _print_progress(outcome: str, worker_id: int, phase: str, arm: str | None,
                    completed: int, total: int, generated: int, started: float) -> None:
    elapsed = max(1e-6, time.monotonic() - started)
    rate = completed / elapsed
    eta = (total - completed) / rate if rate > 0 else None
    print(json.dumps({"event": "progress", "outcome": outcome, "worker": worker_id,
                      "phase": phase, "arm": arm, "completed_cases": completed,
                      "total_cases": total, "generated_outputs": generated,
                      "percent": round(100 * completed / total, 3) if total else 100.0,
                      "cases_per_second": round(rate, 4),
                      "eta_seconds": round(eta, 1) if eta is not None else None},
                     ensure_ascii=False), flush=True)


def _write_run_manifest(root: Path, args: argparse.Namespace, dataset: Path,
                        metadata: dict[str, Any], cases: list[dict[str, Any]], signature: str,
                        teacher_models: dict[str, Path] | None = None) -> None:
    if teacher_models is None:
        # Resolve the same source of truth used by ``main``.  Direct callers
        # that still carry ``teacher_model`` receive the legacy one-entry map;
        # ordinary callers see every configured Teacher.
        teacher_models = resolve_teacher_models(args)
    normalized_teachers = {
        _validate_teacher_key(key): Path(path) for key, path in teacher_models.items()
    }
    path = root / "run_manifest.json"
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment": EXPERIMENT,
        "config_signature": signature,
        "outcome": args.outcome,
        "dataset": str(dataset.resolve()),
        "dataset_sha256": sha256_file(dataset),
        "dataset_metadata": metadata,
        "response_count": len({int(c["record_index"]) for c in cases}),
        "selected_step_count": len(cases),
        "world_size": args.world_size,
        "shard_cases": SHARD_CASES,
        "prompts_per_generate": PROMPTS_PER_GENERATE,
        "expanded_rollouts_per_generate": EXPANDED_ROLLOUTS_PER_GENERATE,
        "samples_per_arm": SAMPLES_PER_ARM,
        "completion_fields_saved": [*COMPLETION_FIELDS_TO_SAVE, "reward"],
        "completion_fields_omitted": ["continuation_text", "vllm_completion_text"],
        "arms": [
            "keep",
            "delete",
            *("replace" if len(normalized_teachers) == 1 and
              getattr(args, "teacher_model", None) is not None else
              f"replace:{key}" for key in normalized_teachers),
        ],
        "student_model": model_id(args.student_model),
        # ``teacher_models`` is the authoritative multi-Teacher provenance.
        # Retain ``teacher_model`` only when there is one configured model so
        # older analysis scripts continue to discover it.
        "teacher_models": {
            key: model_id(model) for key, model in normalized_teachers.items()
        },
        "teacher_keys": list(normalized_teachers),
        **({"teacher_model": model_id(next(iter(normalized_teachers.values())))}
           if len(normalized_teachers) == 1 else {}),
        "max_response_tokens": MAX_RESPONSE_TOKENS,
        "teacher_proposal_max_new_tokens": TEACHER_PROPOSAL_MAX_NEW_TOKENS,
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "min_free_gpu_gib": float(args.min_free_gpu_gib),
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "seed": SEED,
        "created_at": utc_now(),
        "runner": str(Path(__file__).resolve()),
    }
    overwrite = bool(getattr(args, "overwrite", False))
    if not path.exists() or overwrite:
        atomic_json(path, payload)
    else:
        old = load_json(path)
        if old.get("config_signature") != signature:
            raise ValueError("existing run_manifest.json belongs to a different experiment")
    plan_path = root / "worker_plan.json"
    if not plan_path.exists() or overwrite:
        batches = shard_plan(cases)
        atomic_json(plan_path, {
            "config_signature": signature,
            "shard_cases": SHARD_CASES,
            "total_shards": len(batches),
            "workers": {
                str(worker): {
                    "shard_indices": worker_shards(len(batches), worker, args.world_size),
                    "target_cases": sum(len(batches[i]) for i in worker_shards(len(batches), worker, args.world_size)),
                }
                for worker in range(args.world_size)
            },
            "created_at": utc_now(),
        })
    else:
        old_plan = load_json(plan_path)
        if old_plan.get("config_signature") != signature or int(old_plan.get("shard_cases", -1)) != SHARD_CASES:
            raise ValueError("existing worker_plan.json belongs to a different shard layout")


def run_proposals(
    args: argparse.Namespace,
    root: Path,
    cases: list[dict[str, Any]],
    batches: list[list[dict[str, Any]]],
    signature: str,
    max_model_len: int,
    *,
    teacher_key: str | None = None,
    teacher_model: Path | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Generate one semantic proposal per case for one named Teacher.

    ``teacher_key`` is optional to retain the original single-Teacher API.  In
    a multi-Teacher run each invocation uses a distinct key and therefore a
    distinct proposal directory, while all proposal logic remains shared.
    """

    model = Path(teacher_model if teacher_model is not None
                 else getattr(args, "teacher_model", None) or DEFAULT_TEACHER_MODEL)
    shard_ids = worker_shards(len(batches), args.worker_id, args.world_size)
    if args.overwrite and not args.validate_only:
        # A keyed Teacher owns only its proposal and Replace shards.  Keep and
        # Delete are shared baselines and must not be erased once for every
        # Teacher in a multi-Teacher run.
        clear_arms = ("replace",) if teacher_key is not None else ARMS
        clear_worker_shards(root, args.worker_id, shard_ids,
                            clear_proposals=True, clear_arms=clear_arms,
                            teacher_key=teacher_key)
    proposal_map, failures = _proposal_map_for_worker(
        root,
        args.worker_id,
        shard_ids,
        batches,
        signature,
        teacher_key=teacher_key,
        teacher_model=model,
    )
    if args.overwrite:
        # ``_proposal_map_for_worker`` may have found valid old shards, but an
        # overwrite must not fall back to one of them if the fresh attempts
        # fail.  The newly written failure row is the only valid outcome for
        # that case in this invocation.
        proposal_map.clear()
        failures.clear()
    if args.validate_only:
        missing = [
            i for i in shard_ids
            if not _valid_proposal_shard(
                _proposal_paths(root, args.worker_id, i, teacher_key), batches[i], signature,
                teacher_key=teacher_key, teacher_model=model,
            )
        ]
        print(json.dumps({"event": "validate_proposals", "worker": args.worker_id,
                          "total_shards": len(shard_ids), "valid_shards": len(shard_ids)-len(missing),
                          "missing_shards": missing}, ensure_ascii=False), flush=True)
        return proposal_map, failures
    llm = None
    started_all = time.monotonic()
    completed_cases = sum(len(batches[i]) for i in shard_ids
                          if _valid_proposal_shard(
                              _proposal_paths(root, args.worker_id, i, teacher_key), batches[i], signature,
                              teacher_key=teacher_key, teacher_model=model,
                          ))
    save_progress(args, root, signature, {"status": "running", "phase": "proposals",
                  "proposal": {"completed_cases": completed_cases, "total_cases": sum(len(batches[i]) for i in shard_ids),
                                "success_cases": len(proposal_map), "failed_cases": len(failures)},
                  "teacher_key": teacher_key,
                  "continuations": {}, "gpu_id": args.gpu_id})
    pending_shards = [
        i for i in shard_ids
        if args.overwrite or not _valid_proposal_shard(
            _proposal_paths(root, args.worker_id, i, teacher_key), batches[i], signature,
            teacher_key=teacher_key, teacher_model=model,
        )
    ]
    for shard_id in pending_shards:
        for case in batches[shard_id]:
            proposal_map.pop(str(case["case_id"]), None)
            failures.pop(str(case["case_id"]), None)
    if not pending_shards:
        save_progress(args, root, signature, {"status": "complete", "phase": "proposals",
                      "proposal": {"completed_cases": sum(len(batches[i]) for i in shard_ids),
                                    "total_cases": sum(len(batches[i]) for i in shard_ids),
                                    "success_cases": len(proposal_map), "failed_cases": len(failures)},
                      "teacher_key": teacher_key,
                      "continuations": {}, "gpu_id": args.gpu_id})
        atomic_json(root / f"worker_{args.worker_id:02d}" / "summary.json", {
            "schema_version": SCHEMA_VERSION, "experiment": EXPERIMENT,
            "worker_id": args.worker_id, "config_signature": signature,
            "phase": "proposals", "proposal_success_cases": len(proposal_map),
            "proposal_failed_cases": len(failures), "updated_at": utc_now(),
        })
        return proposal_map, failures
    tokenizer = _load_tokenizer(model)
    try:
        _check_headroom(args, "before Teacher initialization")
        with base.GPUHeartbeat(
            gpu_id=args.gpu_id,
            minimum_free_mib=int(math.ceil(args.min_free_gpu_gib * base.MIB_PER_GIB)),
            check_interval=args.memory_check_seconds,
            report_interval=args.heartbeat_seconds,
            stage="Teacher initialization",
        ):
            llm = _new_llm(model, args, max_model_len)
        for shard_id in shard_ids:
            expected = batches[shard_id]
            path = _proposal_paths(root, args.worker_id, shard_id, teacher_key)
            if not args.overwrite and _valid_proposal_shard(
                    path, expected, signature, teacher_key=teacher_key, teacher_model=model):
                continue
            records: list[dict[str, Any]] = []
            failed_rows: list[dict[str, Any]] = []
            pending = list(expected)
            errors_by_case: dict[str, list[str]] = {str(c["case_id"]): [] for c in pending}
            for attempt in range(1, TEACHER_PROPOSAL_ATTEMPTS + 1):
                if not pending:
                    break
                prompts = [list(c["pre_step_input_token_ids"]) for c in pending]
                params = [_sampling(1, min(TEACHER_PROPOSAL_MAX_NEW_TOKENS,
                                           MAX_RESPONSE_TOKENS - len(c["pre_step_response_token_ids"])),
                                     proposal_seed(int(c["case_index"]), attempt)) for c in pending]
                teacher_label = teacher_key or "legacy"
                outputs = _generate(
                    llm,
                    prompts,
                    params,
                    args,
                    f"proposals/{teacher_label}/shard_{shard_id:05d}/attempt_{attempt}",
                )
                still_pending: list[dict[str, Any]] = []
                for case, output in zip(pending, outputs, strict=True):
                    try:
                        row = _proposal_record(case, output, tokenizer,
                                               proposal_seed(int(case["case_index"]), attempt), attempt,
                                               model, teacher_key)
                        records.append(row)
                        proposal_map[str(case["case_id"])] = row
                    except Exception as exc:
                        errors_by_case[str(case["case_id"])].append(repr(exc))
                        still_pending.append(case)
                pending = still_pending
            for case in pending:
                row = {"case_id": case["case_id"], "case_index": int(case["case_index"]),
                       "record_index": int(case["record_index"]),
                       "error": "; ".join(errors_by_case[str(case["case_id"])]),
                       "attempts": TEACHER_PROPOSAL_ATTEMPTS}
                failed_rows.append(row)
                failures[str(case["case_id"])] = row["error"]
            records.sort(key=lambda r: int(r["case_index"]))
            atomic_json(path, {"schema_version": SCHEMA_VERSION, "experiment": EXPERIMENT,
                               "phase": "proposals", "worker_id": args.worker_id,
                               "shard_index": shard_id, "config_signature": signature,
                               **({"teacher_key": _validate_teacher_key(teacher_key)}
                                  if teacher_key is not None else {}),
                               "teacher_model": str(model),
                               "target_cases": len(expected), "generated_outputs": len(records),
                               "records": records, "failures": failed_rows,
                               "saved_at": utc_now()})
            completed_cases += len(expected)
            _print_progress(args.outcome, args.worker_id, "proposals", None, completed_cases,
                            sum(len(batches[i]) for i in shard_ids), len(proposal_map), started_all)
            save_progress(args, root, signature, {"status": "running", "phase": "proposals",
                          "proposal": {"completed_cases": completed_cases, "total_cases": sum(len(batches[i]) for i in shard_ids),
                                        "success_cases": len(proposal_map), "failed_cases": len(failures)},
                          "continuations": {}, "gpu_id": args.gpu_id})
    finally:
        if llm is not None:
            base.shutdown_vllm(llm)
        del tokenizer
        gc.collect()
        with contextlib.suppress(Exception):
            import torch
            torch.cuda.empty_cache()
    save_progress(args, root, signature, {"status": "complete", "phase": "proposals",
                  "proposal": {"completed_cases": sum(len(batches[i]) for i in shard_ids),
                                "total_cases": sum(len(batches[i]) for i in shard_ids),
                                "success_cases": len(proposal_map), "failed_cases": len(failures)},
                  "teacher_key": teacher_key,
                  "continuations": {}, "gpu_id": args.gpu_id})
    atomic_json(root / f"worker_{args.worker_id:02d}" / "summary.json", {
        "schema_version": SCHEMA_VERSION, "experiment": EXPERIMENT,
        "worker_id": args.worker_id, "config_signature": signature,
        "phase": "proposals", "proposal_success_cases": len(proposal_map),
        "proposal_failed_cases": len(failures), "updated_at": utc_now(),
    })
    return proposal_map, failures


def run_continuations(
    args: argparse.Namespace,
    root: Path,
    cases: list[dict[str, Any]],
    batches: list[list[dict[str, Any]]],
    signature: str,
    max_model_len: int,
    proposal_map: dict[str, dict[str, Any]],
    proposal_failures: dict[str, str],
    *,
    teacher_key: str | None = None,
    teacher_proposals: dict[str, tuple[dict[str, dict[str, Any]], dict[str, str]]] | None = None,
) -> None:
    """Generate Student continuations for shared and named Teacher arms.

    The original positional arguments still describe a single Teacher.  When
    ``teacher_proposals`` is supplied, each mapping entry is a ``(proposals,
    failures)`` pair and one Student instance serves Keep/Delete plus every
    named Replace arm.  This avoids reloading the Student for each Teacher.
    """

    shard_ids = worker_shards(len(batches), args.worker_id, args.world_size)
    worker_total_cases = sum(len(batches[i]) for i in shard_ids)

    # Build a uniform list of arm specifications.  ``label`` is used only for
    # progress keys; the on-disk arm remains ``replace`` for every Teacher.
    specs: list[dict[str, Any]] = []
    if teacher_proposals is None:
        requested = ARMS if args.arm == "all" else (args.arm,)
        for arm in requested:
            specs.append({
                "label": arm,
                "arm": arm,
                "teacher_key": teacher_key if arm == "replace" else None,
                "proposals": proposal_map if arm == "replace" else {},
                "failures": proposal_failures if arm == "replace" else {},
            })
    else:
        normalized: list[tuple[str, dict[str, dict[str, Any]], dict[str, str]]] = []
        for raw_key, payload in teacher_proposals.items():
            key = _validate_teacher_key(raw_key)
            if isinstance(payload, tuple) and len(payload) == 2:
                pmap, pfail = payload
            elif isinstance(payload, dict) and "proposals" in payload:
                pmap = payload.get("proposals", {})
                pfail = payload.get("failures", {})
            else:
                raise TypeError(
                    "teacher_proposals values must be (proposal_map, failure_map) pairs"
                )
            if not isinstance(pmap, dict) or not isinstance(pfail, dict):
                raise TypeError(f"invalid proposal maps for Teacher {key!r}")
            normalized.append((key, pmap, pfail))
        if args.arm in {"all", "keep"}:
            specs.append({"label": "keep", "arm": "keep", "teacher_key": None,
                          "proposals": {}, "failures": {}})
        if args.arm in {"all", "delete"}:
            specs.append({"label": "delete", "arm": "delete", "teacher_key": None,
                          "proposals": {}, "failures": {}})
        if args.arm in {"all", "replace"}:
            for key, pmap, pfail in normalized:
                specs.append({"label": f"replace:{key}", "arm": "replace",
                              "teacher_key": key, "proposals": pmap, "failures": pfail})
        if not specs:
            raise ValueError(f"unsupported continuation arm: {args.arm!r}")

    if args.overwrite and not args.validate_only:
        for spec in specs:
            clear_worker_shards(
                root, args.worker_id, shard_ids, clear_proposals=False,
                clear_arms=(spec["arm"],), teacher_key=spec["teacher_key"],
            )

    worker_case_ids = {
        str(case["case_id"])
        for shard_id in shard_ids
        for case in batches[shard_id]
    }
    if not args.validate_only:
        for spec in specs:
            if spec["arm"] != "replace":
                continue
            accounted = set(spec["proposals"]) | set(spec["failures"])
            unaccounted = sorted(worker_case_ids - accounted)
            if unaccounted:
                preview = ", ".join(unaccounted[:5])
                suffix = "..." if len(unaccounted) > 5 else ""
                key_suffix = f" ({spec['teacher_key']})" if spec["teacher_key"] else ""
                raise RuntimeError(
                    "Teacher proposals are missing"
                    f"{key_suffix} for {len(unaccounted)} cases ({preview}{suffix}). "
                    "Run the proposal phase first or resume the same output directory."
                )

    if args.validate_only:
        report: dict[str, Any] = {}
        for spec in specs:
            valid = [
                i for i in shard_ids
                if _valid_arm_shard(
                    _arm_paths(root, args.worker_id, spec["arm"], i, spec["teacher_key"]),
                    batches[i], signature, spec["arm"],
                    spec["proposals"] if spec["arm"] == "replace" else None,
                    teacher_key=spec["teacher_key"],
                    proposal_failures=spec["failures"] if spec["arm"] == "replace" else None,
                )
            ]
            report[spec["label"]] = {
                "valid_shards": len(valid), "total_shards": len(shard_ids),
                "missing_shards": [i for i in shard_ids if i not in valid],
            }
        print(json.dumps({"event": "validate_continuations", "worker": args.worker_id,
                          "arms": report}, ensure_ascii=False), flush=True)
        return

    tokenizer = _load_tokenizer(args.student_model)
    # Verify each Teacher's token IDs using the Student tokenizer before any
    # replacement request is scheduled.  Keep/Delete remain runnable if one
    # Teacher has incompatible tokenizer semantics.
    for spec in specs:
        if spec["arm"] != "replace":
            continue
        pmap = spec["proposals"]
        failures = spec["failures"]
        print(json.dumps({"event": "teacher_student_tokenizer_check_start",
                          "outcome": args.outcome, "worker": args.worker_id,
                          "teacher_key": spec["teacher_key"],
                          "proposal_cases": len(pmap)}, ensure_ascii=False), flush=True)
        mismatch_count = 0
        for case_id, proposal in list(pmap.items()):
            try:
                ids = [int(v) for v in proposal["replacement_step_token_ids"]]
                decoded = base.decode_ids(tokenizer, ids, skip_special_tokens=False)
                compatible = decoded == str(proposal["replacement_step_decoded_text"])
            except Exception:
                compatible = False
            if not compatible:
                failures[case_id] = "student_teacher_tokenizer_decode_mismatch"
                del pmap[case_id]
                mismatch_count += 1
        print(json.dumps({"event": "teacher_student_tokenizer_check_complete",
                          "outcome": args.outcome, "worker": args.worker_id,
                          "teacher_key": spec["teacher_key"],
                          "compatible_cases": len(pmap),
                          "mismatched_cases": mismatch_count},
                         ensure_ascii=False), flush=True)

    llm = None
    started_all = time.monotonic()
    total = worker_total_cases
    progress_arms: dict[str, Any] = {
        str(spec["label"]): {"completed_cases": 0, "total_cases": total,
                              "generated_outputs": 0, "skipped_cases": 0}
        for spec in specs
    }

    def proposal_progress() -> dict[str, int]:
        replaces = [s for s in specs if s["arm"] == "replace"]
        return {
            "completed_cases": worker_total_cases * len(replaces),
            "total_cases": worker_total_cases * len(replaces),
            "success_cases": sum(len(s["proposals"]) for s in replaces),
            "failed_cases": sum(len(s["failures"]) for s in replaces),
        }

    save_progress(args, root, signature, {
        "status": "running", "phase": "continuations",
        "proposal": proposal_progress(), "continuations": progress_arms,
        "gpu_id": args.gpu_id,
    })
    try:
        _check_headroom(args, "before Student initialization")
        with base.GPUHeartbeat(
            gpu_id=args.gpu_id,
            minimum_free_mib=int(math.ceil(args.min_free_gpu_gib * base.MIB_PER_GIB)),
            check_interval=args.memory_check_seconds,
            report_interval=args.heartbeat_seconds,
            stage="Student initialization",
        ):
            llm = _new_llm(args.student_model, args, max_model_len)
        for spec in specs:
            label = str(spec["label"])
            arm = str(spec["arm"])
            key = spec["teacher_key"]
            pmap = spec["proposals"]
            failures = spec["failures"]
            for shard_id in shard_ids:
                expected = batches[shard_id]
                path = _arm_paths(root, args.worker_id, arm, shard_id, key)
                if not args.overwrite and _valid_arm_shard(
                        path, expected, signature, arm,
                        pmap if arm == "replace" else None, teacher_key=key,
                        proposal_failures=failures if arm == "replace" else None):
                    payload = load_json_maybe_gzip(path)
                    progress_arms[label]["completed_cases"] += len(expected)
                    progress_arms[label]["generated_outputs"] += len(payload.get("records", [])) * SAMPLES_PER_ARM
                    progress_arms[label]["skipped_cases"] += len(payload.get("skipped", []))
                    continue
                records: list[dict[str, Any]] = []
                skipped: list[dict[str, Any]] = []
                runnable: list[dict[str, Any]] = []
                for case in expected:
                    proposal = pmap.get(str(case["case_id"])) if arm == "replace" else None
                    if arm == "replace" and proposal is None:
                        skipped.append({
                            "case_id": case["case_id"],
                            "case_index": int(case["case_index"]),
                            "reason": failures.get(str(case["case_id"]), "missing_teacher_proposal"),
                        })
                    else:
                        runnable.append(case)
                for micro_start in range(0, len(runnable), PROMPTS_PER_GENERATE):
                    micro = runnable[micro_start:micro_start + PROMPTS_PER_GENERATE]
                    prompts: list[list[int]] = []
                    params: list[Any] = []
                    for case in micro:
                        proposal = pmap.get(str(case["case_id"])) if arm == "replace" else None
                        if arm == "keep":
                            prefix = list(case["keep_response_prefix_token_ids"])
                        elif arm == "delete":
                            prefix = list(case["pre_step_response_token_ids"])
                        else:
                            prefix = list(case["pre_step_response_token_ids"]) + [
                                int(v) for v in proposal["replacement_step_token_ids"]
                            ]
                        budget = MAX_RESPONSE_TOKENS - len(prefix)
                        prompts.append(list(case["prompt_token_ids"]) + prefix)
                        params.append(_sampling(
                            SAMPLES_PER_ARM, budget,
                            continuation_seed(int(case["case_index"])),
                        ))
                    outputs = _generate(
                        llm, prompts, params, args,
                        f"continuations/{label}/shard_{shard_id:05d}/micro_{micro_start // PROMPTS_PER_GENERATE:03d}",
                    )
                    for case, output in zip(micro, outputs, strict=True):
                        proposal = pmap.get(str(case["case_id"])) if arm == "replace" else None
                        records.append(_continuation_record(
                            case, arm, proposal, output, tokenizer, args.student_model,
                            continuation_seed(int(case["case_index"])), teacher_key=key,
                        ))
                    progress_arms[label]["completed_cases"] += len(micro)
                    progress_arms[label]["generated_outputs"] += len(micro) * SAMPLES_PER_ARM
                    _print_progress(
                        args.outcome, args.worker_id, "continuations", label,
                        progress_arms[label]["completed_cases"], total,
                        progress_arms[label]["generated_outputs"], started_all,
                    )
                    save_progress(args, root, signature, {
                        "status": "running", "phase": "continuations",
                        "proposal": proposal_progress(), "continuations": progress_arms,
                        "gpu_id": args.gpu_id, "current_arm": label,
                        "current_shard": shard_id,
                    })
                progress_arms[label]["completed_cases"] += len(skipped)
                progress_arms[label]["skipped_cases"] += len(skipped)
                records.sort(key=lambda r: int(r["case_index"]))
                skipped.sort(key=lambda r: int(r["case_index"]))
                atomic_gzip_json(path, {
                    "schema_version": SCHEMA_VERSION,
                    "experiment": EXPERIMENT,
                    "phase": "continuations",
                    "arm": arm,
                    "worker_id": args.worker_id,
                    "shard_index": shard_id,
                    "config_signature": signature,
                    **({"teacher_key": _validate_teacher_key(key)} if key is not None else {}),
                    "target_cases": len(expected),
                    "runnable_cases": len(records),
                    "skipped_cases": len(skipped),
                    "generated_outputs": len(records) * SAMPLES_PER_ARM,
                    "completion_fields_saved": [*COMPLETION_FIELDS_TO_SAVE, "reward"],
                    "completion_fields_omitted": ["continuation_text", "vllm_completion_text"],
                    "records": records,
                    "skipped": skipped,
                    "saved_at": utc_now(),
                })
    finally:
        if llm is not None:
            base.shutdown_vllm(llm)
        del tokenizer
        gc.collect()
        with contextlib.suppress(Exception):
            import torch
            torch.cuda.empty_cache()
    save_progress(args, root, signature, {
        "status": "complete", "phase": "complete",
        "proposal": proposal_progress(), "continuations": progress_arms,
        "gpu_id": args.gpu_id,
    })
    total_success = sum(len(s["proposals"]) for s in specs if s["arm"] == "replace")
    total_failures = sum(len(s["failures"]) for s in specs if s["arm"] == "replace")
    atomic_json(root / f"worker_{args.worker_id:02d}" / "summary.json", {
        "schema_version": SCHEMA_VERSION, "experiment": EXPERIMENT,
        "worker_id": args.worker_id, "config_signature": signature,
        "phase": "complete", "proposal_success_cases": total_success,
        "proposal_failed_cases": total_failures,
        "continuations": progress_arms, "updated_at": utc_now(),
    })
    print(json.dumps({"event": "worker_complete", "outcome": args.outcome,
                      "worker": args.worker_id,
                      "proposal_success": total_success,
                      "proposal_failures": total_failures,
                      "continuations": progress_arms}, ensure_ascii=False), flush=True)


def _monitor_arm_labels(
    args: argparse.Namespace,
    teacher_models: dict[str, Path] | None,
) -> tuple[str, ...]:
    """Return the progress labels expected for this invocation.

    A keyed Replace arm is written as ``replace:<teacher-key>`` in progress
    files, while the historical single-Teacher API continues to use simply
    ``replace``.  Keep/Delete are always shared and therefore appear once.
    """

    if args.phase == "proposals":
        return ()
    legacy = getattr(args, "teacher_model", None) is not None
    if args.arm == "keep":
        return ("keep",)
    if args.arm == "delete":
        return ("delete",)
    if args.arm == "replace":
        if legacy or teacher_models is None:
            return ("replace",)
        return tuple(f"replace:{key}" for key in teacher_models)
    if args.arm == "all":
        labels = ["keep", "delete"]
        if legacy or teacher_models is None:
            labels.append("replace")
        else:
            labels.extend(f"replace:{key}" for key in teacher_models)
        return tuple(labels)
    raise ValueError(f"unsupported arm {args.arm!r}")


def monitor(
    args: argparse.Namespace,
    root: Path,
    signature: str,
    expected_cases: int,
    teacher_models: dict[str, Path] | None = None,
) -> None:
    """Monitor all workers, including one Replace stream per Teacher."""

    started = time.monotonic()
    last_line = ""
    run_id = str(getattr(args, "run_id", ""))
    labels = _monitor_arm_labels(args, teacher_models)
    # In a multi-Teacher proposal phase every configured Teacher must finish.
    # During an ``all`` run workers enter continuations only after all proposal
    # loops, so this same barrier also gives an honest aggregate percentage.
    teacher_count = len(teacher_models or {})
    if teacher_count == 0:
        teacher_count = 1
    legacy = getattr(args, "teacher_model", None) is not None
    proposal_required = args.phase == "proposals" or (
        args.phase == "all" and args.arm in {"all", "replace"}
    )
    proposal_target = expected_cases * (1 if legacy else teacher_count) if proposal_required else 0

    while True:
        rows = []
        for worker in range(args.world_size):
            path = _progress_path(root, worker)
            if path.is_file():
                with contextlib.suppress(Exception):
                    row = load_json(path)
                    if (row.get("config_signature") == signature
                            and (not run_id or str(row.get("run_id", "")) == run_id)):
                        rows.append(row)
        proposals_done = sum(int(r.get("proposal", {}).get("completed_cases", 0)) for r in rows)
        proposal_success = sum(int(r.get("proposal", {}).get("success_cases", 0)) for r in rows)
        proposal_failed = sum(int(r.get("proposal", {}).get("failed_cases", 0)) for r in rows)
        arm_stats: dict[str, dict[str, int]] = {}
        for label in labels:
            arm_stats[label] = {
                "completed_cases": sum(int(r.get("continuations", {}).get(label, {}).get("completed_cases", 0)) for r in rows),
                "generated_outputs": sum(int(r.get("continuations", {}).get(label, {}).get("generated_outputs", 0)) for r in rows),
                "skipped_cases": sum(int(r.get("continuations", {}).get(label, {}).get("skipped_cases", 0)) for r in rows),
            }
        elapsed = max(1e-6, time.monotonic() - started)
        complete_workers = sum(r.get("status") == "complete" for r in rows)
        arm_eta: dict[str, float | None] = {}
        for label in labels:
            done = arm_stats[label]["completed_cases"]
            rate = done / elapsed
            arm_eta[label] = (expected_cases - done) / rate if rate > 0 else None
        proposal_pct = 100.0 * proposals_done / proposal_target if proposal_target else 100.0
        proposal_text = (
            f"proposal={proposals_done}/{proposal_target} ({proposal_pct:.1f}%,"
            f"ok={proposal_success},fail={proposal_failed})"
        )
        arm_text = " ".join(
            f"{label}={arm_stats[label]['completed_cases']}/{expected_cases} "
            f"({100.0 * arm_stats[label]['completed_cases'] / expected_cases:.1f}%),"
            f"out={arm_stats[label]['generated_outputs']},skip={arm_stats[label]['skipped_cases']},"
            f"ETA={arm_eta[label] / 60:.1f}m"
            if arm_eta[label] is not None else
            f"{label}=0/{expected_cases} (0.0%),out=0,skip=0,ETA=?"
            for label in labels
        )
        line = (
            f"MONITOR outcome={args.outcome} workers={complete_workers}/{args.world_size} "
            f"{proposal_text} {arm_text}"
        ).rstrip()
        if line != last_line:
            print(line + f" elapsed={elapsed / 60:.1f}m", flush=True)
            last_line = line
        atomic_json(root / "progress.json", {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "config_signature": signature,
            "run_id": run_id,
            "status": "running",
            "workers_seen": len(rows),
            "proposal_target_cases": proposal_target,
            "proposal_completed_cases": proposals_done,
            "proposal_success_cases": proposal_success,
            "proposal_failed_cases": proposal_failed,
            "teacher_keys": list(teacher_models or {}),
            "arms": arm_stats,
            "updated_at": utc_now(),
        })
        proposal_complete = (not proposal_required) or proposals_done >= proposal_target
        arms_complete = all(
            arm_stats[label]["completed_cases"] >= expected_cases for label in labels
        )
        if complete_workers == args.world_size and proposal_complete and arms_complete:
            summary = {
                "outcome": args.outcome,
                "workers": args.world_size,
                "proposal_target_cases": proposal_target,
                "proposal_success": proposal_success,
                "proposal_failures": proposal_failed,
                "arms": arm_stats,
                "elapsed_seconds": round(elapsed, 2),
            }
            print("MONITOR COMPLETE " + json.dumps(summary, ensure_ascii=False), flush=True)
            atomic_json(root / "progress.json", {
                "schema_version": SCHEMA_VERSION,
                "experiment": EXPERIMENT,
                "config_signature": signature,
                "run_id": run_id,
                "status": "complete",
                "workers": rows,
                "teacher_keys": list(teacher_models or {}),
                "arms": arm_stats,
                "proposal_target_cases": proposal_target,
                "proposal_success": proposal_success,
                "proposal_failures": proposal_failed,
                "updated_at": utc_now(),
            })
            atomic_json(root / "summary.json", {
                "schema_version": SCHEMA_VERSION,
                "experiment": EXPERIMENT,
                "outcome": args.outcome,
                "config_signature": signature,
                "run_id": run_id,
                "status": "complete",
                "selected_steps": expected_cases,
                "teacher_keys": list(teacher_models or {}),
                "proposal_target_cases": proposal_target,
                "proposal_success_cases": proposal_success,
                "proposal_failed_cases": proposal_failed,
                "arms": arm_stats,
                "updated_at": utc_now(),
            })
            return
        time.sleep(5.0)


def install_shutdown_handler() -> None:
    """Turn shell termination into a normal Python exception with cleanup."""

    def _handle_sigterm(signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, _handle_sigterm)


def main() -> None:
    install_shutdown_handler()
    args = parse_args()
    if not args.run_id:
        # Standalone invocations still get stale-progress isolation. The
        # companion shell supplies one shared ID to its monitor and workers.
        args.run_id = f"pid{os.getpid()}-{time.time_ns()}"
    # Resolve the user-editable Teacher mapping once per worker.  Absolute
    # paths make the run manifest and resume signature independent of the
    # caller's current working directory.
    teacher_models = resolve_teacher_model_paths(resolve_teacher_models(args))
    validate_args(args, teacher_models)
    dataset = (args.dataset or DEFAULT_DATASET[args.outcome]).resolve()
    output_root = args.output_root.resolve() / args.outcome
    if not dataset.is_file():
        raise FileNotFoundError(dataset)
    metadata, cases = load_dataset(dataset, args.outcome)
    batches = shard_plan(cases)
    max_model_len = derive_max_model_len(cases, args.max_model_len)
    if getattr(args, "teacher_model", None) is not None:
        # Preserve the original single-Teacher signature shape for callers
        # that still use ``--teacher-model``.  The normal mapping path binds
        # the complete named Teacher set below.
        signature = run_signature(
            sha256_file(dataset),
            args.outcome,
            args.student_model,
            teacher=next(iter(teacher_models.values())),
            world_size=args.world_size,
            max_model_len=max_model_len,
        )
    else:
        signature = run_signature(
            sha256_file(dataset),
            args.outcome,
            args.student_model,
            world_size=args.world_size,
            max_model_len=max_model_len,
            teacher_models=teacher_models,
        )
    output_root.mkdir(parents=True, exist_ok=True)
    _write_run_manifest(
        output_root, args, dataset, metadata, cases, signature,
        teacher_models=teacher_models,
    )
    legacy_single = getattr(args, "teacher_model", None) is not None
    replace_labels = (
        ("replace",) if legacy_single
        else tuple(f"replace:{key}" for key in teacher_models)
    )
    if args.arm == "all":
        planned_arms = ["keep", "delete", *replace_labels]
    elif args.arm == "keep":
        planned_arms = ["keep"]
    elif args.arm == "delete":
        planned_arms = ["delete"]
    else:
        planned_arms = list(replace_labels)
    print(json.dumps({"event": "plan", "outcome": args.outcome, "dataset": str(dataset),
                      "output_root": str(output_root), "responses": len({int(c['record_index']) for c in cases}),
                      "steps": len(cases), "shards": len(batches),
                      "worker": args.worker_id, "world_size": args.world_size,
                      "run_id": args.run_id,
                      "worker_shards": worker_shards(len(batches), args.worker_id, args.world_size),
                      "samples_per_arm": SAMPLES_PER_ARM,
                      "prompts_per_generate": PROMPTS_PER_GENERATE,
                      "expanded_rollouts_per_generate": EXPANDED_ROLLOUTS_PER_GENERATE,
                      "arms": planned_arms,
                      "teacher_models": {key: str(model) for key, model in teacher_models.items()},
                      "teacher_keys": list(teacher_models),
                      "max_response_tokens": MAX_RESPONSE_TOKENS,
                      "teacher_proposal_max_new_tokens": TEACHER_PROPOSAL_MAX_NEW_TOKENS,
                      "gpu_memory_utilization": float(args.gpu_memory_utilization),
                      "min_free_gpu_gib": float(args.min_free_gpu_gib),
                      "dataset_preparation_caps": {
                          "max_response_tokens": metadata.get("max_response_tokens"),
                          "teacher_proposal_max_new_tokens": metadata.get(
                              "teacher_proposal_max_new_tokens"
                          ),
                      },
                      "max_model_len": max_model_len,
                      "config_signature": signature}, ensure_ascii=False), flush=True)
    if args.mode == "monitor":
        monitor(args, output_root, signature, len(cases), teacher_models)
        return
    _set_cuda_visibility(args.gpu_id)

    # Each configured Teacher gets an independent proposal map.  The maps are
    # deliberately kept in a keyed dictionary so a failed proposal from one
    # model can never suppress another model's Replace arm.
    proposal_maps: dict[str, tuple[dict[str, dict[str, Any]], dict[str, str]]] = {}
    need_proposals = args.phase == "proposals" or (
        args.phase == "all" and args.arm in {"all", "replace"}
    )
    if need_proposals:
        worker_total_cases = sum(
            len(batches[i])
            for i in worker_shards(len(batches), args.worker_id, args.world_size)
        )
        teacher_items = list(teacher_models.items())
        for teacher_index, (key, model) in enumerate(teacher_items, start=1):
            # Explicit --teacher-model retains the historical unkeyed paths;
            # the normal Python mapping always uses a key namespace.
            effective_key = None if legacy_single else key
            pmap, pfail = run_proposals(
                args,
                output_root,
                cases,
                batches,
                signature,
                max_model_len,
                teacher_key=effective_key,
                teacher_model=model,
            )
            proposal_maps[key] = (pmap, pfail)
            if not args.validate_only:
                successes = sum(len(pair[0]) for pair in proposal_maps.values())
                failures = sum(len(pair[1]) for pair in proposal_maps.values())
                save_progress(args, output_root, signature, {
                    "status": "running",
                    "phase": "proposals",
                    "proposal": {
                        "completed_cases": worker_total_cases * teacher_index,
                        "total_cases": worker_total_cases * len(teacher_items),
                        "success_cases": successes,
                        "failed_cases": failures,
                    },
                    "teacher_keys_completed": [
                        item[0] for item in teacher_items[:teacher_index]
                    ],
                    "continuations": {},
                    "gpu_id": args.gpu_id,
                })

    if args.phase == "proposals":
        # ``run_proposals`` writes a per-Teacher summary while it is running.
        # Replace that transient view with an aggregate before returning so a
        # proposal-only job reports every configured Teacher rather than just
        # whichever model happened to run last.
        atomic_json(output_root / f"worker_{args.worker_id:02d}" / "summary.json", {
            "schema_version": SCHEMA_VERSION,
            "experiment": EXPERIMENT,
            "worker_id": args.worker_id,
            "config_signature": signature,
            "phase": "proposals",
            "teacher_keys": list(teacher_models),
            "proposal_success_cases_by_teacher": {
                key: len(pair[0]) for key, pair in proposal_maps.items()
            },
            "proposal_failed_cases_by_teacher": {
                key: len(pair[1]) for key, pair in proposal_maps.items()
            },
            "proposal_success_cases": sum(len(pair[0]) for pair in proposal_maps.values()),
            "proposal_failed_cases": sum(len(pair[1]) for pair in proposal_maps.values()),
            "updated_at": utc_now(),
        })
        return

    if args.phase in {"all", "continuations"}:
        if legacy_single:
            key = next(iter(teacher_models))
            pmap, pfail = proposal_maps.get(key, ({}, {}))
            if not pmap and args.arm in {"all", "replace"}:
                pmap, pfail = _proposal_map_for_worker(
                    output_root,
                    args.worker_id,
                    worker_shards(len(batches), args.worker_id, args.world_size),
                    batches,
                    signature,
                    teacher_key=None,
                    teacher_model=teacher_models[key],
                )
            run_continuations(
                args, output_root, cases, batches, signature, max_model_len,
                pmap, pfail,
            )
        else:
            if args.arm in {"all", "replace"}:
                worker_shard_ids = worker_shards(
                    len(batches), args.worker_id, args.world_size
                )
                for key, model in teacher_models.items():
                    if key not in proposal_maps:
                        proposal_maps[key] = _proposal_map_for_worker(
                            output_root,
                            args.worker_id,
                            worker_shard_ids,
                            batches,
                            signature,
                            teacher_key=key,
                            teacher_model=model,
                        )
            # One Student instance serves the shared Keep/Delete baselines and
            # every per-Teacher Replace arm.
            run_continuations(
                args, output_root, cases, batches, signature, max_model_len,
                {}, {}, teacher_proposals=proposal_maps,
            )


if __name__ == "__main__":
    main()
