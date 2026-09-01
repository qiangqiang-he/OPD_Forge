#!/usr/bin/env python3
"""Evaluate every checkpoint in one OPD_Forge training output with vLLM.

The module deliberately keeps vLLM and CUDA imports inside spawned GPU workers.
Configuration validation, checkpoint discovery, dataset loading, prompt-length
checks, metric aggregation, and ``--validate-only`` therefore run on CPU.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import multiprocessing as mp
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from omegaconf import OmegaConf

from utils.prompts import get_prompt_template, render_prompt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATTERN = re.compile(r"^global_step_(\d+)$")
REQUIRED_METRICS = ("avg_at_n", "pass_at_n", "mean_length", "truncation_rate")
CHECKPOINT_SOURCE = "checkpoints"
DIRECT_MODEL_SOURCE = "models"
HF_MODELS_TYPE = "hf_models"


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path


@dataclass(frozen=True)
class ModelSpec:
    name: str
    path: Path


@dataclass(frozen=True)
class EvaluationConfig:
    config_path: Path
    project_root: Path
    source_type: str
    training_output: Path | None
    models: tuple[ModelSpec, ...]
    run_name: str
    datasets: tuple[DatasetSpec, ...]
    prompt_name: str
    seed: int
    temperature: float
    top_p: float
    n: int
    max_new_tokens: int
    length_control: tuple[int, ...]
    gpus: tuple[int, ...]
    tensor_parallel_size: int
    rollouts_per_gpu_batch: int
    dtype: str
    trust_remote_code: bool
    gpu_memory_utilization: float
    max_model_len: int
    enable_prefix_caching: bool
    result_dir: Path

    @property
    def questions_per_gpu_batch(self) -> int:
        return self.rollouts_per_gpu_batch // self.n

    @property
    def result_path(self) -> Path:
        return self.result_dir / f"{self.run_name}.json"


@dataclass(frozen=True)
class CheckpointSpec:
    step_num: int
    checkpoint_dir: Path
    source_kind: str
    source_dir: Path
    tokenizer_dir: Path
    model_dir: Path | None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a YAML mapping.")
    return value


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer, got {value!r}.")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}, got {value}.")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric, got {value!r}.")
    return float(value)


def _resolve_project_path(value: Any, project_root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty path string.")
    path = Path(value)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def load_evaluation_config(
    config_path: str | Path,
    *,
    project_root: Path = PROJECT_ROOT,
    source_type: str = CHECKPOINT_SOURCE,
) -> EvaluationConfig:
    """Load and strictly validate the YAML evaluation contract."""

    project_root = project_root.resolve()
    config_path = Path(config_path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Evaluation config does not exist: {config_path}")

    raw_config = OmegaConf.load(config_path)
    OmegaConf.resolve(raw_config)
    raw = _mapping(OmegaConf.to_container(raw_config, resolve=True), "config")

    if source_type not in {CHECKPOINT_SOURCE, DIRECT_MODEL_SOURCE}:
        raise ValueError(f"Unsupported evaluation source type: {source_type!r}")

    training_output: Path | None
    models: tuple[ModelSpec, ...]
    if source_type == CHECKPOINT_SOURCE:
        if "models" in raw or "model_source" in raw:
            raise ValueError(
                "The checkpoint evaluation config must contain training_output, not "
                "model_source/models. Checkpoints are discovered automatically."
            )
        training_output = _resolve_project_path(
            raw.get("training_output"), project_root, "training_output"
        )
        if not training_output.is_dir():
            raise NotADirectoryError(
                f"training_output is not a directory: {training_output}"
            )
        run_name = training_output.name
        if not run_name:
            raise ValueError(
                f"Cannot derive run_name from training_output: {training_output}"
            )
        models = ()
    else:
        if "training_output" in raw:
            raise ValueError(
                "The direct-model evaluation config must contain model_source, not "
                "training_output."
            )
        if "models" in raw:
            raise ValueError(
                "models must be nested under model_source in a direct-model config."
            )
        run_name = raw.get("run_name")
        if not isinstance(run_name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*", run_name
        ):
            raise ValueError(
                "run_name must be a filesystem-safe name containing only letters, "
                "digits, dots, underscores, and hyphens."
            )
        model_source = _mapping(raw.get("model_source"), "model_source")
        if model_source.get("type") != HF_MODELS_TYPE:
            raise ValueError(
                f"model_source.type must be {HF_MODELS_TYPE!r} for direct-model "
                "evaluation."
            )
        raw_models = model_source.get("models")
        if not isinstance(raw_models, list) or not raw_models:
            raise ValueError("model_source.models must be a non-empty YAML list.")
        parsed_models: list[ModelSpec] = []
        seen_model_names: set[str] = set()
        seen_model_paths: set[Path] = set()
        for index, item in enumerate(raw_models):
            model = _mapping(item, f"model_source.models[{index}]")
            name = model.get("name")
            if not isinstance(name, str) or not re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._-]*", name
            ):
                raise ValueError(
                    f"model_source.models[{index}].name must be a filesystem-safe "
                    "non-empty name."
                )
            if name in seen_model_names:
                raise ValueError(f"Duplicate model name: {name}")
            model_path = _resolve_project_path(
                model.get("path"),
                project_root,
                f"model_source.models[{index}].path",
            )
            if model_path in seen_model_paths:
                raise ValueError(f"Duplicate model path: {model_path}")
            if not is_huggingface_model_dir(model_path):
                raise ValueError(
                    f"model_source.models[{index}].path is not a complete Hugging "
                    f"Face model directory: {model_path}"
                )
            seen_model_names.add(name)
            seen_model_paths.add(model_path)
            parsed_models.append(ModelSpec(name=name, path=model_path))
        training_output = None
        models = tuple(parsed_models)

    raw_datasets = raw.get("datasets")
    if not isinstance(raw_datasets, list) or not raw_datasets:
        raise ValueError("datasets must be a non-empty YAML list.")
    datasets: list[DatasetSpec] = []
    seen_dataset_names: set[str] = set()
    for index, item in enumerate(raw_datasets):
        dataset = _mapping(item, f"datasets[{index}]")
        name = dataset.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"datasets[{index}].name must be a non-empty string.")
        if name in seen_dataset_names:
            raise ValueError(f"Duplicate dataset name: {name}")
        configured_path = dataset.get("path")
        if not isinstance(configured_path, str) or not configured_path.strip():
            raise ValueError(f"datasets[{index}].path must be a non-empty string.")
        if Path(configured_path).is_absolute():
            raise ValueError(
                f"datasets[{index}].path must be relative to OPD_Forge: {configured_path}"
            )
        dataset_path = _resolve_project_path(
            configured_path, project_root, f"datasets[{index}].path"
        )
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Dataset does not exist: {dataset_path}")
        seen_dataset_names.add(name)
        datasets.append(DatasetSpec(name=name, path=dataset_path))

    prompt_name = raw.get("prompt")
    if not isinstance(prompt_name, str) or not prompt_name:
        raise ValueError("prompt must be a registered prompt name.")
    get_prompt_template(prompt_name)

    seed = _integer(raw.get("seed"), "seed", minimum=0)
    sampling = _mapping(raw.get("sampling"), "sampling")
    temperature = _number(sampling.get("temperature"), "sampling.temperature")
    if temperature < 0:
        raise ValueError("sampling.temperature must be non-negative.")
    top_p = _number(sampling.get("top_p"), "sampling.top_p")
    if not 0 < top_p <= 1:
        raise ValueError("sampling.top_p must be in (0, 1].")
    n = _integer(sampling.get("n"), "sampling.n", minimum=1)
    max_new_tokens = _integer(
        sampling.get("max_new_tokens"), "sampling.max_new_tokens", minimum=1
    )

    raw_lengths = raw.get("length_control")
    if not isinstance(raw_lengths, list) or not raw_lengths:
        raise ValueError("length_control must be a non-empty list of token limits.")
    length_control = tuple(
        _integer(value, f"length_control[{index}]", minimum=1)
        for index, value in enumerate(raw_lengths)
    )
    if tuple(sorted(set(length_control))) != length_control:
        raise ValueError("length_control values must be unique and strictly ascending.")
    if length_control[-1] != max_new_tokens:
        raise ValueError(
            "The last length_control value must equal sampling.max_new_tokens so the "
            "full-length result is reported."
        )

    report = _mapping(raw.get("report"), "report")
    raw_metrics = report.get("metrics")
    if not isinstance(raw_metrics, list) or tuple(raw_metrics) != REQUIRED_METRICS:
        raise ValueError(
            "report.metrics must be exactly: " + ", ".join(REQUIRED_METRICS)
        )

    runtime = _mapping(raw.get("runtime"), "runtime")
    raw_gpus = runtime.get("gpus")
    if not isinstance(raw_gpus, list) or not raw_gpus:
        raise ValueError("runtime.gpus must be a non-empty list.")
    gpus = tuple(
        _integer(value, f"runtime.gpus[{index}]", minimum=0)
        for index, value in enumerate(raw_gpus)
    )
    if len(set(gpus)) != len(gpus):
        raise ValueError("runtime.gpus must contain distinct GPU IDs.")
    tensor_parallel_size = _integer(
        runtime.get("tensor_parallel_size"), "runtime.tensor_parallel_size", minimum=1
    )
    if tensor_parallel_size != 1:
        raise ValueError(
            "runtime.tensor_parallel_size must be 1 because evaluation uses one worker per GPU."
        )
    rollouts_per_gpu_batch = _integer(
        runtime.get("rollouts_per_gpu_batch"),
        "runtime.rollouts_per_gpu_batch",
        minimum=1,
    )
    if rollouts_per_gpu_batch % n:
        raise ValueError(
            "runtime.rollouts_per_gpu_batch must be divisible by sampling.n; "
            f"got {rollouts_per_gpu_batch} and n={n}."
        )
    if source_type == DIRECT_MODEL_SOURCE:
        if len(gpus) != 8:
            raise ValueError(
                "Direct-model evaluation requires exactly eight independent GPU "
                f"workers; got {len(gpus)} GPU IDs."
            )
        if rollouts_per_gpu_batch != 256:
            raise ValueError(
                "Direct-model evaluation requires exactly 256 rollouts per GPU "
                f"generate call; got {rollouts_per_gpu_batch}."
            )

    engine = _mapping(raw.get("engine"), "engine")
    dtype = engine.get("dtype")
    if not isinstance(dtype, str) or not dtype:
        raise ValueError("engine.dtype must be a non-empty string.")
    trust_remote_code = engine.get("trust_remote_code")
    if not isinstance(trust_remote_code, bool):
        raise ValueError("engine.trust_remote_code must be true or false.")
    gpu_memory_utilization = _number(
        engine.get("gpu_memory_utilization"), "engine.gpu_memory_utilization"
    )
    if not 0 < gpu_memory_utilization < 1:
        raise ValueError("engine.gpu_memory_utilization must be in (0, 1).")
    max_model_len = _integer(engine.get("max_model_len"), "engine.max_model_len", minimum=1)
    if max_model_len <= max_new_tokens:
        raise ValueError(
            "engine.max_model_len must exceed sampling.max_new_tokens to leave room for prompts."
        )
    enable_prefix_caching = engine.get("enable_prefix_caching")
    if not isinstance(enable_prefix_caching, bool):
        raise ValueError("engine.enable_prefix_caching must be true or false.")

    result_dir = _resolve_project_path(
        raw.get("result_dir", "./eval_results"), project_root, "result_dir"
    )
    required_result_dir = (project_root / "eval_results").resolve()
    if result_dir != required_result_dir:
        raise ValueError(
            f"result_dir must resolve to {required_result_dir}, got {result_dir}."
        )

    return EvaluationConfig(
        config_path=config_path,
        project_root=project_root,
        source_type=source_type,
        training_output=training_output,
        models=models,
        run_name=run_name,
        datasets=tuple(datasets),
        prompt_name=prompt_name,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        n=n,
        max_new_tokens=max_new_tokens,
        length_control=length_control,
        gpus=gpus,
        tensor_parallel_size=tensor_parallel_size,
        rollouts_per_gpu_batch=rollouts_per_gpu_batch,
        dtype=dtype,
        trust_remote_code=trust_remote_code,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enable_prefix_caching=enable_prefix_caching,
        result_dir=result_dir,
    )


def _weight_index_is_complete(path: Path, index_name: str) -> bool:
    index_path = path / index_name
    if not index_path.is_file():
        return False
    try:
        with index_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        weight_map = payload["weight_map"]
        shard_names = set(weight_map.values())
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return bool(shard_names) and all(
        (path / shard).is_file() and (path / shard).stat().st_size > 0
        for shard in shard_names
    )


def is_huggingface_model_dir(path: Path) -> bool:
    """Return whether ``path`` contains config plus complete HF weight files."""

    if not path.is_dir() or not (path / "config.json").is_file():
        return False
    index_names = (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    )
    present_indexes = [name for name in index_names if (path / name).is_file()]
    if present_indexes:
        return any(_weight_index_is_complete(path, name) for name in present_indexes)
    weight_files = [
        *path.glob("*.safetensors"),
        *path.glob("pytorch_model*.bin"),
    ]
    return bool(weight_files) and all(item.stat().st_size > 0 for item in weight_files)


def _validate_fsdp_source(path: Path) -> None:
    config_path = path / "fsdp_config.json"
    metadata_dir = path / "huggingface"
    if not config_path.is_file() or not (metadata_dir / "config.json").is_file():
        raise ValueError(f"Incomplete FSDP actor metadata: {path}")
    try:
        with config_path.open(encoding="utf-8") as handle:
            world_size = _integer(json.load(handle).get("world_size"), "FSDP world_size", minimum=1)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {config_path}: {exc}") from exc
    missing = [
        path / f"model_world_size_{world_size}_rank_{rank}.pt"
        for rank in range(world_size)
        if not (path / f"model_world_size_{world_size}_rank_{rank}.pt").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"FSDP actor {path} is missing model shards: "
            + ", ".join(item.name for item in missing)
        )


def _choose_hf_model(checkpoint_dir: Path) -> Path | None:
    candidates: list[Path] = []
    preferred = (
        checkpoint_dir,
        checkpoint_dir / "actor" / "merged_hf",
        checkpoint_dir / "merged_hf",
    )
    for path in preferred:
        if is_huggingface_model_dir(path) and path not in candidates:
            candidates.append(path)
    for config_path in checkpoint_dir.rglob("config.json"):
        path = config_path.parent
        if is_huggingface_model_dir(path) and path not in candidates:
            candidates.append(path)
    if not candidates:
        return None
    candidates.sort(
        key=lambda path: (
            0 if "actor" in path.relative_to(checkpoint_dir).parts else 1,
            len(path.relative_to(checkpoint_dir).parts),
            str(path),
        )
    )
    return candidates[0]


def _choose_fsdp_source(checkpoint_dir: Path) -> Path:
    candidates: list[Path] = []
    preferred = (checkpoint_dir / "actor", checkpoint_dir)
    for path in preferred:
        if (path / "fsdp_config.json").is_file() and path not in candidates:
            candidates.append(path)
    for config_path in checkpoint_dir.rglob("fsdp_config.json"):
        if config_path.parent not in candidates:
            candidates.append(config_path.parent)
    if not candidates:
        raise ValueError(
            f"No directly loadable HF model or FSDP actor shards found in {checkpoint_dir}."
        )
    actor_candidates = [path for path in candidates if path.name == "actor"]
    if len(actor_candidates) == 1:
        source = actor_candidates[0]
    elif len(candidates) == 1:
        source = candidates[0]
    else:
        raise ValueError(
            f"Ambiguous FSDP model sources in {checkpoint_dir}: "
            + ", ".join(str(path) for path in candidates)
        )
    _validate_fsdp_source(source)
    return source


def discover_checkpoints(config: EvaluationConfig) -> list[CheckpointSpec]:
    """Discover all numeric global_step directories and inspect their model format."""

    if config.source_type != CHECKPOINT_SOURCE or config.training_output is None:
        raise ValueError("Checkpoint discovery requires a checkpoint evaluation config.")
    step_directories: dict[int, Path] = {}
    for path in config.training_output.iterdir():
        match = CHECKPOINT_PATTERN.fullmatch(path.name)
        if not match or not path.is_dir():
            continue
        step_num = int(match.group(1))
        if step_num in step_directories:
            raise ValueError(
                f"Duplicate numeric checkpoint step {step_num}: "
                f"{step_directories[step_num]} and {path}"
            )
        step_directories[step_num] = path.resolve()
    if not step_directories:
        raise ValueError(
            f"No global_step_<integer> checkpoint directories found in {config.training_output}."
        )

    specs: list[CheckpointSpec] = []
    for step_num, checkpoint_dir in sorted(step_directories.items()):
        hf_model = _choose_hf_model(checkpoint_dir)
        if hf_model is not None:
            specs.append(
                CheckpointSpec(
                    step_num=step_num,
                    checkpoint_dir=checkpoint_dir,
                    source_kind="huggingface",
                    source_dir=hf_model,
                    tokenizer_dir=hf_model,
                    model_dir=hf_model,
                )
            )
            continue
        fsdp_source = _choose_fsdp_source(checkpoint_dir)
        specs.append(
            CheckpointSpec(
                step_num=step_num,
                checkpoint_dir=checkpoint_dir,
                source_kind="fsdp",
                source_dir=fsdp_source,
                tokenizer_dir=fsdp_source / "huggingface",
                model_dir=None,
            )
        )
    return specs


def load_tasks(config: EvaluationConfig) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Load all datasets and render complete prompts without tokenizing them."""

    tasks: list[dict[str, Any]] = []
    question_counts: dict[str, int] = {}
    for dataset in config.datasets:
        try:
            with dataset.path.open(encoding="utf-8") as handle:
                examples = json.load(handle)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON dataset {dataset.path}: {exc}") from exc
        if not isinstance(examples, list) or not examples:
            raise ValueError(f"Dataset must be a non-empty JSON list: {dataset.path}")
        question_counts[dataset.name] = len(examples)
        for question_index, example in enumerate(examples):
            if not isinstance(example, Mapping):
                raise ValueError(
                    f"{dataset.name}[{question_index}] must be a JSON object."
                )
            if "question" not in example or "answer" not in example:
                raise ValueError(
                    f"{dataset.name}[{question_index}] must contain question and answer."
                )
            question = str(example["question"])
            if not question.strip():
                raise ValueError(f"{dataset.name}[{question_index}].question is empty.")
            tasks.append(
                {
                    "dataset": dataset.name,
                    "question_index": question_index,
                    "answer": str(example["answer"]),
                    "prompt": render_prompt(config.prompt_name, question=question),
                }
            )
    return tasks, question_counts


def make_batches(
    tasks: Sequence[dict[str, Any]], questions_per_batch: int
) -> list[list[dict[str, Any]]]:
    if questions_per_batch < 1:
        raise ValueError("questions_per_batch must be positive.")
    if not tasks:
        raise ValueError("Cannot batch an empty task list.")
    return [
        list(tasks[start : start + questions_per_batch])
        for start in range(0, len(tasks), questions_per_batch)
    ]


def validate_prompt_budget(
    config: EvaluationConfig,
    tasks: Sequence[dict[str, Any]],
    tokenizer_dir: Path,
) -> int:
    """Tokenize prompts on CPU and ensure the configured context budget is valid."""

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_dir,
        trust_remote_code=config.trust_remote_code,
        local_files_only=True,
        # Qwen's tokenizer intentionally uses its own regex; explicitly keep
        # it instead of triggering Transformers' Mistral-regex auto-warning.
        fix_mistral_regex=False,
    )
    max_prompt_tokens = 0
    max_prompt_task: dict[str, Any] | None = None
    for task in tasks:
        prompt_tokens = len(tokenizer.encode(task["prompt"], add_special_tokens=False))
        if prompt_tokens > max_prompt_tokens:
            max_prompt_tokens = prompt_tokens
            max_prompt_task = task
    available_prompt_tokens = config.max_model_len - config.max_new_tokens
    if max_prompt_tokens > available_prompt_tokens:
        assert max_prompt_task is not None
        raise ValueError(
            f"Longest prompt has {max_prompt_tokens} tokens but only "
            f"{available_prompt_tokens} fit before max_new_tokens; offending task is "
            f"{max_prompt_task['dataset']}[{max_prompt_task['question_index']}]."
        )

    model_config_path = tokenizer_dir / "config.json"
    if model_config_path.is_file():
        with model_config_path.open(encoding="utf-8") as handle:
            model_config = json.load(handle)
        model_context = model_config.get("max_position_embeddings")
        if isinstance(model_context, int) and config.max_model_len > model_context:
            raise ValueError(
                f"engine.max_model_len={config.max_model_len} exceeds model context "
                f"max_position_embeddings={model_context}."
            )
    return max_prompt_tokens


@contextmanager
def prepare_model(
    config: EvaluationConfig, checkpoint: CheckpointSpec
) -> Iterator[Path]:
    """Yield a loadable model and remove any temporary FSDP merge afterward."""

    if checkpoint.model_dir is not None:
        yield checkpoint.model_dir
        return
    if checkpoint.source_kind != "fsdp":
        raise ValueError(f"Unsupported checkpoint source kind: {checkpoint.source_kind}")
    if config.training_output is None:
        raise ValueError("FSDP checkpoint merging requires training_output.")

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".eval_merged_global_step_{checkpoint.step_num}.",
            dir=config.training_output,
        )
    )
    merged_model_dir = temporary_root / "model"
    command = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        "fsdp",
        "--local_dir",
        str(checkpoint.source_dir),
        "--target_dir",
        str(merged_model_dir),
        "--trust-remote-code",
        "--use_cpu_initialization",
    ]
    environment = os.environ.copy()
    python_paths = [str(config.project_root / "verl"), str(config.project_root)]
    if environment.get("PYTHONPATH"):
        python_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    try:
        print(
            f"MERGE step={checkpoint.step_num} source={checkpoint.source_dir} "
            f"target={merged_model_dir}",
            flush=True,
        )
        subprocess.run(command, cwd=config.project_root, env=environment, check=True)
        if not is_huggingface_model_dir(merged_model_dir):
            raise RuntimeError(
                "Model merger finished but did not create complete HF weights: "
                f"{merged_model_dir}"
            )
        yield merged_model_dir
    finally:
        shutil.rmtree(temporary_root)
        print(
            f"MERGED_MODEL_REMOVED step={checkpoint.step_num} path={temporary_root}",
            flush=True,
        )


def score_completion_at_lengths(
    *,
    token_ids: Sequence[int],
    full_text: str,
    finish_reason: str | None,
    answer: str,
    length_control: Sequence[int],
    decode: Callable[[Sequence[int]], str],
    verify: Callable[[str, str], float],
) -> dict[int, dict[str, int]]:
    """Score one generated completion at every post-hoc token limit."""

    token_ids = list(token_ids)
    full_correct: int | None = None
    values: dict[int, dict[str, int]] = {}
    for limit in length_control:
        cut_by_posthoc_limit = len(token_ids) > limit
        if cut_by_posthoc_limit:
            response = decode(token_ids[:limit])
            correct = int(bool(verify(response, answer)))
        else:
            if full_correct is None:
                full_correct = int(bool(verify(full_text, answer)))
            correct = full_correct
        values[int(limit)] = {
            "correct": correct,
            "tokens": min(len(token_ids), int(limit)),
            "truncated": int(cut_by_posthoc_limit or finish_reason == "length"),
        }
    return values


def aggregate_question_results(
    *,
    question_results: Sequence[Mapping[str, Any]],
    dataset_question_counts: Mapping[str, int],
    dataset_order: Sequence[str],
    length_control: Sequence[int],
    n: int,
) -> dict[str, dict[str, dict[str, float]]]:
    """Aggregate per-question counters into the four requested metrics."""

    expected_keys = {
        (dataset, question_index)
        for dataset, count in dataset_question_counts.items()
        for question_index in range(count)
    }
    actual_keys = [
        (str(record["dataset"]), int(record["question_index"]))
        for record in question_results
    ]
    if len(actual_keys) != len(set(actual_keys)):
        raise RuntimeError("Evaluation returned duplicate question results.")
    if set(actual_keys) != expected_keys:
        missing = sorted(expected_keys - set(actual_keys))[:10]
        unexpected = sorted(set(actual_keys) - expected_keys)[:10]
        raise RuntimeError(
            f"Incomplete question results: missing={missing}, unexpected={unexpected}."
        )

    avg_label = f"Avg@{n}"
    pass_label = f"Pass@{n}"
    summaries: dict[str, dict[str, dict[str, float]]] = {}
    for dataset in dataset_order:
        records = [record for record in question_results if record["dataset"] == dataset]
        question_count = dataset_question_counts[dataset]
        rollout_count = question_count * n
        length_summaries: dict[str, dict[str, float]] = {}
        for limit in length_control:
            correct_count = 0
            passed_questions = 0
            token_count = 0
            truncated_count = 0
            for record in records:
                counters = record["lengths"][str(limit)]
                per_question_rollouts = int(counters["rollouts"])
                if per_question_rollouts != n:
                    raise RuntimeError(
                        f"{dataset}[{record['question_index']}] at length {limit} has "
                        f"{per_question_rollouts} rollouts, expected {n}."
                    )
                per_question_correct = int(counters["correct_count"])
                if not 0 <= per_question_correct <= n:
                    raise RuntimeError("Invalid correct_count in worker result.")
                correct_count += per_question_correct
                passed_questions += int(per_question_correct > 0)
                token_count += int(counters["token_count"])
                truncated_count += int(counters["truncated_count"])
            length_summaries[str(limit)] = {
                avg_label: correct_count / rollout_count,
                pass_label: passed_questions / question_count,
                "mean_length": token_count / rollout_count,
                "truncation_rate": truncated_count / rollout_count,
            }
        summaries[dataset] = length_summaries
    return summaries


def _decode_prefix(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def _gpu_worker(
    gpu_id: int,
    model_dir: str,
    worker_config: Mapping[str, Any],
    job_queue: Any,
    result_queue: Any,
) -> None:
    """Load one vLLM replica on one visible GPU and consume question batches."""

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        from utils.math_verifier import verify_response_answer

        tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            trust_remote_code=bool(worker_config["trust_remote_code"]),
            local_files_only=True,
            fix_mistral_regex=False,
        )
        llm = LLM(
            model=model_dir,
            tensor_parallel_size=int(worker_config["tensor_parallel_size"]),
            dtype=str(worker_config["dtype"]),
            trust_remote_code=bool(worker_config["trust_remote_code"]),
            max_model_len=int(worker_config["max_model_len"]),
            max_num_seqs=int(worker_config["rollouts_per_gpu_batch"]),
            gpu_memory_utilization=float(worker_config["gpu_memory_utilization"]),
            enable_prefix_caching=bool(worker_config["enable_prefix_caching"]),
            seed=int(worker_config["seed"]),
        )
        sampling_params = SamplingParams(
            n=int(worker_config["n"]),
            temperature=float(worker_config["temperature"]),
            top_p=float(worker_config["top_p"]),
            top_k=-1,
            max_tokens=int(worker_config["max_new_tokens"]),
            seed=int(worker_config["seed"]),
            skip_special_tokens=True,
        )
        result_queue.put(
            {"kind": "ready", "ok": True, "gpu": gpu_id, "time": utc_now()}
        )
    except BaseException as exc:
        result_queue.put(
            {
                "kind": "ready",
                "ok": False,
                "gpu": gpu_id,
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
            questions_per_batch = int(worker_config["questions_per_gpu_batch"])
            if not 1 <= len(real_tasks) <= questions_per_batch:
                raise ValueError(f"Invalid real question batch size: {len(real_tasks)}")
            padded_prompts = [task["prompt"] for task in real_tasks]
            padding_prompt = real_tasks[-1]["prompt"]
            padded_prompts.extend(
                padding_prompt for _ in range(questions_per_batch - len(real_tasks))
            )
            generated_rollouts = len(padded_prompts) * int(worker_config["n"])
            if generated_rollouts != int(worker_config["rollouts_per_gpu_batch"]):
                raise AssertionError(
                    f"Each GPU call must contain exactly "
                    f"{worker_config['rollouts_per_gpu_batch']} rollouts; got {generated_rollouts}."
                )

            started = time.monotonic()
            outputs = llm.generate(padded_prompts, sampling_params, use_tqdm=False)
            if len(outputs) != questions_per_batch:
                raise RuntimeError(
                    f"Expected {questions_per_batch} request outputs, got {len(outputs)}."
                )

            question_results: list[dict[str, Any]] = []
            for task, request_output in zip(real_tasks, outputs[: len(real_tasks)], strict=True):
                completions = request_output.outputs
                if len(completions) != int(worker_config["n"]):
                    raise RuntimeError(
                        f"{task['dataset']}[{task['question_index']}] returned "
                        f"{len(completions)} rollouts, expected {worker_config['n']}."
                    )
                counters = {
                    str(limit): {
                        "rollouts": 0,
                        "correct_count": 0,
                        "token_count": 0,
                        "truncated_count": 0,
                    }
                    for limit in worker_config["length_control"]
                }
                for completion in completions:
                    raw_finish_reason = completion.finish_reason
                    finish_reason = (
                        None
                        if raw_finish_reason is None
                        else str(getattr(raw_finish_reason, "value", raw_finish_reason))
                    )
                    scored = score_completion_at_lengths(
                        token_ids=completion.token_ids,
                        full_text=completion.text,
                        finish_reason=finish_reason,
                        answer=task["answer"],
                        length_control=worker_config["length_control"],
                        decode=lambda ids: _decode_prefix(tokenizer, ids),
                        verify=verify_response_answer,
                    )
                    for limit, values in scored.items():
                        target = counters[str(limit)]
                        target["rollouts"] += 1
                        target["correct_count"] += values["correct"]
                        target["token_count"] += values["tokens"]
                        target["truncated_count"] += values["truncated"]
                question_results.append(
                    {
                        "dataset": task["dataset"],
                        "question_index": task["question_index"],
                        "lengths": counters,
                    }
                )
            result_queue.put(
                {
                    "kind": "batch",
                    "ok": True,
                    "gpu": gpu_id,
                    "batch_id": int(job["batch_id"]),
                    "elapsed_seconds": time.monotonic() - started,
                    "questions": question_results,
                }
            )
        except BaseException as exc:
            result_queue.put(
                {
                    "kind": "batch",
                    "ok": False,
                    "gpu": gpu_id,
                    "batch_id": job.get("batch_id"),
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            return


def _unexpected_worker_exits(processes: Sequence[mp.Process]) -> str | None:
    exited = [
        f"pid={process.pid}, exitcode={process.exitcode}"
        for process in processes
        if process.exitcode is not None
    ]
    return "; ".join(exited) if exited else None


def _await_messages(
    *,
    result_queue: Any,
    processes: Sequence[mp.Process],
    expected_kind: str,
    count: int,
    progress_label: str,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    last_heartbeat = time.monotonic()
    while len(messages) < count:
        try:
            message = result_queue.get(timeout=15)
        except queue.Empty:
            unexpected_exits = _unexpected_worker_exits(processes)
            if unexpected_exits:
                raise RuntimeError(
                    f"GPU workers exited before all {expected_kind} messages arrived at "
                    f"{progress_label}: {unexpected_exits}"
                )
            if time.monotonic() - last_heartbeat >= 60:
                print(
                    f"HEARTBEAT {progress_label} kind={expected_kind} "
                    f"remaining={count - len(messages)} time={utc_now()}",
                    flush=True,
                )
                last_heartbeat = time.monotonic()
            continue
        if message.get("kind") != expected_kind:
            raise RuntimeError(f"Unexpected worker message: {message}")
        if not message.get("ok"):
            raise RuntimeError(
                f"GPU {message.get('gpu')} failed at {progress_label}: "
                f"{message.get('error')}\n{message.get('traceback', '')}"
            )
        messages.append(message)
        if expected_kind == "ready":
            print(
                f"READY {progress_label} gpu={message['gpu']} "
                f"workers={len(messages)}/{count}",
                flush=True,
            )
        else:
            print(
                f"BATCH_DONE {progress_label} batch={message['batch_id']} "
                f"gpu={message['gpu']} elapsed={message['elapsed_seconds']:.1f}s "
                f"completed={len(messages)}/{count}",
                flush=True,
            )
    return messages


def _stop_workers(
    job_queue: Any,
    processes: Sequence[mp.Process],
    *,
    force: bool,
) -> None:
    if force:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)
        return
    for _ in processes:
        job_queue.put(None)
    for process in processes:
        process.join(timeout=30)
    for process in processes:
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)


def _worker_config(config: EvaluationConfig) -> dict[str, Any]:
    return {
        "seed": config.seed,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "n": config.n,
        "max_new_tokens": config.max_new_tokens,
        "length_control": list(config.length_control),
        "tensor_parallel_size": config.tensor_parallel_size,
        "rollouts_per_gpu_batch": config.rollouts_per_gpu_batch,
        "questions_per_gpu_batch": config.questions_per_gpu_batch,
        "dtype": config.dtype,
        "trust_remote_code": config.trust_remote_code,
        "gpu_memory_utilization": config.gpu_memory_utilization,
        "max_model_len": config.max_model_len,
        "enable_prefix_caching": config.enable_prefix_caching,
    }


def evaluate_model(
    *,
    config: EvaluationConfig,
    model_dir: Path,
    batches: Sequence[Sequence[dict[str, Any]]],
    dataset_question_counts: Mapping[str, int],
    progress_label: str,
    process_name_prefix: str,
    start_event: str,
) -> dict[str, dict[str, dict[str, float]]]:
    """Evaluate one loadable model with one independent full replica per GPU."""

    context = mp.get_context("spawn")
    job_queue = context.Queue()
    result_queue = context.Queue()
    payload = _worker_config(config)
    processes = [
        context.Process(
            target=_gpu_worker,
            args=(gpu_id, str(model_dir), payload, job_queue, result_queue),
            name=f"{process_name_prefix}-gpu{gpu_id}",
        )
        for gpu_id in config.gpus
    ]
    print(
        f"{start_event} run={config.run_name} {progress_label} "
        f"gpus={list(config.gpus)} batches={len(batches)} "
        f"rollouts_per_gpu_call={config.rollouts_per_gpu_batch}",
        flush=True,
    )
    for process in processes:
        process.start()
    completed = False
    try:
        _await_messages(
            result_queue=result_queue,
            processes=processes,
            expected_kind="ready",
            count=len(processes),
            progress_label=progress_label,
        )
        for batch_id, batch in enumerate(batches):
            job_queue.put({"batch_id": batch_id, "tasks": list(batch)})
        batch_messages = _await_messages(
            result_queue=result_queue,
            processes=processes,
            expected_kind="batch",
            count=len(batches),
            progress_label=progress_label,
        )
        completed = True
    finally:
        _stop_workers(job_queue, processes, force=not completed)

    batch_messages.sort(key=lambda message: int(message["batch_id"]))
    question_results = [
        question
        for message in batch_messages
        for question in message["questions"]
    ]
    return aggregate_question_results(
        question_results=question_results,
        dataset_question_counts=dataset_question_counts,
        dataset_order=[dataset.name for dataset in config.datasets],
        length_control=config.length_control,
        n=config.n,
    )


def evaluate_checkpoint(
    *,
    config: EvaluationConfig,
    checkpoint: CheckpointSpec,
    model_dir: Path,
    batches: Sequence[Sequence[dict[str, Any]]],
    dataset_question_counts: Mapping[str, int],
) -> dict[str, dict[str, dict[str, float]]]:
    """Evaluate one checkpoint through the shared loadable-model engine."""

    return evaluate_model(
        config=config,
        model_dir=model_dir,
        batches=batches,
        dataset_question_counts=dataset_question_counts,
        progress_label=f"step={checkpoint.step_num}",
        process_name_prefix=f"eval-step{checkpoint.step_num}",
        start_event="STEP_START",
    )


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=4)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def build_validation_plan(
    config: EvaluationConfig,
    checkpoints: Sequence[CheckpointSpec],
    tasks: Sequence[dict[str, Any]],
    dataset_question_counts: Mapping[str, int],
    max_prompt_tokens: int,
) -> dict[str, Any]:
    batches = make_batches(tasks, config.questions_per_gpu_batch)
    return {
        "config": str(config.config_path),
        "training_output": str(config.training_output),
        "run_name": config.run_name,
        "checkpoints": [
            {
                "step_num": checkpoint.step_num,
                "source_kind": checkpoint.source_kind,
                "source_dir": str(checkpoint.source_dir),
            }
            for checkpoint in checkpoints
        ],
        "datasets": dict(dataset_question_counts),
        "total_questions": len(tasks),
        "max_prompt_tokens": max_prompt_tokens,
        "sampling": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "n": config.n,
            "max_new_tokens": config.max_new_tokens,
        },
        "length_control": list(config.length_control),
        "gpus": list(config.gpus),
        "questions_per_gpu_call": config.questions_per_gpu_batch,
        "rollouts_per_gpu_call": config.rollouts_per_gpu_batch,
        "batches_per_checkpoint": len(batches),
        "result_path": str(config.result_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate YAML, datasets, prompts, and checkpoint shards without importing vLLM.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_evaluation_config(args.config)
    checkpoints = discover_checkpoints(config)
    tasks, dataset_question_counts = load_tasks(config)
    max_prompt_tokens = validate_prompt_budget(config, tasks, checkpoints[0].tokenizer_dir)
    batches = make_batches(tasks, config.questions_per_gpu_batch)
    plan = build_validation_plan(
        config,
        checkpoints,
        tasks,
        dataset_question_counts,
        max_prompt_tokens,
    )
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.validate_only:
        print("VALIDATION_OK: no model was merged and no GPU worker was started.", flush=True)
        return

    config.result_dir.mkdir(parents=True, exist_ok=True)
    result_document: dict[str, Any] = {"run_name": config.run_name, "steps": []}
    atomic_write_json(config.result_path, result_document)
    for checkpoint in checkpoints:
        with prepare_model(config, checkpoint) as model_dir:
            datasets = evaluate_checkpoint(
                config=config,
                checkpoint=checkpoint,
                model_dir=model_dir,
                batches=batches,
                dataset_question_counts=dataset_question_counts,
            )
        result_document["steps"].append(
            {"step_num": checkpoint.step_num, "datasets": datasets}
        )
        atomic_write_json(config.result_path, result_document)
        print(
            f"STEP_DONE run={config.run_name} step={checkpoint.step_num} "
            f"result={config.result_path}",
            flush=True,
        )
    print(f"EVALUATION_DONE result={config.result_path}", flush=True)


if __name__ == "__main__":
    main()
