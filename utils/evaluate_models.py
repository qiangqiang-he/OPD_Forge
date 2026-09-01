#!/usr/bin/env python3
"""Evaluate explicitly configured local Hugging Face models with vLLM.

This entry point reuses the checkpoint evaluator's prompt construction, GPU
workers, post-hoc token-length scoring, verification, and metric aggregation.
Only model discovery differs: every configured path must already be a complete
Hugging Face model directory, so no checkpoint merge is performed.

All vLLM and CUDA imports remain inside spawned GPU workers. Consequently,
``--validate-only`` performs model-path, tokenizer, dataset, prompt-budget, and
runtime validation entirely on CPU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from utils.evaluate_checkpoints import (
    DIRECT_MODEL_SOURCE,
    HF_MODELS_TYPE,
    PROJECT_ROOT,
    EvaluationConfig,
    ModelSpec,
    atomic_write_json,
    evaluate_model,
    load_evaluation_config,
    load_tasks,
    make_batches,
    utc_now,
    validate_prompt_budget,
)


def model_result_dir(config: EvaluationConfig) -> Path:
    return config.result_dir / config.run_name


def model_result_path(config: EvaluationConfig, model: ModelSpec) -> Path:
    return model_result_dir(config) / f"{model.name}.json"


def summary_result_path(config: EvaluationConfig) -> Path:
    return model_result_dir(config) / "summary.json"


def _evaluation_metadata(config: EvaluationConfig) -> dict[str, Any]:
    return {
        "prompt": config.prompt_name,
        "seed": config.seed,
        "sampling": {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "n": config.n,
            "max_new_tokens": config.max_new_tokens,
        },
        "length_control": list(config.length_control),
        "runtime": {
            "gpus": list(config.gpus),
            "tensor_parallel_size": config.tensor_parallel_size,
            "rollouts_per_gpu_batch": config.rollouts_per_gpu_batch,
            "questions_per_gpu_batch": config.questions_per_gpu_batch,
            "full_model_replica_per_gpu": True,
        },
        "engine": {
            "dtype": config.dtype,
            "trust_remote_code": config.trust_remote_code,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "max_model_len": config.max_model_len,
            "enable_prefix_caching": config.enable_prefix_caching,
        },
    }


def build_model_validation_plan(
    config: EvaluationConfig,
    *,
    tasks: Sequence[dict[str, Any]],
    dataset_question_counts: Mapping[str, int],
    max_prompt_tokens_by_model: Mapping[str, int],
) -> dict[str, Any]:
    """Build the CPU-only execution plan printed before evaluation."""

    batches = make_batches(tasks, config.questions_per_gpu_batch)
    return {
        "config": str(config.config_path),
        "run_name": config.run_name,
        "models": [
            {
                "name": model.name,
                "path": str(model.path),
                "max_prompt_tokens": int(max_prompt_tokens_by_model[model.name]),
                "result_path": str(model_result_path(config, model)),
            }
            for model in config.models
        ],
        "datasets": dict(dataset_question_counts),
        "total_questions": len(tasks),
        **_evaluation_metadata(config),
        "batches_per_model": len(batches),
        "summary_path": str(summary_result_path(config)),
    }


def _initial_summary(config: EvaluationConfig) -> dict[str, Any]:
    now = utc_now()
    return {
        "run_name": config.run_name,
        "source_type": HF_MODELS_TYPE,
        "status": "running",
        "started_at": now,
        "updated_at": now,
        **_evaluation_metadata(config),
        "models": [
            {
                "name": model.name,
                "path": str(model.path),
                "result_path": str(model_result_path(config, model)),
                "status": "pending",
            }
            for model in config.models
        ],
    }


def _set_model_status(
    summary: dict[str, Any],
    model_name: str,
    status: str,
    *,
    error: str | None = None,
) -> None:
    for item in summary["models"]:
        if item["name"] != model_name:
            continue
        item["status"] = status
        item["updated_at"] = utc_now()
        if error is None:
            item.pop("error", None)
        else:
            item["error"] = error
        summary["updated_at"] = item["updated_at"]
        return
    raise KeyError(f"Unknown model in result summary: {model_name}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate YAML, local HF models, tokenizers, datasets, prompts, and "
            "context budgets without importing vLLM or starting GPU workers."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_evaluation_config(
        args.config,
        project_root=PROJECT_ROOT,
        source_type=DIRECT_MODEL_SOURCE,
    )
    tasks, dataset_question_counts = load_tasks(config)
    batches = make_batches(tasks, config.questions_per_gpu_batch)

    max_prompt_tokens_by_model = {
        model.name: validate_prompt_budget(config, tasks, model.path)
        for model in config.models
    }
    plan = build_model_validation_plan(
        config,
        tasks=tasks,
        dataset_question_counts=dataset_question_counts,
        max_prompt_tokens_by_model=max_prompt_tokens_by_model,
    )
    print(json.dumps(plan, ensure_ascii=False, indent=2), flush=True)
    if args.validate_only:
        print(
            "VALIDATION_OK: no model was loaded by vLLM and no GPU worker was started.",
            flush=True,
        )
        return

    output_dir = model_result_dir(config)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Direct-model result directory is not empty: {output_dir}. "
            "Use a new run_name so results from different attempts cannot be mixed."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_result_path(config)
    summary = _initial_summary(config)
    atomic_write_json(summary_path, summary)

    for model in config.models:
        _set_model_status(summary, model.name, "running")
        atomic_write_json(summary_path, summary)
        try:
            datasets = evaluate_model(
                config=config,
                model_dir=model.path,
                batches=batches,
                dataset_question_counts=dataset_question_counts,
                progress_label=f"model={model.name}",
                process_name_prefix=f"eval-model-{model.name}",
                start_event="MODEL_START",
            )
            result_document = {
                "run_name": config.run_name,
                "model_name": model.name,
                "model_path": str(model.path),
                "completed_at": utc_now(),
                **_evaluation_metadata(config),
                "datasets": datasets,
            }
            atomic_write_json(model_result_path(config, model), result_document)
        except BaseException as exc:
            _set_model_status(summary, model.name, "failed", error=repr(exc))
            summary["status"] = "failed"
            atomic_write_json(summary_path, summary)
            raise

        _set_model_status(summary, model.name, "completed")
        atomic_write_json(summary_path, summary)
        print(
            f"MODEL_DONE run={config.run_name} model={model.name} "
            f"result={model_result_path(config, model)}",
            flush=True,
        )

    summary["status"] = "completed"
    summary["completed_at"] = utc_now()
    summary["updated_at"] = summary["completed_at"]
    atomic_write_json(summary_path, summary)
    print(f"EVALUATION_DONE summary={summary_path}", flush=True)


if __name__ == "__main__":
    main()
