"""Multi-GPU, multi-model, multi-dataset vLLM evaluation entrypoint."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from utils.math_verifier import verify_response_answer
from utils.prompts import render_prompt


def _named_path(item: Any, kind: str) -> tuple[str, Path]:
    if isinstance(item, str):
        path = Path(item)
        return path.stem, path
    value = OmegaConf.to_container(item, resolve=True) if DictConfig and not isinstance(item, dict) else item
    if not isinstance(value, dict) or "path" not in value:
        raise ValueError(f"Each {kind} must be a path string or a mapping containing path and optional name.")
    path = Path(str(value["path"]))
    return str(value.get("name") or path.stem), path


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _has_model_weights(path: Path) -> bool:
    patterns = ("*.safetensors", "pytorch_model*.bin")
    return path.is_dir() and any(next(path.glob(pattern), None) is not None for pattern in patterns)


def _resolve_model_path(path: Path) -> Path:
    """Accept either a HuggingFace model directory or a verl FSDP actor checkpoint."""
    path = path.resolve()
    if _has_model_weights(path):
        return path
    if (path / "fsdp_config.json").is_file() and (path / "huggingface" / "config.json").is_file():
        merged = path / "merged_hf"
        if not _has_model_weights(merged):
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "verl.model_merger",
                    "merge",
                    "--backend",
                    "fsdp",
                    "--local_dir",
                    str(path),
                    "--target_dir",
                    str(merged),
                    "--trust-remote-code",
                ],
                check=True,
            )
        return merged
    raise ValueError(f"Model path is neither a HuggingFace model nor a verl FSDP actor checkpoint: {path}")


def _load_dataset_tasks(data_items: list[Any], prompt_name: str) -> tuple[list[dict], list[str]]:
    tasks: list[dict] = []
    dataset_names: list[str] = []
    for item in data_items:
        name, path = _named_path(item, "dataset")
        if name in dataset_names:
            raise ValueError(f"Duplicate dataset name: {name}")
        dataset_names.append(name)
        with path.open() as handle:
            examples = json.load(handle)
        for index, example in enumerate(examples):
            tasks.append(
                {
                    "dataset": name,
                    "index": index,
                    "answer": str(example["answer"]),
                    "prompt": render_prompt(prompt_name, question=str(example["question"])),
                }
            )
    return tasks, dataset_names


def _evaluate_model(gpu_id: int, model_item: Any, config: dict, result_queue) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    from vllm import LLM, SamplingParams

    model_name, configured_path = _named_path(model_item, "model")
    model_path = _resolve_model_path(configured_path)
    tasks, dataset_names = _load_dataset_tasks(config["datasets"], config["prompt"])
    n = int(config["sampling"]["n"])
    questions_per_batch = 128 // n
    if questions_per_batch < 1:
        raise ValueError(f"sampling.n={n} exceeds the per-GPU batch budget of 128 rollouts.")

    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=1,
        dtype=str(config["engine"]["dtype"]),
        trust_remote_code=bool(config["engine"]["trust_remote_code"]),
        max_model_len=int(config["engine"]["max_model_len"]),
        max_num_seqs=int(config["engine"]["max_num_seqs"]),
        gpu_memory_utilization=float(config["engine"]["gpu_memory_utilization"]),
        enable_prefix_caching=bool(config["engine"]["enable_prefix_caching"]),
        seed=int(config["seed"]),
    )
    sampling = SamplingParams(
        n=n,
        temperature=float(config["sampling"]["temperature"]),
        top_p=float(config["sampling"]["top_p"]),
        top_k=int(config["sampling"]["top_k"]),
        max_tokens=int(config["sampling"]["max_new_tokens"]),
        seed=int(config["seed"]),
    )

    stats = {name: {"correct": 0, "tokens": 0, "rollouts": 0} for name in dataset_names}
    rollout_handle = None
    if config["save_rollouts"]:
        rollout_dir = Path(config["output_dir"]) / "rollouts"
        rollout_dir.mkdir(parents=True, exist_ok=True)
        rollout_handle = (rollout_dir / f"{_slug(model_name)}.jsonl").open("w")
    try:
        for start in range(0, len(tasks), questions_per_batch):
            batch = tasks[start : start + questions_per_batch]
            outputs = llm.generate([task["prompt"] for task in batch], sampling)
            for task, request_output in zip(batch, outputs, strict=True):
                if len(request_output.outputs) != n:
                    raise RuntimeError(f"Expected {n} rollouts, received {len(request_output.outputs)}.")
                dataset_stats = stats[task["dataset"]]
                for sample_index, completion in enumerate(request_output.outputs):
                    correct = int(verify_response_answer(completion.text, task["answer"]))
                    token_count = len(completion.token_ids)
                    dataset_stats["correct"] += correct
                    dataset_stats["tokens"] += token_count
                    dataset_stats["rollouts"] += 1
                    if rollout_handle:
                        rollout_handle.write(
                            json.dumps(
                                {
                                    "model": model_name,
                                    "dataset": task["dataset"],
                                    "question_index": task["index"],
                                    "sample_index": sample_index,
                                    "correct": correct,
                                    "num_tokens": token_count,
                                    "response": completion.text,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
    finally:
        if rollout_handle:
            rollout_handle.close()

    summaries = []
    for dataset_name in dataset_names:
        values = stats[dataset_name]
        count = values["rollouts"]
        summaries.append(
            {
                "model": model_name,
                "dataset": dataset_name,
                f"Avg@{n}": values["correct"] / count,
                "mean_inference_length": values["tokens"] / count,
            }
        )
    result_queue.put({"ok": True, "gpu": gpu_id, "model": model_name, "results": summaries})


def _gpu_worker(gpu_id: int, model_queue, result_queue, config: dict) -> None:
    while True:
        model_item = model_queue.get()
        if model_item is None:
            return
        try:
            _evaluate_model(gpu_id, model_item, config, result_queue)
        except Exception as exc:
            result_queue.put({"ok": False, "gpu": gpu_id, "model": str(model_item), "error": repr(exc)})


@hydra.main(config_path=None, config_name=None, version_base=None)
def main(config: DictConfig) -> None:
    OmegaConf.resolve(config)
    config_data = OmegaConf.to_container(config, resolve=True)
    gpu_ids = [int(gpu) for gpu in config_data["gpus"]]
    models = list(config_data["models"])
    if not gpu_ids or not models or not config_data["datasets"]:
        raise ValueError("gpus, models, and datasets must all be non-empty.")

    output_value = config_data.get("output_dir")
    if output_value in (None, "", "null"):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        output_dir = Path("outputs") / f"summary_{stamp}"
    else:
        output_dir = Path(str(output_value))
    output_dir.mkdir(parents=True, exist_ok=True)
    config_data["output_dir"] = str(output_dir.resolve())

    context = mp.get_context("spawn")
    model_queue = context.Queue()
    result_queue = context.Queue()
    for model in models:
        model_queue.put(model)
    worker_count = min(len(gpu_ids), len(models))
    for _ in range(worker_count):
        model_queue.put(None)
    workers = [
        context.Process(target=_gpu_worker, args=(gpu_ids[index], model_queue, result_queue, config_data))
        for index in range(worker_count)
    ]
    for worker in workers:
        worker.start()

    results = []
    failures = []
    for _ in models:
        try:
            message = result_queue.get(timeout=None)
        except Empty:
            break
        if message["ok"]:
            results.extend(message["results"])
            print(f"GPU {message['gpu']} finished {message['model']}", flush=True)
        else:
            failures.append(message)
            print(f"GPU {message['gpu']} failed {message['model']}: {message['error']}", flush=True)
    for worker in workers:
        worker.join()

    results.sort(key=lambda row: (row["model"], row["dataset"]))
    with (output_dir / "eval.json").open("w") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2)
    if failures:
        with (output_dir / "failures.json").open("w") as handle:
            json.dump(failures, handle, ensure_ascii=False, indent=2)
        raise RuntimeError(f"{len(failures)} model evaluations failed; see {output_dir / 'failures.json'}")
    print(f"Evaluation summary: {output_dir / 'eval.json'}")


if __name__ == "__main__":
    main()
