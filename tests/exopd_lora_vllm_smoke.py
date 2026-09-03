"""Single-GPU vLLM smoke test for ExOPD's shared-base LoRA policy.

This test constructs a real PEFT adapter for the local Qwen3-1.7B checkpoint,
saves it as safetensors, and exercises vLLM twice on exactly the same prompt:

* no ``LoRARequest``: the frozen Student base/reference policy;
* rank-64 ``LoRARequest``: the trainable Student policy.

All adapter weights are deliberately non-zero so the chosen-token prompt
log-probabilities must change when vLLM applies the request.  The temporary
adapter is kept below ``tests/`` and removed after the vLLM engine shuts down.
No Teacher is loaded and no production configuration is modified.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = PROJECT_ROOT / "tests"
EXPECTED_MODEL_NAME = "Qwen3-1.7B"
EXPECTED_DENSE_SUFFIXES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}
LORA_RANK = 64
LORA_ALPHA = 128
MAX_ALLOWED_MODEL_LEN = 1024


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / EXPECTED_MODEL_NAME,
        help="Local Qwen3-1.7B safetensors checkpoint (never downloaded).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=256,
        help=f"Small local vLLM context, at most {MAX_ALLOWED_MODEL_LEN}.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.40,
        help="vLLM memory reservation fraction for the one visible GPU.",
    )
    parser.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> Path:
    model_path = args.model.expanduser().resolve()
    if model_path.name != EXPECTED_MODEL_NAME:
        raise ValueError(
            f"This local smoke requires {EXPECTED_MODEL_NAME}, got {model_path}."
        )
    if not model_path.is_dir():
        raise FileNotFoundError(f"Local Student checkpoint is missing: {model_path}")
    safetensors_files = sorted(model_path.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(
            f"No safetensors model weights were found under {model_path}."
        )
    if not 32 <= args.max_model_len <= MAX_ALLOWED_MODEL_LEN:
        raise ValueError(
            f"max-model-len must be in [32, {MAX_ALLOWED_MODEL_LEN}], got "
            f"{args.max_model_len}."
        )
    if not 0.10 <= args.gpu_memory_utilization <= 0.95:
        raise ValueError("gpu-memory-utilization must be in [0.10, 0.95].")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this real-vLLM smoke test.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Expose exactly one GPU for this local smoke; visible devices="
            f"{torch.cuda.device_count()}."
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The visible GPU must support bfloat16.")
    return model_path


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for package in ("torch", "transformers", "peft", "vllm", "safetensors"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "missing"
    return versions


def _gpu_memory_mib() -> dict[str, int | str] | None:
    """Read whole-device memory without coupling to vLLM's worker process."""

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    rows = [row.strip() for row in output.splitlines() if row.strip()]
    if len(rows) != 1:
        return None
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 4:
        return None
    try:
        return {
            "name": fields[0],
            "used": int(fields[1]),
            "free": int(fields[2]),
            "total": int(fields[3]),
        }
    except ValueError:
        return None


def _adapter_modules(model: Any) -> list[str]:
    return [
        name
        for name, module in model.named_modules()
        if hasattr(module, "lora_A") and "default" in module.lora_A
    ]


def _build_adapter(model_path: Path, adapter_path: Path, seed: int) -> dict[str, Any]:
    """Attach all-linear LoRA to local weights and save only the adapter."""

    from peft import LoraConfig, get_peft_model
    from safetensors import safe_open
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    try:
        if str(model.config.model_type) != "qwen3":
            raise AssertionError(
                f"Expected a Qwen3 checkpoint, got {model.config.model_type!r}."
            )
        model = get_peft_model(
            model,
            LoraConfig(
                r=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                target_modules="all-linear",
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
        module_names = _adapter_modules(model)
        suffixes = {name.rsplit(".", 1)[-1] for name in module_names}
        expected_count = int(model.config.num_hidden_layers) * len(
            EXPECTED_DENSE_SUFFIXES
        )
        if suffixes != EXPECTED_DENSE_SUFFIXES:
            raise AssertionError(
                "PEFT all-linear did not cover exactly the Qwen Transformer dense "
                f"projections: {sorted(suffixes)}."
            )
        if len(module_names) != expected_count:
            raise AssertionError(
                f"Expected {expected_count} adapted dense layers, got "
                f"{len(module_names)}."
            )
        if any(name.endswith("lm_head") for name in module_names):
            raise AssertionError("The output lm_head must not be adapted.")

        # PEFT initializes LoRA-B to zero.  Make both low-rank factors small,
        # deterministic, and non-zero so vLLM application is observable while
        # preserving numerically well-behaved logits.
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            for module_name in module_names:
                module = model.get_submodule(module_name)
                factor_a = module.lora_A["default"].weight
                factor_b = module.lora_B["default"].weight
                factor_a.copy_(
                    torch.randn(
                        factor_a.shape,
                        generator=generator,
                        dtype=torch.float32,
                    ).to(factor_a.dtype)
                    * 1.0e-2
                )
                factor_b.copy_(
                    torch.randn(
                        factor_b.shape,
                        generator=generator,
                        dtype=torch.float32,
                    ).to(factor_b.dtype)
                    * 2.0e-3
                )

        model.save_pretrained(
            adapter_path,
            safe_serialization=True,
            save_embedding_layers=False,
        )
        config_path = adapter_path / "adapter_config.json"
        weights_path = adapter_path / "adapter_model.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            raise AssertionError("PEFT did not save the expected adapter artifacts.")
        adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
        if int(adapter_config["r"]) != LORA_RANK:
            raise AssertionError("Saved adapter rank is not 64.")
        if int(adapter_config["lora_alpha"]) != LORA_ALPHA:
            raise AssertionError("Saved adapter alpha is not 128.")
        if float(adapter_config["lora_dropout"]) != 0.0:
            raise AssertionError("Saved adapter dropout is not zero.")
        saved_targets = set(adapter_config["target_modules"])
        if saved_targets != EXPECTED_DENSE_SUFFIXES:
            raise AssertionError(
                f"Saved adapter targets are incomplete: {sorted(saved_targets)}."
            )

        with safe_open(weights_path, framework="pt", device="cpu") as handle:
            tensor_keys = list(handle.keys())
            tensor_dtypes = {str(handle.get_tensor(key).dtype) for key in tensor_keys}
        expected_tensor_count = expected_count * 2
        if len(tensor_keys) != expected_tensor_count:
            raise AssertionError(
                f"Expected {expected_tensor_count} A/B tensors, got "
                f"{len(tensor_keys)}."
            )
        if not all(
            ".lora_A.weight" in key or ".lora_B.weight" in key
            for key in tensor_keys
        ):
            raise AssertionError("Saved adapter contains a non-LoRA tensor.")

        adapter_parameters = sum(
            parameter.numel()
            for name, parameter in model.named_parameters()
            if "lora_" in name
        )
        return {
            "dense_modules": len(module_names),
            "adapter_parameters": int(adapter_parameters),
            "adapter_size_mib": weights_path.stat().st_size / (1024**2),
            "adapter_tensor_dtypes": sorted(tensor_dtypes),
            "target_modules": sorted(saved_targets),
        }
    finally:
        del model
        gc.collect()


def _thinking_prompt_ids(tokenizer: Any, max_model_len: int) -> list[int]:
    messages = [
        {"role": "system", "content": "You are a helpful math assistant."},
        {
            "role": "user",
            "content": "Solve 2x + 3 = 11. Explain the algebra briefly.",
        },
    ]
    prompt_ids = list(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    )
    if len(prompt_ids) + 1 > max_model_len:
        raise ValueError(
            f"Rendered prompt ({len(prompt_ids)} tokens) exceeds max-model-len "
            f"{max_model_len}."
        )
    if len(prompt_ids) < 2:
        raise AssertionError("Tokenizer produced an unexpectedly short prompt.")
    return [int(token_id) for token_id in prompt_ids]


def _chosen_prompt_logprobs(output: Any, token_ids: list[int]) -> list[float]:
    if list(output.prompt_token_ids) != token_ids:
        raise AssertionError("vLLM changed the supplied original prompt token IDs.")
    rows = output.prompt_logprobs
    if rows is None or len(rows) != len(token_ids):
        raise AssertionError("vLLM did not return one prompt-logprob row per token.")
    chosen: list[float] = []
    # Causal log-probability for token i is represented at prompt row i; row
    # zero has no preceding context and is correctly None.
    for position, token_id in enumerate(token_ids[1:], start=1):
        row = rows[position]
        if row is None or token_id not in row:
            raise AssertionError(
                f"vLLM omitted chosen token {token_id} at prompt position {position}."
            )
        value = float(row[token_id].logprob)
        if not math.isfinite(value) or value > 1.0e-5:
            raise AssertionError(
                f"Invalid chosen-token log-probability {value} at {position}."
            )
        chosen.append(value)
    return chosen


def _shutdown_vllm(llm: Any) -> None:
    """Stop vLLM 0.17's engine subprocess before deleting the adapter."""

    engine = getattr(llm, "llm_engine", None)
    engine_core = getattr(engine, "engine_core", None)
    if engine_core is not None and hasattr(engine_core, "shutdown"):
        engine_core.shutdown()


def _run_vllm(
    model_path: Path,
    adapter_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    prompt_ids = _thinking_prompt_ids(tokenizer, args.max_model_len)
    del tokenizer

    memory_before = _gpu_memory_mib()
    load_started = time.perf_counter()
    llm = LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        runner="generate",
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
        load_format="safetensors",
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_model_len,
        max_num_seqs=1,
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
        enforce_eager=True,
        disable_log_stats=True,
        enable_lora=True,
        max_loras=1,
        max_cpu_loras=1,
        max_lora_rank=LORA_RANK,
    )
    load_seconds = time.perf_counter() - load_started
    try:
        memory_loaded = _gpu_memory_mib()
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=1,
            prompt_logprobs=1,
            logprobs=1,
            seed=args.seed,
        )
        prompt = {"prompt_token_ids": prompt_ids}
        base_result = llm.generate(
            [prompt], sampling_params=sampling, use_tqdm=False
        )[0]
        adapter_result = llm.generate(
            [prompt],
            sampling_params=sampling,
            use_tqdm=False,
            lora_request=LoRARequest(
                lora_name="exopd_r64_policy",
                lora_int_id=1,
                lora_path=str(adapter_path),
            ),
        )[0]
        memory_generated = _gpu_memory_mib()

        base_logprobs = _chosen_prompt_logprobs(base_result, prompt_ids)
        adapter_logprobs = _chosen_prompt_logprobs(adapter_result, prompt_ids)
        if len(base_logprobs) != len(adapter_logprobs):
            raise AssertionError("Base and adapter prompt scores are not aligned.")
        deltas = [
            abs(adapter - base)
            for base, adapter in zip(base_logprobs, adapter_logprobs, strict=True)
        ]
        changed = sum(delta > 1.0e-6 for delta in deltas)
        max_delta = max(deltas, default=0.0)
        if changed == 0:
            raise AssertionError(
                "vLLM accepted the LoRARequest but non-zero adapter weights did "
                "not change any aligned prompt-token log-probability."
            )

        base_completion = base_result.outputs[0]
        adapter_completion = adapter_result.outputs[0]
        return {
            "prompt_tokens": len(prompt_ids),
            "scored_tokens": len(base_logprobs),
            "changed_prompt_logprobs": changed,
            "max_abs_logprob_delta": max_delta,
            "mean_abs_logprob_delta": sum(deltas) / len(deltas),
            "base_generated_token_ids": [int(x) for x in base_completion.token_ids],
            "adapter_generated_token_ids": [
                int(x) for x in adapter_completion.token_ids
            ],
            "load_seconds": load_seconds,
            "gpu_memory_before_mib": memory_before,
            "gpu_memory_loaded_mib": memory_loaded,
            "gpu_memory_generated_mib": memory_generated,
        }
    finally:
        _shutdown_vllm(llm)
        del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    # Explicit offline mode makes an accidental Hub fallback fail instead of
    # silently downloading a checkpoint or tokenizer.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    model_path = _validate_args(args)
    started = time.perf_counter()
    temporary_root: Path | None = None
    with tempfile.TemporaryDirectory(
        dir=TESTS_ROOT, prefix=".exopd_lora_vllm_"
    ) as temporary_directory:
        temporary_root = Path(temporary_directory)
        adapter_path = temporary_root / "adapter"
        adapter_summary = _build_adapter(model_path, adapter_path, args.seed)
        # The full HF model has been released before vLLM claims GPU memory.
        gc.collect()
        vllm_summary = _run_vllm(model_path, adapter_path, args)

    if temporary_root is None or temporary_root.exists():
        raise AssertionError(f"Temporary adapter cleanup failed: {temporary_root}")
    return {
        "status": "PASS",
        "model": model_path.name,
        "base_weight_format": "safetensors",
        "lora_rank": LORA_RANK,
        "lora_alpha": LORA_ALPHA,
        "max_model_len": args.max_model_len,
        "versions": _package_versions(),
        "adapter": adapter_summary,
        "vllm": vllm_summary,
        "elapsed_seconds": time.perf_counter() - started,
        "temporary_adapter_cleaned": True,
    }


def main() -> None:
    summary = run_smoke(_parse_args())
    print("EXOPD VLLM LORA SMOKE PASS")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
