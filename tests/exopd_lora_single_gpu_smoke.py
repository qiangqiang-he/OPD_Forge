"""Single-GPU real-model smoke test for the ExOPD LoRA reference path.

This driver intentionally lives under ``tests/``.  It validates the production
ExOPD objective on a local Qwen3-1.7B Student without changing the production
four-Student/four-Teacher topology.  The frozen reference is obtained by
temporarily disabling the Student's adapter; a deterministic tensor stands in
for the separately hosted Teacher log-probabilities.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MODEL_NAME = "Qwen3-1.7B"
DEFAULT_SEQUENCE_LENGTH = 1024
DEFAULT_RESPONSE_LENGTH = 64
MAX_SEQUENCE_LENGTH = 1024
EXPECTED_DENSE_SUFFIXES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / EXPECTED_MODEL_NAME,
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=DEFAULT_SEQUENCE_LENGTH,
        help=f"Total prompt-plus-response length (hard limit: {MAX_SEQUENCE_LENGTH}).",
    )
    parser.add_argument(
        "--response-length",
        type=int,
        default=DEFAULT_RESPONSE_LENGTH,
        help="Number of trailing original token IDs treated as sampled response tokens.",
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> Path:
    model_path = args.model.expanduser().resolve()
    if model_path.name != EXPECTED_MODEL_NAME:
        raise ValueError(
            "The ExOPD LoRA single-GPU smoke fixes the Student to "
            f"{EXPECTED_MODEL_NAME}; got {model_path}."
        )
    if not model_path.is_dir():
        raise FileNotFoundError(f"Student model directory does not exist: {model_path}")
    if not 8 <= args.sequence_length <= MAX_SEQUENCE_LENGTH:
        raise ValueError(
            f"sequence-length must be in [8, {MAX_SEQUENCE_LENGTH}], got "
            f"{args.sequence_length}."
        )
    if not 1 <= args.response_length < args.sequence_length:
        raise ValueError(
            "response-length must be positive and smaller than sequence-length; "
            f"got {args.response_length} and {args.sequence_length}."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this real-model smoke test.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "This local smoke must expose exactly one GPU. Set CUDA_VISIBLE_DEVICES "
            f"to one device; currently visible={torch.cuda.device_count()}."
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The visible GPU must support bfloat16.")
    return model_path


def _build_original_token_ids(tokenizer, sequence_length: int) -> torch.Tensor:
    """Build one exact token sequence once; no text reconstruction is used later."""

    seed_text = (
        "<|im_start|>system\nYou are a helpful math assistant.<|im_end|>\n"
        "<|im_start|>user\nSolve 2x + 3 = 11 step by step.<|im_end|>\n"
        "<|im_start|>assistant\n<think>We solve the equation carefully.</think>"
    )
    seed_ids = tokenizer.encode(seed_text, add_special_tokens=False)
    if not seed_ids:
        raise AssertionError("Tokenizer unexpectedly returned an empty sequence.")
    repeats = (sequence_length + len(seed_ids) - 1) // len(seed_ids)
    exact_ids = (seed_ids * repeats)[:sequence_length]
    return torch.tensor([exact_ids], dtype=torch.long)


def _sampled_response_log_probs(
    model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    response_length: int,
) -> torch.Tensor:
    """Score the trailing sampled IDs at their original causal token indexes."""

    prompt_length = input_ids.shape[1] - response_length
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        return_dict=True,
    )
    # Token input_ids[:, prompt_length] is predicted by causal-logit row
    # prompt_length - 1.  This preserves the rollout token IDs and indexes.
    response_logits = outputs.logits[:, prompt_length - 1 : -1].float()
    response_ids = input_ids[:, prompt_length:]
    if response_logits.shape[:2] != response_ids.shape:
        raise AssertionError(
            "Response logit/token alignment failed: "
            f"logits={tuple(response_logits.shape)}, ids={tuple(response_ids.shape)}."
        )
    return torch.log_softmax(response_logits, dim=-1).gather(
        -1, response_ids.unsqueeze(-1)
    ).squeeze(-1)


def _lora_dense_modules(model) -> list[str]:
    return [
        name
        for name, module in model.named_modules()
        if hasattr(module, "lora_A") and "default" in module.lora_A
    ]


def _release_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def run_smoke(args: argparse.Namespace) -> dict[str, object]:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from verl.trainer.distillation.losses import compute_exopd_advantage

    model_path = _validate_args(args)
    torch.manual_seed(20260903)
    torch.cuda.manual_seed_all(20260903)
    torch.cuda.reset_peak_memory_stats()

    model = None
    optimizer = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        original_ids = _build_original_token_ids(tokenizer, args.sequence_length)
        del tokenizer

        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        )
        if str(base_model.config.model_type) != "qwen3":
            raise AssertionError(
                f"Expected a Qwen3 checkpoint, got {base_model.config.model_type!r}."
            )
        model = get_peft_model(
            base_model,
            LoraConfig(
                r=64,
                lora_alpha=128,
                target_modules="all-linear",
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        ).to("cuda")
        del base_model
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()

        lora_config = model.peft_config["default"]
        if int(lora_config.r) != 64 or int(lora_config.lora_alpha) != 128:
            raise AssertionError("Loaded adapter is not LoRA r=64, alpha=128.")
        dense_modules = _lora_dense_modules(model)
        dense_suffixes = {name.rsplit(".", 1)[-1] for name in dense_modules}
        expected_count = int(model.config.num_hidden_layers) * len(
            EXPECTED_DENSE_SUFFIXES
        )
        if dense_suffixes != EXPECTED_DENSE_SUFFIXES:
            raise AssertionError(
                "all-linear did not cover exactly the Qwen dense projections: "
                f"{sorted(dense_suffixes)}."
            )
        if len(dense_modules) != expected_count:
            raise AssertionError(
                f"Expected {expected_count} LoRA dense modules, got "
                f"{len(dense_modules)}."
            )
        if any(name.endswith("lm_head") for name in dense_modules):
            raise AssertionError("all-linear must not adapt the output lm_head.")

        trainable = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        frozen = {
            name: parameter
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad
        }
        if not trainable or not frozen:
            raise AssertionError("Expected both trainable LoRA and frozen base tensors.")
        if any("lora_" not in name for name in trainable):
            raise AssertionError("A non-LoRA model parameter is unexpectedly trainable.")
        if any("lora_" in name for name in frozen):
            raise AssertionError("A LoRA parameter is unexpectedly frozen.")

        # PEFT initializes LoRA-B at zero. Seed it before the reference snapshot
        # so the enabled Student is observably distinct from its disabled base.
        with torch.no_grad():
            for name, parameter in trainable.items():
                if "lora_B" in name:
                    parameter.normal_(mean=0.0, std=1.0e-3)

        input_ids = original_ids.to("cuda", non_blocking=True)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        position_ids = torch.arange(
            args.sequence_length, device="cuda", dtype=torch.long
        ).unsqueeze(0)
        prompt_length = args.sequence_length - args.response_length
        if position_ids[0, prompt_length - 1].item() != prompt_length - 1:
            raise AssertionError("Position IDs do not match original causal indexes.")
        if not bool(attention_mask.all()):
            raise AssertionError("Unexpected padding in the synthetic smoke batch.")

        model.eval()
        with torch.no_grad(), model.disable_adapter():
            reference_before = _sampled_response_log_probs(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                response_length=args.response_length,
            ).clone()

        model.train()
        frozen_versions = {
            name: parameter._version for name, parameter in frozen.items()
        }
        trainable_versions = {
            name: parameter._version for name, parameter in trainable.items()
        }
        optimizer = torch.optim.AdamW(trainable.values(), lr=args.learning_rate)
        optimizer.zero_grad(set_to_none=True)
        student_before = _sampled_response_log_probs(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            response_length=args.response_length,
        )
        if torch.equal(student_before.detach(), reference_before):
            raise AssertionError(
                "Enabled LoRA Student unexpectedly equals the disabled reference."
            )

        teacher_log_probs = reference_before + torch.linspace(
            -0.25,
            0.25,
            args.response_length,
            device="cuda",
            dtype=torch.float32,
        ).unsqueeze(0)
        advantage = compute_exopd_advantage(
            student_before,
            teacher_log_probs,
            reference_before,
            exopd_lambda=1.25,
        )
        if advantage.requires_grad or advantage.shape != student_before.shape:
            raise AssertionError("ExOPD advantage must be detached and token-aligned.")
        if not bool(torch.isfinite(advantage).all()):
            raise AssertionError("ExOPD advantage contains non-finite values.")

        loss = -(advantage * student_before).mean()
        if not bool(torch.isfinite(loss)):
            raise AssertionError("ExOPD policy loss is non-finite.")
        loss.backward()

        missing_grads = [
            name for name, parameter in trainable.items() if parameter.grad is None
        ]
        invalid_grads = [
            name
            for name, parameter in trainable.items()
            if parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all())
        ]
        base_grads = [
            name for name, parameter in frozen.items() if parameter.grad is not None
        ]
        if missing_grads:
            raise AssertionError(
                f"LoRA parameters missing gradients: {missing_grads[:5]}"
            )
        if invalid_grads:
            raise AssertionError(
                f"LoRA parameters have invalid gradients: {invalid_grads[:5]}"
            )
        if base_grads:
            raise AssertionError(f"Frozen base parameters received gradients: {base_grads[:5]}")

        optimizer.step()
        updated_adapters = [
            name
            for name, parameter in trainable.items()
            if parameter._version > trainable_versions[name]
        ]
        changed_base = [
            name
            for name, parameter in frozen.items()
            if parameter._version != frozen_versions[name]
        ]
        if not updated_adapters:
            raise AssertionError("The optimizer step did not update a LoRA tensor.")
        if changed_base:
            raise AssertionError(f"Frozen base tensors changed: {changed_base[:5]}")

        model.eval()
        with torch.no_grad():
            student_after = _sampled_response_log_probs(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                response_length=args.response_length,
            )
            with model.disable_adapter():
                reference_after = _sampled_response_log_probs(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    response_length=args.response_length,
                )
        torch.testing.assert_close(reference_after, reference_before, rtol=0, atol=0)
        if torch.equal(student_after, student_before.detach()):
            raise AssertionError("Adapter-enabled Student output did not change after step.")

        torch.cuda.synchronize()
        peak_gib = torch.cuda.max_memory_allocated() / 2**30
        result = {
            "status": "PASS",
            "model": model_path.name,
            "gpu": torch.cuda.get_device_name(0),
            "dtype": "bfloat16",
            "attention": "sdpa",
            "sequence_length": args.sequence_length,
            "response_length": args.response_length,
            "lora_rank": int(lora_config.r),
            "lora_alpha": int(lora_config.lora_alpha),
            "lora_dense_modules": len(dense_modules),
            "trainable_parameters": sum(p.numel() for p in trainable.values()),
            "frozen_parameters": sum(p.numel() for p in frozen.values()),
            "updated_lora_tensors": len(updated_adapters),
            "loss": float(loss.detach().cpu()),
            "advantage_abs_mean": float(advantage.abs().mean().cpu()),
            "student_logprob_max_change": float(
                (student_after - student_before.detach()).abs().max().cpu()
            ),
            "peak_allocated_gib": round(peak_gib, 3),
        }
        return result
    finally:
        optimizer = None
        model = None
        _release_cuda()


def main() -> None:
    result = run_smoke(_parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
