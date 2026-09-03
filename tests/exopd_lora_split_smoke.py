"""Sequential single-GPU real-model smoke test for LoRA ExOPD.

The three modes are intentionally separate processes:

``student``
    Loads only Qwen3-1.7B, attaches rank-64 LoRA to every Transformer
    dense projection, samples a response, obtains the frozen reference by
    disabling the adapter, and runs one ExOPD backward/optimizer step.
``teacher``
    Loads only the Teacher and scores the exact response token IDs saved by
    the Student process.  It never reconstructs the response from text.
``joint``
    Runs on CPU tensors only and validates alignment, masks, causal indexes,
    the production ExOPD formula, stop-gradient behavior, and the shared
    token-mean REINFORCE path.

This is test-only code.  It does not alter the formal four-Student/four-Teacher
training topology.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_ROOT = PROJECT_ROOT / "tests"
VERL_ROOT = PROJECT_ROOT / "verl"
for _path in (PROJECT_ROOT, VERL_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

SCHEMA_VERSION = "exopd-lora-split-v1"
EXPECTED_STUDENT_DIRNAME = "Qwen3-1.7B"
DEFAULT_TOTAL_LENGTH = 1024
DEFAULT_RESPONSE_LENGTH = 64
DEFAULT_EXOPD_LAMBDA = 1.25
EXPECTED_DENSE_SUFFIXES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def _require_single_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("The student and teacher modes require a CUDA GPU.")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "Expose exactly one GPU to this local split smoke; visible devices="
            f"{torch.cuda.device_count()}."
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The visible GPU must support bfloat16.")


def _model_path(path: Path, *, student: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {resolved}")
    if student and resolved.name != EXPECTED_STUDENT_DIRNAME:
        raise ValueError(
            "This smoke intentionally fixes the local Student to Qwen3-1.7B; "
            f"got {resolved}."
        )
    return resolved


def _fixture_path(path: Path, *, create_parent: bool) -> Path:
    resolved = path.expanduser().resolve()
    if resolved != TESTS_ROOT and TESTS_ROOT not in resolved.parents:
        raise ValueError(
            "Local-test fixtures must remain under tests/: " f"{resolved}"
        )
    if create_parent:
        resolved.parent.mkdir(parents=True, exist_ok=True)
    elif not resolved.is_file():
        raise FileNotFoundError(f"Fixture does not exist: {resolved}")
    return resolved


def _release_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _token_fingerprint(tokenizer, input_ids: torch.Tensor) -> str:
    """Fingerprint token meanings without saving decoded/re-tokenized text."""

    digest = hashlib.sha256()
    for token_id in input_ids.reshape(-1).tolist():
        token = tokenizer.convert_ids_to_tokens(int(token_id))
        digest.update(str(int(token_id)).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(token).encode("utf-8", errors="surrogatepass"))
        digest.update(b"\n")
    return digest.hexdigest()


def _thinking_prompt_ids(tokenizer, target_length: int) -> torch.Tensor:
    """Render one Qwen3 thinking prompt at an exact token length."""

    def render(filler_count: int) -> list[int]:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful math assistant.",
            },
            {
                "role": "user",
                "content": (
                    "Solve 2x + 3 = 11 step by step. The repeated word below "
                    "is harmless context for a long-sequence systems test:"
                    + " x" * filler_count
                ),
            },
        ]
        return list(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        )

    base = render(0)
    if len(base) > target_length:
        raise ValueError(
            f"Requested prompt length {target_length} is below template length "
            f"{len(base)}."
        )
    # For the Qwen3 tokenizer, each appended `` x`` is exactly one token.  The
    # small correction loop also fails loudly if a future tokenizer changes it.
    filler_count = target_length - len(base)
    for _ in range(8):
        prompt_ids = render(filler_count)
        difference = target_length - len(prompt_ids)
        if difference == 0:
            decoded_tail = tokenizer.decode(prompt_ids[-16:])
            # Qwen3's current template represents thinking-enabled generation
            # by leaving the assistant turn open.  ``enable_thinking=False``
            # instead pre-fills an empty ``<think>...</think>`` block.
            if not decoded_tail.endswith("<|im_start|>assistant\n") or "</think>" in decoded_tail:
                raise AssertionError(
                    "Qwen3 thinking mode did not leave an open assistant turn."
                )
            return torch.tensor([prompt_ids], dtype=torch.long)
        filler_count += difference
        if filler_count < 0:
            break
    raise AssertionError(
        "Could not render the thinking prompt at the requested exact length; "
        "the tokenizer no longer maps the filler monotonically one-for-one."
    )


def _causal_lm_parts(model):
    causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    prefix = str(causal_lm.base_model_prefix)
    backbone = getattr(causal_lm, prefix)
    output_head = causal_lm.get_output_embeddings()
    if output_head is None:
        raise AssertionError("Causal LM does not expose an output embedding head.")
    return backbone, output_head


def _sampled_token_log_probs(
    model,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor,
    prompt_length: int,
    response_length: int,
) -> torch.Tensor:
    """Score saved response IDs at their original causal-logit rows."""

    if input_ids.shape != attention_mask.shape or input_ids.shape != position_ids.shape:
        raise AssertionError("input_ids, attention_mask, and position_ids must align.")
    if input_ids.shape[0] != 1:
        raise AssertionError("This memory-bounded smoke intentionally uses batch size 1.")
    if prompt_length + response_length != input_ids.shape[1]:
        raise AssertionError("Prompt/response lengths do not cover the input sequence.")

    backbone, output_head = _causal_lm_parts(model)
    outputs = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        return_dict=True,
    )
    causal_start = prompt_length - 1
    causal_stop = causal_start + response_length
    selected_hidden = outputs.last_hidden_state[:, causal_start:causal_stop]
    target_ids = input_ids[:, prompt_length:]
    if selected_hidden.shape[:2] != target_ids.shape:
        raise AssertionError(
            "Causal row/token alignment failed: "
            f"hidden={tuple(selected_hidden.shape)}, targets={tuple(target_ids.shape)}."
        )
    # Only materialize vocabulary logits for response positions.  This keeps
    # the 1024-token backward well within one 48 GB card while retaining the
    # exact full-context Transformer path.
    logits = output_head(selected_hidden).float()
    return logits.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(
        logits, dim=-1
    )


def _lora_dense_modules(model) -> list[str]:
    return [
        name
        for name, module in model.named_modules()
        if hasattr(module, "lora_A") and "default" in module.lora_A
    ]


def _assert_fixture_schema(payload: dict, expected_phase: str) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise AssertionError(
            f"Unexpected fixture schema: {payload.get('schema_version')!r}."
        )
    if payload.get("phase") != expected_phase:
        raise AssertionError(
            f"Expected {expected_phase!r} fixture, got {payload.get('phase')!r}."
        )


def run_student(args: argparse.Namespace) -> dict[str, object]:
    """Run rollout, reference/policy forwards, backward, and one LoRA step."""

    _require_single_cuda()
    model_path = _model_path(args.model, student=True)
    output_path = _fixture_path(args.output, create_parent=True)
    if not 256 <= args.total_length <= 1024:
        raise ValueError("total-length must be in [256, 1024] for this local smoke.")
    if not 8 <= args.response_length < args.total_length - 64:
        raise ValueError("response-length leaves too little room for the prompt.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive.")

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from verl.trainer.distillation.losses import compute_exopd_advantage

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
        prompt_length = args.total_length - args.response_length
        prompt_ids_cpu = _thinking_prompt_ids(tokenizer, prompt_length)

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
                f"Expected Qwen3 Student, got {base_model.config.model_type!r}."
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

        lora_config = model.peft_config["default"]
        if int(lora_config.r) != 64 or int(lora_config.lora_alpha) != 128:
            raise AssertionError("Student adapter is not LoRA rank64/alpha128.")
        # PEFT resolves the ``all-linear`` sentinel to the concrete Qwen
        # projection names when the adapter is injected.
        if set(lora_config.target_modules) != EXPECTED_DENSE_SUFFIXES:
            raise AssertionError(
                "Expected all-linear to resolve to every Qwen dense projection, "
                f"got {lora_config.target_modules!r}."
            )
        dense_modules = _lora_dense_modules(model)
        dense_suffixes = {name.rsplit(".", 1)[-1] for name in dense_modules}
        expected_module_count = int(model.config.num_hidden_layers) * len(
            EXPECTED_DENSE_SUFFIXES
        )
        if dense_suffixes != EXPECTED_DENSE_SUFFIXES:
            raise AssertionError(
                "all-linear did not cover exactly all Qwen Transformer dense "
                f"projections: {sorted(dense_suffixes)}."
            )
        if len(dense_modules) != expected_module_count:
            raise AssertionError(
                f"Expected {expected_module_count} adapted dense modules, got "
                f"{len(dense_modules)}."
            )
        if any(name.endswith("lm_head") for name in dense_modules):
            raise AssertionError("The tied output head must not be adapted.")

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
            raise AssertionError("Expected both LoRA and frozen base parameters.")
        if any("lora_" not in name for name in trainable):
            raise AssertionError("A non-LoRA parameter is unexpectedly trainable.")
        if any("lora_" in name for name in frozen):
            raise AssertionError("A LoRA parameter is unexpectedly frozen.")

        # PEFT initializes LoRA-B at zero.  A small deterministic perturbation
        # represents a policy after some training and makes enabled/disabled
        # adapter behavior directly observable in this one-step smoke.
        with torch.no_grad():
            for name, parameter in trainable.items():
                if "lora_B" in name:
                    parameter.normal_(mean=0.0, std=args.adapter_init_std)

        prompt_ids = prompt_ids_cpu.to("cuda")
        prompt_mask = torch.ones_like(prompt_ids, dtype=torch.long)
        model.eval()
        model.config.use_cache = True
        with torch.inference_mode():
            generated_ids = model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                do_sample=False,
                min_new_tokens=args.response_length,
                max_new_tokens=args.response_length,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
        # Generation under inference_mode returns inference tensors, which
        # autograd cannot save during the later policy forward.
        full_ids = generated_ids.detach().clone()
        del generated_ids
        if full_ids.shape != (1, args.total_length):
            raise AssertionError(
                "Student rollout did not produce the requested exact total length: "
                f"{tuple(full_ids.shape)}."
            )
        if not torch.equal(full_ids[:, :prompt_length], prompt_ids):
            raise AssertionError("Generation changed the original prompt token IDs.")

        response_ids = full_ids[:, prompt_length:].clone()
        input_ids = full_ids
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        position_ids = torch.arange(
            args.total_length, device="cuda", dtype=torch.long
        ).unsqueeze(0)
        response_positions = torch.arange(
            prompt_length, args.total_length, device="cuda", dtype=torch.long
        ).unsqueeze(0)
        causal_rows = response_positions - 1
        response_mask = torch.ones_like(response_ids, dtype=torch.bool)
        if not torch.equal(input_ids.gather(1, response_positions), response_ids):
            raise AssertionError("Saved response positions do not select rollout IDs.")

        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
        model.eval()
        with torch.no_grad(), model.disable_adapter():
            reference_before = _sampled_token_log_probs(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                prompt_length=prompt_length,
                response_length=args.response_length,
            ).clone()

        model.train()
        frozen_versions = {
            name: parameter._version for name, parameter in frozen.items()
        }
        adapter_versions = {
            name: parameter._version for name, parameter in trainable.items()
        }
        probe_name, probe_parameter = next(
            (name, parameter)
            for name, parameter in trainable.items()
            if "lora_B" in name
        )
        probe_before = probe_parameter.detach().float().cpu().clone()
        optimizer = torch.optim.AdamW(trainable.values(), lr=args.learning_rate)
        optimizer.zero_grad(set_to_none=True)
        student_before = _sampled_token_log_probs(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            prompt_length=prompt_length,
            response_length=args.response_length,
        )
        policy_reference_gap = float(
            (student_before.detach() - reference_before).abs().max().cpu()
        )
        if policy_reference_gap <= 0.0:
            raise AssertionError(
                "Enabled LoRA policy unexpectedly equals its disabled reference."
            )

        # The real Teacher runs in the next process.  This bounded deterministic
        # proxy is used only to prove the real Student LoRA backward/step here.
        mock_teacher = reference_before + torch.linspace(
            -0.25,
            0.25,
            args.response_length,
            device="cuda",
            dtype=torch.float32,
        ).unsqueeze(0)
        advantage = compute_exopd_advantage(
            student_before,
            mock_teacher,
            reference_before,
            exopd_lambda=args.exopd_lambda,
        )
        explicit_advantage = args.exopd_lambda * (
            mock_teacher - reference_before
        ) - (student_before.detach() - reference_before)
        torch.testing.assert_close(advantage, explicit_advantage, rtol=1e-6, atol=1e-6)
        if advantage.requires_grad or advantage.grad_fn is not None:
            raise AssertionError("ExOPD advantage is not stop-gradient.")
        policy_loss = -(
            advantage * student_before * response_mask.to(student_before.dtype)
        ).sum() / response_mask.sum()
        if not bool(torch.isfinite(policy_loss)):
            raise AssertionError("Student ExOPD loss is non-finite.")
        policy_loss.backward()

        missing_grad = [
            name for name, parameter in trainable.items() if parameter.grad is None
        ]
        invalid_grad = [
            name
            for name, parameter in trainable.items()
            if parameter.grad is not None
            and not bool(torch.isfinite(parameter.grad).all())
        ]
        base_grad = [
            name for name, parameter in frozen.items() if parameter.grad is not None
        ]
        if missing_grad:
            raise AssertionError(f"LoRA parameters missing gradients: {missing_grad[:5]}")
        if invalid_grad:
            raise AssertionError(f"Non-finite LoRA gradients: {invalid_grad[:5]}")
        if base_grad:
            raise AssertionError(f"Frozen base parameters got gradients: {base_grad[:5]}")

        optimizer.step()
        changed_base = [
            name
            for name, parameter in frozen.items()
            if parameter._version != frozen_versions[name]
        ]
        updated_adapters = [
            name
            for name, parameter in trainable.items()
            if parameter._version > adapter_versions[name]
        ]
        if changed_base:
            raise AssertionError(f"Frozen base parameters changed: {changed_base[:5]}")
        if not updated_adapters:
            raise AssertionError("Optimizer did not update any LoRA parameter.")
        if torch.equal(probe_before, probe_parameter.detach().float().cpu()):
            raise AssertionError(f"LoRA update probe did not change: {probe_name}")

        model.eval()
        with torch.no_grad():
            student_after = _sampled_token_log_probs(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                prompt_length=prompt_length,
                response_length=args.response_length,
            )
            with model.disable_adapter():
                reference_after = _sampled_token_log_probs(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    prompt_length=prompt_length,
                    response_length=args.response_length,
                )
        torch.testing.assert_close(reference_after, reference_before, rtol=0, atol=0)
        policy_update = float(
            (student_after - student_before.detach()).abs().max().cpu()
        )
        if policy_update <= 0.0:
            raise AssertionError("Enabled policy log-probabilities did not change.")

        input_ids_cpu = input_ids.detach().cpu()
        fixture = {
            "schema_version": SCHEMA_VERSION,
            "phase": "student",
            "student_model": model_path.name,
            "tokenizer_vocab_size": int(len(tokenizer)),
            "token_fingerprint": _token_fingerprint(tokenizer, input_ids_cpu),
            "total_length": int(args.total_length),
            "prompt_length": int(prompt_length),
            "response_length": int(args.response_length),
            "exopd_lambda": float(args.exopd_lambda),
            "lora_rank": int(lora_config.r),
            "lora_alpha": int(lora_config.lora_alpha),
            "lora_target_modules": "all-linear",
            "lora_dense_module_count": int(len(dense_modules)),
            "input_ids": input_ids_cpu,
            "prompt_ids": input_ids_cpu[:, :prompt_length].clone(),
            "response_ids": response_ids.detach().cpu(),
            "attention_mask": attention_mask.detach().cpu(),
            "position_ids": position_ids.detach().cpu(),
            "response_positions": response_positions.detach().cpu(),
            "causal_rows": causal_rows.detach().cpu(),
            "response_mask": response_mask.detach().cpu(),
            # These are the on-policy, pre-optimizer values used by ExOPD.
            "student_log_probs": student_before.detach().float().cpu(),
            "reference_log_probs": reference_before.detach().float().cpu(),
            "student_after_log_probs": student_after.detach().float().cpu(),
            "reference_after_log_probs": reference_after.detach().float().cpu(),
        }
        torch.save(fixture, output_path)
        torch.cuda.synchronize()
        result = {
            "status": "PASS",
            "phase": "student",
            "model": model_path.name,
            "gpu": torch.cuda.get_device_name(0),
            "total_tokens": int(input_ids.shape[1]),
            "prompt_tokens": int(prompt_length),
            "response_tokens": int(args.response_length),
            "lora_rank": int(lora_config.r),
            "lora_alpha": int(lora_config.lora_alpha),
            "adapted_dense_modules": len(dense_modules),
            "trainable_parameters": sum(p.numel() for p in trainable.values()),
            "updated_lora_tensors": len(updated_adapters),
            "policy_reference_max_gap": policy_reference_gap,
            "policy_update_max_change": policy_update,
            "reference_update_max_change": float(
                (reference_after - reference_before).abs().max().cpu()
            ),
            "loss": float(policy_loss.detach().cpu()),
            "peak_allocated_gib": round(
                torch.cuda.max_memory_allocated() / 2**30, 3
            ),
            "fixture_bytes": output_path.stat().st_size,
        }
        return result
    finally:
        optimizer = None
        model = None
        _release_cuda()


def run_teacher(args: argparse.Namespace) -> dict[str, object]:
    """Score the Student's exact sampled IDs under a separately loaded Teacher."""

    _require_single_cuda()
    model_path = _model_path(args.model)
    student_path = _fixture_path(args.student_fixture, create_parent=False)
    output_path = _fixture_path(args.output, create_parent=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    student = torch.load(student_path, map_location="cpu", weights_only=True)
    _assert_fixture_schema(student, "student")
    input_ids_cpu = student["input_ids"].long()
    response_ids_cpu = student["response_ids"].long()
    prompt_length = int(student["prompt_length"])
    response_length = int(student["response_length"])
    if input_ids_cpu.shape != (1, int(student["total_length"])):
        raise AssertionError("Student input shape disagrees with its metadata.")
    if not torch.equal(input_ids_cpu[:, prompt_length:], response_ids_cpu):
        raise AssertionError("Student response is not the saved sequence suffix.")

    torch.cuda.reset_peak_memory_stats()
    model = None
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        if len(tokenizer) != int(student["tokenizer_vocab_size"]):
            raise AssertionError(
                "Student and Teacher tokenizer vocabulary sizes differ: "
                f"{student['tokenizer_vocab_size']} vs {len(tokenizer)}."
            )
        teacher_fingerprint = _token_fingerprint(tokenizer, input_ids_cpu)
        if teacher_fingerprint != student["token_fingerprint"]:
            raise AssertionError(
                "Student and Teacher tokenizers assign different meanings to the "
                "saved token IDs."
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=True,
            dtype=torch.bfloat16,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
        ).to("cuda")
        if str(model.config.model_type) != "qwen3":
            raise AssertionError(
                f"Expected Qwen3 Teacher, got {model.config.model_type!r}."
            )
        if int(model.config.vocab_size) <= int(input_ids_cpu.max()):
            raise AssertionError("A saved Student token exceeds Teacher vocabulary.")
        if input_ids_cpu.shape[1] > int(model.config.max_position_embeddings):
            raise AssertionError("Saved sequence exceeds Teacher context capacity.")

        input_ids = input_ids_cpu.to("cuda")
        attention_mask = student["attention_mask"].long().to("cuda")
        position_ids = student["position_ids"].long().to("cuda")
        model.eval()
        model.config.use_cache = False
        with torch.inference_mode():
            teacher_log_probs = _sampled_token_log_probs(
                model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                prompt_length=prompt_length,
                response_length=response_length,
            )
        if teacher_log_probs.shape != response_ids_cpu.shape:
            raise AssertionError("Teacher sampled-token log-probability shape is wrong.")
        if not bool(torch.isfinite(teacher_log_probs).all()):
            raise AssertionError("Teacher produced non-finite sampled-token logprobs.")

        fixture = {
            "schema_version": SCHEMA_VERSION,
            "phase": "teacher",
            "teacher_model": model_path.name,
            "student_model": student["student_model"],
            "tokenizer_vocab_size": int(len(tokenizer)),
            "token_fingerprint": teacher_fingerprint,
            "total_length": int(input_ids_cpu.shape[1]),
            "prompt_length": prompt_length,
            "response_length": response_length,
            # Preserve the exact IDs so joint mode can prove end-to-end equality.
            "input_ids": input_ids_cpu.clone(),
            "response_ids": response_ids_cpu.clone(),
            "response_positions": student["response_positions"].long().clone(),
            "teacher_log_probs": teacher_log_probs.float().cpu(),
        }
        torch.save(fixture, output_path)
        torch.cuda.synchronize()
        return {
            "status": "PASS",
            "phase": "teacher",
            "model": model_path.name,
            "gpu": torch.cuda.get_device_name(0),
            "total_tokens": int(input_ids.shape[1]),
            "response_tokens": response_length,
            "sampled_token_probability_mean": float(
                teacher_log_probs.float().exp().mean().cpu()
            ),
            "peak_allocated_gib": round(
                torch.cuda.max_memory_allocated() / 2**30, 3
            ),
            "fixture_bytes": output_path.stat().st_size,
        }
    finally:
        model = None
        student = None
        _release_cuda()


def run_joint(args: argparse.Namespace) -> dict[str, object]:
    """Validate ExOPD alignment, formula, stop-gradient, and masking on CPU."""

    student_path = _fixture_path(args.student_fixture, create_parent=False)
    teacher_path = _fixture_path(args.teacher_fixture, create_parent=False)
    student = torch.load(student_path, map_location="cpu", weights_only=True)
    teacher = torch.load(teacher_path, map_location="cpu", weights_only=True)
    _assert_fixture_schema(student, "student")
    _assert_fixture_schema(teacher, "teacher")

    tensor_values = [
        value
        for payload in (student, teacher)
        for value in payload.values()
        if isinstance(value, torch.Tensor)
    ]
    if any(value.device.type != "cpu" for value in tensor_values):
        raise AssertionError("Joint validation must use CPU fixtures only.")

    input_ids = student["input_ids"].long()
    prompt_ids = student["prompt_ids"].long()
    response_ids = student["response_ids"].long()
    attention_mask = student["attention_mask"]
    position_ids = student["position_ids"].long()
    response_positions = student["response_positions"].long()
    causal_rows = student["causal_rows"].long()
    response_mask = student["response_mask"]
    prompt_length = int(student["prompt_length"])
    response_length = int(student["response_length"])
    total_length = int(student["total_length"])

    if input_ids.shape != attention_mask.shape or input_ids.shape != position_ids.shape:
        raise AssertionError("Full-sequence tensor shapes do not align.")
    if input_ids.shape != (1, total_length):
        raise AssertionError("Unexpected full-sequence shape.")
    if prompt_ids.shape != (1, prompt_length):
        raise AssertionError("Unexpected prompt shape.")
    if not (response_ids.shape == response_mask.shape == (1, response_length)):
        raise AssertionError("Response IDs and mask shapes do not align.")
    if response_mask.dtype != torch.bool or not bool(response_mask.all()):
        raise AssertionError("The unpadded real rollout must have an all-true bool mask.")
    if attention_mask.dtype not in (torch.bool, torch.int32, torch.int64):
        raise AssertionError("Attention mask has an unexpected dtype.")
    if not bool(attention_mask.bool().all()):
        raise AssertionError("The smoke sequence unexpectedly contains padding.")
    expected_positions = torch.arange(total_length).unsqueeze(0)
    torch.testing.assert_close(position_ids, expected_positions)
    expected_response_positions = torch.arange(
        prompt_length, total_length
    ).unsqueeze(0)
    torch.testing.assert_close(response_positions, expected_response_positions)
    torch.testing.assert_close(causal_rows, expected_response_positions - 1)
    torch.testing.assert_close(input_ids[:, :prompt_length], prompt_ids)
    torch.testing.assert_close(input_ids.gather(1, response_positions), response_ids)

    # Teacher mode must have carried the same sequence and original response IDs.
    torch.testing.assert_close(teacher["input_ids"].long(), input_ids)
    torch.testing.assert_close(teacher["response_ids"].long(), response_ids)
    torch.testing.assert_close(
        teacher["response_positions"].long(), response_positions
    )
    if teacher["token_fingerprint"] != student["token_fingerprint"]:
        raise AssertionError("Student/Teacher token fingerprints differ.")
    for key in ("total_length", "prompt_length", "response_length"):
        if int(teacher[key]) != int(student[key]):
            raise AssertionError(f"Student/Teacher metadata differs for {key}.")

    student_log_probs = student["student_log_probs"].float().requires_grad_(True)
    reference_log_probs = student["reference_log_probs"].float().requires_grad_(True)
    teacher_log_probs = teacher["teacher_log_probs"].float().requires_grad_(True)
    expected_shape = (1, response_length)
    if not (
        student_log_probs.shape
        == reference_log_probs.shape
        == teacher_log_probs.shape
        == expected_shape
    ):
        raise AssertionError("Student/Reference/Teacher log-prob shapes differ.")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (student_log_probs, reference_log_probs, teacher_log_probs)
    ):
        raise AssertionError("A joint log-probability tensor is non-finite.")

    from verl.trainer.distillation import losses
    from verl.trainer.ppo.core_algos import compute_policy_loss_reinforce

    exopd_lambda = float(student["exopd_lambda"])
    advantage = losses.compute_exopd_advantage(
        student_log_probs,
        teacher_log_probs,
        reference_log_probs,
        exopd_lambda=exopd_lambda,
    )
    with torch.no_grad():
        expected_advantage = exopd_lambda * (
            teacher_log_probs - reference_log_probs
        ) - (student_log_probs - reference_log_probs)
    torch.testing.assert_close(advantage, expected_advantage, rtol=1e-6, atol=1e-6)
    if advantage.requires_grad or advantage.grad_fn is not None:
        raise AssertionError("Production ExOPD advantage is not stop-gradient.")

    distillation_config = SimpleNamespace(
        distillation_loss=SimpleNamespace(exopd_lambda=exopd_lambda)
    )
    loss_data = {
        "teacher_logprobs": teacher_log_probs.unsqueeze(-1),
        "ref_log_prob": reference_log_probs,
        "response_mask": response_mask,
    }
    with patch.object(losses, "no_padding_2_padding", lambda value, _data: value):
        per_token_loss, metrics = losses.compute_exopd_sampled_token_loss(
            None,
            distillation_config,
            {"log_probs": student_log_probs},
            loss_data,
        )
    torch.testing.assert_close(per_token_loss, -advantage, rtol=0, atol=0)
    if per_token_loss.requires_grad:
        raise AssertionError("ExOPD per-token loss should remain detached.")
    expected_metric_keys = {
        "distillation/exopd_advantage_mean",
        "distillation/exopd_advantage_abs_mean",
        "distillation/student_sampled_token_prob",
        "distillation/teacher_sampled_token_prob",
        "distillation/exopd_reference_sampled_token_prob",
    }
    if set(metrics) != expected_metric_keys:
        raise AssertionError(f"Unexpected ExOPD metrics: {sorted(metrics)}")

    # Mask one valid token only for this audit and prove token-mean REINFORCE
    # zeroes its gradient while retaining the exact gradients elsewhere.
    audit_mask = response_mask.clone()
    audit_mask[:, -1] = False
    policy_loss, policy_metrics = compute_policy_loss_reinforce(
        rollout_log_prob=student_log_probs.detach(),
        log_prob=student_log_probs,
        advantages=advantage,
        response_mask=audit_mask,
        loss_agg_mode="token-mean",
        config=SimpleNamespace(global_batch_info={}),
    )
    policy_loss.backward()
    valid_count = audit_mask.sum()
    expected_gradient = torch.where(
        audit_mask,
        -advantage / valid_count,
        torch.zeros_like(advantage),
    )
    torch.testing.assert_close(
        student_log_probs.grad, expected_gradient, rtol=1e-6, atol=1e-7
    )
    if teacher_log_probs.grad is not None or reference_log_probs.grad is not None:
        raise AssertionError("Teacher/Reference received gradients through advantage.")
    if float(policy_metrics["actor/ppo_kl"]) != 0.0:
        raise AssertionError("Identical rollout/current logprobs should have zero KL.")

    return {
        "status": "PASS",
        "phase": "joint",
        "device": "cpu",
        "student_model": student["student_model"],
        "teacher_model": teacher["teacher_model"],
        "total_tokens": total_length,
        "response_tokens": response_length,
        "exopd_lambda": exopd_lambda,
        "advantage_abs_mean": float(advantage.abs().mean()),
        "policy_loss": float(policy_loss.detach()),
        "masked_token_gradient": float(student_log_probs.grad[0, -1]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)

    student = subparsers.add_parser("student", help="real Qwen3-1.7B LoRA phase")
    student.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / EXPECTED_STUDENT_DIRNAME,
    )
    student.add_argument("--output", type=Path, required=True)
    student.add_argument("--total-length", type=int, default=DEFAULT_TOTAL_LENGTH)
    student.add_argument(
        "--response-length", type=int, default=DEFAULT_RESPONSE_LENGTH
    )
    student.add_argument("--learning-rate", type=float, default=1.0e-3)
    student.add_argument("--adapter-init-std", type=float, default=1.0e-3)
    student.add_argument("--exopd-lambda", type=float, default=DEFAULT_EXOPD_LAMBDA)

    teacher = subparsers.add_parser("teacher", help="separate real Teacher phase")
    teacher.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "Qwen3-4B-Thinking-2507",
    )
    teacher.add_argument("--student-fixture", type=Path, required=True)
    teacher.add_argument("--output", type=Path, required=True)

    joint = subparsers.add_parser("joint", help="CPU-only aligned fixture phase")
    joint.add_argument("--student-fixture", type=Path, required=True)
    joint.add_argument("--teacher-fixture", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.phase == "student":
        result = run_student(args)
    elif args.phase == "teacher":
        result = run_teacher(args)
    else:
        result = run_joint(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
