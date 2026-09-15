"""One-GPU Student-only / Teacher-only alignment smoke test for R²OPL-base.

Run from WSL2 after activating the ``verl`` environment:

    PYTHONPATH=.:verl python tests/r2opl_base_real_model_smoke.py

The script deliberately never keeps both models on GPU at the same time.  It
preserves the Student's sampled response IDs and detached log-probabilities on
CPU, frees the Student, and then makes the Teacher score those exact token IDs.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from algorithms.r2opl_base import compute_r2opl_base_batch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_STUDENT = PROJECT_ROOT / "models" / "Qwen3-1.7B"
DEFAULT_TEACHER = PROJECT_ROOT / "models" / "Qwen3-4B-Instruct-2507"


def _response_log_probs(
    model: AutoModelForCausalLM, full_ids: torch.Tensor, prompt_length: int
) -> torch.Tensor:
    """Return causal log-probabilities for the immutable sampled response IDs."""

    with torch.inference_mode():
        logits = model(input_ids=full_ids, use_cache=False).logits
        next_token_log_probs = logits[:, :-1].float().log_softmax(dim=-1)
        target_ids = full_ids[:, 1:]
        sampled_log_probs = next_token_log_probs.gather(
            dim=-1, index=target_ids.unsqueeze(-1)
        ).squeeze(-1)
    # Logit row p-1 predicts the first response token at absolute index p.
    return sampled_log_probs[:, prompt_length - 1 :]


def _response_log_probs_with_grad(
    model: AutoModelForCausalLM, full_ids: torch.Tensor, prompt_length: int
) -> torch.Tensor:
    """Differentiable counterpart used after the Teacher has been released."""

    logits = model(input_ids=full_ids, use_cache=False).logits
    next_token_log_probs = logits[:, :-1].float().log_softmax(dim=-1)
    target_ids = full_ids[:, 1:]
    sampled_log_probs = next_token_log_probs.gather(
        dim=-1, index=target_ids.unsqueeze(-1)
    ).squeeze(-1)
    return sampled_log_probs[:, prompt_length - 1 :]


def _release_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _gradient_norm(model: AutoModelForCausalLM) -> float:
    squared_norm = torch.zeros((), device="cuda", dtype=torch.float32)
    for parameter in model.parameters():
        if parameter.grad is not None:
            squared_norm += parameter.grad.detach().float().square().sum()
    return float(squared_norm.sqrt().cpu())


def _real_student_branch_gradients(
    student_path: Path,
    full_ids: torch.Tensor,
    prompt_length: int,
    teacher_log_probs: torch.Tensor,
    result,
) -> dict[str, float]:
    """Measure both R²OPL branches with the real Student, sequentially."""

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    student = AutoModelForCausalLM.from_pretrained(
        student_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device)
    student.eval()
    batch_ids = full_ids.repeat(8, 1).to(device)
    teacher = teacher_log_probs.repeat(8, 1).to(device)
    response_mask = torch.ones_like(teacher, dtype=torch.bool)
    difficulty = result.difficulty.to(device).unsqueeze(-1)
    correct_mask = result.correct_mask.to(device)
    error_mask = result.error_mask.to(device)

    def branch_loss(branch: str) -> torch.Tensor:
        student_log_probs = _response_log_probs_with_grad(
            student, batch_ids, prompt_length
        )
        correct_advantage = difficulty.expand_as(student_log_probs)
        error_advantage = (
            difficulty
            * (1.0 / 20.0)
            * (teacher - student_log_probs).detach()
        )
        if branch == "correct":
            advantage = correct_advantage * correct_mask
        elif branch == "error":
            advantage = error_advantage * error_mask
        else:  # pragma: no cover - internal caller only
            raise ValueError(branch)
        return (
            (-(advantage * student_log_probs) * response_mask)
            .sum(dim=-1)
            .div(response_mask.sum(dim=-1))
            .mean()
        )

    student.zero_grad(set_to_none=True)
    correct_loss = branch_loss("correct")
    correct_loss.backward()
    correct_grad_norm = _gradient_norm(student)
    correct_gradients = [
        parameter.grad.detach().clone() if parameter.grad is not None else None
        for parameter in student.parameters()
    ]

    student.zero_grad(set_to_none=True)
    error_loss = branch_loss("error")
    error_loss.backward()
    error_grad_norm = _gradient_norm(student)
    for parameter, correct_gradient in zip(
        student.parameters(), correct_gradients, strict=True
    ):
        if correct_gradient is None:
            continue
        if parameter.grad is None:
            parameter.grad = correct_gradient
        else:
            parameter.grad.add_(correct_gradient)
    total_grad_norm = _gradient_norm(student)
    gradient_peak_gb = torch.cuda.max_memory_allocated() / 1024**3

    measurements = {
        "real_correct_grad_norm": correct_grad_norm,
        "real_error_grad_norm": error_grad_norm,
        "real_total_grad_norm": total_grad_norm,
        "real_correct_loss": float(correct_loss.detach().cpu()),
        "real_error_loss": float(error_loss.detach().cpu()),
        "real_gradient_student_peak_gb": float(gradient_peak_gb),
    }
    # Release the live autograd graphs before reporting post-release memory;
    # otherwise their activations keep the last Student allocation alive and
    # make this sequential-model smoke test look like it retained a model.
    del correct_loss, error_loss, branch_loss
    del student, batch_ids, teacher, correct_gradients
    _release_cuda()
    measurements["real_gradient_student_memory_after_release_gb"] = float(
        torch.cuda.memory_allocated() / 1024**3
    )
    return measurements


def run(
    student_path: Path,
    teacher_path: Path,
    *,
    max_new_tokens: int,
    measure_gradients: bool = False,
) -> dict[str, float | int]:
    if not torch.cuda.is_available():
        raise RuntimeError("This smoke test requires one CUDA GPU.")
    device = torch.device("cuda")
    prompt = "What is 2 + 2? Give only the final answer."

    # Student-only rollout and Student log-probability test.
    torch.cuda.reset_peak_memory_stats(device)
    student_tokenizer = AutoTokenizer.from_pretrained(student_path)
    if student_tokenizer.pad_token_id is None:
        student_tokenizer.pad_token_id = student_tokenizer.eos_token_id
    student = AutoModelForCausalLM.from_pretrained(
        student_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device)
    student.eval()
    prompt_ids = student_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(device)
    prompt_length = int(prompt_ids.shape[1])
    prompt_attention_mask = torch.ones_like(prompt_ids)
    with torch.inference_mode():
        full_ids = student.generate(
            prompt_ids,
            attention_mask=prompt_attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            min_new_tokens=max_new_tokens,
            pad_token_id=student_tokenizer.pad_token_id,
        )
    response_ids = full_ids[:, prompt_length:]
    if response_ids.numel() == 0:
        raise RuntimeError("Student rollout unexpectedly returned no response tokens.")
    student_log_probs = _response_log_probs(student, full_ids, prompt_length).cpu()
    immutable_full_ids = full_ids.cpu()
    immutable_response_ids = response_ids.cpu()
    student_peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    del student, prompt_ids, prompt_attention_mask, full_ids, response_ids
    _release_cuda()
    student_released_gb = torch.cuda.memory_allocated() / 1024**3

    # Teacher-only probe over exactly the Student sampled IDs.  The two Qwen3
    # checkpoints share the token ID vocabulary; this explicit range check
    # guards accidental text re-tokenization or an incompatible tokenizer.
    torch.cuda.reset_peak_memory_stats(device)
    teacher_tokenizer = AutoTokenizer.from_pretrained(teacher_path)
    teacher_vocab_size = len(teacher_tokenizer)
    if int(immutable_full_ids.max()) >= teacher_vocab_size:
        raise RuntimeError("Student sampled token IDs are outside the Teacher vocabulary.")
    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_path,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to(device)
    teacher.eval()
    teacher_log_probs = _response_log_probs(
        teacher, immutable_full_ids.to(device), prompt_length
    ).cpu()
    teacher_peak_gb = torch.cuda.max_memory_allocated() / 1024**3
    del teacher
    _release_cuda()
    teacher_released_gb = torch.cuda.memory_allocated() / 1024**3

    if student_log_probs.shape != teacher_log_probs.shape:
        raise RuntimeError(
            "Student and Teacher response log-probability shapes differ: "
            f"{tuple(student_log_probs.shape)} vs {tuple(teacher_log_probs.shape)}."
        )
    if not torch.isfinite(student_log_probs).all() or not torch.isfinite(teacher_log_probs).all():
        raise RuntimeError("Student or Teacher returned non-finite sampled-token log-probabilities.")

    # Joint logic stays model-free after both large models are released.  The
    # repeated rows emulate one 8-response prompt group and verify that the
    # exact preserved token axis reaches R²OPL unchanged.
    width = int(immutable_response_ids.shape[1])
    response_mask = torch.ones((8, width), dtype=torch.bool)
    result = compute_r2opl_base_batch(
        old_log_probs=student_log_probs.repeat(8, 1),
        teacher_log_probs=teacher_log_probs.repeat(8, 1),
        response_mask=response_mask,
        verifier_rewards=torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
        prompt_group_ids=["real-smoke-prompt"] * 8,
        lambda_=1.0 / 20.0,
    )
    if not torch.allclose(result.difficulty, torch.full((8,), 0.5)):
        raise RuntimeError("R²OPL group difficulty did not preserve the 4/8 success rate.")

    measurements: dict[str, float | int] = {
        "prompt_tokens": prompt_length,
        "response_tokens": width,
        "student_peak_gb": float(student_peak_gb),
        "student_memory_after_release_gb": float(student_released_gb),
        "teacher_peak_gb": float(teacher_peak_gb),
        "teacher_memory_after_release_gb": float(teacher_released_gb),
        "r2opl_difficulty": float(result.difficulty[0]),
        "response_id_checksum": int(immutable_response_ids.sum()),
        # These are controller-side, trajectory-normalized diagnostics.  They
        # use the exact same 4-correct / 4-error group and preserved sampled
        # token IDs as the real-model branch-gradient measurement below.
        "r2opl_correct_advantage_mean": result.metrics[
            "r2opl_base/train/correct_advantage_mean"
        ],
        "r2opl_correct_advantage_abs_mean": abs(
            result.metrics["r2opl_base/train/correct_advantage_mean"]
        ),
        "r2opl_error_opd_advantage_mean": result.metrics[
            "r2opl_base/train/error_opd_advantage_mean"
        ],
        "r2opl_error_opd_abs_advantage_mean": result.metrics[
            "r2opl_base/train/error_opd_abs_advantage_mean"
        ],
    }
    if measure_gradients:
        measurements.update(
            _real_student_branch_gradients(
                student_path,
                immutable_full_ids,
                prompt_length,
                teacher_log_probs,
                result,
            )
        )
    return measurements


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student", type=Path, default=DEFAULT_STUDENT)
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--measure-gradients", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.student,
                args.teacher,
                max_new_tokens=args.max_new_tokens,
                measure_gradients=args.measure_gradients,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
