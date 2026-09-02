"""Single-GPU phased smoke driver for Sol-OPD.

Each GPU-heavy phase is launched as a separate process by
``tests/run_sol_opd_single_gpu_smoke.sh``. Student and Teacher models are
therefore never resident on the RTX 4090 at the same time. GPU response
length is capped at 1024 tokens; the separate CPU/tokenizer dataset audit
checks the formal 8192-token solution-prompt truncation boundary. Saved
artifacts carry the Student's original response token IDs between phases.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


SHORT_QUESTION = "Compute 20 + 22 and justify the result."
SHORT_SOLUTION = (
    "Add the two integers: 20 + 22 = 42. Therefore the final answer is "
    r"\boxed{42}."
)


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("The Sol-OPD model smoke phases require one CUDA GPU.")
    if torch.cuda.device_count() != 1:
        print(
            f"NOTE: {torch.cuda.device_count()} GPUs are visible; this driver uses cuda:0 only."
        )


def _release_cuda(*objects) -> None:
    del objects
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _load_tokenizer(model_path: Path):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )


def _token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def dataset_audit(model_path: Path, data_path: Path) -> None:
    """Audit all 10,634 solution prompts with the real Qwen tokenizer."""

    from utils.prompts import render_prompt
    from verl.experimental.agent_loop.agent_loop import (
        _tokenize_solution_privileged_prompt,
    )

    tokenizer = _load_tokenizer(model_path)
    with data_path.open("r", encoding="utf-8") as stream:
        records = json.load(stream)
    if not isinstance(records, list) or len(records) != 10_634:
        raise AssertionError(
            f"Expected 10,634 Sol-OPD records, got {type(records).__name__} "
            f"with length {len(records) if isinstance(records, list) else 'n/a'}."
        )

    full_lengths: list[int] = []
    effective_lengths: list[int] = []
    truncated_count = 0
    marker = "__SOL_OPD_AUDIT_MARKER__"
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise AssertionError(f"Record {index} is not a JSON object.")
        question = record.get("question")
        answer = record.get("answer")
        solution = record.get("solution")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (question, answer, solution)
        ):
            raise AssertionError(
                f"Record {index} lacks non-empty lowercase question/answer/solution strings."
            )

        full_prompt = render_prompt(
            "qwen3_privileged_solution_thinking",
            question=question,
            privileged_solution=solution,
        )
        full_length = _token_count(tokenizer, full_prompt)
        effective_ids = _tokenize_solution_privileged_prompt(
            tokenizer,
            prompt_name="qwen3_privileged_solution_thinking",
            question=question,
            privileged_solution=solution,
            max_prompt_length=8192,
        )
        effective_length = len(effective_ids)
        if effective_length > 8192:
            raise AssertionError(
                f"Record {index} remains over capacity: {effective_length}."
            )
        full_lengths.append(full_length)
        effective_lengths.append(effective_length)
        truncated_count += int(full_length > 8192)

        # Decode the complete effective rendering and prove that only a prefix
        # of the solution may have disappeared. The question, fixed template,
        # assistant boundary, and solution tail must remain byte-for-byte.
        marked_prompt = render_prompt(
            "qwen3_privileged_solution_thinking",
            question=question,
            privileged_solution=marker,
        )
        prefix, suffix = marked_prompt.split(marker)
        decoded = tokenizer.decode(
            effective_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if not decoded.startswith(prefix) or not decoded.endswith(suffix):
            raise AssertionError(
                f"Record {index} lost fixed prompt structure or its question."
            )
        retained_solution = decoded[len(prefix) : -len(suffix)]
        if not retained_solution or not solution.endswith(retained_solution):
            raise AssertionError(
                f"Record {index} did not preserve a non-empty solution suffix."
            )
        if not decoded.endswith("<|im_start|>assistant\n"):
            raise AssertionError(f"Record {index} lost the thinking assistant boundary.")

    if truncated_count == 0:
        raise AssertionError("The audit did not exercise solution-only truncation.")
    ordered = sorted(full_lengths)

    def percentile(fraction: float) -> int:
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]

    print(
        "SOL DATASET AUDIT PASS: "
        f"records={len(records)}, truncated={truncated_count}, "
        f"full_p50={percentile(0.50)}, full_p95={percentile(0.95)}, "
        f"full_p99={percentile(0.99)}, full_max={max(full_lengths)}, "
        f"effective_max={max(effective_lengths)}"
    )


def student_rollout(
    model_path: Path,
    output_path: Path,
    *,
    response_tokens: int,
) -> None:
    """Run a real vLLM Student rollout and save its immutable token IDs."""

    _require_cuda()
    if not 1 <= response_tokens <= 1024:
        raise ValueError("Local response_tokens must lie in [1, 1024].")

    from vllm import LLM, SamplingParams

    from utils.prompts import render_prompt

    tokenizer = _load_tokenizer(model_path)
    question = SHORT_QUESTION
    solution = SHORT_SOLUTION
    prompt_text = render_prompt("qwen3_thinking_prompt", question=question)
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if len(prompt_ids) > 2048:
        raise AssertionError(f"Student prompt exceeds 2048 tokens: {len(prompt_ids)}.")
    if len(prompt_ids) + response_tokens > 10240:
        raise AssertionError("Student prompt + response exceeds formal max_model_len=10240.")
    local_max_model_len = max(2048, len(prompt_ids) + response_tokens + 1)

    torch.cuda.reset_peak_memory_stats()
    llm = LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        runner="generate",
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
        seed=42,
        gpu_memory_utilization=0.40,
        max_model_len=local_max_model_len,
        max_num_batched_tokens=local_max_model_len,
        max_num_seqs=1,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        enforce_eager=True,
        max_logprobs=16,
    )
    sampling = SamplingParams(
        n=1,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        min_tokens=response_tokens,
        max_tokens=response_tokens,
        ignore_eos=True,
        logprobs=1,
        seed=42,
    )
    outputs = llm.generate(
        [{"prompt_token_ids": prompt_ids}],
        sampling_params=sampling,
        use_tqdm=response_tokens >= 1024,
    )
    if len(outputs) != 1 or len(outputs[0].outputs) != 1:
        raise AssertionError("Student vLLM returned an unexpected output count.")
    request_output = outputs[0]
    candidate = request_output.outputs[0]
    if list(request_output.prompt_token_ids) != prompt_ids:
        raise AssertionError("Student vLLM changed the supplied prompt token IDs.")
    response_ids = list(candidate.token_ids)
    if len(response_ids) != response_tokens:
        raise AssertionError(
            f"Expected {response_tokens} response tokens, got {len(response_ids)}."
        )
    if candidate.logprobs is None or len(candidate.logprobs) != response_tokens:
        raise AssertionError("Student sampled-token logprob rows are incomplete.")
    response_logprobs = []
    for token_id, row in zip(response_ids, candidate.logprobs, strict=True):
        if row is None or token_id not in row:
            raise AssertionError(f"Student logprobs omit sampled token {token_id}.")
        response_logprobs.append(float(row[token_id].logprob))
    response_logprobs_tensor = torch.tensor(
        [response_logprobs], dtype=torch.float32
    )
    if not bool(torch.isfinite(response_logprobs_tensor).all()):
        raise AssertionError("Student produced non-finite sampled-token logprobs.")

    prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long)
    response_tensor = torch.tensor([response_ids], dtype=torch.long)
    sequence_length = len(prompt_ids) + response_tokens
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_path.name,
            "question": question,
            "privileged_solution": solution,
            "prompt_text": prompt_text,
            "prompt_ids": prompt_tensor,
            "response_ids": response_tensor,
            "student_logprobs": response_logprobs_tensor,
            "attention_mask": torch.ones((1, sequence_length), dtype=torch.long),
            "position_ids": torch.arange(sequence_length).unsqueeze(0),
            "response_positions": torch.arange(
                len(prompt_ids), sequence_length
            ).unsqueeze(0),
            "causal_rows": torch.arange(
                len(prompt_ids) - 1, sequence_length - 1
            ).unsqueeze(0),
        },
        output_path,
    )
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    print(
        "SOL STUDENT ROLLOUT PASS: "
        f"model={model_path.name}, prompt_tokens={len(prompt_ids)}, "
        f"response_tokens={response_tokens}, peak_allocated_gib={peak_gib:.2f}"
    )
    del candidate, request_output, outputs, llm, tokenizer
    del prompt_tensor, response_tensor, response_logprobs_tensor
    _release_cuda()


def _score_exact_response(
    llm,
    sampling_params,
    *,
    prompt_ids: list[int],
    response_ids: list[int],
) -> tuple[list[float], list[int]]:
    """Score saved response IDs under one prompt without text reconstruction."""

    sequence_ids = list(prompt_ids) + list(response_ids)
    outputs = llm.generate(
        [{"prompt_token_ids": sequence_ids}],
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    result = outputs[0]
    if list(result.prompt_token_ids) != sequence_ids:
        raise AssertionError("Teacher vLLM changed the supplied sequence token IDs.")
    if result.prompt_logprobs is None or len(result.prompt_logprobs) != len(
        sequence_ids
    ):
        raise AssertionError("Teacher prompt-logprob rows do not cover its sequence.")
    response_logprobs = []
    for position, token_id in enumerate(response_ids, start=len(prompt_ids)):
        row = result.prompt_logprobs[position]
        if row is None or token_id not in row:
            raise AssertionError(
                f"Teacher logprobs omit response token {token_id} at {position}."
            )
        response_logprobs.append(float(row[token_id].logprob))
    if not bool(torch.isfinite(torch.tensor(response_logprobs)).all()):
        raise AssertionError("Teacher returned non-finite sampled-token logprobs.")
    return response_logprobs, sequence_ids


def teacher_dual_forward(
    model_path: Path,
    student_path: Path,
    output_path: Path,
    *,
    proxy_used: bool,
) -> None:
    """Run solution-privileged and base Teacher forwards on identical IDs."""

    _require_cuda()
    from vllm import LLM, SamplingParams

    from verl.experimental.agent_loop.agent_loop import (
        _tokenize_solution_privileged_prompt,
    )

    student = torch.load(student_path, map_location="cpu", weights_only=True)
    response_ids = student["response_ids"].squeeze(0).long().tolist()
    base_prompt_ids = student["prompt_ids"].squeeze(0).long().tolist()
    tokenizer = _load_tokenizer(model_path)
    if response_ids and max(response_ids) >= len(tokenizer):
        raise AssertionError("Student response IDs exceed the Teacher vocabulary.")

    solution_prompt_ids = _tokenize_solution_privileged_prompt(
        tokenizer,
        prompt_name="qwen3_privileged_solution_thinking",
        question=str(student["question"]),
        privileged_solution=str(student["privileged_solution"]),
        max_prompt_length=8192,
    )
    if len(solution_prompt_ids) > 8192 or len(base_prompt_ids) > 8192:
        raise AssertionError("A Teacher prompt exceeds its 8192-token capacity.")
    if solution_prompt_ids == base_prompt_ids:
        raise AssertionError("Privileged and base Teacher prompts unexpectedly match.")
    if len(solution_prompt_ids) + len(response_ids) + 1 > 16385:
        raise AssertionError("Privileged Teacher request exceeds max_model_len=16385.")
    if len(base_prompt_ids) + len(response_ids) + 1 > 16385:
        raise AssertionError("Base Teacher request exceeds max_model_len=16385.")
    local_max_model_len = max(
        2048,
        len(solution_prompt_ids) + len(response_ids) + 1,
        len(base_prompt_ids) + len(response_ids) + 1,
    )

    torch.cuda.reset_peak_memory_stats()
    llm = LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        runner="generate",
        trust_remote_code=True,
        tensor_parallel_size=1,
        dtype="bfloat16",
        seed=42,
        gpu_memory_utilization=0.55,
        max_model_len=local_max_model_len,
        max_num_batched_tokens=128,
        max_num_seqs=1,
        enable_chunked_prefill=True,
        enable_prefix_caching=True,
        enforce_eager=True,
        max_logprobs=16,
    )
    solution_diagnostics = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=16,
        seed=42,
    )
    sampled_only = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=0,
        seed=42,
    )
    solution_logprobs, solution_sequence_ids = _score_exact_response(
        llm,
        solution_diagnostics,
        prompt_ids=solution_prompt_ids,
        response_ids=response_ids,
    )
    base_logprobs, base_sequence_ids = _score_exact_response(
        llm,
        sampled_only,
        prompt_ids=base_prompt_ids,
        response_ids=response_ids,
    )
    if solution_sequence_ids[-len(response_ids) :] != response_ids:
        raise AssertionError("Solution Teacher lost the Student response ID suffix.")
    if base_sequence_ids[-len(response_ids) :] != response_ids:
        raise AssertionError("Base Teacher lost the Student response ID suffix.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_path.name,
            "student_model_name": student["model_name"],
            "proxy_used": bool(proxy_used),
            "response_ids": torch.tensor([response_ids], dtype=torch.long),
            "solution_prompt_ids": torch.tensor(
                [solution_prompt_ids], dtype=torch.long
            ),
            "base_prompt_ids": torch.tensor([base_prompt_ids], dtype=torch.long),
            "solution_sequence_ids": torch.tensor(
                [solution_sequence_ids], dtype=torch.long
            ),
            "base_sequence_ids": torch.tensor(
                [base_sequence_ids], dtype=torch.long
            ),
            "solution_teacher_logprobs": torch.tensor(
                [solution_logprobs], dtype=torch.float32
            ),
            "base_teacher_logprobs": torch.tensor(
                [base_logprobs], dtype=torch.float32
            ),
        },
        output_path,
    )
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    print(
        "SOL TEACHER DUAL-FORWARD PASS: "
        f"model={model_path.name}, proxy={proxy_used}, "
        f"solution_prompt_tokens={len(solution_prompt_ids)}, "
        f"base_prompt_tokens={len(base_prompt_ids)}, "
        f"response_tokens={len(response_ids)}, peak_allocated_gib={peak_gib:.2f}"
    )
    del llm, tokenizer, student
    _release_cuda()


def student_backward(
    model_path: Path,
    student_path: Path,
    teacher_path: Path,
    *,
    selected_tokens: int,
) -> None:
    """Run a real Student forward/backward/step using saved Sol advantages."""

    _require_cuda()
    from transformers import AutoModelForCausalLM

    student = torch.load(student_path, map_location="cpu", weights_only=True)
    teacher = torch.load(teacher_path, map_location="cpu", weights_only=True)
    torch.testing.assert_close(student["response_ids"], teacher["response_ids"])
    prompt_ids = student["prompt_ids"].long()
    response_ids = student["response_ids"].long()
    prompt_length = prompt_ids.shape[1]
    response_length = response_ids.shape[1]
    sequence_length = prompt_length + response_length
    if sequence_length > 10240:
        raise AssertionError("Student backward input exceeds formal max_model_len=10240.")
    if not 1 <= selected_tokens <= response_length:
        raise ValueError("selected_tokens must lie inside the response.")

    torch.cuda.reset_peak_memory_stats()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).to("cuda")
    model.train()
    model.config.use_cache = False

    # A one-parameter optimizer smoke keeps the 1024-token local phase light
    # while still traversing the real model, causal indexing, LM head,
    # detached advantage, backward, and optimizer-step code paths.
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    named_parameters = list(model.named_parameters())
    try:
        probe_name, probe_parameter = next(
            (name, parameter)
            for name, parameter in reversed(named_parameters)
            if name.endswith("norm.weight") and parameter.numel() < 100_000
        )
    except StopIteration as exc:
        raise AssertionError("Could not find a small final norm parameter.") from exc
    probe_parameter.requires_grad_(True)
    optimizer = torch.optim.AdamW([probe_parameter], lr=1.0e-2)
    before_step = probe_parameter.detach().float().clone()

    input_ids = torch.cat((prompt_ids, response_ids), dim=-1).to("cuda")
    attention_mask = student["attention_mask"].to("cuda")
    position_ids = student["position_ids"].to("cuda")
    response_indexes = torch.linspace(
        0,
        response_length - 1,
        steps=selected_tokens,
        dtype=torch.float64,
    ).round().long().unique().to("cuda")
    causal_rows = prompt_length + response_indexes - 1
    target_ids = response_ids.to("cuda").index_select(1, response_indexes)
    solution_teacher_logprobs = teacher[
        "solution_teacher_logprobs"
    ].to("cuda").index_select(1, response_indexes)

    optimizer.zero_grad(set_to_none=True)
    outputs = model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=False,
        return_dict=True,
    )
    selected_hidden = outputs.last_hidden_state.index_select(1, causal_rows)
    selected_logits = model.lm_head(selected_hidden).float()
    selected_logprobs = selected_logits.gather(
        -1, target_ids.unsqueeze(-1)
    ).squeeze(-1) - torch.logsumexp(selected_logits, dim=-1)
    advantage = (
        solution_teacher_logprobs - selected_logprobs
    ).detach().clamp(-20.0, 20.0)
    policy_loss = -(advantage * selected_logprobs).mean()
    policy_loss.backward()
    if probe_parameter.grad is None or not bool(
        torch.isfinite(probe_parameter.grad).all()
    ):
        raise AssertionError("Student backward produced an invalid gradient.")
    optimizer.step()
    after_step = probe_parameter.detach().float()
    if torch.equal(before_step, after_step):
        raise AssertionError("Student optimizer step did not update its probe parameter.")
    peak_gib = torch.cuda.max_memory_allocated() / 2**30
    print(
        "SOL STUDENT BACKWARD PASS: "
        f"model={model_path.name}, sequence_tokens={sequence_length}, "
        f"loss_tokens={response_indexes.numel()}, probe={probe_name}, "
        f"loss={policy_loss.detach().item():.6f}, peak_allocated_gib={peak_gib:.2f}"
    )
    del outputs, selected_hidden, selected_logits, selected_logprobs, advantage
    del policy_loss, optimizer, before_step, after_step, probe_parameter
    del input_ids, attention_mask, position_ids, response_indexes, causal_rows
    del target_ids, solution_teacher_logprobs, model, student, teacher
    _release_cuda()


def joint_fixture(student_path: Path, teacher_path: Path) -> None:
    """Exercise aligned Sol/base advantages and policy gradients on CPU."""

    from verl.trainer.distillation import losses
    from verl.trainer.ppo.core_algos import compute_policy_loss_reinforce

    student = torch.load(student_path, map_location="cpu", weights_only=True)
    teacher = torch.load(teacher_path, map_location="cpu", weights_only=True)
    response_ids = student["response_ids"].long()
    torch.testing.assert_close(teacher["response_ids"].long(), response_ids)
    solution_prompt_length = teacher["solution_prompt_ids"].shape[1]
    base_prompt_length = teacher["base_prompt_ids"].shape[1]
    torch.testing.assert_close(
        teacher["solution_sequence_ids"][:, solution_prompt_length:], response_ids
    )
    torch.testing.assert_close(
        teacher["base_sequence_ids"][:, base_prompt_length:], response_ids
    )
    response_length = response_ids.shape[1]
    prompt_length = student["prompt_ids"].shape[1]
    torch.testing.assert_close(
        student["causal_rows"],
        torch.arange(
            prompt_length - 1, prompt_length + response_length - 1
        ).unsqueeze(0),
    )

    trainable_student = student["student_logprobs"].float().requires_grad_(True)
    solution_teacher = teacher["solution_teacher_logprobs"].float()
    base_teacher = teacher["base_teacher_logprobs"].float()
    response_mask = torch.ones_like(trainable_student, dtype=torch.bool)
    distillation_config = SimpleNamespace(
        distillation_loss=SimpleNamespace(
            selection_ratio=1.0,
            selection_method="random",
            sol_opd_epsilon=1.0e-6,
            loss_max_clamp=20.0,
        )
    )
    data = {
        "teacher_logprobs": solution_teacher.unsqueeze(-1),
        "sol_unprivileged_teacher_logprobs": base_teacher.unsqueeze(-1),
        "response_mask": response_mask,
    }
    with patch.object(losses, "no_padding_2_padding", lambda tensor, _data: tensor):
        reverse_kl, metrics = losses.compute_solution_privileged_reverse_kl(
            None,
            distillation_config,
            {"log_probs": trainable_student},
            data,
        )
    torch.testing.assert_close(
        reverse_kl,
        trainable_student - solution_teacher,
    )
    advantage = -reverse_kl.detach()
    policy_loss, _ = compute_policy_loss_reinforce(
        rollout_log_prob=trainable_student.detach(),
        log_prob=trainable_student,
        advantages=advantage,
        response_mask=response_mask,
        loss_agg_mode="token-mean",
        config=SimpleNamespace(global_batch_info={}),
    )
    policy_loss.backward()
    if trainable_student.grad is None or not bool(
        torch.isfinite(trainable_student.grad).all()
    ):
        raise AssertionError("Joint Sol-OPD fixture produced invalid gradients.")

    values = {key: float(metric.aggregate()) for key, metric in metrics.items()}
    prefix = "distillation/sol_stats/"
    magnitude_ratio = values[f"{prefix}sol_abs_sum"] / max(
        values[f"{prefix}opd_abs_sum"], 1.0e-12
    )
    deviation_ratio = values[f"{prefix}deviation_abs_sum"] / max(
        values[f"{prefix}opd_abs_sum"], 1.0e-12
    )
    print(
        "SOL JOINT FIXTURE PASS: "
        f"response_tokens={response_length}, R_mag={magnitude_ratio:.6f}, "
        f"R_tsd={deviation_ratio:.6f}, loss={policy_loss.detach().item():.6f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="phase", required=True)

    audit = subparsers.add_parser("dataset-audit")
    audit.add_argument("--model", type=Path, required=True)
    audit.add_argument("--data", type=Path, required=True)

    rollout = subparsers.add_parser("student-rollout")
    rollout.add_argument("--model", type=Path, required=True)
    rollout.add_argument("--output", type=Path, required=True)
    rollout.add_argument("--response-tokens", type=int, required=True)

    teacher = subparsers.add_parser("teacher-dual")
    teacher.add_argument("--model", type=Path, required=True)
    teacher.add_argument("--student", type=Path, required=True)
    teacher.add_argument("--output", type=Path, required=True)
    teacher.add_argument("--proxy-used", action="store_true")

    backward = subparsers.add_parser("student-backward")
    backward.add_argument("--model", type=Path, required=True)
    backward.add_argument("--student", type=Path, required=True)
    backward.add_argument("--teacher", type=Path, required=True)
    backward.add_argument("--selected-tokens", type=int, default=16)

    joint = subparsers.add_parser("joint")
    joint.add_argument("--student", type=Path, required=True)
    joint.add_argument("--teacher", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "dataset-audit":
        dataset_audit(args.model.resolve(), args.data.resolve())
    elif args.phase == "student-rollout":
        student_rollout(
            args.model.resolve(),
            args.output.resolve(),
            response_tokens=args.response_tokens,
        )
    elif args.phase == "teacher-dual":
        teacher_dual_forward(
            args.model.resolve(),
            args.student.resolve(),
            args.output.resolve(),
            proxy_used=args.proxy_used,
        )
    elif args.phase == "student-backward":
        student_backward(
            args.model.resolve(),
            args.student.resolve(),
            args.teacher.resolve(),
            selected_tokens=args.selected_tokens,
        )
    else:
        joint_fixture(args.student.resolve(), args.teacher.resolve())


if __name__ == "__main__":
    main()
