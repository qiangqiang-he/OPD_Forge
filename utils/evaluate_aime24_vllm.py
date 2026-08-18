"""Evaluate one HuggingFace-format model on AIME24 with vLLM."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils.math_verifier import extract_final_answer, verify_response_answer
from utils.prompts import render_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--question-shard", type=int, default=0)
    parser.add_argument("--num-question-shards", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.data.open() as handle:
        examples = json.load(handle)
    indexed_examples = [
        (index, example)
        for index, example in enumerate(examples)
        if index % args.num_question_shards == args.question_shard
    ]

    # Load the tokenizer explicitly to fail before allocating vLLM memory if a
    # merged checkpoint is incomplete.
    AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=args.max_new_tokens + 2048,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=0.90,
        enable_prefix_caching=True,
        seed=args.seed,
    )
    sampling = SamplingParams(
        n=16,
        temperature=0.6,
        top_p=0.95,
        top_k=-1,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )
    prompt_name = "qwen3_no_thinking_prompt"
    outputs = llm.generate(
        [render_prompt(prompt_name, question=example["question"]) for _, example in indexed_examples], sampling
    )

    records = []
    for (question_index, example), request_output in zip(indexed_examples, outputs, strict=True):
        if len(request_output.outputs) != 16:
            raise RuntimeError(f"Question {question_index} returned {len(request_output.outputs)} outputs, expected 16")
        for sample_index, completion in enumerate(request_output.outputs):
            text = completion.text
            answer, format_valid = extract_final_answer(text)
            correct = int(verify_response_answer(text, str(example["answer"])))
            records.append(
                {
                    "question_index": question_index,
                    "sample_index": sample_index,
                    "ground_truth": str(example["answer"]),
                    "extracted_answer": answer,
                    "correct": correct,
                    "format_valid": int(format_valid),
                    "finish_reason": completion.finish_reason,
                    "num_tokens": len(completion.token_ids),
                    "response": text,
                }
            )

    result = {
        "model_name": args.model_name,
        "model": str(args.model),
        "dataset": str(args.data),
        "student_prompt": prompt_name,
        "temperature": 0.6,
        "top_p": 0.95,
        "n": 16,
        "max_new_tokens": args.max_new_tokens,
        "question_shard": args.question_shard,
        "num_question_shards": args.num_question_shards,
        "num_questions": len(indexed_examples),
        "num_samples": len(records),
        "avg_at_16": sum(record["correct"] for record in records) / len(records),
        "format_valid_rate": sum(record["format_valid"] for record in records) / len(records),
        "truncated_rate": sum(record["finish_reason"] == "length" for record in records) / len(records),
        "mean_response_tokens": sum(record["num_tokens"] for record in records) / len(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, args.output)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
