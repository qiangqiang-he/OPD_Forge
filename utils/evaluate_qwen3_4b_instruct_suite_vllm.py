"""Evaluate one question shard of the five-dataset math suite with one vLLM load."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from utils.math_verifier import extract_final_answer, verify_response_answer


SYSTEM_PROMPT = (
    "You are a helpful math assistant.\n"
    "Please solve the math problem step by step clearly and concisely.\n"
    "You must enclose your final answer exactly within \\boxed{}."
)


def no_thinking_prompt(question: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


DATASETS = (
    ("AIME24", "AIME-2024.json"),
    ("AIME25", "AIME-2025.json"),
    ("AIME26", "AIME-2026.json"),
    ("AMC23", "AMC-2023.json"),
    ("MATH500", "MATH500.json"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--question-shard", type=int, required=True)
    parser.add_argument("--num-question-shards", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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

    for dataset_name, filename in DATASETS:
        output = args.output_root / dataset_name / "shards" / f"shard_{args.question_shard}.json"
        if output.exists():
            print(f"SKIP {dataset_name} shard {args.question_shard}", flush=True)
            continue
        with (args.data_root / filename).open() as handle:
            examples = json.load(handle)
        indexed_examples = [
            (index, example)
            for index, example in enumerate(examples)
            if index % args.num_question_shards == args.question_shard
        ]
        print(f"START {dataset_name} shard {args.question_shard}", flush=True)
        outputs = llm.generate(
            [no_thinking_prompt(example["question"]) for _, example in indexed_examples], sampling
        )
        records = []
        for (question_index, example), request_output in zip(indexed_examples, outputs, strict=True):
            if len(request_output.outputs) != 16:
                raise RuntimeError(
                    f"Question {question_index} returned {len(request_output.outputs)} outputs, expected 16"
                )
            for sample_index, completion in enumerate(request_output.outputs):
                text = completion.text
                answer, format_valid = extract_final_answer(text)
                records.append(
                    {
                        "question_index": question_index,
                        "sample_index": sample_index,
                        "ground_truth": str(example["answer"]),
                        "extracted_answer": answer,
                        "correct": int(verify_response_answer(text, str(example["answer"]))),
                        "format_valid": int(format_valid),
                        "finish_reason": completion.finish_reason,
                        "num_tokens": len(completion.token_ids),
                        "response": text,
                    }
                )
        result = {
            "model_name": f"Qwen3-4B-Instruct-2507-{dataset_name}",
            "model": str(args.model),
            "dataset": str(args.data_root / filename),
            "thinking": "off",
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
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + f".tmp.{os.getpid()}")
        with temporary.open("w") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
        os.replace(temporary, output)
        print(
            json.dumps({key: value for key, value in result.items() if key != "records"}, ensure_ascii=False),
            flush=True,
        )


if __name__ == "__main__":
    main()
