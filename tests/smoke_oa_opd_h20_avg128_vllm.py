#!/usr/bin/env python3
"""One-request Student/vLLM compatibility smoke test for the Avg@128 runner."""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from tests import run_oa_opd_h20_avg128 as runner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--role", choices=("student", "teacher"), default="student")
    parser.add_argument("--model", type=Path, default=runner.DEFAULT_STUDENT_MODEL)
    parser.add_argument("--dataset", type=Path, default=runner.DEFAULT_DATASET["correct"])
    args = parser.parse_args()
    runner._set_cuda_visibility(args.gpu)
    _, cases = runner.load_dataset(args.dataset, "correct")
    case = cases[0]
    opts = argparse.Namespace(gpu_id=args.gpu, min_free_gpu_gib=5.0,
                              gpu_memory_utilization=0.75, heartbeat_seconds=30.0,
                              memory_check_seconds=2.0)
    llm = None
    try:
        llm = runner._new_llm(args.model, opts, runner.derive_max_model_len(cases))
        if args.role == "student":
            outputs = runner._generate(
                llm,
                [case["keep_input_token_ids"]],
                [runner._sampling(2, 1, runner.continuation_seed(case["case_index"]))],
                opts,
                "compatibility_smoke/student",
            )
            assert len(outputs) == 1 and len(outputs[0].outputs) == 2
        else:
            outputs = runner._generate(
                llm,
                [case["pre_step_input_token_ids"]],
                [runner._sampling(1, runner.TEACHER_PROPOSAL_MAX_NEW_TOKENS,
                                  runner.proposal_seed(case["case_index"], 1))],
                opts,
                "compatibility_smoke/teacher",
            )
            assert len(outputs) == 1 and len(outputs[0].outputs) == 1
            row = runner._proposal_record(
                case, outputs[0], runner._load_tokenizer(args.model),
                runner.proposal_seed(case["case_index"], 1), 1, args.model
            )
            assert row["replacement_step_token_ids"]
        print("REAL_VLLM_OK", flush=True)
    finally:
        if llm is not None:
            runner.base.shutdown_vllm(llm)
        gc.collect()


if __name__ == "__main__":
    main()
