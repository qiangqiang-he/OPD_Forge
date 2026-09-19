#!/usr/bin/env bash
set -euo pipefail

# Analysis 03: use every GPU reported by nvidia-smi for Student answer probes,
# then release vLLM and run prompt-cluster-bootstrap statistics on CPU.
RESULTS_PATH="${1:-outputs/ersr_dapo17k_qwen3_1p7b_mc128/ersr_results.json}"
OUTPUT_DIR="${2:-outputs/ersr_dapo17k_qwen3_1p7b_mc128/probe_analysis_03}"
STUDENT_MODEL="${3:-models/Qwen3-1.7B}"
BOOTSTRAP_ROUNDS="${BOOTSTRAP_ROUNDS:-2000}"

python tests/collect_ersr_answer_probes_03.py \
  --results "${RESULTS_PATH}" \
  --student-model "${STUDENT_MODEL}" \
  --output-dir "${OUTPUT_DIR}" \
  --gpus auto

python tests/analyze_ersr_answer_probes_03.py \
  --step-scores "${OUTPUT_DIR}/03_probe_step_scores.jsonl" \
  --output-dir "${OUTPUT_DIR}" \
  --bootstrap-rounds "${BOOTSTRAP_ROUNDS}"
