#!/usr/bin/env bash
set -euo pipefail

# Analysis 04 is CPU-only and reads the individual rewards already stored in
# the finalized MC@128 ERSR results.
RESULTS_PATH="${1:-outputs/ersr_dapo17k_qwen3_1p7b_mc128/ersr_results.json}"
OUTPUT_DIR="${2:-outputs/ersr_dapo17k_qwen3_1p7b_mc128/mc_analysis_04}"
REFERENCE_K="${REFERENCE_K:-128}"
K_VALUES="${K_VALUES:-2,4,8,16,32,64}"
M_VALUES="${M_VALUES:-25,50,100,200,500,1000,2000,5000,10000}"
REPETITIONS="${REPETITIONS:-1000}"
INDIVIDUAL_REPETITIONS="${INDIVIDUAL_REPETITIONS:-100}"
SAMPLING_SETTINGS="${SAMPLING_SETTINGS:-prompt_cluster,step_random}"

python tests/analyze_ersr_low_budget_mc_04.py \
  --results "${RESULTS_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --reference-k "${REFERENCE_K}" \
  --k-values "${K_VALUES}" \
  --m-values "${M_VALUES}" \
  --repetitions "${REPETITIONS}" \
  --individual-repetitions "${INDIVIDUAL_REPETITIONS}" \
  --sampling-settings "${SAMPLING_SETTINGS}"
