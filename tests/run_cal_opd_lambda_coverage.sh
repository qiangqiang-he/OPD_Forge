#!/usr/bin/env bash

# Collect one matched 2048-question rollout batch for each Qwen3 thinking mode.

set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
workspace_root=$(cd "$project_root/.." && pwd)
python_bin=${RLVR_PYTHON_BIN:-$workspace_root/anaconda3/envs/verl/bin/python}
ray_bin=${RLVR_RAY_BIN:-$workspace_root/anaconda3/envs/verl/bin/ray}

cd "$project_root"
export PYTHONPATH="$project_root/verl:$project_root${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export WANDB_MODE=disabled
unset WANDB_API_KEY WANDB_RUN_ID WANDB_RESUME

configs=(
  cal_opd_qwen3_4b_to_1p7b_no_thinking_lambda_coverage_2048
  cal_opd_qwen3_4b_to_1p7b_thinking_lambda_coverage_2048
)

cleanup_ray() {
  "$ray_bin" stop --force >/dev/null 2>&1 || true
}
trap cleanup_ray EXIT

for config_name in "${configs[@]}"; do
  run_name="temp_${config_name}"
  output_dir="$project_root/outputs/$run_name"
  stats_path="$output_dir/cal_opd_lambda_stats.jsonl"

  if [[ -f "$stats_path" ]] && [[ $(wc -l < "$stats_path") -eq 1 ]]; then
    echo "COVERAGE_SKIP completed $run_name"
    continue
  fi
  if [[ -e "$output_dir" ]]; then
    echo "Refusing to mix results with incomplete output: $output_dir" >&2
    exit 1
  fi

  mkdir -p "$output_dir"
  cleanup_ray
  echo "COVERAGE_START $run_name $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  "$python_bin" tests/run_cal_opd_lambda_experiment.py \
    --config-path "$project_root/configs/temp" \
    --config-name "$config_name" \
    "hydra.searchpath=[file://$project_root/configs,pkg://verl.trainer.config]" \
    2>&1 | tee "$output_dir/rollout.log"

  if [[ ! -f "$stats_path" ]] || [[ $(wc -l < "$stats_path") -ne 1 ]]; then
    echo "Expected one statistics row, got: $stats_path" >&2
    exit 1
  fi
  echo "COVERAGE_DONE $run_name $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
done

echo "COVERAGE_COMPLETE $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
