#!/usr/bin/env bash

# Run the six local-only, five-step Cal-OPD lambda experiments sequentially.

set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
workspace_root=$(cd "$project_root/.." && pwd)
conda_root=${RLVR_CONDA_ROOT:-$workspace_root/anaconda3}
source "$conda_root/etc/profile.d/conda.sh"
conda activate "${RLVR_CONDA_ENV:-verl}"

cd "$project_root"
export PYTHONPATH="$project_root/verl:$project_root${PYTHONPATH:+:$PYTHONPATH}"
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export WANDB_MODE=disabled
unset WANDB_API_KEY WANDB_RUN_ID WANDB_RESUME

configs=(
  cal_opd_qwen3_4b_to_1p7b_no_thinking_lambda1_5steps
  cal_opd_qwen3_4b_to_1p7b_no_thinking_lambda2_5steps
  cal_opd_qwen3_4b_to_1p7b_no_thinking_lambda3_5steps
  cal_opd_qwen3_4b_to_1p7b_thinking_lambda1_5steps
  cal_opd_qwen3_4b_to_1p7b_thinking_lambda2_5steps
  cal_opd_qwen3_4b_to_1p7b_thinking_lambda3_5steps
)

cleanup_ray() {
  ray stop --force >/dev/null 2>&1 || true
}
trap cleanup_ray EXIT

for config_name in "${configs[@]}"; do
  config_path="$project_root/configs/temp/$config_name.yaml"
  run_name=$(awk -F: '/^run_name:/ {gsub(/[[:space:]]/, "", $2); print $2; exit}' "$config_path")
  output_dir="$project_root/outputs/$run_name"
  stats_path="$output_dir/cal_opd_lambda_stats.jsonl"

  if [[ -f "$stats_path" ]] && [[ $(wc -l < "$stats_path") -eq 5 ]]; then
    echo "SWEEP_SKIP completed $run_name"
    continue
  fi
  if [[ -e "$output_dir" ]]; then
    echo "Refusing to mix results with incomplete output: $output_dir" >&2
    exit 1
  fi

  mkdir -p "$output_dir"
  cleanup_ray
  echo "SWEEP_START $run_name $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  python tests/run_cal_opd_lambda_experiment.py \
    --config-path "$project_root/configs/temp" \
    --config-name "$config_name" \
    "hydra.searchpath=[file://$project_root/configs,pkg://verl.trainer.config]" \
    2>&1 | tee "$output_dir/training.log"

  if [[ ! -f "$stats_path" ]] || [[ $(wc -l < "$stats_path") -ne 5 ]]; then
    echo "Expected five statistics rows, got: $stats_path" >&2
    exit 1
  fi
  echo "SWEEP_DONE $run_name $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
done

echo "SWEEP_COMPLETE $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
