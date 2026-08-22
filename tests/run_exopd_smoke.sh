#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$project_root/../anaconda3/etc/profile.d/conda.sh"
conda activate verl

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export WANDB_MODE=offline
export WANDB_DIR="$project_root/wandb"
export PYTHONPATH="$project_root/verl:$project_root${PYTHONPATH:+:$PYTHONPATH}"

if [[ $(python -c 'import torch; print(torch.cuda.device_count())') -ne 8 ]]; then
  echo "ExOPD smoke tests require exactly 8 visible GPUs." >&2
  exit 1
fi

configs=(
  exopd_qwen3_4b_to_0p6b_no_thinking_smoke_2steps
  exopd_qwen3_4b_to_0p6b_thinking_smoke_2steps
  exopd_qwen3_4b_to_1p7b_no_thinking_smoke_2steps
  exopd_qwen3_4b_to_1p7b_thinking_smoke_2steps
  exopd_qwen3_8b_to_1p7b_no_thinking_smoke_2steps
  exopd_qwen3_8b_to_1p7b_thinking_smoke_2steps
)

mkdir -p "$project_root/outputs/exopd_smoke_logs" "$WANDB_DIR"

run_smoke() {
  local config_name=$1
  local run_name="temp_${config_name}"
  local log_file="$project_root/outputs/exopd_smoke_logs/${config_name}.log"
  local output_dir="$project_root/outputs/$run_name"

  echo "START $config_name"
  ray stop --force >/dev/null 2>&1 || true
  if ! python -m runners.training_entrypoint \
    --config-path "$project_root/configs/temp" \
    --config-name "$config_name" \
    "hydra.searchpath=[file://$project_root/configs,pkg://verl.trainer.config]" \
    >"$log_file" 2>&1; then
    tail -n 200 "$log_file" >&2
    return 1
  fi

  grep -q "step:0 - val-amc23-core/" "$log_file"
  [[ $(grep -c "step:0 -" "$log_file") -eq 1 ]]
  ! grep -Eq "step:0 .*val-aime|step:0 .*val-AIME" "$log_file"
  grep -q "step:2 -" "$log_file"
  grep -q "ExOPD/train/policy_loss" "$log_file"
  grep -q "offline-run" "$log_file"
  ! grep -Eq "Error executing job|CUDA out of memory|OutOfMemoryError|RayTaskError|ActorDiedError|NCCL error" "$log_file"
  [[ -d "$output_dir/global_step_2" ]]
  [[ $(<"$output_dir/latest_checkpointed_iteration.txt") == 2 ]]
  echo "PASS $config_name"
}

trap 'ray stop --force >/dev/null 2>&1 || true' EXIT
cd "$project_root"
for config_name in "${configs[@]}"; do
  run_smoke "$config_name"
done
echo "All ExOPD GPU smoke tests passed."
