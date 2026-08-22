#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source /mnt/afs/infrastructure_dev/rlvr_opd/anaconda3/bin/activate
conda activate verl

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
export WANDB_MODE=disabled
export PYTHONPATH="$project_root/verl:$project_root${PYTHONPATH:+:$PYTHONPATH}"

if [[ $(python -c 'import torch; print(torch.cuda.device_count())') -ne 8 ]]; then
  echo "The unified OPD smoke test requires exactly 8 visible GPUs." >&2
  exit 1
fi

mkdir -p "$project_root/outputs/smoke_logs"

run_smoke() {
  local config_name=$1
  local log_file="$project_root/outputs/smoke_logs/${config_name}.log"
  ray stop --force >/dev/null 2>&1 || true
  python -m runners.training_entrypoint \
    --config-path "$project_root/configs/smoke" \
    --config-name "$config_name" \
    "hydra.searchpath=[file://$project_root/configs,pkg://verl.trainer.config]" \
    2>&1 | tee "$log_file"
  grep -q "step:2" "$log_file"
  grep -q "val-amc23-core/Avg@4" "$log_file"
}

trap 'ray stop --force >/dev/null 2>&1 || true' EXIT
cd "$project_root"
run_smoke gkd_opd_thinking_2steps
run_smoke pg_opd_no_thinking_2steps
echo "Unified OPD thinking/no-thinking smoke tests passed."
