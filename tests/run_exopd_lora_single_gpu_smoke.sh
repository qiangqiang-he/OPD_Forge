#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
default_python=/home/qqh/miniconda3/envs/verl/bin/python
python_bin=${VERL_PYTHON:-$default_python}

if [[ ! -x "$python_bin" ]]; then
    python_bin=$(command -v python)
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$project_root:$project_root/verl${PYTHONPATH:+:$PYTHONPATH}"

cd "$project_root"
"$python_bin" tests/exopd_lora_single_gpu_smoke.py "$@"
