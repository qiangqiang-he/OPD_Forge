#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/start_model_eval.sh configs/eval/CONFIG.yaml [--validate-only]" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "$script_dir/.." && pwd)
workspace_root=$(cd -- "$project_root/.." && pwd)

config_argument=$1
shift
if [[ "$config_argument" == /* ]]; then
  config_candidate=$config_argument
elif [[ -f "$PWD/${config_argument#./}" ]]; then
  config_candidate=$PWD/${config_argument#./}
else
  config_candidate=$project_root/${config_argument#./}
fi
if [[ ! -f "$config_candidate" ]]; then
  echo "Evaluation config does not exist: $config_candidate" >&2
  exit 2
fi
config_file=$(realpath "$config_candidate")
if [[ ! -f "$config_file" ]]; then
  echo "Evaluation config does not exist: $config_file" >&2
  exit 2
fi

eval_config_root=$project_root/configs/eval
case "$config_file" in
  "$eval_config_root"/*.yaml | "$eval_config_root"/*.yml) ;;
  *)
    echo "Evaluation configs must be located directly under $eval_config_root" >&2
    exit 2
    ;;
esac

conda_root=${RLVR_CONDA_ROOT:-$workspace_root/anaconda3}
if [[ ! -f "$conda_root/etc/profile.d/conda.sh" ]]; then
  echo "Cannot locate Conda initialization under $conda_root" >&2
  exit 1
fi
source "$conda_root/etc/profile.d/conda.sh"
conda activate "${RLVR_CONDA_ENV:-verl}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$project_root/verl:$project_root${PYTHONPATH:+:$PYTHONPATH}"

cd "$project_root"
exec python -m utils.evaluate_models --config "$config_file" "$@"
