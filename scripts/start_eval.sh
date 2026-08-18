#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/start_eval.sh CONFIG.yaml [Hydra overrides...]" >&2
  exit 2
fi
if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux is required." >&2
  exit 1
fi
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "No Conda environment is active. Activate the OPD_Forge environment first." >&2
  exit 1
fi
if ! command -v python >/dev/null 2>&1; then
  echo "python is not available in the active environment." >&2
  exit 1
fi

project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
config_file=$(realpath "$1")
shift
if [[ ! -f "$config_file" ]]; then
  echo "Config not found: $config_file" >&2
  exit 2
fi

config_dir=$(dirname "$config_file")
config_name=$(basename "$config_file" .yaml)
session=${TMUX_SESSION:-eval_$config_name}
session=${session//./_}
if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $session" >&2
  exit 1
fi

printf -v command 'cd %q && python -m runners.evaluation_entrypoint --config-path %q --config-name %q %q' \
  "$project_root" "$config_dir" "$config_name" \
  "hydra.searchpath=[file://$project_root/configs]"
for override in "$@"; do
  printf -v command '%s %q' "$command" "$override"
done
tmux_env=(-e "PATH=$PATH" -e "CONDA_PREFIX=$CONDA_PREFIX")
for variable in CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER PYTHONPATH LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES WANDB_API_KEY WANDB_ENTITY; do
  if [[ -n "${!variable:-}" ]]; then
    tmux_env+=(-e "$variable=${!variable}")
  fi
done
tmux new-session -d -s "$session" "${tmux_env[@]}" "bash -c $(printf '%q' "$command")"
echo "Evaluation started in tmux session: $session"
echo "Environment: ${CONDA_DEFAULT_ENV:-$CONDA_PREFIX}"
echo "Attach with: tmux attach -t $session"
