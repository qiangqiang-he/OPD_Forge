#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/start_train.sh CONFIG.yaml [Hydra overrides...]" >&2
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
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  echo "WANDB_API_KEY is not set. Export it before starting training." >&2
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

session=${TMUX_SESSION:-train_$(basename "$config_file" .yaml)}
session=${session//./_}
if tmux has-session -t "$session" 2>/dev/null; then
  echo "tmux session already exists: $session" >&2
  exit 1
fi

printf -v command 'cd %q && bash scripts/run_training.sh %q' "$project_root" "$config_file"
for override in "$@"; do
  printf -v command '%s %q' "$command" "$override"
done

# A long-lived tmux server may contain an older environment. Pass the active
# Conda/Python/CUDA/W&B values explicitly when creating this session.
tmux_env=(-e "PATH=$PATH" -e "CONDA_PREFIX=$CONDA_PREFIX" -e "WANDB_API_KEY=$WANDB_API_KEY")
for variable in CONDA_DEFAULT_ENV CONDA_PROMPT_MODIFIER PYTHONPATH LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES WANDB_ENTITY RUN_NAME RUN_PREFIX GROUP_NAME; do
  if [[ -n "${!variable:-}" ]]; then
    tmux_env+=(-e "$variable=${!variable}")
  fi
done
tmux new-session -d -s "$session" "${tmux_env[@]}" "bash -c $(printf '%q' "$command")"
echo "Training started in tmux session: $session"
echo "Environment: ${CONDA_DEFAULT_ENV:-$CONDA_PREFIX}"
echo "Attach with: tmux attach -t $session"
