#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd -- "$script_dir/.." && pwd)
workspace_root=${RLVR_WORKSPACE_ROOT:-$(cd -- "$project_root/.." && pwd)}
if [[ -n "${RLVR_CONDA_ROOT:-}" ]]; then
  conda_root=$RLVR_CONDA_ROOT
elif command -v conda >/dev/null 2>&1; then
  conda_root=$(conda info --base)
elif [[ -d "$workspace_root/anaconda3" ]]; then
  conda_root=$workspace_root/anaconda3
else
  echo "Cannot locate Conda. Activate it first or set RLVR_CONDA_ROOT." >&2
  exit 1
fi
source "$conda_root/etc/profile.d/conda.sh"
conda activate "${RLVR_CONDA_ENV:-verl}"

export RLVR_PROJECT_ROOT=$project_root
export RLVR_WORKSPACE_ROOT=$workspace_root
config_root=$project_root/configs
cd "$project_root"
launch_args=("$@")

if [[ $# -lt 1 || "$1" != *.yaml ]]; then
  echo "Usage: bash scripts/run_training.sh CONFIG.yaml [Hydra overrides...]" >&2
  exit 2
fi

config_file=$1
shift
if [[ "$config_file" != /* ]]; then
  config_file=$project_root/${config_file#./}
fi
if [[ ! -f "$config_file" ]]; then
  echo "Training config does not exist: $config_file" >&2
  exit 2
fi

config_file=$(realpath "$config_file")
case "$config_file" in
  "$config_root"/*.yaml) ;;
  *)
    echo "Training configs must be located under $config_root" >&2
    exit 2
    ;;
esac
config_dir=$(dirname "$config_file")
config_name=$(basename "$config_file" .yaml)

# Optional resume controls.  They are regular launcher arguments so they are
# preserved when the outer invocation re-enters this script inside tmux.
resume_run_name=${RESUME_RUN_NAME:-}
wandb_run_id=${WANDB_RUN_ID:-}
wandb_skip_until_step=${RLVR_WANDB_SKIP_UNTIL_STEP:-}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume-run-name)
      [[ $# -ge 2 ]] || { echo "--resume-run-name requires a value" >&2; exit 2; }
      resume_run_name=$2
      shift 2
      ;;
    --wandb-run-id)
      [[ $# -ge 2 ]] || { echo "--wandb-run-id requires a value" >&2; exit 2; }
      wandb_run_id=$2
      shift 2
      ;;
    --wandb-skip-until-step)
      [[ $# -ge 2 ]] || { echo "--wandb-skip-until-step requires a value" >&2; exit 2; }
      wandb_skip_until_step=$2
      shift 2
      ;;
    *)
      break
      ;;
  esac
done

export WANDB_MODE=${WANDB_MODE:-online}
: "${WANDB_API_KEY:?WANDB_API_KEY must be set before starting training.}"
export WANDB_API_KEY
export WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-600}

# Training is detached by default so it survives SSH and terminal disconnects.
# The fixed session name is intentional: this machine runs at most one local
# experiment at a time. Inside tmux, continue directly to the training logic.
if [[ -z "${TMUX:-}" ]]; then
  tmux_session=${TMUX_SESSION_NAME:-rlvr_training}
  if tmux has-session -t "=$tmux_session" 2>/dev/null; then
    echo "tmux session '$tmux_session' already exists; refusing to start another training." >&2
    echo "Attach with: tmux attach -t $tmux_session" >&2
    exit 1
  fi

  tmux_env=(env)
  # tmux servers retain the environment from server creation time, so values
  # supplied only to this launcher invocation must be forwarded explicitly.
  for var_name in RUN_NAME RUN_PREFIX GROUP_NAME WANDB_MODE WANDB_API_KEY WANDB_RESUME RLVR_WORKSPACE_ROOT RLVR_CONDA_ROOT RLVR_CONDA_ENV; do
    [[ -n "${!var_name:-}" ]] && tmux_env+=("$var_name=${!var_name}")
  done
  printf -v tmux_command '%q ' "${tmux_env[@]}" bash "$project_root/scripts/run_training.sh" "${launch_args[@]}"
  tmux new-session -d -s "$tmux_session" -c "$project_root" "$tmux_command"
  echo "Training started in tmux session: $tmux_session"
  echo "Attach with: tmux attach -t $tmux_session"
  exit 0
fi

export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=true
# Always use this project's bundled VERL checkout instead of an editable
# installation that may point at the original RLVR_VERL directory.
export PYTHONPATH="$project_root/verl:$project_root${PYTHONPATH:+:$PYTHONPATH}"

resume_overrides=()
initial_step=0
config_group_name=$(awk -F: '/^[[:space:]]*group_name[[:space:]]*:/ {sub(/^[^:]*:[[:space:]]*/, ""); gsub(/^["[:space:]]+|["[:space:]]+$/, ""); print; exit}' "$config_file")
if [[ -n "$resume_run_name" ]]; then
  run_name=$resume_run_name
  output_dir=$project_root/outputs/$run_name
  tracker_file=$output_dir/latest_checkpointed_iteration.txt
  if [[ ! -f "$tracker_file" ]]; then
    echo "Resume tracker does not exist: $tracker_file" >&2
    exit 2
  fi
  resume_step=$(<"$tracker_file")
  if [[ ! "$resume_step" =~ ^[0-9]+$ || ! -d "$output_dir/global_step_$resume_step" ]]; then
    echo "Resume checkpoint is incomplete or invalid: global_step_$resume_step" >&2
    exit 2
  fi
  if [[ -z "$wandb_run_id" ]]; then
    echo "--wandb-run-id is required when --resume-run-name is used." >&2
    exit 2
  fi
  export WANDB_RUN_ID=$wandb_run_id
  # `allow` resumes the specified run when it exists while also tolerating a
  # run that has already been finalized by a previous short smoke run.
  export WANDB_RESUME=${WANDB_RESUME:-allow}
  if [[ -n "$wandb_skip_until_step" ]]; then
    [[ "$wandb_skip_until_step" =~ ^[0-9]+$ ]] || {
      echo "--wandb-skip-until-step must be a non-negative integer" >&2
      exit 2
    }
    export RLVR_WANDB_SKIP_UNTIL_STEP=$wandb_skip_until_step
  fi
  resume_overrides+=(trainer.resume_mode=auto trainer.val_before_train=false)
  initial_step=$resume_step
  echo "Resuming run '$run_name' from global_step_$resume_step; initial validation disabled."
  echo "Continuing W&B run '$WANDB_RUN_ID' (resume=$WANDB_RESUME)."
else
  config_run_name=$(awk -F: '/^[[:space:]]*run_name[[:space:]]*:/ {sub(/^[^:]*:[[:space:]]*/, ""); gsub(/^["[:space:]]+|["[:space:]]+$/, ""); print; exit}' "$config_file")
  config_run_prefix=$(awk -F: '/^[[:space:]]*run_name_prefix[[:space:]]*:/ {sub(/^[^:]*:[[:space:]]*/, ""); gsub(/["[:space:]]/, ""); print; exit}' "$config_file")
  config_run_prefix=${config_run_prefix%${config_run_prefix##*[![:space:]]}}
  run_prefix=${RUN_PREFIX:-${config_run_prefix:-${config_name//\//_}}}
  run_timestamp="$(date -u +%Y%m%d_%H%M%S)_p$$"
  run_name=${RUN_NAME:-${config_run_name:-${run_prefix}_${run_timestamp}}}
  output_dir=$project_root/outputs/$run_name
fi
group_name=${GROUP_NAME:-${config_group_name:-temp}}

token_overrides=()
if [[ -n "${PPO_MAX_TOKEN_LEN_PER_GPU:-}" ]]; then
  token_overrides+=(
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU"
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU"
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU"
  )
fi
log_file=${LOG_FILE:-$output_dir/training.log}
process_log=${PROCESS_LOG:-$output_dir/progress.log}
run_start_epoch=$(date +%s)
mkdir -p "$output_dir"

# Preserve the original console log when continuing an existing run.  A plain
# `tee "$log_file"` truncates it before Python has a chance to load the
# checkpoint, which makes resumed runs unnecessarily hard to audit.
tee_args=(--output-error=warn-nopipe)
if [[ -n "$resume_run_name" ]]; then
  tee_args+=(--append)
fi

# Never terminate another experiment implicitly. Set ALLOW_ACTIVE_TRAINING_STOP=1
# only when replacing the currently running local experiment is intentional.
if pgrep -f 'python -m runners.training_entrypoint.*--config-name' >/dev/null; then
  if [[ "${ALLOW_ACTIVE_TRAINING_STOP:-0}" != "1" ]]; then
    echo "An existing training process is active; refusing to stop it." >&2
    exit 1
  fi
  ray stop --force
else
  # Clear stale Ray state left by a previously completed or failed run.
  ray stop --force
fi

python -m runners.training_entrypoint \
  --config-path "$config_dir" \
  --config-name "$config_name" \
  "hydra.searchpath=[file://$config_root,pkg://verl.trainer.config]" \
  run_name="$run_name" \
  group_name="$group_name" \
  trainer.experiment_name="$run_name" \
  trainer.default_local_dir="$output_dir" \
  "${token_overrides[@]}" \
  "${resume_overrides[@]}" \
  "$@" \
  2>&1 | tee "${tee_args[@]}" "$log_file" >(
    python "$project_root/utils/progress_logger.py" \
      --output "$process_log" \
      --run-name "$run_name" \
      --initial-step "$initial_step" \
      --start-epoch "$run_start_epoch"
  )
