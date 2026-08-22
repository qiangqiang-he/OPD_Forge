#!/usr/bin/env bash

# Wait for local training jobs to release the GPUs, then keep the long-running
# evaluation watcher alive.  The waiter and watcher use separate sessions so a
# task-local cleanup of the evaluation cannot remove this retry loop.

set -u

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${RESOURCE_WAITER_PYTHON:-/mnt/afs/infrastructure_dev/rlvr_opd/anaconda3/envs/verl/bin/python}"
watcher_path="$project_dir/runners/watch_repeated_avg16.py"
success_path="$project_dir/outputs/repeated_avg16_thinking_5datasets/SUCCESS.json"
retry_seconds="${RESOURCE_WAITER_RETRY_SECONDS:-60}"
monitor_seconds="${RESOURCE_WAITER_MONITOR_SECONDS:-10}"
watch_log="$project_dir/outputs/repeated_avg16_thinking_5datasets/watchdog.log"

cd "$project_dir"

training_is_active() {
  local pid cmdline

  if pgrep -f '[s]cripts/run_training\.sh' >/dev/null; then
    return 0
  fi

  # Hydra config expansion uses the training entrypoint too, but --cfg job does
  # not initialize Ray or touch a GPU.  Only treat a real entrypoint invocation
  # as a workload that must preempt evaluation.
  while read -r pid; do
    [[ -n "$pid" && -r "/proc/$pid/cmdline" ]] || continue
    cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline")"
    if [[ "$cmdline" != *"--cfg job"* ]]; then
      return 0
    fi
  done < <(pgrep -f '[r]unners\.training_entrypoint' || true)

  return 1
}

waiter_log() {
  printf '%s RESOURCE_WAITER %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%S%:z')" "$*" >> "$watch_log"
}

terminate_process_group() {
  local pid="$1"
  local pgid

  [[ -n "$pid" ]] || return 0
  pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')"
  [[ -n "$pgid" ]] || return 0
  kill -TERM -- "-$pgid" 2>/dev/null || true
}

pause_evaluation() {
  local watcher_pid="$1"
  local pid

  # Stop the watcher first so it cannot restart an evaluator while a training
  # job is claiming the GPUs.  Evaluators use their own process groups, so all
  # vLLM children are released together while atomic batch shards stay intact.
  terminate_process_group "$watcher_pid"
  while read -r pid; do
    terminate_process_group "$pid"
  done < <(pgrep -f '[e]valuate_repeated_avg16\.py' || true)

  for _ in $(seq 1 30); do
    if ! pgrep -f '[e]valuate_repeated_avg16\.py' >/dev/null; then
      return 0
    fi
    sleep 1
  done

  # TERM normally tears vLLM down promptly.  Limit KILL to the evaluation
  # process groups if a worker is stuck so the incoming training job is safe.
  while read -r pid; do
    local pgid
    pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')"
    [[ -n "$pgid" ]] && kill -KILL -- "-$pgid" 2>/dev/null || true
  done < <(pgrep -f '[e]valuate_repeated_avg16\.py' || true)
}

while [[ ! -f "$success_path" ]]; do
  if training_is_active; then
    sleep "$retry_seconds"
    continue
  fi

  setsid "$python_bin" "$watcher_path" --interval-seconds 300 &
  watcher_pid=$!
  waiter_log "WATCHER_LAUNCHED pid=$watcher_pid"

  while kill -0 "$watcher_pid" 2>/dev/null; do
    if training_is_active; then
      waiter_log "TRAINING_DETECTED pausing_watcher=$watcher_pid"
      pause_evaluation "$watcher_pid"
      break
    fi
    sleep "$monitor_seconds"
  done

  wait "$watcher_pid" 2>/dev/null || true

  if [[ -f "$success_path" ]]; then
    break
  fi

  sleep "$retry_seconds"
done
