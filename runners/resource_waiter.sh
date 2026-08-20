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

cd "$project_dir"

while [[ ! -f "$success_path" ]]; do
  if pgrep -f '[r]unners\.training_entrypoint|[s]cripts/run_training\.sh' >/dev/null; then
    sleep "$retry_seconds"
    continue
  fi

  setsid --wait "$python_bin" "$watcher_path" --interval-seconds 300

  if [[ -f "$success_path" ]]; then
    break
  fi

  sleep "$retry_seconds"
done

