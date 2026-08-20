#!/usr/bin/env bash

# Keep the long-running evaluation recoverable when a task-local cleanup kills
# its Python processes.  Resource ownership is still decided by the watcher's
# advisory lock, so this guard never bypasses another authorized GPU workload.

set -u

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${EVAL_GUARD_PYTHON:-/mnt/afs/infrastructure_dev/rlvr_opd/anaconda3/envs/verl/bin/python}"
watcher_path="$project_dir/runners/watch_repeated_avg16.py"
success_path="$project_dir/outputs/repeated_avg16_thinking_5datasets/SUCCESS.json"
retry_seconds="${EVAL_GUARD_RETRY_SECONDS:-60}"

cd "$project_dir"

while [[ ! -f "$success_path" ]]; do
  # Some local training entrypoints do not take the evaluation lock.  Treat
  # their process presence as an additional resource reservation so the guard
  # cannot race them during startup.
  if pgrep -f '[r]unners\.training_entrypoint|[s]cripts/run_training\.sh' >/dev/null; then
    printf '%s training process detected; retrying in %ss\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$retry_seconds"
    sleep "$retry_seconds"
    continue
  fi

  # Keep the guard outside the watcher's process group.  A task-local cleanup
  # may terminate the watcher/evaluator group, but must not remove this retry
  # loop as collateral damage.
  setsid --wait "$python_bin" "$watcher_path" --interval-seconds 300
  rc=$?

  if [[ -f "$success_path" ]]; then
    break
  fi

  printf '%s watcher exited rc=%s; retrying in %ss\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$retry_seconds"
  sleep "$retry_seconds"
done
