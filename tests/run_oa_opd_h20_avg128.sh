#!/usr/bin/env bash
# Launch the eight independent H20 workers for the OA--OPD Avg@128 study.
# Run from WSL2:
#   bash tests/run_oa_opd_h20_avg128.sh correct
#   bash tests/run_oa_opd_h20_avg128.sh incorrect
# Teacher models are configured in run_oa_opd_h20_avg128.py (TEACHER_MODELS).
# Optional flags: --dataset PATH --output-root PATH --student-model PATH
#   --teacher-key KEY (optional subset; repeat for several keys)
#   --teacher-model PATH [--teacher-key KEY] (legacy single-Teacher override)
#   --world-size 8 --python PATH --gpu-memory-utilization 0.90
#   --overwrite --validate-only --phase all --arm all --run-id ID
#   --gpu-ids 0,1,2,3,4,5,6,7 (physical IDs; useful with a scheduler)

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RUNNER="$SCRIPT_DIR/run_oa_opd_h20_avg128.py"

if [[ $# -lt 1 ]]; then
  echo "用法: $0 {correct|incorrect} [选项]" >&2
  exit 2
fi
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  sed -n '1,24p' "$0"
  exit 0
fi
OUTCOME="$1"
shift
case "$OUTCOME" in
  correct|incorrect) ;;
  *) echo "outcome 必须是 correct 或 incorrect" >&2; exit 2 ;;
esac

PYTHON_BIN="${PYTHON_BIN:-}"
WORLD_SIZE="${WORLD_SIZE:-8}"
DATASET="$REPO_ROOT/data/${OUTCOME}_2000_trajectories_10000_steps.json"
OUTPUT_ROOT="$REPO_ROOT/outputs/oa_opd_h20_avg128_20k"
STUDENT_MODEL="$REPO_ROOT/models/Qwen3-1.7B"
TEACHER_MODEL=""
TEACHER_KEYS=()
PHASE="all"
ARM="all"
OVERWRITE=0
VALIDATE_ONLY=0
GPU_UTILIZATION="0.90"
MIN_FREE_GIB="5.0"
HEARTBEAT_SECONDS="30.0"
MEMORY_CHECK_SECONDS="2.0"
MAX_MODEL_LEN="0"
RUN_ID="${RUN_ID:-}"
GPU_IDS_CSV="${GPU_IDS:-}"
ORIGINAL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset) [[ $# -ge 2 ]] || { echo "--dataset 需要路径" >&2; exit 2; }; DATASET="$2"; shift 2 ;;
    --output-root) [[ $# -ge 2 ]] || { echo "--output-root 需要路径" >&2; exit 2; }; OUTPUT_ROOT="$2"; shift 2 ;;
    --student-model) [[ $# -ge 2 ]] || { echo "--student-model 需要路径" >&2; exit 2; }; STUDENT_MODEL="$2"; shift 2 ;;
    --teacher-model) [[ $# -ge 2 ]] || { echo "--teacher-model 需要路径" >&2; exit 2; }; TEACHER_MODEL="$2"; shift 2 ;;
    --teacher-key) [[ $# -ge 2 ]] || { echo "--teacher-key 需要名称" >&2; exit 2; }; TEACHER_KEYS+=("$2"); shift 2 ;;
    --python) [[ $# -ge 2 ]] || { echo "--python 需要路径" >&2; exit 2; }; PYTHON_BIN="$2"; shift 2 ;;
    --world-size) [[ $# -ge 2 ]] || { echo "--world-size 需要整数" >&2; exit 2; }; WORLD_SIZE="$2"; shift 2 ;;
    --phase) [[ $# -ge 2 ]] || { echo "--phase 需要 all/proposals/continuations" >&2; exit 2; }; PHASE="$2"; shift 2 ;;
    --arm) [[ $# -ge 2 ]] || { echo "--arm 需要 all/keep/delete/replace" >&2; exit 2; }; ARM="$2"; shift 2 ;;
    --gpu-memory-utilization) [[ $# -ge 2 ]] || { echo "--gpu-memory-utilization 需要小数" >&2; exit 2; }; GPU_UTILIZATION="$2"; shift 2 ;;
    --min-free-gpu-gib) [[ $# -ge 2 ]] || { echo "--min-free-gpu-gib 需要数值" >&2; exit 2; }; MIN_FREE_GIB="$2"; shift 2 ;;
    --heartbeat-seconds) [[ $# -ge 2 ]] || { echo "--heartbeat-seconds 需要数值" >&2; exit 2; }; HEARTBEAT_SECONDS="$2"; shift 2 ;;
    --memory-check-seconds) [[ $# -ge 2 ]] || { echo "--memory-check-seconds 需要数值" >&2; exit 2; }; MEMORY_CHECK_SECONDS="$2"; shift 2 ;;
    --max-model-len) [[ $# -ge 2 ]] || { echo "--max-model-len 需要整数" >&2; exit 2; }; MAX_MODEL_LEN="$2"; shift 2 ;;
    --run-id) [[ $# -ge 2 ]] || { echo "--run-id 需要字符串" >&2; exit 2; }; RUN_ID="$2"; shift 2 ;;
    --gpu-ids) [[ $# -ge 2 ]] || { echo "--gpu-ids 需要逗号分隔的整数" >&2; exit 2; }; GPU_IDS_CSV="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    --validate-only) VALIDATE_ONLY=1; shift ;;
    -h|--help) sed -n '1,22p' "$0"; exit 0 ;;
    *) echo "未知选项: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  else
    echo "找不到 Python。请先激活 verl 环境，或用 --python /path/to/python 指定。" >&2
    exit 1
  fi
fi
if ! [[ "$WORLD_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "--world-size 必须是正整数" >&2
  exit 2
fi
WORLD_SIZE=$((10#$WORLD_SIZE))
if [[ ! -f "$DATASET" ]]; then echo "找不到数据集: $DATASET" >&2; exit 1; fi
if [[ ! -f "$RUNNER" ]]; then echo "找不到 runner: $RUNNER" >&2; exit 1; fi
if [[ "$PYTHON_BIN" == */* ]]; then
  [[ -x "$PYTHON_BIN" ]] || { echo "Python 不可执行: $PYTHON_BIN" >&2; exit 1; }
elif ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "找不到 Python: $PYTHON_BIN。请用 --python 指定环境中的解释器。" >&2
  exit 1
fi
if (( ! VALIDATE_ONLY )) && [[ ! -d "$STUDENT_MODEL" ]]; then
  echo "找不到 Student 模型目录: $STUDENT_MODEL" >&2
  exit 1
fi
if (( ! VALIDATE_ONLY )) && [[ -n "$TEACHER_MODEL" && ! -d "$TEACHER_MODEL" ]]; then
  echo "找不到 Teacher 模型目录: $TEACHER_MODEL" >&2
  exit 1
fi
if [[ -z "$RUN_ID" ]]; then
  RUN_ID="oa_$(date -u +%Y%m%dT%H%M%SZ)_$$"
fi

GPU_ID_LIST=()
if [[ -n "$GPU_IDS_CSV" ]]; then
  IFS=',' read -r -a GPU_ID_LIST <<< "$GPU_IDS_CSV"
elif [[ -n "$ORIGINAL_CUDA_VISIBLE_DEVICES" &&
        "$ORIGINAL_CUDA_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  # Slurm/Docker commonly exposes a numeric subset through
  # CUDA_VISIBLE_DEVICES. Treat those values as physical IDs before each
  # child narrows visibility to one card.
  IFS=',' read -r -a GPU_ID_LIST <<< "$ORIGINAL_CUDA_VISIBLE_DEVICES"
else
  for ((worker=0; worker<WORLD_SIZE; worker++)); do GPU_ID_LIST+=("$worker"); done
fi
if (( ${#GPU_ID_LIST[@]} != WORLD_SIZE )); then
  echo "GPU ID 数量 (${#GPU_ID_LIST[@]}) 必须等于 world-size ($WORLD_SIZE)" >&2
  exit 2
fi
declare -A SEEN_GPU_IDS=()
for gpu_id in "${GPU_ID_LIST[@]}"; do
  if ! [[ "$gpu_id" =~ ^[0-9]+$ ]]; then
    echo "GPU ID 必须是非负整数: $gpu_id" >&2
    exit 2
  fi
  if [[ -n "${SEEN_GPU_IDS[$gpu_id]:-}" ]]; then
    echo "GPU ID 重复，会导致多个 worker 争用同一张卡: $gpu_id" >&2
    exit 2
  fi
  SEEN_GPU_IDS[$gpu_id]=1
done
if [[ -n "$ORIGINAL_CUDA_VISIBLE_DEVICES" &&
      ! "$ORIGINAL_CUDA_VISIBLE_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ &&
      -z "$GPU_IDS_CSV" ]]; then
  echo "提示: CUDA_VISIBLE_DEVICES 不是物理整数列表；如需使用调度器分配的卡，请显式传 --gpu-ids。" >&2
fi

RUN_DIR="$OUTPUT_ROOT/$OUTCOME"
mkdir -p "$RUN_DIR/logs"
LOG_FILE="$RUN_DIR/run.log"
echo "============================================================" | tee -a "$LOG_FILE"
echo "OA-OPD Avg@128: outcome=$OUTCOME" | tee -a "$LOG_FILE"
echo "dataset=$DATASET" | tee -a "$LOG_FILE"
echo "output=$RUN_DIR" | tee -a "$LOG_FILE"
echo "run_id=$RUN_ID" | tee -a "$LOG_FILE"
echo "gpu_ids=${GPU_ID_LIST[*]}" | tee -a "$LOG_FILE"
echo "world_size=$WORLD_SIZE, each call=2 prompts x 128 = 256 rollouts" | tee -a "$LOG_FILE"
echo "teacher proposal max tokens=1024, cumulative response max tokens=10240" | tee -a "$LOG_FILE"
echo "arms=$ARM, phase=$PHASE, GPU reserve >=5 GiB" | tee -a "$LOG_FILE"
echo "============================================================" | tee -a "$LOG_FILE"

# The Python worker maps each physical --gpu-id to a single visible device.
unset CUDA_VISIBLE_DEVICES || true
export PYTHONUNBUFFERED=1

CHILD_PIDS=()
WORKER_PIDS=()
declare -A PID_TO_WORKER=()
stop_children() {
  for pid in "${CHILD_PIDS[@]:-}"; do kill -INT "$pid" 2>/dev/null || true; done
  sleep 1
  for pid in "${CHILD_PIDS[@]:-}"; do kill -TERM "$pid" 2>/dev/null || true; done
}
cleanup() {
  local code=$?
  trap - INT TERM EXIT
  stop_children
  for pid in "${CHILD_PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done
  exit "$code"
}
trap cleanup INT TERM EXIT

COMMON_ARGS=(
  --outcome "$OUTCOME"
  --dataset "$DATASET"
  --output-root "$OUTPUT_ROOT"
  --student-model "$STUDENT_MODEL"
  --world-size "$WORLD_SIZE"
  --phase "$PHASE"
  --arm "$ARM"
  --gpu-memory-utilization "$GPU_UTILIZATION"
  --min-free-gpu-gib "$MIN_FREE_GIB"
  --heartbeat-seconds "$HEARTBEAT_SECONDS"
  --memory-check-seconds "$MEMORY_CHECK_SECONDS"
  --max-model-len "$MAX_MODEL_LEN"
  --run-id "$RUN_ID"
)
if [[ -n "$TEACHER_MODEL" ]]; then COMMON_ARGS+=(--teacher-model "$TEACHER_MODEL"); fi
for teacher_key in "${TEACHER_KEYS[@]}"; do
  COMMON_ARGS+=(--teacher-key "$teacher_key")
done
if (( OVERWRITE )); then COMMON_ARGS+=(--overwrite); fi
if (( VALIDATE_ONLY )); then COMMON_ARGS+=(--validate-only); fi

# CPU-only aggregate monitor.  Its output and every worker's output are saved.
if (( ! VALIDATE_ONLY )); then
  "$PYTHON_BIN" "$RUNNER" --mode monitor "${COMMON_ARGS[@]}" \
    > >(tee -a "$LOG_FILE") 2>&1 &
  MONITOR_PID=$!
  CHILD_PIDS+=("$MONITOR_PID")
fi

for ((worker=0; worker<WORLD_SIZE; worker++)); do
  "$PYTHON_BIN" "$RUNNER" --mode worker "${COMMON_ARGS[@]}" \
    --worker-id "$worker" --gpu-id "${GPU_ID_LIST[$worker]}" \
    > >(tee -a "$RUN_DIR/logs/worker_${worker}.log" | tee -a "$LOG_FILE") 2>&1 &
  worker_pid="$!"
  CHILD_PIDS+=("$worker_pid")
  WORKER_PIDS+=("$worker_pid")
  PID_TO_WORKER["$worker_pid"]="$worker"
done

WORKER_FAILED=0
# Reap whichever worker finishes first.  A sequential ``wait worker_0;
# wait worker_1; ...`` can hide a failed GPU worker behind an earlier worker
# that is still generating for hours, defeating the memory-safety stop.  Bash
# 5.1 (the WSL/server default) provides ``wait -n -p``; retain a sequential
# fallback for older shells so the launcher remains portable.
ACTIVE_WORKER_PIDS=("${WORKER_PIDS[@]}")
if (( BASH_VERSINFO[0] > 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] >= 1) )); then
  # Bash can remove several already-completed children from its job table in
  # one SIGCHLD batch.  In that situation a subsequent `wait -n -p PID ...`
  # may return 127 with an empty PID even though the individual `wait PID`
  # calls still have the true exit statuses.  Probe process state first and
  # reap an exited PID directly; this keeps simultaneous worker completion
  # from being misreported as a failed run.
  reap_finished_worker() {
    local pid state
    for pid in "${ACTIVE_WORKER_PIDS[@]}"; do
      state="$(ps -o stat= -p "$pid" 2>/dev/null || true)"
      state="${state//[[:space:]]/}"
      if [[ -z "$state" || "$state" == Z* ]]; then
        FINISHED_PID="$pid"
        if wait "$pid"; then
          WAIT_STATUS=0
        else
          WAIT_STATUS=$?
        fi
        return 0
      fi
    done
    return 1
  }

  while (( ${#ACTIVE_WORKER_PIDS[@]} )); do
    FINISHED_PID=""
    if ! reap_finished_worker; then
      if wait -n -p FINISHED_PID "${ACTIVE_WORKER_PIDS[@]}"; then
        WAIT_STATUS=0
      else
        WAIT_STATUS=$?
      fi
      # A simultaneous-exit SIGCHLD race can leave FINISHED_PID unset.  Scan
      # again and retry instead of assigning an arbitrary active PID the
      # failure status (which was the old false-negative behavior).
      if [[ -z "$FINISHED_PID" ]]; then
        if ! reap_finished_worker; then
          sleep 0.1
          continue
        fi
      fi
    fi
    NEXT_ACTIVE_PIDS=()
    for pid in "${ACTIVE_WORKER_PIDS[@]}"; do
      [[ "$pid" == "$FINISHED_PID" ]] || NEXT_ACTIVE_PIDS+=("$pid")
    done
    # ``wait -n -p`` should identify one of the active PIDs.  Never remove an
    # arbitrary PID if a shell returns an unexpected job identifier: doing so
    # can hide a real failure or block on a still-running worker.  The latter
    # is treated as a launcher error and siblings are stopped safely.
    if [[ -z "$FINISHED_PID" || ${#NEXT_ACTIVE_PIDS[@]} -eq ${#ACTIVE_WORKER_PIDS[@]} ]]; then
      WORKER_FAILED=1
      echo "wait -n did not identify an active worker PID; stopping the run." | tee -a "$LOG_FILE"
      stop_children
      break
    fi
    ACTIVE_WORKER_PIDS=("${NEXT_ACTIVE_PIDS[@]}")
    if (( WAIT_STATUS != 0 )); then
      worker="${PID_TO_WORKER[$FINISHED_PID]:-unknown}"
      WORKER_FAILED=1
      echo "worker $worker failed (status=$WAIT_STATUS); see $RUN_DIR/logs/worker_${worker}.log" | tee -a "$LOG_FILE"
      # Stop sibling workers immediately.  In particular, a GPU safety
      # heartbeat may have interrupted one worker; do not leave the other
      # cards under load while a failed worker is being diagnosed.
      stop_children
      break
    fi
  done
else
  echo "提示: Bash <5.1，无法使用 wait -n；将按 worker 顺序回收进程。" | tee -a "$LOG_FILE"
  for worker in "${!WORKER_PIDS[@]}"; do
    if ! wait "${WORKER_PIDS[$worker]}"; then
      WORKER_FAILED=1
      echo "worker $worker failed; see $RUN_DIR/logs/worker_${worker}.log" | tee -a "$LOG_FILE"
      stop_children
      break
    fi
  done
fi

if (( WORKER_FAILED )); then
  for pid in "${CHILD_PIDS[@]:-}"; do wait "$pid" 2>/dev/null || true; done
fi

if (( ! VALIDATE_ONLY )); then
  if (( WORKER_FAILED )); then kill "$MONITOR_PID" 2>/dev/null || true; fi
  wait "$MONITOR_PID" || WORKER_FAILED=1
fi

if (( WORKER_FAILED )); then
  echo "OA-OPD run failed/interrupted; completed shards remain resumable." | tee -a "$LOG_FILE"
  exit 1
fi
echo "OA-OPD run finished. Results: $RUN_DIR" | tee -a "$LOG_FILE"
trap - INT TERM EXIT
