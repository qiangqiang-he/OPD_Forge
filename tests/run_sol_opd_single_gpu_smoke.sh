#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${RLVR_PYTHON_BIN:-$(command -v python)}
model_root=${RLVR_MODEL_ROOT:-$project_root/models}
student_model=${SOL_OPD_STUDENT_MODEL:-$model_root/Qwen3-1.7B}
formal_teacher=${SOL_OPD_TEACHER_MODEL:-$model_root/Qwen3-4B-Thinking-2507}
driver=$project_root/tests/sol_opd_single_gpu_smoke.py
mode=${1:-long1024}

if [[ -d "$formal_teacher" ]]; then
  teacher_model=$formal_teacher
  proxy_flag=()
else
  teacher_model=$model_root/Qwen3-4B
  proxy_flag=(--proxy-used)
  echo "Exact Qwen3-4B-Thinking-2507 is unavailable; using Qwen3-4B as the Teacher runtime proxy."
fi

for required_path in "$student_model" "$teacher_model"; do
  if [[ ! -d "$required_path" ]]; then
    echo "Required local model directory does not exist: $required_path" >&2
    exit 2
  fi
done

artifact_dir=$(mktemp -d "$project_root/tests/.sol_opd_single_gpu.XXXXXX")
cleanup() {
  local resolved_artifact_dir
  resolved_artifact_dir=$(realpath -m "$artifact_dir")
  case "$resolved_artifact_dir" in
    "$project_root/tests/"*) rm -rf -- "$resolved_artifact_dir" ;;
    *) echo "Refusing to remove unexpected artifact directory: $resolved_artifact_dir" >&2 ;;
  esac
}
trap cleanup EXIT

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$project_root/verl:$project_root${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

"$python_bin" -m pytest \
  "$project_root/tests/test_sol_opd.py" \
  "$project_root/tests/test_algorithm_independence.py" \
  "$project_root/tests/test_teacher_response_alignment.py" \
  -q

if [[ "${SOL_OPD_RUN_DATASET_AUDIT:-1}" == "1" ]]; then
  "$python_bin" "$driver" dataset-audit \
    --model "$teacher_model" \
    --data "$project_root/data/DAPO-17k-English-Qwen3-4B-Instruct-2507-Correct.json"
fi

run_case() {
  local label=$1
  local response_tokens=$2
  local selected_tokens=$3

  local student_artifact=$artifact_dir/${label}_student.pt
  local teacher_artifact=$artifact_dir/${label}_teacher.pt

  # Separate Python processes guarantee complete Student/Teacher GPU release.
  "$python_bin" "$driver" student-rollout \
    --model "$student_model" \
    --output "$student_artifact" \
    --response-tokens "$response_tokens"

  "$python_bin" "$driver" teacher-dual \
    --model "$teacher_model" \
    --student "$student_artifact" \
    --output "$teacher_artifact" \
    "${proxy_flag[@]}"

  "$python_bin" "$driver" student-backward \
    --model "$student_model" \
    --student "$student_artifact" \
    --teacher "$teacher_artifact" \
    --selected-tokens "$selected_tokens"

  CUDA_VISIBLE_DEVICES="" "$python_bin" "$driver" joint \
    --student "$student_artifact" \
    --teacher "$teacher_artifact"
}

case "$mode" in
  short)
    run_case short "${SOL_OPD_SHORT_RESPONSE_TOKENS:-64}" \
      "${SOL_OPD_SHORT_BACKWARD_TOKENS:-16}"
    ;;
  long1024)
    run_case long1024 "${SOL_OPD_LONG_RESPONSE_TOKENS:-1024}" \
      "${SOL_OPD_LONG_BACKWARD_TOKENS:-16}"
    ;;
  both)
    run_case short "${SOL_OPD_SHORT_RESPONSE_TOKENS:-64}" \
      "${SOL_OPD_SHORT_BACKWARD_TOKENS:-16}"
    run_case long1024 "${SOL_OPD_LONG_RESPONSE_TOKENS:-1024}" \
      "${SOL_OPD_LONG_BACKWARD_TOKENS:-16}"
    ;;
  *)
    echo "Usage: bash tests/run_sol_opd_single_gpu_smoke.sh [short|long1024|both]" >&2
    exit 2
    ;;
esac

echo "Sol-OPD single-GPU $mode smoke passed."
