#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
model_root=${RLVR_MODEL_ROOT:-$project_root/models}
student_model=${EXOPD_STUDENT_MODEL:-$model_root/Qwen3-1.7B}
teacher_model=${EXOPD_TEACHER_MODEL:-$model_root/Qwen3-4B-Thinking-2507}
driver=$project_root/tests/exopd_lora_split_smoke.py
total_length=${EXOPD_TOTAL_LENGTH:-1024}
response_length=${EXOPD_RESPONSE_LENGTH:-64}

if [[ -n "${RLVR_PYTHON_BIN:-}" ]]; then
  python_bin=$RLVR_PYTHON_BIN
elif [[ -x /home/qqh/miniconda3/envs/verl/bin/python ]]; then
  python_bin=/home/qqh/miniconda3/envs/verl/bin/python
else
  python_bin=$(command -v python)
fi

for required_path in "$student_model" "$teacher_model"; do
  if [[ ! -d "$required_path" ]]; then
    echo "Required local model directory does not exist: $required_path" >&2
    exit 2
  fi
done

# Artifacts remain test-only and are removed on exit.  Resolve the directory
# before deletion so an unexpected path can never become a recursive target.
artifact_dir=$(mktemp -d "$project_root/tests/.exopd_lora_split.XXXXXX")
cleanup() {
  local resolved_artifact_dir
  resolved_artifact_dir=$(realpath -m -- "$artifact_dir")
  case "$resolved_artifact_dir" in
    "$project_root/tests/"*) rm -rf -- "$resolved_artifact_dir" ;;
    *) echo "Refusing to remove unexpected artifact directory: $resolved_artifact_dir" >&2 ;;
  esac
}
trap cleanup EXIT

student_fixture=$artifact_dir/student.pt
teacher_fixture=$artifact_dir/teacher.pt

# Pin one GPU even when this helper is accidentally run on a multi-GPU host.
export CUDA_VISIBLE_DEVICES=${EXOPD_CUDA_DEVICE:-0}
export PYTHONPATH="$project_root/verl:$project_root${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

"$python_bin" "$driver" student \
  --model "$student_model" \
  --output "$student_fixture" \
  --total-length "$total_length" \
  --response-length "$response_length"

# This is a new Python process: the Student and its CUDA allocator no longer
# exist when the Teacher checkpoint is loaded.
"$python_bin" "$driver" teacher \
  --model "$teacher_model" \
  --student-fixture "$student_fixture" \
  --output "$teacher_fixture"

# Hide CUDA to make the final Teacher/Student integration check explicitly CPU-only.
CUDA_VISIBLE_DEVICES="" "$python_bin" "$driver" joint \
  --student-fixture "$student_fixture" \
  --teacher-fixture "$teacher_fixture"

echo "LoRA ExOPD split single-GPU smoke passed (total_length=$total_length)."
