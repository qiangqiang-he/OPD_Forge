#!/usr/bin/env bash
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${RLVR_PYTHON_BIN:-$(command -v python)}
model_root=${RLVR_MODEL_ROOT:-$project_root/models}
pg_driver=$project_root/tests/test_pub_pg_opd_thinking_configs.py
cal_driver=$project_root/tests/test_pub_cal_opd_thinking_configs.py
artifact_dir=$(mktemp -d "$project_root/tests/.cal_opd_interventions_4096.XXXXXX")

cleanup() {
  local resolved_artifact_dir
  resolved_artifact_dir=$(realpath -m "$artifact_dir")
  case "$resolved_artifact_dir" in
    "$project_root/tests/"*) rm -rf -- "$resolved_artifact_dir" ;;
    *) echo "Refusing to remove unexpected artifact directory: $resolved_artifact_dir" >&2 ;;
  esac
}
trap cleanup EXIT

assert_gpu_available() {
  local free_mib
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  if (( free_mib < 7168 )); then
    echo "At least 7 GiB must be free before a smoke phase; found ${free_mib} MiB." >&2
    exit 1
  fi
}

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$project_root/verl:$project_root/tests:$project_root${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=offline

"$python_bin" -m pytest \
  "$project_root/tests/test_cal_opd.py" \
  "$project_root/tests/test_cal_opd_prompts.py" \
  "$project_root/tests/test_cal_opd_interventions.py" \
  "$project_root/tests/test_cal_opd_ablation_configs.py" \
  -q

assert_gpu_available
"$python_bin" "$pg_driver" vllm-student \
  --model "$model_root/Qwen3-1.7B" \
  --response-tokens 4096 \
  --full-context \
  --output "$artifact_dir/student_1p7b.pt"

for intervention in inst eval answer solution; do
  assert_gpu_available
  "$python_bin" "$cal_driver" teacher \
    --model "$model_root/Qwen3-4B-Thinking-2507" \
    --intervention "$intervention" \
    --pair \
      "$artifact_dir/student_1p7b.pt" \
      "$artifact_dir/teacher_${intervention}.pt"
done

"$python_bin" "$cal_driver" joint \
  --pair "$artifact_dir/student_1p7b.pt" "$artifact_dir/teacher_inst.pt" \
  --pair "$artifact_dir/student_1p7b.pt" "$artifact_dir/teacher_eval.pt" \
  --pair "$artifact_dir/student_1p7b.pt" "$artifact_dir/teacher_answer.pt" \
  --pair "$artifact_dir/student_1p7b.pt" "$artifact_dir/teacher_solution.pt"

assert_gpu_available
"$python_bin" "$cal_driver" full-backward \
  --model "$model_root/Qwen3-1.7B" \
  --student "$artifact_dir/student_1p7b.pt" \
  --teacher "$artifact_dir/teacher_solution.pt" \
  --selected-tokens 128

echo "All Cal-OPD intervention 4096-token smoke phases passed."
