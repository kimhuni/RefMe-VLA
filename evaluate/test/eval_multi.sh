#!/usr/bin/env bash
set -euo pipefail

# ===== User-configurable inputs =====
VIDEO="/data/piper_press/videos/chunk-000/observation.images.table/episode_000000.mp4"
TASK="press the blue button"
OUTPUT_ROOT="/result/VLM_test"
#MAX_NEW_TOKENS=80
#WARMUP=2

# Select GPU externally or override here
: "${CUDA_VISIBLE_DEVICES:=1}"

# Models to run sequentially (must match your Python script's --model choices)
MODELS=(
  "qwen"
  "intern"
  "llava"
  "minicpm"
)

# Map each model key to its local model directory (edit paths as needed)
function model_dir_of() {
  case "$1" in
    qwen)     echo "/ckpt/Qwen2.5-VL-7B-Instruct" ;;
    intern)   echo "/ckpt/InternVL3-8B-hf" ;;
    llava)    echo "/ckpt/llava-onevision-7b-hf" ;;
    minicpm)  echo "/ckpt/MiniCPM-V-4_5" ;;
    *)        echo ""; return 1 ;;
  esac
}

# ===== Run all models sequentially on the same video =====
for MODEL in "${MODELS[@]}"; do
  MODEL_DIR="$(model_dir_of "$MODEL" || true)"
  if [[ -z "${MODEL_DIR}" ]]; then
    echo "[ERROR] Unknown model key: ${MODEL}"
    exit 1
  fi
  if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "[WARN] Skipping ${MODEL}: model dir not found at ${MODEL_DIR}"
    continue
  fi

  echo "=== Running ${MODEL} ==="
  echo "    VIDEO       : ${VIDEO}"
  echo "    MODEL_DIR   : ${MODEL_DIR}"
  echo "    CUDA DEVICE : ${CUDA_VISIBLE_DEVICES}"
  echo

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  python evaluate_real_time.py \
    --video_path "${VIDEO}" \
    --model "${MODEL}" \
    --task "${TASK}" \
#    --max_new_tokens "${MAX_NEW_TOKENS}" \
#    --warmup "${WARMUP}"

  echo
done

echo "All runs finished."