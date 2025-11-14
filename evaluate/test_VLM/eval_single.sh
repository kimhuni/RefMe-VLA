#!/usr/bin/env bash
set -euo pipefail

# ===== User-configurable inputs =====
MODEL="intern"
VIDEO="/data/piper_press/videos/chunk-000/observation.images.table/episode_000000.mp4"
TASK="press the blue button"
OUTPUT_ROOT="/result/VLM_test_short"
MODEL_DIR="/ckpt/InternVL3-1B"
#MAX_NEW_TOKENS=80
#WARMUP=2

# Select GPU externally or override here
: "${CUDA_VISIBLE_DEVICES:=1}"

echo "=== Running ${MODEL} ==="
echo "    VIDEO       : ${VIDEO}"
echo "    MODEL_DIR   : ${MODEL_DIR}"
echo "    CUDA DEVICE : ${CUDA_VISIBLE_DEVICES}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
python evaluate_real_time.py \
  --video_path "${VIDEO}" \
  --model "${MODEL}" \
  --task "${TASK}" \
  --model_dir "${MODEL_DIR}" \
#    --max_new_tokens "${MAX_NEW_TOKENS}" \
#    --warmup "${WARMUP}"
done

echo "All runs finished."