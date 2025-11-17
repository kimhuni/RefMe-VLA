#!/usr/bin/env bash
set -euo pipefail

TASK="press the blue button"
HLP_BASE_MODEL="/home/minji/Desktop/data/Qwen2.5-VL-7B-Instruct"
HLP_ADAPTER_PATH="/home/minji/Desktop/data/finetuned_model/ghkim/HLP_qwen_2.5_7b_QLoRA_r16_press_the_blue_button_ep60_1114_final/checkpoint-2000"
LLP_BASE_MODEL="/home/minji/Desktop/data/finetuned_model/ghkim/pi0_press_the_blue_button_ep60/030000/pretrained_model"
LLP_DATASET="/home/minji/Desktop/data/data_config/ep60_press_the_blue_button"

# LLP 관련 (필요시 config 시스템에 맞춰 수정)
MAX_STEPS=200

# ============================
python -m evaluate.eval_real_time_main \
  --use_hlp=false \
  --task="${TASK}" \
  --max_steps="${MAX_STEPS}" \
  --llp_model_path="${LLP_BASE_MODEL}" \
  --dataset_repo_id="${LLP_DATASET}" \
  --dataset_root="${LLP_DATASET}" \
  --use_devices=true \
  --hlp_model_path="${HLP_BASE_MODEL}" \
  --hlp_adapter_path="${HLP_ADAPTER_PATH}" \
  --hlp_use_qlora=true