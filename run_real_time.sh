#!/usr/bin/env bash
set -euo pipefail

# ==== 사용자 설정 영역 ====
PYTHON=python

# 기본 task
TASK="press the blue button"

# HLP 사용 여부 (1 또는 0)
USE_HLP=1

# HLP 모델 경로
HLP_BASE_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
HLP_ADAPTER_PATH="/path/to/your/hlp_adapter"

# LLP 관련 (필요시 config 시스템에 맞춰 수정)
MAX_STEPS=200
USE_DEVICES=1   # 1 -> 실제 로봇/카메라 사용, 0 -> 랜덤 mock
# ============================

# boolean → true/false 문자열로 변환 (parser.wrap 스타일 쓴다면)
if [ "$USE_HLP" -eq 1 ]; then
  USE_HLP_STR="true"
else
  USE_HLP_STR="false"
fi

if [ "$USE_DEVICES" -eq 1 ]; then
  USE_DEVICES_STR="true"
else
  USE_DEVICES_STR="false"
fi

$PYTHON -m evaluate.eval_real_time_main \
  use_hlp="${USE_HLP_STR}" \
  llp.task="${TASK}" \
  llp.use_devices="${USE_DEVICES_STR}" \
  llp.max_steps="${MAX_STEPS}" \
  hlp.base_model_path="${HLP_BASE_MODEL}" \
  hlp.adapter_path="${HLP_ADAPTER_PATH}" \
  hlp.is_qlora=true \
  hlp.device="cuda:0"