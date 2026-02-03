from transformers import AutoProcessor
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt

# 1. 문제의 설정 그대로 로드 (훈련 때 쓴 설정)
model_path = "/ckpt/Qwen2.5-VL-7B-Instruct"
min_pixels = 256 * 28 * 28  # 200,704
max_pixels = 100352         # 100,352 (훈련 시 실수한 값)

print(f"DEBUG: Loading processor with min={min_pixels}, max={max_pixels}")

try:
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        min_pixels=min_pixels,
        max_pixels=max_pixels
    )
except Exception as e:
    print(f"Error loading processor: {e}")
    exit()

# 2. 테스트 이미지 생성 (랜덤 노이즈 대신 실제 이미지 권장)
# 로컬에 있는 아무 이미지나 경로를 넣어주세요
image_path = "/data/ghkim/helm_data/press_button_in_order/frames_5hz/chunk-000/episode_000000/table/frame_000039.jpg" # press blue
# 이미지가 없다면 더미 생성
if "..." in image_path:
    img = Image.new('RGB', (640, 480), color='red')
else:
    img = Image.open(image_path)

# 3. 프로세싱 진행
text = "Describe this image."
messages = [
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}
]
prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = processor(
    text=[prompt],
    images=[img],
    return_tensors="pt"
)

# 4. 결과 분석
pixel_values = inputs["pixel_values"]
image_grid = inputs["image_grid_thw"]

print("="*30)
print(f"Image Grid Shape (T, H, W): {image_grid}")
print(f"Pixel Values Shape: {pixel_values.shape}")
print(f"Pixel Values Range: Min={pixel_values.min().item():.3f}, Max={pixel_values.max().item():.3f}")
print("="*30)

# 5. 진단
# (1) Grid가 0이거나 매우 이상한 값이면 -> 리사이징 실패
# (2) Pixel Value가 전부 0이거나 NaN이면 -> 이미지 정보 소실
if pixel_values.std().item() < 0.01:
    print("🚨 CRITICAL: Pixel values are flat (empty). Model is BLIND.")
else:
    print("✅ Pixel values contain variance. Proceed to verify visualization.")