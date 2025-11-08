# 📂 inspect_model.py
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# ⚠️ 모델 경로를 실제 경로로 수정하세요
MODEL_PATH = "/ckpt/Qwen2.5-VL-7B-Instruct"

print(f"Loading model from: {MODEL_PATH}")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    trust_remote_code=True,
)

print("\n--- Model Architecture (All Modules) ---")
# 모든 모듈의 이름을 출력합니다.
all_modules = {name for name, mod in model.named_modules()}

# LoRA의 주 대상이 되는 Linear 레이어 이름만 필터링해서 봅니다.
print("\n--- Candidate Linear Layers for LoRA ---")
count = 0
for name in all_modules:
    # 'q_proj', 'k_proj', 'v_proj', 'o_proj' 또는
    # 'gate_proj', 'up_proj', 'down_proj' 같은
    # 일반적인 LoRA 타겟 레이어 이름을 포함하는지 확인합니다.
    if any(target in name for target in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']):
        print(name)
        count += 1

print(f"\nFound {count} candidate layers.")
print("\n---")
print("이제 이 리스트에서 반복되는 핵심 이름(예: 'q_proj', 'k_proj', 'v_proj', 'o_proj')을 찾으세요.")
print("그 이름들을 --target_modules 인자로 사용해야 합니다.")