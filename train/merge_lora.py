from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor

# 1. 원본 베이스 모델을 로드합니다 (훈련 시 사용한 설정과 동일하게)
base_model = AutoModelForVision2Seq.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", # 예시 경로
    device_map="auto",
    trust_remote_code=True,
    # ... (필요시 4-bit/8-bit 로드 옵션) ...
)

# 2. 저장된 어댑터 경로로 PeftModel을 로드합니다
# (예: "output_dir/checkpoint-2000")
peft_model = PeftModel.from_pretrained(base_model, "/path/to/your/checkpoint-2000")

# 3. 병합 후 메모리에서 어댑터 레이어 제거
print("Merging adapter...")
merged_model = peft_model.merge_and_unload()
print("Merge complete.")

# 4. 병합된 풀 모델(full model)을 저장합니다.
# (이 모델은 LoRA가 없는 일반 모델이며 용량이 큽니다)
merged_model_path = "/path/to/your/merged_model_directory"
merged_model.save_pretrained(merged_model_path)

# 5. (중요) Processor도 함께 저장해야 나중에 불러올 수 있습니다.
processor = AutoProcessor.from_pretrained("/path/to/your/checkpoint-2000", trust_remote_code=True)
processor.save_pretrained(merged_model_path)