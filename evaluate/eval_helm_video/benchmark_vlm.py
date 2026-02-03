import argparse
import os
import json
import time
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from peft import PeftModel

"""
CUDA_VISIBLE_DEVICES=4 python evaluate/eval_helm_video/benchmark_vlm.py \
  --model_path /ckpt/Qwen2.5-VL-7B-Instruct \
  --adapter /backups/ghkim/HLP_HeLM_video/HeLM_video_task_10_ddp_0127_re/checkpoint-5000 \
  --jsonl_path /data/ghkim/helm_data/helm_video_task_inter/merged/all_train.jsonl \
  --max_pixels 310000
"""

# ==========================================
# 1. 성능 측정용 클래스 (Context Manager)
# ==========================================
class PerformanceMonitor:
    def __init__(self):
        self.latency = 0
        self.max_mem_gb = 0

    def __enter__(self):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        self.end_time = time.time()
        self.latency = self.end_time - self.start_time
        self.max_mem_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)


# ==========================================
# 2. JSONL 데이터 로더 (커스텀 필요 시 수정)
# ==========================================
def load_and_group_data(jsonl_path, max_samples_per_group=10):
    grouped_data = defaultdict(list)
    print(f"📂 Loading data from {jsonl_path}...")

    with open(jsonl_path, 'r') as f:
        for line in f:
            item = json.loads(line)

            # 1. 이미지 개수 추출
            image_paths = item.get("images", [])
            num_images = len(image_paths)

            # 2. [수정] 태스크 유형 판별 (제공해주신 'mode' 키 사용)
            # JSONL 샘플에 "mode": "DETECT" 또는 "mode": "UPDATE"가 명시되어 있습니다.
            task_type = item.get("mode", "UNKNOWN").upper()

            # 만약 mode 키가 없는 예외 상황을 대비한 백업 로직
            if task_type == "UNKNOWN":
                task_type = "DETECT" if num_images == 1 else "UPDATE"

            # 3. 그룹별 수집
            group_key = (task_type, num_images)
            if len(grouped_data[group_key]) < max_samples_per_group:
                grouped_data[group_key].append({
                    "images": image_paths,
                    "text": item.get("user_prompt", ""),
                    "task_type": task_type
                })
    return grouped_data


# ==========================================
# 3. 메인 실행 함수
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--jsonl_path", type=str, required=True, help="Path to validation jsonl")
    parser.add_argument("--min_pixels", type=int, default=50176)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--max_pixels", type=int, default=310000)
    parser.add_argument("--samples_per_grCUDA_VISIBLE_DEVICES=4 python evaluate/eval_helm_v4/benchmark_helm_v4.py \
  --model_name_or_path /ckpt/Qwen2.5-VL-7B-Instruct \
  --ckpt_adapter /backups/ghkim/HeLM_v4/HLP_HeLM_v4_qwen_7b_all_extended_0122/checkpoint-3500 \
  --eval_jsonl /data/ghkim/helm_data/helm_video_task_10/merged/all_val.jsonl \
  --device cuda \
  --max_samples 50 \
  --warmup 10 \
  --max_new_tokens_detect 32 \
  --max_new_tokens_update 128oup", type=int, default=10,
                        help="Number of samples to measure per image count")
    args = parser.parse_args()

    # 1. 모델 & 프로세서 로드
    print(f"🚀 Loading model from {args.model_path}...")
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels
    )

    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    #     args.model_path,
    #     device_map="auto",
    #     torch_dtype=torch.bfloat16,
    #     attn_implementation="flash_attention_2"  # Flash Attention 2 필수
    # )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    if args.adapter:
        print(f"Loading LoRA adapter from {args.adapter}")
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    # 2. 데이터 로드
    grouped_data = load_and_group_data(args.jsonl_path, 10)

    # 3. 측정 시작
    print("\n⚡ Starting Performance Measurement...")
    final_results = []

    # (태스크, 이미지 개수) 순으로 정렬하여 실행
    sorted_keys = sorted(grouped_data.keys(), key=lambda x: (x[0], x[1]))

    for task_type, num_images in sorted_keys:
        samples = grouped_data[(task_type, num_images)]
        mode_label = f"{task_type} ({num_images} img)"

        print(f"\n[{mode_label}] Measuring {len(samples)} samples...")
        latencies = []
        vrams = []
        gen_token_counts = []

        for sample in tqdm(samples):
            with PerformanceMonitor() as pm:
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": img_path} for img_path in sample["images"]] +
                                   [{"type": "text", "text": sample["text"]}]
                    }
                ]

                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True,
                                   return_tensors="pt").to("cuda")


                with torch.no_grad():
                    # 실제 상황 반영을 위해 max_new_tokens는 넉넉히 주되,
                    # 실제 생성된 토큰 수도 함께 기록합니다.
                    gen_ids = model.generate(**inputs, max_new_tokens=128, do_sample=False)

                    in_len = inputs["input_ids"].shape[1]
                    gen_tokens = gen_ids[0, in_len:]
                    gen_token_counts.append(len(gen_tokens))

            latencies.append(pm.latency)
            vrams.append(pm.max_mem_gb)

        avg_lat = np.mean(latencies)
        avg_vram = np.mean(vrams)
        avg_gen_tokens = np.mean(gen_token_counts)

        final_results.append({
            "task": task_type,
            "images": num_images,
            "latency": avg_lat,
            "vram": avg_vram,
            "gen_tokens": avg_gen_tokens
        })

    # 5. 최종 리포트 출력 (분리된 테이블)
    print("\n" + "=" * 65)
    print(f"{'Task':<8} | {'Images':<6} | {'Latency (s)':<12} | {'VRAM (GB)':<10} | {'Gen Tokens':<10}")
    print("-" * 65)
    for res in final_results:
        print(
            f"{res['task']:<8} | {res['images']:<6} | {res['latency']:<12.4f} | {res['vram']:<10.2f} | {res['gen_tokens']:<10.1f}")
    print("=" * 65)


if __name__ == "__main__":
    main()