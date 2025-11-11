# 🚀 evaluation.py
# 이 스크립트는 훈련된 모델의 성능을 .jsonl 데이터셋과 비교하여 평가합니다.
"""
CUDA_VISIBLE_DEVICES=0 python evaluate/eval_HLP.py \
    --base_model_path /ckpt/Qwen2.5-VL-7B-Instruct \
    --adapter_path /result/ghkim/HLP_qwen_2.5_7b_LoRA_r16_press_the_blue_button_ep60_1109_RAM_test/checkpoint-2000 \
    --dataset_file /data/ghkim/piper_press_the_blue_button_ep60/gpt-5-mini/eval_final/shards/chunk-000.jsonl \
    --output_file /data/ghkim/piper_press_the_blue_button_ep60/eval_qwen_LoRA_RAM_test_2k/shards/chunk_000_evaluation.jsonl \
    --is_qlora True

CUDA_VISIBLE_DEVICES=2 python evaluate/eval_HLP.py \
    --base_model_path /ckpt/Qwen2.5-VL-7B-Instruct \
    --adapter_path /result/ghkim/HLP_qwen_2.5_7b_LoRA_r16_press_the_blue_button_ep60_1109_RAM_test/checkpoint-2000 \
    --dataset_file /data/ghkim/piper_press_the_blue_button_ep60/gpt-5-mini/eval_final/shards/chunk-000.jsonl \
    --output_file /data/ghkim/piper_press_the_blue_button_ep60/eval_qwen_LoRA_RAM_test_2k/shards/chunk_000_evaluation.jsonl \
    --is_qlora False

"""

# 🚀 evaluation.py
# 이 스크립트는 훈련된 모델의 성능을 .jsonl 데이터셋과 비교하여 평가합니다.
import torch
import json
import argparse
import os
from tqdm import tqdm
from PIL import Image
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig # [추가] QLoRA 로딩을 위해
)
from peft import PeftModel


# --- 1. dataset_vlm.py에서 프롬프트 생성기 가져오기 ---
# (훈련 시 사용한 것과 *정확히* 동일한 프롬프트가 필요합니다)
def make_train_prompt(task: str, prev: str, prev_status: str) -> str:
    """
    Generate a concise training prompt for image analysis in robot manipulation.
    (train_vlm.py가 훈련 시 사용했던 바로 그 함수)
    """
    return (
        "You are an image-analysis expert for robot manipulation.\n"
        "INPUT_IMAGES: [SIDE]=global scene view; [WRIST]=close-up wrist camera.\n"
        f"TASK: {task}\n"
        f"PREV_DESC: {prev}\n"
        f"PREV_STATUS: {prev_status}\n"
        "Describe what is visibly happening now (desc_1) and the visible evidence for completion (desc_2).\n"
        "Then decide the status: DONE / NOT_DONE / UNCERTAIN.\n"
        "Output JSON: {\"desc_1\":\"...\",\"desc_2\":\"...\",\"status\":\"...\"}"
    )


def run_evaluation(
    base_model_path: str,
    adapter_path: str,
    dataset_file: str,
    output_file: str,
    load_in_4bit: bool = False, # QLoRA 훈련 시 True
    device: str = "cuda"
):
    """
    .jsonl 파일을 읽어 훈련된 모델의 추론을 수행하고, 정답과 함께 저장합니다.
    """

    # QLoRA로 훈련했다면, 베이스 모델도 4비트로 로드해야 병합할 수 있습니다.
    bnb_config = None
    if load_in_4bit:
        print("Loading base model in 4-bit (for QLoRA merge)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    print(f"Loading base model from: {base_model_path}")
    # (1) 베이스 모델 로드
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        quantization_config=bnb_config,  # 4비트 로드
        device_map=device,
        torch_dtype="auto",
        attn_implementation="eager",
    )

    # (2) 프로세서는 베이스 모델 경로에서 로드
    processor = AutoProcessor.from_pretrained(base_model_path)
    tokenizer = processor.tokenizer

    print(f"Loading adapter from: {adapter_path}")
    # (3) PEFT 모델로 어댑터를 베이스 모델 위에 "덮어씌움"
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        ignore_mismatched_sizes=True
    )

    # (4) ★★★ 요청하신 "임시 병합" (메모리상에서 병합 후 PEFT 래퍼 제거) ★★★
    print("Merging adapter into base model (in memory)...")
    model = model.merge_and_unload()

    model.eval()  # 평가 모드

    results = []

    print(f"Loading dataset from: {dataset_file}")
    with open(dataset_file, 'r', encoding='utf-8') as f_in:
        # .jsonl의 모든 라인을 리스트로 읽어옴
        lines = f_in.readlines()

    print(f"Running inference on {len(lines)} samples...")
    # tqdm으로 진행률 표시
    for line in tqdm(lines):
        if not line.strip():
            continue

        try:
            data = json.loads(line)

            # --- 3. .jsonl에서 입력값 및 정답 추출 ---
            # 입력값
            side_img_path = data['images']['side']
            wrist_img_path = data['images']['wrist']
            task = data['task']
            prev_desc = data['prev_desc']
            prev_status = data['prev_status']

            # 정답 (비교용)
            ground_truth_output = data['api_output']

            # 이미지 로드
            images_list = [
                Image.open(side_img_path).convert('RGB'),
                Image.open(wrist_img_path).convert('RGB')
            ]

            # --- 4. 훈련과 동일한 프롬프트 생성 (핵심) ---
            user_prompt_text = make_train_prompt(task, prev_desc, prev_status)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},  # side
                        {"type": "image"},  # wrist
                        {"type": "text", "text": user_prompt_text}
                    ]
                }
            ]

            # 템플릿 적용 (evaluate_qwen.py 방식)
            prompt_string = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # 4-B. [수정] processor는 변환된 *문자열*과 이미지를 받아 텐서로 만듭니다.
            inputs = processor(
                # [수정] 딕셔너리(messages)가 아닌 문자열(prompt_string)을 리스트에 담아 전달
                text=[prompt_string],
                images=images_list,
                padding=True,
                return_tensors="pt"
                # add_generation_prompt는 이미 apply_chat_template에서 처리됨
            ).to(device)

            # --- 5. 모델 추론 (model.generate) ---
            with torch.no_grad():  # 추론 시에는 그래디언트 계산 불필요
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,  # JSON 출력이므로 넉넉하게
                    do_sample=False  # 평가 시에는 항상 False
                )

            # 입력 토큰을 제외한 순수 출력 토큰만 분리
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            # 토큰을 텍스트로 디코딩
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            # --- 6. 결과 저장 (이미지 경로/메타 포함) ---
            uid = data.get("uid", "unknown")
            task = data.get("task", "")
            chunk_id = data.get("chunk_id", "")
            episode_id = data.get("episode_id", "")
            timestamp_ms = data.get("timestamp_ms", None)

            results.append({
                "uid": uid,
                "task": task,
                "chunk_id": chunk_id,
                "episode_id": episode_id,
                "timestamp_ms": timestamp_ms,
                "images": {
                    "side": side_img_path,
                    "wrist": wrist_img_path
                },
                "model_output_raw": output_text,
                "gt_output": ground_truth_output,
                "prompt": user_prompt_text  # 디버깅용 (원치 않으면 제거해도 됨)
            })

        except Exception as e:
            print(f"Error processing line (uid: {data.get('uid', 'N/A')}): {e}")
            # 실패 케이스도 동일한 스키마로 기록 (가능한 한 메타/이미지 포함)
            try:
                uid = data.get("uid", "N/A")
                task = data.get("task", "")
                chunk_id = data.get("chunk_id", "")
                episode_id = data.get("episode_id", "")
                timestamp_ms = data.get("timestamp_ms", None)
                side_img_path = data.get("images", {}).get("side", None)
                wrist_img_path = data.get("images", {}).get("wrist", None)
                gt_out = data.get("api_output", {})
            except Exception:
                uid = "N/A"
                task = chunk_id = episode_id = ""
                timestamp_ms = None
                side_img_path = wrist_img_path = None
                gt_out = {}

            results.append({
                "uid": uid,
                "task": task,
                "chunk_id": chunk_id,
                "episode_id": episode_id,
                "timestamp_ms": timestamp_ms,
                "images": {
                    "side": side_img_path,
                    "wrist": wrist_img_path
                },
                "model_output_raw": f"ERROR: {str(e)}",
                "gt_output": gt_out
            })

    # --- 7. 최종 결과를 별도 jsonl 파일로 저장 ---
    print(f"Saving {len(results)} results to {output_file}...")

    output_dir = os.path.dirname(output_file)
    # 디렉토리가 존재하지 않으면 생성합니다.
    # (output_dir가 비어있지 않은 경우에만, 즉 상대 경로가 아닌 경우)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for res in results:
            f_out.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned Qwen-VL model by merging adapters in memory")

    # [수정됨] 2개의 경로를 받도록 변경
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="Path to the ORIGINAL base model directory (e.g., /ckpt/Qwen2.5-VL-7B-Instruct)")
    parser.add_argument("--adapter_path", type=str, required=True,
                        help="Path to the trained LoRA/QLoRA adapter directory (e.g., ./results/.../final-adapter)")

    parser.add_argument("--dataset_file", type=str, required=True,
                        help="Path to the .jsonl dataset file to evaluate")
    parser.add_argument("--output_file", type=str, default="./evaluation_results.jsonl",
                        help="Path to save the evaluation results")

    # [추가] 훈련 방식에 따라 설정
    parser.add_argument("--is_qlora", type=bool, default=False,
                        help="Set this if you trained with *standard* LoRA (not QLoRA)")

    args = parser.parse_args()

    run_evaluation(
        args.base_model_path,
        args.adapter_path,
        args.dataset_file,
        args.output_file,
        load_in_4bit=args.is_qlora  # 플래그의 반대
    )