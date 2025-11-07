#!/usr/bin/env python3
# run_vlm.py
import argparse
import os
from textwrap import wrap

import torch
from PIL import Image, ImageDraw, ImageFont

# ========== 공통 프롬프트 ==========
# SYSTEM_PROMPT = (
#     "You are an expert in robotic manipulation image analysis. "
#     "Use only the current image and provided text. Respond in exactly two sentences."
# )

SYSTEM_PROMPT = (
    "You are an image analysis expert specialized in robotic manipulation. "
    "You will be given an image showing a robot arm and a text input which consists of robot task and description you generated previously."
    "Describe visible robot actions and task completion strictly based on the image and the input text"
    "Describe in two sentences what the robot is doing and "
    "whether the task is done or not."
)

DEFAULT_TASK = "Press the blue button on the table."
DEFAULT_PREV = "The robot's gripper is positioned above the blue button, ready to press it. The task is not done."

# ========== 공통 유틸: 주석 이미지 저장 ==========
def annotate_and_save(image_path: str, text: str, output_dir: str, suffix: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    extra_space = max(100, int(h * 0.15))
    canvas = Image.new("RGB", (w, h + extra_space), (255, 255, 255))
    canvas.paste(img, (0, 0))

    # 폰트
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    except Exception:
        font = ImageFont.load_default()

    # 줄바꿈
    draw = ImageDraw.Draw(canvas)
    max_width_px = w - 20
    # 글자폭 대략 추정: 폰트 크기/2.2 근사
    char_px = max(7, font.size / 2.2)
    lines = []
    for line in str(text).split("\n"):
        lines.extend(wrap(line, width=max(8, int(max_width_px / char_px))))

    y = h + 10
    for line in lines:
        draw.text((10, y), line, fill=(0, 0, 0), font=font)
        y += font.size + 5

    base = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(output_dir, f"{base}_{suffix}.jpg")
    canvas.save(save_path)
    return save_path


# ========== Qwen2.5-VL-7B ==========
def run_qwen(model_dir: str, image_path: str, task: str, prev_desc: str) -> str:
    """
    Qwen2.5-VL-7B-Instruct 분기.
    요구: qwen_vl_utils, Qwen2_5_VLForConditionalGeneration
    """
    try:
        from transformers import AutoProcessor
        from transformers import Qwen2_5_VLForConditionalGeneration
        try:
            # qwen_vl_utils가 설치되어 있어야 이미지/비디오 포맷 처리 편함
            from qwen_vl_utils import process_vision_info
        except Exception as e:
            raise RuntimeError(
                "qwen_vl_utils 가 필요합니다. `pip install qwen-vl-utils` 혹은 배포된 패키지를 설치하세요."
            ) from e
    except Exception as e:
        raise RuntimeError("Qwen 패키지(Transformers + Qwen 전용 클래스) 혹은 의존성이 없습니다.") from e

    device_map = "auto"
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir,
        device_map=device_map,
        dtype=torch.float16,  # 24GB면 fp16/4bit 선택
        local_files_only=True,
    )

    user_prompt = (
        f"Task: {task}\n"
        f"Previous description: {prev_desc}\n"
        "Describe what the robot is doing in this image and whether the task is done."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=80, do_sample=False, temperature=0.0)

    # prompt 부분 제거 후 디코딩
    prompt_len = inputs["input_ids"].shape[1]
    output = processor.batch_decode(out_ids[:, prompt_len:], skip_special_tokens=True)[0].strip()
    return output


# ========== InternVL3-8B-hf ==========
def run_internvl(model_dir: str, image_path: str, task: str, prev_desc: str) -> str:
    """
    InternVL3-8B-hf 분기 (Transformers 네이티브).
    """
    from transformers import AutoProcessor, AutoModelForCausalLM

    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map="auto",
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
    )

    user_prompt = (
        f"Task: {task}\n"
        f"Previous description: {prev_desc}\n"
        "Describe what the robot is doing in this image and whether the task is done."
    )

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": image_path},
                                     {"type": "text", "text": user_prompt}]},
    ]

    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], images=[Image.open(image_path).convert("RGB")],
                       return_tensors="pt", padding=True)
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=80, do_sample=False, temperature=0.0)

    # prompt 부분 제거
    tok = getattr(processor, "tokenizer", processor)
    prompt_len = inputs["input_ids"].shape[1]
    output = tok.batch_decode(out_ids[:, prompt_len:], skip_special_tokens=True)[0].strip()
    return output


# ========== LLaVA-OneVision-7B-hf ==========
def run_llava(model_dir: str, image_path: str, task: str, prev_desc: str) -> str:
    """
    LLaVA-OneVision-7B-hf 분기 (Transformers 네이티브 ImageTextToText).
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText

    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir,
        device_map="auto",
        dtype=torch.bfloat16,
        local_files_only=True,
        trust_remote_code=True,
    )

    user_prompt = (
        f"Task: {task}\n"
        f"Previous description: {prev_desc}\n"
        "Describe what the robot is doing in this image and whether the task is done."
    )
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type": "image", "image": image_path},
                                     {"type": "text", "text": user_prompt}]},
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=80, do_sample=False, temperature=0.0)

    output = processor.decode(out_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return output


# ========== MiniCPM-V (2.6/4.5 공용) ==========
def run_minicpm(model_dir: str, image_path: str, task: str, prev_desc: str, stream: bool = True) -> str:
    """
    MiniCPM-V 계열: model.chat(stream=True) → generator 반환.
    system role 미지원 → system 내용을 user로.
    """
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained(
        model_dir,
        trust_remote_code=True,
        attn_implementation="sdpa",
        dtype=torch.bfloat16
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    # MiniCPM은 system 역할 비권장 → system 내용을 user 텍스트에 합치기
    user_prompt_full = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Task: {task}\n"
        f"Previous description: {prev_desc}\n"
        "Describe what the robot is doing in this image and whether the task is done."
    )

    image = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "user", "content": [image, user_prompt_full]},
    ]

    answer = model.chat(
        msgs=messages,
        tokenizer=tokenizer,
        enable_thinking=False,
        stream=stream
    )

    if stream:
        generated_text = ""
        for new_text in answer:
            generated_text += new_text
            print(new_text, flush=True, end="")
        print()
    else:
        generated_text = answer

    return generated_text.strip()


# ========== 메인 ==========
def main():
    parser = argparse.ArgumentParser(description="Unified VLM evaluator with argparse routing.")
    parser.add_argument("--model", required=True, choices=["qwen", "internvl", "llava", "minicpm"],
                        help="Model selector key.")
    parser.add_argument(
        "--model_dir",
        required=False,
        default="",
        help="Local model directory. If omitted, a default path will be used per model key."
    )
    parser.add_argument("--image_path", required=True, help="Path to input image.")
    parser.add_argument("--task", default=DEFAULT_TASK, help="Task description text.")
    parser.add_argument("--prev_desc", default=DEFAULT_PREV, help="Previous step description.")
    parser.add_argument("--output_dir", default="", help="Where to save annotated image. Default: image_dir/{ModelName}/")
    args = parser.parse_args()

    if not os.path.isfile(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    # Determine model_dir: use CLI if provided, else fallback by model key
    default_model_dirs = {
        "qwen": "/ckpt/Qwen2.5-VL-7B-Instruct",
        "internvl": "/ckpt/InternVL3-8B-hf",
        "llava": "/ckpt/llava-onevision-7b-hf",
        "minicpm": "/ckpt/MiniCPM-V-4_5",
    }
    model_dir = args.model_dir if args.model_dir else default_model_dirs[args.model]
    if not os.path.isdir(model_dir):
        raise NotADirectoryError(f"Model dir not found: {model_dir}")


    # 모델 선택
    if args.model == "qwen":
        model_name = "Qwen2.5-VL-7B"
        output_text = run_qwen(model_dir, args.image_path, args.task, args.prev_desc)
    elif args.model == "internvl":
        model_name = "InternVL3-8B-hf"
        output_text = run_internvl(model_dir, args.image_path, args.task, args.prev_desc)
    elif args.model == "llava":
        model_name = "LLaVA-OneVision-7B-hf"
        output_text = run_llava(model_dir, args.image_path, args.task, args.prev_desc)
    elif args.model == "minicpm":
        model_name = "MiniCPM-V"
        output_text = run_minicpm(model_dir, args.image_path, args.task, args.prev_desc, stream=True)
    else:
        raise ValueError("Unsupported model key.")

    print("\n" + "="*80)
    print(f"[{model_name}] OUTPUT:\n{output_text}")
    print("="*80 + "\n")

    # 저장 경로 구성
    if args.output_dir:
        out_dir = args.output_dir
    else:
        img_dir = os.path.dirname(os.path.abspath(args.image_path))
        out_dir = os.path.join(img_dir, model_name.replace("/", "_"))

    save_path = annotate_and_save(args.image_path, output_text, out_dir, suffix=model_name.replace("/", "_"))
    print(f"Annotated image saved to: {save_path}")


if __name__ == "__main__":
    # 예: CUDA_VISIBLE_DEVICES=1 python run_vlm.py --model llava --model_dir /ckpt/llava-onevision-7b-hf --image_path /result/VLM_test/piper_press_the_blue_button_screenshot/output_0002.jpg
    main()