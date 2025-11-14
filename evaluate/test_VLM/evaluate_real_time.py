#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import json
import pickle
import subprocess
from typing import Dict, Any, List, Tuple, Optional

import torch
from PIL import Image, ImageDraw, ImageFont
from textwrap import wrap

# =========================
# Config & Prompts
# =========================
DEFAULT_OUTPUT_ROOT = "/result/VLM_test_short"

SYSTEM_PROMPT = (
    "You are an expert in robotic manipulation image analysis. "
    "Use only the current image and provided text. Respond in exactly two sentences."
)
# SYSTEM_PROMPT = (
#     "You are an expert in robotic manipulation image analysis."
#     "Given: Task description, Previous output (last frame’s action + done status), Current image  "
#
#     "Write exactly two sentences:"
#     "1. Describe what the robot is visibly doing.  "
#     "2. Judge if the task is done, not done, or uncertain."
#
#     "Use only visible evidence from the current image."
#     "Refer to the previous output only for context."
#     "Mark “done” only when physical contact or the final result is clearly seen; otherwise say “not done” or “uncertain.”"
#     "Use short, visual verbs like grasping, pressing, placing, releasing."
#
#     "Task: Press the blue button on the table."
#     "Previous description: The robot's gripper is in contact with the blue button, indicating that it is attempting to press it. The task is not done as the button is still not visibly depressed."
# )

DEFAULT_TASK = "Press the blue button on the table."
FIRST_PREV_DESC = "No previous output."

# 모델별 기본 경로 (필요 시 --model_dir 로 override)
DEFAULT_MODEL_DIRS = {
    "qwen": "/ckpt/Qwen2.5-VL-7B-Instruct",
    "intern": "/ckpt/InternVL3-8B-hf",
    "llava": "/ckpt/llava-onevision-7b-hf",
    "minicpm": "/ckpt/MiniCPM-V-4_5",
}

# =========================
# Utilities
# =========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def video_to_frames(video_path: str, frames_dir: str) -> List[str]:
    """
    Extract frames every 1 second including t=0.
    1) Try: ffmpeg -vf fps=1 -start_number 0
    2) Fallback: OpenCV sampling by time
    Return sorted list of frame paths.
    """
    ensure_dir(frames_dir)
    out_pattern = os.path.join(frames_dir, "frame_%05d.jpg")

    # Try ffmpeg
    try:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vf", "fps=1", "-start_number", "0", out_pattern
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode == 0:
            frames = sorted(
                [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")]
            )
            if frames:
                return frames
            else:
                # fallthrough to cv2
                pass
        else:
            # fallthrough to cv2
            pass
    except FileNotFoundError:
        # ffmpeg not installed → fallback
        pass

    # Fallback: OpenCV (sample by 1 second)
    try:
        import cv2
    except ImportError:
        raise RuntimeError("ffmpeg unavailable and OpenCV not installed. Install opencv-python or install ffmpeg.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        # Fallback guess: 5Hz
        fps = 5.0
    step = int(round(fps * 1.0))  # 1-second interval in frames

    idx = 0
    saved = []
    frame_id = 0
    while True:
        ret = cap.grab()
        if not ret:
            break
        if idx % step == 0:
            ret2, frame = cap.retrieve()
            if not ret2:
                break
            save_path = os.path.join(frames_dir, f"frame_{frame_id:05d}.jpg")
            cv2.imwrite(save_path, frame)
            saved.append(save_path)
            frame_id += 1
        idx += 1

    cap.release()
    if not saved:
        raise RuntimeError("No frames extracted via OpenCV fallback.")
    return saved


def annotate_image(image_path: str, text: str, save_path: str):
    """
    Draws `text` under the image with robust pixel-aware wrapping and a dynamically
    sized caption area so that no line is clipped.
    """
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # Font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    # Layout params
    left_margin = 12
    right_margin = 12
    top_margin = 10
    bottom_margin = 12
    line_spacing = 4
    max_width_px = max(50, w - (left_margin + right_margin))

    # Prepare wrapper helpers
    def break_long_token_by_pixels(token: str, draw: ImageDraw.ImageDraw) -> list:
        """Break a very long token (no spaces) into pixel-fitting chunks."""
        pieces = []
        if not token:
            return [token]
        start = 0
        while start < len(token):
            lo, hi = 1, len(token) - start
            best = 1
            while lo <= hi:
                mid = (lo + hi) // 2
                seg = token[start:start + mid]
                if draw.textlength(seg, font=font) <= max_width_px:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            pieces.append(token[start:start + best])
            start += best
        return pieces

    def wrap_text_by_pixels(txt: str, draw: ImageDraw.ImageDraw) -> list:
        """Wraps text to max_width_px using pixel measurements."""
        wrapped_lines = []
        for raw_line in str(txt).splitlines():
            words = raw_line.split(" ")
            cur = ""
            for wtok in words:
                # If token itself is too wide, break it
                tokens = [wtok]
                if draw.textlength(wtok, font=font) > max_width_px:
                    tokens = break_long_token_by_pixels(wtok, draw)
                for tk in tokens:
                    candidate = tk if cur == "" else (cur + " " + tk)
                    if draw.textlength(candidate, font=font) <= max_width_px:
                        cur = candidate
                    else:
                        if cur != "":
                            wrapped_lines.append(cur)
                        cur = tk
            wrapped_lines.append(cur)
        return wrapped_lines

    # Measure line height accurately
    ascent, descent = font.getmetrics()
    line_height = ascent + descent + line_spacing

    # Wrap text using pixel measurement
    tmp_draw = ImageDraw.Draw(img)
    lines = wrap_text_by_pixels(text, tmp_draw)

    # Compute caption box height
    caption_height = top_margin + bottom_margin + len(lines) * line_height
    total_h = h + caption_height

    # Compose final canvas
    canvas = Image.new("RGB", (w, total_h), (255, 255, 255))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)

    # Draw lines
    y = h + top_margin + ascent  # baseline start uses ascent
    x = left_margin
    for line in lines:
        draw.text((x, y - ascent), line, fill=(0, 0, 0), font=font)
        y += line_height

    ensure_dir(os.path.dirname(save_path))
    canvas.save(save_path)


def preview_text(s: str, n: int = 120) -> str:
    s = s.replace("\n", " ")
    return s[:n]

# Helper to extract dataset name from video path
def extract_dataset_name(video_path: str) -> str:
    """
    Try to infer dataset name from a path like:
    /data/<dataset>/videos/chunk-000/.../episode_xxx.mp4
    Heuristic:
      - If 'videos' is in the path, take the directory right before 'videos' as dataset.
      - Else, fall back to the grandparent directory name.
    """
    norm = os.path.normpath(video_path)
    parts = norm.split(os.sep)
    if "videos" in parts:
        idx = parts.index("videos")
        if idx > 0:
            return parts[idx - 1]
    # Fallback: parent of parent
    parent = os.path.dirname(norm)
    grandparent = os.path.dirname(parent)
    return os.path.basename(grandparent) or os.path.basename(parent)


# =========================
# Model Loaders & Inference
# =========================
def load_qwen(model_dir: str):
    from transformers import AutoProcessor
    from transformers import Qwen2_5_VLForConditionalGeneration
    from qwen_vl_utils import process_vision_info

    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_dir, device_map="auto", dtype=torch.bfloat16, local_files_only=True
    )
    model.eval()
    return {"processor": processor, "model": model, "process_vision_info": process_vision_info}

def infer_qwen(bundle: Dict[str, Any], image_path: str, task: str, prev_desc: str, max_new_tokens: int) -> Tuple[str, Optional[int], Optional[int]]:
    processor = bundle["processor"]
    model = bundle["model"]
    process_vision_info = bundle["process_vision_info"]

    user_prompt = (
        f"Task: {task}\n"
        f"Previous description: {prev_desc}\n"
        "Describe what the robot is doing in this image and whether the task is done."
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [{"type":"image","image": image_path}, {"type":"text","text": user_prompt}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = {k: (v.to(model.device) if hasattr(v,"to") else v) for k,v in inputs.items()}

    in_tokens = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else None

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0)

    prompt_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
    out_text = processor.batch_decode(out_ids[:, prompt_len:], skip_special_tokens=True)[0].strip()
    out_tokens = int(out_ids.shape[1] - prompt_len) if prompt_len else None
    return out_text, in_tokens, out_tokens


def load_internvl(model_dir: str):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, device_map="auto", dtype=torch.bfloat16, local_files_only=True, trust_remote_code=True
    )
    model.eval()
    return {"processor": processor, "model": model}

def infer_internvl(bundle: Dict[str, Any], image_path: str, task: str, prev_desc: str, max_new_tokens: int) -> Tuple[str, Optional[int], Optional[int]]:
    processor = bundle["processor"]
    model = bundle["model"]

    user_prompt = (
        f"Task: {task}\n"
        f"Previous description: {prev_desc}\n"
        "Describe what the robot is doing in this image and whether the task is done."
    )
    messages = [
        {"role": "system", "content": [{"type":"text","text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type":"image","image": image_path}, {"type":"text","text": user_prompt}]}
    ]

    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[chat], images=[Image.open(image_path).convert("RGB")],
                       return_tensors="pt", padding=True)
    inputs = {k: (v.to(model.device) if hasattr(v,"to") else v) for k,v in inputs.items()}
    in_tokens = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else None

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0)

    tok = getattr(processor, "tokenizer", processor)
    prompt_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0
    out_text = tok.batch_decode(out_ids[:, prompt_len:], skip_special_tokens=True)[0].strip()
    out_tokens = int(out_ids.shape[1] - prompt_len) if prompt_len else None
    return out_text, in_tokens, out_tokens


def load_llava(model_dir: str):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_dir, device_map="auto", dtype=torch.bfloat16, local_files_only=True, trust_remote_code=True
    )
    model.eval()
    return {"processor": processor, "model": model}

def infer_llava(bundle: Dict[str, Any], image_path: str, task: str, prev_desc: str, max_new_tokens: int) -> Tuple[str, Optional[int], Optional[int]]:
    processor = bundle["processor"]
    model = bundle["model"]

    user_prompt = (
        f"Task: {task}\n"
        f"Previous description: {prev_desc}\n"
        "Describe what the robot is doing in this image and whether the task is done."
    )
    messages = [
        {"role": "system", "content": [{"type":"text","text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [{"type":"image","image": image_path}, {"type":"text","text": user_prompt}]}
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    in_tokens = int(inputs["input_ids"].shape[1]) if "input_ids" in inputs else None

    with torch.inference_mode():
        out_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=0.0)

    out_text = processor.decode(out_ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    out_tokens = int(out_ids.shape[1] - inputs["input_ids"].shape[1]) if "input_ids" in inputs else None
    return out_text, in_tokens, out_tokens


def load_minicpm(model_dir: str):
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained(
        model_dir, trust_remote_code=True, attn_implementation="sdpa", dtype=torch.bfloat16
    ).eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return {"model": model, "tokenizer": tokenizer}

def infer_minicpm(bundle: Dict[str, Any], image_path: str, task: str, prev_desc: str, max_new_tokens: int) -> Tuple[str, Optional[int], Optional[int]]:
    """
    MiniCPM은 system 역할 미지원 → system 내용을 user에 합쳐서 한 덩어리로 입력.
    stream=False로 한 방 응답 받아서 엔드투엔드 시간 측정 단순화.
    """
    model = bundle["model"]
    tokenizer = bundle["tokenizer"]
    img = Image.open(image_path).convert("RGB")

    user_full = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Task: {task}\n"
        f"Previous description: {prev_desc}\n"
        "Describe what the robot is doing in this image and whether the task is done."
    )
    messages = [{"role": "user", "content": [img, user_full]}]

    # MiniCPM chat은 토큰 길이 직접 얻기 어렵므로 NA 처리
    in_tokens = None
    out = model.chat(
        msgs=messages,
        tokenizer=tokenizer,
        enable_thinking=False,
        stream=False,   # 한 번에 받아서 단순화
        max_new_tokens=max_new_tokens
    )
    out_text = out.strip()
    out_tokens = None
    return out_text, in_tokens, out_tokens


# =========================
# Main flow
# =========================
def main():
    parser = argparse.ArgumentParser(description="Video → frames(1s) → sequential VLM inference with prev_desc chaining.")
    parser.add_argument("--video_path", required=True, help="Path to video file.")
    parser.add_argument("--model", required=True, choices=["qwen", "intern", "llava", "minicpm"])
    parser.add_argument("--model_dir", default="", help="Local model dir. If empty, use default map.")
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT, help="Root folder to save outputs.")
    parser.add_argument("--max_new_tokens", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=1, help="Number of initial frames to exclude from average.")
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--frames_dir", default="", help="If provided and exists, reuse extracted frames here.")
    args = parser.parse_args()

    if not os.path.isfile(args.video_path):
        raise FileNotFoundError(args.video_path)

    model_dir = args.model_dir or DEFAULT_MODEL_DIRS[args.model]
    if not os.path.isdir(model_dir):
        raise NotADirectoryError(f"Model dir not found: {model_dir}")

    # Paths
    video_basename = os.path.splitext(os.path.basename(args.video_path))[0]  # e.g., episode_000000
    model_name_for_path = {
        "qwen": "Qwen",
        "intern": "InternVL",
        "llava": "LLaVA",
        "minicpm": "MiniCPM",
    }[args.model]
    dataset_name = extract_dataset_name(args.video_path)  # e.g., piper_press_the_blue_button_ep60

    # Extract frames (or reuse)
    frames_base = args.frames_dir if args.frames_dir else os.path.join(args.output_root, dataset_name, video_basename, "frames")
    ensure_dir(frames_base)
    if args.frames_dir and os.path.isdir(args.frames_dir) and len(os.listdir(args.frames_dir)) > 0:
        frame_paths = sorted([os.path.join(args.frames_dir, f) for f in os.listdir(args.frames_dir) if f.endswith(".jpg")])
        if not frame_paths:
            frame_paths = video_to_frames(args.video_path, frames_base)
    else:
        frame_paths = video_to_frames(args.video_path, frames_base)

    if not frame_paths:
        raise RuntimeError("No frames extracted.")

    # Load model once (bf16)
    if args.model == "qwen":
        bundle = load_qwen(model_dir)
        infer_fn = infer_qwen
    elif args.model == "intern":
        bundle = load_internvl(model_dir)
        infer_fn = infer_internvl
    elif args.model == "llava":
        bundle = load_llava(model_dir)
        infer_fn = infer_llava
    elif args.model == "minicpm":
        bundle = load_minicpm(model_dir)
        infer_fn = infer_minicpm
    else:
        raise ValueError("Unsupported model key.")

    # Results & logging (defer image saving to the end)
    results: List[Dict[str, Any]] = []
    prev_desc = FIRST_PREV_DESC

    # Warmup note
    warmup = max(0, int(args.warmup))

    # Inference loop
    for i, fp in enumerate(sorted(frame_paths)):
        t0 = time.perf_counter()
        try:
            out_text, in_tok, out_tok = infer_fn(bundle, fp, args.task, prev_desc, args.max_new_tokens)
        except Exception as e:
            out_text, in_tok, out_tok = f"[ERROR] {e}", None, None
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0  # end-to-end as requested

        results.append({
            "index": i,
            "frame_path": fp,
            "latency_ms": round(latency_ms, 3),
            "input_tokens": in_tok if in_tok is not None else "NA",
            "output_tokens": out_tok if out_tok is not None else "NA",
            "output_text": out_text,
            "output_text_preview": preview_text(out_text, 120),
        })

        # Chain prev_desc
        prev_desc = out_text

        # Progress print
        print(f"[{i:03d}] {os.path.basename(fp)} | {latency_ms:.1f} ms | {preview_text(out_text)}")

    # Compute average latency excluding warmup frames
    valid_latencies = [r["latency_ms"] for r in results[warmup:]] if len(results) > warmup else []
    avg_ms = sum(valid_latencies)/len(valid_latencies) if valid_latencies else float("nan")
    print("\n" + "="*80)
    print(f"Frames: {len(results)} | Warmup skipped: {warmup} | Avg latency (E2E, excl. saving): {avg_ms:.2f} ms")
    print("="*80 + "\n")

    # Save logs & annotated images (post-process; saving time excluded from avg)
    out_dir_model = os.path.join(args.output_root, dataset_name, video_basename, model_name_for_path)
    ensure_dir(out_dir_model)

    # CSV (Option B: preview only)
    csv_path = os.path.join(out_dir_model, "results.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("index,frame_path,latency_ms,input_tokens,output_tokens,output_text_preview\n")
        for r in results:
            line = f"{r['index']},{r['frame_path']},{r['latency_ms']},{r['input_tokens']},{r['output_tokens']},\"{r['output_text_preview']}\"\n"
            f.write(line)

    # JSONL (full text)
    jsonl_path = os.path.join(out_dir_model, "results.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Pickle
    pkl_path = os.path.join(out_dir_model, "results.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(results, f)

    # Annotated images (deferred save)
    for r in results:
        # "{output_root}/{video_basename}/{model_name}/frame_00000_{model_name}.jpg"
        base = os.path.splitext(os.path.basename(r["frame_path"]))[0]  # frame_00000
        save_path = os.path.join(out_dir_model, f"{base}_{model_name_for_path}.jpg")
        try:
            annotate_image(r["frame_path"], r["output_text"], save_path)
        except Exception as e:
            print(f"[WARN] annotate failed for {r['frame_path']}: {e}")

    print(f"Logs saved: {csv_path}")
    print(f"JSONL saved: {jsonl_path}")
    print(f"Pickle saved: {pkl_path}")
    print(f"Annotated images saved under: {out_dir_model}")


if __name__ == "__main__":
    # CUDA_VISIBLE_DEVICES는 외부에서 설정
    main()

# Example ============================================
# CUDA_VISIBLE_DEVICES=1 python run_vlm_sequence.py \
#   --video_path /data/piper_press_the_blue_button_ep60/videos/chunk-000/observation.images.table/episode_000000.mp4 \
#   --model intern \
#   --model_dir /ckpt/InternVL3-8B-hf \
#   --output_root /result/VLM_test \
#   --max_new_tokens 80 \
#   --warmup 2