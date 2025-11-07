#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import argparse
import time
from typing import Optional, Tuple, List

import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    AutoProcessor,
    PaliGemmaForConditionalGeneration,
    StoppingCriteria,
    StoppingCriteriaList,
)

# ---------------------------
# Utils
# ---------------------------

def enable_tf32():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img

def crop_roi(img: Image.Image, crop: Optional[List[float]]) -> Image.Image:
    """Crop ROI if crop is [x, y, w, h] in 0~1 normalized fractions."""
    if not crop:
        return img
    w, h = img.size
    x, y, cw, ch = crop
    x0 = int(max(0, min(w - 1, x * w)))
    y0 = int(max(0, min(h - 1, y * h)))
    x1 = int(max(1, min(w, x0 + cw * w)))
    y1 = int(max(1, min(h, y0 + ch * h)))
    if x1 <= x0 or y1 <= y0:
        return img
    return img.crop((x0, y0, x1, y1))

def annotate_image(image_path: str, text: str, save_path: str):
    """Pixel-aware wrapping with dynamic caption height (no clipping)."""
    img = Image.open(image_path).convert("RGB")
    w, h = img.size

    # Font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    left_margin = 12
    right_margin = 12
    top_margin = 10
    bottom_margin = 12
    line_spacing = 4
    max_width_px = max(50, w - (left_margin + right_margin))

    # Helpers
    def break_long_token_by_pixels(token: str, draw: ImageDraw.ImageDraw) -> list:
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
        wrapped_lines = []
        for raw_line in str(txt).splitlines():
            words = raw_line.split(" ")
            cur = ""
            for wtok in words:
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

    ascent, descent = font.getmetrics()
    line_height = ascent + descent + line_spacing
    tmp_draw = ImageDraw.Draw(img)
    lines = wrap_text_by_pixels(text, tmp_draw)

    caption_height = top_margin + bottom_margin + len(lines) * line_height
    total_h = h + caption_height

    canvas = Image.new("RGB", (w, total_h), (255, 255, 255))
    canvas.paste(img, (0, 0))
    draw = ImageDraw.Draw(canvas)

    y = h + top_margin + ascent
    x = left_margin
    for line in lines:
        draw.text((x, y - ascent), line, fill=(0, 0, 0), font=font)
        y += line_height

    ensure_dir(os.path.dirname(save_path))
    canvas.save(save_path)

def normalize_two_sentences(text: str) -> str:
    # Remove numbered bullets like "1) " or "2. "
    text = re.sub(r"\b[12]\)\s*", "", text)
    text = re.sub(r"\b[12]\.\s*", "", text)
    # Collapse spaces/commas
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r",(?:\s*,)+", ", ", text)
    # Avoid "unanswerable" wording
    text = text.replace("unanswerable", "uncertain")
    # Keep exactly two sentences by punctuation
    parts = re.split(r"(?<=[\.\?\!])\s+", text)
    parts = [p.strip() for p in parts if p.strip()]
    two = " ".join(parts[:2])
    if two and two[-1] not in ".?!":
        two += "."
    return two

# ---------------------------
# Stopping criteria: stop after two sentence-terminating punctuations
# ---------------------------

class TwoSentencesStop(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len):
        self.tok = tokenizer
        self.prompt_len = prompt_len
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        gen_ids = input_ids[0, self.prompt_len:]
        txt = self.tok.decode(gen_ids, skip_special_tokens=True)
        count = sum(txt.count(c) for c in [".", "?", "!"])
        return count >= 2

# ---------------------------
# Core
# ---------------------------

def build_prompt(task: str, prev_desc: str) -> str:
    return (
#        "<image>\n"
#        "You are an expert in robotic manipulation. "
#        f"Task: {task} "
#        f"Previous: {prev_desc} "
#        "Use only visible evidence.\n"
#        "Write exactly two sentences: "
#        "1) what the robot arm is doing now; "
#        "2) whether the task is done, not done, or uncertain. "
#        "Say 'done' only if the gripper visibly presses the button or the button visibly deforms.\n"
#        "Answer:"
        "<image>"
        "Answer in exactly two sentences."
        "1) What is the robot arm doing now?    "
        "2) Is the task(press th  blue button) done, not done, or uncertain?"
    )

def load_model_and_processor(model_dir: str):
    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)
    model = PaliGemmaForConditionalGeneration.from_pretrained(
        model_dir,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).eval()
    # Ensure pad token id is set to something reasonable
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = model.config.eos_token_id
    return processor, model

def infer_once(
    processor: AutoProcessor,
    model: PaliGemmaForConditionalGeneration,
    image: Image.Image,
    prompt_text: str,
    gen_cfg: dict,
) -> Tuple[str, float]:
    inputs = processor(text=[prompt_text], images=[image], return_tensors="pt")
    # Move to device and set correct dtypes:
    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            if k == "pixel_values":
                inputs[k] = v.to(device=model.device, dtype=torch.bfloat16, non_blocking=True)
            else:
                inputs[k] = v.to(device=model.device, non_blocking=True)

    prompt_len = inputs["input_ids"].shape[1]
    stoppers = StoppingCriteriaList([TwoSentencesStop(processor.tokenizer, prompt_len)])

    t0 = time.perf_counter()
    with torch.inference_mode():
        out_ids = model.generate(
            **inputs,
            stopping_criteria=stoppers,
            **gen_cfg,
        )
    dt = (time.perf_counter() - t0) * 1000.0

    gen = out_ids[0, prompt_len:]
    text = processor.decode(gen, skip_special_tokens=True).strip()
    text = normalize_two_sentences(text)
    return text, dt

# ---------------------------
# Main
# ---------------------------v

def main():
    parser = argparse.ArgumentParser(description="Evaluate PaliGemma v1/v2 on a single image with robust decoding.")
    parser.add_argument("--version", type=int, choices=[1, 2], default=1, help="1: paligemma-3b-mix-448, 2: paligemma2-3b-mix-448")
    parser.add_argument("--model_dir", type=str, default="", help="Override local model dir.")
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--task", type=str, default="Press the blue button on the table.")
    parser.add_argument("--prev_desc", type=str, default="The robot arm is positioned above the blue button, ready to press it. The task is not done.")
    parser.add_argument("--out_dir", type=str, default="/result/VLM_test/piper_press_the_blue_button_screenshot/PaliGemma")
    parser.add_argument("--crop", type=float, nargs=4, metavar=("X","Y","W","H"), help="Optional ROI crop in fractions (0~1), e.g., --crop 0.25 0.25 0.5 0.5")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--min_new_tokens", type=int, default=20)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3)
    parser.add_argument("--repetition_penalty", type=float, default=1.02)
    args = parser.parse_args()

    enable_tf32()

    # Resolve model dir
    if args.model_dir:
        model_dir = args.model_dir
    else:
        model_dir = "/ckpt/paligemma-3b-mix-448" if args.version == 1 else "/ckpt/paligemma2-3b-mix-448"

    print(f"[INFO] Using PaliGemma at: {model_dir}")
    processor, model = load_model_and_processor(model_dir)

    # Load image (with optional crop)
    img = load_image(args.image_path)
    img_for_model = crop_roi(img, args.crop)

    prompt_text = build_prompt(args.task, args.prev_desc)

    gen_cfg = dict(
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        do_sample=False,
        num_beams=1,
        eos_token_id=processor.tokenizer.eos_token_id,
        # pad_token_id intentionally not forced here; model.config holds it
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        repetition_penalty=args.repetition_penalty,
    )

    out_text, latency_ms = infer_once(processor, model, img_for_model, prompt_text, gen_cfg)
    print(out_text)

    # Save annotated image (uses the original image to visualize)
    ensure_dir(args.out_dir)
    base = os.path.splitext(os.path.basename(args.image_path))[0]
    suffix = f"_PaliGemma_v{args.version}.jpg"
    save_path = os.path.join(args.out_dir, base + suffix)
    annotate_image(args.image_path, out_text, save_path)
    print(f"Annotated image saved to: {save_path}")
    print(f"Inference time (E2E generate only): {latency_ms:.1f} ms")


if __name__ == "__main__":
    main()
