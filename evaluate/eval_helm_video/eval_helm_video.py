#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
eval_video_baseline.py

Evaluation script for HeLM Video Baseline (Multi-frame Input).
Supports Qwen2.5-VL with variable number of input images.

export PYTHONPATH=$(pwd)
CUDA_VISIBLE_DEVICES=6 python evaluate/eval_helm_video/eval_helm_video.py \
  --jsonl /data/ghkim/helm_data/helm_video_task_inter/merged/all_val.jsonl \
  --base_model /backups/ghkim/HLP_HeLM_video/HeLM_video_qwen3b_task_inter_0128_ddp_re/checkpoint-4000 \
  --out_jsonl /data/ghkim/helm_data/helm_video_task_inter/merged/eval_results/video_4k_preds.jsonl \
  --batch_size 1 \
  --attn_impl sdpa \
  --max_samples 50

CUDA_VISIBLE_DEVICES=4 python evaluate/eval_helm_video/eval_helm_video.py \
  --jsonl /data/ghkim/helm_data/helm_video_task_inter/merged/all_val.jsonl \
  --base_model /backups/ghkim/HLP_HeLM_video/HeLM_video_qwen3b_task_inter_0128_ddp_re/checkpoint-5000 \
  --out_jsonl /data/ghkim/helm_data/helm_video_task_inter/merged/eval_results/video_4k_preds.jsonl \
  --batch_size 1 \
  --attn_impl sdpa \
  --max_samples 100
  
CUDA_VISIBLE_DEVICES=5 python evaluate/eval_helm_video/eval_helm_video.py \
  --jsonl /data/ghkim/helm_data/helm_video_task_5/find_object_in_drawer/visual_memory_jsonl/val.jsonl \
  --base_model /ckpt/Qwen2.5-VL-7B-Instruct \
  --adapter /backups/ghkim/HLP_HeLM_video/HeLM_video_find_object_in_drawer_0129_ddpre/checkpoint-500 \
  --out_jsonl /data/ghkim/helm_data/helm_video_task_5/find_object_in_drawer/eval_results/video_500_preds.jsonl \
  --batch_size 1 \
  --attn_impl sdpa \
  --max_samples 100
  
CUDA_VISIBLE_DEVICES=6 python evaluate/eval_helm_video/eval_helm_video.py \
  --jsonl /data/ghkim/helm_data/helm_video_task_5/find_object_in_drawer/visual_memory_jsonl/val.jsonl \
  --base_model /ckpt/Qwen2.5-VL-7B-Instruct \
  --out_jsonl /data/ghkim/helm_data/helm_video_task_5/find_object_in_drawer/eval_results/video_5k_preds.jsonl \
  --batch_size 1 \
  --attn_impl sdpa \
  --max_samples 100
"""
max_pixels=310000
# min_pixels=50176

import argparse
import json
import logging
import random
import re
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import yaml

from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)
from peft import PeftModel

logger = logging.getLogger("eval_video")


# -------------------------
# Utils: JSONL & YAML
# -------------------------
def read_jsonl(path_or_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    p = Path(path_or_dir)
    rows: List[Dict[str, Any]] = []
    if p.is_file():
        files = [p]
    else:
        files = sorted(p.rglob("*.jsonl"))

    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue
                try:
                    rows.append(json.loads(line))
                except Exception as e:
                    logger.warning(f"JSON parse error {fp}:{ln} - {e}")
    return rows


def write_jsonl(path: Union[str, Path], rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_yaml_loose(text: str) -> Dict[str, Any]:
    if text is None: return {}
    s = text.strip()
    s = re.sub(r"^```(?:yaml)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    if "\nassistant\n" in s:
        s = s.split("\nassistant\n", 1)[-1].strip()
    try:
        out = yaml.safe_load(s)
        if isinstance(out, dict): return out
    except:
        pass
    # Fallback parsing
    lines = [ln for ln in s.splitlines() if ":" in ln]
    try:
        out = yaml.safe_load("\n".join(lines))
        if isinstance(out, dict): return out
    except:
        return {}
    return {}


def norm_str(x: Any) -> str:
    return str(x).strip().replace("\n", " ") if x is not None else "None"


def norm_bool(x: Any) -> Optional[bool]:
    if isinstance(x, bool): return x
    if x is None: return None
    s = str(x).strip().lower()
    return True if s in ["true", "yes", "1"] else False if s in ["false", "no", "0"] else None


# -------------------------
# Dataset & Collator
# -------------------------
def _load_images_from_list(image_paths: Union[List[str], Dict[str, str], str]) -> List[Image.Image]:
    """
    Load images from list of paths (Video Baseline) OR dict (HeLM standard).
    Handles:
      - List: ["/path/1.jpg", "/path/2.jpg"]
      - Dict: {"table": "/path/1.jpg"}
      - String: "/path/1.jpg"
    """
    # 1. 딕셔너리 처리
    if isinstance(image_paths, dict):
        if "table" in image_paths:
            image_paths = [image_paths["table"]]
        else:
            image_paths = list(image_paths.values())

    # 2. 단일 문자열 처리
    elif isinstance(image_paths, str):
        image_paths = [image_paths]

    # 3. 리스트 처리
    imgs = []
    for p in image_paths:
        if os.path.exists(p):
            try:
                imgs.append(Image.open(p).convert("RGB"))
            except Exception as e:
                logger.warning(f"Error opening image {p}: {e}")
                imgs.append(Image.new("RGB", (224, 224), (0, 0, 0)))
        else:
            logger.warning(f"Image not found: {p}")
            imgs.append(Image.new("RGB", (224, 224), (0, 0, 0)))

    return imgs


class VideoEvalDataset(Dataset):
    def __init__(self, rows: List[Dict[str, Any]], processor: AutoProcessor):
        self.rows = rows
        self.processor = processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        row = self.rows[i]
        user_prompt = str(row.get("user_prompt", ""))
        gt_text = str(row.get("gt_text", ""))

        # 1. Load Images (Variable Length)
        imgs = _load_images_from_list(row.get("images", []))
        num_images = len(imgs)

        # 2. Construct Chat Messages
        user_content = [{"type": "image"} for _ in range(num_images)]
        user_content.append({"type": "text", "text": user_prompt})

        messages = [{"role": "user", "content": user_content}]

        # Apply Template
        prompt_string = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process Inputs
        model_inputs = self.processor(
            text=prompt_string,
            images=imgs,
            return_tensors="pt",
            padding=False,
        )

        input_ids = model_inputs["input_ids"].squeeze(0)
        attention_mask = model_inputs["attention_mask"].squeeze(0)
        pixel_values = model_inputs["pixel_values"].squeeze(0)

        # Grid THW handling
        grid_thw = model_inputs.get("image_grid_thw", None)
        if grid_thw is not None:
            if grid_thw.ndim >= 3:
                grid_thw = grid_thw.squeeze(0)
            elif grid_thw.ndim == 1 and grid_thw.numel() == 3:
                grid_thw = grid_thw.unsqueeze(0)

            if grid_thw.ndim != 2 or grid_thw.size(-1) != 3:
                if grid_thw.numel() % 3 == 0:
                    grid_thw = grid_thw.view(-1, 3)
                else:
                    raise ValueError(f"Bad grid_thw shape: {tuple(grid_thw.shape)}")

        return {
            "uid": row.get("uid", f"idx{i}"),
            "label": row.get("label", "UNKNOWN"),
            "mode": row.get("mode", "UNKNOWN"),
            "row": row,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thw,
        }


class VideoEvalCollator:
    def __init__(self, processor):
        self.tokenizer = processor.tokenizer
        self.tokenizer.padding_side = "left"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        pad_id = self.tokenizer.pad_token_id
        input_ids = self._left_pad([f["input_ids"] for f in features], pad_id)
        attention_mask = self._left_pad([f["attention_mask"] for f in features], 0)
        pixel_values = torch.cat([f["pixel_values"] for f in features], dim=0)

        batch = {
            "uids": [f["uid"] for f in features],
            "labels": [f["label"] for f in features],
            "modes": [f["mode"] for f in features],
            "rows": [f["row"] for f in features],
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }

        if features[0].get("image_grid_thw") is not None:
            grids = []
            for f in features:
                g = f["image_grid_thw"]
                if g.ndim == 1: g = g.unsqueeze(0)
                grids.append(g)
            batch["image_grid_thw"] = torch.cat(grids, dim=0)
        else:
            batch["image_grid_thw"] = None

        return batch

    def _left_pad(self, tensors, pad_val):
        max_len = max(t.size(0) for t in tensors)
        out = tensors[0].new_full((len(tensors), max_len), pad_val)
        for i, t in enumerate(tensors):
            out[i, -t.size(0):] = t
        return out


# -------------------------
# Evaluation Logic
# -------------------------
def load_model(base_model, adapter, use_qlora, attn_impl):
    base_model_path = "/ckpt/Qwen2.5-VL-3B-Instruct"

    processor = AutoProcessor.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        # min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    bnb_config = None
    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map="auto",
    )

    if adapter:
        print(f"Loading LoRA adapter from {adapter}")
        model = PeftModel.from_pretrained(model, adapter)

    model.eval()
    return model, processor


@torch.no_grad()
def run_eval(args):
    # 1. Load Data
    rows = read_jsonl(args.jsonl)
    if args.max_samples > 0:
        random.seed(args.seed)
        random.shuffle(rows)
        rows = rows[:args.max_samples]
    print(f"Evaluating {len(rows)} samples...")

    # 2. Load Model
    model, processor = load_model(args.base_model, args.adapter, args.use_qlora, args.attn_impl)

    # 3. DataLoader
    ds = VideoEvalDataset(rows, processor)
    collate = VideoEvalCollator(processor)
    dl = DataLoader(ds, batch_size=args.batch_size, collate_fn=collate, num_workers=4)

    # 4. Loop
    results = []
    stats = defaultdict(lambda: {"n": 0, "ok": 0})

    for batch in dl:
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        pixel_values = batch["pixel_values"].to(model.device)
        grid_thw = batch["image_grid_thw"].to(model.device) if batch["image_grid_thw"] is not None else None

        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=grid_thw,
            max_new_tokens=128,
            do_sample=False,
        )

        decoded_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)

        input_lens = attention_mask.sum(dim=1)
        final_preds = []
        for i, seq in enumerate(generated_ids):
            # Remove input tokens from output
            pred = processor.decode(seq[input_ids.shape[1]:], skip_special_tokens=True)
            final_preds.append(pred.strip())

        # Metric Calculation
        for i, pred_text in enumerate(final_preds):
            row = batch["rows"][i]
            gt_text = row.get("gt_text", "")
            gt_yaml = row.get("gt_yaml") or parse_yaml_loose(gt_text)
            pred_yaml = parse_yaml_loose(pred_text)

            mode = batch["modes"][i]
            label = batch["labels"][i]

            is_correct = False
            if mode == "DETECT":
                gt = norm_bool(gt_yaml.get("Event_Detected"))
                pr = norm_bool(pred_yaml.get("Event_Detected"))
                is_correct = (gt == pr) and (gt is not None)
                if not is_correct:
                    # Debug print
                    print(f"[Incorrect DETECT] UID: {batch['uids'][i]} | GT: {gt} | PR: {pr}")
            else:
                gt_act = norm_str(gt_yaml.get("Action_Command"))
                pr_act = norm_str(pred_yaml.get("Action_Command"))
                is_correct = (gt_act == pr_act)
                if not is_correct:
                    # Debug print
                    print(f"[Incorrect UPDATE] UID: {batch['uids'][i]} | GT: {gt_act} | PR: {pr_act}")

            stats["total"]["n"] += 1
            stats["total"]["ok"] += int(is_correct)
            stats[mode]["n"] += 1
            stats[mode]["ok"] += int(is_correct)
            stats[label]["n"] += 1
            stats[label]["ok"] += int(is_correct)

            # [수정] 결과 저장 시 이미지 경로 포함
            results.append({
                "uid": batch["uids"][i],
                "mode": mode,
                "label": label,
                "images": row.get("images", []),  # <--- 이미지 경로 저장
                "gt_text": gt_text,
                "pred_text": pred_text,
                "correct": is_correct
            })

    # 5. Save & Print
    write_jsonl(args.out_jsonl, results)

    print("=" * 30)
    print("Evaluation Results")
    print("=" * 30)
    for k, v in stats.items():
        acc = v['ok'] / v['n'] * 100 if v['n'] > 0 else 0
        print(f"[{k}] Acc: {acc:.2f}% ({v['ok']}/{v['n']})")
    print(f"Saved to {args.out_jsonl}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter", type=str, default=None)
    parser.add_argument("--out_jsonl", type=str, required=True)

    parser.add_argument("--batch_size", type=int, default=1, help="Small batch size for video")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_qlora", action="store_true", default=True)
    parser.add_argument("--attn_impl", type=str, default="sdpa")

    args = parser.parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()