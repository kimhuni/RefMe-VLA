#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import logging
import random
import re
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


logger = logging.getLogger("eval_helm_v3")

"""
CUDA_VISIBLE_DEVICES=4 python evaluate/eval_helm_v2/eval_helm_hlp_v3.py \
  --jsonl /data/ghkim/helm_data/wipe_the_window/jsonl_v3/merged/wipe_the_window/all_val.jsonl \
  --base_model /ckpt/Qwen2.5-VL-7B-Instruct \
  --adapter /result/ghkim/HLP_HeLM_v3_wipe_the_window_2240_0101/checkpoint-1900 \
  --max_samples 200 --seed 123 \
  --out_jsonl /data/ghkim/helm_data/result/HLP_HeLM_v3_wipe_the_window_2240_0101/eval_preds_1900.jsonl
  
CUDA_VISIBLE_DEVICES=5 python evaluate/eval_helm_v2/eval_helm_hlp_v3.py \
  --jsonl /data/ghkim/helm_data/press_the_button_nolight/jsonl_v3/merged/all_val.jsonl \
  --base_model /ckpt/Qwen2.5-VL-7B-Instruct \
  --adapter /result/ghkim/HLP_HeLM_v3_press_N_3320_0101/checkpoint-3200 \
  --max_samples 200 --seed 123 \
  --out_jsonl /data/ghkim/helm_data/result/HLP_HeLM_v3_press_N_3320_0101/eval_preds_vanilla.jsonl
"""

# -------------------------
# JSONL IO
# -------------------------
def read_jsonl(path_or_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    p = Path(path_or_dir)
    rows: List[Dict[str, Any]] = []
    if p.is_file():
        files = [p]
    else:
        files = sorted(p.rglob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No .jsonl under: {p}")

    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception as e:
                    raise RuntimeError(f"JSON parse error: {fp} line {ln}: {e}")
    return rows


def write_jsonl(path: Union[str, Path], rows: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# -------------------------
# YAML parsing helpers
# -------------------------
def parse_yaml_loose(text: str) -> Dict[str, Any]:
    """
    Robust-ish YAML parse:
    - strips code fences
    - tries to parse as YAML
    """
    if text is None:
        return {}
    s = text.strip()

    # strip ```yaml fences
    s = re.sub(r"^```(?:yaml)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    # If the decoded text still contains role headers (rare), keep only the assistant tail.
    # This protects against templates that include 'system/user/assistant' markers.
    if "\nassistant\n" in s:
        s = s.split("\nassistant\n", 1)[-1].strip()

    # sometimes model outputs extra text before yaml; try to find first "Key:"
    # For DETECT: "Event_Detected:"; For UPDATE: "Action_Command:" etc.
    # We'll just attempt full parse first.
    try:
        out = yaml.safe_load(s)
        if isinstance(out, dict):
            return out
    except Exception:
        pass

    # fallback: keep only lines that look like "k: v"
    lines = []
    for ln in s.splitlines():
        if ":" in ln:
            lines.append(ln)
    try:
        out = yaml.safe_load("\n".join(lines))
        if isinstance(out, dict):
            return out
    except Exception:
        return {}
    return {}


def norm_str(x: Any) -> str:
    if x is None:
        return "None"
    if isinstance(x, bool):
        return "true" if x else "false"
    s = str(x).strip()
    # normalize whitespace
    s = re.sub(r"\s+", " ", s)
    return s


def norm_bool(x: Any) -> Optional[bool]:
    if isinstance(x, bool):
        return x
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ["true", "yes", "y", "1"]:
        return True
    if s in ["false", "no", "n", "0"]:
        return False
    return None


# -------------------------
# Dataset / Collator (eval)
# -------------------------
@dataclass
class EvalConfig:
    jsonl_path: str
    model_name_or_path: str
    num_images: int = 1
    trust_remote_code: bool = True
    use_fast: bool = True
    padding_side: str = "left"


def _load_images(row: Dict[str, Any], num_images: int) -> List[Image.Image]:
    imgs: List[Image.Image] = []
    im = row.get("images", {})
    if not isinstance(im, dict):
        raise ValueError("row['images'] must be a dict")

    table = im.get("table", None)
    if table is None:
        raise ValueError("row['images']['table'] missing")
    imgs.append(Image.open(table).convert("RGB"))

    if num_images == 2:
        wrist = im.get("wrist", None)
        if wrist is None:
            raise ValueError("num_images=2 but row['images']['wrist'] missing")
        imgs.append(Image.open(wrist).convert("RGB"))
    return imgs


class HelmEvalDatasetV3(Dataset):
    def __init__(self, cfg: EvalConfig, rows: List[Dict[str, Any]], processor: AutoProcessor):
        self.cfg = cfg
        self.rows = rows
        self.processor = processor

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        row = self.rows[i]
        user_prompt = str(row.get("user_prompt", ""))
        gt_text = str(row.get("gt_text", ""))

        imgs = _load_images(row, self.cfg.num_images)

        user_content = []
        for _ in range(self.cfg.num_images):
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": user_prompt})

        messages = [
            {"role": "user", "content": user_content},
        ]

        prompt_string = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        model_inputs = self.processor(
            text=prompt_string,
            images=imgs,
            return_tensors="pt",
            padding=False,
        )

        # flatten to sample tensors (no batch dim)
        input_ids = model_inputs["input_ids"].squeeze(0)
        attention_mask = model_inputs["attention_mask"].squeeze(0)
        pixel_values = model_inputs["pixel_values"].squeeze(0)  # (N,C,H,W)
        grid_thw = model_inputs.get("image_grid_thw", None)

        # ---- grid_thw normalize (IMPORTANT) ----
        if grid_thw is not None:
            if grid_thw.ndim >= 3:
                grid_thw = grid_thw.squeeze(0)          # (N,3)
            elif grid_thw.ndim == 2:
                pass
            elif grid_thw.ndim == 1 and grid_thw.numel() == 3:
                grid_thw = grid_thw.unsqueeze(0)        # (1,3)
            else:
                raise ValueError(f"Bad image_grid_thw shape: {tuple(grid_thw.shape)}")

            if not (grid_thw.ndim == 2 and grid_thw.size(-1) == 3):
                raise ValueError(f"Bad image_grid_thw final shape: {tuple(grid_thw.shape)}")

        return {
            "uid": row.get("uid", f"idx{i}"),
            "label": row.get("label", "UNKNOWN"),
            "mode": row.get("mode", "UNKNOWN"),
            "user_prompt": user_prompt,
            "gt_text": gt_text,
            "gt_yaml": row.get("gt_yaml", None),   # optional
            "meta": row.get("meta", {}),
            "row": row,  # keep raw

            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thw,
        }


class EvalCollatorQwenVL:
    def __init__(self, processor: AutoProcessor):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        assert self.tokenizer is not None
        self.tokenizer.padding_side = "left"

    def _left_pad_1d(self, xs: List[torch.Tensor], pad_val: int) -> torch.Tensor:
        max_len = max(x.size(0) for x in xs)
        out = xs[0].new_full((len(xs), max_len), pad_val)
        for i, x in enumerate(xs):
            out[i, -x.size(0):] = x
        return out

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # meta fields
        uids = [f["uid"] for f in features]
        labels = [f["label"] for f in features]
        modes = [f["mode"] for f in features]
        user_prompts = [f["user_prompt"] for f in features]
        gt_texts = [f["gt_text"] for f in features]
        rows = [f["row"] for f in features]

        # tensors
        pad_id = self.tokenizer.pad_token_id
        input_ids = self._left_pad_1d([f["input_ids"] for f in features], pad_id)
        attention_mask = self._left_pad_1d([f["attention_mask"] for f in features], 0)
        pixel_values = torch.cat([f["pixel_values"] for f in features], dim=0)  # (B*N,C,H,W)

        batch: Dict[str, Any] = {
            "uids": uids,
            "labels": labels,
            "modes": modes,
            "user_prompts": user_prompts,
            "gt_texts": gt_texts,
            "rows": rows,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }

        # grid_thw concat
        if features[0].get("image_grid_thw") is not None:
            grids = []
            for f in features:
                g = f["image_grid_thw"]
                if g is None:
                    raise ValueError("Mixed None/non-None image_grid_thw in batch.")
                if g.ndim == 1 and g.numel() == 3:
                    g = g.unsqueeze(0)
                if not (g.ndim == 2 and g.size(-1) == 3):
                    raise ValueError(f"Bad per-sample grid_thw: {tuple(g.shape)}")
                grids.append(g)
            grid_thw = torch.cat(grids, dim=0)  # (B*N,3)
            if not (grid_thw.ndim == 2 and grid_thw.size(-1) == 3):
                raise ValueError(f"Bad batched grid_thw: {tuple(grid_thw.shape)}")
            batch["image_grid_thw"] = grid_thw
        else:
            batch["image_grid_thw"] = None

        return batch


# -------------------------
# Model loading
# -------------------------
def load_model_and_processor(
    base_model: str,
    adapter_path: Optional[str],
    use_qlora: bool,
    bf16: bool,
    attn_impl: str,
):
    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True, use_fast=True)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        processor.tokenizer.padding_side = "left"

    if use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if bf16 else torch.float16,
        )
    else:
        bnb_config = None

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if bf16 else torch.float16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )

    # if adapter_path:
    #     model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    return model, processor


# -------------------------
# Metrics
# -------------------------
def eval_detect(gt_yaml: Dict[str, Any], pred_yaml: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    # key is Event_Detected
    gt = norm_bool(gt_yaml.get("Event_Detected", None))
    pr = norm_bool(pred_yaml.get("Event_Detected", None))
    ok = (gt is not None) and (pr is not None) and (gt == pr)
    info = {"gt_event": gt, "pred_event": pr}
    return ok, info


def eval_update(gt_yaml: Dict[str, Any], pred_yaml: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    keys = ["Action_Command", "Working_Memory", "Episodic_Context"]
    per_key = {}
    all_ok = True
    for k in keys:
        gt = norm_str(gt_yaml.get(k, None))
        pr = norm_str(pred_yaml.get(k, None))
        match = (gt == pr)
        per_key[f"match_{k}"] = match
        per_key[f"gt_{k}"] = gt
        per_key[f"pred_{k}"] = pr
        all_ok = all_ok and match
    return all_ok, per_key


# -------------------------
# Main eval
# -------------------------
@torch.no_grad()
def run_eval(
    model,
    processor,
    rows: List[Dict[str, Any]],
    num_images: int,
    device: str,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    out_jsonl: str,
    log_every: int = 20,
):
    cfg = EvalConfig(
        jsonl_path="",
        model_name_or_path="",
        num_images=num_images,
    )
    ds = HelmEvalDatasetV3(cfg, rows, processor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4,
                    collate_fn=EvalCollatorQwenVL(processor))

    model.to(device)

    # metrics accumulators
    total = 0
    by_label = defaultdict(lambda: {"n": 0, "ok": 0})
    by_mode = defaultdict(lambda: {"n": 0, "ok": 0})
    update_key_stats = defaultdict(lambda: {"n": 0, "ok": 0})

    outputs: List[Dict[str, Any]] = []

    for step, batch in enumerate(dl):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        pixel_values = batch["pixel_values"].to(device)
        image_grid_thw = batch["image_grid_thw"]
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(device)

        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
        )

        # Decode ONLY the generated continuation (not the echoed prompt).
        # With left-padding, each sample has a different effective prompt length,
        # so we slice per-sample using attention_mask.sum().
        prompt_lens = attention_mask.sum(dim=1).tolist()  # number of non-pad tokens per sample
        texts: List[str] = []
        for bi in range(gen.size(0)):
            pl = int(prompt_lens[bi])
            # guard: if generation returns shorter than prompt (shouldn't happen)
            pl = min(pl, gen.size(1))
            gen_only = gen[bi, pl:]
            txt = processor.tokenizer.decode(gen_only, skip_special_tokens=True)
            texts.append(txt.strip())

        for i in range(len(batch["uids"])):
            uid = batch["uids"][i]
            label = batch["labels"][i]
            mode = batch["modes"][i]
            gt_text = batch["gt_texts"][i]
            row = batch["rows"][i]

            pred_text = texts[i]

            gt_yaml = row.get("gt_yaml", None)
            if not isinstance(gt_yaml, dict):
                gt_yaml = parse_yaml_loose(gt_text)
            pred_yaml = parse_yaml_loose(pred_text)

            if str(mode).upper() == "DETECT":
                ok, info = eval_detect(gt_yaml, pred_yaml)
            else:
                ok, info = eval_update(gt_yaml, pred_yaml)
                # per-key stats
                for k in ["Action_Command", "Working_Memory", "Episodic_Context"]:
                    update_key_stats[k]["n"] += 1
                    update_key_stats[k]["ok"] += int(info.get(f"match_{k}", False))

            total += 1
            by_label[label]["n"] += 1
            by_label[label]["ok"] += int(ok)
            by_mode[str(mode).upper()]["n"] += 1
            by_mode[str(mode).upper()]["ok"] += int(ok)

            outputs.append({
                "uid": uid,
                "label": label,
                "mode": mode,
                "images": row.get("images", {}),
                "user_prompt": row.get("user_prompt", ""),
                "gt_text": gt_text,
                "pred_text": pred_text,
                "gt_yaml": gt_yaml,
                "pred_yaml": pred_yaml,
                "match": ok,
                **info,
                "meta": row.get("meta", {}),
            })

        if (step + 1) % log_every == 0:
            logger.info(f"[EVAL] step={step+1} seen={total}")

    write_jsonl(out_jsonl, outputs)

    # summary
    def acc(d):
        return (d["ok"] / max(d["n"], 1)) * 100.0

    logger.info("========== EVAL SUMMARY ==========")
    logger.info(f"Total: {total}")
    for m, d in sorted(by_mode.items(), key=lambda x: x[0]):
        logger.info(f"[MODE {m}] n={d['n']} acc={acc(d):.2f}%")
    for lab, d in sorted(by_label.items(), key=lambda x: x[0]):
        logger.info(f"[LABEL {lab}] n={d['n']} acc={acc(d):.2f}%")

    if update_key_stats:
        for k, d in update_key_stats.items():
            logger.info(f"[UPDATE key {k}] n={d['n']} acc={(d['ok']/max(d['n'],1))*100.0:.2f}%")

    logger.info(f"Saved predictions -> {out_jsonl}")


def main():
    logging.basicConfig(level=logging.INFO)

    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, required=True, help="eval jsonl (or dir containing jsonl)")
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--adapter", type=str, default=None)

    ap.add_argument("--num_images", type=int, default=1, choices=[1, 2])
    ap.add_argument("--device", type=str, default="cuda:0")

    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=128)

    ap.add_argument("--do_sample", type=bool, default=False)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)

    ap.add_argument("--use_qlora", type=bool, default=True)
    ap.add_argument("--bf16", type=bool, default=True)
    ap.add_argument("--attn_impl", type=str, default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])

    ap.add_argument("--max_samples", type=int, default=0, help="0=all, else random sample")
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--out_jsonl", type=str, required=True)
    ap.add_argument("--log_every", type=int, default=20)

    args = ap.parse_args()

    rows = read_jsonl(args.jsonl)

    if args.max_samples and args.max_samples > 0 and args.max_samples < len(rows):
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        rows = rows[: args.max_samples]
        logger.info(f"Using subset: {len(rows)} samples")

    model, processor = load_model_and_processor(
        base_model=args.base_model,
        adapter_path=args.adapter,
        use_qlora=bool(args.use_qlora),
        bf16=bool(args.bf16),
        attn_impl=args.attn_impl,
    )

    run_eval(
        model=model,
        processor=processor,
        rows=rows,
        num_images=args.num_images,
        device=args.device,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=bool(args.do_sample),
        out_jsonl=args.out_jsonl,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()