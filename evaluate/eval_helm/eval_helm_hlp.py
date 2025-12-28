# evaluate/eval_helm_hlp.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from peft import PeftModel

try:
    import yaml
except Exception:
    yaml = None

try:
    from tqdm import tqdm
except Exception:
    tqdm = None
"""
python evaluate/eval_helm_hlp.py \
  --eval_jsonl /path/all_val.jsonl \
  --out_jsonl  /path/preds_val.jsonl \
  --base_model /ckpt/Qwen2.5-VL-7B-Instruct \
  --adapter    /result/.../checkpoint-XXXX \
  --num_image 2
  
======================================================

CUDA_VISIBLE_DEVICES=7 python evaluate/eval_helm/eval_helm_hlp.py \
  --eval_jsonl /data/ghkim/helm_data/wipe_the_window/jsonl/merged/wipe_only_tableview/all_val.jsonl \
  --out_jsonl  /result/ghkim/HLP_HeLM_wipe_only_tableview_1226/checkpoint-800/HeLM/val_800.jsonl \
  --base_model /ckpt/Qwen2.5-VL-7B-Instruct \
  --adapter    /result/ghkim/HLP_HeLM_wipe_only_tableview_1226/checkpoint-800 \
  --num_image 1
"""

HLP_HEADER_1 = (
    "Role: High-Level Planner (HLP).\n"
    "Given the table view image and Previous_Memory, update the memory and choose the next atomic command.\n"
    "- Only advance Progress when the event has occurred in the current frame.\n"
    "- World_State should be concise and persistent (use None if no state).\n"
    '- Command should be either the task command or "done" if finished.\n'
    "Return YAML with keys Progress, World_State, Command.\n"
)

HLP_HEADER_2 = (
    "Role: High-Level Planner (HLP).\n"
    "Given the two images and Previous_Memory, update the memory and choose the next atomic command.\n"
    "- Only advance Progress when the event has occurred in the current frame.\n"
    "- World_State should be concise and persistent (use None if no state).\n"
    '- Command should be either the task command or "done" if finished.\n'
    "Return YAML with keys Progress, World_State, Command.\n"
)


def _drop_lines_with_prefix(text: str, prefixes: Tuple[str, ...]) -> str:
    lines = text.splitlines()
    kept = []
    for ln in lines:
        s = ln.strip()
        if any(s.startswith(p) for p in prefixes):
            continue
        kept.append(ln)
    return "\n".join(kept).strip()


def build_user_text(
    raw_user_text: str,
    num_image: int,
    add_hlp_header: bool,
    drop_frame_line: bool,
    drop_images_line: bool,
    drop_return_yaml_line: bool,
) -> str:
    prefixes = []
    if drop_frame_line:
        prefixes.append("Frame:")
    if drop_images_line:
        prefixes.append("Images:")
    if drop_return_yaml_line:
        prefixes.append("Return YAML")

    if prefixes:
        raw_user_text = _drop_lines_with_prefix(raw_user_text, tuple(prefixes))

    if add_hlp_header:
        header = HLP_HEADER_1 if int(num_image) == 1 else HLP_HEADER_2
        return (header + "\n\n" + raw_user_text.strip()).strip()
    return raw_user_text.strip()


def load_row_user_assistant(row: Dict[str, Any]) -> Tuple[str, str]:
    conv = row.get("conversations", [])
    if not isinstance(conv, list):
        raise ValueError("row['conversations'] must be a list")

    user_text, asst_text = "", ""
    for m in conv:
        if not isinstance(m, dict):
            continue
        if m.get("from") == "user":
            user_text = str(m.get("value", ""))
        elif m.get("from") == "assistant":
            asst_text = str(m.get("value", ""))
    if not user_text or not asst_text:
        raise ValueError("Missing user/assistant in conversations")
    return user_text, asst_text


def load_images(row: Dict[str, Any], num_image: int) -> List[Image.Image]:
    imgs = row.get("images", {})
    if not isinstance(imgs, dict):
        raise ValueError("row['images'] must be dict")

    p_table = imgs.get("table", None)
    p_wrist = imgs.get("wrist", None)

    if not p_table:
        raise ValueError(f"images must include 'table'. got keys={list(imgs.keys())}")

    table = Image.open(p_table).convert("RGB")

    if int(num_image) == 1:
        return [table]

    if not p_wrist:
        raise ValueError(f"num_image=2 requires 'wrist'. got keys={list(imgs.keys())}")

    wrist = Image.open(p_wrist).convert("RGB")
    return [table, wrist]


def parse_yaml_maybe(text: str) -> Optional[Dict[str, Any]]:
    if yaml is None:
        return None
    try:
        obj = yaml.safe_load(text)
        if isinstance(obj, dict):
            return obj
        return None
    except Exception:
        return None


@torch.no_grad()
def generate_yaml(
    model,
    processor,
    user_text: str,
    images: List[Image.Image],
    num_image: int,
    max_new_tokens: int,
) -> str:
    # messages: image placeholder count must match num_image
    if int(num_image) == 1:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": user_text},
                ],
            }
        ]

    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=[prompt],
        images=images,         # 반드시 list length == num_image
        padding=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    # trim prompt
    in_ids = inputs["input_ids"]
    gen_trim = generated_ids[:, in_ids.shape[1]:]
    out_text = processor.batch_decode(
        gen_trim,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return out_text.strip()


def load_model_and_processor(
    base_model: str,
    adapter: Optional[str],
    device: str,
    load_in_4bit: bool,
    attn_impl: str,
):
    bnb_config = None
    if load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
        device_map=device if (not load_in_4bit) else "auto",
    )

    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True, use_fast=True)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return model, processor


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--eval_jsonl", type=str, required=True)
    p.add_argument("--out_jsonl", type=str, required=True)

    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--adapter", type=str, default=None)

    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--load_in_4bit", type=bool, default=True)
    p.add_argument("--attn_impl", type=str, default="sdpa")

    # ★ 핵심: 1장/2장 스위치
    p.add_argument("--num_image", type=int, default=2, choices=[1, 2])

    p.add_argument("--max_new_tokens", type=int, default=128)

    p.add_argument("--add_hlp_header", type=bool, default=True)
    p.add_argument("--drop_frame_line", type=bool, default=True)
    p.add_argument("--drop_images_line", type=bool, default=True)
    p.add_argument("--drop_return_yaml_line", type=bool, default=True)

    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--show_tqdm", type=bool, default=True)
    p.add_argument("--print_every", type=int, default=200)

    args = p.parse_args()

    model, processor = load_model_and_processor(
        base_model=args.base_model,
        adapter=args.adapter,
        device=args.device,
        load_in_4bit=bool(args.load_in_4bit),
        attn_impl=args.attn_impl,
    )

    in_path = Path(args.eval_jsonl)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    n_parsed = 0
    n_match_cmd = 0
    n_match_progress = 0

    with in_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        it = f_in
        if bool(args.show_tqdm) and tqdm is not None:
            it = tqdm(f_in, desc="Evaluating", unit="lines")

        for line in it:
            if args.limit and n >= args.limit:
                break
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            uid = row.get("uid", f"row{n}")

            raw_user, gt = load_row_user_assistant(row)
            user = build_user_text(
                raw_user_text=raw_user,
                num_image=args.num_image,
                add_hlp_header=bool(args.add_hlp_header),
                drop_frame_line=bool(args.drop_frame_line),
                drop_images_line=bool(args.drop_images_line),
                drop_return_yaml_line=bool(args.drop_return_yaml_line),
            )

            images = load_images(row, args.num_image)

            pred = generate_yaml(
                model=model,
                processor=processor,
                user_text=user,
                images=images,
                num_image=args.num_image,
                max_new_tokens=args.max_new_tokens,
            )

            out = {
                "uid": uid,
                "user_prompt": user,
                "gt_text": gt,
                "pred_text": pred,
            }

            gt_yaml = parse_yaml_maybe(gt)
            pred_yaml = parse_yaml_maybe(pred)
            if gt_yaml is not None and pred_yaml is not None:
                n_parsed += 1
                out["gt_yaml"] = gt_yaml
                out["pred_yaml"] = pred_yaml

                # exact-match metrics for two keys
                gt_cmd = str(gt_yaml.get("Command", "")).strip()
                pr_cmd = str(pred_yaml.get("Command", "")).strip()
                gt_prog = str(gt_yaml.get("Progress", "")).strip()
                pr_prog = str(pred_yaml.get("Progress", "")).strip()

                match_cmd = (gt_cmd == pr_cmd)
                match_progress = (gt_prog == pr_prog)

                out["match_cmd"] = match_cmd
                out["match_progress"] = match_progress

                n_match_cmd += int(match_cmd)
                n_match_progress += int(match_progress)

            f_out.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1

            if args.print_every and (n % args.print_every == 0):
                if bool(args.show_tqdm) and tqdm is not None:
                    pass
                else:
                    print(f"[progress] n={n} parsed={n_parsed}", flush=True)

    print(f"[DONE] wrote: {out_path}")
    print(f"num_samples={n}")
    if n_parsed > 0:
        print(f"parsed_yaml={n_parsed}/{n}")
        print(f"cmd_acc={n_match_cmd/n_parsed:.3f}  progress_acc={n_match_progress/n_parsed:.3f}")
    else:
        print("parsed_yaml=0 (pyyaml not installed or parsing failed)")


if __name__ == "__main__":
    main()