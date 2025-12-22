# evaluate/eval_helm_hlp.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
)
from peft import PeftModel
try:
    from tqdm import tqdm  # type: ignore
except Exception:
    tqdm = None

try:
    import yaml
except Exception:
    yaml = None

"""
CUDA_VISIBLE_DEVICES=5 python evaluate/eval_helm_hlp.py \
  --base_model_path /ckpt/Qwen2.5-VL-7B-Instruct \
  --adapter_path /result/ghkim/HLP_HeLM_press_1_2_3/checkpoint-400 \
  --dataset_file /data/ghkim/helm_data/press_the_button_N_times_ep60/jsonl/merged/press_1_2_3/all_val.jsonl \
  --output_file /result/ghkim/HLP_HeLM_press_1_2_3/eval/all_val_pred_4k.jsonl \
  --attn_impl sdpa \
  --max_new_tokens 128 \
  --log_every 20 \
  --show_tqdm 1
"""


HLP_HEADER = (
    "Role: High-Level Planner (HLP).\n"
    "Given the two images and Previous_Memory, update the memory and choose the next atomic command.\n"
    "- Only advance Progress when the event has occurred in the current frame.\n"
    "- World_State should be concise and persistent (use None if no state).\n"
    "- Command should be either the task command or \"done\" if finished.\n"
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
    add_hlp_header: bool = True,
    drop_frame_line: bool = True,
    drop_images_line: bool = True,
    drop_return_yaml_line: bool = True,
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
        return (HLP_HEADER + "\n\n" + raw_user_text.strip()).strip()
    return raw_user_text.strip()


def load_row_user_assistant(row: Dict[str, Any]) -> Tuple[str, str]:
    conv = row.get("conversations", [])
    if not isinstance(conv, list):
        raise ValueError("row['conversations'] must be a list")

    user_text = ""
    gt_text = ""
    for m in conv:
        if not isinstance(m, dict):
            continue
        if m.get("from") == "user":
            user_text = str(m.get("value", ""))
        elif m.get("from") == "assistant":
            gt_text = str(m.get("value", ""))

    if not user_text or not gt_text:
        raise ValueError("Missing user/assistant in conversations")
    return user_text, gt_text


def load_images(row: Dict[str, Any]) -> list[Image.Image]:
    imgs = row.get("images", {})
    if not isinstance(imgs, dict):
        raise ValueError("row['images'] must be dict")

    # new schema: table/wrist
    p0 = imgs.get("table")
    p1 = imgs.get("wrist")
    if not p0 or not p1:
        raise ValueError(f"images must include table/wrist. got keys={list(imgs.keys())}")

    return [Image.open(p0).convert("RGB"), Image.open(p1).convert("RGB")]


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
    images: list[Image.Image],
    max_new_tokens: int = 128,
) -> str:
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

    prompt_string = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = processor(
        text=prompt_string,
        images=images,
        return_tensors="pt",
        padding=True,
    )

    # device_map="auto"일 때는 모델이 알아서 분산되지만, 입력 텐서는 한 디바이스로 보내야 함.
    # 보통 첫 파라미터 디바이스로 보냄.
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )

    # 프롬프트 부분 잘라내기
    in_len = inputs["input_ids"].shape[1]
    gen_trim = gen_ids[:, in_len:]

    out_text = processor.batch_decode(
        gen_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return out_text.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_path", type=str, required=True)
    ap.add_argument("--adapter_path", type=str, required=True)
    ap.add_argument("--dataset_file", type=str, required=True)   # all_val.jsonl 같은 파일
    ap.add_argument("--output_file", type=str, required=True)

    ap.add_argument("--is_qlora", type=int, default=1)
    ap.add_argument("--attn_impl", type=str, default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="0이면 전체, >0이면 앞에서 N개만 평가")
    ap.add_argument("--log_every", type=int, default=50, help="print progress every N samples (0 disables)")
    ap.add_argument("--show_tqdm", type=int, default=1, help="use tqdm progress bar if available")

    # prompt normalize options (train과 맞추는 게 중요)
    ap.add_argument("--add_hlp_header", type=int, default=1)
    ap.add_argument("--drop_frame_line", type=int, default=1)
    ap.add_argument("--drop_images_line", type=int, default=1)
    ap.add_argument("--drop_return_yaml_line", type=int, default=1)

    args = ap.parse_args()

    dataset_path = Path(args.dataset_file)
    out_path = Path(args.output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # processor
    processor = AutoProcessor.from_pretrained(args.base_model_path, trust_remote_code=True, use_fast=True)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # base model (QLoRA면 4bit로)
    bnb_config = None
    if args.is_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model_path,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_impl,
        trust_remote_code=True,
    )

    # adapter 로드
    model = PeftModel.from_pretrained(base, args.adapter_path)
    model.eval()

    n = 0
    n_parsed = 0
    n_match_cmd = 0
    n_match_progress = 0

    with dataset_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        it = f_in
        if args.show_tqdm and tqdm is not None:
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
                raw_user,
                add_hlp_header=bool(args.add_hlp_header),
                drop_frame_line=bool(args.drop_frame_line),
                drop_images_line=bool(args.drop_images_line),
                drop_return_yaml_line=bool(args.drop_return_yaml_line),
            )

            images = load_images(row)
            pred = generate_yaml(
                model=model,
                processor=processor,
                user_text=user,
                images=images,
                max_new_tokens=args.max_new_tokens,
            )

            gt_obj = parse_yaml_maybe(gt)
            pred_obj = parse_yaml_maybe(pred)

            match_cmd = None
            match_progress = None
            if gt_obj and pred_obj:
                n_parsed += 1
                match_cmd = (str(pred_obj.get("Command")) == str(gt_obj.get("Command")))
                match_progress = (str(pred_obj.get("Progress")) == str(gt_obj.get("Progress")))
                n_match_cmd += int(match_cmd)
                n_match_progress += int(match_progress)

            out_rec = {
                "uid": uid,
                "task_id": row.get("task_id"),
                "images": row.get("images"),
                "user_prompt": user,          # 실제 모델에 넣은 prompt (디버깅용)
                "gt_text": gt,
                "pred_text": pred,
                "gt_yaml": gt_obj,
                "pred_yaml": pred_obj,
                "match_cmd": match_cmd,
                "match_progress": match_progress,
            }
            f_out.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            n += 1

            # Periodic progress logging
            if args.log_every and (n % args.log_every == 0):
                if n_parsed > 0:
                    cmd_acc = n_match_cmd / n_parsed
                    prog_acc = n_match_progress / n_parsed
                    print(f"[progress] n={n} parsed={n_parsed} cmd_acc={cmd_acc:.3f} progress_acc={prog_acc:.3f}", flush=True)
                else:
                    print(f"[progress] n={n} parsed={n_parsed}", flush=True)

    # 간단 요약 출력
    print(f"[DONE] wrote: {out_path}")
    print(f"num_samples={n}")
    if n_parsed > 0:
        print(f"parsed_yaml={n_parsed}/{n}")
        print(f"cmd_acc={n_match_cmd/n_parsed:.3f}  progress_acc={n_match_progress/n_parsed:.3f}")
    else:
        print("parsed_yaml=0 (pyyaml not installed or parsing failed)")


if __name__ == "__main__":
    main()