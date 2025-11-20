#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen2.5-VL 베이스 + (Q)LoRA 어댑터를 로드 → 메모리상 병합 → 로컬에 저장하는 최소 스크립트
- eval_HLP.py와 동일한 방식(Qwen2_5_VLForConditionalGeneration, AutoProcessor, PeftModel, merge_and_unload)으로 동작
예)
python train/merge_lora.py \
  --base_model_path /ckpt/Qwen2.5-VL-7B-Instruct \
  --adapter_path    /result/ghkim/HLP_qwen_2.5_7b_QLoRA_r16_press_the_blue_button_ep60_1114_final/checkpoint-2000 \
  --out_dir         /result/ghkim/HLP_qwen_2.5_7b_QLoRA_r16_press_the_blue_button_ep60_1114_final/merged_model_2k \
#  --load_in_4bit
# QLoRA로 학습했다면 4bit 로드:
#  --load_in_4bit 를 추가
"""

import argparse
import os
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_path", type=str, required=True,
                    help="Qwen2.5-VL-7B-Instruct 로컬 경로(또는 허브 ID)")
    ap.add_argument("--adapter_path", type=str, required=True,
                    help="LoRA/QLoRA 어댑터 디렉토리 (checkpoint-XXXX)")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="병합 결과 저장 폴더")
    ap.add_argument("--device", type=str, default="auto",
                    help='device_map (기본: "auto")')
    ap.add_argument("--load_in_4bit", action="store_true",
                    help="QLoRA로 학습했다면 베이스를 4bit로 로드")
    ap.add_argument("--save_dtype", type=str, default="bf16",
                    choices=["bf16", "fp16", "fp32"],
                    help="병합 결과를 저장할 dtype (기본: bf16)")
    args = ap.parse_args()

    # Determine save dtype tensor
    if args.save_dtype == "bf16":
        _save_dtype = torch.bfloat16
    elif args.save_dtype == "fp16":
        _save_dtype = torch.float16
    else:
        _save_dtype = torch.float32

    bnb_config = None
    if args.load_in_4bit:
        print("[INFO] Loading base in 4-bit (for QLoRA).")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    print(f"[INFO] Loading base model from: {args.base_model_path}")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model_path,
        quantization_config=bnb_config,
        device_map=args.device,
        torch_dtype=None if args.load_in_4bit else _save_dtype,
        attn_implementation="eager",
    )

    # --- Processor & Tokenizer loading strategy ---
    # 1) Always load the full Processor from the BASE model path (it has image processor config).
    # 2) If the adapter dir contains tokenizer files, load tokenizer from adapter (to match training),
    #    otherwise fall back to base tokenizer.
    # Prefer FAST tokenizer/processor; fall back to slow only if fast not available.
    try:
        processor = AutoProcessor.from_pretrained(args.base_model_path, use_fast=True, trust_remote_code=True)
    except Exception as e:
        print(f"[WARN] Fast processor load failed ({e}); falling back to slow processor.")
        processor = AutoProcessor.from_pretrained(args.base_model_path, use_fast=False, trust_remote_code=True)

    def _has_tokenizer_files(p):
        return any(os.path.exists(os.path.join(p, n)) for n in [
            "tokenizer.json", "tokenizer.model", "spiece.model", "vocab.json", "tokenizer_config.json"
        ])

    tok_src = args.adapter_path if _has_tokenizer_files(args.adapter_path) else args.base_model_path
    print(f"[INFO] Loading tokenizer from: {tok_src}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True, trust_remote_code=True)
    except Exception as e:
        print(f"[WARN] Fast tokenizer load failed ({e}); falling back to slow tokenizer.")
        tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=False, trust_remote_code=True)

    # Robust vocab-size detection for both fast/slow tokenizers
    target_vocab_size = getattr(tokenizer, "vocab_size", None)
    if target_vocab_size is None:
        try:
            target_vocab_size = tokenizer.get_vocab_size()
        except Exception:
            try:
                target_vocab_size = len(tokenizer)
            except Exception:
                target_vocab_size = None

    if target_vocab_size is not None and getattr(base_model.config, "vocab_size", None) != target_vocab_size:
        print(f"[INFO] Resizing token embeddings: {getattr(base_model.config, 'vocab_size', None)} -> {target_vocab_size}")
        base_model.resize_token_embeddings(target_vocab_size)
        base_model.config.vocab_size = target_vocab_size

    print(f"[INFO] Loading adapter from: {args.adapter_path}")
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_path,
        ignore_mismatched_sizes=True
    )

    print("[INFO] Merging adapter into base (merge_and_unload)...")
    model = model.merge_and_unload()
    model.eval()
    # === Cast merged model to target 16-bit/32-bit dtype before saving ===
    # Note: If --load_in_4bit was used, this will dequantize to the chosen dtype for saving.
    print(f"[INFO] Casting merged model to {args.save_dtype} before saving...")
    model = model.to(_save_dtype)

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[INFO] Saving merged model to: {args.out_dir}")
    model.save_pretrained(args.out_dir, safe_serialization=True, max_shard_size="4GB")
    processor.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)  # ensure merged dir uses the training tokenizer

    print("[DONE] Merged model + processor saved.")

if __name__ == "__main__":
    main()