#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen2.5‑VL + QLoRA 어댑터 병합 스크립트
- 베이스 Qwen2.5‑VL 모델과 QLoRA 어댑터(PEFT)를 로드해서 완전 가중치로 병합/저장합니다.
- 다양한 transformers 버전에 대응하도록 로더를 여러 단계로 시도합니다.
- processor(tokenizer+image_processor)도 함께 저장합니다.

예)
python train/merge_lora.py \
  --base_model /home/minji/Desktop/data/Qwen2.5-VL-7B-Instruct \
  --adapter /home/minji/Desktop/data/finetuned_model/ghkim/HLP_qwen_2.5_7b_QLoRA_r16_press_the_blue_button_ep60_1114_final/checkpoint-2000 \
  --out     /home/minji/Desktop/data/finetuned_model/ghkim/HLP_qwen_2.5_7b_QLoRA_r16_press_the_blue_button_ep60_1114_final/merged_model_2k \
  --dtype bf16 \
  --load_in_4bit
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional, List

import torch
from peft import PeftModel
from transformers import AutoProcessor, BitsAndBytesConfig

# ------------------------------------------------------------
# Logging
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("merge_lora")


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    return v.lower() in ("1", "true", "t", "yes", "y")


def get_dtype(name: str):
    name = (name or "bf16").lower()
    if name in ["bf16", "bfloat16"]:
        return torch.bfloat16
    if name in ["fp16", "float16", "half"]:
        return torch.float16
    if name in ["fp32", "float32"]:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def try_import_candidates(candidates: List[str]):
    """
    candidates 에 있는 클래스들을 순서대로 import 한 뒤 반환.
    import 실패하면 다음 후보로 넘어감.
    """
    last_err = None
    for dotted in candidates:
        mod, cls = dotted.rsplit(".", 1)
        try:
            module = __import__(mod, fromlist=[cls])
            return getattr(module, cls)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"모델 클래스를 임포트하지 못했습니다. 시도한 후보들: {candidates}\n마지막 에러: {last_err}")


def load_qwen_base(
    base_model: str,
    device: str = "auto",
    dtype_str: str = "bf16",
    load_in_4bit: bool = False,
    attn_implementation: str = "eager",
    trust_remote_code: bool = True,
):
    """
    다양한 transformers 버전에 대응하기 위해 여러 클래스를 시도해 로드.
    우선순위:
      1) transformers.models.qwen2_5_vl.Qwen2_5_VLForConditionalGeneration
      2) AutoModelForVision2Seq
      3) AutoModelForCausalLM (trust_remote_code=True 가정)
    """
    dtype = get_dtype(dtype_str)

    bnb_config = None
    if load_in_4bit:
        logger.info("4-bit quant 로 베이스 모델을 로드합니다.")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tried = []
    # 1) 공식 Qwen2.5-VL 클래스
    try:
        QwenCls = try_import_candidates([
            "transformers.Qwen2_5_VLForConditionalGeneration",
            "transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLForConditionalGeneration",
        ])
        model = QwenCls.from_pretrained(
            base_model,
            torch_dtype=dtype if not load_in_4bit else None,
            quantization_config=bnb_config,
            device_map=device,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
        )
        logger.info("Loaded base with Qwen2_5_VLForConditionalGeneration.")
        return model
    except Exception as e:
        tried.append(("Qwen2_5_VLForConditionalGeneration", repr(e)))

    # 2) AutoModelForVision2Seq (일부 구버전에서 Qwen2.5-VL이 이 경로로 노출됨)
    try:
        from transformers import AutoModelForVision2Seq
        model = AutoModelForVision2Seq.from_pretrained(
            base_model,
            torch_dtype=dtype if not load_in_4bit else None,
            quantization_config=bnb_config,
            device_map=device,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
        )
        logger.info("Loaded base with AutoModelForVision2Seq.")
        return model
    except Exception as e:
        tried.append(("AutoModelForVision2Seq", repr(e)))

    # 3) AutoModelForCausalLM (trust_remote_code=True 로 커스텀 모델 로딩)
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype if not load_in_4bit else None,
            quantization_config=bnb_config,
            device_map=device,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        logger.info("Loaded base with AutoModelForCausalLM (trust_remote_code=True).")
        return model
    except Exception as e:
        tried.append(("AutoModelForCausalLM", repr(e)))
        raise RuntimeError(f"베이스 모델 로딩 실패.\nTried: {tried}")


def maybe_load_processor(base_model: str, adapter_path: Optional[str]) -> AutoProcessor:
    """
    Processor 우선순위:
      1) 어댑터 디렉토리에 processor가 있으면 (preprocessor_config.json) 그걸 사용
      2) 없으면 베이스 모델에서 불러옴
    """
    use_adapter_proc = False
    if adapter_path:
        adapter_proc = Path(adapter_path) / "preprocessor_config.json"
        if adapter_proc.exists():
            use_adapter_proc = True

    src = adapter_path if use_adapter_proc else base_model
    processor = AutoProcessor.from_pretrained(src, trust_remote_code=True)
    logger.info(f"Loaded processor from: {src}")
    return processor


# ------------------------------------------------------------
# Merge
# ------------------------------------------------------------
def merge_and_save(
    base_model: str,
    adapter_path: str,
    out_dir: str,
    dtype: str = "bf16",
    device: str = "auto",
    load_in_4bit: bool = False,
    attn_implementation: str = "eager",
    trust_remote_code: bool = True,
    save_dtype: Optional[str] = None,
    safe_serialization: bool = True,
):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1) Base load
    model = load_qwen_base(
        base_model=base_model,
        device=device,
        dtype_str=dtype,
        load_in_4bit=load_in_4bit,
        attn_implementation=attn_implementation,
        trust_remote_code=trust_remote_code,
    )

    # 2) Adapter attach
    logger.info(f"Loading adapter from: {adapter_path}")
    peft_model = PeftModel.from_pretrained(
        model,
        adapter_path,
        is_trainable=False,
        # 일부 환경에서 모듈 크기 불일치가 있어도 병합을 시도할 수 있게 합니다.
        ignore_mismatched_sizes=True,
    )

    # 3) Merge
    logger.info("Merging adapter into base (merge_and_unload)...")
    merged = peft_model.merge_and_unload()
    logger.info("Merge complete.")

    # 4) (옵션) 저장 dtype 변환
    if save_dtype and save_dtype.lower() != "auto":
        sdtype = get_dtype(save_dtype)
        merged = merged.to(sdtype)
        logger.info(f"Cast merged model to {save_dtype} before saving.")

    # 5) Save model
    logger.info(f"Saving merged model to: {out}")
    merged.save_pretrained(
        out,
        safe_serialization=safe_serialization,  # safetensors
        max_shard_size="4GB",
        from_pt=True,
    )

    # 6) Save processor
    processor = maybe_load_processor(base_model, adapter_path)
    processor.save_pretrained(out)
    logger.info("Processor saved.")

    logger.info("All done.")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser("Qwen2.5‑VL QLoRA merge script")
    p.add_argument("--base_model", type=str, required=True, help="HF 허브 ID 또는 로컬 경로 (예: Qwen/Qwen2.5-VL-7B-Instruct)")
    p.add_argument("--adapter", type=str, required=True, help="PEFT 어댑터(LoRA/QLoRA) 체크포인트 경로 (예: checkpoint-2000)")
    p.add_argument("--out", type=str, required=True, help="병합 결과를 저장할 디렉토리")

    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="베이스 로딩 dtype (4bit면 무시)")
    p.add_argument("--save_dtype", type=str, default="auto", choices=["auto", "bf16", "fp16", "fp32"], help="저장 전 캐스팅 dtype")
    p.add_argument("--device", type=str, default="auto", help='device_map 인자 (예: "auto", "cuda:0", "cpu")')
    p.add_argument("--load_in_4bit", action="store_true", help="베이스를 4-bit 양자화로 로드 (메모리 절약)")
    p.add_argument("--attn_implementation", type=str, default="eager", help='트랜스포머 attention 구현 (예: "eager", "sdpa")')
    p.add_argument("--trust_remote_code", type=str, default="true", help="원격 코드 신뢰 (true/false)")
    p.add_argument("--no_safe_serialization", action="store_true", help="safetensors 대신 pt 형식으로 저장하고 싶으면 지정")

    return p


def main():
    args = build_parser().parse_args()

    merge_and_save(
        base_model=args.base_model,
        adapter_path=args.adapter,
        out_dir=args.out,
        dtype=args.dtype,
        device=args.device,
        load_in_4bit=bool(args.load_in_4bit),
        attn_implementation=args.attn_implementation,
        trust_remote_code=str2bool(args.trust_remote_code),
        save_dtype=args.save_dtype,
        safe_serialization=not args.no_safe_serialization,
    )


if __name__ == "__main__":
    main()