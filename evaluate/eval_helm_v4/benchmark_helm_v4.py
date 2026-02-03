"""
CUDA_VISIBLE_DEVICES=4 python evaluate/eval_helm_v4/benchmark_helm_v4.py \
  --model_name_or_path /ckpt/Qwen2.5-VL-7B-Instruct \
  --ckpt_adapter /backups/ghkim/HeLM_v4/HLP_HeLM_v4_qwen_7b_all_extended_0122/checkpoint-3500 \
  --eval_jsonl /data/ghkim/helm_data/helm_v4_task_10_extended_re/merged/all_val.jsonl \
  --device cuda \
  --max_samples 50 \
  --warmup 10 \
  --max_new_tokens_detect 32 \
  --load_in_4bit \
  --max_new_tokens_update 128
"""

import argparse
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image

# Qwen2.5-VL (your training likely used this)
from transformers import AutoProcessor
try:
    from transformers import Qwen2_5_VLForConditionalGeneration
except Exception:
    # if your env exposes it differently, adjust
    from transformers import AutoModelForCausalLM as Qwen2_5_VLForConditionalGeneration

# PEFT (for LoRA/QLoRA adapters)
try:
    from peft import PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False

import time
import torch

class PerformanceMonitor:
    def __init__(self, device: str = "cuda"):
        self.device = device
        self.latency = 0.0
        self.max_mem_gb = 0.0

    def __enter__(self):
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated()
            self.max_mem_gb = peak / (1024 ** 3)
        self.latency = time.perf_counter() - self._t0
        return False


@dataclass
class BenchResult:
    mode: str
    n: int
    mean_latency_s: float
    p50_latency_s: float
    p90_latency_s: float
    p99_latency_s: float
    peak_vram_gb: float


def load_jsonl(path: str, max_samples: int, mode: Optional[str]) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if mode is not None and r.get("mode") != mode:
                continue
            rows.append(r)
            if max_samples > 0 and len(rows) >= max_samples:
                break
    return rows


def get_table_image_path(r: Dict[str, Any]) -> str:
    images = r.get("images")
    if isinstance(images, dict):
        if "table" in images:
            return images["table"]
        # fallback: any first value
        for v in images.values():
            if isinstance(v, str):
                return v
        raise KeyError("Row images dict has no usable string path")

    if isinstance(images, list):
        # common cases:
        # 1) list of dicts like [{"table": "/path"}, ...]
        for item in images:
            if isinstance(item, dict) and "table" in item and isinstance(item["table"], str):
                return item["table"]
        # 2) list of strings (assume first is the table image)
        for item in images:
            if isinstance(item, str):
                return item
        raise TypeError("Row images list has no usable string path")

    raise TypeError(f"Unsupported images type: {type(images)}")


def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _bytes_to_gb(x: int) -> float:
    return float(x) / (1024.0 ** 3)


def _percentile(sorted_vals: List[float], q: float) -> float:
    # q in [0,1]
    if not sorted_vals:
        return 0.0
    idx = int(round((len(sorted_vals) - 1) * q))
    idx = max(0, min(len(sorted_vals) - 1, idx))
    return sorted_vals[idx]


def build_model_and_processor(
    model_name_or_path: str,
    ckpt_adapter: str,
    device: str,
    bf16: bool,
    use_flash_attn: bool,
    load_in_4bit: bool,
) -> Tuple[Any, Any]:
    """
    - If ckpt_adapter is empty: loads base model
    - If ckpt_adapter points to adapter: loads base then attaches adapter (PEFT)
    """
    torch_dtype = torch.bfloat16 if bf16 else torch.float16

    # DDP/QLoRA safe device selection (single process benchmark still OK)
    if device.startswith("cuda"):
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        device_map = {"": local_rank}
    else:
        device_map = None

    quant_cfg = None
    if load_in_4bit:
        # optional, only if you want 4bit base load in benchmark
        from transformers import BitsAndBytesConfig
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    attn_impl = "flash_attention_2" if use_flash_attn else "sdpa"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        quantization_config=quant_cfg,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )

    # Attach adapter if provided
    if ckpt_adapter:
        if not _HAS_PEFT:
            raise RuntimeError("peft is not installed but ckpt_adapter was provided.")
        model = PeftModel.from_pretrained(model, ckpt_adapter)
        # If you want faster inference:
        # model = model.merge_and_unload()  # optional (only if you won't train further)

    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    model.eval()
    return model, processor


@torch.inference_mode()
def run_one_generate(
    model,
    processor,
    prompt: str,
    image_path: str,
    device: str,
    max_new_tokens: int,
    uid: str = "",
) -> str:
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Failed to open image: {image_path} (uid={uid})") from e

    # Qwen2.5-VL expects image tokens in the text; use chat template with an explicit image placeholder.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=text,
        images=img,
        return_tensors="pt",
    )

    # move tensors
    for k, v in inputs.items():
        if hasattr(v, "to"):
            inputs[k] = v.to(device)

    _cuda_sync()
    t0 = time.perf_counter()
    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )
    _cuda_sync()
    t1 = time.perf_counter()

    out_text = processor.batch_decode(out_ids, skip_special_tokens=True)[0]
    # Heuristic: keep only the tail after the prompt text if it appears.
    if isinstance(out_text, str) and isinstance(text, str) and text.strip() and out_text.startswith(text):
        out_text = out_text[len(text):].lstrip()

    print("[OUTPUT]: ", out_text)
    return out_text, (t1 - t0)

@torch.inference_mode()
def run_one_generate_e2e(model, processor, prompt: str, image_path: str, device: str, max_new_tokens: int, uid: str = ""):
    img = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=text, images=img, return_tensors="pt")
    for k, v in inputs.items():
        if hasattr(v, "to"):
            inputs[k] = v.to(device)

    gen_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        use_cache=True,
    )

    in_len = inputs["input_ids"].shape[1]
    gen_trim = gen_ids[:, in_len:]
    out_text = processor.batch_decode(gen_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

    # 여기서 print까지 포함시킬 거면 여기서 출력 (실환경 반영)
    print(f"\n[GEN uid={uid}]")
    print(out_text)
    print("---------------------------------------------------")

    return out_text, int(gen_trim.shape[1])

def benchmark_mode(
    rows: List[Dict[str, Any]],
    model,
    processor,
    device: str,
    max_new_tokens: int,
    warmup: int,
) -> BenchResult:
    latencies: List[float] = []
    vrams: List[float] = []

    # warmup (측정 제외)
    for i in range(min(warmup, len(rows))):
        r = rows[i]
        prompt = r["user_prompt"]
        img_path = get_table_image_path(r)
        _ = run_one_generate_e2e(model, processor, prompt, img_path, device, max_new_tokens, uid=r.get("uid", ""))

    # measure (E2E 포함: apply_chat_template + processor + to(device) + generate + decode + print)
    for r in rows:
        prompt = r["user_prompt"]
        img_path = get_table_image_path(r)

        with PerformanceMonitor(device=device) as pm:
            _out_text, _gen_tokens = run_one_generate_e2e(
                model, processor, prompt, img_path, device, max_new_tokens, uid=r.get("uid", "")
            )

        latencies.append(pm.latency)
        vrams.append(pm.max_mem_gb)

    lat_sorted = sorted(latencies)
    mean_lat = sum(latencies) / max(1, len(latencies))

    peak_vram = max(vrams) if vrams else 0.0

    return BenchResult(
        mode=str(rows[0].get("mode", "UNKNOWN")) if rows else "UNKNOWN",
        n=len(rows),
        mean_latency_s=mean_lat,
        p50_latency_s=_percentile(lat_sorted, 0.50),
        p90_latency_s=_percentile(lat_sorted, 0.90),
        p99_latency_s=_percentile(lat_sorted, 0.99),
        peak_vram_gb=peak_vram,
    )


def print_result(r: BenchResult):
    print(f"\n=== Benchmark: {r.mode} ===")
    print(f"N samples: {r.n}")
    print(f"Latency mean: {r.mean_latency_s:.4f} s")
    print(f"Latency p50 : {r.p50_latency_s:.4f} s")
    print(f"Latency p90 : {r.p90_latency_s:.4f} s")
    print(f"Latency p99 : {r.p99_latency_s:.4f} s")
    print(f"Peak VRAM   : {r.peak_vram_gb:.2f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name_or_path", type=str, required=True)
    ap.add_argument("--ckpt_adapter", type=str, default="")
    ap.add_argument("--eval_jsonl", type=str, required=True)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--use_flash_attn", action="store_true")
    ap.add_argument("--load_in_4bit", action="store_true")

    ap.add_argument("--max_samples", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=10)

    ap.add_argument("--max_new_tokens_detect", type=int, default=64)
    ap.add_argument("--max_new_tokens_update", type=int, default=128)

    args = ap.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")

    model, processor = build_model_and_processor(
        model_name_or_path=args.model_name_or_path,
        ckpt_adapter=args.ckpt_adapter,
        device=device,
        bf16=bool(args.bf16),
        use_flash_attn=bool(args.use_flash_attn),
        load_in_4bit=bool(args.load_in_4bit),
    )

    # DETECT
    detect_rows = load_jsonl(args.eval_jsonl, args.max_samples, mode="DETECT")
    if detect_rows:
        r_det = benchmark_mode(
            detect_rows, model, processor, device,
            max_new_tokens=args.max_new_tokens_detect,
            warmup=args.warmup,
        )
        print_result(r_det)
    else:
        print("\n(No DETECT rows found)")

    # UPDATE
    update_rows = load_jsonl(args.eval_jsonl, args.max_samples, mode="UPDATE")
    if update_rows:
        r_upd = benchmark_mode(
            update_rows, model, processor, device,
            max_new_tokens=args.max_new_tokens_update,
            warmup=args.warmup,
        )
        print_result(r_upd)
    else:
        print("\n(No UPDATE rows found)")


if __name__ == "__main__":
    main()