# eval_helm_v2.py
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel

import yaml

import helm_dataset as hd  # <-- train_helm.py에서 쓰는 동일 파일

"""
CUDA_VISIBLE_DEVICES=6 python evaluate/eval_helm_v2/eval_helm_hlp.py \
  --jsonl /data/ghkim/helm_data/wipe_the_window/jsonl_v2/merged/wipe_the_window/all_val.jsonl \
  --out_jsonl /result/ghkim/HLP_HeLM_v2_wipe_the_window_1230/eval/eval_step-400.jsonl \
  --base_model /ckpt/Qwen2.5-VL-7B-Instruct \
  --adapter /result/ghkim/HLP_HeLM_v2_wipe_the_window_1230/checkpoint-400 \
  --use_qlora 1 \
  --num_image 1 --camera table \
  --attn_impl sdpa \
  --max_new_tokens 64
"""
# -----------------------------
# utils
# -----------------------------
def read_jsonl(path_or_dir: str) -> List[Dict[str, Any]]:
    p = Path(path_or_dir)
    files: List[Path] = []
    if p.is_file():
        files = [p]
    elif p.is_dir():
        files = sorted(p.rglob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No jsonl under: {p}")
    else:
        raise FileNotFoundError(p)

    rows: List[Dict[str, Any]] = []
    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception as e:
                    raise RuntimeError(f"JSON parse error: {fp} line {ln}: {e}")
    return rows


def parse_yaml(text: str) -> Optional[Dict[str, Any]]:
    if not text or not isinstance(text, str):
        return None
    s = text.strip()
    # strip ```yaml fences
    s = re.sub(r"^```yaml\s*", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"^```\s*", "", s).strip()
    s = re.sub(r"\s*```$", "", s).strip()
    try:
        obj = yaml.safe_load(s)
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def norm_bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        t = v.strip().lower()
        if t in ("true", "yes", "1"):
            return True
        if t in ("false", "no", "0"):
            return False
    if isinstance(v, (int, float)):
        return bool(v)
    return None


def norm_str(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    return str(v).strip()


# -----------------------------
# model load (mirrors train_helm build_model, but for eval)
# -----------------------------
def load_eval_model(
    base_model: str,
    adapter: Optional[str],
    use_qlora: bool,
    device_map: str,
    attn_impl: str,
):
    bnb = None
    if use_qlora:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model,
        quantization_config=bnb,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )

    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
        # eval 안정성: merge 추천(원하면 끌 수 있음)
        model = model.merge_and_unload()

    model.eval()

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True, use_fast=True)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    return model, processor


# -----------------------------
# prompt + image loading (EXACTLY AS TRAIN V2)
# -----------------------------
def build_prompt_and_target(row: Dict[str, Any]) -> Tuple[str, str]:
    # adds DETECT_HEADER / UPDATE_HEADER exactly the same way as training
    return hd._build_v2_prompt_and_target(row)


def load_images_v2(
    row: Dict[str, Any],
    num_image: int,
    camera: str,
    frames_root: Optional[str],
    require_images_for_update: bool,
) -> List[Image.Image]:
    return hd._load_images_v2(
        row=row,
        num_image=int(num_image),
        camera=str(camera),
        frames_root=Path(frames_root) if frames_root else None,
        require_images_for_update=bool(require_images_for_update),
    )


# -----------------------------
# generation (Qwen2.5-VL)
# -----------------------------
@torch.inference_mode()
def run_one(
    model,
    processor,
    user_text: str,
    images: List[Image.Image],
    max_new_tokens: int,
) -> str:
    # train dataset uses one image token + text
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        }
    ]

    prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,  # inference
    )

    inputs = processor(
        text=[prompt],
        images=images,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    out_ids = model.generate(
        **inputs,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
    )

    in_len = inputs["input_ids"].shape[1]
    gen = out_ids[:, in_len:]
    txt = processor.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return txt.strip()


# -----------------------------
# scoring (v2)
# -----------------------------
def score_detect(gt: Dict[str, Any], pred: Dict[str, Any]) -> Dict[str, Any]:
    gt_ev = norm_bool(gt.get("Event_Detected"))
    pr_ev = norm_bool(pred.get("Event_Detected"))
    return {"match_event": (gt_ev is not None and pr_ev is not None and gt_ev == pr_ev)}


def score_update(gt: Dict[str, Any], pred: Dict[str, Any]) -> Dict[str, Any]:
    # UPDATE keys are fixed by UPDATE_HEADER
    gt_wm = norm_str(gt.get("Working_Memory"))
    pr_wm = norm_str(pred.get("Working_Memory"))
    gt_ec = norm_str(gt.get("Episodic_Context"))
    pr_ec = norm_str(pred.get("Episodic_Context"))
    gt_ac = norm_str(gt.get("Action_Command"))
    pr_ac = norm_str(pred.get("Action_Command"))

    return {
        "match_working_memory": (gt_wm == pr_wm),
        "match_episodic_context": (gt_ec == pr_ec),
        "match_action_command": (gt_ac == pr_ac),
        "match_all": (gt_wm == pr_wm and gt_ec == pr_ec and gt_ac == pr_ac),
    }


# -----------------------------
# main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--jsonl", type=str, required=True, help="v2 jsonl file/dir (detect/update/merged)")
    ap.add_argument("--out_jsonl", type=str, required=True, help="where to save per-row eval outputs")

    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--use_qlora", type=int, default=1)

    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--attn_impl", type=str, default="sdpa", choices=["flash_attention_2", "sdpa", "eager"])

    ap.add_argument("--num_image", type=int, default=1)
    ap.add_argument("--camera", type=str, default="table")
    ap.add_argument("--frames_root", type=str, default=None)
    ap.add_argument("--require_images_for_update", action="store_true")

    ap.add_argument("--mode", type=str, default="both", choices=["both", "detect", "update"])
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=0)

    args = ap.parse_args()

    rows = read_jsonl(args.jsonl)
    if args.limit and int(args.limit) > 0:
        rows = rows[: int(args.limit)]

    model, processor = load_eval_model(
        base_model=args.base_model,
        adapter=args.adapter,
        use_qlora=bool(args.use_qlora),
        device_map=args.device_map,
        attn_impl=args.attn_impl,
    )

    n_det = n_up = 0
    ok_det = 0
    ok_up_all = 0
    ok_up_ac = 0
    parse_fail = 0

    outp = Path(args.out_jsonl)
    outp.parent.mkdir(parents=True, exist_ok=True)

    with outp.open("w", encoding="utf-8") as fw:
        for i, row in enumerate(rows):
            mode = str(row.get("mode", "")).strip().lower()
            if mode not in ("detect", "update"):
                continue
            if args.mode != "both" and mode != args.mode:
                continue

            user_text, tgt_text = build_prompt_and_target(row)
            images = load_images_v2(
                row=row,
                num_image=int(args.num_image),
                camera=str(args.camera),
                frames_root=args.frames_root,
                require_images_for_update=bool(args.require_images_for_update),
            )

            pred_text = run_one(
                model=model,
                processor=processor,
                user_text=user_text,
                images=images,
                max_new_tokens=int(args.max_new_tokens),
            )

            gt_yaml = parse_yaml(tgt_text)
            pred_yaml = parse_yaml(pred_text)

            scored: Dict[str, Any] = {}
            if gt_yaml is None or pred_yaml is None:
                parse_fail += 1
            else:
                if mode == "detect":
                    s = score_detect(gt_yaml, pred_yaml)
                    n_det += 1
                    ok_det += int(s["match_event"])
                    scored.update(s)
                else:
                    s = score_update(gt_yaml, pred_yaml)
                    n_up += 1
                    ok_up_all += int(s["match_all"])
                    ok_up_ac += int(s["match_action_command"])
                    scored.update(s)

            out_row = dict(row)
            out_row["eval_user_prompt"] = user_text
            out_row["gt_text"] = tgt_text
            out_row["pred_text"] = pred_text
            out_row["gt_yaml"] = gt_yaml
            out_row["pred_yaml"] = pred_yaml
            out_row.update(scored)

            fw.write(json.dumps(out_row, ensure_ascii=False) + "\n")

            if (i + 1) % 50 == 0:
                det_acc = (ok_det / n_det) if n_det else 0.0
                up_all = (ok_up_all / n_up) if n_up else 0.0
                up_ac = (ok_up_ac / n_up) if n_up else 0.0
                print(f"[eval_v2] {i+1}/{len(rows)} det_acc={det_acc:.3f} up_all={up_all:.3f} up_ac={up_ac:.3f} parse_fail={parse_fail}")

    def div(a: int, b: int) -> float:
        return float(a) / float(b) if b > 0 else 0.0

    print("==== HeLM Eval v2 (aligned with train_helm.py) ====")
    print(f"rows_total={len(rows)} parse_fail={parse_fail}")
    print(f"[DETECT] n={n_det} event_acc={div(ok_det, n_det):.4f}")
    print(f"[UPDATE] n={n_up} all_acc={div(ok_up_all, n_up):.4f} action_acc={div(ok_up_ac, n_up):.4f}")
    print(f"[saved] {outp}")


if __name__ == "__main__":
    main()