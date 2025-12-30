# evaluate/eval_helm_hlp.py
from __future__ import annotations

import argparse
import json
import re
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

import yaml

# -------------------------
# Prompt templates (v2)
# -------------------------
DETECT_HEADER = (
    "You are the robot arm Visual Event Detector.\n"
    "Goal: Verify whether the CURRENT action has been fully completed in the image.\n"
    "Input: An image + Global_Instruction describing what counts as action completion.\n"
    "Decision rule:\n"
    "- Use the Global_Instruction and image as the ONLY completion criterion.\n"
    "- Event_Detected: true ONLY when the completion condition is clearly and unambiguously visible.\n"
    "- Otherwise (partial progress / occlusion / uncertainty) -> Event_Detected: false.\n"
    "Constraints:\n"
    "- Do not propose next actions.\n"
    "- Do not update or rewrite memory.\n"
    "- Do not output any text except YAML.\n"
    "Return YAML with exactly one key: Event_Detected (boolean).\n"
)

UPDATE_HEADER = (
    "You are the robot arm Logic State Manager.\n"
    "Context: Event_Detected=true or a Task Change has occurred.\n"
    "Inputs:\n"
    "- Global_Instruction defining the overall task.\n"
    "- Previous memory state (Working_Memory, Episodic_Context, Action_Command).\n"
    "Goal: Update internal memory and decide the next Action_Command based on the Global_Instruction.\n"
    "Logic Rules:\n"
    "1) Update Working_Memory to reflect the action that has just been completed.\n"
    "2) Check task status using Working_Memory and Global_Instruction:\n"
    "   - If the task continues: keep Episodic_Context unchanged and select the next Action_Command.\n"
    "   - If the task is finished: promote/summarize the final result into Episodic_Context and set Action_Command: done.\n"
    "Constraints:\n"
    "- Action_Command must be selected ONLY from Allowed_Action_Commands.\n"
    "- Do not add new actions or explanations.\n"
    "- Output YAML only with keys: Working_Memory, Episodic_Context, Action_Command.\n"
)

# ----------------------------
# YAML parsing
# ----------------------------

def parse_yaml_maybe(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse YAML output to dict safely.
    Returns None if parse fails.
    """
    if not text or not isinstance(text, str):
        return None
    s = text.strip()

    # common cleanup: sometimes model wraps with ```yaml ... ```
    s = re.sub(r"^```yaml\s*", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"^```\s*", "", s).strip()
    s = re.sub(r"\s*```$", "", s).strip()

    try:
        obj = yaml.safe_load(s)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    # normalize keys to exact expected capitalization if user uses consistent keys
    return obj


def normalize_bool(v: Any) -> Optional[bool]:
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


def normalize_str(v: Any) -> str:
    if v is None:
        return "None"
    if isinstance(v, str):
        return v.strip()
    return str(v).strip()


# ----------------------------
# Row loading
# ----------------------------

def load_row_user_assistant(row: Dict[str, Any]) -> Tuple[str, str]:
    """
    v2: row has user_prompt, gt_text
    v1: row has conversations (ShareGPT)
    """
    if "user_prompt" in row and "gt_text" in row:
        return str(row["user_prompt"]), str(row["gt_text"])

    conv = row.get("conversations", None)
    if isinstance(conv, list):
        user_text = ""
        asst_text = ""
        for m in conv:
            if m.get("from") == "user":
                user_text = str(m.get("value", ""))
            elif m.get("from") == "assistant":
                asst_text = str(m.get("value", ""))
        if user_text and asst_text:
            return user_text, asst_text

    raise ValueError("Row must contain either (user_prompt, gt_text) or conversations")


def infer_num_images(row: Dict[str, Any], arg_num_images: Optional[int]) -> int:
    """
    Priority:
      1) argparse --num_images if set
      2) row['views'] length if present
      3) row['images'] keys
    """
    if arg_num_images is not None:
        return int(arg_num_images)

    views = row.get("views", None)
    if isinstance(views, list) and len(views) in (1, 2):
        return len(views)

    imgs = row.get("images", {})
    if isinstance(imgs, dict):
        if "wrist" in imgs:
            return 2
        return 1

    return 1


def load_images(row: Dict[str, Any], num_image: int, camera_if_one: str = "table") -> List[Image.Image]:
    imgs = row.get("images", {})
    if not isinstance(imgs, dict):
        raise ValueError("row['images'] must be dict")

    if int(num_image) == 1:
        # choose camera
        if camera_if_one == "wrist":
            p = imgs.get("wrist", None)
            if not p:
                raise ValueError(f"num_image=1 camera=wrist requires 'wrist' in images. keys={list(imgs.keys())}")
        else:
            p = imgs.get("table", None)
            if not p:
                raise ValueError(f"num_image=1 camera=table requires 'table' in images. keys={list(imgs.keys())}")
        return [Image.open(p).convert("RGB")]

    # num_image == 2
    p_table = imgs.get("table", None)
    p_wrist = imgs.get("wrist", None)
    if not p_table or not p_wrist:
        raise ValueError(f"num_image=2 requires images.table and images.wrist. keys={list(imgs.keys())}")
    return [Image.open(p_table).convert("RGB"), Image.open(p_wrist).convert("RGB")]


def infer_mode(row: Dict[str, Any], user_text: str) -> str:
    m = row.get("mode", None)
    if isinstance(m, str) and m.strip():
        return m.strip().lower()
    # fallback: parse from prompt
    if "MODE: DETECT" in user_text:
        return "detect"
    if "MODE: UPDATE" in user_text:
        return "update"
    return "unknown"


# ----------------------------
# Generation
# ----------------------------

@torch.inference_mode()
def generate_text(
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
        images=images,  # 반드시 list length == num_image
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
        device_map=device,
    )

    if adapter:
        model = PeftModel.from_pretrained(model, adapter)
        model = model.merge_and_unload()  # 평가 안정성 (선택)

    processor = AutoProcessor.from_pretrained(base_model, trust_remote_code=True, use_fast=True)
    processor.tokenizer.padding_side = "left"

    model.eval()
    return model, processor


# ----------------------------
# Metrics
# ----------------------------

def score_detect(gt: Dict[str, Any], pred: Dict[str, Any]) -> Dict[str, Any]:
    gt_ev = normalize_bool(gt.get("Event_Detected"))
    pr_ev = normalize_bool(pred.get("Event_Detected"))

    gt_cmd = normalize_str(gt.get("Command"))
    pr_cmd = normalize_str(pred.get("Command"))

    return {
        "match_event": (gt_ev is not None and pr_ev is not None and gt_ev == pr_ev),
        "match_command": (gt_cmd == pr_cmd),
    }


def score_update(gt: Dict[str, Any], pred: Dict[str, Any]) -> Dict[str, Any]:
    gt_prog = normalize_str(gt.get("Progress"))
    pr_prog = normalize_str(pred.get("Progress"))

    gt_ws = normalize_str(gt.get("World_State"))
    pr_ws = normalize_str(pred.get("World_State"))

    return {
        "match_progress": (gt_prog == pr_prog),
        "match_world_state": (gt_ws == pr_ws),
    }


# ----------------------------
# Main
# ----------------------------

def main():
    p = argparse.ArgumentParser()

    p.add_argument("--jsonl", type=str, required=True, help="Path to jsonl (detect/update) or directory of shards")
    p.add_argument("--out_jsonl", type=str, default=None, help="Write per-row predictions to jsonl")

    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--adapter", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--load_in_4bit", action="store_true")
    p.add_argument("--attn_impl", type=str, default="flash_attention_2", choices=["flash_attention_2", "sdpa", "eager"])

    # image config
    p.add_argument("--num_images", type=int, default=None, choices=[1, 2], help="override images count")
    p.add_argument("--camera_if_one", type=str, default="table", choices=["table", "wrist"],
                   help="when num_images=1, which camera image to use")

    # eval config
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--limit", type=int, default=0, help="0=all")
    p.add_argument("--mode", type=str, default="both", choices=["both", "detect", "update"],
                   help="evaluate only detect/update or both")

    args = p.parse_args()

    model, processor = load_model_and_processor(
        base_model=args.base_model,
        adapter=args.adapter,
        device=args.device,
        load_in_4bit=bool(args.load_in_4bit),
        attn_impl=args.attn_impl,
    )

    # load rows
    jsonl_path = Path(args.jsonl)
    files: List[Path] = []
    if jsonl_path.is_dir():
        files = sorted(jsonl_path.rglob("*.jsonl"))
    else:
        files = [jsonl_path]

    rows: List[Dict[str, Any]] = []
    for f in files:
        with f.open("r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))

    if args.limit and args.limit > 0:
        rows = rows[: int(args.limit)]

    # stats accumulators
    n_detect = n_update = 0
    ok_detect_event = ok_detect_cmd = 0
    ok_update_prog = ok_update_ws = 0
    parse_fail = 0

    out_rows = []

    for i, row in enumerate(rows):
        user_text, gt_text = load_row_user_assistant(row)
        mode = infer_mode(row, user_text)

        if args.mode != "both" and mode != args.mode:
            continue

        num_image = infer_num_images(row, args.num_images)
        images = load_images(row, num_image=num_image, camera_if_one=args.camera_if_one)

        pred_text = generate_text(
            model=model,
            processor=processor,
            user_text=user_text,
            images=images,
            num_image=num_image,
            max_new_tokens=int(args.max_new_tokens),
        )

        gt_yaml = parse_yaml_maybe(gt_text)
        pred_yaml = parse_yaml_maybe(pred_text)

        if gt_yaml is None or pred_yaml is None:
            parse_fail += 1
            scored = {}
        else:
            if mode == "detect":
                s = score_detect(gt_yaml, pred_yaml)
                n_detect += 1
                ok_detect_event += int(s["match_event"])
                ok_detect_cmd += int(s["match_command"])
                scored = s
            elif mode == "update":
                s = score_update(gt_yaml, pred_yaml)
                n_update += 1
                ok_update_prog += int(s["match_progress"])
                ok_update_ws += int(s["match_world_state"])
                scored = s
            else:
                # unknown mode: do nothing but still store
                scored = {}

        out_row = dict(row)
        out_row["pred_text"] = pred_text
        out_row["gt_yaml"] = gt_yaml
        out_row["pred_yaml"] = pred_yaml
        out_row.update(scored)
        out_rows.append(out_row)

        if (i + 1) % 50 == 0:
            print(f"[eval] processed {i+1}/{len(rows)} rows...")

    # summary
    def safe_div(a: int, b: int) -> float:
        return float(a) / float(b) if b > 0 else 0.0

    print("==== Evaluation Summary (v2) ====")
    print(f"rows_total={len(rows)}  parse_fail={parse_fail}")
    if n_detect > 0:
        print(f"[DETECT] n={n_detect}  event_acc={safe_div(ok_detect_event, n_detect):.4f}  cmd_acc={safe_div(ok_detect_cmd, n_detect):.4f}")
    if n_update > 0:
        print(f"[UPDATE] n={n_update}  progress_acc={safe_div(ok_update_prog, n_update):.4f}  world_state_acc={safe_div(ok_update_ws, n_update):.4f}")

    if args.out_jsonl:
        out_path = Path(args.out_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fp:
            for r in out_rows:
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[eval] wrote predictions to: {out_path}")


if __name__ == "__main__":
    main()