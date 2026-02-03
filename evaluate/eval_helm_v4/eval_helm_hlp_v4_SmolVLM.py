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
from tqdm import tqdm  # 진행 상황 확인을 위해 tqdm 추가 권장

# [수정 1] 최신 Transformers 클래스 및 Peft 사용
from transformers import AutoModelForImageTextToText, AutoProcessor
# from peft import PeftModel
"""
CUDA_VISIBLE_DEVICES=6 python evaluate/eval_helm_v4/eval_helm_hlp_v4_SmolVLM.py \
  --base_model /backups/ghkim/HeLM_v4/HLP_HeLM_v4_SmolVLM_Full_FT_v4_task_10/checkpoint-3500 \
  --jsonl /data/ghkim/helm_data/helm_v4_task_10/merged/all_val.jsonl \
  --out_jsonl /data/ghkim/helm_data/helm_v4_task_10/merged/test/eval_result_3500_val.jsonl \
  --num_images 1 \
  --batch_size 8 \
  --max_new_tokens 128 \
  --bf16 True \
  --max_samples 1000 \
  --attn_impl flash_attention_2
"""
logger = logging.getLogger("eval_helm_v4")


# -------------------------
# JSONL IO (기존과 동일)
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
                if not line: continue
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
# YAML parsing helpers (기존과 동일)
# -------------------------
def parse_yaml_loose(text: str) -> Dict[str, Any]:
    if text is None: return {}
    s = text.strip()
    s = re.sub(r"^```(?:yaml)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)
    if "\nassistant\n" in s:
        s = s.split("\nassistant\n", 1)[-1].strip()

    # [수정 2] SmolVLM은 "Assistant: "를 붙이는 경우가 많음 -> 후처리 강화
    if "Assistant:" in s:
        s = s.split("Assistant:")[-1].strip()

    try:
        out = yaml.safe_load(s)
        if isinstance(out, dict): return out
    except Exception:
        pass

    lines = []
    for ln in s.splitlines():
        if ":" in ln: lines.append(ln)
    try:
        out = yaml.safe_load("\n".join(lines))
        if isinstance(out, dict): return out
    except Exception:
        return {}
    return {}


def norm_str(x: Any) -> str:
    if x is None: return "None"
    if isinstance(x, bool): return "true" if x else "false"
    s = str(x).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def norm_bool(x: Any) -> Optional[bool]:
    if isinstance(x, bool): return x
    if x is None: return None
    s = str(x).strip().lower()
    if s in ["true", "yes", "y", "1"]: return True
    if s in ["false", "no", "n", "0"]: return False
    return None


# -------------------------
# Dataset / Collator (SmolVLM 전용으로 수정됨)
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
    if not isinstance(im, dict): raise ValueError("row['images'] must be a dict")

    table = im.get("table", None)
    if table is None: raise ValueError("row['images']['table'] missing")
    imgs.append(Image.open(table).convert("RGB"))

    if num_images == 2:
        wrist = im.get("wrist", None)
        if wrist is None: raise ValueError("num_images=2 but row['images']['wrist'] missing")
        imgs.append(Image.open(wrist).convert("RGB"))
    return imgs


class HelmEvalDatasetSmolVLM(Dataset):
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

        # [수정 3] SmolVLM Chat Template 적용
        # 이미지 토큰과 텍스트를 메시지로 구성
        messages = [
            {
                "role": "user",
                "content": [{"type": "image"}] * self.cfg.num_images + [{"type": "text", "text": user_prompt}]
            }
        ]

        # 템플릿 적용 (토큰화 X, 텍스트 반환)
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        # [수정 4] processor 호출 시 image_grid_thw 관련 옵션 제거
        model_inputs = self.processor(text=prompt, images=imgs, return_tensors="pt")

        input_ids = model_inputs["input_ids"].squeeze(0)
        attention_mask = model_inputs["attention_mask"].squeeze(0)
        pixel_values = model_inputs["pixel_values"].squeeze(0)

        # SmolVLM은 pixel_attention_mask가 있을 수 있음
        pixel_attention_mask = model_inputs.get("pixel_attention_mask")
        if pixel_attention_mask is not None:
            pixel_attention_mask = pixel_attention_mask.squeeze(0)

        # image_grid_thw는 제거됨 (None)

        return {
            "uid": row.get("uid", f"idx{i}"),
            "label": row.get("label", "UNKNOWN"),
            "mode": row.get("mode", "UNKNOWN"),
            "user_prompt": user_prompt,
            "gt_text": gt_text,
            "gt_yaml": row.get("gt_yaml", None),
            "meta": row.get("meta", {}),
            "row": row,

            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "pixel_attention_mask": pixel_attention_mask,  # 추가
        }


class EvalCollatorSmolVLM:
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
        # 데이터셋에서 리턴하는 키 이름("uid", "label" 등)과 정확히 일치시켜야 합니다.
        batch = {k: [f[k] for f in features] for k in ["uid", "label", "mode", "user_prompt", "gt_text", "row"]}

        # Tensor padding
        pad_id = self.tokenizer.pad_token_id
        batch["input_ids"] = self._left_pad_1d([f["input_ids"] for f in features], pad_id)
        batch["attention_mask"] = self._left_pad_1d([f["attention_mask"] for f in features], 0)

        # Pixel values stack (B, N, C, H, W) -> SmolVLM usually handles batch dim in forward
        # [수정] cat 대신 stack을 사용하여 (B, num_images, C, H, W) 5차원 유지
        batch["pixel_values"] = torch.stack([f["pixel_values"] for f in features], dim=0)

        if features[0].get("pixel_attention_mask") is not None:
            # 마스크도 동일하게 stack으로 묶어줌
            batch["pixel_attention_mask"] = torch.stack([f["pixel_attention_mask"] for f in features], dim=0)
        else:
            batch["pixel_attention_mask"] = None

        return batch

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
    # [수정 5] AutoModelForImageTextToText 및 dtype 사용
    print(f"Loading base model: {base_model}")
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        dtype=torch.bfloat16 if bf16 else torch.float16,
        _attn_implementation=attn_impl,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained("/ckpt/SmolVLM-500M-Instruct")

    # if adapter_path:
    #     print(f"Loading adapter: {adapter_path}")
    #     model = PeftModel.from_pretrained(model, adapter_path)
    #     print("Merged with Adapter")

    model.eval()
    return model, processor


# -------------------------
# Metrics (기존과 동일)
# -------------------------
def eval_detect(gt_yaml, pred_yaml):
    gt = norm_bool(gt_yaml.get("Event_Detected", None))
    pr = norm_bool(pred_yaml.get("Event_Detected", None))
    ok = (gt is not None) and (pr is not None) and (gt == pr)
    return ok, {"gt_event": gt, "pred_event": pr}


def eval_update(gt_yaml, pred_yaml):
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
        device: str,  # device_map="auto" 사용 시 모델 device 속성 따름
        batch_size: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        out_jsonl: str,
        log_every: int = 20,
):
    cfg = EvalConfig(jsonl_path="", model_name_or_path="", num_images=num_images)
    ds = HelmEvalDatasetSmolVLM(cfg, rows, processor)

    # [수정 6] Collator 교체
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=4,
                    collate_fn=EvalCollatorSmolVLM(processor))

    # Metric accumulators
    total = 0
    by_label = defaultdict(lambda: {"n": 0, "ok": 0})
    by_mode = defaultdict(lambda: {"n": 0, "ok": 0})
    update_key_stats = defaultdict(lambda: {"n": 0, "ok": 0})
    outputs = []

    print(f"Start inference on {len(rows)} samples...")

    for step, batch in enumerate(tqdm(dl, desc="Evaluating")):
        # device_map="auto"를 썼다면 model.device로 보내야 함.
        # 하지만 batch 처리는 수동으로 해야 하므로, model의 첫 파라미터 device를 참조하거나
        # accelerate가 있다면 그쪽 device를 씀. 여기선 안전하게 input tensor들을 model device로 이동.

        target_device = model.device

        input_ids = batch["input_ids"].to(target_device)
        attention_mask = batch["attention_mask"].to(target_device)
        pixel_values = batch["pixel_values"].to(target_device, dtype=model.dtype)

        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }

        if batch.get("pixel_attention_mask") is not None:
            gen_kwargs["pixel_attention_mask"] = batch["pixel_attention_mask"].to(target_device)

        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        # [수정 7] Generate 호출 (image_grid_thw 제거됨)
        gen = model.generate(**gen_kwargs)

        # [수정 8] 결과 디코딩 로직 개선
        # Prompt 길이를 이용하여 뒷부분만 잘라냄
        # input_ids가 left padding 되어 있으므로, 실제 프롬프트 길이는 배치 내에서 다를 수 있음.
        # 하지만 generate 출력은 input_ids를 포함해서 나옴.

        texts = []
        input_len = input_ids.shape[1]

        # batch decode and strip prompt
        # (SmolVLM은 보통 input tokens + new tokens를 반환)
        gen_tokens = gen[:, input_len:]
        decoded = processor.batch_decode(gen_tokens, skip_special_tokens=True)

        for txt in decoded:
            texts.append(txt.strip())

        # Metric Calculation (기존 로직 유지)
        for i in range(len(batch["uid"])):
            uid = batch["uid"][i]
            label = batch["label"][i]
            mode = batch["mode"][i]
            gt_text = batch["gt_text"][i]
            row = batch["row"][i]
            pred_text = texts[i]

            gt_yaml = row.get("gt_yaml", None)
            if not isinstance(gt_yaml, dict):
                gt_yaml = parse_yaml_loose(gt_text)
            pred_yaml = parse_yaml_loose(pred_text)

            if str(mode).upper() == "DETECT":
                ok, info = eval_detect(gt_yaml, pred_yaml)
            else:
                ok, info = eval_update(gt_yaml, pred_yaml)
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
                "gt_text": gt_text,
                "pred_text": pred_text,
                "match": ok,
                **info,
            })

    write_jsonl(out_jsonl, outputs)

    # Summary Logging
    def acc(d):
        return (d["ok"] / max(d["n"], 1)) * 100.0

    logger.info("========== EVAL SUMMARY ==========")
    logger.info(f"Total: {total}")
    for m, d in sorted(by_mode.items(), key=lambda x: x[0]):
        logger.info(f"[MODE {m}] n={d['n']} acc={acc(d):.2f}%")
    logger.info(f"Saved -> {out_jsonl}")


def main():
    logging.basicConfig(level=logging.INFO)
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, required=True)
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--num_images", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--do_sample", type=bool, default=False)  # python bool parsing 주의
    ap.add_argument("--bf16", type=bool, default=True)
    ap.add_argument("--attn_impl", type=str, default="flash_attention_2")
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--out_jsonl", type=str, required=True)

    # argparse bool issue handling
    args = ap.parse_args()

    rows = read_jsonl(args.jsonl)
    if args.max_samples > 0:
        random.shuffle(rows)
        rows = rows[:args.max_samples]

    model, processor = load_model_and_processor(
        args.base_model, args.adapter, False, args.bf16, args.attn_impl
    )

    run_eval(
        model, processor, rows, args.num_images, "cuda",
        args.batch_size, args.max_new_tokens, 0.0, 1.0,
        args.do_sample, args.out_jsonl
    )


if __name__ == "__main__":
    main()