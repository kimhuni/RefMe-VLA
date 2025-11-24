# hlp_server.py  (HLP 전용 conda/env에서 실행)
# hlp_server.py  — FastAPI server for HLP (Qwen2.5‑VL)
import os
import io
import re
import time
import json
import base64
import logging
from dataclasses import dataclass
from typing import Optional

import torch
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import (
    AutoProcessor, AutoTokenizer, AutoConfig,
    Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig,
)
from peft import PeftModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# =========================
# Helpers
# =========================

def b64_to_pil(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

def make_hlp_prompt(task: str, prev_desc: str, prev_status: str) -> str:
    return (
        "You are an image-analysis expert for robot manipulation.\n"
        "INPUT_IMAGES: [SIDE]=global scene view; [WRIST]=close-up wrist camera.\n"
        f"TASK: {task}\n"
        f"PREV_DESC: {prev_desc}\n"
        f"PREV_STATUS: {prev_status}\n"
        "Describe what is visibly happening now (desc_1) and the visible evidence for completion (desc_2).\n"
        "Then decide the status: DONE / NOT_DONE / UNCERTAIN.\n"
        "Output JSON: {\"desc_1\":\"...\",\"desc_2\":\"...\",\"status\":\"...\",\"subtask\":\"...\"}"
    )

def parse_hlp_json(text: str) -> dict:
    # extract the first JSON object; be lenient
    m = re.search(r"\{.*\}", text, flags=re.S)
    d = {"desc_1": "", "desc_2": "", "status": "UNCERTAIN", "subtask": ""}
    if m:
        try:
            d.update(json.loads(m.group(0)))
        except Exception:
            pass
    # sanitize
    d["status"] = str(d.get("status", "UNCERTAIN")).upper()
    if d["status"] not in {"DONE", "NOT_DONE", "UNCERTAIN"}:
        d["status"] = "UNCERTAIN"
    d["desc_1"] = str(d.get("desc_1", ""))[:512]
    d["desc_2"] = str(d.get("desc_2", ""))[:512]
    d["subtask"] = str(d.get("subtask", ""))[:128]
    return d

# =========================
# Config + Server
# =========================

@dataclass
class ServerConfig:
    merged_dir: Optional[str] = None           # preferred (merged model directory)
    base_dir: Optional[str] = None             # fallback for processor / or base for adapter path
    adapter_dir: Optional[str] = None          # if provided, load base+adapter instead of merged_dir
    use_4bit: bool = False
    device_map: str = "auto"
    max_new_tokens: int = 192

class HLPServer:
    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        self.prev_desc = ""
        self.prev_status = "NOT_DONE"

        if cfg.merged_dir is None and (cfg.base_dir is None or cfg.adapter_dir is None):
            raise ValueError("Provide either merged_dir or (base_dir + adapter_dir).")

        # Processor: safest from base_dir
        proc_src = cfg.base_dir if cfg.base_dir else cfg.merged_dir
        logging.info(f"[HLP] Loading processor from: {proc_src}")
        self.processor = AutoProcessor.from_pretrained(proc_src, use_fast=True, trust_remote_code=True)

        # Tokenizer: from merged_dir if available (to match training vocab)
        tok_src = cfg.merged_dir if cfg.merged_dir else cfg.base_dir
        logging.info(f"[HLP] Loading tokenizer from: {tok_src}")
        self.tokenizer = AutoTokenizer.from_pretrained(tok_src, use_fast=True, trust_remote_code=True)

        # Config: from merged_dir if available
        cfg_src = cfg.merged_dir if cfg.merged_dir else cfg.base_dir
        logging.info(f"[HLP] Loading config from: {cfg_src}")
        config = AutoConfig.from_pretrained(cfg_src, trust_remote_code=True)

        # Align vocab size
        tv = getattr(self.tokenizer, "vocab_size", None)
        if tv is not None and getattr(config, "vocab_size", None) != tv:
            logging.info(f"[HLP] Adjusting vocab_size: {getattr(config, 'vocab_size', None)} -> {tv}")
            config.vocab_size = tv

        # Quantization config
        bnb = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
        ) if cfg.use_4bit else None

        # Model
        if cfg.adapter_dir:
            # base + adapter → merge in memory
            if not cfg.base_dir:
                raise ValueError("base_dir is required when adapter_dir is provided.")
            logging.info(f"[HLP] Loading BASE from: {cfg.base_dir}")
            base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                cfg.base_dir,
                quantization_config=bnb,
                torch_dtype=None if bnb else torch.bfloat16,
                device_map=cfg.device_map,
                attn_implementation="eager",
                trust_remote_code=True,
            )
            logging.info(f"[HLP] Loading ADAPTER from: {cfg.adapter_dir}")
            peft = PeftModel.from_pretrained(base, cfg.adapter_dir, ignore_mismatched_sizes=True)
            logging.info("[HLP] Merging adapter into base (in-memory).")
            self.model = peft.merge_and_unload().eval()
        else:
            # merged model direct load
            if not cfg.merged_dir:
                raise ValueError("merged_dir must be provided when adapter_dir is None.")
            logging.info(f"[HLP] Loading MERGED model from: {cfg.merged_dir}")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                cfg.merged_dir,
                config=config,
                quantization_config=bnb,
                torch_dtype=None if bnb else torch.bfloat16,
                device_map=cfg.device_map,
                attn_implementation="eager",
                trust_remote_code=True,
            ).eval()

        logging.info("[HLP] Model ready.")

    @torch.inference_mode()
    def infer(self, task: str, prev_desc: str, prev_status: str, side: Image.Image, wrist: Image.Image):
        t0 = time.time()
        prompt = make_hlp_prompt(task, prev_desc, prev_status)
        messages = [
            {"role": "user", "content": [
                {"type": "image"},  # SIDE
                {"type": "image"},  # WRIST
                {"type": "text", "text": prompt},
            ]}
        ]
        tmpl = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[tmpl], images=[side, wrist], return_tensors="pt").to(self.model.device)

        out_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=False,
            temperature=0.0,
        )
        gen_ids = out_ids[:, inputs.input_ids.shape[1]:]
        out_text = self.processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

        parsed = parse_hlp_json(out_text)
        # update internal state (optional)
        self.prev_desc = (parsed.get("desc_1") or self.prev_desc)
        self.prev_status = parsed.get("status", self.prev_status)

        return {
            "desc_1": parsed.get("desc_1", ""),
            "desc_2": parsed.get("desc_2", ""),
            "status": parsed.get("status", "UNCERTAIN"),
            "subtask": parsed.get("subtask", ""),
            "raw_text": out_text,
            "latency_ms": (time.time() - t0) * 1000.0,
        }

    def reset(self):
        self.prev_desc, self.prev_status = "", "NOT_DONE"

# =========================
# FastAPI app + schemas
# =========================

app = FastAPI()
SERVER: Optional[HLPServer] = None

class InferReq(BaseModel):
    task: str
    prev_desc: str = ""
    prev_status: str = "NOT_DONE"
    side_b64: str
    wrist_b64: str

class InferResp(BaseModel):
    desc_1: str
    desc_2: str
    status: str
    subtask: str
    raw_text: str
    latency_ms: float

@app.get("/health")
def health():
    return {"ok": SERVER is not None}

@app.post("/reset")
def reset():
    assert SERVER is not None, "Server not initialized"
    SERVER.reset()
    return {"ok": True}

@app.post("/infer", response_model=InferResp)
def infer(req: InferReq):
    assert SERVER is not None, "Server not initialized"
    side = b64_to_pil(req.side_b64)
    wrist = b64_to_pil(req.wrist_b64)
    out = SERVER.infer(req.task, req.prev_desc, req.prev_status, side, wrist)
    return InferResp(**out)

# =========================
# Entrypoint
# =========================

if __name__ == "__main__":
    import argparse
    import uvicorn

    p = argparse.ArgumentParser()
    p.add_argument("--merged_dir", type=str, default=None, help="Path to merged Qwen2.5‑VL model directory")
    p.add_argument("--base_dir", type=str, default=None, help="Path to base Qwen2.5‑VL (processor or base for adapter)")
    p.add_argument("--adapter_dir", type=str, default=None, help="Optional PEFT adapter to merge at startup")
    p.add_argument("--use_4bit", action="store_true", help="Load model in 4‑bit (bnb) for VRAM savings")
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--max_new_tokens", type=int, default=192)
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    args = p.parse_args()

    # Minimal validation
    if args.merged_dir is None and (args.base_dir is None or args.adapter_dir is None):
        p.error("Provide either --merged_dir or both --base_dir and --adapter_dir.")

    cfg = ServerConfig(
        merged_dir=args.merged_dir,
        base_dir=args.base_dir,
        adapter_dir=args.adapter_dir,
        use_4bit=args.use_4bit,
        device_map=args.device_map,
        max_new_tokens=args.max_new_tokens,
    )
    SERVER = HLPServer(cfg)
    uvicorn.run(app, host=args.host, port=args.port, workers=1)