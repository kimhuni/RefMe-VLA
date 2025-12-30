# eval_real_time_qwen.py
from __future__ import annotations
import re
import time
from typing import Any, Dict, Optional

import torch
import yaml
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from peft import PeftModel


# === training(helm_dataset.py)과 동일 헤더 ===
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


def _safe_yaml_dict(text: str) -> Dict[str, Any]:
    """
    모델 출력이 깨져도 key 라인만 최대한 추출하고 yaml.safe_load 시도.
    """
    raw = text.strip()
    try:
        d = yaml.safe_load(raw)
        if isinstance(d, dict):
            return d
    except Exception:
        pass

    # fallback: key: value 라인만 긁기
    out = {}
    for ln in raw.splitlines():
        m = re.match(r"^\s*([A-Za-z_]+)\s*:\s*(.*)\s*$", ln)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def parse_detect_yaml(text: str) -> bool:
    d = _safe_yaml_dict(text)
    v = d.get("Event_Detected", False)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1")
    return False


def parse_update_yaml(text: str) -> Dict[str, str]:
    d = _safe_yaml_dict(text)
    out = {
        "Working_Memory": "",
        "Episodic_Context": "",
        "Action_Command": "",
    }
    for k in out.keys():
        v = d.get(k, "")
        if v is None:
            v = ""
        out[k] = str(v).strip()
    return out


class HLPQwenV2:
    """
    Qwen2.5-VL + (Q)LoRA adapter inference
    - detect(): Event_Detected(bool)
    - update(): memory(dict: Working_Memory/Episodic_Context/Action_Command)
    """

    def __init__(
        self,
        base_model_path: str,
        adapter_path: str,
        device: str = "cuda:0",
        attn_impl: str = "sdpa",
        load_in_4bit: bool = True,
        max_new_tokens_detect: int = 32,
        max_new_tokens_update: int = 128,
    ):
        self.device = device
        self.max_new_tokens_detect = max_new_tokens_detect
        self.max_new_tokens_update = max_new_tokens_update

        self.processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        bnb_config = None
        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        t0 = time.time()
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map=device,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, adapter_path)
        self.model = model.merge_and_unload().eval()
        print(f"[HLP] load done: {time.time()-t0:.2f}s")

    @torch.no_grad()
    def _generate_text(self, batch: Dict[str, torch.Tensor], max_new_tokens: int) -> str:
        inputs = {k: v.to(self.device) for k, v in batch.items()}
        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        in_len = inputs["input_ids"].shape[1]
        gen_trim = gen_ids[:, in_len:]
        out_text = self.tokenizer.batch_decode(
            gen_trim, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        return out_text

    def detect(self, batch: Dict[str, torch.Tensor]) -> bool:
        print("[HLP]===DETECT MODE===")
        out_text = self._generate_text(batch, self.max_new_tokens_detect)
        print("[HLP] raw output: \n", out_text)

        return parse_detect_yaml(out_text)

    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, str]:
        print("[HLP]===UPDATE MODE===")
        out_text = self._generate_text(batch, self.max_new_tokens_update)
        print("[HLP] raw output: \n", out_text)

        return parse_update_yaml(out_text)