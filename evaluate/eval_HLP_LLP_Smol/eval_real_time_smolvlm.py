# eval_real_time_qwen.py v3
from __future__ import annotations
import re
import time
from typing import Any, Dict, Optional, List, Tuple

import torch
import yaml
from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
from peft import PeftModel

DETECT_SYSTEM = (
    "You are the robot arm Visual Event Detector.\n"
    "Goal: Decide whether the target EVENT is detected in the current image.\n"
    "The EVENT corresponds to a meaningful completion moment for the current stage of the Global_Instruction."
    "Input: An image + Global_Instruction describing what counts as action completion"
    " + Memory (may help interpret the current stage/goal)\n"
    "Decision rule:\n"
    "- Use the Global_Instruction  as the primary criterion.\n"
    "- You MAY use Memory only to understand what “completion” means for the current stage."
    "- Event_Detected: true when the completion (or clearly post-completion state) is visible.\n"
    "- Otherwise (partial progress / occlusion / uncertainty) -> Event_Detected: false.\n"
    "Constraints:\n"
    "- Do not propose next actions.\n"
    "- Do not update or rewrite memory.\n"
    "- Do not output any text except YAML.\n"
    "Return YAML with exactly one key: Event_Detected (boolean).\n"
)

UPDATE_SYSTEM = (
    "You are the robot arm Logic State Manager.\n"
    "Context: Event_Detected=true or a Task Change has occurred.\n"
    "Inputs:\n"
    "- Global_Instruction defining the overall task.\n"
    "- Previous memory state (with keys: Working_Memory, Episodic_Context, Action_Command).\n"
    "- Allowed_Action_Commands (a small fixed list)"
    "Goal: Produce the next memory state after the event, preserving information"
    "and decide the next Action_Command based on the Global_Instruction.\n"
    "Logic Rules ((copy-first, lossless)):\n"
    "1) Start by COPYING Previous_Memory fields.\n"
    "2) Update Working_Memory to reflect the newly completed step."
    "- Prefer appending or small edits over rewriting."
    "3) Episodic_Context:"
    "- If the task is not finished, keep it EXACTLY unchanged."
    "- If the task is finished, update it to summarize the final outcome."
    "4) Action_Command:"
    "- Must be EXACTLY one of Allowed_Action_Commands."
    "- Use done only when the task is finished."
    "Constraints:\n"
    "- Action_Command must be selected ONLY from Allowed_Action_Commands.\n"
    "- Do not add new actions or explanations.\n"
    "- Output YAML only with keys: Action_Command, Working_Memory, Episodic_Context.\n"
)


def _safe_yaml_dict(text: str) -> Dict[str, Any]:
    raw = text.strip()
    # Assistant: 접두사가 붙어 나오는 경우 제거
    if "Assistant:" in raw:
        raw = raw.split("Assistant:")[-1].strip()
    try:
        d = yaml.safe_load(raw)
        if isinstance(d, dict): return d
    except Exception: pass
    out = {}
    for ln in raw.splitlines():
        m = re.match(r"^\s*([A-Za-z_]+)\s*:\s*(.*)\s*$", ln)
        if m: out[m.group(1)] = m.group(2)
    return out


def parse_detect_yaml(text: str) -> bool:
    d = _safe_yaml_dict(text)
    v = d.get("Event_Detected", False)
    if isinstance(v, bool):
        detected = v
    elif isinstance(v, str):
        detected = v.strip().lower() in ("true", "yes", "1")
    else:
        detected = False

    event = str(d.get("Event", "none")).strip()
    if not detected:
        event = "none"
    if event == "":
        event = "none"
    return detected, event


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


class HLPSmolVLM:
    """ SmolVLM (Full-FT) 실시간 추론 클래스 """

    def __init__(
            self,
            base_model_path: str,
            adapter_path: Optional[str] = None,  # Full-FT면 None
            device: str = "cuda:0",
            attn_impl: str = "flash_attention_2",
            max_new_tokens_detect: int = 32,
            max_new_tokens_update: int = 128,
    ):
        self.device = device
        self.max_new_tokens_detect = max_new_tokens_detect
        self.max_new_tokens_update = max_new_tokens_update

        self.model = AutoModelForImageTextToText.from_pretrained(
            base_model_path,
            dtype="bfloat16",
            _attn_implementation=attn_impl,
            device_map=device,
            trust_remote_code=True,
        )
        # [수정] AutoModelForImageTextToText 사용
        self.processor = AutoProcessor.from_pretrained("/home/minji/Desktop/data/ckpt/SmolVLM-500M-Instruct", trust_remote_code=True)

        self.model.eval()

        # if adapter_path:
        #     from peft import PeftModel
        #     self.model = PeftModel.from_pretrained(self.model, adapter_path)
        #     print(f"[HLP] Adapter Loaded: {adapter_path}")

    @torch.no_grad()
    def _generate(self, images: List[Image.Image], system_prompt: str, user_text: str, max_tokens: int) -> str:
        # [수정] system_prompt를 따로 넣지 않습니다.
        # 이미 user_text(make_detect_prompt의 결과)에 포함되어 있기 때문입니다.

        # 오프라인 코드와 동일하게 단일 user 메시지 구조로 변경
        messages = [
            {
                "role": "user",
                # SmolVLM의 이미지 토큰 위치를 오프라인 95% 성공 시와 동일하게 배치
                "content": [{"type": "image"}] * len(images) + [{"type": "text", "text": user_text}]
            }
        ]

        # 템플릿 적용
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

        # [중요] 실시간 이미지가 BGR이라면 RGB로 변환되어 들어와야 함
        # main 루프에서 처리하지 않았다면 여기서 강제 변환 (필요시)
        # images = [img.convert("RGB") for img in images]

        inputs = self.processor(text=prompt, images=images, return_tensors="pt").to(self.device, torch.bfloat16)

        gen_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=self.processor.tokenizer.pad_token_id,
        )

        # 정확한 슬라이싱 (프롬프트 제외)
        prompt_len = inputs["input_ids"].shape[1]
        response = self.processor.batch_decode(gen_ids[:, prompt_len:], skip_special_tokens=True)[0]

        return response.strip()

    def detect(self, obs_pil: List[Image.Image], user_text: str) -> Tuple[bool, str]:
        raw = self._generate(obs_pil, DETECT_SYSTEM, user_text, self.max_new_tokens_detect)
        print("--------------------------------------------------------")
        print("\n[DETECT] Output:", raw)
        print("--------------------------------------------------------")
        return parse_detect_yaml(raw)

    def update(self, obs_pil: List[Image.Image], user_text: str) -> Dict[str, str]:
        raw = self._generate(obs_pil, UPDATE_SYSTEM, user_text, self.max_new_tokens_update)
        print("--------------------------------------------------------")
        print("\n[UPDATE] Output:", raw)
        print("--------------------------------------------------------")
        return parse_update_yaml(raw)