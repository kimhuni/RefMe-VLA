# realtime_HLP_qwen.py

from dataclasses import dataclass
from typing import Dict

import json
import logging

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
)
from peft import PeftModel
from torchvision.transforms.functional import to_pil_image


@dataclass
class HLPConfig:
    base_model_path: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    adapter_path: str = "path/to/hlp_adapter"
    is_qlora: bool = True
    device: str = "cuda:0"
    max_new_tokens: int = 256


def make_hlp_prompt(task: str, prev: str, prev_status: str) -> str:
    """
    이전에 정리했던 HLP system prompt를 함수로 분리.
    필요하면 여기서 SYSTEM 버전 / PARTIALLY_DONE 버전 등 갈라 써도 됨.
    """
    return (
        "You are an image-analysis expert for robot manipulation.\n"
        "INPUT_IMAGES: [SIDE]=global scene view; [WRIST]=close-up wrist camera.\n"
        f"TASK: {task}\n"
        f"PREV_DESC: {prev}\n"
        f"PREV_STATUS: {prev_status}\n"
        "Describe what is visibly happening now (desc_1) and the visible evidence for completion (desc_2).\n"
        "Then decide the status: DONE / NOT_DONE / UNCERTAIN.\n"
        # subtask까지 HLP가 뽑아줄 수 있게 확장 (원하면 제거 가능)
        "If useful, also infer a finer-grained textual subtask (subtask), "
        "like 'approach the button', 'press the button fully', etc.\n"
        "Output JSON: {\"desc_1\":\"...\",\"desc_2\":\"...\",\"status\":\"...\",\"subtask\":\"...\"}"
    )


def _safe_tensor_to_pil(t):
    """카메라에서 나오는 torch.Tensor(C,H,W) 또는 (1,C,H,W) → PIL.Image."""
    if isinstance(t, torch.Tensor):
        if t.ndim == 4:
            t = t[0]
        return to_pil_image(t.cpu())
    raise ValueError(f"[HLP] Unsupported image type: {type(t)}")


def _parse_hlp_json(text: str) -> Dict[str, str]:
    """
    HLP 출력 문자열에서 JSON 부분만 찾아서 파싱.
    실패 시 status=UNCERTAIN / subtask="" 로 fallback.
    """
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        json_str = text[start:end]
        data = json.loads(json_str)
        return {
            "desc_1": data.get("desc_1", ""),
            "desc_2": data.get("desc_2", ""),
            "status": data.get("status", "UNCERTAIN"),
            "subtask": data.get("subtask", ""),
        }
    except Exception as e:
        logging.warning(f"[HLP] JSON parse failed: {e} | raw: {text[:200]}...")
        return {"desc_1": "", "desc_2": "", "status": "UNCERTAIN", "subtask": ""}


class HighLevelPlanner:
    """
    Qwen2.5-VL + (Q)LoRA HLP를 실시간으로 한 step씩 돌리는 래퍼.

    사용법:
      cfg_hlp = HLPConfig(...)
      hlp = HighLevelPlanner(cfg_hlp)
      out = hlp.step(task, side_img_tensor, wrist_img_tensor)
      # out: {"desc_1": ..., "desc_2": ..., "status": ..., "subtask": ...}
    """

    def __init__(self, cfg: HLPConfig):
        self.cfg = cfg
        self.device = cfg.device
        self.max_new_tokens = cfg.max_new_tokens

        bnb_config = None
        if cfg.is_qlora:
            logging.info("[HLP] Loading base model in 4-bit (QLoRA).")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        logging.info(f"[HLP] Loading base model from: {cfg.base_model_path}")
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            cfg.base_model_path,
            quantization_config=bnb_config,
            device_map=cfg.device,
            torch_dtype="auto",
            attn_implementation="eager",
        )

        logging.info(f"[HLP] Loading adapter from: {cfg.adapter_path}")
        model = PeftModel.from_pretrained(
            base_model,
            cfg.adapter_path,
            ignore_mismatched_sizes=True,
        )

        logging.info("[HLP] Merging adapter into base model (memory-only).")
        self.model = model.merge_and_unload()
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(cfg.base_model_path)
        self.tokenizer = self.processor.tokenizer

        # HLP 내부 상태
        self.prev_desc: str = ""
        self.prev_status: str = "NOT_DONE"

    def reset(self):
        """ESC 등으로 HLP 히스토리를 리셋할 때 호출."""
        logging.info("[HLP] Reset prev_desc / prev_status")
        self.prev_desc = ""
        self.prev_status = "NOT_DONE"

    @torch.inference_mode()
    def step(
        self,
        task: str,
        side_img_tensor: torch.Tensor,
        wrist_img_tensor: torch.Tensor,
    ) -> Dict[str, str]:
        """
        한 step:
          - SIDE, WRIST 이미지 + task + prev_desc/status 로 프롬프트 생성
          - Qwen 추론
          - JSON 파싱
          - prev_desc, prev_status 업데이트
        """
        side_img = _safe_tensor_to_pil(side_img_tensor)
        wrist_img = _safe_tensor_to_pil(wrist_img_tensor)
        images_list = [side_img, wrist_img]

        user_prompt_text = make_hlp_prompt(task, self.prev_desc, self.prev_status)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # SIDE
                    {"type": "image"},  # WRIST
                    {"type": "text", "text": user_prompt_text},
                ],
            }
        ]

        prompt_string = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[prompt_string],
            images=images_list,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        parsed = _parse_hlp_json(output_text)

        # 내부 상태 업데이트
        if parsed["desc_1"]:
            self.prev_desc = parsed["desc_1"]
        self.prev_status = parsed["status"]

        # 로그 (요청 2번)
        logging.info(
            f"[HLP] status={parsed['status']} | subtask={parsed['subtask']} | "
            f"desc_1={parsed['desc_1']} | desc_2={parsed['desc_2']}"
        )

        return parsed
