# realtime_HLP_qwen.py

from dataclasses import dataclass
from typing import Dict

import json
import logging

import torch
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    BitsAndBytesConfig,
)

from common.utils.utils import (
    init_devices,
    get_safe_torch_device,
    init_keyboard_listener
)

from peft import PeftModel
from torchvision.transforms.functional import to_pil_image

import time


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
        "Output JSON: {\"desc_1\":\"...\",\"desc_2\":\"...\",\"status\":\"...\"}"
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

        hlp_loading_start = time.time()

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
            attn_implementation="flash_attention_2",
            # attn_implementation="sdpa",
        )

        # processor = AutoProcessor.from_pretrained(cfg.base_model_path, use_fast=True, trust_remote_code=True)
        # tokenizer = processor.tokenizer
        processor = AutoProcessor.from_pretrained(cfg.adapter_path, trust_remote_code=True)
        tokenizer = processor.tokenizer

        self.processor = processor
        self.tokenizer = tokenizer

        # pad_token을 "추가"하지 말고, "ID만" 재지정 (vocab 안 늘어남)
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Ensure FA2-safe batching: left padding + valid pad token
        self.tokenizer.padding_side = "left"

        print(f"[eval] tokenizer padding_side={tokenizer.padding_side}, "
              f"pad={self.tokenizer.pad_token_id}, eos={self.tokenizer.eos_token_id}")

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        print(
            f"[eval] tokenizer padding_side={self.tokenizer.padding_side}, pad={self.tokenizer.pad_token_id}, eos={self.tokenizer.eos_token_id}")

        # ✅ vocab 크기 불일치시에만 임베딩 리사이즈
        tok_len = len(self.tokenizer)
        if getattr(base_model.config, "vocab_size", None) != tok_len:
            base_model.resize_token_embeddings(tok_len)
            base_model.config.vocab_size = tok_len  # 동기화

        print(f"Loading adapter from: {cfg.adapter_path}")
        # (3) PEFT 모델로 어댑터를 베이스 모델 위에 "덮어씌움"
        model = PeftModel.from_pretrained(
            base_model,
            cfg.adapter_path,
            ignore_mismatched_sizes=True
        )

        print("[HLP] model load & merging time: ", time.time() - hlp_loading_start, "sec")


        # (4) ★★★ 요청하신 "임시 병합" (메모리상에서 병합 후 PEFT 래퍼 제거) ★★★
        print("Merging adapter into base model (in memory)...")
        model = model.merge_and_unload()

        # Propagate pad id to model configs to avoid generation-side surprises
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        if hasattr(model, "config"):
            model.config.pad_token_id = self.tokenizer.pad_token_id

        self.model = model
        model.eval()  # 평가 모드

        # HLP 내부 상태
        self.prev_desc: str = ""
        self.prev_status: str = "NOT_DONE"

        # Optional keyboard listener to reset HLP state with ESC
        self._listener = None
        self._event = None
        if init_keyboard_listener is not None:
            try:
                self._listener, self._event, _ = init_keyboard_listener()
                logging.info("[HLP] Keyboard listener initialized: press ESC to reset HLP state.")
            except Exception as e:
                logging.warning(f"[HLP] Keyboard listener init failed: {e}")

    def reset(self):
        """ESC 등으로 HLP 히스토리를 리셋할 때 호출."""
        logging.info("[HLP] Reset prev_desc / prev_status")
        self.prev_desc = ""
        self.prev_status = "NOT_DONE"


    def attach_keyboard_event(self, event: dict):
        """Optionally attach an external keyboard event dict (e.g., from main loop)."""
        self._event = event


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
        # Reset internal history if ESC was pressed
        if self._event and self._event.get("set initial", False):
            self.reset()
            # clear the flag so we don't reset every step
            self._event["set initial"] = False
            logging.info("[HLP] ESC detected: state reset to prev_desc='' / prev_status='NOT_DONE'.")

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

        infer_start_time = time.time()

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
        print(
            # f"[HLP] status={parsed['status']} | subtask={parsed['subtask']} | "
            f"[HLP] status={parsed['status']} | subtask={task} | "
            f"\n[HLP] desc_1={parsed['desc_1']} | desc_2={parsed['desc_2']}"
            f"\n[HLP] inference time : {time.time() - infer_start_time}"
        )
        # print("raw", output_text)

        return parsed
