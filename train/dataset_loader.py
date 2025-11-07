# dataset_loader.py (REWRITE v5 - PROCESS-LOCAL INIT FIX)
import json
from typing import Dict, Any, List
from dataclasses import dataclass
import glob
import logging
import os  # <-- os.getpid()를 위해 추가

import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoProcessor  # <-- AutoProcessor 임포트
from torch.nn.utils.rnn import pad_sequence

# 로거 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CAM_FALLBACK_ORDER = ["side", "wrist"]


def make_train_prompt(task: str, prev: str, prev_status: str) -> str:
    """
    Generate a concise training prompt for image analysis in robot manipulation.
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


# ===== 1) Dataset: 모든 전처리를 여기서 수행 (num_workers 병렬 처리) =====
class VLMJSONDataset(Dataset):
    """
    [V5] __getitem__에서 프로세스-로컬(process-local) 'processor'를 초기화합니다.
    """

    def __init__(self, paths: list[str], model_name_or_path: str, image_key: str = "image_path"):
        self.rows = []
        for jsonl_path in paths:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for ln, line in enumerate(f, start=1):
                    try:
                        obj = json.loads(line)
                        obj["_src_file"] = jsonl_path
                        obj["_src_line"] = ln
                        self.rows.append(obj)
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping bad JSON line {ln} in {jsonl_path}")

        # [수정] processor 객체 대신 'model_name_or_path' (문자열)을 저장
        self.model_name_or_path = model_name_or_path
        self.image_key = image_key
        # [수정] processor를 None으로 초기화. 각 워커가 스스로 로드할 것임.
        self.processor = None

        print(f"[VLMJSONDataset] files={len(paths)} samples={len(self.rows)}")
        if len(self.rows) > 0:
            print(f"[VLMJSONDataset] sample_keys={list(self.rows[0].keys())}")

    def __len__(self):
        return len(self.rows)

    def _context_info(self, row: Dict[str, Any]) -> str:
        parts = []
        for k in ["uid", "chunk_id", "episode_id", "timestamp_ms"]:
            if k in row:
                parts.append(f"{k}={row[k]}")
        return ", ".join(parts) if parts else "no context"

    def _load_images(self, row: Dict[str, Any]) -> List[Image.Image]:
        # (이 함수는 V4와 동일)
        context = self._context_info(row)
        if "images" in row and isinstance(row["images"], dict):
            cams = row.get("meta", {}).get("capture", {}).get("cameras")
            if not cams:
                present = list(row["images"].keys())
                ordered = [c for c in CAM_FALLBACK_ORDER if c in present]
                ordered += sorted([c for c in present if c not in CAM_FALLBACK_ORDER])
                cams = ordered

            imgs = []
            for cam in cams:
                if cam in row["images"]:
                    path = row["images"][cam]
                    try:
                        imgs.append(Image.open(path).convert("RGB"))
                    except Exception as e:
                        raise IOError(
                            f"Failed to load '{cam}' image '{path}' ({context}) at {row.get('_src_file', '?')}:{row.get('_src_line', '?')}: {e}") from e
            if not imgs:
                raise ValueError(f"No images could be loaded for sample ({context})")
            return imgs

        elif self.image_key in row:
            try:
                return [Image.open(row[self.image_key]).convert("RGB")]
            except Exception as e:
                raise IOError(f"Error loading image '{row[self.image_key]}' for sample ({context}): {e}") from e

        return []

    def __getitem__(self, i: int) -> Dict[str, Any]:

        # --- [핵심 수정] 프로세스-로컬(워커-로컬) processor 초기화 ---
        if self.processor is None:
            # 이 로그는 num_workers 수만큼 (여기서는 16번) 출력될 것입니다.
            logger.info(f"[Worker PID: {os.getpid()}] Initializing processor for this worker...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_name_or_path,
                trust_remote_code=True,
                use_fast=False  # train_vlm.py와 일관성 유지
            )
        # --- 수정 끝 ---

        row = self.rows[i]
        context = self._context_info(row)
        debug_id = row.get("uid", f"idx_{i}")

        # (이하 V4 로직과 동일)
        # 1. 이미지 로드 (PIL)
        try:
            images = self._load_images(row)
        except Exception as e:
            logger.error(f"Error in _load_images for index {i} ({debug_id}): {e}")
            raise e

        # 2. 텍스트 생성 (str)
        task = row["task"]
        prev = row.get("prev_desc", row.get("prev", ""))
        prev_status = row.get("prev_status", "UNCERTAIN")
        prompt_text = make_train_prompt(task=task, prev=prev, prev_status=prev_status)

        if "target_text" in row:
            target_text = row["target_text"]
        elif "api_output" in row:
            target_text = json.dumps(
                {"desc_1": row["api_output"].get("desc_1", ""),
                 "desc_2": row["api_output"].get("desc_2", ""),
                 "status": row["api_output"].get("status", "UNCERTAIN")},
                ensure_ascii=False,
            )
        else:
            raise KeyError(f"Missing 'target_text' or 'api_output' in sample ({context})")

        # 3. Chat Template 구성
        user_content = []
        if images:
            for _ in images:
                user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": prompt_text})

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": target_text}
        ]
        user_messages = [messages[0]]

        # 4. 템플릿 -> 문자열 변환
        try:
            full_prompt_string = self.processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception as e:
            logger.error(f"Failed to apply chat template (full) for sample {debug_id}: {e}")
            raise e

        # 5. Processor 호출 (이미지+텍스트 동시 전처리)
        try:
            enc = self.processor(
                text=full_prompt_string,
                images=images,
                return_tensors="pt",
                padding=False,
                truncation=False
            )
        except Exception as e:
            logger.error(f"Processor failed for sample {debug_id}: {e}")
            logger.error(f"Failed sample string (first 500 chars): {full_prompt_string[:500]}")
            raise e

        # 6. 수동 레이블 마스킹
        try:
            user_part_string = self.processor.tokenizer.apply_chat_template(
                user_messages, tokenize=False, add_generation_prompt=True
            )
            user_token_ids = self.processor.tokenizer(
                user_part_string, return_tensors="pt", add_special_tokens=True
            ).input_ids.squeeze(0)
            user_tokens_len = len(user_token_ids)
        except Exception as e:
            logger.error(f"Failed to tokenize user_part for masking on sample {debug_id}: {e}")
            user_tokens_len = 0

        input_ids = enc.input_ids.squeeze(0)
        labels = input_ids.clone()

        if user_tokens_len >= len(labels):
            logger.warning(
                f"Label/prompt length mismatch for {debug_id}. Full: {len(labels)}, User: {user_tokens_len}. Masking nothing.")
            user_tokens_len = 0

        labels[:user_tokens_len] = -100

        pixel_values = enc.pixel_values.squeeze(0)

        grid_thw = enc.get("image_grid_thw")
        if grid_thw is not None:
            grid_thw = grid_thw.squeeze(0)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thw,  # None일 수도 있음
        }


# ===== 2) Collator: 텐서를 받아 패딩만 수행 (V4와 동일, 빠름) =====
@dataclass
class DataCollatorVLM:
    """
    Dataset에서 이미 전처리된 텐서 딕셔너리를 받아와 배치로 묶고 패딩합니다.
    """
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

        input_ids = pad_sequence(
            [f["input_ids"] for f in features],
            batch_first=True,
            padding_value=self.pad_token_id
        )
        labels = pad_sequence(
            [f["labels"] for f in features],
            batch_first=True,
            padding_value=-100
        )
        attention_mask = (input_ids != self.pad_token_id).long()

        try:
            pixel_values = torch.stack([f["pixel_values"] for f in features])
        except Exception as e:
            shapes = [f["pixel_values"].shape for f in features]
            logger.error(f"Failed to stack pixel_values. Shapes: {shapes}. Error: {e}")
            raise e

        batch = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if features[0]["image_grid_thw"] is not None:
            try:
                concatenated_grid_thw = torch.cat([f["image_grid_thw"] for f in features], dim=0)
                batch["image_grid_thw"] = concatenated_grid_thw
                batch["grid_thw"] = concatenated_grid_thw
            except Exception as e:
                shapes = [f["image_grid_thw"].shape for f in features]
                logger.error(f"Failed to concatenate image_grid_thw. Shapes: {shapes}. Error: {e}")
                raise e

        return batch