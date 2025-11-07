# dataset_loader.py (REWRITE v6 - PRE-CACHING FIX)
import json
from typing import Dict, Any, List
from dataclasses import dataclass
import glob
import logging
import os
from tqdm import tqdm  # <-- 진행률 표시를 위해 tqdm 추가

import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoProcessor
from torch.nn.utils.rnn import pad_sequence

# 로거 설정
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CAM_FALLBACK_ORDER = ["side", "wrist"]


def make_train_prompt(task: str, prev: str, prev_status: str) -> str:
    # (V5와 동일)
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


# ===== 1) Dataset: [수정] 모든 샘플을 RAM으로 사전 캐싱 =====
class VLMJSONDataset(Dataset):
    """
    [V6] __init__에서 모든 샘플을 미리 전처리하여 RAM에 보관합니다.
    __getitem__은 단지 리스트에서 텐서를 꺼내기만 합니다 (매우 빠름).
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

        self.image_key = image_key

        # [핵심 수정] 메인 프로세서에서 전용 'processor'를 즉시 로드
        logger.info(f"Loading processor '{model_name_or_path}' for pre-caching...")
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            use_fast=True  # <-- 빠른 토크나이저 사용
        )

        logger.info(f"Pre-processing and caching {len(self.rows)} samples into RAM...")
        # self.processed_samples 리스트에 전처리된 텐서를 저장
        self.processed_samples = []
        for i in tqdm(range(len(self.rows)), desc="Caching dataset"):
            try:
                # _process_one_sample이 텐서 딕셔너리를 반환
                self.processed_samples.append(
                    self._process_one_sample(self.rows[i], processor, i)
                )
            except Exception as e:
                logger.error(f"Failed to process sample {i} ({self.rows[i].get('uid', 'N/A')}): {e}")

        logger.info(f"Caching complete. {len(self.processed_samples)} samples loaded into RAM.")
        print(f"[VLMJSONDataset] files={len(paths)} samples={len(self.processed_samples)}")

    def __len__(self):
        return len(self.processed_samples)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        # [핵심 수정] __getitem__은 RAM에 캐시된 딕셔너리를 반환 (초고속)
        return self.processed_samples[i]

    # --- 전처리를 위한 헬퍼 함수들 ---

    def _process_one_sample(self, row: Dict[str, Any], processor: AutoProcessor, i: int) -> Dict[str, Any]:
        """
        V5의 __getitem__ 로직을 그대로 가져와 단일 샘플을 전처리합니다.
        """
        context = self._context_info(row)
        debug_id = row.get("uid", f"idx_{i}")

        # 1. 이미지 로드 (PIL)
        images = self._load_images(row)

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
        full_prompt_string = processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # 5. Processor 호출 (이미지+텍스트 동시 전처리)
        enc = processor(
            text=full_prompt_string,
            images=images,
            return_tensors="pt",
            padding=False,
            truncation=False
        )

        # 6. 수동 레이블 마스킹
        user_part_string = processor.tokenizer.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True
        )
        user_token_ids = processor.tokenizer(
            user_part_string, return_tensors="pt", add_special_tokens=True
        ).input_ids.squeeze(0)
        user_tokens_len = len(user_token_ids)

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

    def _context_info(self, row: Dict[str, Any]) -> str:
        # (V5와 동일)
        parts = []
        for k in ["uid", "chunk_id", "episode_id", "timestamp_ms"]:
            if k in row:
                parts.append(f"{k}={row[k]}")
        return ", ".join(parts) if parts else "no context"

    def _load_images(self, row: Dict[str, Any]) -> List[Image.Image]:
        # (V5와 동일)
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


# ===== 2) Collator: 텐서를 받아 패딩만 수행 (V5와 동일, 빠름) =====
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