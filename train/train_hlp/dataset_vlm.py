# 📦 dataset_loader.py (V7 - Pre-Caching + All Bug Fixes)
# '100s/it' 병목 현상을 해결하기 위해 V6의 '사전 캐싱' 아키텍처를 사용합니다.
# V5에서 수정한 모든 버그(마스킹, image_grid_thw)를 V6 로직에 적용한 최종본입니다.
# 경고: 훈련 시작 시 모든 데이터를 RAM에 캐시하므로, RAM 사용량이 매우 큽니다.

import json
import os
import glob
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor
from PIL import Image
import logging
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm  # 진행률 표시
from typing import List, Dict, Any

# 로거 설정
logger = logging.getLogger(__name__)


# (train_vlm.py에서 logging.basicConfig를 호출해야 함)


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


# ===== 1) Dataset: V6 아키텍처 (사전 캐싱) =====
class VlmDataset(Dataset):
    """
    [V7] __init__에서 모든 샘플을 미리 전처리하여 RAM에 보관합니다.
    __getitem__은 단지 리스트에서 텐서를 꺼내기만 합니다 (매우 빠름).
    """

    def __init__(self, dataset_dir: str, model_name_or_path: str):
        self.processor = None
        self.data = []

        # 1. 샤드 파일 검색
        shard_pattern = os.path.join(dataset_dir, "shards", "chunk-*.json*")
        shard_files = sorted(glob.glob(shard_pattern))
        if not shard_files:
            raise FileNotFoundError(f"No shards found at {shard_pattern}")

        # 2. .jsonl 파일의 모든 라인을 우선 RAM에 로드
        for shard_file in shard_files:
            try:
                with open(shard_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            self.data.append(json.loads(line))
            except Exception as e:
                logger.warning(f"Error reading or parsing {shard_file}: {e}")

        logger.info(f"Loaded {len(self.data)} data points. Starting pre-caching...")

        # 3. [핵심] 메인 프로세서에서 즉시 'processor'를 로드
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            use_fast=True
        )

        # 4. 모든 샘플을 미리 전처리하여 RAM 리스트에 저장
        self.processed_samples = []
        # tqdm을 사용해 캐싱 진행률 표시
        for i in tqdm(range(len(self.data)), desc="Pre-caching dataset into RAM"):
            try:
                # _process_one_sample이 텐서 딕셔너리를 반환
                self.processed_samples.append(
                    self._process_one_sample(self.data[i], i)
                )
            except Exception as e:
                logger.error(f"Failed to process sample {i} ({self.data[i].get('uid', 'N/A')}): {e}")

        logger.info(f"Caching complete. {len(self.processed_samples)} samples loaded into RAM.")
        # 원본 데이터는 메모리에서 해제
        del self.data

    def __len__(self):
        return len(self.processed_samples)

    def __getitem__(self, i: int):
        # [핵심] __getitem__은 RAM에 캐시된 딕셔너리를 즉시 반환 (초고속)
        return self.processed_samples[i]

    # --- 전처리를 위한 헬퍼 함수 ---
    def _process_one_sample(self, item: dict, idx: int) -> dict:
        """
        [BUG FIXED] V5의 버그 수정 로직을 V6 아키텍처에 적용합니다.
        """

        # --- 1. 이미지 로드 (PIL) ---
        # (V6) 캐싱을 위해 여기서 이미지를 로드하고 텐서로 변환합니다.
        try:
            images_list = [
                Image.open(item['images']['side']).convert('RGB'),
                Image.open(item['images']['wrist']).convert('RGB')
            ]
        except Exception as e:
            logger.error(f"Error loading images for {item.get('uid', idx)}: {e}")
            raise e  # 캐싱 중단

        # --- 2. 텍스트 생성 ---
        user_prompt_text = make_train_prompt(
            item['task'], item.get('prev_desc', ''), item.get('prev_status', 'NOT_DONE')
        )
        target_text = json.dumps(item['api_output'])

        # --- 3. 채팅 템플릿 구성 ---
        messages = [
            {"role": "user",
             "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": user_prompt_text}]},
            {"role": "assistant", "content": target_text}
        ]

        # --- 4. 토큰화 (String 변환 -> Processor 호출) ---
        # (AttributeError: 'dict' object has no attribute 'replace' 버그 수정)
        try:
            prompt_string = self.processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False  # 이미 assistant 턴 포함
            )
        except Exception as e:
            logger.error(f"Error applying chat template for {item.get('uid', idx)}: {e}")
            raise e

        model_inputs = self.processor(
            text=prompt_string,  # 딕셔너리가 아닌 문자열 전달
            images=images_list,
            return_tensors="pt",
            padding=False
        )

        # (V6) RAM 캐싱을 위해 텐서에서 배치 차원(0)을 제거합니다.
        input_ids = model_inputs['input_ids'].squeeze(0)
        labels = input_ids.clone()
        attention_mask = model_inputs['attention_mask'].squeeze(0)
        # (V6) pixel_values도 캐시합니다. (OOM 위험!)
        pixel_values = model_inputs['pixel_values'].squeeze(0)

        # --- 5. [FINAL MASKING LOGIC] ---
        # (loss=6, `,,` 출력 버그 수정)
        assistant_content_str = "\n" + target_text
        target_tokens = self.processor.tokenizer(
            assistant_content_str, add_special_tokens=False
        ).input_ids

        target_len = len(target_tokens) + 1  # +1 for <|im_end|>
        mask_len = len(labels) - target_len

        if mask_len < 0:
            logger.warning(
                f"Masking error for {item.get('uid', idx)}: Target length ({target_len}) is longer than total input ({len(labels)}). Not masking.")
        else:
            labels[:mask_len] = -100

        # 6. [BUG FIX] `image_grid_thw` "올바르게" 추가

        # model_inputs에서 텐서를 가져옴 (V6 로직)
        grid_thw = model_inputs.get("image_grid_thw")
        if grid_thw is not None:
            grid_thw = grid_thw.squeeze(0)  # 0-dim 에러 방지

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,  # RAM에 이미지 텐서 캐시
            "image_grid_thw": grid_thw,  # [추가] (None일 수도 있음)
        }


# ===== 2) Collator: 텐서를 받아 패딩만 수행 (V6와 거의 동일) =====
@dataclass
class DataCollatorForVLM:
    """
    VlmDataset에서 이미 RAM에 캐시된 텐서 딕셔너리를 받아 패딩만 수행합니다.
    """
    tokenizer: AutoProcessor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

        features = [f for f in features if f]
        if not features:
            logger.warning("Data collator received an empty batch.")
            return {}

        pad_token_id = self.tokenizer.pad_token_id

        # 1. 텍스트 관련 텐서 패딩
        input_ids = pad_sequence(
            [f["input_ids"] for f in features], batch_first=True, padding_value=pad_token_id
        )
        labels = pad_sequence(
            [f["labels"] for f in features], batch_first=True, padding_value=-100
        )
        attention_mask = pad_sequence(
            [f["attention_mask"] for f in features], batch_first=True, padding_value=0
        )

        # 2. 이미지 텐서 스택
        try:
            pixel_values = torch.stack([f["pixel_values"] for f in features])
        except Exception as e:
            shapes = [f["pixel_values"].shape for f in features]
            logger.error(f"Failed to stack pixel_values. Shapes: {shapes}. Error: {e}")
            # V6는 pixel_values를 캐시하므로, 여기서 크기가 다르면 치명적 에러임
            raise e

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,  # [V6] pixel_values도 전달
        }

        if features[0]["image_grid_thw"] is not None:
            try:
                concatenated_grid_thw = torch.cat([f["image_grid_thw"] for f in features], dim=0)
                batch["image_grid_thw"] = concatenated_grid_thw
            except Exception as e:
                shapes = [f["image_grid_thw"].shape for f in features if f["image_grid_thw"] is not None]
                logger.error(f"Failed to concatenate image_grid_thw. Shapes: {shapes}. Error: {e}")
                # 이 경우, None으로 두어 모델이 처리하도록 함
                batch["image_grid_thw"] = None

        else:
            batch["image_grid_thw"] = None  # 명시적으로 None 전달

        # --- [BUG FIX] `image_grid_thw` 제거 ---

        return batch