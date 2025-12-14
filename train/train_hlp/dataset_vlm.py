# 📦 dataset_loader.py (V7 - Pre-Caching + All Bug Fixes)
# '100s/it' 병목 현상을 해결하기 위해 V6의 '사전 캐싱' 아키텍처를 사용합니다.
# V5에서 수정한 모든 버그(마스킹, image_grid_thw)를 V6 로직에 적용한 최종본입니다.
# 경고: 훈련 시작 시 모든 데이터를 RAM에 캐시하므로, RAM 사용량이 매우 큽니다.

import json
import os
import glob
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor, PreTrainedTokenizer
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
        self.processor.tokenizer.padding_side = "left"

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
        [THE REAL FINAL MASKING LOGIC]
        "길이 추측" (+1) 대신 "내용 검색"을 사용하여 마스킹 버그를 수정합니다.
        """

        # --- 1. 이미지 로드 (PIL) ---
        try:
            images_list = [
                Image.open(item['images']['side']).convert('RGB'),
                Image.open(item['images']['wrist']).convert('RGB')
            ]
        except Exception as e:
            logger.error(f"Error loading images for {item.get('uid', idx)}: {e}")
            raise e

        # --- 2. 텍스트 생성 ---
        user_prompt_text = make_train_prompt(
            item['task'], item.get('prev_desc', ''), item.get('prev_status', 'NOT_DONE')
        )
        target_text = json.dumps(item['api_output'])  # 이것이 "순수 JSON"

        # --- 3. 채팅 템플릿 구성 ---
        messages = [
            {"role": "user",
             "content": [{"type": "image"}, {"type": "image"}, {"type": "text", "text": user_prompt_text}]},
            {"role": "assistant", "content": target_text}
        ]

        # --- 4. 토큰화 (String 변환 -> Processor 호출) ---
        try:
            prompt_string = self.processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        except Exception as e:
            logger.error(f"Error applying chat template for {item.get('uid', idx)}: {e}")
            raise e

        model_inputs = self.processor(
            text=prompt_string,
            images=images_list,
            return_tensors="pt",
            padding=False
        )

        input_ids = model_inputs['input_ids'].squeeze(0)
        labels = input_ids.clone()
        attention_mask = model_inputs['attention_mask'].squeeze(0)
        pixel_values = model_inputs['pixel_values'].squeeze(0)

        # --- 5. [수정됨] "내용 검색" 기반 마스킹 ---

        # (1) 우리가 예측해야 할 *순수 JSON* 토큰을 가져옵니다.
        target_tokens_ids = self.processor.tokenizer(target_text, add_special_tokens=False).input_ids

        # (2) input_ids를 리스트로 변환 (검색용)
        full_ids_list = input_ids.tolist()

        # (3) input_ids *끝*에서부터 *순수 JSON* 시퀀스를 검색합니다.
        start_index = -1
        # (단순하지만 확실한 역방향 검색)
        for i in range(len(full_ids_list) - len(target_tokens_ids), -1, -1):
            if full_ids_list[i: i + len(target_tokens_ids)] == target_tokens_ids:
                start_index = i
                break  # 찾았으면 중단

        if start_index != -1:
            # (4) JSON이 시작하는 지점(start_index) *앞*을 모두 -100으로 마스킹
            labels[:start_index] = -100
        else:
            # (5) [치명적] 정답 JSON을 input_ids에서 찾지 못함.
            #     이 샘플은 훈련하면 안 됨. (loss=0, garbage output의 원인)
            logger.error(
                f"CRITICAL MASKING ERROR: Target JSON not found in input_ids for {item.get('uid', idx)}. Masking all labels.")
            labels[:] = -100  # 이 샘플 전체를 마스킹

        # --- 6. `image_grid_thw` 복원 ---
        grid_thw = model_inputs.get("image_grid_thw")
        if grid_thw is not None:
            grid_thw = grid_thw.squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thw,  # [복원]
        }


# ===== 2) Collator: 텐서를 받아 패딩만 수행 (V6와 거의 동일) =====
@dataclass
class DataCollatorForVLM:
    """
    VlmDataset에서 이미 RAM에 캐시된 텐서 딕셔너리를 받아 패딩만 수행합니다.
    """
    # tokenizer: AutoProcessor
    def __init__(self, tokenizer, processor):
        self.processor = processor
        self.tokenizer = tokenizer

        # ★ 안전장치: 좌측 패딩 강제 확인
        assert getattr(self.tokenizer, "padding_side", None) == "left", \
            f"tokenizer.padding_side is {self.tokenizer.padding_side}, must be 'left' for Qwen2.5-VL + FA2"

    def _left_pad(self, tensors, pad_value):
        """
        Left-pad a list of 1D torch tensors to the same length.
        """
        max_len = max(t.size(0) for t in tensors)
        out = tensors[0].new_full((len(tensors), max_len), pad_value)
        for i, t in enumerate(tensors):
            out[i, -t.size(0):] = t  # right-align sequence => left padding
        return out

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

        features = [f for f in features if f]
        if not features:
            logger.warning("Data collator received an empty batch.")
            return {}

        pad_token_id = self.tokenizer.pad_token_id

        if not hasattr(self, "_pad_side_logged"):
            logger.info(f"[DataCollatorForVLM] padding_side={self.tokenizer.padding_side}")
            self._pad_side_logged = True

        # 1. 텍스트 관련 텐서 패딩
        input_ids = self._left_pad([f["input_ids"] for f in features], pad_token_id)
        labels = self._left_pad([f["labels"] for f in features], -100)
        attention_mask = self._left_pad([f["attention_mask"] for f in features], 0)

        # Transformers 버전에 맞게 수정
        pixel_values = torch.cat([f["pixel_values"] for f in features], dim=0)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,  # 이제 2D Tensor입니다.
        }
        ##################
        # 3. Grid 정보 처리 및 안전장치 추가
        if features[0].get("image_grid_thw") is not None:
            concatenated_grid_thw = torch.cat([f["image_grid_thw"] for f in features], dim=0)
            batch["image_grid_thw"] = concatenated_grid_thw
        else:
            batch["image_grid_thw"] = None

        return batch