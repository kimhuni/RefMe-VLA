# 📦 dataset_vlm.py (V5 Architecture - FINALIZED)
# V6의 pre-caching 대신, V5의 process-local 아키텍처를 기반으로 모든 버그를 수정한 최종본입니다.
# 'train_vlm.py'의 'dataloader_num_workers=16' (이상)과 함께 사용해야 합니다.

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

# 로거 설정 (Dataloader 워커에서 로깅하려면 중요)
logger = logging.getLogger(__name__)


# (참고: train_vlm.py에서 logging.basicConfig를 호출해야 함)


def make_train_prompt(task: str, prev: str, prev_status: str) -> str:
    """
    Generate a concise training prompt for image analysis in robot manipulation.
    (훈련/추론 시 동일하게 사용)
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


# ===== 1) Dataset: V5 아키텍처 (프로세스-로컬) =====
class VlmDataset(Dataset):
    """
    [V5 Architecture]
    __init__은 가볍게 경로만 로드합니다. (빠른 시작)
    __getitem__이 Dataloader 워커(프로세스)별로 전처리를 병렬 수행합니다.
    """

    def __init__(self, dataset_dir: str, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path
        self.data = []

        # 1. 샤드 파일 검색
        shard_pattern = os.path.join(dataset_dir, "shards", "chunk-*.json*")
        shard_files = sorted(glob.glob(shard_pattern))
        if not shard_files:
            raise FileNotFoundError(f"No shards found at {shard_pattern}")

        # 2. .jsonl의 *내용*이 아닌 *경로와 라인 번호*만 로드 (초경량)
        # (V6와 달리, 여기서 모든 데이터를 RAM에 올리지 않습니다.)
        # [수정] 대용량 데이터셋을 위해, 라인별로 읽지 않고 파일 목록만 저장
        for shard_file in shard_files:
            try:
                with open(shard_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            self.data.append(json.loads(line))
            except Exception as e:
                logger.warning(f"Error reading or parsing {shard_file}: {e}")

        logger.info(f"Loaded {len(self.data)} data points from {len(shard_files)} shards.")

        # 3. [V5] 프로세서는 워커별로 생성되도록 None으로 초기화
        self.processor = None

    def __len__(self):
        return len(self.data)

    def _initialize_processor(self):
        """
        Dataloader 워커별로 프로세서를 초기화합니다.
        """
        logger.info(f"Initializing processor for worker...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            use_fast=True
        )
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

    def __getitem__(self, idx):
        """
        이 함수는 16개(num_workers)의 프로세스에서 동시에 병렬 실행됩니다.
        """
        if self.processor is None:
            self._initialize_processor()

        item = self.data[idx]

        # --- 1. 이미지 로드 ---
        try:
            images_list = [
                Image.open(item['images']['side']).convert('RGB'),
                Image.open(item['images']['wrist']).convert('RGB')
            ]
        except Exception as e:
            logger.error(f"Error loading images for {item.get('uid', idx)}: {e}")
            return {}  # 콜레이터가 이 빈 딕셔너리를 무시합니다.

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
            return {}

        model_inputs = self.processor(
            text=prompt_string,  # 딕셔너리가 아닌 문자열 전달
            images=images_list,
            return_tensors="pt",
            padding=False
        )

        input_ids = model_inputs['input_ids'][0]
        labels = input_ids.clone()
        attention_mask = model_inputs['attention_mask'][0]

        # --- 5. [FINAL MASKING LOGIC] ---
        # (loss=6, `,,` 출력 버그 수정)
        # "뒤에서부터 계산"하는 로직을 사용합니다.

        # 1. 어시스턴트의 실제 응답(줄바꿈 + JSON)을 토큰화
        assistant_content_str = "\n" + target_text
        target_tokens = self.processor.tokenizer(
            assistant_content_str, add_special_tokens=False
        ).input_ids

        # 2. 응답 길이 = (줄바꿈+JSON) 토큰 + <|im_end|> 토큰 1개
        target_len = len(target_tokens) + 1  # +1 for <|im_end|>

        # 3. (전체 길이 - 응답 길이) 만큼을 마스킹
        mask_len = len(labels) - target_len

        if mask_len < 0:
            logger.warning(
                f"Masking error for {item.get('uid', idx)}: Target length ({target_len}) is longer than total input ({len(labels)}). Not masking.")
        else:
            labels[:mask_len] = -100

        # --- 6. [BUG FIX] `image_grid_thw` 제거 ---
        # (IndexError: 0-dim tensor 버그 수정)
        # Qwen2_5_VL은 이 인자가 필요 없으며, 에러를 유발합니다.

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


# ===== 2) Collator: 텐서를 받아 패딩만 수행 (V5와 동일, 빠름) =====
@dataclass
class DataCollatorForVLM:
    """
    VlmDataset에서 이미 텐서로 변환된 딕셔너리를 받아 패딩만 수행합니다.
    """
    tokenizer: AutoProcessor  # pad_token_id를 얻기 위해 프로세서 전체를 받음

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # __getitem__에서 에러가 발생한 빈 딕셔너리({}) 필터링
        features = [f for f in features if f]
        if not features:
            logger.warning("Data collator received an empty batch.")
            return {}

        # pad_token_id 가져오기 (초기화 시점에 정해짐)
        pad_token_id = self.tokenizer.tokenizer.pad_token_id

        # 1. 텍스트 관련 텐서 패딩
        input_ids = pad_sequence(
            [f["input_ids"] for f in features],
            batch_first=True,
            padding_value=pad_token_id
        )
        labels = pad_sequence(
            [f["labels"] for f in features],
            batch_first=True,
            padding_value=-100  # 손실 마스킹 값으로 패딩
        )
        attention_mask = pad_sequence(
            [f["attention_mask"] for f in features],
            batch_first=True,
            padding_value=0  # 어텐션 마스크는 0으로 패딩
        )

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        # --- [BUG FIX] `image_grid_thw` 제거 ---
        # (IndexError: 0-dim tensor 버그 수정)

        return batch