# 📦 dataset_vlm.py
import json
import os  # [Added]
import glob  # [Added]
import torch
from torch.utils.data import Dataset
from transformers import AutoProcessor
from PIL import Image
import logging

# Setup logging (worker-process-safe)
logger = logging.getLogger(__name__)


def make_train_prompt(task: str, prev: str, prev_status: str) -> str:
    """
    Generate a concise training prompt for image analysis in robot manipulation.
    (Function provided in the requirements)
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


class VlmDataset(Dataset):
    """
    Dataset class implementing the 'v5 architecture'.
    __init__ is lightweight; __getitem__ handles all preprocessing (image loading, tokenizing, masking).
    """

    # [Modified] __init__ signature changed (jsonl_path -> dataset_dir)
    def __init__(self, dataset_dir: str, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path
        self.data = []

        # [Modified] Logic to find sharded .jsonl files
        # User mentioned "{dataset_dir}/shards/chunk-000.json" format.
        # Use "chunk-*.json*" to find both .jsonl and .json files.
        shard_pattern = os.path.join(dataset_dir, "shards", "chunk-*.json*")
        logger.info(f"Looking for dataset shards at: {shard_pattern}")

        # Use glob to get all matching files (sorted)
        shard_files = sorted(glob.glob(shard_pattern))

        if not shard_files:
            logger.error(f"No dataset shards found at {shard_pattern}. "
                         f"Please check --dataset_dir path.")
            raise FileNotFoundError(f"No shards found at {shard_pattern}")

        logger.info(f"Found {len(shard_files)} shards. Loading...")

        # Iterate over all shard files and append data
        for shard_file in shard_files:
            try:
                with open(shard_file, 'r', encoding='utf-8') as f:
                    # Assume .jsonl format (one JSON object per line)
                    for line in f:
                        if line.strip():  # Skip empty lines
                            self.data.append(json.loads(line))
            except Exception as e:
                logger.warning(f"Error reading or parsing {shard_file}: {e}")

        logger.info(f"Successfully loaded {len(self.data)} total data points from {len(shard_files)} shards.")

        # [v5 Architecture] Initialize processor to None (process-local)
        self.processor = None

    def __len__(self):
        return len(self.data)

    def _initialize_processor(self):
        """
        [v5 Architecture] Initializes the processor for each Dataloader worker.
        """
        logger.info(f"Initializing processor for worker...")
        self.processor = AutoProcessor.from_pretrained(
            self.model_name_or_path,
            trust_remote_code=True,
            use_fast=True  # [Requirement] Use fast Rust tokenizer
        )
        # Set pad token to EOS token if it's not defined (for Collator)
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

    def __getitem__(self, idx):
        # [v5 Architecture] Create processor if it doesn't exist for this worker
        if self.processor is None:
            self._initialize_processor()

        item = self.data[idx]

        # 1. Load images
        try:
            side_image = Image.open(item['images']['side']).convert('RGB')
            wrist_image = Image.open(item['images']['wrist']).convert('RGB')
            images_list = [side_image, wrist_image]
        except Exception as e:
            logger.error(f"Error loading images for index {idx}, path: {item['images']}. Error: {e}")
            # Return empty dict, DataCollator will filter this
            return {}

            # 2. Configure prompt and target [Core Preprocessing Logic]
        user_prompt_text = make_train_prompt(
            item['task'], item['prev_desc'], item['prev_status']
        )
        target_text = json.dumps(item['api_output'])

        # 3. VLM Input Template (Qwen-VL Chat)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},  # side image
                    {"type": "image"},  # wrist image
                    {"type": "text", "text": user_prompt_text}
                ]
            },
            {
                "role": "assistant",
                "content": target_text
            }
        ]

        # 4. [Loss Masking] Calculate length of the User turn
        # Create string for User turn + Assistant start prompt
        user_messages = [messages[0]]
        user_part_string = self.processor.tokenizer.apply_chat_template(
            user_messages,
            tokenize=False,
            add_generation_prompt=True  # Adds "assistant\n" prompt
        )

        # Tokenize to get the actual token length
        # (Note: add_special_tokens=False to avoid double <bos>)
        user_part_tokens = self.processor.tokenizer(
            user_part_string, add_special_tokens=False
        ).input_ids

        user_part_len = len(user_part_tokens)

        # 5. Apply full template and tokenize (using Processor)
        # apply_chat_template internally handles <image> tokens
        prompt_with_images = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False  # Assistant turn is already included
        )

        # Processor converts text + images into model inputs
        model_inputs = self.processor(
            text=prompt_with_images,
            images=images_list,
            return_tensors="pt",
            padding=False  # Collator handles padding
        )

        # Remove batch dimension (0)
        input_ids = model_inputs['input_ids'][0]
        attention_mask = model_inputs['attention_mask'][0]

        # 6. [Loss Masking] Create labels and apply mask
        labels = input_ids.clone()
        labels[:user_part_len] = -100  # [Requirement] Mask User + Assistant prompt

        # 7. [Qwen-VL Special Args] Handle image_grid_thw
        #image_grid_thw = model_inputs.get('image_grid_thw')
        #if image_grid_thw is not None:
        #    image_grid_thw = image_grid_thw[0]

        return_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

        if idx == 0:  # 첫 번째 데이터만 확인
            print("--- 🐛 DEBUGGING DATA MASKING ---")

            # 마스킹(-100)이 안 된 라벨 토큰만 필터링
            target_tokens = [l for l in labels.tolist() if l != -100]

            # 타겟 토큰 디코딩
            decoded_target = self.processor.tokenizer.decode(target_tokens)
            print(f"Decoded Target (Should be JSON): {decoded_target}")

            # 원본 JSON (비교 대상)
            original_target = json.dumps(item['api_output'])
            print(f"Original Target (For comparison): {original_target}")

        return return_dict


class DataCollatorForVLM:
    """
    [v5 Architecture] Data collator that only performs padding on tensorized samples.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, features):
        # Filter out failed samples (which returned {} in __getitem__)
        features = [f for f in features if f]
        if not features:
            return {}

        # 1. Pad text-related tensors
        input_ids = [f['input_ids'] for f in features]
        attention_mask = [f['attention_mask'] for f in features]
        labels = [f['labels'] for f in features]

        # Right-padding
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=-100  # Pad with loss mask value
        )

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

        # 2. [Qwen-VL Special Args] Pad (torch.cat)
        #if 'image_grid_thw' in features[0]:
        #    image_grids = [f['image_grid_thw'] for f in features]
        #    # Combine into (num_images_in_batch, 3, grid_H, grid_W)
        #    batch['image_grid_thw'] = torch.cat(image_grids, dim=0)

        return batch