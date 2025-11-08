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
                "content": target_text  # processor.tokenizer가 \n + target_text + <|im_end|>로 변환
            }
        ]

        # 4. Convert the chat dictionary into a single string
        try:
            prompt_string_with_placeholders = self.processor.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        except Exception as e:
            logger.error(f"Error applying chat template at index {idx}: {e}")
            logger.error(f"Messages: {messages}")
            return {}

        # 5. Processor converts the *string* + images into model inputs
        model_inputs = self.processor(
            text=prompt_string_with_placeholders,
            images=images_list,
            return_tensors="pt",
            padding=False
        )

        # Remove batch dimension (0)
        input_ids = model_inputs['input_ids'][0]
        attention_mask = model_inputs['attention_mask'][0]

        # 6. [FINAL MASKING LOGIC]
        labels = input_ids.clone()

        # [FIX] The template adds a newline: 'assistant\n{JSON...}'
        # We must tokenize exactly what the assistant content will be.
        assistant_content_str = "\n" + target_text

        target_tokens = self.processor.tokenizer(
            assistant_content_str, add_special_tokens=False
        ).input_ids

        # The total length to *keep* (unmask) is:
        # The length of (\n + JSON) tokens + 1 (for the <|im_end|> token)
        target_len = len(target_tokens) + 1

        # Mask everything *except* for the last 'target_len' tokens
        mask_len = len(labels) - target_len
        labels[:mask_len] = -100

        # [Debugging code] (This is still useful)
        if idx == 0:
            print("--- 🐛 DEBUGGING FINAL MASKING ---")
            target_tokens_debug = [l for l in labels.tolist() if l != -100]
            decoded_target = self.processor.tokenizer.decode(target_tokens_debug)
            print(f"Decoded Target (Should be JSON): {decoded_target}")
            print(f"Original Target (For comparison): {target_text}")
        # --- [디버깅 코드 끝] ---

        return_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

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