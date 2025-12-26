# helm_dataset.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from transformers import AutoProcessor

HLP_HEADER_1 = (
    "Role: High-Level Robot Policy.\n"
    "Given the table view image and Previous_Memory, update the memory and choose the next atomic command.\n"
    "- Only advance Progress when the event has occurred in the current frame.\n"
    "- World_State should be concise and persistent (use None if no state).\n"
    "- Command should be either the task command or \"done\" if finished.\n"
)

HLP_HEADER_2 = (
    "Role: High-Level Robot Policy.\n"
    "Given the two images and Previous_Memory, update the memory and choose the next atomic command.\n"
    "- Only advance Progress when the event has occurred in the current frame.\n"
    "- World_State should be concise and persistent (use None if no state).\n"
    "- Command should be either the task command or \"done\" if finished.\n"
)

def _drop_lines_with_prefix(text: str, prefixes: Tuple[str, ...]) -> str:
    lines = text.splitlines()
    kept = []
    for ln in lines:
        s = ln.strip()
        if any(s.startswith(p) for p in prefixes):
            continue
        kept.append(ln)
    return "\n".join(kept).strip()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"JSON parse error: {path} line {ln}: {e}")
    return rows


def _get_user_assistant(row: Dict[str, Any]) -> Tuple[str, str]:
    conv = row.get("conversations", [])
    if not isinstance(conv, list):
        raise ValueError("row['conversations'] must be a list")

    user_text, asst_text = "", ""
    for m in conv:
        if not isinstance(m, dict):
            continue
        if m.get("from") == "user":
            user_text = str(m.get("value", ""))
        elif m.get("from") == "assistant":
            asst_text = str(m.get("value", ""))
    if not user_text or not asst_text:
        raise ValueError("Missing user/assistant in conversations")
    return user_text, asst_text


def _load_images(row: Dict[str, Any], num_image: int) -> List[Image.Image]:
    """
    Load 1 or 2 images from the row.

    - num_image == 1: requires 'table'
    - num_image == 2: requires 'table' and 'wrist'

    Returns a list of PIL Images whose length equals num_image.
    """
    images = row.get("images", {})
    if not isinstance(images, dict):
        raise ValueError("row['images'] must be a dict")

    p_table = images.get("table", None)
    p_wrist = images.get("wrist", None)

    if not p_table:
        raise ValueError(f"images must include 'table'. got keys={list(images.keys())}")

    img0 = Image.open(p_table).convert("RGB")

    if int(num_image) == 1:
        return [img0]

    if not p_wrist:
        raise ValueError(f"num_image=2 requires 'wrist'. got keys={list(images.keys())}")

    img1 = Image.open(p_wrist).convert("RGB")
    return [img0, img1]


def _find_subsequence(haystack: List[int], needle: List[int]) -> int:
    """Return start index of needle in haystack, or -1 if not found."""
    if not needle or len(needle) > len(haystack):
        return -1
    L = len(needle)
    for i in range(len(haystack) - L, -1, -1):  # reverse search tends to be more stable here
        if haystack[i : i + L] == needle:
            return i
    return -1


class HelmJsonlDataset(Dataset):
    """
    Loads HeLM JSONL rows and produces Qwen2.5-VL training tensors.

    Key point:
      - user prompt is row.conversations[user]
      - assistant target is YAML string row.conversations[assistant]
      - images are [table, wrist]
      - labels are masked to compute loss only on assistant content
    """

    def __init__(self, jsonl_path: str, num_image: int, model_name_or_path: str, add_hlp_header: bool = True, drop_frame_line: bool = True, drop_images_line: bool = True, drop_return_yaml_line: bool = True):
        self.jsonl_path = Path(jsonl_path)
        self.rows = _read_jsonl(self.jsonl_path)

        self.num_image = num_image

        self.add_hlp_header = add_hlp_header
        self.drop_frame_line = drop_frame_line
        self.drop_images_line = drop_images_line
        self.drop_return_yaml_line = drop_return_yaml_line

        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            use_fast=True,
        )
        # for causal LM padding
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token is None:
            # safe default
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        user_text, target_text = _get_user_assistant(row)

        # Optionally transform the user prompt on-the-fly (no need to regenerate JSONL).
        prefixes = []
        if self.drop_frame_line:
            prefixes.append("Frame:")
        if self.drop_images_line:
            prefixes.append("Images:")
        if self.drop_return_yaml_line:
            prefixes.append("Return YAML")
        if prefixes:
            user_text = _drop_lines_with_prefix(user_text, tuple(prefixes))

        # Add canonical header BEFORE building messages so it is actually included in the chat template.
        if self.add_hlp_header:
            HLP_HEADER = HLP_HEADER_1 if int(self.num_image) == 1 else HLP_HEADER_2
            user_text = HLP_HEADER + "\n\n" + user_text.strip()

        images = _load_images(row, int(self.num_image))

        if int(self.num_image) == 1:
            # Qwen2.5-VL style messages: 1 image + text
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": user_text},
                    ],
                },
                {"role": "assistant", "content": target_text},
            ]
        else:
            # Qwen2.5-VL style messages: 2 images + text
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "text", "text": user_text},
                    ],
                },
                {"role": "assistant", "content": target_text},
            ]

        prompt = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        model_inputs = self.processor(
            text=prompt,
            images=images,
            return_tensors="pt",
            padding=False,
        )

        input_ids = model_inputs["input_ids"].squeeze(0)
        attention_mask = model_inputs["attention_mask"].squeeze(0)
        pixel_values = model_inputs["pixel_values"].squeeze(0)

        labels = input_ids.clone()

        # mask: find target tokens within full input ids
        tgt_ids = self.processor.tokenizer(target_text, add_special_tokens=False).input_ids
        full = input_ids.tolist()

        start = _find_subsequence(full, tgt_ids)
        if start == -1:
            # common variant: template inserts a newline before assistant content
            tgt_ids2 = self.processor.tokenizer("\n" + target_text, add_special_tokens=False).input_ids
            start = _find_subsequence(full, tgt_ids2)

        if start == -1:
            # safest: no loss (avoid training on wrong span)
            labels[:] = -100
            print(f"⚠️ Masking Failed! Target not found in input. Target snippet: {target_text[:30]}...")
        else:
            labels[:start] = -100

        out: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }

        grid = model_inputs.get("image_grid_thw", None)
        if grid is not None:
            grid = grid.squeeze(0)
            if grid.ndim == 1:
                grid = grid.unsqueeze(0)  # (3,) -> (1,3)
            out["image_grid_thw"] = grid


        return out


@dataclass
class HelmDataCollator:
    pad_id: int = 0

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = pad_sequence([f["input_ids"] for f in features], batch_first=True, padding_value=self.pad_id)
        attention_mask = pad_sequence([f["attention_mask"] for f in features], batch_first=True, padding_value=0)
        labels = pad_sequence([f["labels"] for f in features], batch_first=True, padding_value=-100)

        # Qwen2.5-VL visual encoder expects images concatenated across the batch:
        # pixel_values: (num_images_total, C, H, W) (or similar)
        # image_grid_thw: (num_images_total, 3)
        batch: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": torch.cat([f["pixel_values"] for f in features], dim=0),
        }

        if "image_grid_thw" in features[0]:
            batch["image_grid_thw"] = torch.cat([f["image_grid_thw"] for f in features], dim=0)
        # If any sample lacks grid info, disable it for the whole batch.
        if any(("image_grid_thw" not in f) or (f.get("image_grid_thw") is None) for f in features):
            batch["image_grid_thw"] = None

        return batch