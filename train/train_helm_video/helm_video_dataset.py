# train/train_video/video_dataset.py
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoProcessor


def read_jsonl(path_or_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    p = Path(path_or_dir)
    rows: List[Dict[str, Any]] = []
    if p.is_file():
        files = [p]
    else:
        files = sorted(p.rglob("*.jsonl"))

    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line: continue
                try:
                    rows.append(json.loads(line))
                except Exception as e:
                    print(f"[Warning] Parse error {fp}:{ln} - {e}")
    return rows


def _load_images(item: Dict[str, Any]) -> List[Image.Image]:
    """
    Load images from a list of paths.
    item['images'] can be a list of strings.
    """
    image_paths = item.get("images", [])
    if not isinstance(image_paths, list):
        # 만약 문자열 하나만 있으면 리스트로 감쌈
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        else:
            raise ValueError(f"Expected list of image paths, got {type(image_paths)}")

    imgs: List[Image.Image] = []
    for p in image_paths:
        try:
            # Qwen2.5-VL expects RGB
            imgs.append(Image.open(p).convert("RGB"))
        except Exception as e:
            raise RuntimeError(f"Failed to load image {p}: {e}")
    return imgs


def _derive_label(item: Dict[str, Any]) -> str:
    """
    데이터에 'label' 키가 없으면 mode와 gt 정보를 이용해 추론
    """
    if "label" in item:
        return str(item["label"])

    mode = item.get("mode", "UNKNOWN")
    if mode == "DETECT":
        # gt_text or gt_yaml parse
        # 간단히 gt_text에 'Event_Detected: true'가 있는지 등으로 판단하거나
        # build 단계에서 label을 넣어주는 것이 가장 좋음.
        # 여기서는 임시로 'detect'로 통합하거나, 내용 파싱
        gt = str(item.get("gt_text", "")).lower()
        if "event_detected: true" in gt:
            return "detect_pos"
        else:
            return "detect_neg"
    elif mode == "UPDATE":
        return "update"

    return "UNKNOWN"


def _find_subsequence(haystack: List[int], needle: List[int]) -> int:
    if len(needle) == 0 or len(needle) > len(haystack):
        return -1
    for i in range(len(haystack) - len(needle), -1, -1):
        if haystack[i: i + len(needle)] == needle:
            return i
    return -1


@dataclass
class VideoDatasetConfig:
    jsonl_path: str
    model_name_or_path: str
    min_pixels: int = 256 * 28 * 28  # Qwen2.5-VL defaults
    max_pixels: int = 1280 * 28 * 28
    padding_side: str = "left"


class VideoJsonlDataset(Dataset):
    def __init__(self, cfg: VideoDatasetConfig):
        super().__init__()
        self.cfg = cfg
        self.rows = read_jsonl(cfg.jsonl_path)

        self.processor = AutoProcessor.from_pretrained(
            cfg.model_name_or_path,
            trust_remote_code=True,
            min_pixels=cfg.min_pixels,
            max_pixels=cfg.max_pixels,
        )
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer:
            self.processor.tokenizer.padding_side = cfg.padding_side

        # Pools for sampling
        self.pools: Dict[str, List[int]] = {}
        for i, r in enumerate(self.rows):
            lab = _derive_label(r)
            self.pools.setdefault(lab, []).append(i)

    def __len__(self) -> int:
        return len(self.rows)

    def get_pools(self) -> Dict[str, List[int]]:
        return {k: list(v) for k, v in self.pools.items()}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.rows[idx]

        user_prompt = str(item.get("user_prompt", ""))
        target_text = str(item.get("gt_text", ""))

        # 1. Load Images (Variable Length)
        imgs = _load_images(item)
        num_images = len(imgs)

        # 2. Construct Message with Dynamic Image Tokens
        # Qwen2.5-VL expects each image to have a corresponding {"type": "image"} content
        user_content = [{"type": "image"} for _ in range(num_images)]
        user_content.append({"type": "text", "text": user_prompt})

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": target_text},
        ]

        # 3. Apply Template
        prompt_string = self.processor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

        # 4. Process Inputs
        # processor handles list of images automatically
        model_inputs = self.processor(
            text=prompt_string,
            images=imgs,
            return_tensors="pt",
            padding=False
        )

        input_ids = model_inputs["input_ids"].squeeze(0)
        attention_mask = model_inputs["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        # 5. Masking (Supervise only Output)
        target_ids = self.processor.tokenizer(
            target_text, add_special_tokens=False
        ).input_ids

        start_idx = _find_subsequence(input_ids.tolist(), target_ids)
        if start_idx >= 0:
            labels[:start_idx] = -100
        else:
            labels[:] = -100  # mask all if target not found

        # 6. Handle Pixel Values
        # Qwen2.5-VL returns flattened pixel_values for all images in the conversation
        pixel_values = model_inputs["pixel_values"].squeeze(0)

        # image_grid_thw: (N_total_images, 3)
        grid_thw = model_inputs.get("image_grid_thw", None)
        if grid_thw is not None:
            # 1. 0차원(스칼라) 방지
            if grid_thw.ndim == 0:
                raise ValueError("image_grid_thw became 0-d tensor.")

            # 2. Batch 차원이 포함되어 오면 제거 (1, N, 3) -> (N, 3)
            if grid_thw.ndim >= 3:
                grid_thw = grid_thw.squeeze(0)

            # 3. 이미지가 1장이라서 (3,)으로 온 경우 (1, 3)으로 확장
            if grid_thw.ndim == 1:
                if grid_thw.numel() == 3:
                    grid_thw = grid_thw.unsqueeze(0)
                else:
                    raise ValueError(f"Unexpected shape: {grid_thw.shape}")

            # 4. 최종 확인: 항상 (N, 3) 형태의 2차원 텐서여야 함
            if grid_thw.ndim != 2 or grid_thw.size(-1) != 3:
                # 강제로 (이미지수, 3)으로 재구성 시도
                grid_thw = grid_thw.view(-1, 3)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thw,
        }


# DataCollator는 기존 것을 거의 그대로 쓰되, import 경로만 주의
class DataCollatorForVideoBaseline:
    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer

    def _left_pad_1d(self, tensors, pad_value):
        max_len = max(t.size(0) for t in tensors)
        out = tensors[0].new_full((len(tensors), max_len), pad_value)
        for i, t in enumerate(tensors):
            out[i, -t.size(0):] = t
        return out

    def __call__(self, features):
        features = [f for f in features if f is not None]
        if not features: return {}

        pad_id = self.tokenizer.pad_token_id
        input_ids = self._left_pad_1d([f["input_ids"] for f in features], pad_id)
        attention_mask = self._left_pad_1d([f["attention_mask"] for f in features], 0)
        labels = self._left_pad_1d([f["labels"] for f in features], -100)

        # pixel_values는 이미 (N, C, H, W) 형태이므로 그대로 합침
        pixel_values = torch.cat([f["pixel_values"] for f in features], dim=0)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }

        # image_grid_thw 병합 로직 강화
        if features[0].get("image_grid_thw") is not None:
            grids = []
            for f in features:
                g = f["image_grid_thw"]
                # 각 샘플의 grid가 (N, 3)인지 확인하고 아니면 보정
                if g.ndim == 1:
                    g = g.unsqueeze(0)
                grids.append(g)

            # 이제 모든 텐서가 2차원이므로 cat이 성공함
            batch["image_grid_thw"] = torch.cat(grids, dim=0)

        return batch