# helm_dataset_v3.py
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
        if not files:
            raise FileNotFoundError(f"No .jsonl files found under: {p}")

    for fp in files:
        with fp.open("r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception as e:
                    raise RuntimeError(f"JSON parse error: {fp} line {ln}: {e}")
    return rows


def count_labels(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in rows:
        k = str(r.get("label", "UNKNOWN"))
        out[k] = out.get(k, 0) + 1
    return out


def _load_images(item: Dict[str, Any], num_images: int) -> List[Image.Image]:
    imgs: List[Image.Image] = []
    images = item.get("images", {})
    if not isinstance(images, dict):
        raise ValueError("row['images'] must be a dict")

    table = images.get("table", None)
    if table is None:
        raise ValueError("row['images']['table'] is missing")
    imgs.append(Image.open(table).convert("RGB"))

    if num_images == 2:
        wrist = images.get("wrist", None)
        if wrist is None:
            raise ValueError("num_images=2 but row['images']['wrist'] is missing")
        imgs.append(Image.open(wrist).convert("RGB"))
    return imgs


def _find_subsequence(haystack: List[int], needle: List[int]) -> int:
    """Return start index of needle in haystack scanning from end; -1 if not found."""
    if len(needle) == 0 or len(needle) > len(haystack):
        return -1
    for i in range(len(haystack) - len(needle), -1, -1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return -1


@dataclass
class V3DatasetConfig:
    jsonl_path: str
    model_name_or_path: str
    num_images: int = 1
    trust_remote_code: bool = True
    use_fast: bool = True
    # qwen2.5-vl + FA2에서는 left padding 권장
    padding_side: str = "left"


class HelmJsonlDatasetV3(Dataset):
    """
    v3 row schema:
      - user_prompt: str
      - gt_text: str (YAML)
      - images: {"table":..., "wrist":...}
      - label: detect_pos/detect_neg/update_intra/update_transition
      - mode: DETECT/UPDATE
    """
    def __init__(self, cfg: V3DatasetConfig):
        super().__init__()
        self.cfg = cfg
        self.rows = read_jsonl(cfg.jsonl_path)

        self.processor = AutoProcessor.from_pretrained(
            cfg.model_name_or_path,
            trust_remote_code=cfg.trust_remote_code,
            use_fast=cfg.use_fast,
        )
        # tokenizer side
        if hasattr(self.processor, "tokenizer") and self.processor.tokenizer is not None:
            self.processor.tokenizer.padding_side = cfg.padding_side

        # pools for mixed sampling
        self.pools: Dict[str, List[int]] = {}
        for i, r in enumerate(self.rows):
            lab = str(r.get("label", "UNKNOWN"))
            self.pools.setdefault(lab, []).append(i)

    def __len__(self) -> int:
        return len(self.rows)

    def get_pools(self) -> Dict[str, List[int]]:
        return {k: list(v) for k, v in self.pools.items()}

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.rows[idx]

        user_prompt = str(item.get("user_prompt", ""))
        target_text = str(item.get("gt_text", ""))

        imgs = _load_images(item, self.cfg.num_images)

        # Qwen2.5-VL chat messages (images are separate tokens, not embedded in text)
        # user_prompt에 "Images: <image_table>" 같은 줄이 있어도 상관은 없지만,
        # 실제 이미지 토큰은 content에 {"type":"image"}로 들어가야 합니다.
        user_content = []
        for _ in range(self.cfg.num_images):
            user_content.append({"type": "image"})
        user_content.append({"type": "text", "text": user_prompt})

        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": target_text},
        ]

        prompt_string = self.processor.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        model_inputs = self.processor(
            text=prompt_string,
            images=imgs,
            return_tensors="pt",
            padding=False,
        )

        input_ids = model_inputs["input_ids"].squeeze(0)
        attention_mask = model_inputs["attention_mask"].squeeze(0)

        labels = input_ids.clone()

        # mask: only supervise the target_text part
        target_ids = self.processor.tokenizer(
            target_text, add_special_tokens=False
        ).input_ids

        start_idx = _find_subsequence(input_ids.tolist(), target_ids)
        if start_idx >= 0:
            labels[:start_idx] = -100
        else:
            # fail-safe: ignore this sample
            labels[:] = -100

        # pixel_values: (1, N, C, H, W) -> squeeze(0) => (N, C, H, W)
        pixel_values = model_inputs["pixel_values"].squeeze(0)

        # image_grid_thw: (1, N, 3) -> (N, 3), later collator concatenates => (B*N, 3)
        grid_thw = model_inputs.get("image_grid_thw", None)
        if grid_thw is not None:
            # 예상 shape: (1, num_images, 3) or (num_images, 3) or (1,3) or (3,)
            # 먼저 batch 차원(맨 앞)이 있으면 제거
            if grid_thw.ndim >= 3:
                grid_thw = grid_thw.squeeze(0)  # (num_images, 3)
            elif grid_thw.ndim == 2:
                # (num_images, 3) or (1,3) OK
                pass
            elif grid_thw.ndim == 1:
                # (3,) -> (1,3)
                if grid_thw.numel() == 3:
                    grid_thw = grid_thw.unsqueeze(0)
                else:
                    raise ValueError(f"Unexpected image_grid_thw shape: {tuple(grid_thw.shape)}")
            else:
                # 0-d는 절대 허용하면 안 됨
                raise ValueError("image_grid_thw became 0-d tensor (scalar).")

            # 최종 안전장치: (num_images, 3) 강제
            if not (grid_thw.ndim == 2 and grid_thw.size(-1) == 3):
                raise ValueError(f"Bad image_grid_thw final shape: {tuple(grid_thw.shape)}")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thw,
        }


class DataCollatorForQwenVL:
    def __init__(self, processor: AutoProcessor):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        assert self.tokenizer is not None
        assert getattr(self.tokenizer, "padding_side", None) == "left", \
            f"tokenizer.padding_side must be 'left' (got {self.tokenizer.padding_side})"

    def _left_pad_1d(self, tensors: List[torch.Tensor], pad_value: int) -> torch.Tensor:
        max_len = max(t.size(0) for t in tensors)
        out = tensors[0].new_full((len(tensors), max_len), pad_value)
        for i, t in enumerate(tensors):
            out[i, -t.size(0):] = t
        return out

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        features = [f for f in features if f is not None]
        if not features:
            return {}

        pad_id = self.tokenizer.pad_token_id

        input_ids = self._left_pad_1d([f["input_ids"] for f in features], pad_id)
        attention_mask = self._left_pad_1d([f["attention_mask"] for f in features], 0)
        labels = self._left_pad_1d([f["labels"] for f in features], -100)

        # flatten images across batch: (N,C,H,W) cat -> (B*N,C,H,W)
        pixel_values = torch.cat([f["pixel_values"] for f in features], dim=0)

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values,
        }

        if features[0].get("image_grid_thw") is not None:
            grids = []
            for f in features:
                g = f["image_grid_thw"]
                if g is None:
                    raise ValueError("Some samples have image_grid_thw=None while others are not None.")
                # 샘플이 혹시라도 (3,)이면 (1,3)으로 복원
                if g.ndim == 1 and g.numel() == 3:
                    g = g.unsqueeze(0)
                if not (g.ndim == 2 and g.size(-1) == 3):
                    raise ValueError(f"Bad per-sample grid_thw shape in collator: {tuple(g.shape)}")
                grids.append(g)

            grid_thw = torch.cat(grids, dim=0)  # (B*num_images, 3)
            if not (grid_thw.ndim == 2 and grid_thw.size(-1) == 3):
                raise ValueError(f"Bad batched grid_thw shape: {tuple(grid_thw.shape)}")

            batch["image_grid_thw"] = grid_thw
        else:
            batch["image_grid_thw"] = None

        return batch