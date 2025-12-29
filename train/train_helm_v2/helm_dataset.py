# helm_dataset.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image
from transformers import AutoProcessor


# -------------------------
# Dummy image cache (for update rows)
# -------------------------
_DUMMY_IMG_CACHE: Dict[Tuple[int, int], Image.Image] = {}


def _get_dummy_image(size: Tuple[int, int] = (224, 224)) -> Image.Image:
    """Return a cached black RGB image."""
    if size not in _DUMMY_IMG_CACHE:
        _DUMMY_IMG_CACHE[size] = Image.new("RGB", size, color=(0, 0, 0))
    return _DUMMY_IMG_CACHE[size]


# -------------------------
# Simple YAML dump (no PyYAML dependency)
# -------------------------
def _yaml_dump(d: Dict[str, Any]) -> str:
    lines = []
    for k, v in d.items():
        if isinstance(v, bool):
            vv = "true" if v else "false"
        elif v is None:
            vv = "null"
        else:
            vv = str(v)
        lines.append(f"{k}: {vv}")
    return "\n".join(lines)


def _read_jsonl_file(path: Path) -> List[Dict[str, Any]]:
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


def _read_jsonl(path_or_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    p = Path(path_or_dir)
    if p.is_file():
        return _read_jsonl_file(p)
    if p.is_dir():
        rows: List[Dict[str, Any]] = []
        files = sorted(p.rglob("*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No jsonl files under: {p}")
        for fp in files:
            rows.extend(_read_jsonl_file(fp))
        return rows
    raise FileNotFoundError(f"Not found: {p}")


def _find_subsequence(haystack: List[int], needle: List[int]) -> int:
    """Return start index of needle in haystack, or -1 if not found."""
    if not needle or len(needle) > len(haystack):
        return -1
    L = len(needle)
    for i in range(len(haystack) - L, -1, -1):
        if haystack[i : i + L] == needle:
            return i
    return -1


# -------------------------
# Prompt templates (v2)
# -------------------------
DETECT_HEADER = (
    "You are the robot arm Visual Event Detector.\n"
    "Goal: Verify whether the CURRENT action has been fully completed in the image.\n"
    "Input: An image + Global_Instruction describing what counts as action completion.\n"
    "Decision rule:\n"
    "- Use the Global_Instruction and image as the ONLY completion criterion.\n"
    "- Event_Detected: true ONLY when the completion condition is clearly and unambiguously visible.\n"
    "- Otherwise (partial progress / occlusion / uncertainty) -> Event_Detected: false.\n"
    "Constraints:\n"
    "- Do not propose next actions.\n"
    "- Do not update or rewrite memory.\n"
    "- Do not output any text except YAML.\n"
    "Return YAML with exactly one key: Event_Detected (boolean).\n"
)

UPDATE_HEADER = (
    "You are the robot arm Logic State Manager.\n"
    "Context: Event_Detected=true or a Task Change has occurred.\n"
    "Inputs:\n"
    "- Global_Instruction defining the overall task.\n"
    "- Previous memory state (Working_Memory, Episodic_Context, Action_Command).\n"
    "Goal: Update internal memory and decide the next Action_Command based on the Global_Instruction.\n"
    "Logic Rules:\n"
    "1) Update Working_Memory to reflect the action that has just been completed.\n"
    "2) Check task status using Working_Memory and Global_Instruction:\n"
    "   - If the task continues: keep Episodic_Context unchanged and select the next Action_Command.\n"
    "   - If the task is finished: promote/summarize the final result into Episodic_Context and set Action_Command: done.\n"
    "Constraints:\n"
    "- Action_Command must be selected ONLY from Allowed_Action_Commands.\n"
    "- Do not add new actions or explanations.\n"
    "- Output YAML only with keys: Working_Memory, Episodic_Context, Action_Command.\n"
)


def _build_v2_prompt_and_target(row: Dict[str, Any]) -> Tuple[str, str]:
    mode = row.get("mode", None)
    gi = str(row.get("global_instruction", "")).strip()
    mem_in = row.get("memory_in", {})
    if not isinstance(mem_in, dict):
        mem_in = {}

    label = row.get("label", {})
    if not isinstance(label, dict):
        label = {}

    if mode == "detect":
        user = (
            DETECT_HEADER
            + "\nGlobal_Instruction:\n"
            + gi
            + "\n\nCurrent_Memory:\n"
            + _yaml_dump(mem_in)
        )
        # target yaml
        tgt = _yaml_dump({"Event_Detected": bool(label.get("Event_Detected", False))})
        return user, tgt

    if mode == "update":
        # allowed actions (optional)
        allowed = None
        pc = row.get("prompt_context", None)
        if isinstance(pc, dict):
            allowed = pc.get("allowed_actions", None)
        if allowed is None:
            allowed = row.get("action_candidates", None)

        user = (
            UPDATE_HEADER
            + "\nGlobal_Instruction:\n"
            + gi
            + "\n\nPrev_Memory:\n"
            + _yaml_dump(mem_in)
        )
        if allowed is not None:
            user += "\n\nAllowed_Action_Commands:\n"
            if isinstance(allowed, list):
                user += "\n".join([f"- {str(x)}" for x in allowed])
            else:
                user += str(allowed)

        # target yaml
        tgt = _yaml_dump({
            "Working_Memory": label.get("Working_Memory", ""),
            "Episodic_Context": label.get("Episodic_Context", ""),
            "Action_Command": label.get("Action_Command", ""),
        })
        return user, tgt

    # legacy fallback (expects conversations)
    conv = row.get("conversations", [])
    if not isinstance(conv, list):
        raise ValueError("Unknown row schema (no mode, no conversations).")
    user_text, asst_text = "", ""
    for m in conv:
        if not isinstance(m, dict):
            continue
        if m.get("from") == "user":
            user_text = str(m.get("value", ""))
        elif m.get("from") == "assistant":
            asst_text = str(m.get("value", ""))
    if not user_text or not asst_text:
        raise ValueError("Missing user/assistant in legacy conversations")
    return user_text, asst_text


def _load_images_v2(
    row: Dict[str, Any],
    num_image: int,
    camera: str,
    frames_root: Optional[Path],
    require_images_for_update: bool,
) -> List[Image.Image]:
    """
    v2 rules:
    - detect: must have row['images'][camera] OR cannot train (error)
    - update: row often has no images; if frames_root is provided and chunk/episode/t_event exist,
             we reconstruct a table image path at t_event.
    """
    mode = row.get("mode", None)

    # Helper to open path
    def open_rgb(p: Union[str, Path]) -> Image.Image:
        return Image.open(str(p)).convert("RGB")

    images = row.get("images", {})
    if isinstance(images, dict) and camera in images and images[camera]:
        img0 = open_rgb(images[camera])
        if int(num_image) == 1:
            return [img0]
        # if you ever support wrist later, extend here
        raise ValueError("num_image=2 not supported in this v2 minimal loader (no wrist in v2).")

    # No images in row
    if mode == "update":
        # Update is text-only; we keep the VLM input format consistent by providing a dummy image.
        # This avoids conditioning update on any real visual content and removes the need for --frames_root.
        return [_get_dummy_image()]

    # detect with missing images -> hard error (data bug)
    raise ValueError(f"Row has no usable images for camera='{camera}' (mode={mode}). keys={list(images.keys()) if isinstance(images, dict) else type(images)}")


class HelmJsonlDataset(Dataset):
    """
    Supports both:
      - legacy HeLM JSONL (row['conversations'], row['images'])
      - v2 JSONL (row['mode'] in {'detect','update'}, plus fields shown in your examples)
    """

    def __init__(
        self,
        jsonl_path: str,
        model_name_or_path: str,
        num_image: int = 1,
        camera: str = "table",
        frames_root: Optional[str] = None,
        require_images_for_update: bool = True,
    ):
        self.jsonl_path = jsonl_path
        self.rows = _read_jsonl(jsonl_path)

        self.num_image = int(num_image)
        self.camera = str(camera)

        self.frames_root = Path(frames_root) if frames_root else None
        self.require_images_for_update = bool(require_images_for_update)

        # NOTE: With dummy-update images enabled, frames_root/require_images_for_update are no longer needed
        # for v2 update rows. They are kept only for backward compatibility.

        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            use_fast=True,
        )
        self.processor.tokenizer.padding_side = "left"
        if self.processor.tokenizer.pad_token is None:
            self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]

        user_text, target_text = _build_v2_prompt_and_target(row)

        # Messages: for Qwen2.5-VL we include one image token + text
        # (We keep update rows also image-conditioned by reconstructing the table image at t_event.)
        images = _load_images_v2(
            row=row,
            num_image=self.num_image,
            camera=self.camera,
            frames_root=self.frames_root,
            require_images_for_update=self.require_images_for_update,
        )

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

        tgt_ids = self.processor.tokenizer(target_text, add_special_tokens=False).input_ids
        full = input_ids.tolist()

        start = _find_subsequence(full, tgt_ids)
        if start == -1:
            tgt_ids2 = self.processor.tokenizer("\n" + target_text, add_special_tokens=False).input_ids
            start = _find_subsequence(full, tgt_ids2)

        if start == -1:
            labels[:] = -100
            # keep it silent-ish; too noisy otherwise
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
                grid = grid.unsqueeze(0)
            out["image_grid_thw"] = grid

        return out


@dataclass
class HelmDataCollator:
    pad_id: int = 0

    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids = pad_sequence([f["input_ids"] for f in features], batch_first=True, padding_value=self.pad_id)
        attention_mask = pad_sequence([f["attention_mask"] for f in features], batch_first=True, padding_value=0)
        labels = pad_sequence([f["labels"] for f in features], batch_first=True, padding_value=-100)

        batch: Dict[str, torch.Tensor] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": torch.cat([f["pixel_values"] for f in features], dim=0),
        }

        if "image_grid_thw" in features[0]:
            batch["image_grid_thw"] = torch.cat([f["image_grid_thw"] for f in features], dim=0)

        if any(("image_grid_thw" not in f) or (f.get("image_grid_thw") is None) for f in features):
            batch["image_grid_thw"] = None

        return batch