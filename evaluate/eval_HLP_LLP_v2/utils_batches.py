# utils_batches.py
from __future__ import annotations
from typing import Any, Dict, List, Optional

import yaml


def _yaml_dump(d: Dict[str, Any]) -> str:
    # training과 동일하게 stable yaml (짧고 안전)
    return yaml.safe_dump(d, sort_keys=False, allow_unicode=True).strip()


def build_detect_user_text(detect_header: str, global_instruction: str, memory_in: Dict[str, Any]) -> str:
    return (
        detect_header
        + "\nGlobal_Instruction:\n"
        + str(global_instruction).strip()
        + "\n\nCurrent_Memory:\n"
        + _yaml_dump(memory_in if isinstance(memory_in, dict) else {})
    )


def build_update_user_text(
    update_header: str,
    global_instruction: str,
    memory_in: Dict[str, Any],
    allowed: str = None,
) -> str:
    user = (
        update_header
        + "\nGlobal_Instruction:\n"
        + str(global_instruction).strip()
        + "\n\nPrev_Memory:\n"
        + _yaml_dump(memory_in if isinstance(memory_in, dict) else {})
    )
    if allowed is not None:
        user += "\n\nAllowed_Action_Commands:\n"
        if isinstance(allowed, list):
            user += "\n".join([f"- {str(x)}" for x in allowed])
        else:
            user += str(allowed)
    return user


def create_hlp_detect_batch(processor, table_pil, user_text: str):
    """
    Qwen2.5-VL chat template는 placeholder image 개수와 실제 images 개수가 맞아야 안정적.
    -> 이미지 1장 고정.
    """
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return processor(
        text=[prompt],
        images=[table_pil],
        padding=True,
        return_tensors="pt",
    )


def create_hlp_update_batch(processor, dummy_table_pil, user_text: str):
    messages = [
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": user_text}]},
    ]
    prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return processor(
        text=[prompt],
        images=[dummy_table_pil],
        padding=True,
        return_tensors="pt",
    )