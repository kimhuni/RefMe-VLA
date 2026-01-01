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


def _make_user_messages_with_images(user_text: str, num_images: int):
    """
    Qwen2.5-VL chat template: placeholder image 개수 == 실제 images 개수 여야 함.
    """
    if num_images not in (1, 2):
        raise ValueError(f"num_images must be 1 or 2, got {num_images}")

    content = [{"type": "image"} for _ in range(num_images)]
    content.append({"type": "text", "text": user_text})
    return [{"role": "user", "content": content}]


def create_hlp_detect_batch(processor, images_pil, user_text: str, num_images: int = 1):
    """
    DETECT batch 생성.
    images_pil: List[PIL] (len==num_images)
    """
    if not isinstance(images_pil, (list, tuple)):
        images_pil = [images_pil]
    if len(images_pil) != num_images:
        raise ValueError(f"[DETECT] len(images_pil)={len(images_pil)} != num_images={num_images}")

    messages = _make_user_messages_with_images(user_text=user_text, num_images=num_images)
    prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return processor(
        text=[prompt],
        images=[list(images_pil)],
        padding=True,
        return_tensors="pt",
    )


def create_hlp_update_batch(processor, images_pil, user_text: str, num_images: int = 1):
    """
    UPDATE batch 생성.
    v3 정합: dummy 이미지 금지 -> 실시간 캡처 이미지(1장 또는 2장)를 그대로 넣는다.
    """
    if not isinstance(images_pil, (list, tuple)):
        images_pil = [images_pil]
    if len(images_pil) != num_images:
        raise ValueError(f"[UPDATE] len(images_pil)={len(images_pil)} != num_images={num_images}")

    messages = _make_user_messages_with_images(user_text=user_text, num_images=num_images)
    prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return processor(
        text=[prompt],
        images=[list(images_pil)],
        padding=True,
        return_tensors="pt",
    )