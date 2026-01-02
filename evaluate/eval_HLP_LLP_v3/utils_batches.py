# utils_batches.py
from __future__ import annotations
from typing import Any, Dict, List, Optional

import yaml


def _yaml_dump(d: Dict[str, Any]) -> str:
    # training과 동일하게 stable yaml (짧고 안전)
    return yaml.safe_dump(d, sort_keys=False, allow_unicode=True).strip()


def build_detect_user_text(detect_header: str, global_instruction: str, memory_in: Dict[str, Any]) -> str:
    ac = memory_in.get("Action_Command", "None")
    wm = memory_in.get("Working_Memory", "None")
    ec = memory_in.get("Episodic_Context", "None")
    mem =  f"Action_Command: {ac} | Working_Memory: {wm} | Episodic_Context: {ec}"
    user_d = (
        detect_header
        + f"Task: {str(global_instruction).strip()}\n"
        + f"Memory: {mem}\n"
        + f"Images: <image_table>\n"
        # + _yaml_dump(memory_in if isinstance(memory_in, dict) else {})
    )
    # print("[DETECT] user text input \n")
    # print(user_d)

    return user_d


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

    # print("[UPDATE] user text input \n")
    # print(user)
    return user


def _make_user_messages_with_images(user_text: str):
    """
    Qwen2.5-VL chat template: placeholder image 개수 == 실제 images 개수 여야 함.
    """
    content = [{"type": "image"}]
    content.append({"type": "text", "text": user_text})
    return [{"role": "user", "content": content}]


def create_hlp_detect_batch(processor, obs_pil, user_text: str):
    """
    DETECT batch 생성.
    images_pil: List[PIL] (len==num_images)
    """
    messages = _make_user_messages_with_images(user_text=user_text)

    # print("----------Detect input text--------------\n", messages)
    prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return processor(
        text=prompt,
        images = obs_pil,  # IMPORTANT: single PIL, no nested list
        padding = True,
        return_tensors = "pt",
    )


def create_hlp_update_batch(processor, obs_pil, user_text: str, num_images: int = 1):
    """
    UPDATE batch 생성.
    v3 정합: dummy 이미지 금지 -> 실시간 캡처 이미지(1장 또는 2장)를 그대로 넣는다.
    """
    messages = _make_user_messages_with_images(user_text=user_text)
    # print("----------Detect input text--------------\n", messages)

    prompt = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return processor(
            text=prompt,
            images = obs_pil,  # IMPORTANT: single PIL, no nested list
            padding = True,
            return_tensors = "pt",
        )