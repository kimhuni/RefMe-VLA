from __future__ import annotations
import yaml
from typing import Dict, Optional


# v3: output은 YAML (학습 대상)
def dump_yaml(d: Dict) -> str:
    # 사람이 읽기 쉬운 YAML, 키 순서 유지
    return yaml.safe_dump(d, sort_keys=False, allow_unicode=True).strip()


DETECT_SYSTEM = (
    "Role: HLP-DETECT.\n"
    "Given image(s) and Memory, decide whether the target event has occurred in the current frame.\n"
    "Return YAML with keys: Event.\n"
    "- Event must be true or false.\n"
)

UPDATE_SYSTEM = (
    "Role: HLP-UPDATE.\n"
    "Given image(s), Task and Previous_Memory, update the memory AFTER the event.\n"
    "Return YAML with keys: Action_Command, Working_Memory, Episodic_Context.\n"
    "Keep values concise.\n"
)


def render_memory_one_line(mem: Dict[str, str]) -> str:
    # prompt 입력은 한 줄로 짧게, 출력은 YAML로 강제
    ac = mem.get("Action_Command", "None")
    wm = mem.get("Working_Memory", "None")
    ec = mem.get("Episodic_Context", "None")
    return f"Action_Command: {ac} | Working_Memory: {wm} | Episodic_Context: {ec}"


def make_detect_prompt(task_text: str, memory: Dict[str, str], n_images: int) -> str:
    img_tokens = "<image_table>" + (" <image_wrist>" if n_images == 2 else "")
    return (
        f"{DETECT_SYSTEM}\n"
        f"Task: {task_text}\n"
        f"Memory: {render_memory_one_line(memory)}\n"
        f"Images: {img_tokens}\n"
    )


def make_update_prompt(task_text: str, prev_memory: Dict[str, str], n_images: int) -> str:
    img_tokens = "<image_table>" + (" <image_wrist>" if n_images == 2 else "")
    return (
        f"{UPDATE_SYSTEM}\n"
        f"Task: {task_text}\n"
        f"Previous_Memory: {render_memory_one_line(prev_memory)}\n"
        f"Images: {img_tokens}\n"
    )