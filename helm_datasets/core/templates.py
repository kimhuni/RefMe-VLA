from __future__ import annotations

from typing import Any, Dict, Optional

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    yaml = None
    _HAS_YAML = False


def render_user_prompt(
    task_text: str,
    previous_memory: str,
    images: Dict[str, str],
    frame_idx: int,
) -> str:
    img_tokens = " ".join([f"<image_{k}>" for k in images.keys()])
    return (
        f"Task: {task_text}\n"
        f"Previous_Memory: {previous_memory}\n"
        f"Frame: {frame_idx}\n"
        f"Images: {img_tokens}\n"
        f"Return YAML with keys Progress, World_State, Command."
    )


def render_assistant_yaml(progress: str, world_state: Optional[str], command: str) -> str:
    obj: Dict[str, Any] = {
        "Progress": progress,
        "World_State": world_state,
        "Command": command,
    }

    if _HAS_YAML:
        return yaml.safe_dump(obj, sort_keys=False, allow_unicode=True).strip()

    ws = "null" if world_state is None else f"\"{world_state}\""
    return f'Progress: "{progress}"\nWorld_State: {ws}\nCommand: "{command}"'
