from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

from .spec import TaskSpecV4


def load_taskspec(path: Path) -> TaskSpecV4:
    raw = json.loads(path.read_text(encoding="utf-8"))
    spec = TaskSpecV4(
        task_id=raw["task_id"],
        inter=int(raw["inter"]),
        intra=[int(x) for x in raw["intra"]],
        task_text=list(raw["task_text"]),
        episode_filters=raw["episode_filters"],
        memory_grid=raw["memory_grid"],
        llp_commands=str(raw.get("llp_commands", "")),
        event_list=str(raw.get("event_list", "")),
        event_grid=raw.get("event_grid", None),
    )
    spec.validate()
    return spec


def load_all_taskspecs(taskspecs_dir: Path) -> Dict[str, TaskSpecV4]:
    """
    taskspecs_dir 내부의 모든 *.json 을 재귀적으로 읽는다.
    예: taskspecs/wipe_the_window/*.json, taskspecs/press_button/*.json ...
    """
    if not taskspecs_dir.exists():
        raise FileNotFoundError(f"taskspecs_dir not found: {taskspecs_dir}")

    specs: Dict[str, TaskSpecV4] = {}
    for p in sorted(taskspecs_dir.rglob("*.json")):
        spec = load_taskspec(p)
        if spec.task_id in specs:
            raise ValueError(f"Duplicate task_id={spec.task_id} in {p}")
        specs[spec.task_id] = spec
    return specs