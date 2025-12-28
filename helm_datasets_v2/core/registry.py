# helm_datasets/core/registry.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

from helm_datasets_v2.core.spec import TaskSpec


def load_taskspec(path: Path) -> TaskSpec:
    raw: Dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))

    # v1/v2 공통
    task_id = raw["task_id"]
    max_inter = int(raw["max_inter"])

    # max_intra는 반드시 list
    mi = raw["max_intra"]
    if isinstance(mi, int):
        max_intra = [int(mi)]
    else:
        max_intra = [int(x) for x in list(mi)]

    task_text = raw["task_text"]
    if isinstance(task_text, str):
        task_text = [task_text]

    episode_filters = raw.get("episode_filters", None)

    command_grid = raw["command_grid"]
    progress_grid = raw["progress_grid"]
    world_state_grid = raw.get("world_state_grid", "None")

    # v2
    llp_command = raw.get("llp_command", "")
    if not llp_command:
        # v1에서 llp_commands 같은 이름을 썼다면
        llp_commands = raw.get("llp_commands", None)
        if isinstance(llp_commands, list) and llp_commands:
            llp_command = str(llp_commands[0])

    init_memory = raw.get("init_memory", "")

    spec = TaskSpec(
        task_id=task_id,
        max_inter=max_inter,
        max_intra=max_intra,
        task_text=list(task_text),
        episode_filters=episode_filters,
        command_grid=command_grid,
        progress_grid=progress_grid,
        world_state_grid=world_state_grid,
        llp_command=str(llp_command),
        init_memory=str(init_memory),
    )
    spec.validate()
    return spec


def get_task_registry() -> Dict[str, TaskSpec]:
    """
    너의 레포에서 taskspec json들이 있는 경로로 맞춰줘.
    예: helm_datasets/taskspecs/*.json
    """
    root = Path(__file__).resolve().parent.parent / "taskspecs"
    paths = sorted(root.glob("*.json"))
    reg: Dict[str, TaskSpec] = {}
    for p in paths:
        spec = load_taskspec(p)
        reg[spec.task_id] = spec
    return reg
