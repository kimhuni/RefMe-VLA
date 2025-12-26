from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Any

from .spec import TaskSpec


def _default_taskspecs_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "taskspecs"


def _parse_tasks_filter(tasks: Optional[str]) -> Optional[Set[str]]:
    if tasks is None:
        return None
    tasks = tasks.strip()
    if not tasks:
        return None
    return {t.strip() for t in tasks.split(",") if t.strip()}


def load_taskspec(path: Path) -> TaskSpec:
    raw: Any = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict), f"taskspec must be a json object: {path}"

    ws_grid = raw.get("world_state_grid", [])
    if isinstance(ws_grid, list):
        norm_ws: List[Optional[str]] = []
        for v in ws_grid:
            norm_ws.append(str(v))
        raw["world_state_grid"] = norm_ws

    spec = TaskSpec(
        task_id=str(raw["task_id"]),
        task_text=list(raw["task_text"]),
        episode_filters=list(raw.get("episode_filters", [{} for _ in range(int(raw["max_inter"]) + 1)])),
        max_inter=int(raw["max_inter"]),
        max_intra=[int(x) for x in list(raw["max_intra"])],
        command_grid=[list(r) for r in list(raw["command_grid"])],
        progress_grid=[list(r) for r in list(raw["progress_grid"])],
        world_state_grid=list(raw.get("world_state_grid", [])),
    )
    spec.validate()
    return spec


def get_task_registry(
    taskspecs_dir: Optional[Path] = None,
    tasks: Optional[str] = None,
    allow_task_ids: Optional[List[str]] = None,
) -> Dict[str, TaskSpec]:
    taskspecs_dir = taskspecs_dir or _default_taskspecs_dir()
    if not taskspecs_dir.exists():
        raise FileNotFoundError(f"taskspecs_dir not found: {taskspecs_dir}")

    wanted = _parse_tasks_filter(tasks)
    if allow_task_ids is not None:
        wanted = set(allow_task_ids) if wanted is None else (wanted & set(allow_task_ids))

    reg: Dict[str, TaskSpec] = {}
    for p in sorted(taskspecs_dir.glob("*.json")):
        spec = load_taskspec(p)
        if wanted is not None and spec.task_id not in wanted:
            continue
        reg[spec.task_id] = spec

    if wanted is not None:
        missing = [t for t in sorted(wanted) if t not in reg]
        if missing:
            raise KeyError(f"Requested tasks not found in {taskspecs_dir}: {missing}")

    return reg
