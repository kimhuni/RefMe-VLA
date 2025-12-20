from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Set, Any

from .spec import TaskSpec


def _default_taskspecs_dir() -> Path:
    """Return `<package_root>/taskspecs`.

    Assumes this file lives in `<package_root>/core/registry.py` (or similar).
    """
    # registry.py (this file) -> .../helm_datasets/core/registry.py
    # package root -> .../helm_datasets
    return Path(__file__).resolve().parent.parent / "taskspecs"


def _parse_tasks_filter(tasks: Optional[str]) -> Optional[Set[str]]:
    """Parse comma-separated task ids from `--tasks` style string."""
    if tasks is None:
        return None
    tasks = tasks.strip()
    if not tasks:
        return None
    return {t.strip() for t in tasks.split(",") if t.strip()}


def _normalize_world_state_grid(raw: Any, max_inter: int) -> list[Optional[str]]:
    """Normalize world_state_grid to a 1D list over [inter].

    - Accepts either:
      1) 1D: [null, "total_count: 1", ...]
      2) 2D-ish: [[null], ["total_count: 1"], ...] (we take the first element)

    Returns a list of length (max_inter + 1).
    """
    if raw is None:
        out = [None] * (max_inter + 1)
        return out

    if not isinstance(raw, list):
        raise ValueError("world_state_grid must be a list")

    out: list[Optional[str]] = []
    for item in raw:
        if isinstance(item, list):
            # taskspecs example: [ [null], ["total_count: 1"] ]
            if len(item) == 0:
                out.append(None)
            else:
                v = item[0]
                out.append(None if v is None else str(v))
        else:
            out.append(None if item is None else str(item))

    # Pad/trim to expected length
    need = max_inter + 1
    if len(out) < need:
        out.extend([None] * (need - len(out)))
    elif len(out) > need:
        out = out[:need]

    return out


def get_task_registry(
    tasks: Optional[str] = None,
    taskspecs_dir: Optional[Path] = None,
) -> Dict[str, TaskSpec]:
    """Load TaskSpecs from JSON files under `taskspecs/`.

    - `tasks`: comma-separated list (same format as build_helm `--tasks`). If provided,
      only those task_ids are loaded.
    - `taskspecs_dir`: override directory for taskspec files.

    Expected file layout:
      <taskspecs_dir>/<task_id>.json

    Each JSON must include at least:
      task_id, task_text, max_inter, max_intra, command_grid, progress_grid

    And may include:
      world_state_grid  (either 1D [inter] or 2D-ish [[...]]; normalized to 1D)

    Returns:
      dict mapping task_id -> TaskSpec
    """
    allow = _parse_tasks_filter(tasks)
    root = taskspecs_dir or _default_taskspecs_dir()

    reg: Dict[str, TaskSpec] = {}
    if not root.exists():
        raise FileNotFoundError(f"taskspecs_dir not found: {root}")

    for p in sorted(root.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            raise RuntimeError(f"Failed to read taskspec JSON: {p} ({e})") from e

        task_id = str(data.get("task_id", p.stem))
        if allow is not None and task_id not in allow:
            continue

        # Required fields
        task_text = str(data["task_text"])
        max_inter = int(data["max_inter"])
        max_intra = int(data["max_intra"])
        command_grid = data["command_grid"]
        progress_grid = data["progress_grid"]

        # Optional world_state_grid (normalized to 1D [inter])
        world_state_grid = _normalize_world_state_grid(data.get("world_state_grid", None), max_inter)

        spec = TaskSpec(
            task_id=task_id,
            task_text=task_text,
            max_inter=max_inter,
            max_intra=max_intra,
            command_grid=command_grid,
            progress_grid=progress_grid,
            world_state_grid=world_state_grid,
        )
        spec.validate()

        if spec.task_id in reg:
            raise ValueError(f"Duplicate task_id '{spec.task_id}' loaded from: {p}")
        reg[spec.task_id] = spec

    if allow is not None:
        missing = sorted(list(allow - set(reg.keys())))
        if missing:
            raise KeyError(f"Requested tasks not found in {root}: {missing}")

    return reg