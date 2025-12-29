# helm_datasets_v2/core/spec.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json


EpisodeFilter = Union[Dict[str, Any], List[Dict[str, Any]]]


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    inter: int
    intra: List[int]
    task_text: List[str]

    # NEW: episode_filters is [inter][intra] where each item can be:
    #  - {"tasks": "..."}  (single dict)
    #  - [{"tasks": "..."}, ...]  (list of dicts; OR semantics)
    episode_filters: List[List[EpisodeFilter]]

    memory_grid: List[List[Dict[str, Any]]]  # [inter][state_index]
    require_event: bool = True

    # detect sampling knobs
    camera: str = "table"
    negatives_per_positive: int = 3
    min_gap: int = 1

    # episode json schema
    event_frames_key: str = "event_frames"

    # NEW: allowed action candidates for update prompt context
    llp_commands: Optional[str] = None
    llp_command_list: Optional[List[str]] = None


def _parse_llp_commands(s: Optional[str]) -> List[str]:
    if not s:
        return []
    # split by newline, strip, drop empties
    cmds = []
    for line in s.splitlines():
        t = line.strip()
        if t:
            cmds.append(t)
    return cmds


def load_taskspec(path: Path) -> TaskSpec:
    with path.open("r", encoding="utf-8") as f:
        d = json.load(f)

    d.setdefault("require_event", True)

    # sampling override (optional)
    camera = d.get("camera", None)
    npp = d.get("negatives_per_positive", None)
    mgap = d.get("min_gap", None)
    if "sampling" in d and isinstance(d["sampling"], dict):
        det = d["sampling"].get("detect", {})
        if isinstance(det, dict):
            camera = det.get("camera", camera)
            npp = det.get("negatives_per_positive", npp)
            mgap = det.get("min_gap", mgap)

    event_schema = d.get("event_schema", {})
    if isinstance(event_schema, dict):
        event_key = event_schema.get("event_frames_key", d.get("event_frames_key", "event_frames"))
    else:
        event_key = d.get("event_frames_key", "event_frames")

    llp = d.get("llp_commands", None)
    llp_list = _parse_llp_commands(llp)

    # infer inter if missing
    inferred_inter = int(d.get("inter", len(d.get("memory_grid", [])) - 1))

    spec = TaskSpec(
        task_id=d["task_id"],
        inter=inferred_inter,
        intra=list(d["intra"]),
        task_text=list(d["task_text"]),
        episode_filters=d.get("episode_filters", []),
        memory_grid=d["memory_grid"],
        require_event=bool(d.get("require_event", True)),
        camera=camera or "table",
        negatives_per_positive=int(npp) if npp is not None else 3,
        min_gap=int(mgap) if mgap is not None else 1,
        event_frames_key=str(event_key),
        llp_commands=llp,
        llp_command_list=llp_list,
    )
    validate_spec(spec)
    return spec


def validate_spec(spec: TaskSpec) -> None:
    expected_inters = spec.inter + 1

    if len(spec.memory_grid) != expected_inters:
        raise ValueError(f"memory_grid inter dim mismatch: {len(spec.memory_grid)} vs {expected_inters}")

    # ✅ typo fix
    if len(spec.intra) != expected_inters:
        raise ValueError(f"intra dim mismatch: {len(spec.intra)} vs {expected_inters}")

    # episode_filters optional but if provided, must align
    if spec.episode_filters:
        if len(spec.episode_filters) != expected_inters:
            raise ValueError(f"episode_filters inter dim mismatch: {len(spec.episode_filters)} vs {expected_inters}")
        for i in range(expected_inters):
            if len(spec.episode_filters[i]) != spec.intra[i]:
                raise ValueError(
                    f"episode_filters intra dim mismatch at inter={i}: {len(spec.episode_filters[i])} vs intra={spec.intra[i]}"
                )

    for i in range(expected_inters):
        states = spec.memory_grid[i]
        if len(states) != spec.intra[i] + 1:
            raise ValueError(f"memory_grid[{i}] length must be intra+1={spec.intra[i]+1}, got {len(states)}")

        for j, cell in enumerate(states):
            if not isinstance(cell, dict):
                raise ValueError(f"memory_grid[{i}][{j}] must be a dict")
            for k in ("Action_Command", "Working_Memory", "Episodic_Context"):
                if k not in cell:
                    raise ValueError(f"memory_grid[{i}][{j}] missing key {k}")

        if str(states[-1].get("Action_Command", "")).strip().lower() != "done":
            raise ValueError(f"memory_grid[{i}][-1].Action_Command should be 'done'")

    # ✅ allow npp <= 0 to mean "all negatives"
    if spec.negatives_per_positive is None:
        raise ValueError("negatives_per_positive must be set (use -1 for ALL)")

    # llp_commands optional but if provided, recommend it contains 'done'
    if spec.llp_command_list and not any(cmd.strip().lower() == "done" for cmd in spec.llp_command_list):
        raise ValueError("llp_commands provided but does not include 'done'")