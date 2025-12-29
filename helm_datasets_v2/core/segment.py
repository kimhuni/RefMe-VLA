# helm_datasets_v2/core/segment.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from .spec import EpisodeFilter


@dataclass(frozen=True)
class StepAssignment:
    inter: int
    intra: int
    task_match: str


def _allowed_tasks(filter_item: EpisodeFilter) -> List[str]:
    """
    filter_item can be:
      - {"tasks": "..."}
      - [{"tasks": "..."}, {"tasks": "..."}]
    OR semantics across list.
    """
    if isinstance(filter_item, dict):
        if "tasks" in filter_item:
            return [str(filter_item["tasks"]).strip()]
        return []

    if isinstance(filter_item, list):
        out = []
        for f in filter_item:
            if isinstance(f, dict) and "tasks" in f:
                out.append(str(f["tasks"]).strip())
        return [x for x in out if x]

    return []


def assign_episode_to_step(
    episode_task: str,
    episode_filters: List[List[EpisodeFilter]],
) -> Optional[StepAssignment]:
    """
    Episode-level assignment:
    - episode_task is a single string (episode performs one task)
    - find first (inter,intra) whose allowed task list contains episode_task
    - return that assignment, else None
    """
    t = (episode_task or "").strip()
    if not t:
        return None

    for inter_idx, intra_list in enumerate(episode_filters):
        for intra_idx, filt in enumerate(intra_list):
            allowed = _allowed_tasks(filt)
            if t in allowed:
                return StepAssignment(inter=inter_idx, intra=intra_idx, task_match=t)

    return None