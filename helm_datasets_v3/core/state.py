# helm_datasets_v2/core/state.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .spec import TaskSpec
from .segment import StepAssignment


@dataclass(frozen=True)
class Transition:
    kind: str            # "intra" | "task_change"
    inter_from: int
    intra_from: int
    inter_to: int
    intra_to: int
    global_task: str
    memory_in: Dict[str, Any]
    memory_out: Dict[str, Any]


def validate_episode_against_spec(
    spec: TaskSpec,
    assignment: StepAssignment,
    event_frames: List[int],
) -> None:
    if spec.require_event and not event_frames:
        raise ValueError("require_event=True but episode has no event_frames")
    # for this dataset, each episode should have exactly one main event (recommended)
    # we won't hard-enforce ==1, but we must be able to pick one.
    # assignment exists by construction.


def build_task_change_transitions(spec: TaskSpec) -> List[Transition]:
    trans: List[Transition] = []
    for inter in range(spec.inter):
        mem_in = spec.memory_grid[inter][-1]       # done
        mem_out = spec.memory_grid[inter + 1][0]   # next task init
        trans.append(
            Transition(
                kind="task_change",
                inter_from=inter,
                intra_from=spec.intra[inter],       # conceptual done index
                inter_to=inter + 1,
                intra_to=0,
                global_task=spec.task_text[inter + 1],
                memory_in=mem_in,
                memory_out=mem_out,
            )
        )
    return trans