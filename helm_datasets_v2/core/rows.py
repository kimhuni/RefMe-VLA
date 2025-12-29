# helm_datasets_v2/core/rows.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional
import random
import hashlib

from .spec import TaskSpec
from .episode import DataEpisode
from .segment import StepAssignment
from .state import build_task_change_transitions


def _uid(parts: List[str]) -> str:
    h = hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()
    return h[:16]


def _pick_event_t(ep: DataEpisode) -> Optional[int]:
    # your schema normally has single event_frame_idx -> event_frames[0]
    if not ep.event_frames:
        return None
    return int(min(ep.event_frames))


def _sample_negatives(start_t: int, end_t_exclusive: int, n: int, min_gap: int, rng: random.Random) -> List[int]:
    """Return negative frame indices.

    If n <= 0: return ALL frames in [start_t, end_t_exclusive) (optionally thinned by min_gap).
    If n > 0: sample n frames uniformly without replacement from that range (after min_gap thinning).
    """
    pool = list(range(start_t, end_t_exclusive))
    if min_gap > 1 and pool:
        pool = pool[::min_gap]

    # ALL negatives mode
    if n <= 0:
        return pool

    n = min(n, len(pool))
    return rng.sample(pool, n) if n > 0 else []


def build_detect_rows(
    spec: TaskSpec,
    ep: DataEpisode,
    assign: StepAssignment,
    split: str,
    seed: int = 0,
) -> Iterable[Dict[str, Any]]:
    """
    Episode-level detect:
      - negatives sampled from [0, event_t)
      - positive at event_t
    """
    rng = random.Random(seed)

    event_t = _pick_event_t(ep)
    if event_t is None:
        if spec.require_event:
            return
        else:
            return

    # guard
    T = len(ep.frame_paths)
    if event_t < 0 or event_t >= T:
        return

    neg_frames = _sample_negatives(0, event_t, spec.negatives_per_positive, spec.min_gap, rng)
    frames = neg_frames + [event_t]

    mem_in = spec.memory_grid[assign.inter][assign.intra]

    for t in frames:
        is_pos = (t == event_t)
        img_path = str(ep.frame_paths[t])

        yield {
            "uid": _uid([spec.task_id, "detect", ep.chunk, ep.episode, str(assign.inter), str(assign.intra), str(t)]),
            "task_id": spec.task_id,
            "mode": "detect",
            "split": split,
            "chunk": ep.chunk,
            "episode": ep.episode,
            "inter": assign.inter,
            "intra": assign.intra,
            "t": int(t),
            "images": {spec.camera: img_path},
            "global_instruction": spec.task_text[assign.inter],
            "memory_in": {
                "Working_Memory": mem_in["Working_Memory"],
                "Episodic_Context": mem_in["Episodic_Context"],
                "Action_Command": mem_in["Action_Command"],
            },
            "label": {"Event_Detected": bool(is_pos)},
            "meta": {"episode_task": ep.task_str, "event_t": int(event_t), "task_match": assign.task_match},
        }


def build_update_rows(
    spec: TaskSpec,
    ep: DataEpisode,
    assign: StepAssignment,
    split: str,
) -> Iterable[Dict[str, Any]]:
    """
    Episode-level update (text-only):
      - intra update: memory_grid[inter][intra] -> [inter][intra+1]
      - include allowed_actions from llp_commands (if provided)
    """
    event_t = _pick_event_t(ep)
    if event_t is None and spec.require_event:
        return

    mem_in = spec.memory_grid[assign.inter][assign.intra]
    mem_out = spec.memory_grid[assign.inter][assign.intra + 1]

    allowed_actions = spec.llp_command_list or []

    yield {
        "uid": _uid([spec.task_id, "update", "intra", ep.chunk, ep.episode, str(assign.inter), str(assign.intra)]),
        "task_id": spec.task_id,
        "mode": "update",
        "split": split,
        "chunk": ep.chunk,
        "episode": ep.episode,
        "transition": "intra",
        "inter_from": assign.inter,
        "intra_from": assign.intra,
        "inter_to": assign.inter,
        "intra_to": assign.intra + 1,
        "t_event": int(event_t) if event_t is not None else None,
        "global_instruction": spec.task_text[assign.inter],
        "prompt_context": {
            "allowed_actions": allowed_actions
        },
        "memory_in": {
            "Working_Memory": mem_in["Working_Memory"],
            "Episodic_Context": mem_in["Episodic_Context"],
            "Action_Command": mem_in["Action_Command"],
        },
        "label": {
            "Working_Memory": mem_out["Working_Memory"],
            "Episodic_Context": mem_out["Episodic_Context"],
            "Action_Command": mem_out["Action_Command"],
        },
        "meta": {"episode_task": ep.task_str, "task_match": assign.task_match},
    }


def build_task_change_update_rows(
    spec: TaskSpec,
    split: str,
    chunk: str = "N/A",
    episode: str = "N/A",
) -> Iterable[Dict[str, Any]]:
    """
    Spec-only inter transition rows (no image).
    """
    allowed_actions = spec.llp_command_list or []
    for tr in build_task_change_transitions(spec):
        yield {
            "uid": _uid([spec.task_id, "update", "task_change", str(tr.inter_from), str(tr.inter_to)]),
            "task_id": spec.task_id,
            "mode": "update",
            "split": split,
            "chunk": chunk,
            "episode": episode,
            "transition": "task_change",
            "inter_from": tr.inter_from,
            "intra_from": tr.intra_from,
            "inter_to": tr.inter_to,
            "intra_to": tr.intra_to,
            "t_event": None,
            "global_instruction": tr.global_task,
            "prompt_context": {
                "allowed_actions": allowed_actions
            },
            "memory_in": {
                "Working_Memory": tr.memory_in["Working_Memory"],
                "Episodic_Context": tr.memory_in["Episodic_Context"],
                "Action_Command": tr.memory_in["Action_Command"],
            },
            "label": {
                "Working_Memory": tr.memory_out["Working_Memory"],
                "Episodic_Context": tr.memory_out["Episodic_Context"],
                "Action_Command": tr.memory_out["Action_Command"],
            },
            "meta": {"note": "task boundary update (no image)"},
        }