from __future__ import annotations

from typing import Any, Dict, List, Optional

from .data_index import DataEpisode
from .spec import TaskSpec
from .templates import render_user_prompt, render_assistant_yaml


def intra_for_frame(t: int, event_frame_idxs: List[int], base_intra: int, max_intra: int) -> int:
    """Return intra state at frame t. Increments by the number of events that have occurred up to t (inclusive)."""
    if not event_frame_idxs:
        intra = base_intra
    else:
        k = sum(1 for e in event_frame_idxs if t >= e)
        intra = base_intra + k
    return max(0, min(intra, max_intra))


def merge_world_state_on_done(world_state: Optional[str], progress: str) -> Optional[str]:
    return world_state
    #suffix = f"final_progress = {progress}"
    #if world_state is None or str(world_state).strip().lower() == "none" or str(world_state).strip() == "":
    #    return suffix
    #return f"{world_state}; {suffix}"


def make_previous_memory_text(progress: str, world_state: Optional[str]) -> str:
    if world_state is None:
        return f"Progress: {progress}"
    return f"Progress: {progress} | World_State: {world_state}"


def make_uid(task_id: str, chunk: str, episode: str, inter: int, base_intra: int, frame_idx: int) -> str:
    return f"{task_id}@{chunk}-{episode}-inter{inter}-base{base_intra}-f{frame_idx:06d}"


def make_rows_for_variant(
    ep: DataEpisode,
    task: TaskSpec,
    inter: int,
    base_intra: int,
    cameras: List[str],
) -> List[Dict[str, Any]]:
    max_intra = task.max_intra[inter]
    if base_intra < 0 or base_intra + 1 > max_intra:
        raise ValueError(
            f"Invalid base_intra={base_intra} for task={task.task_id} inter={inter} max_intra={max_intra}"
        )

    if hasattr(ep, "event_frame_idxs") and ep.event_frame_idxs:
        event_frame_idxs = sorted(set(ep.event_frame_idxs))
    else:
        event_frame_idxs = [ep.event_frame_idx] if ep.event_frame_idx is not None else []

    task_text = task.get_task_text(inter)
    ws = task.get_world_state(inter)
    llp_commands = task.llp_commands

    rows: List[Dict[str, Any]] = []
    for t in range(ep.n_frames):
        # Output state for the current frame
        out_intra = intra_for_frame(t, event_frame_idxs, base_intra, max_intra)

        # Input state (Previous_Memory) is 1-step lagged.
        # For the first frame, we set in_intra = out_intra (no prior frame exists).
        if t == 0:
            in_intra = out_intra
        else:
            in_intra = intra_for_frame(t - 1, event_frame_idxs, base_intra, max_intra)

        # Previous memory comes from the *input* (lagged) state.
        in_progress = task.get_progress(inter, in_intra)
        in_world_state = ws  # keep raw ws for memory; do not apply done-merge here
        previous_memory = make_previous_memory_text(progress=in_progress, world_state=in_world_state)

        # Assistant output corresponds to the *current* state.
        command = task.get_command(inter, out_intra)
        progress = task.get_progress(inter, out_intra)

        world_state = ws
        if command == "done":
            world_state = merge_world_state_on_done(ws, progress)

        images = ep.get_frame_paths(cameras=cameras, t=t)
        user_text = render_user_prompt(task_text=task_text, previous_memory=previous_memory, avail_cmd_block=llp_commands)
        assistant_text = render_assistant_yaml(progress=progress, world_state=world_state, command=command)

        rows.append(
            {
                "uid": make_uid(task.task_id, ep.chunk, ep.episode, inter, base_intra, t),
                "task_id": task.task_id,
                "chunk": ep.chunk,
                "episode": ep.episode,
                "inter": inter,
                "base_intra": base_intra,
                "frame_idx": t,
                "event_frame_idx": event_frame_idxs[0] if event_frame_idxs else None,
                "event_frame_idxs": event_frame_idxs,
                "images": images,
                "conversations": [
                    {"from": "user", "value": user_text},
                    {"from": "assistant", "value": assistant_text},
                ],
                "meta": {
                    "data_episode_tasks": ep.tasks,
                    "episode_index": ep.episode_index,
                },
            }
        )

    return rows
