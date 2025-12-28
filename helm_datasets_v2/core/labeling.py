# helm_datasets/core/labeling.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from helm_datasets_v2.core.spec import TaskSpec
from helm_datasets_v2.core.templates import (
    make_prev_memory,
    render_user_prompt_detect,
    render_user_prompt_update,
    render_assistant_yaml_detect,
    render_assistant_yaml_update,
)


def _frame_path(cam_dir: Path, frame_idx: int) -> str:
    return str(cam_dir / f"frame_{frame_idx:06d}.jpg")


def _select_views(num_image: int, camera: str) -> List[str]:
    if num_image == 2:
        return ["table", "wrist"]
    # n_images == 1
    if camera == "wrist":
        return ["wrist"]
    # auto or table
    return ["table"]


def _images_for_episode(ep, frame_idx: int, views: List[str]) -> Dict[str, str]:
    out = {}
    for v in views:
        if v == "table":
            out["table"] = _frame_path(ep.table_dir, frame_idx)
        elif v == "wrist":
            out["wrist"] = _frame_path(ep.wrist_dir, frame_idx)
        else:
            raise ValueError(f"Unknown view: {v}")
    return out


def make_rows_for_task(
    task_id: str,
    spec: TaskSpec,
    train_episodes: List,
    val_episodes: List,
    require_event: bool,
    num_image: int,
    camera: str,
) -> Dict[str, Dict[str, List[dict]]]:
    """
    returns:
      {
        "detect": {"train":[...], "val":[...]},
        "update": {"train":[...], "val":[...]},
      }
    """
    views = _select_views(num_image=num_image, camera=camera)

    buckets = {
        "detect": {"train": [], "val": []},
        "update": {"train": [], "val": []},
    }

    def add_episode(ep, split: str):
        ev = getattr(ep, "event_frame_idx", None)
        n_frames = int(getattr(ep, "n_frames", 0) or 0)
        if n_frames <= 0:
            return
        if require_event and ev is None:
            return

        for inter in range(spec.max_inter + 1):
            task_text = spec.get_task_text(inter)
            ws = spec.get_world_state(inter)

            max_base = spec.max_intra[inter]
            for base_intra in range(0, max_base):
                # before/after progress
                prog_before = spec.progress_grid[inter][base_intra]
                prog_after = spec.progress_grid[inter][base_intra + 1]

                # detect command before / after
                cmd_before = spec.command_grid[inter][base_intra]
                cmd_after = spec.command_grid[inter][base_intra + 1]

                # -------- DETECT rows (all frames) --------
                for f in range(n_frames):
                    # event 이후 프레임은 이미 업데이트된 상태를 previous_memory로 본다
                    if ev is not None and f > ev:
                        mem_prog = prog_after
                        cmd = cmd_after
                    else:
                        mem_prog = prog_before
                        cmd = cmd_before

                    prev_mem = make_prev_memory(mem_prog, ws)
                    event_detected = (ev is not None and f == ev)

                    user_prompt = render_user_prompt_detect(
                        task_text=task_text,
                        llp_command=spec.get_llp_command(),
                        prev_memory=prev_mem,
                    )
                    gt_text = render_assistant_yaml_detect(event_detected, cmd)

                    row = {
                        "uid": f"{task_id}@{ep.chunk}-{ep.episode}-inter{inter}-base{base_intra}-f{f:06d}-detect",
                        "mode": "detect",
                        "task_id": task_id,
                        "chunk": ep.chunk,
                        "episode": ep.episode,
                        "inter": inter,
                        "base_intra": base_intra,
                        "frame_idx": f,
                        "event_frame_idx": ev,
                        "views": views,
                        "images": _images_for_episode(ep, f, views),
                        "user_prompt": user_prompt,
                        "gt_text": gt_text,
                        "meta": {
                            "data_episode_tasks": getattr(ep, "tasks", None),
                            "episode_index": getattr(ep, "episode_index", None),
                        },
                    }
                    buckets["detect"][split].append(row)

                # -------- UPDATE row (only event frame) --------
                if ev is None:
                    continue

                prev_mem_u = make_prev_memory(prog_before, ws)
                user_prompt_u = render_user_prompt_update(
                    task_text=task_text,
                    prev_memory=prev_mem_u,
                )
                gt_text_u = render_assistant_yaml_update(prog_after, ws)

                row_u = {
                    "uid": f"{task_id}@{ep.chunk}-{ep.episode}-inter{inter}-base{base_intra}-f{ev:06d}-update",
                    "mode": "update",
                    "task_id": task_id,
                    "chunk": ep.chunk,
                    "episode": ep.episode,
                    "inter": inter,
                    "base_intra": base_intra,
                    "frame_idx": ev,
                    "event_frame_idx": ev,
                    "views": views,
                    "images": _images_for_episode(ep, ev, views),
                    "user_prompt": user_prompt_u,
                    "gt_text": gt_text_u,
                    "meta": {
                        "data_episode_tasks": getattr(ep, "tasks", None),
                        "episode_index": getattr(ep, "episode_index", None),
                    },
                }
                buckets["update"][split].append(row_u)

    for ep in train_episodes:
        add_episode(ep, "train")
    for ep in val_episodes:
        add_episode(ep, "val")

    return buckets
