from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from helm_datasets_v3.core.registry import load_all_taskspecs
from helm_datasets_v3.core.spec import TaskSpecV3
from helm_datasets_v3.core.data_index import DataEpisodeV3, iter_all_episodes, load_data_episode
from helm_datasets_v3.core.io_utils import frame_path
from helm_datasets_v3.core.templates import dump_yaml, make_detect_prompt, make_update_prompt

"""
python -m helm_datasets_v3.build_helm \
  --out_root "/data/ghkim/helm_data/press_the_button_N_times_ep60" \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_v3/taskspecs" \
  --tasks press_blue_button_3+2 \
  --fps_out 5 \
  --n_images 1 \
  --val_ratio 0.1 \
  --shard_size 5000
"""


# ===== v3 constants =====
TRANSITION_FRAME_POS = 2  # fixed early frame index position (avoid black frame 0)


def episode_matches_filter(ep_meta: Dict[str, Any], flt: Dict[str, Any]) -> bool:
    """
    episode_filters는 {"tasks": "..."} 같이 episode.json의 메타 key/value를 매칭하는 용도.
    """
    for k, v in flt.items():
        if ep_meta.get(k) != v:
            return False
    return True


def split_episodes(pairs: List[tuple[str, str]], val_ratio: float, seed: int) -> tuple[List[tuple[str, str]], List[tuple[str, str]]]:
    rnd = random.Random(seed)
    pairs = list(pairs)
    rnd.shuffle(pairs)
    n_val = int(round(len(pairs) * val_ratio))
    val = pairs[:n_val]
    train = pairs[n_val:]
    return train, val


def shard_write_jsonl(rows: List[Dict[str, Any]], out_dir: Path, prefix: str, shard_size: int) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for i in range(0, len(rows), shard_size):
        shard = rows[i:i+shard_size]
        p = out_dir / f"{prefix}-{i//shard_size:05d}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in shard:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        paths.append(p)
    return paths


def make_uid(task_id: str, chunk: str, episode: str, split: str, kind: str, inter_idx: int, step_idx: int, frame_id: int) -> str:
    # kind: detect / update / transition
    return f"{task_id}@{chunk}-{episode}-{split}-{kind}-i{inter_idx}-s{step_idx}-f{frame_id:06d}"


def build_detect_rows_for_episode(
    spec: TaskSpecV3,
    ep: DataEpisodeV3,
    out_root: Path,
    fps_out: int,
    n_images: int,
    split: str,
    seed: int,
) -> List[Dict[str, Any]]:
    """
    Detect: memory_grid × frame -> Event flag.
    - inter별로 task_text[inter] 사용
    - memory_grid[inter]의 각 state를 모두 사용(원하면 나중에 옵션으로 줄일 수 있음)
    """
    rows: List[Dict[str, Any]] = []
    event_set = set(ep.event_frame_ids)

    for inter_idx in range(spec.inter + 1):
        task_text = spec.task_text[inter_idx]
        for mem_state_idx, mem in enumerate(spec.memory_grid[inter_idx]):
            prompt = make_detect_prompt(task_text, mem, n_images=n_images)
            # 모든 frame 사용 (정제된 프레임이므로 OK)
            for frame_id in ep.frame_ids:
                is_event = frame_id in event_set
                gt = dump_yaml({"Event": bool(is_event)})

                images = {
                    "table": str(frame_path(out_root, fps_out, ep.chunk, ep.episode, "table", frame_id))
                }
                if n_images == 2:
                    images["wrist"] = str(frame_path(out_root, fps_out, ep.chunk, ep.episode, "wrist", frame_id))

                rows.append({
                    "uid": make_uid(spec.task_id, ep.chunk, ep.episode, split, "detect", inter_idx, mem_state_idx, frame_id),
                    "task_id": spec.task_id,
                    "mode": "DETECT",
                    "chunk": ep.chunk,
                    "episode": ep.episode,
                    "inter": inter_idx,
                    "memory_state": mem_state_idx,
                    "frame_id": frame_id,
                    "event_frame_ids": ep.event_frame_ids,
                    "images": images,
                    "user_prompt": prompt,
                    "gt_text": gt,
                    "gt_yaml": {"Event": bool(is_event)},
                    "meta": {
                        "data_episode_tasks": ep.meta.get("tasks"),
                        "episode_index": ep.meta.get("episode_index"),
                    }
                })
    return rows


def select_transition_frame_id(ep: DataEpisodeV3) -> Optional[int]:
    if not ep.frame_ids:
        return None
    pos = TRANSITION_FRAME_POS
    if pos >= len(ep.frame_ids):
        pos = len(ep.frame_ids) - 1
    return ep.frame_ids[pos]


def build_update_rows_for_episode_step(
    spec: TaskSpecV3,
    ep: DataEpisodeV3,
    out_root: Path,
    fps_out: int,
    n_images: int,
    split: str,
    inter_idx: int,
    step_idx: int,
    make_transition_in_step0: bool = True,
) -> List[Dict[str, Any]]:
    """
    Update rows for a given (inter_idx, step_idx) for this episode pool.
    - intra update: memory_grid[inter][step] -> memory_grid[inter][step+1], replicated over ALL event frames.
    - additionally, if inter_idx==1 and step_idx==0 and make_transition_in_step0:
        transition update: (0,last)->(1,0) using fixed early frame (pos=2).
    """
    rows: List[Dict[str, Any]] = []
    task_text = spec.task_text[inter_idx]

    # (1) optional transition update (0,last)->(1,0)
    if make_transition_in_step0 and inter_idx == 1 and step_idx == 0:
        prev_mem, curr_mem = spec.transition_prev_curr()
        frame_id = select_transition_frame_id(ep)
        if frame_id is not None:
            prompt = make_update_prompt(task_text, prev_mem, n_images=n_images)
            gt = dump_yaml(curr_mem)

            images = {"table": str(frame_path(out_root, fps_out, ep.chunk, ep.episode, "table", frame_id))}
            if n_images == 2:
                images["wrist"] = str(frame_path(out_root, fps_out, ep.chunk, ep.episode, "wrist", frame_id))

            rows.append({
                "uid": make_uid(spec.task_id, ep.chunk, ep.episode, split, "transition", inter_idx, step_idx, frame_id),
                "task_id": spec.task_id,
                "mode": "UPDATE",
                "kind": "transition",
                "chunk": ep.chunk,
                "episode": ep.episode,
                "inter": inter_idx,
                "step": step_idx,
                "frame_id": frame_id,
                "images": images,
                "user_prompt": prompt,
                "gt_text": gt,
                "gt_yaml": curr_mem,
                "meta": {
                    "data_episode_tasks": ep.meta.get("tasks"),
                    "episode_index": ep.meta.get("episode_index"),
                }
            })

    # (2) intra update replicated over ALL event frames
    prev_mem, curr_mem = spec.prev_curr_for_step(inter_idx, step_idx)
    prompt = make_update_prompt(task_text, prev_mem, n_images=n_images)
    gt = dump_yaml(curr_mem)

    # event_frame_ids가 비어있으면 update를 만들 수 없음 (event 복제 정책이므로)
    for frame_id in ep.event_frame_ids:
        images = {"table": str(frame_path(out_root, fps_out, ep.chunk, ep.episode, "table", frame_id))}
        if n_images == 2:
            images["wrist"] = str(frame_path(out_root, fps_out, ep.chunk, ep.episode, "wrist", frame_id))

        rows.append({
            "uid": make_uid(spec.task_id, ep.chunk, ep.episode, split, "update", inter_idx, step_idx, frame_id),
            "task_id": spec.task_id,
            "mode": "UPDATE",
            "kind": "intra",
            "chunk": ep.chunk,
            "episode": ep.episode,
            "inter": inter_idx,
            "step": step_idx,
            "frame_id": frame_id,
            "event_frame_ids": ep.event_frame_ids,
            "images": images,
            "user_prompt": prompt,
            "gt_text": gt,
            "gt_yaml": curr_mem,
            "meta": {
                "data_episode_tasks": ep.meta.get("tasks"),
                "episode_index": ep.meta.get("episode_index"),
            }
        })

    return rows


def build_for_task(
    spec: TaskSpecV3,
    out_root: Path,
    fps_out: int,
    n_images: int,
    val_ratio: float,
    seed: int,
    shard_size: int,
    taskspecs_dir: Path,
) -> None:
    # 전체 episode 목록
    all_pairs = iter_all_episodes(out_root, fps_out=fps_out)
    train_pairs, val_pairs = split_episodes(all_pairs, val_ratio=val_ratio, seed=seed)

    # split 별 에피소드 로딩 캐시 (필요시)
    def load_pairs(pairs: List[tuple[str, str]]) -> List[DataEpisodeV3]:
        episodes: List[DataEpisodeV3] = []
        for chunk, episode in pairs:
            ep = load_data_episode(out_root, fps_out, chunk, episode, use_wrist=(n_images == 2))
            episodes.append(ep)
        return episodes

    train_eps = load_pairs(train_pairs)
    val_eps = load_pairs(val_pairs)

    # output root
    task_out = out_root / "jsonl_v3" / spec.task_id
    detect_out = task_out / "detect"
    update_out = task_out / "update"

    # ===== DETECT =====
    detect_train_rows: List[Dict[str, Any]] = []
    detect_val_rows: List[Dict[str, Any]] = []

    for split_name, eps in [("train", train_eps), ("val", val_eps)]:
        for ep in eps:
            # detect는 episode_filters 적용할 수도 있지만, 우선 전체에서 생성(정제된 에피소드만 있다면 OK)
            rows = build_detect_rows_for_episode(spec, ep, out_root, fps_out, n_images, split_name, seed)
            if split_name == "train":
                detect_train_rows.extend(rows)
            else:
                detect_val_rows.extend(rows)

    shard_write_jsonl(detect_train_rows, detect_out, "train", shard_size)
    shard_write_jsonl(detect_val_rows, detect_out, "val", shard_size)

    # ===== UPDATE =====
    update_train_rows: List[Dict[str, Any]] = []
    update_val_rows: List[Dict[str, Any]] = []

    # inter/step별로 episode_filters를 적용해서 해당 pool에 대해 update 생성
    def build_update_for_split(eps: List[DataEpisodeV3], split_name: str) -> List[Dict[str, Any]]:
        out_rows: List[Dict[str, Any]] = []
        for inter_idx in range(spec.inter + 1):
            for step_idx in range(spec.intra[inter_idx]):
                flt = spec.episode_filters[inter_idx][step_idx]
                for ep in eps:
                    if not episode_matches_filter(ep.meta, flt):
                        continue
                    out_rows.extend(build_update_rows_for_episode_step(
                        spec, ep, out_root, fps_out, n_images, split_name,
                        inter_idx=inter_idx, step_idx=step_idx,
                        make_transition_in_step0=True,   # 고정(나중에 argparse 옵션화)
                    ))
        return out_rows

    update_train_rows = build_update_for_split(train_eps, "train")
    update_val_rows = build_update_for_split(val_eps, "val")

    shard_write_jsonl(update_train_rows, update_out, "train", shard_size)
    shard_write_jsonl(update_val_rows, update_out, "val", shard_size)

    # meta 저장
    meta = {
        "task_id": spec.task_id,
        "fps_out": fps_out,
        "n_images": n_images,
        "val_ratio": val_ratio,
        "seed": seed,
        "transition_frame_pos": TRANSITION_FRAME_POS,
        "note": "v3: DETECT=memory_grid×frame, UPDATE=intra over all event frames + transition (0,last)->(1,0) at inter=1 step0.",
    }
    (task_out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", type=str, required=True)
    p.add_argument("--taskspecs_dir", type=str, required=True)
    p.add_argument("--tasks", type=str, nargs="+", required=True, help="task_id list")
    p.add_argument("--fps_out", type=int, default=5, help="frames_{fps}hz")
    p.add_argument("--n_images", type=int, default=2, choices=[1, 2], help="1=table only, 2=table+wrist")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--shard_size", type=int, default=5000)
    args = p.parse_args()

    out_root = Path(args.out_root)
    taskspecs_dir = Path(args.taskspecs_dir)

    specs = load_all_taskspecs(taskspecs_dir)
    for task_id in args.tasks:
        if task_id not in specs:
            raise KeyError(f"task_id not found in taskspecs_dir: {task_id}")
        build_for_task(
            specs[task_id],
            out_root=out_root,
            fps_out=args.fps_out,
            n_images=args.n_images,
            val_ratio=args.val_ratio,
            seed=args.seed,
            shard_size=args.shard_size,
            taskspecs_dir=taskspecs_dir,
        )

    print("[v3] build complete.")


if __name__ == "__main__":
    main()