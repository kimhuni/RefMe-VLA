# helm_datasets/build_helm.py (v2)
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import List, Dict, Tuple

from helm_datasets_v2.core.registry import get_task_registry
from helm_datasets_v2.core.data_index import scan_data_episodes
from helm_datasets_v2.core.labeling import make_rows_for_task
from helm_datasets_v2.utils.io_utils import ensure_dir, atomic_write_text

""" 1 image
python -m helm_datasets.build_helm \
  --out_root "/data/ghkim/helm_data_v2/press_the_button_N_times_ep60" \
  --tasks press_blue_button_1 \
  --num_image 1 \
  --camera table \
  --val_ratio 0.1 \
  --shard_size 5000 \
  --require_event
"""

def _split_train_val_by_episode(episodes: List, val_ratio: float, seed: int) -> Tuple[List, List]:
    rng = random.Random(seed)
    idxs = list(range(len(episodes)))
    rng.shuffle(idxs)

    n_val = int(math.ceil(len(episodes) * val_ratio))
    val_set = set(idxs[:n_val])

    train_eps, val_eps = [], []
    for i, ep in enumerate(episodes):
        (val_eps if i in val_set else train_eps).append(ep)
    return train_eps, val_eps


def _shard_rows(rows: List[dict], shard_size: int) -> List[List[dict]]:
    if shard_size <= 0:
        return [rows]
    return [rows[i:i + shard_size] for i in range(0, len(rows), shard_size)]


def _write_jsonl(path: Path, rows: List[dict]) -> int:
    ensure_dir(path.parent)
    lines = [json.dumps(r, ensure_ascii=False) for r in rows]
    atomic_write_text(path, "\n".join(lines) + ("\n" if lines else ""))
    return len(rows)


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--out_root", type=str, required=True, help="HeLM root: frames_1hz/... + episode jsons")
    p.add_argument("--tasks", type=str, nargs="*", default=None, help="task_ids; default=all")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--shard_size", type=int, default=5000)
    p.add_argument("--require_event", action="store_true", help="skip episodes without event_frame_idx")

    # image views
    p.add_argument("--num_image", type=int, default=2, choices=[1, 2], help="use 1 or 2 camera images")
    p.add_argument("--camera", type=str, default="auto", choices=["auto", "table", "wrist"],
                   help="if n_images=1, choose which camera; auto=table")

    args = p.parse_args()

    out_root = Path(args.out_root)
    reg = get_task_registry()

    task_ids = sorted(reg.keys()) if not args.tasks else args.tasks
    episodes = scan_data_episodes(out_root)

    if not episodes:
        raise RuntimeError(f"No episodes found under out_root={out_root}")

    # episode 단위 split
    train_eps, val_eps = _split_train_val_by_episode(episodes, args.val_ratio, args.seed)

    # output root
    jsonl_root = out_root / "jsonl_v2"
    ensure_dir(jsonl_root)

    total = 0
    for task_id in task_ids:
        if task_id not in reg:
            raise KeyError(f"Unknown task_id='{task_id}'. Available: {sorted(reg.keys())[:20]} ...")
        spec = reg[task_id]
        spec.validate()

        # v2 rows
        buckets = make_rows_for_task(
            task_id=task_id,
            spec=spec,
            train_episodes=train_eps,
            val_episodes=val_eps,
            require_event=bool(args.require_event),
            num_image=int(args.num_image),
            camera=str(args.camera),
        )
        # buckets: {"detect":{"train":[...], "val":[...]}, "update":{...}}

        task_dir = jsonl_root / task_id
        detect_dir = task_dir / "detect"
        update_dir = task_dir / "update"
        ensure_dir(detect_dir)
        ensure_dir(update_dir)

        def write_split(mode_dir: Path, split_name: str, rows: List[dict]) -> int:
            cnt = 0
            for i, shard in enumerate(_shard_rows(rows, args.shard_size)):
                cnt += _write_jsonl(mode_dir / f"{split_name}-{i:05d}.jsonl", shard)
            return cnt

        n_dt = write_split(detect_dir, "train", buckets["detect"]["train"])
        n_dv = write_split(detect_dir, "val", buckets["detect"]["val"])
        n_ut = write_split(update_dir, "train", buckets["update"]["train"])
        n_uv = write_split(update_dir, "val", buckets["update"]["val"])

        meta = {
            "task_id": task_id,
            "schema_version": "v2",
            "counts": {
                "detect_train": n_dt,
                "detect_val": n_dv,
                "update_train": n_ut,
                "update_val": n_uv,
            },
            "images": {
                "n_images": int(args.n_images),
                "camera": str(args.camera),
            },
            "prev_memory_format": "Progress: ... | World_State: ...",
            "detect_output_keys": ["Event_Detected", "Command"],
            "update_output_keys": ["Progress", "World_State"],
            "llp_command": spec.llp_command,
            "init_memory": spec.init_memory,
        }
        atomic_write_text(task_dir / "meta.json", json.dumps(meta, ensure_ascii=False, indent=2))

        total += (n_dt + n_dv + n_ut + n_uv)
        print(f"[v2] {task_id}: detect {n_dt}/{n_dv}, update {n_ut}/{n_uv}")

    print(f"Done. Total rows: {total}")


if __name__ == "__main__":
    main()
