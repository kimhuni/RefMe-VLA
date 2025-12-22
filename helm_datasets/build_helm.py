from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List, Tuple

from helm_datasets.utils.io_utils import ensure_dir, atomic_write_text
from helm_datasets.core.data_index import scan_data_episodes
from helm_datasets.core.registry import get_task_registry
from helm_datasets.core.labeling import make_rows_for_variant

"""

press N + m times: "press_blue_button_1+1","press_blue_button_1+2","press_blue_button_1+3","press_blue_button_2+1","press_blue_button_2+2","press_blue_button_2+3","press_blue_button_3+1","press_blue_button_3+2","press_blue_button_3+3" \
  
################################################################################

export PYTHONPATH=$(pwd)

python -m helm_datasets.build_helm \
  --out_root "/data/ghkim/helm_data/press_the_button_N_times_ep60" \
  --tasks "press_blue_button_1+1","press_blue_button_1+2","press_blue_button_1+3","press_blue_button_2+1","press_blue_button_2+2","press_blue_button_2+3","press_blue_button_3+1","press_blue_button_3+2","press_blue_button_3+3" \
  --val_ratio 0.1 \
  --shard_size 5000
"""


def write_jsonl(path: Path, rows: List[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def episode_split(
    episodes: List[Tuple[str, str]],  # (chunk, episode)
    seed: int,
    val_ratio: float,
) -> Tuple[set, set]:
    keys = list(dict.fromkeys(episodes))
    rnd = random.Random(seed)
    rnd.shuffle(keys)
    n_val = int(len(keys) * val_ratio)
    val = set(keys[:n_val])
    train = set(keys[n_val:])
    return train, val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--taskspecs_dir", type=str, default="/home/ghkim/codes/RefMe-VLA/helm_datasets/taskspecs")
    ap.add_argument("--tasks", type=str, default=None, help="Comma-separated task_ids to build")
    ap.add_argument("--require_event", type=int, default=1)
    ap.add_argument("--cameras", type=str, default="table,wrist")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--shard_size", type=int, default=5000)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    taskspecs_dir = Path(args.taskspecs_dir) if args.taskspecs_dir else None
    cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
    require_event = bool(args.require_event)

    episodes = scan_data_episodes(out_root=out_root, cameras=cameras, require_event=require_event)
    if not episodes:
        raise RuntimeError("No data episodes found. Did you run extract_frames + annotate_app?")

    reg = get_task_registry(taskspecs_dir=taskspecs_dir, tasks=args.tasks)
    if not reg:
        raise RuntimeError("No taskspecs loaded.")

    ep_keys = [(ep.chunk, ep.episode) for ep in episodes]
    train_set, val_set = episode_split(ep_keys, seed=args.seed, val_ratio=args.val_ratio)

    out_jsonl_dir = out_root / "jsonl"
    ensure_dir(out_jsonl_dir)

    for task_id, task in reg.items():
        task.validate()

        train_rows: List[dict] = []
        val_rows: List[dict] = []

        for inter in range(task.max_inter + 1):
            filt = task.episode_filters[inter] if task.episode_filters else {}
            wanted_tasks = filt.get("tasks", None) if isinstance(filt, dict) else None

            pool = episodes if wanted_tasks is None else [ep for ep in episodes if ep.tasks == wanted_tasks]

            for ep in pool:
                for base_intra in range(0, task.max_intra[inter]):
                    rows = make_rows_for_variant(ep=ep, task=task, inter=inter, base_intra=base_intra, cameras=cameras)
                    if (ep.chunk, ep.episode) in val_set:
                        val_rows.extend(rows)
                    else:
                        train_rows.extend(rows)

        task_dir = out_jsonl_dir / task_id
        ensure_dir(task_dir)

        def shard(rows: List[dict], split: str) -> None:
            if not rows:
                return
            for i in range(0, len(rows), args.shard_size):
                shard_rows = rows[i : i + args.shard_size]
                p = task_dir / f"{split}-{i//args.shard_size:05d}.jsonl"
                write_jsonl(p, shard_rows)

        shard(train_rows, "train")
        shard(val_rows, "val")

        meta = {
            "task_id": task_id,
            "num_data_episodes_total": len(episodes),
            "num_rows_train": len(train_rows),
            "num_rows_val": len(val_rows),
            "val_ratio": args.val_ratio,
            "cameras": cameras,
            "require_event": require_event,
            "schema": {
                "assistant": "YAML (Progress, World_State, Command)",
                "world_state_type": "string_or_null",
                "done_included": True,
                "framewise_intra": "t < event => base_intra, else base_intra+1",
            },
        }
        atomic_write_text(task_dir / "meta.json", json.dumps(meta, ensure_ascii=False, indent=2))

        print(f"[{task_id}] train={len(train_rows)} val={len(val_rows)} -> {task_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
