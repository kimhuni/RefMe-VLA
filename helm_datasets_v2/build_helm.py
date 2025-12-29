# helm_datasets_v2/build_helm.py
from __future__ import annotations

import argparse
from pathlib import Path
import random

from helm_datasets_v2.core.spec import load_taskspec
from helm_datasets_v2.core.episode import discover_episodes, load_data_episode
from helm_datasets_v2.core.segment import assign_episode_to_step
from helm_datasets_v2.core.state import validate_episode_against_spec
from helm_datasets_v2.core.rows import build_detect_rows, build_update_rows, build_task_change_update_rows
from helm_datasets_v2.core.io import write_jsonl_sharded

"""
export PYTHONPATH=$(pwd)
python -m helm_datasets_v2.build_helm \
  --out_root "/data/ghkim/helm_data/wipe_the_window" \
  --tasks wipe_the_window \
  --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_v2/taskspecs/wipe_the_window" \
  --camera table \
  --val_ratio 0.1 \
  --shard_size 5000 \
  --require_event \
  --negatives_per_positive -1
"""

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", type=str, required=True)
    p.add_argument("--tasks", type=str, nargs="+", required=True)
    p.add_argument("--taskspecs_dir", type=str, required=True)

    p.add_argument("--camera", type=str, default="table")
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--shard_size", type=int, default=5000)
    p.add_argument("--require_event", action="store_true")

    p.add_argument("--negatives_per_positive", type=int, default=None)
    p.add_argument("--min_gap", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.out_root)
    specs_dir = Path(args.taskspecs_dir)

    rng = random.Random(args.seed)

    for task_id in args.tasks:
        spec_path = specs_dir / f"{task_id}.json"
        if not spec_path.exists():
            raise FileNotFoundError(f"taskspec not found: {spec_path}")

        spec = load_taskspec(spec_path)

        # CLI override
        spec = spec.__class__(**{
            **spec.__dict__,
            "camera": args.camera or spec.camera,
            "require_event": bool(args.require_event) or spec.require_event,
            "negatives_per_positive": args.negatives_per_positive if args.negatives_per_positive is not None else spec.negatives_per_positive,
            "min_gap": args.min_gap if args.min_gap is not None else spec.min_gap,
        })

        episode_dirs = discover_episodes(out_root, camera=spec.camera)
        if not episode_dirs:
            raise RuntimeError(f"No episodes found under out_root={out_root} (camera={spec.camera})")

        episode_dirs = list(episode_dirs)
        rng.shuffle(episode_dirs)
        n_val = int(len(episode_dirs) * args.val_ratio)
        val_set = set(episode_dirs[:n_val])
        splits = [("val", val_set), ("train", set(episode_dirs[n_val:]))]

        for split, split_dirs in splits:
            detect_rows_all = []
            update_rows_all = []

            for ep_dir in split_dirs:
                ep = load_data_episode(out_root, ep_dir, camera=spec.camera, event_frames_key=spec.event_frames_key)

                # Episode-level assignment to a step (inter,intra) based on ep.task_str
                assign = assign_episode_to_step(ep.task_str, spec.episode_filters)
                if assign is None:
                    continue

                try:
                    validate_episode_against_spec(spec, assign, ep.event_frames)
                except ValueError:
                    continue

                detect_rows_all.extend(list(build_detect_rows(spec, ep, assign, split=split, seed=args.seed)))
                update_rows_all.extend(list(build_update_rows(spec, ep, assign, split=split)))

            # Spec-only task-change updates (inter boundaries)
            # Add a few per split (can be many duplicates otherwise; here we add exactly one set)
            update_rows_all.extend(list(build_task_change_update_rows(spec, split=split)))

            base = out_root / "jsonl_v2" / task_id
            detect_dir = base / "detect"
            update_dir = base / "update"

            write_jsonl_sharded(detect_rows_all, detect_dir, split=split, shard_size=args.shard_size)
            write_jsonl_sharded(update_rows_all, update_dir, split=split, shard_size=args.shard_size)

            print(f"[{task_id}] {split}: detect={len(detect_rows_all)} update={len(update_rows_all)} -> {base}")


if __name__ == "__main__":
    main()