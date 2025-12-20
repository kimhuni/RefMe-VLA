from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional, List

from helm_datasets.utils.io_utils import ensure_dir, jsonl_dir, atomic_write_text
from helm_datasets.core.data_index import scan_data_episodes
from helm_datasets.core.registry import get_task_registry
from helm_datasets.core.labeling import make_sample, sample_to_jsonl_dict

"""
python -m helm_datasets.build_helm \
  --out_root /data/ghkim/helm_data/press_the_blue_button_one_time_test_ep3 \
  --require_event \
  --camera table \
  --use_event_frame \
  --shard_by_state \
  --tasks press_blue_button_3
"""

def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    ensure_dir(path.parent)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True, help="dataset root containing frames_1hz/")
    ap.add_argument("--require_event", action="store_true", help="skip episodes without <episode>.json")
    ap.add_argument("--camera", type=str, default="table", choices=["table", "wrist"])
    ap.add_argument("--use_event_frame", default=False, action="store_true", help="use event_frame_idx as representative image")
    ap.add_argument("--tasks", type=str, help="comma-separated task_ids to build (empty=all)")
    ap.add_argument("--shard_by_state", action="store_true", help="write separate jsonl per (task, inter, intra)")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    episodes = scan_data_episodes(out_root, require_event=args.require_event)
    if not episodes:
        raise SystemExit("No data episodes found. Check out_root and episode json files.")

    registry = get_task_registry(tasks=args.tasks)
    if args.tasks.strip():
        allow = set([t.strip() for t in args.tasks.split(",") if t.strip()])
        registry = {k: v for k, v in registry.items() if k in allow}

    if not registry:
        raise SystemExit("No tasks selected.")

    out_dir = jsonl_dir(out_root)
    ensure_dir(out_dir)

    total = 0
    for task_id, task in registry.items():
        task.validate()

        if args.shard_by_state:
            # state별 shard
            for inter, intra in task.states():
                rows = []
                for ep in episodes:
                    s = make_sample(
                        ep=ep,
                        task=task,
                        inter=inter,
                        intra=intra,
                        camera=args.camera,
                        use_event_frame=args.use_event_frame,
                    )
                    rows.append(sample_to_jsonl_dict(s))

                p = out_dir / task_id / f"inter{inter}_intra{intra}.jsonl"
                cnt = write_jsonl(p, rows)
                total += cnt
                print(f"[Wrote] {p}  ({cnt} rows)")
        else:
            # task별 단일 파일
            rows = []
            for inter, intra in task.states():
                for ep in episodes:
                    s = make_sample(
                        ep=ep,
                        task=task,
                        inter=inter,
                        intra=intra,
                        camera=args.camera,
                        use_event_frame=args.use_event_frame,
                    )
                    rows.append(sample_to_jsonl_dict(s))
            p = out_dir / f"{task_id}.jsonl"
            cnt = write_jsonl(p, rows)
            total += cnt
            print(f"[Wrote] {p}  ({cnt} rows)")

    print(f"Done. Total samples: {total}")


if __name__ == "__main__":
    main()