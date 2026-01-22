"""
build_visual_memory.py (Revised)

- Fix: Accumulates data from all tasks before saving to prevent overwriting.
- Fix: Merges Detect and Update data into single 'train.jsonl' and 'val.jsonl'.
- Feature: Explicitly adds 'label' field for MixedBatchSampler.

Usage:
export PYTHONPATH=$(pwd)
python helm_datasets_video/build_videohelm.py \
    --data_root "/data/ghkim/helm_data/press_button_in_order" \
    --out_root "/data/ghkim/helm_data/press_button_in_order" \
    --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_video/taskspec/press_in_order"
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from helm_datasets_v4.core.registry import load_all_taskspecs
from helm_datasets_v4.core.spec import TaskSpecV4
from helm_datasets_v4.core.data_index import iter_all_episodes, load_data_episode, DataEpisodeV4
from helm_datasets_v4.core.io_utils import frame_path
from helm_datasets_video.templates_visual import make_detect_prompt, make_update_prompt, dump_yaml

# --- Helper: Episode Pool ---
class EpisodePool:
    def __init__(self):
        # pool[inter_idx][step_idx] = List[DataEpisodeV4]
        self.pool: Dict[int, Dict[int, List[DataEpisodeV4]]] = defaultdict(lambda: defaultdict(list))

    def add(self, inter: int, step: int, ep: DataEpisodeV4):
        self.pool[inter][step].append(ep)

    def sample_event_frame(self, inter: int, step: int, data_root: Path, fps_out: int, rng: random.Random) -> str:
        episodes = self.pool[inter][step]
        if not episodes: return None
        ep = rng.choice(episodes)
        if not ep.event_frame_ids:
            fid = ep.frame_ids[-1]
        else:
            fid = rng.choice(ep.event_frame_ids)
        return str(frame_path(data_root, fps_out, ep.chunk, ep.episode, "table", fid))

    def sample_start_frame(self, inter: int, step: int, data_root: Path, fps_out: int, rng: random.Random) -> str:
        episodes = self.pool[inter][step]
        if not episodes: return None
        ep = rng.choice(episodes)
        fid = ep.frame_ids[min(10, len(ep.frame_ids)-1)]
        return str(frame_path(data_root, fps_out, ep.chunk, ep.episode, "table", fid))


# --- Row Builders ---

def build_detect_rows(
    spec: TaskSpecV4,
    ep: DataEpisodeV4,
    data_root: Path,
    fps_out: int,
    split: str,
    inter_idx: int,
    step_idx: int
) -> List[Dict[str, Any]]:
    rows = []
    task_text = spec.task_text[inter_idx]
    curr_mem = spec.memory_grid[inter_idx][step_idx]
    action_command = curr_mem.get("Action_Command", "None")

    event_set = set(ep.event_frame_ids)
    step_event = "none"
    if spec.event_grid is not None:
        step_event = spec.event_grid[inter_idx][step_idx]

    prompt = make_detect_prompt(task_text, action_command, spec.event_list)

    for frame_id in ep.frame_ids:
        is_event = frame_id in event_set
        event_str = step_event if is_event else "none"

        # Explicit Labeling for Sampler
        label = "detect_pos" if is_event else "detect_neg"

        gt_yaml = {"Event_Detected": is_event, "Event": event_str}
        image_path = str(frame_path(data_root, fps_out, ep.chunk, ep.episode, "table", frame_id))

        rows.append({
            "uid": f"{spec.task_id}@{ep.episode}-detect-i{inter_idx}-s{step_idx}-f{frame_id}",
            "mode": "DETECT",
            "label": label,  # <--- Added
            "images": [image_path],
            "user_prompt": prompt,
            "gt_text": dump_yaml(gt_yaml),
            "meta": {"split": split, "task": spec.task_id}
        })
    return rows


def build_update_rows(
    spec: TaskSpecV4,
    pool: EpisodePool,
    data_root: Path,
    fps_out: int,
    rng: random.Random,
    split: str,
    inter_idx: int,
    target_step_idx: int,
    augment_factor: int = 20
) -> List[Dict[str, Any]]:

    rows = []
    task_text = spec.task_text[inter_idx]
    target_mem = spec.memory_grid[inter_idx][target_step_idx]
    target_action = target_mem.get("Action_Command", "done")

    if target_step_idx == 0:
        prev_action = "None"
        curr_event = "None"
    else:
        prev_step_idx = target_step_idx - 1
        prev_mem = spec.memory_grid[inter_idx][prev_step_idx]
        prev_action = prev_mem.get("Action_Command", "None")
        curr_event = "none"
        if spec.event_grid:
            curr_event = spec.event_grid[inter_idx][prev_step_idx]

    prompt = make_update_prompt(
        task_text, prev_action, curr_event, spec.llp_commands
    )
    gt_yaml = {"Action_Command": target_action}

    target_episodes = pool.pool[inter_idx][target_step_idx]

    # Validation: if no episodes for this step, skip
    if not target_episodes:
        return []

    for ep in target_episodes:
        for _ in range(augment_factor):
            instance_history = []

            # Start Frame
            s_img = pool.sample_start_frame(inter_idx, 0, data_root, fps_out, rng)
            if s_img: instance_history.append(s_img)

            # Event History
            valid_hist = True
            for step_i in range(target_step_idx):
                e_img = pool.sample_event_frame(inter_idx, step_i, data_root, fps_out, rng)
                if e_img:
                    instance_history.append(e_img)
                else:
                    valid_hist = False
                    break

            if not valid_hist: continue

            unique_id = rng.randint(0, 9999999)
            rows.append({
                "uid": f"{spec.task_id}-update-i{inter_idx}-s{target_step_idx}-{ep.episode}-aug{unique_id}",
                "mode": "UPDATE",
                "label": "update",  # <--- Added
                "images": instance_history,
                "user_prompt": prompt,
                "gt_text": dump_yaml(gt_yaml),
                "meta": {"split": split, "task": spec.task_id, "aug": True}
            })

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--taskspecs_dir", type=str, required=True)
    parser.add_argument("--fps_out", type=int, default=5)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    # Augmentation options
    parser.add_argument("--train_aug", type=int, default=30, help="Augment factor for train set")
    parser.add_argument("--val_aug", type=int, default=5, help="Augment factor for val set")

    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    specs = load_all_taskspecs(Path(args.taskspecs_dir))

    rng = random.Random(args.seed)

    # -------------------------------------------------------
    # 1. Global Containers for Merged Data
    # -------------------------------------------------------
    all_train_rows = []
    all_val_rows = []

    print("[Info] Starting Visual Memory Dataset Build...")

    for task_id, spec in specs.items():
        print(f"Processing Task: {task_id}")

        # Build Pool
        pool = EpisodePool()
        all_pairs = list(iter_all_episodes(data_root, fps_out=args.fps_out))
        rng.shuffle(all_pairs)

        n_val = int(len(all_pairs) * args.val_ratio)
        val_pairs = all_pairs[:n_val]
        train_pairs = all_pairs[n_val:]

        def process_dataset(pairs, split_name, aug_factor):
            # A. Build local pool for this split
            split_pool = EpisodePool()
            for chunk, ep_name in pairs:
                ep = load_data_episode(data_root, args.fps_out, chunk, ep_name, use_wrist=False)
                for inter_i in range(spec.inter + 1):
                    for step_i in range(spec.intra[inter_i]):
                        flt = spec.episode_filters[inter_i][step_i]
                        match = True
                        for k, v in flt.items():
                            if ep.meta.get(k) != v: match = False; break
                        if match:
                            split_pool.add(inter_i, step_i, ep)

            # B. Generate Rows
            local_rows = []

            # Detect
            for inter_i in range(spec.inter + 1):
                for step_i in range(spec.intra[inter_i]):
                    target_eps = split_pool.pool[inter_i][step_i]
                    for ep in target_eps:
                        local_rows.extend(build_detect_rows(
                            spec, ep, data_root, args.fps_out, split_name, inter_i, step_i
                        ))

            # Update (with Augmentation)
            for inter_i in range(spec.inter + 1):
                num_states = len(spec.memory_grid[inter_i])
                for step_i in range(num_states):
                    local_rows.extend(build_update_rows(
                        spec, split_pool, data_root, args.fps_out, rng, split_name, inter_i, step_i,
                        augment_factor=aug_factor
                    ))
            return local_rows

        # Process Splits and Append to Global List
        task_train = process_dataset(train_pairs, "train", args.train_aug)
        task_val = process_dataset(val_pairs, "val", args.val_aug)

        all_train_rows.extend(task_train)
        all_val_rows.extend(task_val)

        print(f"  > Added {len(task_train)} train rows, {len(task_val)} val rows.")

    # -------------------------------------------------------
    # 2. Save Merged Files
    # -------------------------------------------------------
    out_dir = out_root / "visual_memory_jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_jsonl(rows, filename):
        path = out_dir / filename
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[Save] Saved {len(rows)} total rows to {path}")

    save_jsonl(all_train_rows, "train.jsonl")
    save_jsonl(all_val_rows, "val.jsonl")

    print("[Done] Build Complete.")

if __name__ == "__main__":
    main()