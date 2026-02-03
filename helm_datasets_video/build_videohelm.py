"""
build_visual_memory.py (Revised)

- Fix: Accumulates data from all tasks before saving to prevent overwriting.
- Fix: Merges Detect and Update data into single 'train.jsonl' and 'val.jsonl'.
- Feature: Explicitly adds 'label' field for MixedBatchSampler.

Usage:
export PYTHONPATH=$(pwd)
python helm_datasets_video/build_videohelm.py \
    --data_root "/data/ghkim/helm_data/press_button_in_order" \
    --out_root "/data/ghkim/helm_data/press_button_in_order/extended" \
    --taskspecs_dir "/home/ghkim/codes/RefMe-VLA/helm_datasets_video/taskspecs/press_button_in_order_extended"
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


class EpisodePool:
    def __init__(self):
        # pool[inter][step] = List[Episode]
        self.pool: Dict[int, Dict[int, List[DataEpisodeV4]]] = defaultdict(lambda: defaultdict(list))

    def add(self, inter: int, step: int, ep: DataEpisodeV4):
        self.pool[inter][step].append(ep)

    def sample_event_frame(self, inter: int, step: int, data_root: Path, fps_out: int, rng: random.Random) -> str:
        episodes = self.pool[inter][step]
        if not episodes: return None
        ep = rng.choice(episodes)
        if ep.event_frame_ids and len(ep.event_frame_ids) > 0:
            fid = rng.choice(ep.event_frame_ids)
        else:
            fid = ep.frame_ids[-1]
        return str(frame_path(data_root, fps_out, ep.chunk, ep.episode, "table", fid))

    def sample_start_frame(self, inter: int, step: int, data_root: Path, fps_out: int, rng: random.Random) -> str:
        episodes = self.pool[inter][step]
        if not episodes: return None
        ep = rng.choice(episodes)
        fid = ep.frame_ids[rng.randint(0, min(10, len(ep.frame_ids) - 1))]
        return str(frame_path(data_root, fps_out, ep.chunk, ep.episode, "table", fid))

    # [NEW] 특정 Inter 단계의 전체 히스토리(Start + All Events)를 가져오는 함수
    def sample_full_history(self, inter: int, num_steps: int, data_root: Path, fps_out: int, rng: random.Random) -> \
    List[str]:
        """
        과거 Inter 단계의 완성된 히스토리를 생성.
        해당 Inter의 마지막 단계(보통 Done 직전) 에피소드 풀을 사용하여
        Start Frame + Event Frames (step 0 ~ num_steps-1)를 모두 가져옴.
        """
        # 마지막 스텝(완료 상태)의 에피소드를 가져와야 전체 이벤트를 알 수 있음
        # 하지만 pool 구조상 각 step별로 에피소드가 나뉘어 있음.
        # 가장 좋은 건 '마지막 step'에 있는 에피소드를 하나 골라서, 그 에피소드의 전체 흐름을 가져오는 것.
        # 여기서는 pool[inter][last_step]에서 하나를 뽑음.

        last_step_pool = self.pool[inter][num_steps - 1]  # intra 마지막
        if not last_step_pool:
            # Fallback: 아무거나 0번 스텝에서라도 가져옴 (데이터 부족 시)
            last_step_pool = self.pool[inter][0]

        if not last_step_pool: return []

        ep = rng.choice(last_step_pool)

        history = []
        # 1. Start Frame
        fid_start = ep.frame_ids[rng.randint(0, min(10, len(ep.frame_ids) - 1))]
        history.append(str(frame_path(data_root, fps_out, ep.chunk, ep.episode, "table", fid_start)))

        # 2. All Event Frames in that episode
        # 주의: DataEpisodeV4에 저장된 event_frame_ids는 해당 step의 이벤트만 있을 수도 있고 전체일 수도 있음.
        # HeLM 데이터 구조상 보통 해당 step의 이벤트만 가지고 있을 확률이 높음.
        # 따라서, 과거 inter의 '모든 스텝'을 순회하며 각각 이미지를 뽑아야 함.

        # 수정 전략: pool[inter][0] ~ pool[inter][N]을 순회하며 각각 이미지를 뽑음.
        # (서로 다른 에피소드에서 짜집기 하게 됨 -> Data Augmentation 효과)
        history = []

        # Start (from step 0)
        s_img = self.sample_start_frame(inter, 0, data_root, fps_out, rng)
        if s_img: history.append(s_img)

        # Events (0 to num_steps-1)
        for s in range(num_steps):
            e_img = self.sample_event_frame(inter, s, data_root, fps_out, rng)
            if e_img: history.append(e_img)

        return history


def build_detect_rows(
        spec: TaskSpecV4,
        ep: DataEpisodeV4,
        data_root: Path,
        fps_out: int,
        split: str,
        inter_idx: int,
        step_idx: int
) -> List[Dict[str, Any]]:
    # Detect는 현재 프레임만 보므로 Inter history가 필요 없음 (단일 이미지)
    rows = []
    task_text = spec.task_text[inter_idx]
    curr_mem = spec.memory_grid[inter_idx][step_idx]
    action_command = curr_mem.get("Action_Command", "None")

    event_ids = ep.event_frame_ids
    if not event_ids and hasattr(ep, 'event_frame_idx'):
        event_ids = [ep.event_frame_idx]

    event_set = set(event_ids)
    step_event = "none"
    if spec.event_grid is not None:
        step_event = spec.event_grid[inter_idx][step_idx]

    prompt = make_detect_prompt(task_text, action_command, spec.event_list)

    for frame_id in ep.frame_ids:
        is_event = frame_id in event_set
        event_str = step_event if is_event else "none"
        label = "detect_pos" if is_event else "detect_neg"

        gt_yaml = {"Event_Detected": is_event, "Event": event_str}
        image_path = str(frame_path(data_root, fps_out, ep.chunk, ep.episode, "table", frame_id))

        rows.append({
            "uid": f"{spec.task_id}@{ep.episode}-detect-i{inter_idx}-s{step_idx}-f{frame_id}",
            "mode": "DETECT",
            "label": label,
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
    target_action = str(target_mem.get("Action_Command", "done")).strip()

    # Augmentation Factor (Done 10배 삭제됨, 1:1 유지)
    effective_aug = augment_factor
    is_done_step = (target_action.lower() == "done")
    if is_done_step:
        effective_aug = augment_factor  # 1배 (균형)

    # Previous State Info
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

    # Pool Selection (Done fallback)
    pool_idx = target_step_idx
    if target_step_idx not in pool.pool[inter_idx]:
        pool_idx = target_step_idx - 1

    target_episodes = pool.pool[inter_idx][pool_idx]
    if not target_episodes: return []

    for ep in target_episodes:
        for _ in range(effective_aug):
            instance_history = []

            # ------------------------------------------------------------------
            # [CRITICAL] 1. Past Inter-Episodes History Injection
            # 현재 inter가 1이라면, inter=0의 모든 히스토리를 먼저 넣어야 함.
            # ------------------------------------------------------------------
            for past_inter in range(inter_idx):
                # 과거 inter의 총 스텝 수 (Action 단계 수)
                # intra에는 [1, 1] 처럼 들어있음. past_inter의 intra 값만큼 Event가 있었을 것.
                num_past_steps = spec.intra[past_inter]

                # 과거의 Start + Event 프레임들을 가져옴
                past_history = pool.sample_full_history(past_inter, num_past_steps, data_root, fps_out, rng)
                instance_history.extend(past_history)

            # ------------------------------------------------------------------
            # 2. Current Inter-Episode History
            # ------------------------------------------------------------------
            # Start Frame (현재 Inter의 시작)
            s_img = pool.sample_start_frame(inter_idx, 0, data_root, fps_out, rng)
            if s_img: instance_history.append(s_img)

            # Event History Loop (Current Inter)
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
                "label": "update",
                "images": instance_history,  # Past + Current History
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
    parser.add_argument("--train_aug", type=int, default=30)
    parser.add_argument("--val_aug", type=int, default=5)

    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    specs = load_all_taskspecs(Path(args.taskspecs_dir))

    rng = random.Random(args.seed)
    all_train_rows = []
    all_val_rows = []

    print("[Info] Starting Visual Memory Dataset Build...")

    for task_id, spec in specs.items():
        print(f"Processing Task: {task_id}")

        pool = EpisodePool()
        all_pairs = list(iter_all_episodes(data_root, fps_out=args.fps_out))
        rng.shuffle(all_pairs)

        n_val = int(len(all_pairs) * args.val_ratio)
        val_pairs = all_pairs[:n_val]
        train_pairs = all_pairs[n_val:]

        def process_dataset(pairs, split_name, aug_factor):
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

        task_train = process_dataset(train_pairs, "train", args.train_aug)
        task_val = process_dataset(val_pairs, "val", args.val_aug)

        all_train_rows.extend(task_train)
        all_val_rows.extend(task_val)
        print(f"  > Added {len(task_train)} train rows, {len(task_val)} val rows.")

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