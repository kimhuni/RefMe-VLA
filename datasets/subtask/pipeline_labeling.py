
# datasets/subtask/pipeline_labeling.py
"""
python -m datasets.subtask.pipeline_labeling \
  --dataset-root /path/to/piper_mix_v01_ep5 \
  --repo-id piper_mix_v01_ep5 \
  --state-col observation.state
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import VelocityLabelConfig, TASK_INDEX_TO_NUM_SUBTASKS
from .utils import compute_velocity_from_df, find_low_speed_segments, assign_subtask_indices
from .lerobot_io import (
    load_metadata,
    iter_episodes,
    get_episode_parquet_path,
    get_task_index_for_episode,
    append_episode_subtask_record,
)


def process_single_episode(
    dataset_root: Path,
    repo_id: str,
    episode: dict,
    meta,
    state_col: str,
    cfg: VelocityLabelConfig,
    dry_run: bool = False,
) -> None:
    """
    단일 episode 에 대해:
      1) parquet 로딩
      2) 속도 계산
      3) low-speed segment 탐색
      4) subtask_index 생성
      5) parquet 에 subtask_index 저장
      6) episodes_subtask.jsonl 메타 기록
    를 수행하는 함수.
    """
    ep_idx = int(episode["episode_index"])
    task_index = get_task_index_for_episode(meta, episode)

    if task_index not in TASK_INDEX_TO_NUM_SUBTASKS:
        # 설정되지 않은 task_index 는 스킵 (혹은 에러로 처리해도 됨)
        print(f"[WARN] episode {ep_idx}: task_index {task_index} not in TASK_INDEX_TO_NUM_SUBTASKS, skip.")
        return

    num_subtasks = TASK_INDEX_TO_NUM_SUBTASKS[task_index]
    ep_path = get_episode_parquet_path(meta, ep_idx)

    if not ep_path.is_file():
        print(f"[WARN] episode {ep_idx}: parquet not found at {ep_path}, skip.")
        return

    df = pd.read_parquet(ep_path)
    num_frames = len(df)

    # 1) 속도 계산
    t, v = compute_velocity_from_df(df, state_col=state_col, cfg=cfg)

    # 2) low-speed segment 탐색
    segments = find_low_speed_segments(t, v, cfg=cfg)

    # 3) subtask_index 생성
    try:
        subtask_index = assign_subtask_indices(
            num_frames=num_frames,
            low_speed_segments=segments,
            num_subtasks=num_subtasks,
        )
    except RuntimeError as e:
        print(f"[WARN] episode {ep_idx}: {e} → 라벨링 실패, 스킵.")
        return

    # 4) parquet 에 subtask_index 저장
    if not dry_run:
        df["subtask_index"] = subtask_index.astype(np.int32)
        df.to_parquet(ep_path)
        print(f"[INFO] episode {ep_idx}: wrote subtask_index to {ep_path}")

    # 5) episodes_subtask.jsonl 메타 기록
    used_segments = segments[:num_subtasks]
    subtasks_meta = []
    prev_end = -1
    for k, seg in enumerate(used_segments):
        start = max(prev_end + 1, 0)
        end = min(seg.frame_end, num_frames - 1)
        subtasks_meta.append(
            {
                "subtask_index": k,
                "start": int(start),
                "end": int(end),
                "length": int(end - start + 1),
            }
        )
        prev_end = end

    # 마지막 segment 이후 남은 구간도 마지막 subtask 로 포함되어 있으니,
    # 메타에는 "라벨링 기준 segment" 정보만 기록해둔다.
    record = {
        "episode_index": ep_idx,
        "task_index": task_index,
        "num_frames": num_frames,
        "expected_num_subtasks": num_subtasks,
        "detected_num_low_speed_segments": len(segments),
        "used_num_subtasks": len(subtasks_meta),
        "subtasks": subtasks_meta,
    }

    if not dry_run:
        append_episode_subtask_record(dataset_root, record)
        print(f"[INFO] episode {ep_idx}: append subtask meta.")


def main():
    parser = argparse.ArgumentParser(
        description="Velocity-based subtask labeling pipeline for lerobot dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="lerobot dataset root 디렉토리 (meta/, data/, videos/ 가 있는 위치)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="LeRobotDatasetMetadata 에서 사용하는 repo_id (보통 HF dataset 이름 혹은 폴더 이름).",
    )
    parser.add_argument(
        "--state-col",
        type=str,
        default="observation.state",
        help="EE state 가 들어있는 컬럼명 (기본: observation.state).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제로 parquet / episodes_subtask.jsonl 를 수정하지 않고 로그만 찍기.",
    )
    # 나중에 tau_low, ignore_first_sec 같은 것도 CLI로 덮어쓸 수 있게 옵션 추가 가능
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    repo_id = args.repo_id

    # 기본 config 로 시작 (나중에 CLI 인자로 tau_low 등 덮어써도 됨)
    cfg = VelocityLabelConfig()

    print(f"[INFO] Loading metadata from root={dataset_root}, repo_id={repo_id}")
    meta = load_metadata(dataset_root, repo_id)

    for episode in iter_episodes(meta):
        process_single_episode(
            dataset_root=dataset_root,
            repo_id=repo_id,
            episode=episode,
            meta=meta,
            state_col=args.state_col,
            cfg=cfg,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()