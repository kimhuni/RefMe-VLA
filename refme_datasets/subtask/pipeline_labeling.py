# datasets/subtask/pipeline_labeling.py
"""
python -m refme_datasets.subtask.pipeline_labeling.py \
  --dataset-root /data/ghkim/press the blue button two times/lerobot_5hz \
  --state-col observation.state

python -m refme_datasets.subtask.pipeline_labeling   --dataset-root /data/ghkim/press the blue button two times/lerobot_5hz   --state-col observation.state
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from refme_datasets.subtask.config import VelocityLabelConfig, TASK_INDEX_TO_NUM_SUBTASKS
from refme_datasets.subtask.utils import compute_velocity_from_df, find_low_speed_segments, assign_subtask_indices
from refme_datasets.subtask.lerobot_io import (
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

    # boundary-based segmentation: use b-run start indices
    required_boundaries = max(num_subtasks - 1, 0)
    boundaries = sorted([seg.frame_start for seg in segments])[:required_boundaries]

    subtasks_meta = []
    prev = 0
    # subtask 0 .. num_subtasks-2
    for k, b in enumerate(boundaries):
        end = max(b - 1, prev)
        subtasks_meta.append({
            "subtask_index": k,
            "start": int(prev),
            "end": int(end),
            "length": int(end - prev + 1),
        })
        prev = b

    # final subtask: include final rest (A-option)
    last_k = num_subtasks - 1
    subtasks_meta.append({
        "subtask_index": last_k,
        "start": int(prev),
        "end": int(num_frames - 1),
        "length": int(num_frames - prev),
    })

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
        required=False,
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