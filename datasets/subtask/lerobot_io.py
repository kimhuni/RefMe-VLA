# datasets/subtask/lerobot_io.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Iterable

from common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from common.datasets import utils as lerobot_utils


# 새로 만들 subtask 메타 파일 경로
EPISODES_SUBTASK_PATH = "meta/episodes_subtask.jsonl"


def load_metadata(dataset_root: str | Path, repo_id: str) -> LeRobotDatasetMetadata:
    """
    기존 LeRobotDatasetMetadata 를 재사용해서
    info / episodes / tasks 등을 한 번에 불러오는 헬퍼.
    """
    root = Path(dataset_root)
    meta = LeRobotDatasetMetadata(repo_id=repo_id, root=root)
    return meta


def iter_episodes(meta: LeRobotDatasetMetadata) -> Iterable[Dict[str, Any]]:
    """
    episodes.jsonl 에 들어있는 episode dict 를 정렬된 순서로 yield.
    load_episodes() 가 {episode_index: episode_dict} 를 반환하므로,
    그걸 그대로 meta.episodes 에 담고 있다고 가정한다.
    """
    episodes_dict: Dict[int, Dict[str, Any]] = meta.episodes
    for ep_idx in sorted(episodes_dict.keys()):
        yield episodes_dict[ep_idx]


def get_episode_parquet_path(meta: LeRobotDatasetMetadata, episode_index: int) -> Path:
    """
    episode_index 에 해당하는 parquet 파일의 상대 경로를 metadata 로부터 얻어서,
    dataset root 와 합쳐 최종 경로를 반환.
    """
    rel_path = meta.get_data_file_path(ep_index=episode_index)  # Path
    return meta.root / rel_path


def get_task_index_for_episode(
    meta: LeRobotDatasetMetadata,
    episode: Dict[str, Any],
) -> int:
    """
    episode dict 에서 task_index 를 찾아서 반환한다.

    - episodes.jsonl 에 이미 "task_index" 필드가 있으면 그대로 사용.
    - 없고 "tasks" (task 문자열) 만 있다면,
        meta.task_to_task_index 를 이용해 매핑한다.
    - 둘 다 없으면 에러를 던진다.
    """
    if "task_index" in episode:
        return int(episode["task_index"])

    # Hugging Face lerobot spec 에서는 보통 "task_index" 를 쓰지만,
    # 여기서는 사용자가 예시로 "tasks": "press the blue button" 형식도 보여줬으므로
    # 그 경우를 대비해서 task 문자열 기반 매핑도 지원한다.
    task_str_keys = ["task", "tasks"]
    task_name = None
    for k in task_str_keys:
        if k in episode:
            task_name = episode[k]
            break

    if task_name is None:
        raise KeyError(
            f"Episode has neither 'task_index' nor 'task'/'tasks'. "
            f"Episode keys: {list(episode.keys())}"
        )

    if task_name not in meta.task_to_task_index:
        raise KeyError(
            f"Task name '{task_name}' not found in meta.task_to_task_index "
            f"(available: {list(meta.task_to_task_index.keys())[:5]} ...)."
        )

    return int(meta.task_to_task_index[task_name])


def append_episode_subtask_record(
    dataset_root: str | Path,
    record: Dict[str, Any],
) -> None:
    """
    episodes_subtask.jsonl 에 한 줄을 append 한다.
    lerobot 의 append_jsonlines 유틸을 재사용.
    """
    root = Path(dataset_root)
    fpath = root / EPISODES_SUBTASK_PATH
    lerobot_utils.append_jsonlines(record, fpath)