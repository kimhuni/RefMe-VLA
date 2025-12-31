from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .io_utils import episode_json_path, load_episode_json, list_common_frame_ids


@dataclass
class DataEpisodeV3:
    chunk: str
    episode: str
    meta: Dict[str, Any]            # episode json 전체(보존)
    frame_ids: List[int]            # table/wrist 공통 frame_id
    event_frame_ids: List[int]      # episode.json의 event_frame_idxs (frame_id list)


def load_data_episode(
    out_root: Path,
    fps_out: int,
    chunk: str,
    episode: str,
    use_wrist: bool,
) -> DataEpisodeV3:
    ep_json = load_episode_json(episode_json_path(out_root, fps_out, chunk, episode))

    # v3: event_frame_idxs(list)를 1순위로 사용
    event_ids: List[int] = []
    if "event_frame_idxs" in ep_json and isinstance(ep_json["event_frame_idxs"], list):
        event_ids = [int(x) for x in ep_json["event_frame_idxs"]]
    else:
        # backward compatible: event_frame_idx
        if ep_json.get("event_frame_idx") is not None:
            event_ids = [int(ep_json["event_frame_idx"])]

    event_ids = sorted(set(event_ids))

    table_dir = out_root / f"frames_{fps_out}hz" / chunk / episode / "table"
    wrist_dir = (out_root / f"frames_{fps_out}hz" / chunk / episode / "wrist") if use_wrist else None
    frame_ids = list_common_frame_ids(table_dir, wrist_dir)

    return DataEpisodeV3(
        chunk=chunk,
        episode=episode,
        meta=ep_json,
        frame_ids=frame_ids,
        event_frame_ids=event_ids,
    )


def iter_all_episodes(out_root: Path, fps_out: int) -> List[tuple[str, str]]:
    """
    frames_{fps}hz 아래 chunk-xxx/episode_xxx 구조를 스캔.
    """
    root = out_root / f"frames_{fps_out}hz"
    if not root.exists():
        raise FileNotFoundError(f"frames root not found: {root}")

    out: List[tuple[str, str]] = []
    for chunk_dir in sorted(root.glob("chunk-*")):
        if not chunk_dir.is_dir():
            continue
        for ep_dir in sorted(chunk_dir.glob("episode_*")):
            if ep_dir.is_dir():
                out.append((chunk_dir.name, ep_dir.name))
    return out