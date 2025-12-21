from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass(frozen=True)
class DataEpisode:
    out_root: Path
    chunk: str
    episode: str

    tasks: str
    episode_index: int

    fps_frames: int
    n_frames: int
    event_frame_idx: int

    frame_paths: Dict[str, List[Path]]  # {"table":[...], "wrist":[...]}

    def get_frame_paths(self, cameras: List[str], t: int) -> Dict[str, str]:
        return {cam: str(self.frame_paths[cam][t]) for cam in cameras}


def _list_chunks(frames_root: Path) -> List[str]:
    if not frames_root.exists():
        return []
    return sorted([p.name for p in frames_root.iterdir() if p.is_dir() and p.name.startswith("chunk-")])


def _list_episodes(frames_root: Path, chunk: str) -> List[str]:
    base = frames_root / chunk
    if not base.exists():
        return []
    return sorted([p.name for p in base.iterdir() if p.is_dir() and p.name.startswith("episode_")])


def _episode_json_path(frames_root: Path, chunk: str, episode: str) -> Path:
    return frames_root / chunk / episode / f"{episode}.json"


def _camera_dir(frames_root: Path, chunk: str, episode: str, camera: str) -> Path:
    return frames_root / chunk / episode / camera


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def scan_data_episodes(
    out_root: Path,
    cameras: List[str],
    require_event: bool = True,
) -> List[DataEpisode]:
    frames_root = out_root / "frames_1hz"
    chunks = _list_chunks(frames_root)
    episodes: List[DataEpisode] = []

    for chunk in chunks:
        for ep in _list_episodes(frames_root, chunk):
            ep_json = _episode_json_path(frames_root, chunk, ep)
            meta = _read_json(ep_json, default=None)
            if meta is None:
                if require_event:
                    continue
                meta = {}

            tasks = str(meta.get("tasks", ""))
            episode_index = int(meta.get("episode_index", -1))
            fps_frames = int(meta.get("fps_frames", 1))

            frame_paths: Dict[str, List[Path]] = {}
            min_len: Optional[int] = None
            for cam in cameras:
                d = _camera_dir(frames_root, chunk, ep, cam)
                imgs = sorted(d.glob("frame_*.jpg"))
                frame_paths[cam] = imgs
                min_len = len(imgs) if min_len is None else min(min_len, len(imgs))

            if min_len is None or min_len == 0:
                continue

            n_frames = int(meta.get("n_frames", min_len))
            n_frames = max(1, min(n_frames, min_len))

            ev = meta.get("event_frame_idx", None)
            if ev is None:
                if require_event:
                    continue
                event_frame_idx = 0
            else:
                event_frame_idx = int(ev)
                event_frame_idx = max(0, min(event_frame_idx, n_frames - 1))

            episodes.append(
                DataEpisode(
                    out_root=out_root,
                    chunk=chunk,
                    episode=ep,
                    tasks=tasks,
                    episode_index=episode_index,
                    fps_frames=fps_frames,
                    n_frames=n_frames,
                    event_frame_idx=event_frame_idx,
                    frame_paths=frame_paths,
                )
            )

    return episodes
