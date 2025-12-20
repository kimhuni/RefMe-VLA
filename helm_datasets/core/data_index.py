from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from helm_datasets.utils.io_utils import (
    list_chunks_from_frames,
    list_episodes_from_frames,
    frames_dir,
    episode_json_path,
    read_json,
)


@dataclass(frozen=True)
class DataEpisode:
    out_root: Path
    chunk: str
    episode: str

    episode_dir: Path
    episode_json: Path

    fps_frames: int
    n_frames: int
    event_frame_idx: int

    table_dir: Path
    wrist_dir: Path

    def default_image_path(self, use_event_frame: bool = False, camera: str = "table") -> Path:
        """Return representative image path for this episode."""
        cam_dir = self.table_dir if camera == "table" else self.wrist_dir
        idx = self.event_frame_idx if use_event_frame else 0
        return cam_dir / f"frame_{idx:06d}.jpg"


def scan_data_episodes(out_root: Path, require_event: bool = True) -> List[DataEpisode]:
    """Scan frames_1hz/<chunk>/<episode>/<episode>.json and create DataEpisode list."""
    episodes: List[DataEpisode] = []
    chunks = list_chunks_from_frames(out_root)

    for chunk in chunks:
        eps = list_episodes_from_frames(out_root, chunk)
        for ep in eps:
            ep_dir = out_root / "frames_1hz" / chunk / ep
            ep_json = episode_json_path(out_root, chunk, ep)
            ev = read_json(ep_json, default=None)

            if ev is None:
                if require_event:
                    continue
                # fallback: no event file
                fps_frames = 1
                n_frames = _count_frames(frames_dir(out_root, chunk, ep, "table"))
                event_frame_idx = 0
            else:
                fps_frames = int(ev.get("fps_frames", 1))
                n_frames = ev.get("n_frames", None)
                if n_frames is None:
                    n_frames = _count_frames(frames_dir(out_root, chunk, ep, "table"))
                else:
                    n_frames = int(n_frames)

                event_frame_idx = ev.get("event_frame_idx", None)
                if event_frame_idx is None:
                    if require_event:
                        continue
                    event_frame_idx = 0
                event_frame_idx = int(event_frame_idx)

            table = frames_dir(out_root, chunk, ep, "table")
            wrist = frames_dir(out_root, chunk, ep, "wrist")

            episodes.append(
                DataEpisode(
                    out_root=out_root,
                    chunk=chunk,
                    episode=ep,
                    episode_dir=ep_dir,
                    episode_json=ep_json,
                    fps_frames=fps_frames,
                    n_frames=n_frames,
                    event_frame_idx=event_frame_idx,
                    table_dir=table,
                    wrist_dir=wrist,
                )
            )

    return episodes


def _count_frames(cam_dir: Path) -> int:
    if not cam_dir.exists():
        return 0
    # frame_000000.jpg 형태 가정
    return len(list(cam_dir.glob("frame_*.jpg")))