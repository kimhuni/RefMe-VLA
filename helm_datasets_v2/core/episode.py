# helm_datasets_v2/core/episode.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import json
import re


_FRAME_RE = re.compile(r"frame_(\d+)\.(jpg|jpeg|png)$", re.IGNORECASE)


@dataclass
class DataEpisode:
    out_root: Path
    chunk: str
    episode: str
    episode_dir: Path
    camera: str
    frame_paths: List[Path]
    episode_json: Dict[str, Any]
    event_frames: List[int]
    task_str: str


def discover_episodes(out_root: Path, camera: str) -> List[Path]:
    base = out_root / "frames_1hz"
    if not base.exists():
        raise FileNotFoundError(f"frames_1hz not found under: {out_root}")
    episode_dirs = sorted(base.glob("chunk-*/episode_*"))
    episode_dirs = [p for p in episode_dirs if (p / camera).exists()]
    return episode_dirs


def _load_episode_json(episode_dir: Path, episode_name: str) -> Dict[str, Any]:
    p = episode_dir / f"{episode_name}.json"
    if not p.exists():
        cand = list(episode_dir.glob("*.json"))
        if len(cand) == 1:
            p = cand[0]
        else:
            raise FileNotFoundError(f"episode json not found: {episode_dir}/{episode_name}.json")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_frames(episode_dir: Path, camera: str) -> List[Path]:
    cam_dir = episode_dir / camera
    frames = sorted([p for p in cam_dir.iterdir() if p.is_file() and _FRAME_RE.match(p.name)])
    def key(p: Path) -> int:
        m = _FRAME_RE.match(p.name)
        return int(m.group(1)) if m else 0
    return sorted(frames, key=key)


def extract_event_frames(ep_json: Dict[str, Any], key: str = "event_frames") -> List[int]:
    # Supported:
    # 1) {"event_frames": [1, 5, ...]}
    # 2) {"events": [{"frame": 12, "type": "..."}]}
    # 3) {"event_frame_idx": 9}  <-- your schema
    if key in ep_json and isinstance(ep_json[key], list):
        return [int(x) for x in ep_json[key]]

    if "event_frames" in ep_json and isinstance(ep_json["event_frames"], list):
        return [int(x) for x in ep_json["event_frames"]]

    if "events" in ep_json and isinstance(ep_json["events"], list):
        out = []
        for e in ep_json["events"]:
            if isinstance(e, dict) and "frame" in e:
                out.append(int(e["frame"]))
        return out

    if "event_frame_idx" in ep_json:
        try:
            return [int(ep_json["event_frame_idx"])]
        except Exception:
            return []

    return []


def extract_task_str(ep_json: Dict[str, Any]) -> str:
    # your schema: tasks is a single string
    t = ep_json.get("tasks", "")
    return str(t).strip()


def load_data_episode(out_root: Path, episode_dir: Path, camera: str, event_frames_key: str) -> DataEpisode:
    chunk = episode_dir.parent.name
    episode = episode_dir.name

    frames = _load_frames(episode_dir, camera)
    ep_json = _load_episode_json(episode_dir, episode)
    event_frames = extract_event_frames(ep_json, key=event_frames_key)
    task_str = extract_task_str(ep_json)

    return DataEpisode(
        out_root=out_root,
        chunk=chunk,
        episode=episode,
        episode_dir=episode_dir,
        camera=camera,
        frame_paths=frames,
        episode_json=ep_json,
        event_frames=sorted(event_frames),
        task_str=task_str,
    )