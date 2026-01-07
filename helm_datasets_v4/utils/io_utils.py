from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import time


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, ensure_ascii=False, indent=2))


def read_json(path: Path, default: Optional[Any] = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


# ----------------------------
# Chunk/Episode discovery
# ----------------------------

def list_chunks_from_videos(lerobot_root: Path) -> List[str]:
    """
    lerobot_root/videos/chunk-* 스캔
    """
    videos_root = lerobot_root / "videos"
    if not videos_root.exists():
        return []
    chunks = sorted([p.name for p in videos_root.iterdir() if p.is_dir() and p.name.startswith("chunk-")])
    return chunks


def list_chunks_from_frames(out_root: Path, fps_out: int) -> List[str]:
    """
    out_root/frames_1hz/chunk-* 스캔
    """
    base = out_root / f"frames_{fps_out}hz"
    if not base.exists():
        return []
    chunks = sorted([p.name for p in base.iterdir() if p.is_dir() and p.name.startswith("chunk-")])
    return chunks


def list_episodes_from_frames(out_root: Path, fps_out: int, chunk: str) -> List[str]:
    """
    frames_1hz/<chunk>/episode_* 폴더 목록
    """
    base = out_root / f"frames_{fps_out}hz" / chunk
    if not base.exists():
        return []
    eps = sorted([p.name for p in base.iterdir() if p.is_dir() and p.name.startswith("episode_")])
    return eps


# ----------------------------
# Path conventions
# ----------------------------

def frames_dir(out_root: Path, fps_out: int, chunk: str, episode: str, camera: str) -> Path:
    return out_root / f"frames_{fps_out}hz" / chunk / episode / camera


def episode_json_path(out_root: Path, fps_out: int, chunk: str, episode: str) -> Path:
    """Per-episode metadata/event JSON stored alongside frames."""
    # frames_dir(out_root, chunk, episode, camera) -> .../frames_1hz/<chunk>/<episode>/<camera>
    d = frames_dir(out_root, fps_out, chunk, episode, "table").parent
    return d / f"{episode}.json"


def meta_path(out_root: Path) -> Path:
    return out_root / "meta.json"


def index_path(out_root: Path) -> Path:
    return out_root / "index.json"


def jsonl_dir(out_root: Path) -> Path:
    return out_root / "jsonl"


# ----------------------------
# Misc
# ----------------------------

def now_kst_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def uid(task_id: str, chunk: str, episode: str, frame_idx: int, variant_id: str, intra: int) -> str:
    return f"{task_id}@{chunk}-{episode}-v{variant_id}-i{intra}-f{frame_idx:06d}"


def update_index_event(out_root: Path, fps_out: int, chunk: str, episode: str, boundaries: List[int]) -> None:
    """Update index.json for quick discovery.

    Backward-compat: callers may still pass a list; we treat the first element
    as the single event frame idx.
    """
    ep_json = episode_json_path(out_root, fps_out, chunk, episode)
    ev = read_json(ep_json, default={})

    # Prefer on-disk value; fall back to the provided list for backward-compat.
    event_frame_idx = ev.get("event_frame_idx", None)
    if event_frame_idx is None and boundaries:
        event_frame_idx = int(boundaries[0])

    n_frames = ev.get("n_frames", None)

    idxp = index_path(out_root)
    idx = read_json(idxp, default={})
    idx.setdefault("events", {})
    idx["events"].setdefault(chunk, {})
    idx["events"][chunk][episode] = {
        "event_frame_idx": None if event_frame_idx is None else int(event_frame_idx),
        "n_frames": None if n_frames is None else int(n_frames),
        "updated_at": now_kst_iso(),
    }
    atomic_write_json(idxp, idx)