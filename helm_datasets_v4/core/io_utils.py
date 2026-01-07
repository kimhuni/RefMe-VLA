from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def frames_dir(out_root: Path, fps_out: int, chunk: str, episode: str, camera: str) -> Path:
    # 예: /.../frames_5hz/chunk-000/episode_000000/table/
    return out_root / f"frames_{fps_out}hz" / chunk / episode / camera


def episode_json_path(out_root: Path, fps_out: int, chunk: str, episode: str) -> Path:
    return out_root / f"frames_{fps_out}hz" / chunk / episode / f"{episode}.json"


def load_episode_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"episode json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def save_episode_json_preserve(path: Path, patch: Dict[str, Any]) -> None:
    """
    기존 키(tasks, episode_index 등)를 유지하면서 patch만 덮어쓴다.
    """
    base: Dict[str, Any] = {}
    if path.exists():
        base = json.loads(path.read_text(encoding="utf-8"))
    base.update(patch)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_frame_id(name: str) -> Optional[int]:
    # frame_000123.jpg -> 123
    if not name.startswith("frame_") or "." not in name:
        return None
    stem = name.split(".")[0]
    num = stem.replace("frame_", "")
    if not num.isdigit():
        return None
    return int(num)


def list_common_frame_ids(table_dir: Path, wrist_dir: Optional[Path]) -> List[int]:
    table_ids = set()
    if table_dir.exists():
        for p in table_dir.glob("frame_*.jpg"):
            fid = parse_frame_id(p.name)
            if fid is not None:
                table_ids.add(fid)

    if wrist_dir is None:
        return sorted(table_ids)

    wrist_ids = set()
    if wrist_dir.exists():
        for p in wrist_dir.glob("frame_*.jpg"):
            fid = parse_frame_id(p.name)
            if fid is not None:
                wrist_ids.add(fid)

    return sorted(table_ids & wrist_ids)


def frame_path(out_root: Path, fps_out: int, chunk: str, episode: str, camera: str, frame_id: int) -> Path:
    return frames_dir(out_root, fps_out, chunk, episode, camera) / f"frame_{frame_id:06d}.jpg"