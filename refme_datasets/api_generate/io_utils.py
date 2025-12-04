# io_utils.py
import json
import subprocess
from pathlib import Path
from typing import Dict, Iterator, List, Tuple
import logging

logger = logging.getLogger(__name__)

# ---------- Episodes meta ----------

def load_episodes_meta(meta_path: Path) -> Dict[int, Dict]:
    meta: Dict[int, Dict] = {}
    with open(meta_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            idx = int(obj["episode_index"])
            task = obj.get("task") or obj.get("tasks") or "unknown task"
            length = obj.get("length")
            meta[idx] = {"task": task, "length": length}
    return meta

# ---------- Chunk/Episode iteration ----------

def iter_chunks(videos_dir: Path) -> Iterator[Tuple[str, Path]]:
    for chunk_dir in sorted(videos_dir.glob("chunk-*/")):
        logger.info("[iter_chunks] found chunk: %s", chunk_dir)
        yield chunk_dir.name, chunk_dir


def list_episodes_in_chunk(chunk_dir: Path) -> List[int]:
    table_dir = chunk_dir / "observation.images.table"
    eps = []
    for mp4 in sorted(table_dir.glob("episode_*.mp4")):
        try:
            eps.append(int(mp4.stem.split("_")[-1]))
        except Exception:
            continue
    logger.info("[list_episodes_in_chunk] %s -> %d episodes", chunk_dir, len(eps))
    return eps


def episode_video_paths(chunk_dir: Path, episode_idx: int) -> Tuple[Path, Path]:
    table = chunk_dir / "observation.images.table" / f"episode_{episode_idx:06d}.mp4"
    wrist = chunk_dir / "observation.images.wrist" / f"episode_{episode_idx:06d}.mp4"
    return table, wrist

# ---------- Frame extraction (1 FPS, ffmpeg) ----------

def ensure_frames_for_episode(table_mp4: Path, wrist_mp4: Path, out_side: Path, out_wrist: Path, fps: int = 1) -> None:
    out_side.mkdir(parents=True, exist_ok=True)
    out_wrist.mkdir(parents=True, exist_ok=True)

    # 이미 프레임이 있으면 스킵 (간단 기준)
    if not any(out_side.glob("frame_*.jpg")):
        _extract_1fps_ffmpeg(table_mp4, out_side, fps=fps)
    if not any(out_wrist.glob("frame_*.jpg")):
        _extract_1fps_ffmpeg(wrist_mp4, out_wrist, fps=fps)


def _extract_1fps_ffmpeg(video_path: Path, out_dir: Path, fps: int = 1) -> None:
    # PTS(타임스탬프) 기반 파일명: frame_%012d.jpg (ffmpeg의 frame_pts 사용)
    # 일부 ffmpeg 빌드에서 -frame_pts 1 이 작동하지 않으면 -vsync vfr + showinfo 로그로 대체 파싱 필요
    logger.info("[ffmpeg] %s -> %s (fps=%d)", video_path, out_dir, fps)
    cmd = [
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-frame_pts", "1",
        str(out_dir / "frame_%012d.jpg"),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# ---------- Frame pairing (timestamp align by filename) ----------

def iter_aligned_pairs(side_dir: Path, wrist_dir: Path) -> Iterator[Tuple[Path, Path, int]]:
    # 공통 파일명 기준 정렬 (frame_##########.jpg)
    s_map = {p.name: p for p in sorted(side_dir.glob("frame_*.jpg"))}
    w_map = {p.name: p for p in sorted(wrist_dir.glob("frame_*.jpg"))}
    common_keys = set(s_map.keys()) & set(w_map.keys())
    logger.info("[iter_aligned_pairs] %s & %s -> %d aligned frames", side_dir, wrist_dir, len(common_keys))
    for name in sorted(common_keys):
        ts = int(name.split("_")[-1].split(".")[0])  # frame_000037000.jpg -> 000037000
        yield s_map[name], w_map[name], ts

# ---------- JSONL & logging ----------

def write_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def reset_file(path: Path) -> None:
    if path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)