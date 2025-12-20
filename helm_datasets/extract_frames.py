from __future__ import annotations
import argparse
from pathlib import Path
import cv2  # type: ignore
import json
from typing import Any, Dict, Optional
from helm_datasets.utils.io_utils import ensure_dir, frames_dir, list_chunks_from_videos, episode_json_path, read_json
"""
python -m helm_datasets.extract_frames \
  --lerobot_root "/data/ghkim/piper_press_the_blue_button_ep3" \
  --out_root     "/data/ghkim/helm_data/press_the_blue_button_one_time_test_ep3"
"""

def load_lerobot_episodes_meta(lerobot_root: Path) -> Dict[int, Dict[str, Any]]:
    """Load LeRobot meta/episodes.jsonl into a dict keyed by episode_index."""
    meta_path = lerobot_root / "meta" / "episodes.jsonl"
    if not meta_path.exists():
        print(f"[WARN] meta/episodes.jsonl not found: {meta_path}")
        return {}

    out: Dict[int, Dict[str, Any]] = {}
    with meta_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                print(f"[WARN] Failed to parse JSONL at {meta_path}:{line_no}")
                continue
            if "episode_index" not in obj:
                continue
            try:
                idx = int(obj["episode_index"])
            except Exception:
                continue
            out[idx] = obj
    return out


def parse_episode_index(ep_id: str) -> Optional[int]:
    """Parse episode_index from episode id like 'episode_000123'."""
    try:
        return int(ep_id.split("_")[-1])
    except Exception:
        return None


def write_episode_meta(
    out_root: Path,
    chunk: str,
    ep_id: str,
    fps_frames: int,
    n_frames: int,
    tasks: Optional[str],
    episode_index: Optional[int],
) -> None:
    """Write or merge per-episode JSON stored alongside frames."""
    p = episode_json_path(out_root, chunk, ep_id)
    ensure_dir(p.parent)

    # Merge with existing content if present (e.g., annotate_app later adds event_frame_idx)
    existing = read_json(p, default={})
    if not isinstance(existing, dict):
        existing = {}

    merged: Dict[str, Any] = dict(existing)
    merged.setdefault("fps_frames", int(fps_frames))
    merged["n_frames"] = int(n_frames)
    if tasks is not None:
        merged["tasks"] = tasks
    if episode_index is not None:
        merged["episode_index"] = int(episode_index)

    # Do not set a bogus value if annotation hasn't been done yet.
    merged.setdefault("event_frame_idx", None)

    p.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def extract_1hz_frames(mp4_path: Path, out_dir: Path, overwrite: bool) -> int:
    """
    mp4에서 1초당 1프레임을 추출하여 저장.
    - 저장 파일명: frame_{sec:06d}.jpg (sec=0,1,2,...)
    """
    ensure_dir(out_dir)

    if not mp4_path.exists():
        raise FileNotFoundError(f"Video not found: {mp4_path}")

    # 이미 존재하면 스킵
    if not overwrite and any(out_dir.glob("frame_*.jpg")):
        return len(list(out_dir.glob("frame_*.jpg")))

    cap = cv2.VideoCapture(str(mp4_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {mp4_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 5.0  # 너 데이터는 5Hz이므로 fallback

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = int(frame_count / fps) if frame_count > 0 else 0

    saved = 0
    for sec in range(duration_sec + 1):
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000.0)
        ok, frame = cap.read()
        if not ok:
            continue
        out_path = out_dir / f"frame_{sec:06d}.jpg"
        cv2.imwrite(str(out_path), frame)
        saved += 1

    cap.release()
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lerobot_root", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--overwrite", type=int, default=0)
    args = ap.parse_args()

    lerobot_root = Path(args.lerobot_root)
    out_root = Path(args.out_root)
    overwrite = bool(args.overwrite)

    meta_by_index = load_lerobot_episodes_meta(lerobot_root)

    videos_root = lerobot_root / "videos"
    if not videos_root.exists():
        raise FileNotFoundError(f"videos/ not found under: {lerobot_root}")

    chunks = list_chunks_from_videos(lerobot_root)
    if not chunks:
        raise RuntimeError(f"No chunk-* dirs found under: {videos_root}")

    total_eps = 0
    for chunk in chunks:
        table_dir = videos_root / chunk / "observation.images.table"
        wrist_dir = videos_root / chunk / "observation.images.wrist"
        if not table_dir.exists() or not wrist_dir.exists():
            print(f"[SKIP] Missing video dirs in {chunk}: {table_dir} / {wrist_dir}")
            continue

        episodes = sorted([p.stem for p in table_dir.glob("episode_*.mp4")])
        if not episodes:
            print(f"[SKIP] No episodes in {table_dir}")
            continue

        for ep_id in episodes:
            table_mp4 = table_dir / f"{ep_id}.mp4"
            wrist_mp4 = wrist_dir / f"{ep_id}.mp4"

            out_table = frames_dir(out_root, chunk, ep_id, "table")
            out_wrist = frames_dir(out_root, chunk, ep_id, "wrist")

            n1 = extract_1hz_frames(table_mp4, out_table, overwrite=overwrite)
            n2 = extract_1hz_frames(wrist_mp4, out_wrist, overwrite=overwrite)
            n_frames = min(n1, n2) if (n1 > 0 and n2 > 0) else max(n1, n2)

            ep_index = parse_episode_index(ep_id)
            tasks = None
            if ep_index is not None and ep_index in meta_by_index:
                # LeRobot meta commonly uses the key 'tasks' and 'length'
                tasks = meta_by_index[ep_index].get("tasks", None)
                if tasks is not None:
                    tasks = str(tasks)

            write_episode_meta(
                out_root=out_root,
                chunk=chunk,
                ep_id=ep_id,
                fps_frames=1,
                n_frames=n_frames,
                tasks=tasks,
                episode_index=ep_index,
            )

            print(f"[{chunk}/{ep_id}] saved table={n1}, wrist={n2} | n_frames={n_frames} | tasks={tasks}")
            total_eps += 1

    print(f"Done. processed episodes: {total_eps}")


if __name__ == "__main__":
    main()