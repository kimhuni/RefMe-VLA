# tools/split_lerobot_by_segments.py
import os
import json
import math
import glob
import shutil
import subprocess
from pathlib import Path

import pandas as pd

"""
python common/datasets/libero_dataset/split_lerobot_by_segments.py \
  --src_root /data/ghkim/wipe_the_window_ep150 \
  --segments /data/ghkim/wipe_the_window_ep150/segments.jsonl \
  --out_root /data/ghkim/wipe_the_window_ep150_subtasks \
  --fps 5.0 \
  --video_keys observation.images.table observation.images.wrist
"""

def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def read_jsonl(path: str):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def episode_parquet_path(dataset_root: str, episode_index: int, chunk_size: int = 1000) -> str:
    chunk = episode_index // chunk_size
    return os.path.join(dataset_root, f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet")


def episode_video_path(dataset_root: str, episode_index: int, video_key: str, chunk_size: int = 1000) -> str:
    chunk = episode_index // chunk_size
    return os.path.join(dataset_root, f"videos/chunk-{chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")


def write_episode_parquet(df: pd.DataFrame, out_root: str, new_episode_index: int, chunk_size: int = 1000):
    chunk = new_episode_index // chunk_size
    out_dir = os.path.join(out_root, f"data/chunk-{chunk:03d}")
    safe_mkdir(out_dir)
    out_path = os.path.join(out_dir, f"episode_{new_episode_index:06d}.parquet")
    df.to_parquet(out_path, index=False)
    return out_path


def ffmpeg_cut_video(in_path: str, out_path: str, start_sec: float, dur_sec: float, reencode: bool = True):
    safe_mkdir(str(Path(out_path).parent))
    # Accurate seek: -ss after -i (slower but precise)
    if reencode:
        cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "error",
            "-i", in_path,
            "-ss", f"{start_sec:.6f}",
            "-t", f"{dur_sec:.6f}",
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-y",
            out_path,
        ]
    else:
        # stream copy may fail on non-keyframe boundaries
        cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "error",
            "-i", in_path,
            "-ss", f"{start_sec:.6f}",
            "-t", f"{dur_sec:.6f}",
            "-an",
            "-c", "copy",
            "-y",
            out_path,
        ]
    subprocess.check_call(cmd)


def normalize_segment_df(seg_df: pd.DataFrame, new_episode_index: int):
    """
    Make the sliced segment look like a fresh episode:
    - frame_index renumbered from 0
    - episode_index set to new_episode_index
    - (optional) keep timestamp as-is or shift to start at 0 (here we shift)
    """
    seg_df = seg_df.copy()

    if "frame_index" in seg_df.columns:
        seg_df["frame_index"] = range(len(seg_df))

    if "episode_index" in seg_df.columns:
        seg_df["episode_index"] = int(new_episode_index)

    if "timestamp" in seg_df.columns:
        t0 = float(seg_df["timestamp"].iloc[0])
        seg_df["timestamp"] = seg_df["timestamp"].astype("float32") - t0

    # If there's a dataset-level 'index' column, you can optionally reindex it
    if "index" in seg_df.columns:
        seg_df["index"] = range(len(seg_df))

    return seg_df


def main(
    src_root: str,
    segments_jsonl: str,
    out_root: str,
    fps: float = 5.0,
    video_keys=("observation.images.table", "observation.images.wrist"),
    chunk_size: int = 1000,
    drop_last_frame: bool = True,
    reencode_video: bool = True,
):
    safe_mkdir(out_root)
    segs = read_jsonl(segments_jsonl)
    if not segs:
        raise RuntimeError(f"No segments found in {segments_jsonl}")

    # Assign new episode indices sequentially
    new_ep = 0
    manifest = []  # keep mapping for later (debug / TaskSpec linking)

    for s in segs:
        parent_ep = int(s["episode_id"])
        seg_id = int(s.get("segment_id", 0))
        start = int(s["start"])
        end = int(s["end"])
        subtask = s.get("subtask", "")
        event_frame_idxs = s.get("event_frame_idxs", [])

        if end <= start:
            print(f"[SKIP] invalid range: ep={parent_ep} seg={seg_id} {start}-{end}")
            continue

        parquet_path = episode_parquet_path(src_root, parent_ep, chunk_size=chunk_size)
        if not os.path.exists(parquet_path):
            print(f"[SKIP] missing parquet: {parquet_path}")
            continue

        df = pd.read_parquet(parquet_path)

        # Slice by frame_index if exists; else by row indices
        if "frame_index" in df.columns:
            mask = (df["frame_index"] >= start) & (df["frame_index"] < end)
            seg_df = df.loc[mask].reset_index(drop=True)
        else:
            seg_df = df.iloc[start:end].reset_index(drop=True)

        # Handle action alignment: if action is "move to next state", drop last frame
        if drop_last_frame and len(seg_df) >= 2:
            seg_df = seg_df.iloc[:-1].reset_index(drop=True)

        if len(seg_df) < 2:
            print(f"[SKIP] too short after slicing: ep={parent_ep} seg={seg_id} len={len(seg_df)}")
            continue

        seg_df = normalize_segment_df(seg_df, new_episode_index=new_ep)
        out_parquet = write_episode_parquet(seg_df, out_root, new_episode_index=new_ep, chunk_size=chunk_size)

        # Cut videos
        # Convert frame to seconds
        # start_sec uses the same start; if drop_last_frame, duration matches seg_df length
        start_sec = start / float(fps)
        dur_sec = len(seg_df) / float(fps)

        for vk in video_keys:
            in_vid = episode_video_path(src_root, parent_ep, vk, chunk_size=chunk_size)
            if not os.path.exists(in_vid):
                print(f"[WARN] missing video: {in_vid}")
                continue

            out_chunk = new_ep // chunk_size
            out_vid_dir = os.path.join(out_root, f"videos/chunk-{out_chunk:03d}/{vk}")
            safe_mkdir(out_vid_dir)
            out_vid = os.path.join(out_vid_dir, f"episode_{new_ep:06d}.mp4")

            ffmpeg_cut_video(in_vid, out_vid, start_sec=start_sec, dur_sec=dur_sec, reencode=reencode_video)

        manifest.append({
            "new_episode_index": new_ep,
            "parent_episode_id": parent_ep,
            "segment_id": seg_id,
            "start": start,
            "end": end,
            "subtask": subtask,
            "event_frame_idxs": event_frame_idxs,
            "out_parquet": os.path.relpath(out_parquet, out_root),
        })
        print(f"[OK] new_ep={new_ep:06d} from parent_ep={parent_ep:06d} seg={seg_id} frames={len(seg_df)} subtask={subtask}")

        new_ep += 1

    # Save manifest for debugging / TaskSpec episode_filters 연결용
    manifest_path = os.path.join(out_root, "split_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", type=str, required=True)
    ap.add_argument("--segments", type=str, required=True, help="segments.jsonl")
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--chunk_size", type=int, default=1000)
    ap.add_argument("--drop_last_frame", action="store_true", default=True)
    ap.add_argument("--no_drop_last_frame", action="store_false", dest="drop_last_frame")
    ap.add_argument("--reencode_video", action="store_true", default=True)
    ap.add_argument("--no_reencode_video", action="store_false", dest="reencode_video")
    ap.add_argument("--video_keys", type=str, nargs="+",
                    default=["observation.images.table", "observation.images.wrist"])
    args = ap.parse_args()

    main(
        src_root=args.src_root,
        segments_jsonl=args.segments,
        out_root=args.out_root,
        fps=args.fps,
        video_keys=tuple(args.video_keys),
        chunk_size=args.chunk_size,
        drop_last_frame=args.drop_last_frame,
        reencode_video=args.reencode_video,
    )