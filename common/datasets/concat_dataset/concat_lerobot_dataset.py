"""
Concat multiple full LeRobot episodes into new episodes based on recipes.jsonl.

Input (src_root) layout (as you described):
  data/chunk-XXX/episode_YYYYYY.parquet
  videos/chunk-XXX/{video_key}/episode_YYYYYY.mp4

Chunking is controlled by `chunk_size` (default 50), which determines how many episodes per chunk directory.

Recipe format (JSONL, one per line):
{
  "new_episode_index": 0,
  "parts": [{"task":"a","episode_id":12}, {"task":"b","episode_id":7}, {"task":"c","episode_id":21}],
  "boundary_policy": {"noop_frames": 0, "drop_last_frame_each_part": true, "drop_first_frames_each_part": 0}
}

Output (out_root) layout mirrors LeRobot:
  data/chunk-XXX/episode_YYYYYY.parquet
  videos/chunk-XXX/{video_key}/episode_YYYYYY.mp4
  concat_manifest.json  (debug / provenance)

Notes / Policy (defaults are safe for delta-action datasets):
- drop_last_frame_each_part: True  -> drops last row of each part to reduce action alignment issues at boundaries
- drop_first_frames_each_part: int -> drops the first N rows of each part before concatenation (useful to avoid leading transient / reset frames)
- noop_frames: 0 (default) -> if >0, inserts K duplicated frames at boundaries with action=0
  (keeps observation/state as last frame, action zeros; timestamp continues)
- timestamps are regenerated as uniform steps using fps (0, 1/fps, 2/fps, ...)
- drop_first_frames_each_part and drop_last_frame_each_part affect both parquet and video trimming (video clips are trimmed to match the exact frames kept in the parquet).

Requirements:
- pandas, pyarrow
- ffmpeg available in PATH
"""
def ffmpeg_trim_video(
    in_path: str,
    out_path: str,
    start_sec: float,
    dur_sec: float,
    reencode: bool = True,
    fps: float = 5.0,
):
    """Trim a video to [start_sec, start_sec+dur_sec). Accurate seek; re-encode by default."""
    safe_mkdir(str(Path(out_path).parent))
    if reencode:
        cmd = [
            "ffmpeg",
            "-hide_banner", "-loglevel", "error",
            "-i", in_path,
            "-ss", f"{start_sec:.6f}",
            "-t", f"{dur_sec:.6f}",
            "-an",
            "-r", f"{fps}",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-y",
            out_path,
        ]
    else:
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

import os
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd

"""
python common/datasets/concat_dataset/concat_lerobot_dataset.py \
  --src_root /data/ghkim/lerobot_data/wipe_the_window_ep150 \
  --recipes /data/ghkim/concat_data/wipe_the_window/wipe_BMT.jsonl \
  --out_root /data/ghkim/concat_data/wipe_the_window \
  --fps 5.0 \
  --chunk_size 50 \
  --video_keys observation.images.table observation.images.wrist \
  --drop_first_frames_each_part 5
  
python common/datasets/concat_dataset/concat_lerobot_dataset.py \
  --src_root /data/ghkim/data_hub/press_the_button_nolight_full \
  --recipes /data/ghkim/concat_data/press_button_N_time/press_button_2.jsonl \
  --out_root /data/ghkim/concat_data/press_button_N_time/press_button_2 \
  --fps 5.0 \
  --chunk_size 50 \
  --video_keys observation.images.table observation.images.wrist \
  --drop_first_frames_each_part 4
  
python common/datasets/concat_dataset/concat_lerobot_dataset.py \
  --src_root /data/ghkim/data_hub/press_the_button_nolight_full \
  --recipes /data/ghkim/concat_data/press_button_N_time/press_button_3.jsonl \
  --out_root /data/ghkim/concat_data/press_button_N_time/press_button_3 \
  --fps 5.0 \
  --chunk_size 50 \
  --video_keys observation.images.table observation.images.wrist \
  --drop_first_frames_each_part 4

python common/datasets/concat_dataset/concat_lerobot_dataset.py \
  --src_root /data/ghkim/data_hub/press_the_button_nolight_full \
  --recipes /data/ghkim/concat_data/press_button_N_time/press_button_3.jsonl \
  --out_root /data/ghkim/concat_data/press_button_N_time/press_button_3 \
  --fps 5.0 \
  --chunk_size 50 \
  --video_keys observation.images.table observation.images.wrist \
  --drop_first_frames_each_part 5

python common/datasets/concat_dataset/concat_lerobot_dataset.py \
  --src_root /data/ghkim/data_hub/wipe_the_window \
  --recipes /data/ghkim/concat_data/wipe_the_window/wipe_BMT.jsonl \
  --out_root /data/ghkim/concat_data/wipe_the_window \
  --fps 5.0 \
  --chunk_size 50 \
  --video_keys observation.images.table observation.images.wrist \
  --drop_first_frames_each_part 5
"""

def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)


def read_jsonl(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def episode_parquet_path(dataset_root: str, episode_index: int, chunk_size: int = 50) -> str:
    chunk = episode_index // chunk_size
    return os.path.join(dataset_root, f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet")


def episode_video_path(dataset_root: str, episode_index: int, video_key: str, chunk_size: int = 50) -> str:
    chunk = episode_index // chunk_size
    return os.path.join(dataset_root, f"videos/chunk-{chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")


def out_parquet_path(out_root: str, new_episode_index: int, chunk_size: int = 50) -> str:
    out_chunk = new_episode_index // chunk_size
    out_dir = os.path.join(out_root, f"data/chunk-{out_chunk:03d}")
    safe_mkdir(out_dir)
    return os.path.join(out_dir, f"episode_{new_episode_index:06d}.parquet")


def out_video_path(out_root: str, new_episode_index: int, video_key: str, chunk_size: int = 50) -> str:
    out_chunk = new_episode_index // chunk_size
    out_dir = os.path.join(out_root, f"videos/chunk-{out_chunk:03d}/{video_key}")
    safe_mkdir(out_dir)
    return os.path.join(out_dir, f"episode_{new_episode_index:06d}.mp4")


def ffmpeg_concat_videos(
    input_paths: List[str],
    output_path: str,
    reencode: bool = True,
    fps: float = 5.0,
):
    """
    Concatenate MP4 files in order.
    - Uses concat demuxer with a temporary list file.
    - If reencode=True (recommended), outputs H.264/yuv420p to avoid codec mismatch.
    """
    safe_mkdir(str(Path(output_path).parent))

    # concat demuxer requires a file list with: file 'path'
    # Paths must be escaped; easiest is to use absolute and single quotes.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tf:
        list_path = tf.name
        for p in input_paths:
            ap = os.path.abspath(p)
            tf.write(f"file '{ap}'\n")

    try:
        if reencode:
            # Force consistent output
            cmd = [
                "ffmpeg",
                "-hide_banner", "-loglevel", "error",
                "-f", "concat",
                "-safe", "0",
                "-i", list_path,
                "-an",
                "-r", f"{fps}",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-y",
                output_path,
            ]
        else:
            # Stream copy is fast but can fail if codecs/settings differ or non-keyframe boundaries
            cmd = [
                "ffmpeg",
                "-hide_banner", "-loglevel", "error",
                "-f", "concat",
                "-safe", "0",
                "-i", list_path,
                "-an",
                "-c", "copy",
                "-y",
                output_path,
            ]
        subprocess.check_call(cmd)
    finally:
        try:
            os.remove(list_path)
        except OSError:
            pass


def make_noop_rows(last_row: pd.Series, k: int, action_col: str = "action") -> pd.DataFrame:
    """
    Create K rows by copying last_row; set action to zeros (keeping shape).
    Keeps other columns (observation/state) identical.
    """
    if k <= 0:
        return pd.DataFrame(columns=last_row.index)

    rows = []
    for _ in range(k):
        r = last_row.copy()
        # action is usually numpy array/list; set zeros of same shape
        if action_col in r.index and r[action_col] is not None:
            a = np.asarray(r[action_col], dtype=np.float32)
            r[action_col] = np.zeros_like(a, dtype=np.float32)
        rows.append(r)
    return pd.DataFrame(rows)


def normalize_episode_df(df: pd.DataFrame, new_episode_index: int, fps: float) -> pd.DataFrame:
    """
    Reindex episode fields to look like a fresh episode.
    - frame_index: 0..T-1
    - index: 0..T-1 (if present)
    - episode_index: new_episode_index
    - timestamp: uniform steps (float32), starting at 0
    """
    out = df.copy()

    T = len(out)
    if "frame_index" in out.columns:
        out["frame_index"] = np.arange(T, dtype=np.int64)

    if "index" in out.columns:
        out["index"] = np.arange(T, dtype=np.int64)

    if "episode_index" in out.columns:
        out["episode_index"] = np.full(T, new_episode_index, dtype=np.int64)

    # regenerate timestamps uniformly
    if "timestamp" in out.columns:
        dt = 1.0 / float(fps)
        out["timestamp"] = (np.arange(T, dtype=np.float32) * np.float32(dt)).astype(np.float32)

    return out


def concat_parquets_for_recipe(
    src_root: str,
    parts: List[Dict[str, Any]],
    new_episode_index: int,
    fps: float,
    chunk_size: int,
    drop_last_frame_each_part: bool,
    noop_frames: int,
    drop_first_frames_each_part: int,
) -> Tuple[pd.DataFrame, List[dict]]:
    """
    Returns (concatenated_df, provenance_parts)
    """
    dfs = []
    provenance = []

    for i, p in enumerate(parts):
        ep_id = int(p["episode_id"])
        parquet_path = episode_parquet_path(src_root, ep_id, chunk_size=chunk_size)
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Missing parquet: {parquet_path}")

        df = pd.read_parquet(parquet_path)

        # Drop first frames if requested
        if drop_first_frames_each_part > 0:
            if len(df) > drop_first_frames_each_part:
                df = df.iloc[drop_first_frames_each_part:].reset_index(drop=True)
            else:
                raise RuntimeError(f"Part too short after drop_first_frames_each_part: ep={ep_id}, len={len(df)}")

        # Optionally drop last frame to reduce action-alignment issues
        if drop_last_frame_each_part and len(df) >= 2:
            df = df.iloc[:-1].reset_index(drop=True)

        if len(df) < 2:
            raise RuntimeError(f"Part too short after drop_last_frame_each_part: ep={ep_id}, len={len(df)}")

        dfs.append(df)

        provenance.append({
            "part_idx": i,
            "episode_id": ep_id,
            "task": p.get("task", None),
            "frames_used": int(len(df)),
            "dropped_last": bool(drop_last_frame_each_part),
            "dropped_first": int(drop_first_frames_each_part),
        })

        # Boundary noop insertion between parts (after this part, if not last)
        if noop_frames > 0 and i < len(parts) - 1:
            last_row = df.iloc[-1]
            noop_df = make_noop_rows(last_row, noop_frames, action_col="action")
            dfs.append(noop_df)
            provenance.append({
                "part_idx": i,
                "episode_id": ep_id,
                "task": p.get("task", None),
                "frames_used": int(noop_frames),
                "is_noop_boundary_padding": True,
            })

    cat = pd.concat(dfs, ignore_index=True)

    # Normalize indices/timestamps/episode_index for new episode
    cat = normalize_episode_df(cat, new_episode_index=new_episode_index, fps=fps)
    return cat, provenance


def main(
    src_root: str,
    recipes_jsonl: str,
    out_root: str,
    fps: float = 5.0,
    video_keys: Tuple[str, ...] = ("observation.images.table", "observation.images.wrist"),
    chunk_size: int = 50,
    reencode_video: bool = True,
    fail_on_missing_video: bool = False,
    drop_first_frames_each_part_override: int | None = None,
):
    safe_mkdir(out_root)

    recipes = read_jsonl(recipes_jsonl)
    if not recipes:
        raise RuntimeError(f"No recipes found: {recipes_jsonl}")

    manifest = []
    ok = 0
    skipped = 0

    for r in recipes:
        new_ep = int(r["new_episode_index"])
        parts = r["parts"]
        bp = r.get("boundary_policy", {}) or {}
        noop_frames = int(bp.get("noop_frames", 0))
        drop_last_frame_each_part = bool(bp.get("drop_last_frame_each_part", True))
        drop_first_frames_each_part = int(bp.get("drop_first_frames_each_part", 0))
        if drop_first_frames_each_part_override is not None and drop_first_frames_each_part_override >= 0:
            drop_first_frames_each_part = drop_first_frames_each_part_override

        # ---- concat parquet ----
        try:
            cat_df, prov = concat_parquets_for_recipe(
                src_root=src_root,
                parts=parts,
                new_episode_index=new_ep,
                fps=fps,
                chunk_size=chunk_size,
                drop_last_frame_each_part=drop_last_frame_each_part,
                noop_frames=noop_frames,
                drop_first_frames_each_part=drop_first_frames_each_part,
            )
        except Exception as e:
            print(f"[SKIP] new_ep={new_ep:06d} parquet concat failed: {e}")
            skipped += 1
            continue

        out_pq = out_parquet_path(out_root, new_ep, chunk_size=chunk_size)
        cat_df.to_parquet(out_pq, index=False)

        # ---- concat videos (for each key) ----
        video_status = {}
        for vk in video_keys:
            # fast path: if no drop_first or drop_last, use original in_paths
            if drop_first_frames_each_part == 0 and not drop_last_frame_each_part:
                in_paths = []
                missing = False
                for p in parts:
                    ep_id = int(p["episode_id"])
                    vpath = episode_video_path(src_root, ep_id, vk, chunk_size=chunk_size)
                    if not os.path.exists(vpath):
                        missing = True
                        if fail_on_missing_video:
                            raise FileNotFoundError(f"Missing video: {vpath}")
                        else:
                            print(f"[WARN] missing video: {vpath} (new_ep={new_ep:06d}, vk={vk})")
                            break
                    in_paths.append(vpath)
                if missing:
                    video_status[vk] = {"ok": False, "reason": "missing_input_video"}
                    continue
                out_v = out_video_path(out_root, new_ep, vk, chunk_size=chunk_size)
                try:
                    ffmpeg_concat_videos(in_paths, out_v, reencode=reencode_video, fps=fps)
                    video_status[vk] = {"ok": True, "out": os.path.relpath(out_v, out_root)}
                except Exception as e:
                    print(f"[WARN] video concat failed: new_ep={new_ep:06d} vk={vk} err={e}")
                    video_status[vk] = {"ok": False, "reason": f"ffmpeg_failed: {e}"}
                continue

            # otherwise, trim each video part to match the parquet slicing
            with tempfile.TemporaryDirectory() as tmpdir:
                trimmed_paths = []
                missing = False
                for pi, p in enumerate(parts):
                    ep_id = int(p["episode_id"])
                    in_vid = episode_video_path(src_root, ep_id, vk, chunk_size=chunk_size)
                    if not os.path.exists(in_vid):
                        missing = True
                        if fail_on_missing_video:
                            raise FileNotFoundError(f"Missing video: {in_vid}")
                        else:
                            print(f"[WARN] missing video: {in_vid} (new_ep={new_ep:06d}, vk={vk})")
                            break
                    # Find provenance entry for this part index (and not noop)
                    prov_entry = None
                    for entry in prov:
                        if entry.get("part_idx") == pi and not entry.get("is_noop_boundary_padding", False):
                            prov_entry = entry
                            break
                    if prov_entry is None:
                        raise RuntimeError(f"Missing provenance for part {pi} (new_ep={new_ep:06d})")
                    frames_used = prov_entry["frames_used"]
                    dropped_first = prov_entry.get("dropped_first", 0)
                    # Compute trimming params
                    start_sec = float(dropped_first) / float(fps)
                    dur_sec = float(frames_used) / float(fps)
                    tmp_clip_path = os.path.join(tmpdir, f"part{pi:02d}.mp4")
                    ffmpeg_trim_video(
                        in_vid, tmp_clip_path,
                        start_sec=start_sec,
                        dur_sec=dur_sec,
                        reencode=reencode_video,
                        fps=fps,
                    )
                    trimmed_paths.append(tmp_clip_path)
                if missing:
                    video_status[vk] = {"ok": False, "reason": "missing_input_video"}
                    continue
                out_v = out_video_path(out_root, new_ep, vk, chunk_size=chunk_size)
                try:
                    ffmpeg_concat_videos(trimmed_paths, out_v, reencode=reencode_video, fps=fps)
                    video_status[vk] = {"ok": True, "out": os.path.relpath(out_v, out_root)}
                except Exception as e:
                    print(f"[WARN] video concat failed: new_ep={new_ep:06d} vk={vk} err={e}")
                    video_status[vk] = {"ok": False, "reason": f"ffmpeg_failed: {e}"}

        manifest.append({
            "new_episode_index": new_ep,
            "num_frames": int(len(cat_df)),
            "out_parquet": os.path.relpath(out_pq, out_root),
            "boundary_policy": {
                "noop_frames": noop_frames,
                "drop_last_frame_each_part": drop_last_frame_each_part,
                "drop_first_frames_each_part": drop_first_frames_each_part,
            },
            "parts": parts,
            "parquet_provenance": prov,
            "videos": video_status,
        })

        ok += 1
        print(f"[OK] new_ep={new_ep:06d} frames={len(cat_df)} parts={len(parts)} noop={noop_frames} drop_last={drop_last_frame_each_part} drop_first={drop_first_frames_each_part}")

    manifest_path = os.path.join(out_root, "concat_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Done. ok={ok}, skipped={skipped}. Manifest: {manifest_path}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", required=True)
    ap.add_argument("--recipes", required=True, help="recipes.jsonl")
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--chunk_size", type=int, default=50, help="Number of episodes per chunk directory (e.g., 50)")
    ap.add_argument("--video_keys", nargs="+", default=["observation.images.table", "observation.images.wrist"])
    ap.add_argument("--reencode_video", action="store_true", default=True)
    ap.add_argument("--no_reencode_video", action="store_false", dest="reencode_video")
    ap.add_argument("--fail_on_missing_video", action="store_true", default=False)
    ap.add_argument("--drop_first_frames_each_part", type=int, default=0, help="Drop the first N frames of each input part before concatenation")
    args = ap.parse_args()

    main(
        src_root=args.src_root,
        recipes_jsonl=args.recipes,
        out_root=args.out_root,
        fps=args.fps,
        video_keys=tuple(args.video_keys),
        chunk_size=args.chunk_size,
        reencode_video=args.reencode_video,
        fail_on_missing_video=args.fail_on_missing_video,
        drop_first_frames_each_part_override=args.drop_first_frames_each_part,
    )
