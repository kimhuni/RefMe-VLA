# tools/segment_annotator_app.py
import os
import json
import glob
import math
import subprocess
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

"""
streamlit run common/datasets/libero_dataset/segment_annotator_app.py
"""

# ----------------------------
# Config
# ----------------------------
st.set_page_config(layout="wide")

def episode_parquet_path(dataset_root: str, episode_index: int, chunk_size: int = 1000) -> str:
    chunk = episode_index // chunk_size
    return os.path.join(dataset_root, f"data/chunk-{chunk:03d}/episode_{episode_index:06d}.parquet")

def episode_video_path(dataset_root: str, episode_index: int, video_key: str, chunk_size: int = 1000) -> str:
    chunk = episode_index // chunk_size
    return os.path.join(dataset_root, f"videos/chunk-{chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")

def safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def ffmpeg_extract_frame(video_path: str, time_sec: float, out_png: str):
    """
    Extract a single frame at time_sec to out_png using ffmpeg.
    - Accurate seeking: place -ss after -i (slower, but precise)
    """
    safe_mkdir(str(Path(out_png).parent))
    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-ss", f"{time_sec:.6f}",
        "-frames:v", "1",
        "-y",
        out_png,
    ]
    subprocess.check_call(cmd)

def load_episode_df(parquet_path: str) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)

def get_episode_frame_count(df: pd.DataFrame) -> int:
    # frame_index exists; often 0..T-1
    if "frame_index" in df.columns:
        return int(df["frame_index"].max()) + 1
    return len(df)

def append_jsonl(path: str, obj: dict):
    safe_mkdir(str(Path(path).parent))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# ----------------------------
# UI
# ----------------------------
st.title("LeRobot Episode Segment Annotator (start/end frame)")

with st.sidebar:
    dataset_root = st.text_input("Dataset root", value="/data/ghkim/wipe_the_window_ep150")
    chunk_size = st.number_input("Chunk size", value=1000, min_value=1, step=1)
    fps = st.number_input("Video FPS (from dataset.json)", value=5.0, min_value=0.1, step=0.1)
    out_jsonl = st.text_input("Output segments.jsonl", value=os.path.join(dataset_root, "segments.jsonl"))
    video_keys = st.multiselect("Video keys", options=["observation.images.table", "observation.images.wrist"],
                               default=["observation.images.table", "observation.images.wrist"])

# Gather episodes by scanning parquet files
parquet_glob = os.path.join(dataset_root, "data/chunk-*/episode_*.parquet")
parquet_files = sorted(glob.glob(parquet_glob))
if not parquet_files:
    st.error(f"No parquet files found: {parquet_glob}")
    st.stop()

# Extract episode indices from filenames
def parse_ep_idx(p: str) -> int:
    name = Path(p).stem  # episode_000123
    return int(name.split("_")[-1])

episode_indices = [parse_ep_idx(p) for p in parquet_files]
episode_indices_sorted = sorted(episode_indices)

colA, colB = st.columns([1, 2], gap="large")

with colA:
    ep = st.selectbox("Episode index", episode_indices_sorted, index=0)
    parquet_path = episode_parquet_path(dataset_root, ep, chunk_size=chunk_size)
    df = load_episode_df(parquet_path)
    T = get_episode_frame_count(df)
    st.write(f"Parquet: `{parquet_path}`")
    st.write(f"Frames: **{T}**")

    # Segment basics
    seg_id = st.number_input("segment_id (within episode)", value=0, min_value=0, step=1)
    subtask = st.text_input("subtask label", value="")
    notes = st.text_area("notes (optional)", value="")

    start_end = st.slider("start/end frame (end is exclusive)", min_value=0, max_value=max(T, 1), value=(0, min(T, 1)))
    start_f, end_f = int(start_end[0]), int(start_end[1])

    # Preview frame (single frame)
    preview_f = st.slider("preview frame", min_value=0, max_value=max(T-1, 0), value=min(start_f, max(T-1, 0)))
    preview_sec = preview_f / float(fps)
    st.caption(f"preview time: {preview_sec:.3f}s  (frame={preview_f}, fps={fps})")

    # Optional: event_frame_idxs input
    event_frames_str = st.text_input("event_frame_idxs (comma-separated, optional)", value="")
    event_frame_idxs = []
    if event_frames_str.strip():
        try:
            event_frame_idxs = [int(x.strip()) for x in event_frames_str.split(",") if x.strip()]
        except Exception:
            st.warning("Failed to parse event_frame_idxs. Leave blank or use '12, 34' format.")

    save = st.button("Save segment → segments.jsonl", type="primary")

with colB:
    st.subheader("Video preview (single frame extracted by ffmpeg)")
    # Resolve actual video paths: video_key in dataset.json is like "observation.images.table"
    # but folder names are usually the last token "table"/"wrist"
    # Your video_path template: videos/chunk-xxx/{video_key}/episode_xxxxxx.mp4
    # => video_key should be "observation.images.table" folder name in your dataset.
    # If in your dataset the folder is actually "observation.images.table", keep as is.
    # If it's "table", you can adjust mapping below.
    def video_folder_name(vk: str) -> str:
        # default assumes folder is exactly vk
        return vk

    img_cols = st.columns(len(video_keys) if video_keys else 1)
    for i, vk in enumerate(video_keys):
        with img_cols[i]:
            vpath = episode_video_path(dataset_root, ep, video_folder_name(vk), chunk_size=chunk_size)
            st.write(vk)
            st.code(vpath)
            if os.path.exists(vpath):
                tmp_png = os.path.join(dataset_root, ".tmp_preview", f"ep{ep:06d}_{vk.replace('/','_')}_f{preview_f:06d}.png")
                try:
                    ffmpeg_extract_frame(vpath, preview_sec, tmp_png)
                    img = Image.open(tmp_png).convert("RGB")
                    st.image(img, use_container_width=True)
                except Exception as e:
                    st.error(f"ffmpeg extract failed: {e}")
                    st.video(vpath)
            else:
                st.warning("Video not found")

# Save segment
if save:
    if end_f <= start_f:
        st.error("Invalid segment: end must be > start")
        st.stop()

    obj = {
        "episode_id": int(ep),
        "segment_id": int(seg_id),
        "start": int(start_f),
        "end": int(end_f),
        "subtask": subtask,
        "notes": notes,
        "event_frame_idxs": event_frame_idxs,
    }
    append_jsonl(out_jsonl, obj)
    st.success(f"Saved to {out_jsonl}\n{obj}")