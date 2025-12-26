from __future__ import annotations
import argparse
from pathlib import Path

import streamlit as st  # type: ignore
from PIL import Image
from helm_datasets.utils.io_utils import (
    list_chunks_from_frames,
    list_episodes_from_frames,
    frames_dir,
    read_json,
    atomic_write_json,
    update_index_event,
)

"""
export PYTHONPATH=$(pwd)
streamlit run helm_datasets/annotate_app.py --\
    --out_root "/data/ghkim/helm_data/press_the_button_N_times_ep60"
"""

st.set_page_config(layout="wide", page_title="HeLM Event Annotator")


def load_frame(path: Path):
    if not path.exists():
        return None
    return Image.open(path)


def episode_frames(out_root: Path, chunk: str, episode: str, camera: str):
    d = frames_dir(out_root, chunk, episode, camera)
    return sorted(d.glob("frame_*.jpg"))


def episode_json_path(out_root: Path, chunk: str, episode: str) -> Path:
    """Per-episode metadata/event JSON stored alongside frames."""
    # frames_dir(out_root, chunk, episode, camera) -> .../frames_1hz/<chunk>/<episode>/<camera>
    d = frames_dir(out_root, chunk, episode, "table").parent
    return d / f"{episode}.json"


def main(out_root: Path):
    st.title("HeLM Event Annotation (1Hz frames)")
    st.caption("Event = event 발생 프레임 인덱스. Episode 폴더에 <episode>.json으로 저장됩니다.")

    chunks = list_chunks_from_frames(out_root)
    if not chunks:
        st.error("frames_1hz가 없습니다. 먼저 extract_frames를 실행하세요.")
        return

    # ---- Sidebar: chunk 선택 ----
    chunk = st.sidebar.selectbox("Chunk", chunks, index=0)

    episodes = list_episodes_from_frames(out_root, chunk)
    if not episodes:
        st.error(f"{chunk}에 episode가 없습니다.")
        return

    # ---- Session state: episode index ----
    if "ep_idx" not in st.session_state:
        st.session_state.ep_idx = 0
    st.session_state.ep_idx = max(0, min(st.session_state.ep_idx, len(episodes) - 1))


    # Prev/Next 버튼
    c_prev, c_next = st.sidebar.columns(2)
    if c_prev.button("◀ Prev"):
        st.session_state.ep_idx = max(0, st.session_state.ep_idx - 1)
        st.session_state["episode_selectbox"] = episodes[st.session_state.ep_idx]
        st.rerun()
    if c_next.button("Next ▶"):
        st.session_state.ep_idx = min(len(episodes) - 1, st.session_state.ep_idx + 1)
        st.session_state["episode_selectbox"] = episodes[st.session_state.ep_idx]
        st.rerun()

    ep = st.sidebar.selectbox(
        "Episode",
        episodes,
        index=st.session_state.ep_idx,
        key="episode_selectbox",
    )
    # selectbox에서 직접 바꿨을 때도 ep_idx 동기화
    if ep != episodes[st.session_state.ep_idx]:
        st.session_state.ep_idx = episodes.index(ep)


    episode_key = f"{chunk}/{ep}"
    event_key = f"event_frame_idx::{episode_key}"
    frame_key = f"frame_idx::{episode_key}"

    # Initialize event frame index in session state if missing
    if event_key not in st.session_state:
        ep_json = episode_json_path(out_root, chunk, ep)
        ev = read_json(ep_json, default={"fps_frames": 1, "n_frames": None, "event_frame_idx": None})
        v = ev.get("event_frame_idx", None)
        st.session_state[event_key] = None if v is None else int(v)

    # Initialize frame index in session state if missing
    table_imgs = episode_frames(out_root, chunk, ep, "table")
    wrist_imgs = episode_frames(out_root, chunk, ep, "wrist")
    n = min(len(table_imgs), len(wrist_imgs))
    if frame_key not in st.session_state:
        st.session_state[frame_key] = 0
    st.session_state[frame_key] = max(0, min(st.session_state[frame_key], n - 1))

    if n == 0:
        st.error("프레임이 없습니다.")
        return

    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("Event")

        if st.button("Set event at current frame"):
            idx = int(st.session_state[frame_key])
            st.session_state[event_key] = idx
            st.info(f"Set event frame: {idx} (remember to Save)")

        if st.button("Save", type="primary"):
            if st.session_state[event_key] is None:
                st.error("event_frame_idx가 설정되지 않았습니다. 먼저 'Set event at current frame'를 누르세요.")
            else:
                ep_json = episode_json_path(out_root, chunk, ep)
                existing_data = read_json(ep_json, default={})
                if not isinstance(existing_data, dict):
                    existing_data = {}
                existing_data["fps_frames"] = 1
                existing_data["n_frames"] = int(n)
                existing_data["event_frame_idx"] = int(st.session_state[event_key])
                atomic_write_json(ep_json, existing_data)
                # keep index update compatible by passing a single boundary list
                update_index_event(out_root, chunk, ep, [existing_data["event_frame_idx"]])
                st.success(f"Saved: {ep_json}")

        cur = st.session_state.get(event_key, None)
        st.write(cur if cur is not None else "(not set)")

        if cur is not None:
            if st.button(f"Go to frame {cur}", key=f"jump_{chunk}_{ep}_{cur}"):
                st.session_state[frame_key] = int(cur)

        c1, c2, c3 = st.columns(3)
        if c1.button("Clear"):
            st.session_state[event_key] = None
        if c2.button("Reload"):
            ep_json = episode_json_path(out_root, chunk, ep)
            ev = read_json(ep_json, default={"fps_frames": 1, "n_frames": None, "event_frame_idx": None})
            v = ev.get("event_frame_idx", None)
            st.session_state[event_key] = None if v is None else int(v)
            st.info("Reloaded from disk.")
        if c3.button("Set to current"):
            st.session_state[event_key] = int(st.session_state[frame_key])
            st.info("Set to current frame. (Remember to Save)")

        st.caption(f"Episode JSON: {episode_json_path(out_root, chunk, ep)}")

    with col1:
        st.subheader("Frames")

        # slider는 session_state[frame_key]를 사용하도록 key를 줌
        idx = int(st.slider("Frame index", 0, n - 1, st.session_state[frame_key], 1, key=frame_key))

        ev_idx = st.session_state.get(event_key, None)
        if ev_idx is not None and idx == int(ev_idx):
            st.success(f"✅ This frame ({idx}) is the event frame.")
        else:
            st.caption("")

        c1, c2 = st.columns(2)
        with c1:
            st.caption("table")
            img = load_frame(table_imgs[idx])
            if img:
                st.image(img, use_container_width=True)
        with c2:
            st.caption("wrist")
            img2 = load_frame(wrist_imgs[idx])
            if img2:
                st.image(img2, use_container_width=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True)
    args, _ = ap.parse_known_args()
    main(Path(args.out_root))