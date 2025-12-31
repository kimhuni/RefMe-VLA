from __future__ import annotations
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

import streamlit as st  # type: ignore
from PIL import Image
from helm_datasets_v3.utils.io_utils import (
    list_chunks_from_frames,
    list_episodes_from_frames,
    frames_dir,
    read_json,
    atomic_write_json,
    update_index_event,
)

"""
export PYTHONPATH=$(pwd)
streamlit run helm_datasets_v3/annotate_app.py --\
    --out_root "/data/ghkim/helm_data/press_the_button_nolight" \
    --fps_frames 5
"""

st.set_page_config(layout="wide", page_title="HeLM Event Annotator")


def load_frame(path: Path):
    if not path.exists():
        return None
    return Image.open(path)


def episode_frames(out_root: Path, fps_out: int, chunk: str, episode: str, camera: str):
    d = frames_dir(out_root, fps_out, chunk, episode, camera)
    return sorted(d.glob("frame_*.jpg"))

# Helper: parse frame id from frame_000123.jpg -> 123
def _parse_frame_id(p: Path) -> Optional[int]:
    try:
        stem = p.stem
        if stem.startswith("frame_"):
            return int(stem[6:])
    except Exception:
        pass
    return None

# Helper: get path for frame id in camera
def _frame_path(out_root, fps_out, chunk, ep, camera, frame_id) -> Path:
    return frames_dir(out_root, fps_out, chunk, ep, camera) / f"frame_{frame_id:06d}.jpg"


def episode_json_path(out_root: Path, fps_out: int, chunk: str, episode: str) -> Path:
    """Per-episode metadata/event JSON stored alongside frames for the given fps_out."""
    d = frames_dir(out_root, fps_out, chunk, episode, "table").parent
    return d / f"{episode}.json"


def _expand_range_spec(spec: str) -> List[int]:
    spec = spec.strip()
    if not spec:
        return []
    if ":" in spec:
        parts = spec.split(":")
    elif "-" in spec:
        parts = spec.split("-")
    else:
        # single number?
        try:
            val = int(spec)
            return [val]
        except Exception:
            return []
    if len(parts) != 2:
        return []
    try:
        start = int(parts[0].strip())
        end = int(parts[1].strip())
        if start <= end:
            return list(range(start, end + 1))
        else:
            return list(range(end, start + 1))
    except Exception:
        return []


def _coerce_int(x) -> Optional[int]:
    if isinstance(x, int):
        return x
    if isinstance(x, float) and x.is_integer():
        return int(x)
    try:
        return int(x)
    except Exception:
        return None


def _coerce_int_list(lst) -> List[int]:
    if not isinstance(lst, (list, tuple)):
        return []
    result = []
    for x in lst:
        v = _coerce_int(x)
        if v is not None:
            result.append(v)
    return result


def move_frames_to_deleted(frame_paths: List[Path], deleted_dir: Path) -> int:
    deleted_dir.mkdir(parents=True, exist_ok=True)
    moved_count = 0
    for fp in frame_paths:
        if fp.exists():
            target = deleted_dir / fp.name
            try:
                shutil.move(str(fp), str(target))
                moved_count += 1
            except Exception:
                pass
    return moved_count


def main(out_root: Path, fps_out: int):
    st.title(f"HeLM Event Annotation ({fps_out}Hz frames)")
    st.caption(f"Event = event 발생 프레임 인덱스. Episode 폴더에 <episode>.json으로 저장됩니다.")

    chunks = list_chunks_from_frames(out_root, fps_out)
    if not chunks:
        st.error(f"frames_{fps_out}hz가 없습니다. 먼저 extract_frames를 실행하세요.")
        return

    # ---- Sidebar: chunk 선택 ----
    chunk = st.sidebar.selectbox("Chunk", chunks, index=0)

    episodes = list_episodes_from_frames(out_root, fps_out, chunk)
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


    episode_key = f"{fps_out}hz::{chunk}/{ep}"
    events_key = f"event_frame_idxs::{episode_key}"
    event_key = f"event_frame_idx::{episode_key}"  # derived from events_key, not independently edited
    frame_key = f"frame_idx::{episode_key}"

    # Initialize event frame indexes in session state if missing (backward compatible)
    if events_key not in st.session_state:
        ep_json = episode_json_path(out_root, fps_out, chunk, ep)
        ev = read_json(ep_json, default={})
        event_frame_idxs = []
        if isinstance(ev, dict):
            if "event_frame_idxs" in ev and isinstance(ev["event_frame_idxs"], list):
                event_frame_idxs = [int(x) for x in ev["event_frame_idxs"] if isinstance(x, int) or (isinstance(x, float) and x.is_integer())]
            elif "event_frame_idx" in ev and (ev["event_frame_idx"] is not None):
                val = ev["event_frame_idx"]
                if isinstance(val, int) or (isinstance(val, float) and val.is_integer()):
                    event_frame_idxs = [int(val)]
        st.session_state[events_key] = sorted(set(event_frame_idxs))
    # Always keep event_key in sync with first event or None
    st.session_state[event_key] = st.session_state[events_key][0] if st.session_state[events_key] else None

    # --- Load frame lists and frame ids for both cameras ---
    def _get_frame_ids(imgs):
        return [_parse_frame_id(p) for p in imgs if _parse_frame_id(p) is not None]

    def _refresh_frame_lists():
        table_imgs = episode_frames(out_root, fps_out, chunk, ep, "table")
        wrist_imgs = episode_frames(out_root, fps_out, chunk, ep, "wrist")
        frame_ids_table = set(_get_frame_ids(table_imgs))
        frame_ids_wrist = set(_get_frame_ids(wrist_imgs))
        common_frame_ids = sorted(frame_ids_table & frame_ids_wrist)
        return table_imgs, wrist_imgs, list(frame_ids_table), list(frame_ids_wrist), common_frame_ids

    table_imgs, wrist_imgs, frame_ids_table, frame_ids_wrist, common_frame_ids = _refresh_frame_lists()

    n_common = len(common_frame_ids)
    if n_common == 0:
        st.error("No common frames in both cameras.")
        return

    # Clamp slider session state
    if frame_key not in st.session_state:
        st.session_state[frame_key] = 0
    st.session_state[frame_key] = max(0, min(st.session_state[frame_key], n_common - 1))

    # --- UI Layout ---
    col1, col2 = st.columns([2, 1])

    # ---- RIGHT: Event and Frame Delete Controls ----
    with col2:
        st.subheader("Event / Delete")

        # --- Event Controls ---
        cur_pos = int(st.session_state[frame_key])
        cur_frame_id = common_frame_ids[cur_pos]
        event_frame_idxs = list(st.session_state.get(events_key, []))

        if st.button("Add event at current frame"):
            if cur_frame_id not in event_frame_idxs:
                new_events = list(event_frame_idxs)
                new_events.append(cur_frame_id)
                new_events = sorted(set(new_events))
                st.session_state[events_key] = new_events
                st.session_state[event_key] = new_events[0] if new_events else None
                st.info(f"Added event frame id: {cur_frame_id}")
            else:
                st.info(f"Frame id {cur_frame_id} is already an event frame.")

        # --- Add event range ---
        st.markdown("**Event Controls**")
        range_cols = st.columns([3, 1])
        with range_cols[0]:
            range_input = st.text_input(
                "Add event range (inclusive)",
                placeholder="e.g., 35:41 or 35-41",
                key=f"add_range_{episode_key}",
            )
        with range_cols[1]:
            add_range_clicked = st.button("Add range", key=f"add_range_btn_{episode_key}")
        if add_range_clicked:
            frame_ids_to_add = _expand_range_spec(range_input)
            if not frame_ids_to_add:
                st.warning("Invalid range spec. Use 35:41 or 35-41.")
            else:
                # Only add those ids that are not already present
                old_set = set(st.session_state[events_key])
                new_set = set(frame_ids_to_add)
                actually_new = new_set - old_set
                merged = sorted(old_set | new_set)
                st.session_state[events_key] = merged
                st.session_state[event_key] = merged[0] if merged else None
                st.info(f"Added {len(actually_new)} event ids from range.")

        # Delete by index
        count = len(event_frame_idxs)
        if count > 0:
            del_event_idx = st.number_input("Delete event by list index", min_value=0, max_value=count - 1, step=1, value=0, key=f"del_idx_{episode_key}")
            if st.button("Delete event by list index"):
                if 0 <= del_event_idx < len(event_frame_idxs):
                    removed = event_frame_idxs.pop(del_event_idx)
                    st.session_state[events_key] = event_frame_idxs
                    st.session_state[event_key] = event_frame_idxs[0] if event_frame_idxs else None
                    st.info(f"Deleted event frame at index {del_event_idx} (frame id {removed})")
                else:
                    st.warning("Invalid delete index.")

        # Display event frame ids
        event_frame_idxs = st.session_state[events_key]
        count = len(event_frame_idxs)
        preview = ", ".join(str(x) for x in event_frame_idxs[:10]) + ("..." if count > 10 else "")
        st.write(f"Events: {count} [{preview if count > 0 else '(none)'}]")

        # --- Frame Deletion Controls ---
        st.markdown("### Delete frames")
        max_frame_id = max(common_frame_ids) if common_frame_ids else 0
        min_frame_id = min(common_frame_ids) if common_frame_ids else 0
        delete_frame_id = st.number_input("Delete frame id", min_value=min_frame_id, max_value=max_frame_id, value=cur_frame_id, key=f"del_frame_id_{episode_key}")
        if st.button("Delete this frame id"):
            moved_count = 0
            for cam in ["table", "wrist"]:
                fp = _frame_path(out_root, fps_out, chunk, ep, cam, int(delete_frame_id))
                if fp.exists():
                    deleted_dir = frames_dir(out_root, fps_out, chunk, ep, cam).parent / "_deleted_frames" / cam
                    deleted_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.move(str(fp), str(deleted_dir / fp.name))
                        moved_count += 1
                    except Exception:
                        pass
            st.success(f"Moved {moved_count} files for frame id {int(delete_frame_id)} to _deleted_frames/")
            # After deletion, refresh frames and clamp slider
            table_imgs, wrist_imgs, frame_ids_table, frame_ids_wrist, common_frame_ids = _refresh_frame_lists()
            n_common = len(common_frame_ids)
            if n_common == 0:
                st.session_state[frame_key] = 0
            else:
                st.session_state[frame_key] = min(st.session_state[frame_key], n_common - 1)
            st.rerun()

        # --- Compact: Delete frames after last event ---
        if st.button("Delete frames AFTER last event"):
            events = sorted(set(list(st.session_state[events_key])))
            if not events:
                st.warning("No events set. Add an event first.")
            else:
                last_id = max(events)
                max_frame_id = max(common_frame_ids) if common_frame_ids else 0
                if last_id >= max_frame_id:
                    st.info("No frames after last event.")
                else:
                    moved_count = 0
                    for fid in range(last_id+1, max_frame_id+1):
                        for cam in ["table", "wrist"]:
                            fp = _frame_path(out_root, fps_out, chunk, ep, cam, fid)
                            if fp.exists():
                                deleted_dir = frames_dir(out_root, fps_out, chunk, ep, cam).parent / "_deleted_frames" / cam
                                deleted_dir.mkdir(parents=True, exist_ok=True)
                                try:
                                    shutil.move(str(fp), str(deleted_dir / fp.name))
                                    moved_count += 1
                                except Exception:
                                    pass
                    st.success(f"Moved {moved_count} files for frame ids {last_id+1} to {max_frame_id} to _deleted_frames/")
                    # After deletion, refresh frames and clamp slider
                    table_imgs, wrist_imgs, frame_ids_table, frame_ids_wrist, common_frame_ids = _refresh_frame_lists()
                    n_common = len(common_frame_ids)
                    if n_common == 0:
                        st.session_state[frame_key] = 0
                    else:
                        st.session_state[frame_key] = min(st.session_state[frame_key], n_common - 1)
                    st.rerun()

        # Range deletion
        delete_start_id = st.number_input("Delete start id", min_value=min_frame_id, max_value=max_frame_id, value=cur_frame_id, key=f"del_start_id_{episode_key}")
        delete_end_id = st.number_input("Delete end id", min_value=min_frame_id, max_value=max_frame_id, value=cur_frame_id, key=f"del_end_id_{episode_key}")
        if st.button("Delete range [start..end]"):
            s = min(int(delete_start_id), int(delete_end_id))
            e = max(int(delete_start_id), int(delete_end_id))
            moved_count = 0
            for fid in range(s, e+1):
                for cam in ["table", "wrist"]:
                    fp = _frame_path(out_root, fps_out, chunk, ep, cam, fid)
                    if fp.exists():
                        deleted_dir = frames_dir(out_root, fps_out, chunk, ep, cam).parent / "_deleted_frames" / cam
                        deleted_dir.mkdir(parents=True, exist_ok=True)
                        try:
                            shutil.move(str(fp), str(deleted_dir / fp.name))
                            moved_count += 1
                        except Exception:
                            pass
            st.success(f"Moved {moved_count} files for frame ids {s} to {e} to _deleted_frames/")
            # After deletion, refresh frames and clamp slider
            table_imgs, wrist_imgs, frame_ids_table, frame_ids_wrist, common_frame_ids = _refresh_frame_lists()
            n_common = len(common_frame_ids)
            if n_common == 0:
                st.session_state[frame_key] = 0
            else:
                st.session_state[frame_key] = min(st.session_state[frame_key], n_common - 1)
            st.rerun()

        # --- Save Button ---
        events = sorted(set(list(st.session_state[events_key])))
        if st.button("Save", type="primary"):
            ep_json = episode_json_path(out_root, fps_out, chunk, ep)
            existing_data = read_json(ep_json, default={})
            if not isinstance(existing_data, dict):
                existing_data = {}
            sorted_unique = sorted(set(events))
            existing_data["fps_frames"] = int(fps_out)
            existing_data["n_frames"] = int(len(common_frame_ids))
            existing_data["event_frame_idxs"] = sorted_unique
            existing_data["event_frame_idx"] = sorted_unique[0] if sorted_unique else None
            atomic_write_json(ep_json, existing_data)
            update_index_event(out_root, fps_out, chunk, ep, sorted_unique)
            st.success(f"Saved: {ep_json}")

    # ---- LEFT: Frames Viewer ----
    with col1:
        st.subheader("Frames")

        # Optionally keep prev/next frame buttons
        cprev, cnext = st.columns(2)
        if cprev.button("◀ Prev frame"):
            st.session_state[frame_key] = max(0, st.session_state[frame_key] - 1)
            st.rerun()
        if cnext.button("Next frame ▶"):
            st.session_state[frame_key] = min(n_common - 1, st.session_state[frame_key] + 1)
            st.rerun()

        idx = int(st.slider(
            f"Frame [{st.session_state[frame_key]+1}/{n_common}] id {common_frame_ids[st.session_state[frame_key]]}",
            0, n_common - 1, st.session_state[frame_key], 1, key=frame_key
        ))
        # NOTE: Do NOT assign st.session_state[frame_key] here; the slider owns this key.
        cur_frame_id = common_frame_ids[idx]

        event_frame_idxs = st.session_state.get(events_key, [])
        if cur_frame_id in event_frame_idxs:
            first_match = event_frame_idxs.index(cur_frame_id)
            st.success(f"✅ This frame is an event frame (#{first_match})")
        else:
            st.caption("")

        c1, c2 = st.columns(2)
        with c1:
            st.caption("table")
            img = load_frame(_frame_path(out_root, fps_out, chunk, ep, "table", cur_frame_id))
            if img:
                st.image(img, use_container_width=True)
        with c2:
            st.caption("wrist")
            img2 = load_frame(_frame_path(out_root, fps_out, chunk, ep, "wrist", cur_frame_id))
            if img2:
                st.image(img2, use_container_width=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--fps_frames", type=int, default=1)
    args, _ = ap.parse_known_args()
    main(Path(args.out_root), int(args.fps_frames))