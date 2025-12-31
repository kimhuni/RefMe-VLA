from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image
"""
export PYTHONPATH=$(pwd)
streamlit run helm_datasets_v2/validate_helm_data.py
"""

# ---------- IO ----------
def collect_jsonl_files(p: Path) -> List[Path]:
    if p.is_file() and p.suffix == ".jsonl":
        return [p]
    if p.is_dir():
        return sorted(p.rglob("*.jsonl"))
    return []


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                st.warning(f"JSON parse error {path}:{ln}: {e}")
    return rows


def open_image(path: str) -> Optional[Image.Image]:
    try:
        p = Path(path)
        if not p.exists():
            return None
        return Image.open(p)
    except Exception:
        return None


# ---------- Index helpers ----------
def row_episode_key(r: Dict[str, Any]) -> Tuple[Any, Any]:
    return (r.get("chunk"), r.get("episode"))


def build_episode_index(rows: List[Dict[str, Any]]) -> Dict[Tuple[Any, Any], List[int]]:
    idx: Dict[Tuple[Any, Any], List[int]] = {}
    for i, r in enumerate(rows):
        key = row_episode_key(r)
        if key[0] is None or key[1] is None:
            continue
        idx.setdefault(key, []).append(i)
    # keep stable order
    for k in idx:
        idx[k].sort()
    return idx


def current_episode_rows(ep_index: Dict[Tuple[Any, Any], List[int]], rows: List[Dict[str, Any]], cur_i: int) -> List[int]:
    key = row_episode_key(rows[cur_i])
    return ep_index.get(key, [])


def rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


# ---------- UI ----------
def main():
    st.set_page_config(layout="wide", page_title="HeLM v2 Simple Viewer")
    st.title("HeLM v2 Simple Viewer (detect/update)")

    st.sidebar.header("Load JSONL")
    path_str = st.sidebar.text_input("JSONL file or folder", value="")
    if st.sidebar.button("Load"):
        p = Path(path_str)
        files = collect_jsonl_files(p)
        if not files:
            st.sidebar.error("No .jsonl found.")
        else:
            st.session_state.files = [str(x) for x in files]
            st.session_state.file_idx = 0
            st.session_state.rows = read_jsonl(files[0])
            st.session_state.row_idx = 0

    if "files" not in st.session_state:
        st.info("왼쪽에서 jsonl 파일/폴더를 입력하고 Load를 누르세요.")
        return

    files: List[str] = st.session_state.files
    if not files:
        st.warning("No files loaded.")
        return

    # choose file
    file_idx = st.sidebar.selectbox(
        "File",
        options=list(range(len(files))),
        index=int(st.session_state.get("file_idx", 0)),
        format_func=lambda i: Path(files[i]).name,
    )

    if file_idx != st.session_state.get("file_idx", 0):
        st.session_state.file_idx = file_idx
        st.session_state.rows = read_jsonl(Path(files[file_idx]))
        st.session_state.row_idx = 0

    rows: List[Dict[str, Any]] = st.session_state.rows
    if not rows:
        st.warning("Empty JSONL.")
        return

    # build episode index
    ep_index = build_episode_index(rows)
    ep_keys = sorted(ep_index.keys(), key=lambda x: (str(x[0]), str(x[1])))

    st.sidebar.header("Navigate")
    # row slider
    row_idx = st.sidebar.slider("Row", 0, len(rows) - 1, int(st.session_state.get("row_idx", 0)))
    st.session_state.row_idx = int(row_idx)

    # prev/next row
    c1, c2 = st.sidebar.columns(2)
    if c1.button("◀ Prev row"):
        st.session_state.row_idx = max(0, st.session_state.row_idx - 1)
        rerun()
    if c2.button("Next row ▶"):
        st.session_state.row_idx = min(len(rows) - 1, st.session_state.row_idx + 1)
        rerun()

    # episode jump
    st.sidebar.subheader("Episode jump")
    cur_key = row_episode_key(rows[st.session_state.row_idx])
    cur_ep_label = f"{cur_key[0]}/{cur_key[1]}"
    # find current episode index
    cur_ep_pos = 0
    if cur_key in ep_index:
        try:
            cur_ep_pos = ep_keys.index(cur_key)
        except ValueError:
            cur_ep_pos = 0

    ep_pos = st.sidebar.selectbox(
        "Episode",
        options=list(range(len(ep_keys))),
        index=cur_ep_pos,
        format_func=lambda i: f"{ep_keys[i][0]}/{ep_keys[i][1]} ({len(ep_index[ep_keys[i]])} rows)",
    )

    if st.sidebar.button("Go to episode"):
        # jump to first row of that episode
        st.session_state.row_idx = ep_index[ep_keys[ep_pos]][0]
        rerun()

    # show current
    r = rows[st.session_state.row_idx]
    mode = r.get("mode")

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("Image (if exists)")
        images = r.get("images", {})
        if isinstance(images, dict) and images:
            # show all cameras sorted
            cams = sorted(images.keys())
            for cam in cams:
                st.caption(f"camera: {cam}")
                im = open_image(str(images[cam]))
                if im is None:
                    st.warning(f"Missing image: {images[cam]}")
                else:
                    st.image(im, use_container_width=True)
        else:
            st.info("No images in this row (likely update row).")



        # show same-episode row list quick jump
        st.subheader("Rows in this episode")
        ep_rows = current_episode_rows(ep_index, rows, st.session_state.row_idx)
        if ep_rows:
            # show small selector
            pick = st.selectbox(
                "Jump within episode",
                options=ep_rows,
                index=max(0, ep_rows.index(st.session_state.row_idx)) if st.session_state.row_idx in ep_rows else 0,
                format_func=lambda i: f"row {i} | mode={rows[i].get('mode')} | t={rows[i].get('t')} | t_event={rows[i].get('t_event')}",
            )
            if st.button("Go to selected row"):
                st.session_state.row_idx = int(pick)
                rerun()

    with right:
        st.subheader("Key fields")
        st.markdown("**global_instruction**")
        st.code(str(r.get("global_instruction", "")))

        st.markdown("**memory_in**")
        st.json(r.get("memory_in", {}))

        st.markdown("**label**")
        st.json(r.get("label", {}))

        st.subheader("Row JSON")
        st.json(r)

        pc = r.get("prompt_context", {})
        if isinstance(pc, dict) and pc:
            st.markdown("**prompt_context**")
            st.json(pc)


if __name__ == "__main__":
    main()