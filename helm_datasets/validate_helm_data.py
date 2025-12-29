from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
from PIL import Image

try:
    import yaml  # type: ignore
    HAS_YAML = True
except Exception:
    yaml = None
    HAS_YAML = False

"""
export PYTHONPATH=$(pwd)
streamlit run helm_datasets_v2/validate_helm_data.py
"""

# -------------------------
# IO helpers
# -------------------------
def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as e:
                st.warning(f"JSON parse error at {path} line {line_no}: {e}")
    return rows


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def collect_jsonl_files(input_path: Path) -> List[Path]:
    if input_path.is_file() and input_path.suffix == ".jsonl":
        return [input_path]
    if input_path.is_dir():
        return sorted(input_path.rglob("*.jsonl"))
    return []


def safe_open_image(p: str) -> Optional[Image.Image]:
    try:
        path = Path(p)
        if not path.exists():
            return None
        return Image.open(path)
    except Exception:
        return None


# -------------------------
# Validation helpers
# -------------------------
def extract_user_assistant(row: Dict[str, Any]) -> Tuple[str, str]:
    conv = row.get("conversations", [])
    user_text = ""
    asst_text = ""
    if isinstance(conv, list):
        for m in conv:
            if not isinstance(m, dict):
                continue
            if m.get("from") == "user":
                user_text = str(m.get("value", ""))
            if m.get("from") == "assistant":
                asst_text = str(m.get("value", ""))
    return user_text, asst_text


def validate_assistant_yaml(text: str) -> Tuple[bool, str, Optional[Dict[str, Any]]]:
    """
    Validate that assistant text is YAML with keys Progress, World_State, Command.
    Returns (ok, message, parsed_obj)
    """
    required = {"Progress", "World_State", "Command"}

    if HAS_YAML:
        try:
            obj = yaml.safe_load(text)  # type: ignore
        except Exception as e:
            return False, f"YAML parse error: {e}", None

        if not isinstance(obj, dict):
            return False, "YAML must parse to a mapping/dict.", None

        missing = required - set(obj.keys())
        if missing:
            return False, f"Missing keys: {sorted(missing)}", obj

        return True, "OK", obj

    # fallback (no pyyaml): minimal key check
    missing = [k for k in required if f"{k}:" not in text]
    if missing:
        return False, f"(No PyYAML) Missing keys (string check): {missing}", None
    return True, "(No PyYAML) Basic key check OK", None


def update_assistant_in_row(row: Dict[str, Any], new_asst: str) -> None:
    conv = row.get("conversations", [])
    if not isinstance(conv, list):
        conv = []
        row["conversations"] = conv

    found = False
    for m in conv:
        if isinstance(m, dict) and m.get("from") == "assistant":
            m["value"] = new_asst
            found = True
            break
    if not found:
        conv.append({"from": "assistant", "value": new_asst})


# -------------------------
# App
# -------------------------
def main():
    st.set_page_config(layout="wide", page_title="HeLM JSONL Reviewer")

    st.title("HeLM JSONL 검수/수정 도구")
    st.caption("생성된 JSONL(row)을 이미지와 함께 확인하고 assistant YAML을 수정한 뒤 다시 저장합니다.")

    # Input selection
    st.sidebar.header("Load")
    input_path_str = st.sidebar.text_input("JSONL file or folder", value="")
    load_clicked = st.sidebar.button("Load")

    if "rows" not in st.session_state:
        st.session_state.rows = []
        st.session_state.files = []
        st.session_state.file_idx = 0
        st.session_state.row_idx = 0
        st.session_state.dirty = False

    if load_clicked:
        ip = Path(input_path_str)
        files = collect_jsonl_files(ip)
        if not files:
            st.sidebar.error("No .jsonl files found.")
        else:
            st.session_state.files = [str(p) for p in files]
            st.session_state.file_idx = 0
            st.session_state.row_idx = 0
            st.session_state.rows = read_jsonl(Path(st.session_state.files[0]))
            st.session_state.dirty = False
            st.sidebar.success(f"Loaded {len(st.session_state.rows)} rows from {files[0]}")

    files = st.session_state.files
    if not files:
        st.info("좌측에서 JSONL 파일(또는 폴더)을 입력하고 Load를 누르세요.")
        st.stop()

    # File chooser
    file_idx = st.sidebar.selectbox(
        "File",
        options=list(range(len(files))),
        format_func=lambda i: Path(files[i]).name,
        index=st.session_state.file_idx,
    )
    if file_idx != st.session_state.file_idx:
        st.session_state.file_idx = file_idx
        st.session_state.rows = read_jsonl(Path(files[file_idx]))
        st.session_state.row_idx = 0
        st.session_state.dirty = False

    rows: List[Dict[str, Any]] = st.session_state.rows
    if not rows:
        st.warning("This JSONL file is empty or failed to parse.")
        st.stop()

    # Build an index for fast navigation:
    # sequence key = (chunk, episode, inter, base_intra)
    # value = sorted list of (frame_idx, row_idx)
    seq_to_frames: Dict[Tuple[Any, Any, Any, Any], List[Tuple[int, int]]] = {}
    for i, r in enumerate(rows):
        key = (r.get("chunk"), r.get("episode"), r.get("inter"), r.get("base_intra"))
        try:
            frame = int(r.get("frame_idx", -1))
        except Exception:
            continue
        if frame < 0:
            continue
        seq_to_frames.setdefault(key, []).append((frame, i))

    for key, lst in seq_to_frames.items():
        lst.sort(key=lambda x: x[0])

    seq_order: List[Tuple[Any, Any, Any, Any]] = sorted(seq_to_frames.keys())

    def _find_row_idx_for_frame(delta: int) -> Optional[int]:
        cur = rows[st.session_state.row_idx]
        key = (cur.get("chunk"), cur.get("episode"), cur.get("inter"), cur.get("base_intra"))
        frames = seq_to_frames.get(key)
        if not frames:
            return None

        # Find current position in this sequence by row_idx (fallback to frame_idx match)
        cur_pos: Optional[int] = None
        for pos, (_f, idx) in enumerate(frames):
            if idx == st.session_state.row_idx:
                cur_pos = pos
                break
        if cur_pos is None:
            try:
                cur_frame = int(cur.get("frame_idx", -1))
            except Exception:
                cur_frame = -1
            for pos, (f, _idx) in enumerate(frames):
                if f == cur_frame:
                    cur_pos = pos
                    break
        if cur_pos is None:
            return None

        if delta > 0:
            # next within sequence
            if cur_pos + 1 < len(frames):
                return frames[cur_pos + 1][1]
            # move to next sequence
            try:
                seq_i = seq_order.index(key)
            except ValueError:
                return None
            if seq_i + 1 < len(seq_order):
                next_key = seq_order[seq_i + 1]
                next_frames = seq_to_frames.get(next_key, [])
                if next_frames:
                    return next_frames[0][1]
            return None

        if delta < 0:
            # prev within sequence
            if cur_pos - 1 >= 0:
                return frames[cur_pos - 1][1]
            # move to previous sequence
            try:
                seq_i = seq_order.index(key)
            except ValueError:
                return None
            if seq_i - 1 >= 0:
                prev_key = seq_order[seq_i - 1]
                prev_frames = seq_to_frames.get(prev_key, [])
                if prev_frames:
                    return prev_frames[-1][1]
            return None

        return st.session_state.row_idx

    # Search / navigation
    st.sidebar.header("Navigate")
    uid_query = st.sidebar.text_input("Find by uid contains", value="")
    if st.sidebar.button("Find"):
        q = uid_query.strip()
        if q:
            hit = next((i for i, r in enumerate(rows) if q in str(r.get("uid", ""))), None)
            if hit is not None:
                st.session_state.row_idx = hit
            else:
                st.sidebar.warning("No match.")

    row_idx = st.sidebar.slider("Row index", 0, len(rows) - 1, st.session_state.row_idx, 1)
    st.session_state.row_idx = int(row_idx)

    nav_c1, nav_c2 = st.sidebar.columns(2)
    if nav_c1.button("◀ Prev frame"):
        idx2 = _find_row_idx_for_frame(-1)
        if idx2 is not None:
            st.session_state.row_idx = idx2
            st.rerun()
        else:
            st.sidebar.warning("No previous frame (already at beginning of the dataset).")
    if nav_c2.button("Next frame ▶"):
        idx2 = _find_row_idx_for_frame(+1)
        if idx2 is not None:
            st.session_state.row_idx = idx2
            st.rerun()
        else:
            st.sidebar.warning("No next frame (already at end of the dataset).")

    row = rows[st.session_state.row_idx]

    # Layout
    left, right = st.columns([2, 1])

    # LEFT: images + core fields
    with left:
        st.subheader("Images & Metadata")
        images = row.get("images", {})
        if not isinstance(images, dict):
            images = {}

        c1, c2 = st.columns(2)
        with c1:
            st.caption("table")
            im = safe_open_image(str(images.get("table", "")))
            if im is None:
                st.warning("table image not found")
            else:
                st.image(im, use_container_width=True)
        with c2:
            st.caption("wrist")
            im2 = safe_open_image(str(images.get("wrist", "")))
            if im2 is None:
                st.warning("wrist image not found")
            else:
                st.image(im2, use_container_width=True)

        st.write(
            {
                "uid": row.get("uid"),
                "task_id": row.get("task_id"),
                "chunk": row.get("chunk"),
                "episode": row.get("episode"),
                "inter": row.get("inter"),
                "base_intra": row.get("base_intra"),
                "frame_idx": row.get("frame_idx"),
                "event_frame_idx": row.get("event_frame_idx"),
            }
        )

        # Sanity check: intra expectation (optional)
        try:
            frame_idx = int(row.get("frame_idx", 0))
            event_idx = int(row.get("event_frame_idx", 0))
            base_intra = int(row.get("base_intra", 0))
            expected_intra = base_intra if frame_idx < event_idx else base_intra + 1
            st.caption(f"Expected intra by rule: {expected_intra}  (t < event ? base : base+1)")
        except Exception:
            pass

    # RIGHT: conversations, edit assistant
    with right:
        st.subheader("Prompt / Answer")
        user_text, asst_text = extract_user_assistant(row)

        st.markdown("**User**")
        st.text_area("user_text", user_text, height=180, disabled=True)

        st.markdown("**Assistant (YAML)**")
        st.caption("Note: YAML `World_State: null` means Python None. This typically happens when taskspec world_state was set to \"None\" and normalized to null during generation. If you want a literal string, set world_state_grid to a non-None string (e.g., \"total count = 2\") or edit it here.")
        edited = st.text_area("assistant_yaml", asst_text, height=220)

        ok, msg, parsed = validate_assistant_yaml(edited)
        if ok:
            st.success(msg)
        else:
            st.error(msg)

        # Quick structured editor (if YAML available)
        if HAS_YAML and parsed and isinstance(parsed, dict):
            st.markdown("**Structured edit (optional)**")
            p = st.text_input("Progress", value=str(parsed.get("Progress", "")))
            ws = parsed.get("World_State", None)
            ws_str = "" if ws is None else str(ws)
            ws2 = st.text_input("World_State (string, empty->null)", value=ws_str)
            cmd = st.text_input("Command", value=str(parsed.get("Command", "")))

            if st.button("Apply structured edit → YAML"):
                obj = {
                    "Progress": p,
                    "World_State": (None if ws2.strip() == "" else ws2),
                    "Command": cmd,
                }
                new_yaml = yaml.safe_dump(obj, sort_keys=False, allow_unicode=True).strip()  # type: ignore
                edited = new_yaml
                st.session_state["_temp_yaml"] = new_yaml
                st.rerun()

        # Apply update to in-memory row
        if st.button("Update row (in-memory)"):
            update_assistant_in_row(row, edited)
            rows[st.session_state.row_idx] = row
            st.session_state.rows = rows
            st.session_state.dirty = True
            st.info("Updated in memory. Save to write file.")

        # Save/export
        st.divider()
        st.subheader("Save / Export")

        out_suffix = st.text_input("Output suffix", value="_edited")
        if st.button("Export current file as new JSONL"):
            in_file = Path(files[st.session_state.file_idx])
            out_file = in_file.with_name(in_file.stem + out_suffix + in_file.suffix)
            write_jsonl(out_file, rows)
            st.session_state.dirty = False
            st.success(f"Saved: {out_file}")

        if st.session_state.dirty:
            st.warning("Unsaved changes in memory.")


if __name__ == "__main__":
    main()