# viewer_streamlit.py

import json, glob, argparse, time, random
from pathlib import Path
import streamlit as st
import pandas as pd
import re

# --- robust parsing helpers for malformed JSON strings ---
def _clean_json_like(s: str) -> str:
    # normalize common issues: control chars, smart quotes, repeated commas, stray trailing commas
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # remove most control chars except tab/newline
    s = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]", " ", s)
    # unify smart quotes
    s = s.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'")
    # collapse repeated commas
    s = re.sub(r",\s*,+", ", ", s)
    # remove comma before closing brace/bracket
    s = re.sub(r",\s*([}\]])", r" \1", s)
    # collapse excessive whitespace
    s = re.sub(r"[ \t\f\v]+", " ", s)
    return s.strip()

def parse_model_output_any(raw: str) -> dict:
    """
    Try strict JSON first; if it fails, fall back to regex-based extraction for keys:
    desc_1, desc_2, status. Returns a dict with missing keys filled as empty strings.
    """
    cleaned = _clean_json_like(raw)
    # 1) strict JSON
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # 2) fallback: regex extraction with DOTALL and lookahead to next key or end
    out = {"desc_1": "", "desc_2": "", "status": ""}
    # desc_1: up to ,"desc_2" or ,"status" or }
    m1 = re.search(r'"desc_1"\s*:\s*"(.*?)"(?=,\s*"(?:desc_2|status)"|\s*})', cleaned, flags=re.DOTALL)
    if m1:
        out["desc_1"] = m1.group(1)
    # desc_2: up to ,"status" or }
    m2 = re.search(r'"desc_2"\s*:\s*"(.*?)"(?=,\s*"(?:status)"|\s*})', cleaned, flags=re.DOTALL)
    if m2:
        out["desc_2"] = m2.group(1)
    # status: simple capture
    m3 = re.search(r'"status"\s*:\s*"(.*?)"', cleaned, flags=re.DOTALL)
    if m3:
        out["status"] = m3.group(1)

    # final normalization: remove remaining newlines and stray spaces inside values
    for k in ("desc_1", "desc_2", "status"):
        v = out.get(k, "")
        if isinstance(v, str):
            v = v.replace("\n", " ").replace("\r", " ")
            v = re.sub(r"\s+", " ", v).strip()
            # escape any remaining unescaped quotes to make it JSON-safe if needed downstream
            v = v.replace('\\"', '"')  # un-double-escape first
            v = v.replace('"', '\\"')
            # but keep in-memory values unescaped for display
            v = v.replace('\\"', '"')
            out[k] = v
    return out


def load_shards(shards_glob):
    rows = []
    for fp in glob.glob(shards_glob):
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    o = json.loads(line)
                    rows.append({
                        "uid": o.get("uid"),
                        "task": o.get("task"),
                        "chunk_id": o.get("chunk_id"),
                        "episode_id": o.get("episode_id"),
                        "ts": o.get("timestamp_ms"),
                        "side": o.get("images", {}).get("side", ""),
                        "wrist": o.get("images", {}).get("wrist", ""),
                        "prev_desc": o.get("prev_desc", ""),
                        "prev_status": o.get("prev_status", ""),
                        "desc_1": o.get("api_output", {}).get("desc_1", ""),
                        "desc_2": o.get("api_output", {}).get("desc_2", ""),
                        "status": o.get("api_output", {}).get("status", ""),
                        "model_output_raw": o.get("model_output_raw", ""),
                    })
                except Exception:
                    pass
    return pd.DataFrame(rows)

def show_record(r):
    raw_str = r.get("model_output_raw")
    debug_raw = st.session_state.get("debug_raw")
    # show raw first if requested
    if debug_raw and isinstance(raw_str, str) and raw_str.strip():
        st.caption("raw model_output_raw")
        st.code(raw_str[:500], language="json")
        try:
            cleaned_preview = _clean_json_like(raw_str)[:500]
            st.caption("cleaned (preview)")
            st.code(cleaned_preview, language="json")
        except Exception:
            pass

    # --- handle model_output_raw field (robust) ---
    if isinstance(raw_str, str) and raw_str.strip():
        try:
            model_output = parse_model_output_any(raw_str)
            r["desc_1"] = model_output.get("desc_1", r.get("desc_1",""))
            r["desc_2"] = model_output.get("desc_2", r.get("desc_2",""))
            r["status"] = model_output.get("status", r.get("status",""))
        except Exception as e:
            st.warning(f"⚠️ Failed to parse model_output_raw for {r.get('uid')}: {e}")
            if not r.get("desc_1"):
                r["desc_1"] = f"[PARSE_ERROR] {raw_str[:200]}"
            if not r.get("status"):
                r["status"] = "ERROR"

    st.markdown(f"### {r['uid']} — **{r['status']}**")
    c1, c2 = st.columns(2)
    side_path = r.get("side", "")
    wrist_path = r.get("wrist", "")
    with c1:
        if isinstance(side_path, str) and len(side_path) > 0:
            try:
                st.image(side_path, caption=f"side | {side_path}", use_container_width=True)
            except Exception:
                st.info("No side image.")
        else:
            st.info("No side image.")
    with c2:
        if isinstance(wrist_path, str) and len(wrist_path) > 0:
            try:
                st.image(wrist_path, caption=f"wrist | {wrist_path}", use_container_width=True)
            except Exception:
                st.info("No wrist image.")
        else:
            st.info("No wrist image.")
    st.markdown(f"**prev_desc**: {r['prev_desc']} \n **prev_status**: {r['prev_status']}")
    st.markdown(f"**desc_1**: {r['desc_1']}  \n**desc_2**: {r['desc_2']}")
    status = r['status']
    color = "gray"
    if status == "DONE":
        color = "green"
    elif status == "NOT_DONE":
        color = "red"
    elif status == "UNCERTAIN":
        color = "gray"
    st.markdown(f"<h1 style='text-align:center; color:{color};'>{status}</h1>", unsafe_allow_html=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--derived_root", required=True, type=Path,
                        help="/data/..._derived (api_eval_B 아래 shards/*.jsonl 읽음)")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()

    shards_glob = str(args.derived_root / "shards" / "*.jsonl")

    st.session_state["debug_raw"] = st.sidebar.checkbox("debug: show raw model_output_raw", value=False)

    matched = sorted(glob.glob(shards_glob))
    st.sidebar.write(f"📄 matched files: {len(matched)}")
    if len(matched) == 0:
        st.warning(f"No records found. Glob pattern matched nothing:\n{shards_glob}")
    if st.session_state.get("debug_raw", False):
        for p in matched[:20]:
            st.sidebar.caption(p)

    df = load_shards(shards_glob)
    if df.empty:
        st.warning(f"No records found at: {shards_glob}")
        return

    st.set_page_config(page_title="VLM-B Viewer", layout="wide")
    st.title("VLM-B Evaluation Viewer (side + wrist)")

    # 좌측 필터

    col_f1, col_f2, col_f3, col_f4 = st.sidebar.columns(4)
    with col_f1:
        statuses = st.multiselect("status", sorted(df["status"].dropna().unique().tolist()),
                                  default=sorted(df["status"].dropna().unique().tolist()))
    with col_f2:
        chunks = st.multiselect("chunk", sorted(df["chunk_id"].dropna().unique().tolist()),
                                default=sorted(df["chunk_id"].dropna().unique().tolist()))
    with col_f3:
        episodes = st.multiselect("episode", sorted(df["episode_id"].dropna().unique().tolist()),
                                  default=[])
    with col_f4:
        task = st.selectbox("task", ["(any)"] + sorted(df["task"].dropna().unique().tolist()), index=0)

    q = df[df["status"].isin(statuses) & df["chunk_id"].isin(chunks)]
    if episodes:
        q = q[q["episode_id"].isin(episodes)]
    if task != "(any)":
        q = q[q["task"] == task]

    # 정렬 및 페이지네이션
    q = q.sort_values(["chunk_id", "episode_id", "ts"]).reset_index(drop=True)

    with st.sidebar.expander("failures", expanded=False):
        fail_path = args.derived_root / "failures.jsonl"
        if fail_path.exists():
            fails = []
            with open(fail_path, "r", encoding="utf-8") as ff:
                for line in ff:
                    try:
                        fails.append(json.loads(line))
                    except Exception:
                        pass
            st.write(f"count={len(fails)}")
            max_show = st.number_input("show first N", 0, min(500, len(fails)), min(100, len(fails)), step=10)
            for f in fails[:max_show]:
                st.write(f)

    # ----- view mode -----
    mode = st.sidebar.radio("view mode", ["list", "step-through"], index=1)

    if mode == "list":
        # simple paginated list
        page_size = st.sidebar.number_input("page size", 5, 200, 20, step=5)
        page = st.sidebar.number_input("page idx", 0, max(0, (len(q)-1)//page_size), 0, step=1)
        view = q.iloc[page*page_size : (page+1)*page_size]

        with st.expander("table (filtered)", expanded=False):
            st.dataframe(view[["uid","chunk_id","episode_id","ts","status","desc_1","desc_2","prev_desc", "prev_status"]], use_container_width=True)

        for _, r in view.iterrows():
            show_record(r)

    else:
        # step-through (button-triggered navigation)
        if "idx" not in st.session_state:
            st.session_state.idx = 0

        total = len(q)
        st.sidebar.write(f"records: {total}")
        colb1, colb2, colb3 = st.sidebar.columns([1,1,1])
        with colb1:
            if st.button(" ⟵ ", use_container_width=True):
                st.session_state.idx = (st.session_state.idx - 1) % max(1, total)
        with colb2:
            if st.button(" ⟶ ", use_container_width=True):
                st.session_state.idx = (st.session_state.idx + 1) % max(1, total)
        with colb3:
            autoplay = st.checkbox("auto", value=False)

        # optional auto-advance every N seconds
        interval = st.sidebar.number_input("auto interval (s)", 0.5, 10.0, 1.0, step=0.5)

        if total == 0:
            st.warning("No records after filters.")
            return

        cur = q.iloc[st.session_state.idx]
        show_record(cur)

        if autoplay:
            time.sleep(float(interval))
            st.session_state.idx = (st.session_state.idx + 1) % total
            st.experimental_rerun()

if __name__ == "__main__":
    main()