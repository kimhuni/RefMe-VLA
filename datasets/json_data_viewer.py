# viewer_streamlit.py
import json, glob, argparse, time, random
from pathlib import Path
import streamlit as st
import pandas as pd

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
                    })
                except Exception:
                    pass
    return pd.DataFrame(rows)

def show_record(r):
    st.markdown(f"### {r['uid']} — **{r['status']}**")
    c1, c2 = st.columns(2)
    with c1:
        st.image(r["side"], caption=f"side | {r['side']}", use_container_width=True)
    with c2:
        st.image(r["wrist"], caption=f"wrist | {r['wrist']}", use_container_width=True)
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

    shards_glob = str(args.derived_root  / "shards" / "*.jsonl")

    st.set_page_config(page_title="VLM-B Viewer", layout="wide")
    st.title("VLM-B Evaluation Viewer (side + wrist)")

    # 좌측 필터
    df = load_shards(shards_glob)
    if df.empty:
        st.warning(f"No records found at: {shards_glob}")
        return

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