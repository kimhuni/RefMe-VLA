# datasets/view_model_outputs.py
# Streamlit viewer for malformed model_output_raw JSON strings + GT comparison
# Usage:
#   streamlit run datasets/view_model_outputs.py -- --path /data/.../shards/*.jsonl
#   (파일 하나만도 OK: --path /data/.../chunk_000_evaluation.jsonl)
"""
streamlit run datasets/json_data_view_HLP.py -- --path "/data/ghkim/piper_press_the_blue_button_ep60/eval_qwen_step_1k/shards/*.jsonl"
"""

import json, glob, argparse, re
from pathlib import Path
import streamlit as st
import pandas as pd

# ---------- Robust parsing helpers ----------
def _clean_json_like(s: str) -> str:
    """정규 JSON 파싱 전에 흔한 오염 패턴을 정리."""
    if not isinstance(s, str):
        return ""

    # 줄바꿈 통일 & 제어문자 제거(탭/개행 제외)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F]", " ", s)

    # 스마트 따옴표 보정
    s = s.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'")

    # 흔한 잡음 토큰 정리: W + 개행 + addCriterion
    s = re.sub(r'\bW\s*\n\s*addCriterion', ' addCriterion', s)

    # addCriterion("TABLE") 같은 내부 따옴표는 값 문자열을 깨뜨리므로 제거
    s = re.sub(r'addCriterion\(\s*"TABLE"\s*\)', r'addCriterion(TABLE)', s, flags=re.IGNORECASE)
    s = re.sub(r'addCriterion\(\s*"WRIST"\s*\)', r'addCriterion(WRIST)', s, flags=re.IGNORECASE)

    # 값 내부에 주입된 중첩 JSON 블록 시작을 최대한 무력화
    #   예:  "desc_2": " ... \n{"desc_1": " ...  → 중첩 시작 앞에서 끊어 salvage
    s = re.sub(r'(\n\s*\{\s*"desc_(?:1|2)"\s*:)', ' ', s)

    # 중복 콤마 및 닫는 괄호 앞 콤마 제거
    s = re.sub(r",\s*,+", ", ", s)
    s = re.sub(r",\s*([}\]])", r" \1", s)

    # 과도 공백 축약
    s = re.sub(r"[ \t\f\v]+", " ", s)
    return s.strip()


def parse_model_output_any(raw: str) -> dict:
    """
    model_output_raw 문자열을 최대한 복구/파싱.
    우선 json.loads 시도, 실패 시 키별(grab) 추출로 fallback.
    반환: {"desc_1": str, "desc_2": str, "status": str}
    """
    out = {"desc_1": "", "desc_2": "", "status": ""}
    if not isinstance(raw, str) or not raw.strip():
        return out

    cleaned = _clean_json_like(raw)

    # 1) 정규 JSON 시도
    try:
        obj = json.loads(cleaned)
        # 정상 JSON이면 키만 뽑아 리턴
        for k in out.keys():
            if isinstance(obj.get(k), str):
                out[k] = obj[k]
        return out
    except Exception:
        pass

    # 2) Fallback: 키별 안전 추출기
    def grab(text: str, key: str, next_keys):
        """
        "key": " ... "  값을 추출하되, 값 내부의 따옴표/개행을 용인하고
        다음 키(next_keys)나 } 직전까지 스캔.
        """
        m = re.search(rf'"{re.escape(key)}"\s*:\s*"', text)
        if not m:
            return ""
        i = m.end()  # opening quote 뒤
        j = i
        # 종료 패턴: ", "next_key"  또는  " }
        end_pat = rf'"\s*,\s*"(?:{"|".join(map(re.escape, next_keys))})"|"\s*\}}'
        nested_pat = r'\n\s*\{\s*"desc_(?:1|2)"\s*:'

        # 선형 스캔 (문자열 내 따옴표는 허용, 단 종료 패턴 앞의 따옴표로 종결)
        while j < len(text):
            # 중첩 JSON 시작이면 여기서 값 종료
            if re.match(nested_pat, text[j:]):
                return text[i:j]
            m2 = re.search(end_pat, text[j:])
            if m2:
                end_idx = j + m2.start()
                # 종료 패턴 바로 앞쪽의 마지막 따옴표까지를 값으로
                kpos = end_idx - 1
                while kpos >= i and text[kpos] != '"':
                    kpos -= 1
                if kpos >= i:
                    return text[i:kpos]
            j += 1
        return text[i:]

    out["desc_1"] = grab(cleaned, "desc_1", ["desc_2", "status"])
    out["desc_2"] = grab(cleaned, "desc_2", ["status"])

    # status는 대개 깔끔함. 그래도 DOTALL로 한 번 더
    m3 = re.search(r'"status"\s*:\s*"(.*?)"', cleaned, flags=re.DOTALL)
    if m3:
        out["status"] = m3.group(1)

    # 최종 정리: 개행/중복 공백 제거
    for k in list(out.keys()):
        v = out[k]
        if isinstance(v, str):
            v = v.replace("\r", " ").replace("\n", " ")
            v = re.sub(r"\s+", " ", v).strip()
            out[k] = v

    return out


# ---------- IO ----------
def load_rows_from_path(path_pattern: str) -> pd.DataFrame:
    paths = sorted(glob.glob(path_pattern))
    rows = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                rows.append({
                    "uid": o.get("uid", ""),
                    "model_output_raw": o.get("model_output_raw", ""),
                    "gt_desc_1": o.get("gt_output", {}).get("desc_1", ""),
                    "gt_desc_2": o.get("gt_output", {}).get("desc_2", ""),
                    "gt_status": o.get("gt_output", {}).get("status", ""),
                })
    return pd.DataFrame(rows)


# ---------- UI ----------
def render_card(r: pd.Series, show_raw: bool, show_clean_preview: bool):
    uid = r.get("uid", "")
    raw = r.get("model_output_raw", "")

    st.markdown(f"### {uid}")

    col1, col2 = st.columns(2)

    # 왼쪽: Model Output (raw + parsed)
    with col1:
        st.subheader("Model Output")
        if show_raw and isinstance(raw, str) and raw.strip():
            st.caption("raw (first 500 chars)")
            st.code(raw[:500], language="json")
        if show_clean_preview and isinstance(raw, str) and raw.strip():
            st.caption("cleaned preview (first 500 chars)")
            st.code(_clean_json_like(raw)[:500], language="json")

        parsed = parse_model_output_any(raw)
        st.markdown(f"**desc_1**: {parsed.get('desc_1','')}")
        st.markdown(f"**desc_2**: {parsed.get('desc_2','')}")
        status = parsed.get("status", "")
        color = {"DONE":"green", "NOT_DONE":"red", "UNCERTAIN":"gray"}.get(status, "gray")
        st.markdown(f"<h3 style='color:{color};'>status: {status or '(empty)'} </h3>", unsafe_allow_html=True)

    # 오른쪽: Ground Truth
    with col2:
        st.subheader("Ground Truth")
        st.markdown(f"**desc_1**: {r.get('gt_desc_1','')}")
        st.markdown(f"**desc_2**: {r.get('gt_desc_2','')}")
        gt_status = r.get("gt_status", "")
        gt_color = {"DONE":"green", "NOT_DONE":"red", "UNCERTAIN":"gray"}.get(gt_status, "gray")
        st.markdown(f"<h3 style='color:{gt_color};'>status: {gt_status or '(empty)'} </h3>", unsafe_allow_html=True)


# --- Pair card: one sample per page, parsed vs GT side-by-side, raw output below both columns
def render_pair_card(r: pd.Series, show_raw: bool, show_clean_preview: bool):
    uid = r.get("uid", "")
    raw = r.get("model_output_raw", "")
    st.markdown(f"### {uid}")

    # --- top: parsed vs GT side-by-side ---
    left, right = st.columns(2)

    with left:
        st.subheader("Model Output (parsed)")
        parsed = parse_model_output_any(raw)
        st.markdown(f"**desc_1**: {parsed.get('desc_1','')}")
        st.markdown(f"**desc_2**: {parsed.get('desc_2','')}")
        status = parsed.get("status", "")
        color = {"DONE":"green", "NOT_DONE":"red", "UNCERTAIN":"gray"}.get(status, "gray")
        st.markdown(f"<h3 style='color:{color};'>status: {status or '(empty)'} </h3>", unsafe_allow_html=True)

    with right:
        st.subheader("Ground Truth")
        st.markdown(f"**desc_1**: {r.get('gt_desc_1','')}")
        st.markdown(f"**desc_2**: {r.get('gt_desc_2','')}")
        gt_status = r.get("gt_status", "")
        gt_color = {"DONE":"green", "NOT_DONE":"red", "UNCERTAIN":"gray"}.get(gt_status, "gray")
        st.markdown(f"<h3 style='color:{gt_color};'>status: {gt_status or '(empty)'} </h3>", unsafe_allow_html=True)

    # --- bottom: raw / cleaned preview ---
    if show_raw and isinstance(raw, str) and raw.strip():
        st.markdown("---")
        st.subheader("Raw Model Output")
        st.caption("raw (first 1000 chars)")
        st.code(raw[:1000], language="json")
    if show_clean_preview and isinstance(raw, str) and raw.strip():
        st.caption("cleaned preview (first 1000 chars)")
        st.code(_clean_json_like(raw)[:1000], language="json")


def main():
    st.set_page_config(page_title="Model Output Viewer", layout="wide")
    st.title("VLM Model Output (raw & parsed) vs Ground Truth")

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True,
                        help="JSONL 파일 또는 glob 패턴. 예) /data/.../shards/*.jsonl")
    args = parser.parse_args()

    st.sidebar.markdown("### Options")
    show_raw = st.sidebar.checkbox("Show raw model_output_raw", value=True)
    show_clean_preview = st.sidebar.checkbox("Show cleaned preview", value=True)

    view_mode = st.sidebar.radio("View mode", options=["Grid (multi per page)", "Pair (1 per page)"], index=0)

    # 파일 매칭 안내
    matched = sorted(glob.glob(args.path))
    st.sidebar.write(f"matched files: {len(matched)}")
    for p in matched[:30]:
        st.sidebar.caption(p)

    df = load_rows_from_path(args.path)
    if df.empty:
        st.warning(f"No records found for: {args.path}")
        return

    # 필터 & 페이지
    st.sidebar.markdown("---")
    st.sidebar.write(f"records: {len(df)}")

    if view_mode.startswith("Pair"):
        # force page_size = 1 for pair mode
        page_size = 1
        page = st.sidebar.number_input("page idx", 0, max(0, len(df)-1), 0, step=1)
        view = df.iloc[page:page+1].reset_index(drop=True)

        with st.expander("table (current)", expanded=False):
            st.dataframe(view[["uid","gt_status"]], use_container_width=True)

        # render exactly one
        row = view.iloc[0]
        render_pair_card(row, show_raw=show_raw, show_clean_preview=show_clean_preview)
        st.markdown("---")
    else:
        # grid mode (multi per page)
        page_size = st.sidebar.number_input("page size", 5, 200, 20, step=5)
        page = st.sidebar.number_input("page idx", 0, max(0, (len(df)-1)//page_size), 0, step=1)
        view = df.iloc[page*page_size : (page+1)*page_size].reset_index(drop=True)

        with st.expander("table (filtered)", expanded=False):
            st.dataframe(view[["uid","gt_status"]], use_container_width=True)

        for _, row in view.iterrows():
            render_card(row, show_raw=show_raw, show_clean_preview=show_clean_preview)
            st.markdown("---")


if __name__ == "__main__":
    main()