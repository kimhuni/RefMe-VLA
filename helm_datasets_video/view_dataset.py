import streamlit as st
import json
import os
from PIL import Image
from glob import glob

"""
streamlit run helm_datasets_video/view_dataset.py
"""

# 페이지 설정 (와이드 모드)
st.set_page_config(layout="wide", page_title="HeLM Visual Memory Viewer")

def load_jsonl(file_path):
    """JSONL 파일을 읽어서 리스트로 반환"""
    data = []
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def main():
    st.title("🧠 HeLM Visual Memory Dataset Viewer")

    # ---------------------------------------------------------
    # 1. 사이드바: 데이터셋 파일 선택
    # ---------------------------------------------------------
    st.sidebar.header("📁 Dataset Selection")

    # 기본 경로 설정 (사용자 환경에 맞게 수정 가능)
    default_root = "/data/ghkim/helm_data/visual_memory_baseline/visual_memory_jsonl"
    data_root = st.sidebar.text_input("Dataset Root Directory", value=default_root)

    if os.path.exists(data_root):
        files = sorted(glob(os.path.join(data_root, "*.jsonl")))
        files = [os.path.basename(f) for f in files]
    else:
        st.sidebar.error(f"Path not found: {data_root}")
        files = []

    if not files:
        st.warning("No .jsonl files found.")
        return

    selected_file = st.sidebar.selectbox("Select JSONL File", files)
    file_path = os.path.join(data_root, selected_file)

    # 데이터 로드
    data = load_jsonl(file_path)
    total_samples = len(data)
    st.sidebar.markdown(f"**Total Samples:** `{total_samples}`")

    if total_samples == 0:
        st.error("File is empty.")
        return

    # ---------------------------------------------------------
    # 2. 샘플 탐색 (인덱스 선택)
    # ---------------------------------------------------------
    index = st.sidebar.number_input("Sample Index", min_value=0, max_value=total_samples - 1, value=0, step=1)

    # 현재 샘플 가져오기
    sample = data[index]


    # ---------------------------------------------------------
    # 3~4. 이미지 + 텍스트를 한 화면에 (왼쪽: 이미지, 오른쪽: 텍스트)
    # ---------------------------------------------------------
    st.markdown("---")

    # 왼쪽 이미지 패널이 차지할 비율 선택 (1/2 또는 1/3)
    st.sidebar.header("🧩 Layout")
    img_panel_ratio = st.sidebar.selectbox(
        "Image Panel Width",
        options=["1/2", "1/3"],
        index=0,
        help="이미지 패널이 화면에서 차지하는 비율을 선택합니다.",
    )

    if img_panel_ratio == "1/3":
        left_col, right_col = st.columns([1, 2])  # 왼쪽 1/3, 오른쪽 2/3
    else:
        left_col, right_col = st.columns([1, 1])  # 왼쪽 1/2, 오른쪽 1/2

    # ---- 왼쪽: 이미지 (여러 장이면 위아래로 쌓기) ----
    with left_col:
        st.subheader("🖼️ Visual Context")

        image_paths = sample.get('images', [])
        if not isinstance(image_paths, list):
            image_paths = [image_paths]  # 단일 경로일 경우 리스트로 변환

        if image_paths:
            # 이미지 크기(폭) 조절: 컬럼 폭에 맞추되 너무 크지 않게 옵션 제공
            img_width = st.sidebar.slider(
                "Image Width (px)",
                min_value=120,
                max_value=480,
                value=260,
                step=10,
                help="이미지를 더 작게 보고 싶으면 값을 줄이세요.",
            )

            for idx, img_path in enumerate(image_paths):
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    st.image(img, width=img_width)

                    caption = "Current Observation"

                    st.caption(f"**[{idx}]** {caption}\n`...{img_path[-30:]}`")
                    st.markdown("---")
                else:
                    st.error(f"Image not found:\n{img_path}")
        else:
            st.warning("No images found in this sample.")

    # ---- 오른쪽: 텍스트 (Prompt / GT / Raw) ----
    with right_col:
        # ---- 메타데이터 (UID / Mode / Task) ----
        st.markdown("### 🧾 Metadata")
        m1, m2, m3 = st.columns([1, 1, 2])
        with m1:
            st.info(f"**UID:** {sample.get('uid', 'N/A')}")
        with m2:
            mode = sample.get('mode', 'UNKNOWN')
            if mode == 'DETECT':
                st.success(f"**Mode:** {mode}")
            else:
                st.warning(f"**Mode:** {mode}")
        with m3:
            st.text(f"Task ID: {sample.get('meta', {}).get('task', 'N/A')}")

        st.markdown("---")
        st.subheader("📝 Text Data")

        tab_prompt, tab_gt, tab_raw = st.tabs(["User Prompt", "Ground Truth", "Raw JSON"])

        with tab_prompt:
            prompt_content = sample.get('user_prompt', '')
            st.text_area("Full Prompt", value=prompt_content, height=520, disabled=True)

        with tab_gt:
            gt_text = sample.get('gt_text', '')
            # YAML 파싱 시도 (가독성을 위해)
            try:
                import yaml
                gt_json = yaml.safe_load(gt_text)
                st.json(gt_json)
            except Exception:
                st.code(gt_text, language='yaml')

        with tab_raw:
            st.json(sample)


if __name__ == "__main__":
    main()