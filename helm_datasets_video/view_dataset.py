import streamlit as st
import json
import os
from PIL import Image
from glob import glob

"""
streamlit run view_dataset.py
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

    # 메타데이터 표시
    st.markdown("---")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.info(f"**UID:** {sample.get('uid', 'N/A')}")
    with c2:
        mode = sample.get('mode', 'UNKNOWN')
        if mode == 'DETECT':
            st.success(f"**Mode:** {mode}")
        else:
            st.warning(f"**Mode:** {mode}")
    with c3:
        st.text(f"Task ID: {sample.get('meta', {}).get('task', 'N/A')}")

    # ---------------------------------------------------------
    # 3. 이미지 시각화 (핵심 기능)
    # ---------------------------------------------------------
    st.subheader("🖼️ Visual Context (Input Images)")

    image_paths = sample.get('images', [])
    if not isinstance(image_paths, list):
        image_paths = [image_paths]  # 단일 경로일 경우 리스트로 변환

    if image_paths:
        # 이미지를 한 줄에 표시 (개수가 많으면 여러 줄로)
        cols = st.columns(len(image_paths))
        for idx, (col, img_path) in enumerate(zip(cols, image_paths)):
            with col:
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    st.image(img, use_container_width=True)

                    # 캡션 달기
                    if mode == "UPDATE":
                        if idx == 0:
                            caption = "Start Frame (Context)"
                        else:
                            caption = f"Event History #{idx}"
                    else:
                        caption = "Current Observation"

                    st.caption(f"**[{idx}]** {caption}\n`...{img_path[-30:]}`")
                else:
                    st.error(f"Image not found:\n{img_path}")
    else:
        st.warning("No images found in this sample.")

    # ---------------------------------------------------------
    # 4. 프롬프트 및 정답 확인
    # ---------------------------------------------------------
    st.markdown("---")
    st.subheader("📝 Text Data")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("### User Prompt (Input)")
        prompt_content = sample.get('user_prompt', '')
        st.text_area("Full Prompt", value=prompt_content, height=400, disabled=True)

    with col_r:
        st.markdown("### Ground Truth (Output)")
        gt_text = sample.get('gt_text', '')

        # YAML 파싱 시도 (가독성을 위해)
        try:
            import yaml
            gt_json = yaml.safe_load(gt_text)
            st.json(gt_json)
        except:
            st.code(gt_text, language='yaml')

    # ---------------------------------------------------------
    # 5. Raw Data (디버깅용)
    # ---------------------------------------------------------
    with st.expander("🔍 View Raw JSON Sample"):
        st.json(sample)


if __name__ == "__main__":
    main()