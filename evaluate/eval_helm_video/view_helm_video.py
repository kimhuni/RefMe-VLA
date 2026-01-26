import streamlit as st
import json
import os
import pandas as pd
from PIL import Image
import yaml

"""
Run command:
streamlit run evaluate/eval_helm_video/view_helm_video.py
"""

# 페이지 설정
st.set_page_config(layout="wide", page_title="HeLM Video Evaluation Viewer")


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
@st.cache_data
def load_data(file_path):
    data = []
    if not os.path.exists(file_path):
        return []

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except:
                    pass
    return data


def get_image_paths(row_images):
    """
    Handle both Dict (Standard HeLM) and List (Video HeLM) image formats
    """
    if isinstance(row_images, list):
        return row_images
    elif isinstance(row_images, dict):
        if "table" in row_images:
            return [row_images["table"]]
        return list(row_images.values())
    elif isinstance(row_images, str):
        return [row_images]
    return []


def safe_yaml_parse(text):
    if text is None: return {}
    try:
        # Remove markdown code blocks if present
        clean_text = text.replace("```yaml", "").replace("```", "").strip()
        return yaml.safe_load(clean_text)
    except:
        return text


# -----------------------------------------------------------------------------
# Main Layout
# -----------------------------------------------------------------------------
def main():
    st.title("🎬 HeLM Video Evaluation Viewer")

    # Sidebar: File Selection
    st.sidebar.header("📂 Data Source")

    # 기본 경로 (환경에 맞게 수정 가능)
    default_path = "/data/ghkim/helm_data/helm_video_task_3/eval_results/video_2k_preds.jsonl"
    file_path = st.sidebar.text_input("Result JSONL Path", value=default_path)

    if not os.path.exists(file_path):
        st.error(f"File not found: {file_path}")
        return

    # Load Data
    raw_data = load_data(file_path)
    if not raw_data:
        st.warning("Data is empty or failed to load.")
        return

    df = pd.DataFrame(raw_data)

    # -------------------------------------------------------------------------
    # 1. Statistics Dashboard
    # -------------------------------------------------------------------------
    st.markdown("### 📊 Summary Statistics")

    # Overall Accuracy
    if 'correct' in df.columns:
        total_acc = df['correct'].mean() * 100
        # Mode Accuracy
        if 'mode' in df.columns:
            mode_acc = df.groupby('mode')['correct'].mean() * 100
        else:
            mode_acc = {}
    else:
        total_acc = 0.0
        mode_acc = {}
        df['correct'] = False  # Fallback

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Accuracy", f"{total_acc:.2f}%", f"Count: {len(df)}")

    if 'DETECT' in mode_acc:
        c2.metric("DETECT Acc", f"{mode_acc['DETECT']:.2f}%")
    if 'UPDATE' in mode_acc:
        c3.metric("UPDATE Acc", f"{mode_acc['UPDATE']:.2f}%")

    st.markdown("---")

    # -------------------------------------------------------------------------
    # 2. Filters & Navigation
    # -------------------------------------------------------------------------
    st.sidebar.header("🔍 Filters")

    # Filter by Mode
    if 'mode' in df.columns:
        all_modes = ["ALL"] + sorted(df['mode'].dropna().unique().tolist())
        selected_mode = st.sidebar.selectbox("Filter by Mode", all_modes)
    else:
        selected_mode = "ALL"

    # Filter by Result (Correct/Incorrect)
    filter_status = st.sidebar.radio("Filter by Result", ["All", "Incorrect Only", "Correct Only"])

    # Apply Filters
    filtered_df = df.copy()
    if selected_mode != "ALL":
        filtered_df = filtered_df[filtered_df['mode'] == selected_mode]

    if filter_status == "Incorrect Only":
        filtered_df = filtered_df[filtered_df['correct'] == False]
    elif filter_status == "Correct Only":
        filtered_df = filtered_df[filtered_df['correct'] == True]

    # Show Count
    st.sidebar.markdown(f"**Filtered Samples:** {len(filtered_df)} / {len(df)}")

    if len(filtered_df) == 0:
        st.warning("No samples match the filters.")
        return

    # Sample Navigation
    # Get index mapping to original list
    filtered_indices = filtered_df.index.tolist()

    # Create a selector
    selected_idx_loc = st.sidebar.number_input(
        "Sample Index (in filtered list)",
        min_value=0,
        max_value=len(filtered_df) - 1,
        value=0,
        step=1
    )

    # Retrieve the actual row
    current_idx = filtered_indices[selected_idx_loc]
    sample = raw_data[current_idx]

    # -------------------------------------------------------------------------
    # 3. Sample Visualization
    # -------------------------------------------------------------------------
    uid = sample.get('uid', 'N/A')
    st.subheader(f"Sample View (UID: `{uid}`)")

    # Status Badge
    is_correct = sample.get('correct', False)
    status_color = "green" if is_correct else "red"
    status_text = "✅ CORRECT" if is_correct else "❌ INCORRECT"

    mode_str = sample.get('mode', 'N/A')
    label_str = sample.get('label', 'N/A')

    st.markdown(f":{status_color}[**{status_text}**] | Mode: **{mode_str}** | Label: **{label_str}**")

    # A. Input Images (Visual History)
    st.markdown("#### 🖼️ Visual Input (History)")

    # 1. 'images' 키 확인 (우리가 eval 스크립트에 추가한 부분)
    image_source = sample.get('images', [])

    # 2. 없으면 'row' 내부 확인 (Fallback)
    if not image_source and 'row' in sample:
        image_source = sample['row'].get('images', [])

    image_paths = get_image_paths(image_source)

    if image_paths:
        st.caption(f"Found {len(image_paths)} images")  # 이미지 개수 표시 (Done 디버깅용)

        # Display images
        cols = st.columns(min(len(image_paths), 5))
        for i, img_path in enumerate(image_paths):
            col = cols[i % 5]
            with col:
                if os.path.exists(img_path):
                    img = Image.open(img_path)
                    st.image(img, use_container_width=True, caption=f"Frame {i}")
                else:
                    st.error(f"Missing: {os.path.basename(img_path)}")
    else:
        st.warning("No images found in this sample.")

    # B. Text Comparison (GT vs Pred)
    st.markdown("#### 📝 Prediction Comparison")

    col_l, col_r = st.columns(2)

    gt_text = sample.get('gt_text', "")
    pred_text = sample.get('pred_text', "")

    with col_l:
        st.info("Ground Truth (Target)")
        gt_parsed = safe_yaml_parse(gt_text)
        if isinstance(gt_parsed, dict):
            st.json(gt_parsed)
        else:
            st.text(gt_text)

    with col_r:
        if is_correct:
            st.success("Prediction (Model)")
        else:
            st.error("Prediction (Model)")

        pred_parsed = safe_yaml_parse(pred_text)
        if isinstance(pred_parsed, dict):
            st.json(pred_parsed)
        else:
            st.text(pred_text)

    # C. Full Prompt Debug
    with st.expander("🔍 View Full Input Prompt"):
        st.text_area("User Prompt", sample.get('user_prompt', 'N/A'), height=300)

    # D. Raw Data Debug
    with st.expander("🔍 View Raw JSON"):
        st.json(sample)


if __name__ == "__main__":
    main()