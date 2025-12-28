# helm_datasets/streamlit_viewer.py
import json
from pathlib import Path

import streamlit as st
from PIL import Image

"""
streamlit run evaluate/eval_helm/helm_viewer.py
"""

# -----------------------------
# Utils
# -----------------------------
def load_jsonl(path: Path):
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def yaml_block(d):
    if d is None:
        return "❌ YAML parse failed"
    return "\n".join([f"{k}: {v}" for k, v in d.items()])


# -----------------------------
# Streamlit App
# -----------------------------
st.set_page_config(layout="wide")
st.title("🧠 HeLM HLP Evaluation Viewer")

# ---- Sidebar ----
st.sidebar.header("Dataset")

jsonl_path = st.sidebar.text_input(
    "Path to eval jsonl",
    value="/data/ghkim/helm_data/press_the_button_N_times_ep60/jsonl/merged/press_1_2_3/all_val_pred.jsonl",
)

only_cmd_mismatch = st.sidebar.checkbox("Show only Command mismatch", value=False)

# ---- Load ----
if not Path(jsonl_path).exists():
    st.error("JSONL file not found")
    st.stop()

data = load_jsonl(Path(jsonl_path))

if only_cmd_mismatch:
    data = [d for d in data if d.get("match_cmd") is False]

if len(data) == 0:
    st.warning("No samples to display")
    st.stop()

# ---- Index Control ----
if "idx" not in st.session_state:
    st.session_state.idx = 0

col_prev, col_next = st.columns([1, 1])
with col_prev:
    if st.button("⬅ Prev"):
        st.session_state.idx = max(0, st.session_state.idx - 1)
with col_next:
    if st.button("Next ➡"):
        st.session_state.idx = min(len(data) - 1, st.session_state.idx + 1)

idx = st.session_state.idx
sample = data[idx]

st.markdown(f"### Sample {idx + 1} / {len(data)}")
st.code(sample.get("uid", ""), language="text")

# -----------------------------
# Images
# -----------------------------
# img_col1, img_col2 = st.columns(2)
#
# with img_col1:
#     st.subheader("📷 Table")
#     st.image(Image.open(sample["images"]["table"]), use_container_width=True)

# with img_col2:
#     st.subheader("🤖 Wrist")
#     st.image(Image.open(sample["images"]["wrist"]), use_container_width=True)

# -----------------------------
# YAML Comparison
# -----------------------------
st.markdown("## 🔍 YAML Comparison")

col_gt, col_pred = st.columns(2)

with col_gt:
    st.subheader("✅ Ground Truth")
    st.code(yaml_block(sample.get("gt_yaml")), language="yaml")

with col_pred:
    st.subheader("🤖 Prediction")
    st.code(yaml_block(sample.get("pred_yaml")), language="yaml")

# -----------------------------
# Match Indicators
# -----------------------------
st.markdown("## 📊 Match Status")

mc = sample.get("match_cmd")
mp = sample.get("match_progress")

col1, col2 = st.columns(2)
with col1:
    st.metric(
        "Command Match",
        "✅ MATCH" if mc else "❌ MISMATCH",
    )
with col2:
    st.metric(
        "Progress Match",
        "✅ MATCH" if mp else "❌ MISMATCH",
    )

# -----------------------------
# Prompt (Optional)
# -----------------------------
with st.expander("📝 User Prompt"):
    st.code(sample.get("user_prompt", ""), language="text")

with st.expander("🧾 Raw Text Output"):
    st.markdown("**GT Text**")
    st.code(sample.get("gt_text", ""), language="yaml")
    st.markdown("**Pred Text**")
    st.code(sample.get("pred_text", ""), language="yaml")