import streamlit as st
from PIL import Image
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.model_utils import load_model, predict

st.set_page_config(page_title="Dementia MRI Classifier", layout="centered")

st.title("🧠 Dementia Detection from MRI")
st.markdown("Upload a brain MRI image to predict the dementia stage.")

uploaded_file = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Classifying..."):
        model = load_model()
        prediction = predict(image, model)

    st.success(f"🧾 Prediction: **{prediction}**")
