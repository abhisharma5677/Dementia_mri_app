import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from PIL import Image
from utils.model_utils import load_model, predict

# Load model once
model_path = "model/resnet50_dementia.pth"
model = load_model(model_path)

st.title("🧠 Brain MRI Dementia Classifier")
st.markdown("Upload a brain MRI image to predict the dementia stage.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)

    if st.button("Predict"):
        label = predict(image, model)
        st.success(f"🧠 Prediction: **{label}**")


