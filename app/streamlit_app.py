import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import os
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="Mechanical Defect Detection", layout="centered")
st.title("Mechanical Part Crack Detection")

MODEL_PATH = "models/mechanical_defect_model.h5"

# Optionally download model from HF Hub if not present
HF_REPO = st.secrets.get("HF_REPO", "")  # set in Streamlit secrets or leave empty
HF_FILENAME = st.secrets.get("HF_FILE", "models/mechanical_defect_model.h5")
if not os.path.exists(MODEL_PATH) and HF_REPO:
    token = st.secrets.get("HF_TOKEN", None)
    if token:
        path = hf_hub_download(HF_REPO, HF_FILENAME, use_auth_token=token)
        os.makedirs("models", exist_ok=True)
        os.replace(path, MODEL_PATH)

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.warning("Model not found locally. Upload model file models/mechanical_defect_model.h5 or set HF_REPO secrets.")
        return None
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

uploaded = st.file_uploader("Upload a mechanical part image (jpg/png)", type=['jpg','jpeg','png'])
if uploaded and model is not None:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption="Uploaded image", use_column_width=True)
    img_arr = np.array(img.resize((80,80))).astype('float32')/255.0
    x = np.expand_dims(img_arr, 0)
    pred = model.predict(x)[0][0]
    label = "Defective (crack)" if pred >= 0.5 else "Non-defective"
    conf = pred if pred>=0.5 else 1-pred
    st.markdown(f"**Prediction:** {label}  \n**Confidence:** {conf*100:.1f}%")
