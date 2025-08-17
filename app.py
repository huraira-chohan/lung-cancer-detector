import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing import image
import os

# =========================
# 1. Load the trained model
# =========================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("lung_cancer_model.h5")
    return model

model = load_model()

# Class names (update according to your dataset)
CLASS_NAMES = ["lung_aca", "lung_scc", "lung_n"]

# =========================
# 2. Image Preprocessing
# =========================
def preprocess(img):
    img = img.resize((224, 224))   # resize to model input size
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img

# =========================
# 3. Streamlit UI
# =========================
st.title("🫁 Lung Cancer Detection (EfficientNetB0)")
st.write("Upload a histopathology image, and the model will predict the cancer type.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    from PIL import Image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    st.write("🔍 Classifying...")
    processed_img = preprocess(img)

    # Prediction
    preds = model.predict(processed_img)
    pred_class = CLASS_NAMES[np.argmax(preds)]
    confidence = np.max(preds)

    st.write(f"### ✅ Prediction: {pred_class}")
    st.write(f"🔢 Confidence: {confidence:.2f}")
