# app.py
import streamlit as st
import os
import gdown
import zipfile
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# =======================
# 1️⃣ PAGE CONFIG
# =======================
st.set_page_config(
    page_title="Football Club Logo Recognition",
    layout="centered"
)

st.title("⚽ Football Club Logo Recognition")
st.write("Upload a football club logo and the model will predict the club name.")

# =======================
# 2️⃣ PATHS & LINKS
# =======================
MODEL_PATH = "football_logo_model.h5"
DATASET_DIR = "dataset"
TRAIN_DIR = os.path.join(DATASET_DIR, "train")

# 🔴 REPLACE THESE WITH YOUR OWN GOOGLE DRIVE FILE IDs
MODEL_URL = "https://drive.google.com/uc?id=1YGpJpyRA_lA0wTITKCbw3m6Fclfu7dYh"
DATASET_URL = "https://drive.google.com/uc?id=1Yne5NKOzRKFNqDkFwFgjVq39BOPEaHTQ"
DATASET_ZIP = "dataset.zip"

# =======================
# 3️⃣ DOWNLOAD DATASET
# =======================
def download_dataset():
    if not os.path.exists(TRAIN_DIR):
        st.info("Downloading dataset...")
        gdown.download(DATASET_URL, DATASET_ZIP, quiet=False)

        with zipfile.ZipFile(DATASET_ZIP, "r") as zip_ref:
            zip_ref.extractall(DATASET_DIR)

        os.remove(DATASET_ZIP)
        st.success("Dataset downloaded successfully")

# =======================
# 4️⃣ LOAD MODEL + CLASSES
# =======================
@st.cache_resource(show_spinner=True)
def load_model_and_classes():
    # Download dataset if missing
    download_dataset()

    # Download model if missing
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading trained model...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    model = load_model(MODEL_PATH)

    # Read class names from dataset folders
    class_names = sorted([
        d for d in os.listdir(TRAIN_DIR)
        if os.path.isdir(os.path.join(TRAIN_DIR, d))
    ])

    model.class_names = class_names
    return model

model = load_model_and_classes()

# =======================
# 5️⃣ IMAGE UPLOAD
# =======================
uploaded_file = st.file_uploader(
    "Upload a football club logo image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB").resize((224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    if st.button("Predict Club"):
        with st.spinner("Predicting..."):
            prediction = model.predict(img_array)
            index = np.argmax(prediction)
            club = model.class_names[index]
            confidence = prediction[0][index] * 100

        st.success(f"🏆 Predicted Club: **{club}**")
        st.info(f"Confidence: **{confidence:.2f}%**")
