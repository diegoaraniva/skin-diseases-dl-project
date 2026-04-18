from pathlib import Path
import json
import os
import tempfile

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import gdown


# Google Drive folder ID containing deployment assets
GDRIVE_FOLDER_ID = "1ScflbX21RFYYUHOjy7_lOSacWRYRz5uR"

# Expected file names in the Google Drive folder
MODEL_FILENAME = "skin_disease_efficientnet_ft5.keras"
METADATA_FILENAME = "model_metadata.json"
LABELS_FILENAME = "class_labels.json"

# Local cache directory for downloaded assets
CACHE_DIR = Path(tempfile.gettempdir()) / "skin_classifier_assets"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = CACHE_DIR / MODEL_FILENAME
METADATA_PATH = CACHE_DIR / METADATA_FILENAME
LABELS_PATH = CACHE_DIR / LABELS_FILENAME


def download_from_gdrive():
    """Download deployment assets from Google Drive folder if not already cached."""
    progress_placeholder = st.empty()
    
    try:
        if not MODEL_PATH.exists():
            progress_placeholder.info(f"📥 Downloading model from Google Drive...")
            # Download entire folder to cache directory
            gdown.download_folder(
                id=GDRIVE_FOLDER_ID,
                output=str(CACHE_DIR),
                quiet=False,
                use_cookies=False
            )
            progress_placeholder.success("✅ Assets downloaded successfully!")
        else:
            progress_placeholder.success("✅ Assets loaded from cache")
    except Exception as e:
        progress_placeholder.error(f"❌ Error downloading from Google Drive: {str(e)}")
        st.stop()


@st.cache_resource
def load_model_and_metadata():
    """Load model and metadata from local cache or download from Google Drive."""
    # Ensure assets are downloaded
    download_from_gdrive()
    
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            f"Ensure {MODEL_FILENAME} is in the Google Drive folder."
        )

    metadata = {}
    if METADATA_PATH.exists():
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    labels = []
    if LABELS_PATH.exists():
        labels = json.loads(LABELS_PATH.read_text(encoding="utf-8"))

    model = tf.keras.models.load_model(MODEL_PATH)
    return model, metadata, labels


def preprocess_image(image: Image.Image, input_size: tuple[int, int]) -> np.ndarray:
    """Preprocess image for model inference."""
    image = image.convert("RGB").resize(input_size)
    arr = np.array(image, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    return preprocess_input(arr)


def predict_top_k(model, image_tensor: np.ndarray, labels: list[str], top_k: int = 3):
    """Get top-K predictions from model."""
    probs = model.predict(image_tensor, verbose=0)[0]
    top_idx = np.argsort(probs)[::-1][:top_k]

    results = []
    for idx in top_idx:
        label = labels[idx] if idx < len(labels) else f"class_{idx}"
        results.append(
            {
                "label": label,
                "probability": float(probs[idx]),
            }
        )
    return results


def main():
    st.set_page_config(page_title="Skin Disease Classifier", layout="centered")
    st.title("Skin Disease Classification Demo")
    st.caption("Modelo EfficientNetB0 fine-tuned (Etapa 5)")

    # Display info about asset source
    with st.expander("ℹ️ Asset Information"):
        st.write(f"**Source:** Google Drive Folder")
        st.write(f"**Folder ID:** `{GDRIVE_FOLDER_ID}`")
        st.write(f"**Cache Directory:** `{CACHE_DIR}`")

    try:
        model, metadata, labels = load_model_and_metadata()
    except Exception as exc:
        st.error(f"❌ Error loading deployment assets: {exc}")
        st.stop()

    input_size_meta = metadata.get("input_size", [128, 128])
    if isinstance(input_size_meta, list) and len(input_size_meta) == 2:
        input_size = (int(input_size_meta[0]), int(input_size_meta[1]))
    else:
        input_size = (128, 128)

    top_k = st.slider("Top-K predictions", min_value=1, max_value=5, value=3)
    uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png", "webp", "bmp"])

    if uploaded_file is None:
        st.info("📤 Upload an image to run inference.")
        return

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("🔮 Predict", use_container_width=True):
        with st.spinner("Running inference..."):
            image_tensor = preprocess_image(image, input_size)
            predictions = predict_top_k(model, image_tensor, labels, top_k=top_k)

        st.subheader("Predictions")
        for i, item in enumerate(predictions, start=1):
            st.write(f"{i}. **{item['label']}** - {item['probability']:.4f}")

        st.bar_chart({item["label"]: item["probability"] for item in predictions})


if __name__ == "__main__":
    main()