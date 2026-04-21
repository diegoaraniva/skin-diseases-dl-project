from pathlib import Path
import json
import tempfile
import zipfile

import h5py
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf
import torch
from tensorflow.keras.applications.efficientnet import preprocess_input
from transformers import ViTForImageClassification, ViTImageProcessor
import gdown

GDRIVE_FOLDER_ID = "1ScflbX21RFYYUHOjy7_lOSacWRYRz5uR"
MODEL_FILENAME = "skin_disease_efficientnet_ft5.keras"
METADATA_FILENAME = "model_metadata.json"
LABELS_FILENAME = "class_labels.json"

CACHE_DIR = Path(tempfile.gettempdir()) / "skin_classifier_assets"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def get_asset_path(filename: str) -> Path:
    direct = CACHE_DIR / filename
    if direct.exists():
        return direct
    matches = list(CACHE_DIR.rglob(filename))
    if matches:
        return matches[0]
    return direct


def download_from_gdrive() -> None:
    progress_placeholder = st.empty()
    model_path = get_asset_path(MODEL_FILENAME)
    if model_path.exists():
        progress_placeholder.success("✅ Assets loaded from cache")
        return

    try:
        progress_placeholder.info("📥 Downloading model assets from Google Drive...")
        gdown.download_folder(
            id=GDRIVE_FOLDER_ID,
            output=str(CACHE_DIR),
            quiet=False,
            use_cookies=False,
        )
        progress_placeholder.success("✅ Assets downloaded successfully!")
    except Exception as exc:
        progress_placeholder.error(f"❌ Error downloading from Google Drive: {exc}")
        st.stop()


def load_labels() -> list[str]:
    labels_path = get_asset_path(LABELS_FILENAME)
    if not labels_path.exists():
        return []

    raw = json.loads(labels_path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [str(x) for x in raw]

    if isinstance(raw, dict):
        index_to_label = raw.get("index_to_label", {})
        if isinstance(index_to_label, dict):
            ordered = sorted(index_to_label.items(), key=lambda kv: int(kv[0]))
            return [str(v) for _, v in ordered]

    return []


def load_pytorch_from_interop_bundle(model_path: Path):
    with zipfile.ZipFile(model_path, "r") as zf:
        config = json.loads(zf.read("config.json").decode("utf-8"))
        weights_bytes = zf.read("model.weights.h5")

    base_checkpoint = str(config.get("base_checkpoint", "google/vit-base-patch16-224-in21k"))
    num_labels = int(config.get("num_labels", 2))

    model = ViTForImageClassification.from_pretrained(
        base_checkpoint,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
    )

    state_dict = {}
    tmp_h5_path = CACHE_DIR / "_tmp_interop_model.weights.h5"
    tmp_h5_path.write_bytes(weights_bytes)
    try:
        with h5py.File(tmp_h5_path, "r") as h5f:
            group = h5f["pytorch_state_dict"]
            for safe_name in group.keys():
                original_name = safe_name.replace("__", "/")
                state_dict[original_name] = torch.from_numpy(group[safe_name][()])
    finally:
        if tmp_h5_path.exists():
            tmp_h5_path.unlink(missing_ok=True)

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    processor = ViTImageProcessor.from_pretrained(
        base_checkpoint,
        size={"height": 224, "width": 224},
        do_resize=True,
        do_rescale=True,
        do_normalize=True,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
    )
    return model, processor, "pytorch_interop", (224, 224)


@st.cache_resource
def load_model_and_metadata():
    download_from_gdrive()
    model_path = get_asset_path(MODEL_FILENAME)
    metadata_path = get_asset_path(METADATA_FILENAME)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Ensure {MODEL_FILENAME} is in the Google Drive folder."
        )

    metadata = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    labels = load_labels()

    try:
        keras_model = tf.keras.models.load_model(model_path)
        return {
            "backend": "keras",
            "model": keras_model,
            "processor": None,
            "input_size": (224, 224),
            "metadata": metadata,
            "labels": labels,
        }
    except Exception:
        model, processor, backend, input_size = load_pytorch_from_interop_bundle(model_path)
        if not labels:
            id2label = getattr(model.config, "id2label", {})
            if isinstance(id2label, dict) and id2label:
                ordered = sorted(id2label.items(), key=lambda kv: int(kv[0]))
                labels = [str(v) for _, v in ordered]
        return {
            "backend": backend,
            "model": model,
            "processor": processor,
            "input_size": input_size,
            "metadata": metadata,
            "labels": labels,
        }


def preprocess_image(image: Image.Image, input_size: tuple[int, int], backend: str, processor=None):
    image = image.convert("RGB").resize(input_size)
    if backend == "keras":
        arr = np.array(image, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)
        return preprocess_input(arr)

    encoded = processor(images=image, return_tensors="pt")
    return encoded["pixel_values"]


def predict_top_k(model, backend: str, image_tensor, labels: list[str], top_k: int = 3):
    if backend == "keras":
        probs = model.predict(image_tensor, verbose=0)[0]
    else:
        with torch.no_grad():
            logits = model(pixel_values=image_tensor).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    top_idx = np.argsort(probs)[::-1][:top_k]
    results = []
    for idx in top_idx:
        label = labels[idx] if idx < len(labels) else f"class_{idx}"
        results.append({"label": label, "probability": float(probs[idx])})
    return results


def main():
    st.set_page_config(page_title="Skin Disease Classifier", layout="centered")
    st.title("Skin Disease Classification Demo")
    st.caption("Demo de inferencia para artefactos de despliegue (Etapa 5)")

    with st.expander("ℹ️ Asset Information"):
        st.write(f"**Source:** Google Drive Folder")
        st.write(f"**Folder ID:** `{GDRIVE_FOLDER_ID}`")
        st.write(f"**Cache Directory:** `{CACHE_DIR}`")

    try:
        bundle = load_model_and_metadata()
    except Exception as exc:
        st.error(f"❌ Error loading deployment assets: {exc}")
        st.stop()

    backend = bundle["backend"]
    model = bundle["model"]
    processor = bundle["processor"]
    metadata = bundle["metadata"]
    labels = bundle["labels"]
    input_size = bundle["input_size"]

    st.write(f"**Backend cargado:** {backend}")

    top_k = st.slider("Top-K predictions", min_value=1, max_value=5, value=3)
    uploaded_file = st.file_uploader("Upload a skin image", type=["jpg", "jpeg", "png", "webp", "bmp"])

    if uploaded_file is None:
        st.info("📤 Upload an image to run inference.")
        return

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("🔮 Predict", use_container_width=True):
        with st.spinner("Running inference..."):
            image_tensor = preprocess_image(image, input_size, backend, processor)
            predictions = predict_top_k(model, backend, image_tensor, labels, top_k=top_k)

        st.subheader("Predictions")
        for i, item in enumerate(predictions, start=1):
            st.write(f"{i}. **{item['label']}** - {item['probability']:.4f}")

        st.bar_chart({item["label"]: item["probability"] for item in predictions})

    if metadata:
        with st.expander("Model Metadata"):
            st.json(metadata)


if __name__ == "__main__":
    main()