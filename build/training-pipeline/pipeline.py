from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


DEFAULT_CROISSANT_URL = (
    "https://www.kaggle.com/datasets/anieetorudofia/"
    "skin-diseases-cancer-comprehensive-dataset/croissant/download"
)
DEFAULT_KAGGLE_DATASET_SLUG = "anieetorudofia/skin-diseases-cancer-comprehensive-dataset"


@dataclass
class TrainConfig:
    dataset_dir: str
    output_dir: str
    use_kagglehub_dataset: bool = True
    kaggle_dataset_slug: str = DEFAULT_KAGGLE_DATASET_SLUG
    image_size: int = 128
    batch_size: int = 32
    random_seed: int = 42
    val_size: float = 0.15
    test_size: float = 0.15
    head_epochs: int = 3
    ft_epochs: int = 3
    head_lr: float = 1e-3
    ft_lr: float = 1e-5
    unfreeze_last: int = 40
    dropout_rate: float = 0.35


ARTIFACT_MODEL_NAME = "skin_disease_efficientnet_ft5.keras"
ARTIFACT_LABELS_NAME = "class_labels.json"
ARTIFACT_METADATA_NAME = "model_metadata.json"


class DatasetError(ValueError):
    pass


def _resolve_dataset_dir(config: TrainConfig) -> Path:
    dataset_dir = Path(config.dataset_dir).expanduser() if config.dataset_dir else None
    if dataset_dir and dataset_dir.exists() and not config.use_kagglehub_dataset:
        return dataset_dir

    if config.use_kagglehub_dataset:
        try:
            import kagglehub
        except ImportError as e:  # pragma: no cover
            raise DatasetError(
                "Falta la dependencia 'kagglehub'. Instala requirements.txt y vuelve a intentar."
            ) from e

        downloaded_path = Path(kagglehub.dataset_download(config.kaggle_dataset_slug))
        if not downloaded_path.exists():
            raise DatasetError(
                "No se pudo resolver la ruta descargada del dataset por kagglehub."
            )
        return downloaded_path

    raise DatasetError(
        "Debes indicar una ruta válida de dataset o activar el modo automático por kagglehub."
    )


def _list_image_files(dataset_dir: Path) -> Tuple[List[str], List[int], List[str]]:
    class_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
    if not class_dirs:
        raise DatasetError(
            "No se encontraron carpetas de clases en el dataset. "
            "Usa una estructura tipo: dataset/<clase>/<imagenes>."
        )

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    file_paths: List[str] = []
    labels: List[int] = []
    class_names = [c.name for c in class_dirs]

    for class_idx, class_dir in enumerate(class_dirs):
        class_files = [f for f in class_dir.rglob("*") if f.is_file() and f.suffix.lower() in image_exts]
        if not class_files:
            continue
        file_paths.extend(str(f) for f in class_files)
        labels.extend([class_idx] * len(class_files))

    if len(file_paths) < 100:
        raise DatasetError(
            "Se detectaron muy pocas imágenes para entrenar (<100). "
            "Verifica la ruta y la estructura del dataset."
        )

    return file_paths, labels, class_names


def _build_tf_dataset(
    paths: np.ndarray,
    labels: np.ndarray,
    image_size: int,
    batch_size: int,
    training: bool,
) -> tf.data.Dataset:
    autotune = tf.data.AUTOTUNE

    path_ds = tf.data.Dataset.from_tensor_slices(paths)
    label_ds = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((path_ds, label_ds))

    if training:
        ds = ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)

    def _load(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image = tf.image.resize(image, (image_size, image_size))
        image = tf.cast(image, tf.float32)
        image = tf.keras.applications.efficientnet.preprocess_input(image)
        return image, label

    ds = ds.map(_load, num_parallel_calls=autotune)
    ds = ds.batch(batch_size).prefetch(autotune)
    return ds


def train_and_export(config: TrainConfig) -> Dict[str, object]:
    dataset_dir = _resolve_dataset_dir(config)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_paths, labels, class_names = _list_image_files(dataset_dir)
    x = np.array(file_paths)
    y = np.array(labels)

    x_temp, x_test, y_temp, y_test = train_test_split(
        x,
        y,
        test_size=config.test_size,
        random_state=config.random_seed,
        stratify=y,
    )

    val_relative = config.val_size / (1.0 - config.test_size)
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp,
        y_temp,
        test_size=val_relative,
        random_state=config.random_seed,
        stratify=y_temp,
    )

    train_ds = _build_tf_dataset(x_train, y_train, config.image_size, config.batch_size, training=True)
    val_ds = _build_tf_dataset(x_val, y_val, config.image_size, config.batch_size, training=False)
    test_ds = _build_tf_dataset(x_test, y_test, config.image_size, config.batch_size, training=False)

    base_model = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(config.image_size, config.image_size, 3),
    )
    base_model.trainable = False

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(config.image_size, config.image_size, 3)),
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(config.dropout_rate),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(len(class_names), activation="softmax"),
        ]
    )

    early_stop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.head_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.head_epochs,
        callbacks=[early_stop],
        verbose=1,
    )

    base_model.trainable = True
    if config.unfreeze_last < len(base_model.layers):
        for layer in base_model.layers[:-config.unfreeze_last]:
            layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.ft_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    history_ft = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.ft_epochs,
        callbacks=[early_stop],
        verbose=1,
    )

    probs = model.predict(test_ds, verbose=0)
    pred = np.argmax(probs, axis=1)

    acc = float(accuracy_score(y_test, pred))
    f1 = float(f1_score(y_test, pred, average="macro"))

    model_path = output_dir / ARTIFACT_MODEL_NAME
    labels_path = output_dir / ARTIFACT_LABELS_NAME
    metadata_path = output_dir / ARTIFACT_METADATA_NAME

    model.save(model_path)
    labels_path.write_text(json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8")

    metadata = {
        "model_name": "EfficientNetB0 final fine-tuned",
        "input_size": [config.image_size, config.image_size],
        "class_count": int(len(class_names)),
        "labels_file": labels_path.name,
        "preprocess": "tf.keras.applications.efficientnet.preprocess_input",
        "normalization": "ImageNet style preprocessing",
        "selected_backbone": "EfficientNetB0",
        "note": "Pipeline trigger desde Streamlit con guardado y subida automática a Google Drive.",
        "metrics": {
            "test_accuracy": acc,
            "test_f1_macro": f1,
            "head_epochs_run": len(history_head.history.get("loss", [])),
            "ft_epochs_run": len(history_ft.history.get("loss", [])),
        },
        "dataset": {
            "path": str(dataset_dir.resolve()),
            "source": "kagglehub" if config.use_kagglehub_dataset else "manual_path",
            "kaggle_dataset_slug": config.kaggle_dataset_slug if config.use_kagglehub_dataset else "",
            "croissant_url": DEFAULT_CROISSANT_URL,
            "train_samples": int(len(x_train)),
            "val_samples": int(len(x_val)),
            "test_samples": int(len(x_test)),
            "created_at": datetime.utcnow().isoformat() + "Z",
        },
    }

    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "model_path": str(model_path),
        "labels_path": str(labels_path),
        "metadata_path": str(metadata_path),
        "metrics": {"test_accuracy": acc, "test_f1_macro": f1},
        "class_count": len(class_names),
    }
