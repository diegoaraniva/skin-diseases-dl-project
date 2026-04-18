from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlencode

import streamlit as st

from drive_uploader import finish_oauth_flow, start_oauth_flow, upload_files_to_drive
from pipeline import (
    DEFAULT_CROISSANT_URL,
    DEFAULT_KAGGLE_DATASET_SLUG,
    DatasetError,
    TrainConfig,
    train_and_export,
)


DEFAULT_DRIVE_FOLDER_ID = "1ScflbX21RFYYUHOjy7_lOSacWRYRz5uR"
DEFAULT_ARTIFACTS_DIR = "artifacts"
DEFAULT_REDIRECT_URI = "https://skin-diseases-dl-project-vtgprpplmoyca9fud8yreq.streamlit.app/"


st.set_page_config(page_title="Skin Diseases Training Trigger", page_icon="🧪", layout="wide")
st.title("Skin Diseases - Training Pipeline Trigger")
st.caption("Entrena EfficientNetB0 (transfer learning), guarda artefactos y los sube automáticamente a Google Drive.")

if "oauth_state" not in st.session_state:
    st.session_state.oauth_state = ""
if "oauth_client_secret" not in st.session_state:
    st.session_state.oauth_client_secret = ""
if "token_file" not in st.session_state:
    st.session_state.token_file = ""
if "latest_artifacts" not in st.session_state:
    st.session_state.latest_artifacts = []
if "oauth_code_verifier" not in st.session_state:
    st.session_state.oauth_code_verifier = ""

query_params = st.query_params
oauth_code_from_url = query_params.get("code", "")
oauth_state_from_url = query_params.get("state", "")

with st.sidebar:
    st.header("Configuración")
    use_notebook_dataset = st.checkbox(
        "Usar dataset del notebook (mlcroissant + kagglehub)",
        value=True,
    )
    kaggle_dataset_slug = st.text_input(
        "Kaggle dataset slug",
        value=DEFAULT_KAGGLE_DATASET_SLUG,
    )
    st.caption(f"Croissant URL: {DEFAULT_CROISSANT_URL}")
    dataset_dir = st.text_input(
        "Ruta dataset (carpetas por clase)",
        value="",
        placeholder=r"C:\ruta\dataset\train",
        disabled=use_notebook_dataset,
    )
    output_dir = st.text_input("Carpeta de artefactos", value=DEFAULT_ARTIFACTS_DIR)
    drive_folder_id = st.text_input("Google Drive Folder ID", value=DEFAULT_DRIVE_FOLDER_ID)

st.subheader("1) Trigger de entrenamiento")
col1, col2, col3 = st.columns(3)
with col1:
    image_size = st.number_input("Image size", min_value=96, max_value=512, value=128, step=16)
    batch_size = st.number_input("Batch size", min_value=8, max_value=128, value=32, step=8)
with col2:
    head_epochs = st.number_input("Head epochs", min_value=1, max_value=50, value=3)
    ft_epochs = st.number_input("Fine-tuning epochs", min_value=1, max_value=50, value=3)
with col3:
    head_lr = st.number_input("Head learning rate", min_value=1e-6, max_value=1e-1, value=1e-3, format="%.6f")
    ft_lr = st.number_input("Fine-tuning learning rate", min_value=1e-7, max_value=1e-2, value=1e-5, format="%.7f")

if st.button("Iniciar entrenamiento", type="primary"):
    if (not use_notebook_dataset) and (not dataset_dir):
        st.error("Debes indicar la ruta del dataset.")
    else:
        cfg = TrainConfig(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            use_kagglehub_dataset=bool(use_notebook_dataset),
            kaggle_dataset_slug=kaggle_dataset_slug.strip() or DEFAULT_KAGGLE_DATASET_SLUG,
            image_size=int(image_size),
            batch_size=int(batch_size),
            head_epochs=int(head_epochs),
            ft_epochs=int(ft_epochs),
            head_lr=float(head_lr),
            ft_lr=float(ft_lr),
        )

        try:
            with st.spinner("Entrenando modelo y exportando artefactos..."):
                result = train_and_export(cfg)

            st.success("Entrenamiento finalizado.")
            st.json(result)
            st.session_state.latest_artifacts = [
                result["model_path"],
                result["labels_path"],
                result["metadata_path"],
            ]
        except DatasetError as e:
            st.error(f"Dataset inválido: {e}")
        except Exception as e:  # pragma: no cover
            st.exception(e)

st.divider()
st.subheader("2) Login OAuth en Google Drive")
st.write("Sube aquí tu archivo OAuth `client_secret.json` descargado desde Google Cloud Console.")
st.caption(f"Redirect URI actual: {DEFAULT_REDIRECT_URI} (puedes cambiarlo con GOOGLE_OAUTH_REDIRECT_URI)")

client_secret_upload = st.file_uploader("Archivo client_secret.json", type=["json"])

if client_secret_upload is not None:
    oauth_dir = Path(output_dir)
    oauth_dir.mkdir(parents=True, exist_ok=True)
    client_secret_path = oauth_dir / "client_secret.json"
    client_secret_path.write_bytes(client_secret_upload.getvalue())
    st.session_state.oauth_client_secret = str(client_secret_path)
    st.success(f"Archivo OAuth guardado en {client_secret_path}")

col_a, col_b = st.columns([1, 2])
with col_a:
    if st.button("Generar enlace de login"):
        if not st.session_state.oauth_client_secret:
            st.error("Primero sube el archivo client_secret.json")
        else:
            try:
                flow_data = start_oauth_flow(st.session_state.oauth_client_secret)
                st.session_state.oauth_state = flow_data["state"]
                st.session_state.oauth_code_verifier = flow_data.get("code_verifier", "")
                st.success("Abre el enlace y autoriza la cuenta Google.")
                st.markdown(f"[Abrir login Google]({flow_data['auth_url']})")
            except Exception as e:  # pragma: no cover
                st.exception(e)

with col_b:
    auth_code = st.text_input("Pega aquí el authorization code")
    if st.button("Guardar token OAuth"):
        if not st.session_state.oauth_client_secret:
            st.error("Falta client_secret.json")
        elif not st.session_state.oauth_state:
            st.error("Primero genera el enlace de login.")
        elif not auth_code:
            st.error("Debes pegar el authorization code.")
        else:
            try:
                token_file = finish_oauth_flow(
                    client_secret_file=st.session_state.oauth_client_secret,
                    auth_code=auth_code.strip(),
                    state=st.session_state.oauth_state,
                    token_output_dir=output_dir,
                    code_verifier=st.session_state.oauth_code_verifier,
                )
                st.session_state.token_file = token_file
                st.success(f"Token guardado en {token_file}")
            except Exception as e:  # pragma: no cover
                st.exception(e)

if oauth_code_from_url and oauth_state_from_url:
    if st.session_state.oauth_client_secret and not st.session_state.token_file:
        try:
            authorization_response = (
                f"{DEFAULT_REDIRECT_URI}?"
                f"{urlencode({'code': oauth_code_from_url, 'state': oauth_state_from_url})}"
            )
            token_file = finish_oauth_flow(
                client_secret_file=st.session_state.oauth_client_secret,
                auth_code=oauth_code_from_url,
                state=oauth_state_from_url,
                token_output_dir=output_dir,
                code_verifier=st.session_state.oauth_code_verifier,
                authorization_response=authorization_response,
            )
            st.session_state.token_file = token_file
            st.success("Token OAuth creado automáticamente desde la URL de redirección.")
        except Exception as e:  # pragma: no cover
            st.exception(e)

st.divider()
st.subheader("3) Subida automática de artefactos a Google Drive")

if st.session_state.latest_artifacts:
    st.write("Artefactos detectados de la última ejecución:")
    for p in st.session_state.latest_artifacts:
        st.write(f"- {p}")

upload_files = st.text_area(
    "Lista de archivos a subir (uno por línea). Si dejas vacío, usa los artefactos más recientes.",
    value="",
    height=120,
)

if st.button("Subir a Google Drive"):
    if not drive_folder_id:
        st.error("Debes indicar el folder ID de Google Drive.")
    elif not st.session_state.token_file:
        st.error("Debes completar el login OAuth primero.")
    else:
        files = [line.strip() for line in upload_files.splitlines() if line.strip()]
        if not files:
            files = st.session_state.latest_artifacts

        if not files:
            st.error("No hay archivos para subir. Ejecuta primero el entrenamiento o ingresa archivos manualmente.")
        else:
            try:
                with st.spinner("Subiendo archivos a Google Drive..."):
                    uploaded = upload_files_to_drive(
                        token_file=st.session_state.token_file,
                        drive_folder_id=drive_folder_id.strip(),
                        files=files,
                    )

                st.success("Subida completada.")
                st.json(uploaded)
            except Exception as e:  # pragma: no cover
                st.exception(e)
