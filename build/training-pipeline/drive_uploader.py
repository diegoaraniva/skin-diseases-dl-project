from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


SCOPES = ["https://www.googleapis.com/auth/drive.file"]
TOKEN_FILENAME = "gdrive_token.json"
OAUTH_STATE_PREFIX = "gdrive_oauth_state"
DEFAULT_REDIRECT_URI = os.environ.get(
    "GOOGLE_OAUTH_REDIRECT_URI",
    "https://skin-diseases-dl-project-vtgprpplmoyca9fud8yreq.streamlit.app/",
)


def _require_google_deps() -> tuple:
    try:
        Request = importlib.import_module("google.auth.transport.requests").Request
        Credentials = importlib.import_module("google.oauth2.credentials").Credentials
        InstalledAppFlow = importlib.import_module("google_auth_oauthlib.flow").InstalledAppFlow
        build = importlib.import_module("googleapiclient.discovery").build
        MediaFileUpload = importlib.import_module("googleapiclient.http").MediaFileUpload
    except ImportError as e:  # pragma: no cover
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "google-auth",
                "google-auth-oauthlib",
                "google-api-python-client",
            ],
            check=True,
        )
        try:
            Request = importlib.import_module("google.auth.transport.requests").Request
            Credentials = importlib.import_module("google.oauth2.credentials").Credentials
            InstalledAppFlow = importlib.import_module("google_auth_oauthlib.flow").InstalledAppFlow
            build = importlib.import_module("googleapiclient.discovery").build
            MediaFileUpload = importlib.import_module("googleapiclient.http").MediaFileUpload
        except ImportError as retry_error:  # pragma: no cover
            raise ImportError(
                "Faltan dependencias de Google Drive y no pudieron instalarse automáticamente."
            ) from retry_error

    return Request, Credentials, InstalledAppFlow, build, MediaFileUpload


def _oauth_state_path(token_output_dir: str, state: str) -> Path:
    token_dir = Path(token_output_dir)
    token_dir.mkdir(parents=True, exist_ok=True)
    return token_dir / f"{OAUTH_STATE_PREFIX}_{state}.json"


def _save_oauth_state(token_output_dir: str, state: str, code_verifier: str) -> str:
    state_path = _oauth_state_path(token_output_dir, state)
    state_path.write_text(
        json.dumps(
            {
                "state": state,
                "code_verifier": code_verifier,
                "redirect_uri": DEFAULT_REDIRECT_URI,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return str(state_path)


def _load_oauth_state(token_output_dir: str, state: str) -> Dict[str, str]:
    state_path = _oauth_state_path(token_output_dir, state)
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))


def start_oauth_flow(client_secret_file: str, token_output_dir: str) -> Dict[str, str]:
    _, _, InstalledAppFlow, _, _ = _require_google_deps()
    flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, SCOPES)
    flow.redirect_uri = DEFAULT_REDIRECT_URI
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )

    return {
        "auth_url": auth_url,
        "state": state,
        "code_verifier": getattr(flow, "code_verifier", ""),
        "redirect_uri": DEFAULT_REDIRECT_URI,
        "client_secret_file": str(Path(client_secret_file).resolve()),
        "oauth_state_file": _save_oauth_state(token_output_dir, state, getattr(flow, "code_verifier", "")),
    }


def finish_oauth_flow(
    client_secret_file: str,
    auth_code: str,
    state: str,
    token_output_dir: str,
    code_verifier: str = "",
    authorization_response: str = "",
) -> str:
    _, _, InstalledAppFlow, _, _ = _require_google_deps()
    flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, SCOPES, state=state)
    flow.redirect_uri = DEFAULT_REDIRECT_URI
    saved_state = _load_oauth_state(token_output_dir, state)
    if not code_verifier:
        code_verifier = saved_state.get("code_verifier", "")
    if code_verifier:
        flow.code_verifier = code_verifier
    if authorization_response:
        flow.fetch_token(authorization_response=authorization_response, code_verifier=code_verifier or None)
    else:
        flow.fetch_token(code=auth_code, code_verifier=code_verifier or None)

    token_dir = Path(token_output_dir)
    token_dir.mkdir(parents=True, exist_ok=True)
    token_path = token_dir / TOKEN_FILENAME
    token_path.write_text(flow.credentials.to_json(), encoding="utf-8")

    state_path = _oauth_state_path(token_output_dir, state)
    if state_path.exists():
        state_path.unlink()

    return str(token_path)


def _load_credentials(token_file: str):
    Request, Credentials, _, _, _ = _require_google_deps()
    token_path = Path(token_file)
    if not token_path.exists():
        raise FileNotFoundError(
            f"No existe el token OAuth en: {token_file}. Primero autentica en la app Streamlit."
        )

    creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        token_path.write_text(creds.to_json(), encoding="utf-8")
    return creds


def upload_files_to_drive(token_file: str, drive_folder_id: str, files: List[str]) -> List[Dict[str, str]]:
    _, _, _, build, MediaFileUpload = _require_google_deps()
    creds = _load_credentials(token_file)
    service = build("drive", "v3", credentials=creds)

    uploaded: List[Dict[str, str]] = []
    for file_path in files:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"No existe el archivo a subir: {file_path}")

        media = MediaFileUpload(str(path), resumable=True)
        metadata = {"name": path.name, "parents": [drive_folder_id]}

        created = (
            service.files()
            .create(body=metadata, media_body=media, fields="id, name, webViewLink")
            .execute()
        )

        uploaded.append(
            {
                "id": created.get("id", ""),
                "name": created.get("name", path.name),
                "webViewLink": created.get("webViewLink", ""),
            }
        )

    return uploaded
