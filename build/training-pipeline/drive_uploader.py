from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


SCOPES = ["https://www.googleapis.com/auth/drive.file"]
TOKEN_FILENAME = "gdrive_token.json"


def _require_google_deps() -> tuple:
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Faltan dependencias de Google Drive. Instala google-auth, google-auth-oauthlib y google-api-python-client."
        ) from e

    return Request, Credentials, InstalledAppFlow, build, MediaFileUpload


def start_oauth_flow(client_secret_file: str) -> Dict[str, str]:
    _, _, InstalledAppFlow, _, _ = _require_google_deps()
    flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, SCOPES)
    auth_url, _ = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent",
    )

    state_payload = {
        "client_secret_file": str(Path(client_secret_file).resolve()),
        "state": flow.oauth2session.state,
    }
    return {"auth_url": auth_url, "state_payload": json.dumps(state_payload)}


def finish_oauth_flow(client_secret_file: str, auth_code: str, state: str, token_output_dir: str) -> str:
    _, _, InstalledAppFlow, _, _ = _require_google_deps()
    flow = InstalledAppFlow.from_client_secrets_file(client_secret_file, SCOPES, state=state)
    flow.fetch_token(code=auth_code)

    token_dir = Path(token_output_dir)
    token_dir.mkdir(parents=True, exist_ok=True)
    token_path = token_dir / TOKEN_FILENAME
    token_path.write_text(flow.credentials.to_json(), encoding="utf-8")
    return str(token_path)


def _load_credentials(token_file: str) -> Credentials:
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
