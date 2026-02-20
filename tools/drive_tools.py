from __future__ import annotations

from pathlib import Path

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

from auth.google_auth_manager import get_credentials


def list_files(account: str, page_size: int = 10) -> dict:
    if page_size < 1:
        raise ValueError("page_size must be >= 1")

    try:
        creds = get_credentials(account)
        service = build("drive", "v3", credentials=creds)
        results = service.files().list(pageSize=page_size, fields="files(id, name)").execute()
        files = results.get("files", [])
        return {"count": len(files), "files": files}
    except HttpError as exc:
        raise RuntimeError(f"Drive list_files API error: {exc}") from exc


def upload_file(account: str, file_path: str, mime_type: str | None = None) -> dict:
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        creds = get_credentials(account)
        service = build("drive", "v3", credentials=creds)
        media = MediaFileUpload(str(path), mimetype=mime_type, resumable=True)
        created = service.files().create(
            body={"name": path.name},
            media_body=media,
            fields="id, name",
        ).execute()
        return {"id": created.get("id"), "name": created.get("name")}
    except HttpError as exc:
        raise RuntimeError(f"Drive upload_file API error: {exc}") from exc


def delete_file(account: str, file_id: str) -> dict:
    if not file_id:
        raise ValueError("file_id is required")

    try:
        creds = get_credentials(account)
        service = build("drive", "v3", credentials=creds)
        service.files().delete(fileId=file_id).execute()
        return {"deleted": True, "file_id": file_id}
    except HttpError as exc:
        raise RuntimeError(f"Drive delete_file API error: {exc}") from exc
