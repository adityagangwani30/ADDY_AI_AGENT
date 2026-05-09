from __future__ import annotations

from pathlib import Path
from typing import Any

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload

from auth.google_auth_manager import get_credentials


def _build_drive_service(account: str):
    creds = get_credentials(account)
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def list_files(account: str, page_size: int = 10) -> dict:
    """
    List files from the user's Google Drive.

    Args:
        account: The account identifier used to retrieve OAuth credentials.
        page_size: Number of files to return (must be >= 1).

    Returns:
        A dict with ``count`` (int) and ``files`` (list of file objects with id and name).

    Raises:
        ValueError: If page_size is less than 1.
        RuntimeError: If the Drive API returns an HTTP error.
    """
    if page_size < 1:
        raise ValueError("page_size must be >= 1")

    try:
        service = _build_drive_service(account)
        results = service.files().list(pageSize=page_size, fields="files(id, name)").execute()
        files = results.get("files", [])
        return {"count": len(files), "files": files}
    except HttpError as exc:
        raise RuntimeError(f"Drive list_files API error: {exc}") from exc


def upload_file(account: str, file_path: str, mime_type: str | None = None) -> dict:
    """
    Upload a local file to Google Drive.

    Args:
        account: The account identifier used to retrieve OAuth credentials.
        file_path: Absolute or relative path to the local file to upload.
        mime_type: Optional MIME type override. Detected automatically if omitted.

    Returns:
        A dict with ``id`` and ``name`` of the uploaded Drive file.

    Raises:
        FileNotFoundError: If the file at file_path does not exist.
        RuntimeError: If the Drive API returns an HTTP error.
    """
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        service = _build_drive_service(account)
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
    """
    Permanently delete a file from Google Drive.

    Args:
        account: The account identifier used to retrieve OAuth credentials.
        file_id: The Google Drive file ID to delete.

    Returns:
        A dict with ``deleted`` (True) and ``file_id``.

    Raises:
        ValueError: If file_id is empty.
        RuntimeError: If the Drive API returns an HTTP error.
    """
    if not file_id:
        raise ValueError("file_id is required")

    try:
        service = _build_drive_service(account)
        service.files().delete(fileId=file_id).execute()
        return {"deleted": True, "file_id": file_id}
    except HttpError as exc:
        raise RuntimeError(f"Drive delete_file API error: {exc}") from exc


def search_files(account: str, filename: str, max_results: int = 10) -> dict:
    """
    Search Drive files by partial filename match.
    """
    name = (filename or "").strip()
    if not name:
        raise ValueError("filename is required")

    page_size = max(1, min(int(max_results), 50))
    safe_name = name.replace("'", "\\'")
    query = f"name contains '{safe_name}' and trashed = false"
    try:
        service = _build_drive_service(account)
        result = service.files().list(
            q=query,
            pageSize=page_size,
            fields="files(id, name, mimeType, webViewLink, createdTime)",
        ).execute()
        files = result.get("files", [])
        return {"count": len(files), "files": files, "query": name}
    except HttpError as exc:
        raise RuntimeError(f"Drive search_files API error: {exc}") from exc


def retrieve_file(account: str, file_id: str = "", filename: str = "") -> dict:
    """
    Retrieve file metadata by id or by best partial-name match.
    """
    target_id = (file_id or "").strip()
    try:
        service = _build_drive_service(account)
        if not target_id:
            searched = search_files(account=account, filename=filename, max_results=1)
            files = searched.get("files", [])
            if not files:
                raise ValueError("No matching file found")
            target_id = str(files[0].get("id") or "")

        if not target_id:
            raise ValueError("file_id or filename is required")

        meta = service.files().get(
            fileId=target_id,
            fields="id, name, mimeType, size, webViewLink, webContentLink, createdTime, modifiedTime",
        ).execute()
        return {"file": meta}
    except HttpError as exc:
        raise RuntimeError(f"Drive retrieve_file API error: {exc}") from exc


def share_file(account: str, file_id: str = "", filename: str = "") -> dict:
    """
    Create/ensure an anyone-with-link reader share and return a view link.
    """
    target_id = (file_id or "").strip()
    try:
        service = _build_drive_service(account)
        if not target_id:
            searched = search_files(account=account, filename=filename, max_results=1)
            files = searched.get("files", [])
            if not files:
                raise ValueError("No matching file found")
            target_id = str(files[0].get("id") or "")

        if not target_id:
            raise ValueError("file_id or filename is required")

        service.permissions().create(
            fileId=target_id,
            body={"type": "anyone", "role": "reader"},
        ).execute()
        meta = service.files().get(fileId=target_id, fields="id, name, webViewLink").execute()
        return {"shared": True, "file": meta}
    except HttpError as exc:
        raise RuntimeError(f"Drive share_file API error: {exc}") from exc


def drive_upload(account: str, file_path: str, mime_type: str | None = None, overwrite: bool = False) -> dict:
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    if overwrite:
        try:
            existing = search_files(account=account, filename=path.name, max_results=1)
            files = existing.get("files", [])
            if files:
                delete_file(account=account, file_id=str(files[0].get("id") or ""))
        except Exception:
            pass
    return upload_file(account=account, file_path=file_path, mime_type=mime_type)


def drive_search(account: str, filename: str, max_results: int = 10) -> dict:
    return search_files(account=account, filename=filename, max_results=max_results)


def drive_retrieve(account: str, file_id: str = "", filename: str = "") -> dict:
    return retrieve_file(account=account, file_id=file_id, filename=filename)


def drive_share(account: str, file_id: str = "", filename: str = "") -> dict:
    return share_file(account=account, file_id=file_id, filename=filename)
