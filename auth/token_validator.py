from __future__ import annotations

import logging
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from auth.google_auth_manager import ACCOUNTS_FILE, list_available_accounts, load_accounts
from config import (
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GOOGLE_REFRESH_TOKEN_1,
    GOOGLE_REFRESH_TOKEN_2,
    GOOGLE_REFRESH_TOKEN_3,
    GOOGLE_REFRESH_TOKEN_4,
    GOOGLE_SCOPES,
    GOOGLE_TOKEN_URI,
)

LOGGER = logging.getLogger(__name__)


def _bootstrap_refresh_tokens() -> dict[str, str]:
    return {
        "GOOGLE_REFRESH_TOKEN_1": GOOGLE_REFRESH_TOKEN_1,
        "GOOGLE_REFRESH_TOKEN_2": GOOGLE_REFRESH_TOKEN_2,
        "GOOGLE_REFRESH_TOKEN_3": GOOGLE_REFRESH_TOKEN_3,
        "GOOGLE_REFRESH_TOKEN_4": GOOGLE_REFRESH_TOKEN_4,
    }


def _build_credentials_from_account_data(account_name: str) -> Credentials:
    accounts = load_accounts()
    account_data = accounts.get(account_name)
    if not isinstance(account_data, dict) and ACCOUNTS_FILE.exists():
        try:
            account_file_data = json.loads(Path(ACCOUNTS_FILE).read_text(encoding="utf-8"))
            if isinstance(account_file_data, dict):
                account_data = account_file_data.get(account_name)
        except Exception:
            account_data = None
    if not isinstance(account_data, dict):
        raise ValueError(f"Account '{account_name}' not found in stored credentials")

    refresh_token = str(account_data.get("refresh_token") or "").strip()
    access_token = str(account_data.get("access_token") or account_data.get("token") or "").strip() or None
    if not refresh_token and not access_token:
        raise ValueError(f"Account '{account_name}' does not contain usable OAuth tokens")

    client_id = str(account_data.get("client_id") or GOOGLE_CLIENT_ID or "").strip() or None
    client_secret = str(account_data.get("client_secret") or GOOGLE_CLIENT_SECRET or "").strip() or None
    token_uri = str(account_data.get("token_uri") or GOOGLE_TOKEN_URI or "").strip() or GOOGLE_TOKEN_URI
    scopes = account_data.get("scopes") or GOOGLE_SCOPES

    return Credentials(
        token=access_token,
        refresh_token=refresh_token or None,
        token_uri=token_uri,
        client_id=client_id,
        client_secret=client_secret,
        scopes=scopes,
    )


def _check_gmail(creds: Credentials) -> str:
    service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    service.users().labels().list(userId="me").execute()
    return "ok"


def _check_calendar(creds: Credentials) -> str:
    service = build("calendar", "v3", credentials=creds, cache_discovery=False)
    now = datetime.now(timezone.utc).isoformat()
    service.events().list(calendarId="primary", maxResults=1, timeMin=now, singleEvents=True).execute()
    return "ok"


def _check_drive(creds: Credentials) -> str:
    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    service.files().list(pageSize=1, fields="files(id)").execute()
    return "ok"


def validate_refresh_token(
    refresh_token: str | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
    token_uri: str | None = None,
    scopes: list[str] | None = None,
) -> dict[str, Any]:
    token = (refresh_token or GOOGLE_REFRESH_TOKEN_1 or "").strip()
    if not token:
        return {"status": "missing", "refresh_token": "missing"}

    creds = Credentials(
        token=None,
        refresh_token=token,
        token_uri=token_uri or GOOGLE_TOKEN_URI,
        client_id=client_id or GOOGLE_CLIENT_ID,
        client_secret=client_secret or GOOGLE_CLIENT_SECRET,
        scopes=scopes or GOOGLE_SCOPES,
    )

    result: dict[str, Any] = {
        "status": "unknown",
        "refresh_token": "present",
        "expires_at": None,
        "active_scopes": sorted(list(creds.scopes or [])),
        "gmail": "unchecked",
        "calendar": "unchecked",
        "drive": "unchecked",
    }

    try:
        creds.refresh(Request())
        result["expires_at"] = creds.expiry.isoformat() if creds.expiry else None
        result["refresh_token"] = "usable"
    except Exception as exc:
        result["status"] = "error"
        result["refresh_token"] = "invalid"
        result["error"] = type(exc).__name__
        return result

    try:
        result["gmail"] = _check_gmail(creds)
    except Exception as exc:
        result["gmail"] = "error"
        result.setdefault("errors", {})["gmail"] = type(exc).__name__

    try:
        result["calendar"] = _check_calendar(creds)
    except Exception as exc:
        result["calendar"] = "error"
        result.setdefault("errors", {})["calendar"] = type(exc).__name__

    try:
        result["drive"] = _check_drive(creds)
    except Exception as exc:
        result["drive"] = "error"
        result.setdefault("errors", {})["drive"] = type(exc).__name__

    result["status"] = "healthy" if all(result[key] == "ok" for key in ("gmail", "calendar", "drive")) else "degraded"
    return result


def validate_account_health(account_name: str) -> dict[str, Any]:
    result: dict[str, Any] = {"account": account_name}
    try:
        creds = _build_credentials_from_account_data(account_name)
        if not creds.valid:
            creds.refresh(Request())
    except Exception as exc:
        result.update({"status": "error", "error": type(exc).__name__, "message": str(exc)})
        return result

    result.update(
        {
            "status": "healthy" if creds.valid else "degraded",
            "expires_at": creds.expiry.isoformat() if creds.expiry else None,
            "active_scopes": sorted(list(creds.scopes or [])),
            "refresh_token": "present" if creds.refresh_token else "missing",
        }
    )

    try:
        result["gmail"] = _check_gmail(creds)
    except Exception as exc:
        result["gmail"] = "error"
        result.setdefault("errors", {})["gmail"] = type(exc).__name__

    try:
        result["calendar"] = _check_calendar(creds)
    except Exception as exc:
        result["calendar"] = "error"
        result.setdefault("errors", {})["calendar"] = type(exc).__name__

    try:
        result["drive"] = _check_drive(creds)
    except Exception as exc:
        result["drive"] = "error"
        result.setdefault("errors", {})["drive"] = type(exc).__name__

    if result.get("gmail") == result.get("calendar") == result.get("drive") == "ok":
        result["status"] = "healthy"
    else:
        result["status"] = "degraded"
    return result


def validate_oauth_health() -> dict[str, Any]:
    report: dict[str, Any] = {
        "bootstrap": validate_refresh_token(refresh_token=GOOGLE_REFRESH_TOKEN_1),
        "bootstrap_tokens": {},
        "accounts": {},
    }

    for slot_name, slot_token in _bootstrap_refresh_tokens().items():
        report["bootstrap_tokens"][slot_name] = validate_refresh_token(refresh_token=slot_token)

    for account in list_available_accounts():
        report["accounts"][account] = validate_account_health(account)

    all_account_health = all(item.get("status") == "healthy" for item in report["accounts"].values()) if report["accounts"] else True
    bootstrap_health = report["bootstrap"].get("status") in {"healthy", "degraded"}
    report["status"] = "healthy" if bootstrap_health and all_account_health else "degraded"
    return report
