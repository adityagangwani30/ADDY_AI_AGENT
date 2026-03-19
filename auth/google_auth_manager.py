from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
ACCOUNTS_FILE = BASE_DIR / "accounts.json"
CLIENT_SECRET_FILE = BASE_DIR / "client_secret.json"

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/drive",
]


class AuthConfigurationError(RuntimeError):
    """Raised when OAuth configuration is incomplete or invalid."""


# ────────────────────────────────────────────
# Account persistence
# ────────────────────────────────────────────


def load_accounts() -> dict[str, Any]:
    """Load accounts with ENV priority (Render-safe)."""

    # 🔥 Priority 1: ENV (Render)
    env_value = os.getenv("ACCOUNTS_JSON")
    if env_value:
        try:
            data = json.loads(env_value)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError as exc:
            LOGGER.error("Failed to parse ACCOUNTS_JSON env var: %s", exc)

    # 🔥 Priority 2: local file
    if ACCOUNTS_FILE.exists():
        try:
            with open(ACCOUNTS_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    return data
        except (json.JSONDecodeError, IOError) as exc:
            LOGGER.error("Failed to load %s: %s", ACCOUNTS_FILE, exc)

    return {}


def save_accounts(accounts: dict[str, Any]) -> None:
    """Persist credentials locally (skip in cloud)."""

    if os.getenv("ACCOUNTS_JSON"):
        LOGGER.info("Skipping save in cloud environment")
        return

    try:
        ACCOUNTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ACCOUNTS_FILE, "w", encoding="utf-8") as fh:
            json.dump(accounts, fh, indent=2)
    except IOError as exc:
        LOGGER.error("Failed to save %s: %s", ACCOUNTS_FILE, exc)


def list_available_accounts() -> list[str]:
    return list(load_accounts().keys())


# ────────────────────────────────────────────
# Credential retrieval (core function)
# ────────────────────────────────────────────


def get_credentials(account_name: str) -> Credentials:
    """
    Return valid Google OAuth credentials for *account_name*.

    Flow:
      1. Load stored credentials from accounts.json
      2. If expired → auto-refresh via refresh_token
      3. If refresh fails or no stored creds → run InstalledAppFlow (local only)
      4. Save full credential object back to accounts.json
    """
    accounts = load_accounts()
    creds: Credentials | None = None
    account_data = accounts.get(account_name)

    # ── Step 1: build Credentials from stored data ──
    if account_data:
        creds = _build_credentials_from_stored(account_name, account_data)

    # ── Step 2: refresh if needed ──
    if creds and not creds.valid:
        creds = _try_refresh(account_name, creds)

    # ── Step 3: re-authenticate if still invalid ──
    if not creds or not creds.valid:
        LOGGER.info("No valid credentials for '%s'. Starting OAuth flow...", account_name)
        creds = _run_oauth_flow(account_name)

    # ── Step 4: persist full credentials ──
    _save_credential(accounts, account_name, creds)

    return creds


# ────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────


def _build_credentials_from_stored(
    account_name: str,
    account_data: dict[str, Any],
) -> Credentials | None:
    """Try to construct Credentials from stored JSON data (new or old format)."""

    # New format: full credential object (has client_id or token)
    if "client_id" in account_data or "token" in account_data:
        try:
            return Credentials.from_authorized_user_info(account_data, SCOPES)
        except Exception as exc:
            LOGGER.warning(
                "Failed to load full credentials for '%s': %s",
                account_name,
                exc,
            )
            return None

    # Old format: only refresh_token — build manually using config values
    refresh_token = account_data.get("refresh_token")
    if refresh_token:
        from config import GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_TOKEN_URI

        if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
            LOGGER.warning(
                "Cannot build credentials for '%s': "
                "old format (refresh_token only) requires GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET.",
                account_name,
            )
            return None

        return Credentials(
            token=None,
            refresh_token=refresh_token,
            token_uri=GOOGLE_TOKEN_URI,
            client_id=GOOGLE_CLIENT_ID,
            client_secret=GOOGLE_CLIENT_SECRET,
            scopes=SCOPES,
        )

    LOGGER.warning("No usable credential data for '%s'.", account_name)
    return None


def _try_refresh(account_name: str, creds: Credentials) -> Credentials | None:
    """Attempt to refresh expired credentials. Returns None on failure."""
    if not creds.refresh_token:
        LOGGER.warning("No refresh_token for '%s'. Cannot refresh.", account_name)
        return None

    try:
        creds.refresh(Request())
        LOGGER.info("Successfully refreshed token for '%s'.", account_name)
        return creds
    except Exception as exc:
        LOGGER.warning(
            "Token refresh failed for '%s': %s. Will re-authenticate.",
            account_name,
            exc,
        )
        return None


def _run_oauth_flow(account_name: str) -> Credentials:
    """Run InstalledAppFlow to obtain fresh credentials (requires a browser)."""
    if not CLIENT_SECRET_FILE.exists():
        raise AuthConfigurationError(
            f"Cannot authenticate '{account_name}': "
            f"client_secret.json not found at {CLIENT_SECRET_FILE}. "
            "Re-authentication requires a local environment with a browser."
        )

    # Lazy import — only needed when re-authenticating
    from google_auth_oauthlib.flow import InstalledAppFlow

    try:
        flow = InstalledAppFlow.from_client_secrets_file(
            str(CLIENT_SECRET_FILE),
            SCOPES,
        )
        # Use port 8080 to match the redirect_uri configured in client_secret.json
        flow.redirect_uri = "http://localhost:8080/"
        print(f"\n🔐 Authenticating: {account_name}")
        print("   A browser window will open. Sign in with this account.\n")
        creds = flow.run_local_server(port=8080, open_browser=True)
    except Exception as exc:
        raise AuthConfigurationError(
            f"OAuth flow failed for '{account_name}': {exc}"
        ) from exc

    return creds


def _save_credential(
    accounts: dict[str, Any],
    account_name: str,
    creds: Credentials,
) -> None:
    """Store the full credential object for an account and persist to disk."""
    accounts[account_name] = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes) if creds.scopes else SCOPES,
    }
    save_accounts(accounts)
