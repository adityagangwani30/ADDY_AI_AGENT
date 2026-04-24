from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
ACCOUNTS_FILE = BASE_DIR / "accounts.json"
CREDENTIALS_FILE = BASE_DIR / "credentials.json"

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
    "https://www.googleapis.com/auth/calendar",
    "https://www.googleapis.com/auth/drive.readonly",
]


class AuthConfigurationError(RuntimeError):
    """Raised when OAuth configuration is incomplete or invalid."""


class AuthReauthRequired(RuntimeError):
    """Raised when an account needs re-authentication (token revoked, expired, etc.)."""


# ────────────────────────────────────────────
# Account persistence
# ────────────────────────────────────────────


def load_accounts() -> dict[str, Any]:
    """Load accounts with ENV priority (Render-safe)."""

    # Priority 1: ENV (Render / cloud)
    env_value = os.getenv("ACCOUNTS_JSON")
    if env_value:
        try:
            data = json.loads(env_value)
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError as exc:
            LOGGER.error("Failed to parse ACCOUNTS_JSON env var: %s", exc)

    # Priority 2: local file (dev fallback)
    if ACCOUNTS_FILE.exists():
        try:
            with open(ACCOUNTS_FILE, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    return data
        except (json.JSONDecodeError, IOError) as exc:
            LOGGER.error("Failed to load %s: %s", ACCOUNTS_FILE, exc)

    LOGGER.warning("No account data found (neither ACCOUNTS_JSON env nor %s file).", ACCOUNTS_FILE)
    return {}


def save_accounts(accounts: dict[str, Any]) -> None:
    """Persist credentials locally (skip in cloud)."""
    if os.getenv("ACCOUNTS_JSON"):
        LOGGER.info("Skipping save — cloud environment (ACCOUNTS_JSON is set).")
        return

    try:
        ACCOUNTS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(ACCOUNTS_FILE, "w", encoding="utf-8") as fh:
            json.dump(accounts, fh, indent=2)
    except IOError as exc:
        LOGGER.error("Failed to save %s: %s", ACCOUNTS_FILE, exc)


def list_available_accounts() -> list[str]:
    """Return account emails that have stored credentials."""
    return list(load_accounts().keys())


def mark_account_invalid(account_name: str) -> None:
    """
    Mark an account as needing re-authentication.

    Sets an 'invalid' flag in the stored account data so the system
    can prompt the user to re-authenticate.
    """
    accounts = load_accounts()
    if account_name in accounts:
        accounts[account_name]["_invalid"] = True
        accounts[account_name]["_invalid_since"] = datetime.now(timezone.utc).isoformat()
        save_accounts(accounts)
        LOGGER.warning("Account '%s' marked as invalid — re-authentication required.", account_name)


def is_account_invalid(account_name: str) -> bool:
    """Check if an account has been marked as invalid."""
    accounts = load_accounts()
    account_data = accounts.get(account_name, {})
    return account_data.get("_invalid", False)


# ────────────────────────────────────────────
# Automated re-authentication
# ────────────────────────────────────────────


def auto_authenticate(email: str) -> Credentials:
    """
    Run the InstalledAppFlow to re-authenticate an account.

    Opens the browser, captures the redirect, and returns fresh Credentials.
    Also updates accounts.json.
    """
    from google_auth_oauthlib.flow import InstalledAppFlow
    from config import GOOGLE_TOKEN_URI

    if not CREDENTIALS_FILE.exists():
        raise FileNotFoundError(
            f"credentials.json not found at {CREDENTIALS_FILE}. "
            "Download it from Google Cloud Console."
        )

    LOGGER.info("Starting automated OAuth flow for '%s'...", email)

    flow = InstalledAppFlow.from_client_secrets_file(
        str(CREDENTIALS_FILE),
        SCOPES,
    )

    creds = flow.run_local_server(
        port=8080,
        prompt="consent",
        access_type="offline",
        login_hint=email,
    )

    if not creds.refresh_token:
        raise AuthReauthRequired(
            f"Refresh token missing for '{email}'. "
            "Revoke app access at https://myaccount.google.com/permissions and retry."
        )

    # Save to accounts.json
    accounts = load_accounts()
    accounts[email] = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": GOOGLE_TOKEN_URI,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes) if creds.scopes else SCOPES,
        "expiry": creds.expiry.isoformat() if creds.expiry else None,
    }

    # Clear invalid flag if present
    accounts[email].pop("_invalid", None)
    accounts[email].pop("_invalid_since", None)

    save_accounts(accounts)
    LOGGER.info("Successfully authenticated '%s' and saved tokens.", email)

    return creds


# ────────────────────────────────────────────
# Credential retrieval (core function)
# ────────────────────────────────────────────


def get_credentials(account_name: str) -> Credentials:
    """
    Return valid Google OAuth credentials for *account_name*.

    Flow:
      1. Load stored credentials (from ENV or file)
      2. Build Credentials object (supports old and new format)
      3. If expired → auto-refresh via refresh_token
      4. If refresh fails → mark account invalid and raise clear error

    Auto-recovery: If any token refresh fails, the account is marked as
    invalid and an AuthReauthRequired error is raised with instructions
    for the user to reconnect.
    """
    # Check if account is already flagged as invalid
    if is_account_invalid(account_name):
        raise AuthReauthRequired(
            f"⚠️ Account '{account_name}' needs re-authentication. "
            "Please reconnect by running: python reauth.py --email " + account_name
        )

    accounts = load_accounts()
    account_data = accounts.get(account_name)

    if not account_data:
        raise AuthConfigurationError(
            f"Account '{account_name}' not found. "
            f"Available: {list(accounts.keys())}"
        )

    # Build Credentials from stored data
    creds = _build_credentials_from_stored(account_name, account_data)

    if creds is None:
        raise AuthConfigurationError(
            f"Could not build credentials for '{account_name}'. "
            "Check that the account data contains a valid refresh_token, client_id, and client_secret."
        )

    # Auto-refresh if expired
    if not creds.valid:
        creds = _try_refresh(account_name, creds)

    if creds is None:
        # Auto-recovery: mark account as invalid and raise
        mark_account_invalid(account_name)
        raise AuthReauthRequired(
            f"⚠️ Your account '{account_name}' needs re-authentication. "
            "Please reconnect by running: python reauth.py --email " + account_name
        )

    # Persist updated token (locally only)
    _save_credential(accounts, account_name, creds)

    return creds


# ────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────


def _build_credentials_from_stored(
    account_name: str,
    account_data: dict[str, Any],
) -> Credentials | None:
    """Build Credentials from stored JSON data (supports old and new format)."""

    # New format: full credential object (has client_id or token)
    if "client_id" in account_data or "token" in account_data:
        # Filter out internal metadata keys before passing to google-auth
        filtered = {k: v for k, v in account_data.items() if not k.startswith("_")}
        try:
            return Credentials.from_authorized_user_info(filtered, SCOPES)
        except Exception as exc:
            LOGGER.warning("Failed to load full credentials for '%s': %s", account_name, exc)
            return None

    # Old format: only refresh_token — build using config values
    refresh_token = account_data.get("refresh_token")
    if refresh_token:
        from config import GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_TOKEN_URI

        if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
            LOGGER.warning(
                "Cannot build credentials for '%s': "
                "old format requires GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET env vars.",
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
        LOGGER.error("Token refresh failed for '%s': %s", account_name, exc)
        return None


def _save_credential(
    accounts: dict[str, Any],
    account_name: str,
    creds: Credentials,
) -> None:
    """Store the full credential object and persist (local only)."""
    accounts[account_name] = {
        "token": creds.token,
        "refresh_token": creds.refresh_token,
        "token_uri": creds.token_uri,
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "scopes": list(creds.scopes) if creds.scopes else SCOPES,
        "expiry": creds.expiry.isoformat() if creds.expiry else None,
    }
    save_accounts(accounts)
