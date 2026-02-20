from __future__ import annotations

import json
from pathlib import Path

from google.oauth2.credentials import Credentials

from config import ACCOUNTS_FILE, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, GOOGLE_SCOPES, GOOGLE_TOKEN_URI


class AuthConfigurationError(RuntimeError):
    """Raised when OAuth configuration is incomplete or invalid."""


# WARNING: accounts.json contains refresh tokens and must never be committed.
def _load_accounts() -> dict:
    accounts_path = Path(ACCOUNTS_FILE)
    if not accounts_path.exists():
        raise FileNotFoundError(f"Accounts file not found: {accounts_path}")

    with accounts_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def list_available_accounts() -> set[str]:
    accounts = _load_accounts()
    return set(accounts.keys())


def get_credentials(account_name: str) -> Credentials:
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise AuthConfigurationError(
            "Missing GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET in environment variables."
        )

    accounts = _load_accounts()
    account_data = accounts.get(account_name)
    if account_data is None:
        raise KeyError(f"Unknown account '{account_name}'.")

    refresh_token = account_data.get("refresh_token")
    if not refresh_token:
        raise AuthConfigurationError(f"Missing refresh_token for account '{account_name}'.")

    return Credentials(
        token=None,
        refresh_token=refresh_token,
        token_uri=GOOGLE_TOKEN_URI,
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        scopes=GOOGLE_SCOPES,
    )
