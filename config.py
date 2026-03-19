from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

# Centralized environment loading for the entire application.
load_dotenv(ENV_FILE)


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().strip('"').strip("'")


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = _env(name)
    if not raw:
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


LOG_LEVEL = _env("LOG_LEVEL", "INFO")

GEMINI_API_KEY = _env("GEMINI_API_KEY")
GEMINI_MODEL_PRIMARY = _env("GEMINI_MODEL_PRIMARY", "gemini-2.5-flash")
GEMINI_MODEL_FALLBACKS = _env_list("GEMINI_MODEL_FALLBACKS", ["gemini-2.0-flash"])
TELEGRAM_BOT_TOKEN = _env("TELEGRAM_BOT_TOKEN")
GOOGLE_CLIENT_ID = _env("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = _env("GOOGLE_CLIENT_SECRET")
GOOGLE_TOKEN_URI = _env("GOOGLE_TOKEN_URI", "https://oauth2.googleapis.com/token")
GOOGLE_SCOPES = _env_list(
    "GOOGLE_SCOPES",
    [
        "https://www.googleapis.com/auth/gmail.modify",
        "https://www.googleapis.com/auth/calendar",
        "https://www.googleapis.com/auth/drive",
    ],
)

ACCOUNTS_FILE = Path(_env("ACCOUNTS_FILE", str(BASE_DIR / "accounts.json")))
CLIENT_SECRET_FILE = Path(_env("CLIENT_SECRET_FILE", str(BASE_DIR / "client_secret.json")))
MEMORY_DB_PATH = Path(_env("MEMORY_DB_PATH", str(BASE_DIR / "memory" / "assistant_memory.db")))

WEBHOOK_SECRET = _env("WEBHOOK_SECRET")


# --------------- Startup validation ---------------
# Warn loudly about missing critical env vars so Render deployments fail fast.

import sys as _sys

_CRITICAL_VARS = {
    "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN,
    "GEMINI_API_KEY": GEMINI_API_KEY,
}

for _name, _value in _CRITICAL_VARS.items():
    if not _value:
        print(f"WARNING: {_name} is not set. The application will not function correctly.", file=_sys.stderr)

if not _env("ACCOUNTS_JSON"):
    print(
        "WARNING: ACCOUNTS_JSON is not set. Google account credentials will be unavailable.",
        file=_sys.stderr,
    )
