from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
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

GROQ_API_KEY = _env("GROQ_API_KEY")
GROQ_MODEL = _env("GROQ_MODEL", "llama-3.3-70b-versatile")
NVIDIA_API_KEY = _env("NVIDIA_API_KEY")
NVIDIA_MODEL = _env("NVIDIA_MODEL", "meta/llama-3.3-70b-instruct")
LLM_TIMEOUT_SECONDS = int(_env("LLM_TIMEOUT_SECONDS", "15"))
TELEGRAM_BOT_TOKEN = _env("TELEGRAM_BOT_TOKEN")
GOOGLE_CLIENT_ID = _env("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = _env("GOOGLE_CLIENT_SECRET")
GOOGLE_REFRESH_TOKEN = _env("GOOGLE_REFRESH_TOKEN")
GOOGLE_TOKEN_URI = _env("GOOGLE_TOKEN_URI", "https://oauth2.googleapis.com/token")
GOOGLE_SCOPES = _env_list(
    "GOOGLE_SCOPES",
    [
        "https://www.googleapis.com/auth/gmail.readonly",
        "https://www.googleapis.com/auth/gmail.send",
        "https://www.googleapis.com/auth/calendar",
        "https://www.googleapis.com/auth/drive",
    ],
)

MEMORY_DB_PATH = Path(_env("MEMORY_DB_PATH", str(BASE_DIR / "memory" / "assistant_memory.db")))

WEBHOOK_SECRET = _env("WEBHOOK_SECRET")
