from __future__ import annotations

import logging
from pathlib import Path

from . import (
    BASE_DIR,
    GOOGLE_CLIENT_ID,
    GOOGLE_CLIENT_SECRET,
    GOOGLE_REFRESH_TOKEN_1,
    GROQ_API_KEY,
    NVIDIA_API_KEY,
    TELEGRAM_BOT_TOKEN,
    WEBHOOK_SECRET,
)

LOGGER = logging.getLogger(__name__)

CRITICAL_ENV_VARS = {
    "TELEGRAM_BOT_TOKEN": TELEGRAM_BOT_TOKEN,
    "GROQ_API_KEY": GROQ_API_KEY,
    "NVIDIA_API_KEY": NVIDIA_API_KEY,
    "GOOGLE_CLIENT_ID": GOOGLE_CLIENT_ID,
    "GOOGLE_CLIENT_SECRET": GOOGLE_CLIENT_SECRET,
    "GOOGLE_REFRESH_TOKEN_1": GOOGLE_REFRESH_TOKEN_1,
}

SENSITIVE_FILES = (
    BASE_DIR / ".env",
    BASE_DIR / "accounts.json",
    BASE_DIR / "credentials.json",
    BASE_DIR / "client_secret.json",
    BASE_DIR / "token.json",
)


def _present(value: str | None) -> bool:
    return bool(value and value.strip())


def _redacted_missing_vars() -> list[str]:
    return [name for name, value in CRITICAL_ENV_VARS.items() if not _present(value)]


def warn_on_sensitive_files() -> None:
    for path in SENSITIVE_FILES:
        if Path(path).exists():
            LOGGER.warning("Sensitive local artifact detected: %s", path.name)


def validate_environment(strict: bool = True) -> dict[str, list[str]]:
    missing = _redacted_missing_vars()
    if not _present(WEBHOOK_SECRET):
        LOGGER.warning("WEBHOOK_SECRET is not set. Telegram webhook verification will be disabled.")

    warn_on_sensitive_files()

    if missing:
        message = (
            "Missing required environment variables: "
            + ", ".join(missing)
            + ". Configure them before starting the assistant."
        )
        if strict:
            raise RuntimeError(message)
        LOGGER.warning(message)

    return {
        "missing": missing,
        "present": [name for name in CRITICAL_ENV_VARS if name not in missing],
    }
