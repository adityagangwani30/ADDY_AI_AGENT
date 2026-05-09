from __future__ import annotations

import contextvars
import json
import logging
from datetime import datetime, timezone
from typing import Any

from config import LOG_LEVEL

REQUEST_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
        }

        message = record.getMessage()
        parsed: dict[str, Any] | None = None
        if message:
            stripped = message.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                try:
                    value = json.loads(stripped)
                except Exception:
                    value = None
                if isinstance(value, dict):
                    parsed = value

        if parsed is not None:
            payload.update(parsed)
            if "message" not in payload and "event" not in payload:
                payload["message"] = message
        else:
            payload["message"] = message

        request_id = getattr(record, "request_id", None) or REQUEST_ID.get()
        if request_id:
            payload["request_id"] = request_id

        return json.dumps(payload, default=str)


def configure_logging() -> None:
    numeric_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric_level)

    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root.handlers.clear()
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def bind_request_id(request_id: str | None) -> contextvars.Token[str | None]:
    return REQUEST_ID.set(request_id)


def reset_request_id(token: contextvars.Token[str | None]) -> None:
    REQUEST_ID.reset(token)


def log_event(logger: logging.Logger, level: int, **payload: object) -> None:
    logger.log(level, json.dumps(payload, default=str))
