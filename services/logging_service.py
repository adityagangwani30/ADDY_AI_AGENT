from __future__ import annotations

import json
import logging

from config import LOG_LEVEL


def configure_logging() -> None:
    numeric_level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level, format="%(message)s")


def log_event(logger: logging.Logger, level: int, **payload: object) -> None:
    logger.log(level, json.dumps(payload, default=str))
