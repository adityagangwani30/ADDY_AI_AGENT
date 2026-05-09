from __future__ import annotations

import logging
import re
from typing import Optional

from memory.file_index import FileIndex

LOGGER = logging.getLogger("services.alias_service")


def learn_alias_from_text(text: str) -> Optional[tuple[str, str]]:
    """Simple rule: detect "this is my X" or "this is my latest X" and return alias X.

    Returns tuple(alias, file_id) if a recent file exists to map to, else None.
    """
    m = re.search(r"this is my (latest )?(?P<alias>[\w\- ]{2,50})", text, flags=re.I)
    if not m:
        return None
    alias = m.group("alias").strip().lower()

    idx = FileIndex()
    # pick the most recently used file
    cur = idx.conn.cursor()
    cur.execute("SELECT * FROM files ORDER BY recent_usage_ts DESC LIMIT 1")
    row = cur.fetchone()
    if not row:
        return None
    file_id = row["file_id"]
    try:
        idx.add_alias(alias, file_id)
        LOGGER.info("alias_service: mapped alias=%s -> %s", alias, file_id)
        return (alias, file_id)
    except Exception:
        LOGGER.exception("alias_service: failed to add alias %s", alias)
        return None


__all__ = ["learn_alias_from_text"]
