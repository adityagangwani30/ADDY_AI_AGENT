from __future__ import annotations

import logging
import os
from typing import Optional

from memory.file_index import FileIndex

LOGGER = logging.getLogger("services.cleanup")


def run_cleanup(db_path: Optional[str] = None, keep_days: int = 365) -> int:
    idx = FileIndex(db_path=db_path) if db_path else FileIndex()
    removed = idx.cleanup_stale(keep_days=keep_days)
    # remove temp folders used by Telegram downloads
    tmp_folder = os.path.join(os.getenv("TMP", "/tmp"), "ai-assistant-telegram")
    try:
        if os.path.exists(tmp_folder):
            for f in os.listdir(tmp_folder):
                p = os.path.join(tmp_folder, f)
                try:
                    os.remove(p)
                except Exception:
                    pass
    except Exception:
        LOGGER.exception("cleanup: failed cleaning temp folder %s", tmp_folder)

    LOGGER.info("cleanup: removed %d file index entries", removed)
    return removed


__all__ = ["run_cleanup"]
