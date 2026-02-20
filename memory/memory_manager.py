from __future__ import annotations

from memory.storage import SQLiteMemoryRepository


class MemoryManager(SQLiteMemoryRepository):
    """
    Backward-compatible alias over the SQLite-backed repository.
    """

    pass
