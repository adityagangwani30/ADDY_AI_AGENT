from __future__ import annotations

import logging
from typing import Dict, List, Optional

from memory.file_index import FileIndex
from memory.memory_manager import MemoryManager

LOGGER = logging.getLogger("services.unified_search")


class UnifiedSearch:
    def __init__(self, file_index: Optional[FileIndex] = None, memory_mgr: Optional[MemoryManager] = None):
        self.file_index = file_index or FileIndex()
        self.memory = memory_mgr or MemoryManager()

    def search_all(self, user_id: str, query: str, limit: int = 10) -> Dict[str, List[Dict]]:
        """Search files and memories and return combined results.

        Returns dict with keys: files, memories
        """
        files = self.file_index.search(query, limit=limit)
        memories = self.memory.search(user_id, query, limit=limit) if user_id else []

        LOGGER.info("unified_search query=%s files=%d memories=%d", query, len(files), len(memories))

        return {"files": files, "memories": memories}


__all__ = ["UnifiedSearch"]
