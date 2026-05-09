"""High-level memory manager for Phase 2.

Provides a lightweight, deterministic-first API for saving and retrieving
memories, recent context, alias resolution and preference management.
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, Iterable, List

from .storage import SQLiteMemoryRepository

logger = logging.getLogger("memory.manager")


class MemoryManager:
    def __init__(self, db_path: str | None = None):
        self.repo = SQLiteMemoryRepository(db_path=db_path)

    # Basic memory CRUD
    def save_memory(self, user_id: str, category: str, key: str | None, value: Any, importance: int = 0) -> int:
        logger.info("memory:save user=%s category=%s key=%s", user_id, category, key)
        return self.repo.save_memory_entry(user_id, category, key, value, importance)

    def get_memories(self, user_id: str, category: str | None = None, key: str | None = None) -> List[dict]:
        return self.repo.get_memory_entries(user_id, category, key)

    def delete_memory(self, entry_id: int) -> None:
        logger.info("memory:delete id=%s", entry_id)
        self.repo.delete_memory_entry(entry_id)

    def search(self, user_id: str, query: str, limit: int = 10) -> List[dict]:
        return self.repo.search_memory_entries(user_id, query, limit)

    def forget(self, user_id: str, category: str | None = None, key: str | None = None) -> int:
        """Delete matching memories by category/key. Returns number deleted (best-effort).

        This implementation finds matching entries and deletes them individually.
        """
        removed = 0
        items = self.get_memories(user_id, category, key)
        for it in items:
            self.delete_memory(int(it["id"]))
            removed += 1
        logger.info("memory:forgot user=%s category=%s key=%s removed=%d", user_id, category, key, removed)
        return removed

    def show(self, user_id: str, category: str | None = None, key: str | None = None) -> List[dict]:
        """Return matching memories for inspection (non-sensitive display by caller)."""
        return self.get_memories(user_id, category, key)

    # Recent context helpers
    def add_context(self, user_id: str, role: str, message: str) -> int:
        return self.repo.add_recent_context(user_id, role, message)

    def get_recent_context(self, user_id: str, limit: int = 12) -> List[dict]:
        return self.repo.get_recent_context(user_id, limit)

    def clear_context_older_than(self, seconds: int) -> int:
        return self.repo.cleanup_short_term_context(keep_seconds=seconds)

    # Alias resolution
    def set_alias(self, user_id: str | None, alias: str, actual_value: str, entity_type: str | None = None) -> int:
        logger.info("alias:set user=%s alias=%s", user_id, alias)
        return self.repo.set_entity_alias(user_id, alias, actual_value, entity_type)

    def resolve_alias(self, user_id: str | None, alias: str) -> str | None:
        result = self.repo.resolve_entity_alias(user_id, alias)
        logger.info("alias:resolve user=%s alias=%s resolved=%s", user_id, alias, bool(result))
        return result

    def list_aliases(self, user_id: str | None = None) -> List[dict]:
        return self.repo.list_entity_aliases(user_id)

    def set_project_alias(self, user_id: str | None, alias: str, repository: str, metadata: dict[str, Any] | None = None) -> int:
        logger.info("project_alias:set user=%s alias=%s", user_id, alias)
        return self.repo.set_project_alias(user_id, alias, repository, metadata)

    def resolve_project_alias(self, user_id: str | None, alias: str) -> str | None:
        result = self.repo.resolve_project_alias(user_id, alias)
        logger.info("project_alias:resolve user=%s alias=%s resolved=%s", user_id, alias, bool(result))
        return result

    def list_project_context(self, user_id: str) -> List[dict]:
        return self.repo.list_project_context(user_id)

    def set_active_repository(self, user_id: str, repository: str, alias: str | None = None, metadata: dict[str, Any] | None = None) -> int:
        return self.repo.set_active_repository(user_id, repository, alias=alias, metadata=metadata)

    def get_active_repository(self, user_id: str) -> str | None:
        return self.repo.get_active_repository(user_id)

    # Preferences
    def set_preference(self, user_id: str, key: str, value: Any) -> None:
        logger.info("preference:set user=%s key=%s", user_id, key)
        self.repo.set_user_preference(user_id, key, value)

    def get_preference(self, user_id: str, key: str) -> Any | None:
        return self.repo.get_user_preference(user_id, key)

    # Simple deterministic memory extraction from text
    def extract_and_store(self, user_id: str, text: str) -> List[int]:
        """Run simple deterministic extraction rules and store matches as memories.

        Returns list of created memory ids.
        """
        created: List[int] = []

        # preference: tone
        m = re.search(r"use (formal|informal) tone", text, flags=re.I)
        if m:
            tone = m.group(1).lower()
            self.set_preference(user_id, "preferred_tone", tone)
            logger.info("extraction: set preferred_tone=%s", tone)

        # file mapping: if user mentions a filename with resume or .pdf
        m2 = re.search(r"(?P<name>[\w\-]+\.(pdf|docx|doc))", text)
        if m2:
            fname = m2.group("name")
            entry_id = self.save_memory(user_id, "reference", "resume", {"filename": fname}, importance=2)
            created.append(entry_id)
            logger.info("extraction: mapped resume -> %s", fname)

        # contact mapping: "sir -> email@example.com"
        m3 = re.search(r"(?P<alias>\w[\w\s\-]{0,20})\s+is\s+(?P<value>[^\s@]+@[^\s]+)", text)
        if m3:
            alias = m3.group("alias").strip()
            value = m3.group("value").strip()
            self.set_alias(user_id, alias, value, entity_type="contact")
            logger.info("extraction: set alias %s -> %s", alias, value)

        return created
