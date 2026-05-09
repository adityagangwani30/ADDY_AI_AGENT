from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from memory.storage import SQLiteMemoryRepository

CONFIRM_WORDS = {"yes", "confirm", "ok", "approve", "yes confirm", "confirm action"}
CANCEL_WORDS = {"cancel", "no", "reject", "deny", "abort"}
DEFAULT_CONFIRMATION_TTL_SECONDS = 180


class ConfirmationManager:
    def __init__(self, memory_repo: SQLiteMemoryRepository, ttl_seconds: int = DEFAULT_CONFIRMATION_TTL_SECONDS) -> None:
        self.memory_repo = memory_repo
        self.ttl_seconds = ttl_seconds

    def save(self, session_id: str, tool_name: str, account: str, parameters: dict[str, Any]) -> None:
        self.memory_repo.save_pending_confirmation(session_id, tool_name, account, parameters)

    def clear(self, session_id: str) -> None:
        self.memory_repo.clear_pending_confirmation(session_id)

    def classify_reply(self, text: str) -> str:
        normalized = (text or "").strip().lower()
        if normalized in CONFIRM_WORDS:
            return "confirm"
        if normalized in CANCEL_WORDS:
            return "cancel"
        return "other"

    def get_valid_pending(self, session_id: str) -> dict[str, Any] | None:
        pending = self.memory_repo.get_pending_confirmation(session_id)
        if not pending:
            return None

        created_raw = str(pending.get("created_at") or "")
        if not created_raw:
            return pending

        try:
            created = datetime.fromisoformat(created_raw)
        except ValueError:
            return pending

        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)

        if datetime.now(timezone.utc) > created + timedelta(seconds=self.ttl_seconds):
            self.clear(session_id)
            return None

        return pending
