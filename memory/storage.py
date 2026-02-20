from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import MEMORY_DB_PATH


class MemoryRepository(ABC):
    @abstractmethod
    def save_pending_confirmation(
        self,
        session_id: str,
        tool_name: str,
        account: str,
        parameters: dict[str, Any],
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_pending_confirmation(self, session_id: str) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def clear_pending_confirmation(self, session_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def add_conversation(self, session_id: str, role: str, content: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_conversation(self, session_id: str, limit: int = 12) -> list[dict[str, str]]:
        raise NotImplementedError

    @abstractmethod
    def set_account_preference(self, session_id: str, account: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_account_preference(self, session_id: str) -> str | None:
        raise NotImplementedError


class SQLiteMemoryRepository(MemoryRepository):
    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path or MEMORY_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_schema(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS pending_confirmations (
                        session_id TEXT PRIMARY KEY,
                        tool_name TEXT NOT NULL,
                        account TEXT NOT NULL,
                        parameters_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS conversation_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS account_preferences (
                        session_id TEXT PRIMARY KEY,
                        account TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                conn.commit()

    def save_pending_confirmation(
        self,
        session_id: str,
        tool_name: str,
        account: str,
        parameters: dict[str, Any],
    ) -> None:
        payload = json.dumps(parameters)
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO pending_confirmations (session_id, tool_name, account, parameters_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        tool_name=excluded.tool_name,
                        account=excluded.account,
                        parameters_json=excluded.parameters_json,
                        created_at=excluded.created_at
                    """,
                    (session_id, tool_name, account, payload, timestamp),
                )
                conn.commit()

    def get_pending_confirmation(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT session_id, tool_name, account, parameters_json, created_at
                    FROM pending_confirmations
                    WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchone()

        if row is None:
            return None

        return {
            "session_id": row["session_id"],
            "tool_name": row["tool_name"],
            "account": row["account"],
            "parameters": json.loads(row["parameters_json"]),
            "created_at": row["created_at"],
        }

    def clear_pending_confirmation(self, session_id: str) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM pending_confirmations WHERE session_id = ?", (session_id,))
                conn.commit()

    def add_conversation(self, session_id: str, role: str, content: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO conversation_history (session_id, role, content, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (session_id, role, content, timestamp),
                )
                conn.commit()

    def get_conversation(self, session_id: str, limit: int = 12) -> list[dict[str, str]]:
        bounded_limit = max(1, min(limit, 100))
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT role, content
                    FROM conversation_history
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (session_id, bounded_limit),
                ).fetchall()

        items = [{"role": row["role"], "content": row["content"]} for row in rows]
        items.reverse()
        return items

    def set_account_preference(self, session_id: str, account: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO account_preferences (session_id, account, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        account=excluded.account,
                        updated_at=excluded.updated_at
                    """,
                    (session_id, account, timestamp),
                )
                conn.commit()

    def get_account_preference(self, session_id: str) -> str | None:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT account FROM account_preferences WHERE session_id = ?",
                    (session_id,),
                ).fetchone()

        if row is None:
            return None
        return str(row["account"])


class PostgresMemoryRepository(MemoryRepository):
    """Postgres-ready placeholder for production migration."""

    def save_pending_confirmation(
        self,
        session_id: str,
        tool_name: str,
        account: str,
        parameters: dict[str, Any],
    ) -> None:
        raise NotImplementedError("PostgresMemoryRepository is not implemented yet.")

    def get_pending_confirmation(self, session_id: str) -> dict[str, Any] | None:
        raise NotImplementedError("PostgresMemoryRepository is not implemented yet.")

    def clear_pending_confirmation(self, session_id: str) -> None:
        raise NotImplementedError("PostgresMemoryRepository is not implemented yet.")

    def add_conversation(self, session_id: str, role: str, content: str) -> None:
        raise NotImplementedError("PostgresMemoryRepository is not implemented yet.")

    def get_conversation(self, session_id: str, limit: int = 12) -> list[dict[str, str]]:
        raise NotImplementedError("PostgresMemoryRepository is not implemented yet.")

    def set_account_preference(self, session_id: str, account: str) -> None:
        raise NotImplementedError("PostgresMemoryRepository is not implemented yet.")

    def get_account_preference(self, session_id: str) -> str | None:
        raise NotImplementedError("PostgresMemoryRepository is not implemented yet.")
