"""
Persistent memory storage for the AI assistant.

Provides a SQLite-backed implementation of the ``MemoryRepository`` abstract
base class, covering: pending action confirmations, per-session conversation
history, account preferences, and account alias management.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import MEMORY_DB_PATH

# Default account alias memory.
DEFAULT_ACCOUNT_ALIASES: dict[str, str] = {
    "exam": "adityagangwaniexam@gmail.com",
    "college": "1ms23ec007@msrit.edu",
    "personal": "adityabvbvpn0011@gmail.com",
    "private": "ashgangcr7@gmail.com",
}


class MemoryRepository(ABC):
    @abstractmethod
    def save_pending_confirmation(self, session_id: str, tool_name: str, account: str, parameters: dict[str, Any]) -> None:
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
    """
    SQLite-backed implementation of ``MemoryRepository``.

    Thread-safe via an internal ``threading.Lock``.  All writes are committed
    immediately so that concurrent readers always see the latest state.

    Args:
        db_path: Optional path to the SQLite database file.  Falls back to
            ``MEMORY_DB_PATH`` from ``config`` if not provided.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path or MEMORY_DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._initialize_schema()

    def _connect(self) -> sqlite3.Connection:
        """Open a new SQLite connection with row factory enabled."""
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_schema(self) -> None:
        """Create all required tables if they do not already exist."""
        with self._lock:
            with self._connect() as conn:

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS pending_confirmations (
                        session_id TEXT PRIMARY KEY,
                        tool_name TEXT NOT NULL,
                        account TEXT NOT NULL,
                        parameters_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS conversation_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        role TEXT NOT NULL,
                        content TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS account_preferences (
                        session_id TEXT PRIMARY KEY,
                        account TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS account_aliases (
                        alias TEXT PRIMARY KEY,
                        account TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                """)

                conn.commit()

    # ---------------- Pending Confirmation ----------------

    def save_pending_confirmation(
        self, session_id: str, tool_name: str, account: str, parameters: dict[str, Any]
    ) -> None:
        """
        Persist a destructive action awaiting user confirmation.

        Overwrites any existing pending confirmation for the same session.
        """
        payload = json.dumps(parameters)
        timestamp = datetime.now(timezone.utc).isoformat()

        with self._lock:
            with self._connect() as conn:
                conn.execute("""
                    INSERT INTO pending_confirmations (session_id, tool_name, account, parameters_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        tool_name=excluded.tool_name,
                        account=excluded.account,
                        parameters_json=excluded.parameters_json,
                        created_at=excluded.created_at
                """, (session_id, tool_name, account, payload, timestamp))
                conn.commit()

    def get_pending_confirmation(self, session_id: str) -> dict[str, Any] | None:
        """
        Retrieve an unconfirmed pending action for the given session.

        Returns:
            A dict with ``tool_name``, ``account``, and ``parameters``;
            or ``None`` if no pending confirmation exists.
        """
        with self._lock:
            with self._connect() as conn:
                row = conn.execute("""
                    SELECT * FROM pending_confirmations WHERE session_id = ?
                """, (session_id,)).fetchone()

        if not row:
            return None

        return {
            "tool_name": row["tool_name"],
            "account": row["account"],
            "parameters": json.loads(row["parameters_json"]),
            "created_at": row["created_at"],
        }

    def clear_pending_confirmation(self, session_id: str) -> None:
        """Delete the pending confirmation for the given session (after confirmation or cancellation)."""
        with self._lock:
            with self._connect() as conn:
                conn.execute("DELETE FROM pending_confirmations WHERE session_id = ?", (session_id,))
                conn.commit()

    # ---------------- Conversation ----------------

    def add_conversation(self, session_id: str, role: str, content: str) -> None:
        """Append a new message to the conversation history for the given session."""
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._connect() as conn:
                conn.execute("""
                    INSERT INTO conversation_history (session_id, role, content, created_at)
                    VALUES (?, ?, ?, ?)
                """, (session_id, role, content, timestamp))
                conn.commit()

    def get_conversation(self, session_id: str, limit: int = 12) -> list[dict[str, str]]:
        """
        Retrieve the most recent conversation messages for a session.

        Args:
            session_id: The session to query.
            limit: Maximum number of messages to return (most recent first, then reversed).

        Returns:
            A list of dicts with ``role`` and ``content`` in chronological order.
        """
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute("""
                    SELECT role, content FROM conversation_history
                    WHERE session_id = ?
                    ORDER BY id DESC
                    LIMIT ?
                """, (session_id, limit)).fetchall()

        result = [{"role": r["role"], "content": r["content"]} for r in rows]
        result.reverse()
        return result

    # ---------------- Account Preference ----------------

    def set_account_preference(self, session_id: str, account: str) -> None:
        """Persist the most recently used account for a session."""
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._connect() as conn:
                conn.execute("""
                    INSERT INTO account_preferences (session_id, account, updated_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        account=excluded.account,
                        updated_at=excluded.updated_at
                """, (session_id, account, timestamp))
                conn.commit()

    def get_account_preference(self, session_id: str) -> str | None:
        """
        Return the last-used account for a session, or ``None`` if not set.
        """
        with self._lock:
            with self._connect() as conn:
                row = conn.execute("""
                    SELECT account FROM account_preferences WHERE session_id = ?
                """, (session_id,)).fetchone()

        return row["account"] if row else None

    # ---------------- Alias System ----------------

    def set_account_alias(self, alias: str, account: str) -> None:
        """Map a short alias (e.g. ``"college"``) to a full account email address."""
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._connect() as conn:
                conn.execute("""
                    INSERT INTO account_aliases (alias, account, created_at)
                    VALUES (?, ?, ?)
                    ON CONFLICT(alias) DO UPDATE SET
                        account=excluded.account
                """, (alias.lower(), account, timestamp))
                conn.commit()

    def get_account_by_alias(self, alias: str) -> str | None:
        """
        Look up an account email by alias, falling back to the built-in defaults.

        Returns:
            The account email string, or ``None`` if not found.
        """
        alias_key = alias.lower()
        with self._lock:
            with self._connect() as conn:
                row = conn.execute("""
                    SELECT account FROM account_aliases WHERE alias = ?
                """, (alias_key,)).fetchone()

        if row:
            return row["account"]
        return DEFAULT_ACCOUNT_ALIASES.get(alias_key)

    def list_account_aliases(self) -> dict[str, str]:
        """
        Return all known alias → account mappings, merging DB rows over defaults.

        Returns:
            A dict mapping alias strings to email addresses.
        """
        with self._lock:
            with self._connect() as conn:
                rows = conn.execute("""
                    SELECT alias, account FROM account_aliases
                """).fetchall()

        aliases = dict(DEFAULT_ACCOUNT_ALIASES)
        for row in rows:
            aliases[row["alias"]] = row["account"]
        return aliases
