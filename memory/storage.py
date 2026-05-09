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
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any
import logging

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
        self.logger = logging.getLogger("memory.storage")

    def _connect(self) -> sqlite3.Connection:
        """Open a new SQLite connection with row factory enabled."""
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _connection(self):
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    def _initialize_schema(self) -> None:
        """Create all required tables if they do not already exist."""
        with self._lock:
            with self._connection() as conn:

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

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS executed_actions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        action_type TEXT NOT NULL,
                        parameters_json TEXT NOT NULL,
                        success INTEGER NOT NULL,
                        error_message TEXT,
                        user_id TEXT,
                        execution_time_ms INTEGER
                    )
                """)

                # Phase 2: persistent memory tables
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS memory_entries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        category TEXT NOT NULL,
                        key TEXT,
                        value TEXT NOT NULL,
                        importance INTEGER DEFAULT 0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_memory_user_cat_key ON memory_entries (user_id, category, key)
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS recent_context (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        message TEXT NOT NULL,
                        role TEXT NOT NULL,
                        timestamp TEXT NOT NULL
                    )
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_recent_context_user_ts ON recent_context (user_id, timestamp)
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS entity_aliases (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT,
                        alias TEXT NOT NULL,
                        actual_value TEXT NOT NULL,
                        entity_type TEXT,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(user_id, alias)
                    )
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_entity_alias_alias ON entity_aliases (alias)
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_preferences (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        preference_key TEXT NOT NULL,
                        preference_value TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(user_id, preference_key)
                    )
                """)

                conn.execute("""
                    CREATE TABLE IF NOT EXISTS project_context (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        user_id TEXT NOT NULL,
                        alias TEXT,
                        repository TEXT NOT NULL,
                        metadata_json TEXT,
                        active INTEGER DEFAULT 0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        UNIQUE(user_id, repository)
                    )
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_project_context_user_alias ON project_context (user_id, alias)
                """)

                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_project_context_user_active ON project_context (user_id, active)
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
            with self._connection() as conn:
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
            with self._connection() as conn:
                conn.execute("DELETE FROM pending_confirmations WHERE session_id = ?", (session_id,))
                conn.commit()

    # ---------------- Conversation ----------------

    def add_conversation(self, session_id: str, role: str, content: str) -> None:
        """Append a new message to the conversation history for the given session."""
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._connection() as conn:
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
            with self._connection() as conn:
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
            with self._connection() as conn:
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
            with self._connection() as conn:
                row = conn.execute("""
                    SELECT account FROM account_preferences WHERE session_id = ?
                """, (session_id,)).fetchone()

        return row["account"] if row else None

    # ---------------- Alias System ----------------

    def set_account_alias(self, alias: str, account: str) -> None:
        """Map a short alias (e.g. ``"college"``) to a full account email address."""
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._connection() as conn:
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
            with self._connection() as conn:
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
            with self._connection() as conn:
                rows = conn.execute("""
                    SELECT alias, account FROM account_aliases
                """).fetchall()

        aliases = dict(DEFAULT_ACCOUNT_ALIASES)
        for row in rows:
            aliases[row["alias"]] = row["account"]
        return aliases

    # ---------------- Action Audit ----------------

    def record_executed_action(
        self,
        action_type: str,
        parameters: dict[str, Any],
        success: bool,
        error_message: str | None = None,
        user_id: str | None = None,
        execution_time_ms: int | None = None,
    ) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._connection() as conn:
                conn.execute(
                    """
                    INSERT INTO executed_actions (
                        timestamp, action_type, parameters_json, success, error_message, user_id, execution_time_ms
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        timestamp,
                        action_type,
                        json.dumps(parameters, default=str),
                        1 if success else 0,
                        error_message,
                        user_id,
                        execution_time_ms,
                    ),
                )
                conn.commit()

    def health_check(self) -> bool:
        with self._connection() as conn:
            conn.execute("SELECT 1")
        return True

    # ---------------- Memory Entries (Long-Term / Reference) ----------------

    def save_memory_entry(
        self, user_id: str, category: str, key: str | None, value: Any, importance: int = 0
    ) -> int:
        """Insert or update a memory entry. Returns the entry id."""
        now = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(value, default=str)
        with self._lock:
            with self._connection() as conn:
                # Try to find existing
                if key is not None:
                    row = conn.execute(
                        "SELECT id FROM memory_entries WHERE user_id = ? AND category = ? AND key = ?",
                        (user_id, category, key),
                    ).fetchone()
                    if row:
                        conn.execute(
                            """
                            UPDATE memory_entries SET value = ?, importance = ?, updated_at = ? WHERE id = ?
                        """,
                            (payload, importance, now, row["id"]),
                        )
                        conn.commit()
                        self.logger.info("update memory_entry user=%s category=%s key=%s id=%s", user_id, category, key, row["id"])
                        return int(row["id"])

                # Insert new
                cur = conn.execute(
                    """
                    INSERT INTO memory_entries (user_id, category, key, value, importance, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    (user_id, category, key, payload, importance, now, now),
                )
                conn.commit()
                self.logger.info("insert memory_entry user=%s category=%s key=%s id=%s", user_id, category, key, cur.lastrowid)
                return int(cur.lastrowid)

    def get_memory_entries(self, user_id: str, category: str | None = None, key: str | None = None) -> list[dict]:
        """Retrieve memory entries for a user, optionally filtering by category/key."""
        with self._lock:
            with self._connection() as conn:
                if category and key:
                    rows = conn.execute(
                        "SELECT * FROM memory_entries WHERE user_id = ? AND category = ? AND key = ? ORDER BY importance DESC, updated_at DESC",
                        (user_id, category, key),
                    ).fetchall()
                elif category:
                    rows = conn.execute(
                        "SELECT * FROM memory_entries WHERE user_id = ? AND category = ? ORDER BY importance DESC, updated_at DESC",
                        (user_id, category),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT * FROM memory_entries WHERE user_id = ? ORDER BY importance DESC, updated_at DESC",
                        (user_id,),
                    ).fetchall()

        result = []
        for r in rows:
            try:
                val = json.loads(r["value"])
            except Exception:
                val = r["value"]
            result.append(
                {
                    "id": r["id"],
                    "user_id": r["user_id"],
                    "category": r["category"],
                    "key": r["key"],
                    "value": val,
                    "importance": r["importance"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                }
            )
        return result

    def delete_memory_entry(self, entry_id: int) -> None:
        with self._lock:
            with self._connection() as conn:
                conn.execute("DELETE FROM memory_entries WHERE id = ?", (entry_id,))
                conn.commit()
                self.logger.info("delete memory_entry id=%s", entry_id)

    def search_memory_entries(self, user_id: str, query: str, limit: int = 10) -> list[dict]:
        """Lightweight semantic-ish search using substring/LIKE + token overlap scoring.

        This intentionally avoids vector DBs: it finds candidate rows using SQL LIKE
        then ranks by token overlap and importance.
        """
        q = query.lower()
        tokens = set([t for t in q.split() if len(t) > 2])
        with self._lock:
            with self._connection() as conn:
                # candidate selection via LIKE on value and key
                pattern = f"%{q}%"
                rows = conn.execute(
                    """
                    SELECT * FROM memory_entries
                    WHERE user_id = ? AND (LOWER(value) LIKE ? OR LOWER(key) LIKE ? OR LOWER(category) LIKE ?)
                    ORDER BY importance DESC, updated_at DESC
                    LIMIT ?
                """,
                    (user_id, pattern, pattern, pattern, limit * 4),
                ).fetchall()

        # avoid logging personal content directly; log a high-level event
        self.logger.debug("search_memory user=%s query_len=%d candidates=%d", user_id, len(query), len(rows))

        scored = []
        for r in rows:
            text = (r["key"] or "") + " " + str(r["value"])
            text_l = text.lower()
            overlap = sum(1 for t in tokens if t in text_l)
            score = overlap + (int(r["importance"] or 0) * 2)
            scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, r in scored[:limit]:
            try:
                val = json.loads(r["value"])
            except Exception:
                val = r["value"]
            results.append({
                "id": r["id"],
                "category": r["category"],
                "key": r["key"],
                "value": val,
                "importance": r["importance"],
                "score": score,
            })
        return results

    # ---------------- Recent Context (Short-Term) ----------------

    def add_recent_context(self, user_id: str, role: str, message: str) -> int:
        ts = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._connection() as conn:
                cur = conn.execute(
                    "INSERT INTO recent_context (user_id, message, role, timestamp) VALUES (?, ?, ?, ?)",
                    (user_id, message, role, ts),
                )
                conn.commit()
                # log context injection without message contents
                self.logger.info("add_recent_context user=%s role=%s id=%s", user_id, role, cur.lastrowid)
                return int(cur.lastrowid)

    def get_recent_context(self, user_id: str, limit: int = 12, since_iso: str | None = None) -> list[dict]:
        with self._lock:
            with self._connection() as conn:
                if since_iso:
                    rows = conn.execute(
                        "SELECT role, message, timestamp FROM recent_context WHERE user_id = ? AND timestamp >= ? ORDER BY id DESC LIMIT ?",
                        (user_id, since_iso, limit),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT role, message, timestamp FROM recent_context WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                        (user_id, limit),
                    ).fetchall()

        result = []
        for r in rows:
            result.append({"role": r["role"], "message": r["message"], "timestamp": r["timestamp"]})
        result.reverse()
        return result

    def clear_recent_context_older_than(self, seconds: int) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()
        with self._lock:
            with self._connection() as conn:
                cur = conn.execute("DELETE FROM recent_context WHERE timestamp < ?", (cutoff,))
                conn.commit()
                return cur.rowcount

    # ---------------- Entity Aliases ----------------

    def set_entity_alias(self, user_id: str | None, alias: str, actual_value: str, entity_type: str | None = None) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            with self._connection() as conn:
                conn.execute(
                    """
                    INSERT INTO entity_aliases (user_id, alias, actual_value, entity_type, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, alias) DO UPDATE SET actual_value=excluded.actual_value, updated_at=excluded.updated_at
                """,
                    (user_id, alias.lower(), actual_value, entity_type, now, now),
                )
                conn.commit()
                row = conn.execute("SELECT id FROM entity_aliases WHERE user_id IS ? AND alias = ?", (user_id, alias.lower())).fetchone()
                _id = int(row["id"]) if row else 0
                # log alias create/update without including actual_value
                self.logger.info("set_entity_alias user=%s alias=%s id=%s", user_id, alias.lower(), _id)
                return _id

    def resolve_entity_alias(self, user_id: str | None, alias: str) -> str | None:
        key = alias.lower()
        with self._lock:
            with self._connection() as conn:
                row = conn.execute("SELECT actual_value FROM entity_aliases WHERE user_id IS ? AND alias = ?", (user_id, key)).fetchone()
                if row:
                    self.logger.info("resolve_entity_alias user=%s alias=%s hit_local", user_id, key)
                    return row["actual_value"]
                # fallback to account alias table if appropriate
                row2 = conn.execute("SELECT account FROM account_aliases WHERE alias = ?", (key,)).fetchone()
                if row2:
                    self.logger.info("resolve_entity_alias user=%s alias=%s hit_account_alias", user_id, key)
                    return row2["account"]
        return None

    def delete_entity_alias(self, user_id: str | None, alias: str) -> None:
        with self._lock:
            with self._connection() as conn:
                conn.execute("DELETE FROM entity_aliases WHERE user_id IS ? AND alias = ?", (user_id, alias.lower()))
                conn.commit()
                self.logger.info("delete_entity_alias user=%s alias=%s", user_id, alias.lower())

    def list_entity_aliases(self, user_id: str | None = None) -> list[dict]:
        with self._lock:
            with self._connection() as conn:
                if user_id is None:
                    rows = conn.execute("SELECT alias, actual_value, entity_type FROM entity_aliases").fetchall()
                else:
                    rows = conn.execute("SELECT alias, actual_value, entity_type FROM entity_aliases WHERE user_id IS ?", (user_id,)).fetchall()

        return [{"alias": r["alias"], "actual_value": r["actual_value"], "entity_type": r["entity_type"]} for r in rows]

    # ---------------- User Preferences ----------------

    def set_user_preference(self, user_id: str, key: str, value: Any) -> None:
        now = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(value, default=str)
        with self._lock:
            with self._connection() as conn:
                conn.execute(
                    """
                    INSERT INTO user_preferences (user_id, preference_key, preference_value, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, preference_key) DO UPDATE SET preference_value=excluded.preference_value, updated_at=excluded.updated_at
                """,
                    (user_id, key, payload, now, now),
                )
                conn.commit()
                self.logger.info("set_user_preference user=%s key=%s", user_id, key)

    def get_user_preference(self, user_id: str, key: str) -> Any | None:
        with self._lock:
            with self._connection() as conn:
                row = conn.execute("SELECT preference_value FROM user_preferences WHERE user_id = ? AND preference_key = ?", (user_id, key)).fetchone()
        if not row:
            return None
        try:
            return json.loads(row["preference_value"])
        except Exception:
            return row["preference_value"]
    # ---------------- Project Context ----------------

    def set_project_context(self, user_id: str, repository: str, alias: str | None = None, metadata: dict[str, Any] | None = None, active: bool = False) -> int:
        now = datetime.now(timezone.utc).isoformat()
        payload = json.dumps(metadata or {}, default=str)
        with self._lock:
            with self._connection() as conn:
                conn.execute(
                    """
                    INSERT INTO project_context (user_id, alias, repository, metadata_json, active, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, repository) DO UPDATE SET
                        alias=excluded.alias,
                        metadata_json=excluded.metadata_json,
                        active=excluded.active,
                        updated_at=excluded.updated_at
                    """,
                    (user_id, alias, repository, payload, 1 if active else 0, now, now),
                )
                conn.commit()
                row = conn.execute("SELECT id FROM project_context WHERE user_id = ? AND repository = ?", (user_id, repository)).fetchone()
                return int(row["id"]) if row else 0

    def set_active_repository(self, user_id: str, repository: str, alias: str | None = None, metadata: dict[str, Any] | None = None) -> int:
        with self._lock:
            with self._connection() as conn:
                conn.execute("UPDATE project_context SET active = 0 WHERE user_id = ?", (user_id,))
                conn.commit()
        return self.set_project_context(user_id, repository, alias=alias, metadata=metadata, active=True)

    def get_active_repository(self, user_id: str) -> str | None:
        with self._lock:
            with self._connection() as conn:
                row = conn.execute("SELECT repository FROM project_context WHERE user_id = ? AND active = 1 ORDER BY updated_at DESC LIMIT 1", (user_id,)).fetchone()
        return row["repository"] if row else None

    def list_project_context(self, user_id: str) -> list[dict]:
        with self._lock:
            with self._connection() as conn:
                rows = conn.execute("SELECT * FROM project_context WHERE user_id = ? ORDER BY active DESC, updated_at DESC", (user_id,)).fetchall()

        result = []
        for row in rows:
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except Exception:
                metadata = row["metadata_json"]
            result.append(
                {
                    "id": row["id"],
                    "user_id": row["user_id"],
                    "alias": row["alias"],
                    "repository": row["repository"],
                    "metadata": metadata,
                    "active": bool(row["active"]),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            )
        return result

    def set_project_alias(self, user_id: str | None, alias: str, repository: str, metadata: dict[str, Any] | None = None) -> int:
        alias_key = alias.lower().strip()
        repo_id = self.set_entity_alias(user_id, alias_key, repository, entity_type="github_repo")
        if user_id and metadata:
            self.set_project_context(user_id, repository, alias=alias_key, metadata=metadata, active=False)
        return repo_id

    def resolve_project_alias(self, user_id: str | None, alias: str) -> str | None:
        return self.resolve_entity_alias(user_id, alias)

    def list_project_aliases(self, user_id: str | None = None) -> list[dict]:
        with self._lock:
            with self._connection() as conn:
                if user_id is None:
                    rows = conn.execute("SELECT alias, actual_value, entity_type FROM entity_aliases WHERE entity_type = 'github_repo'").fetchall()
                else:
                    rows = conn.execute("SELECT alias, actual_value, entity_type FROM entity_aliases WHERE user_id IS ? AND entity_type = 'github_repo'", (user_id,)).fetchall()
        return [{"alias": r["alias"], "actual_value": r["actual_value"], "entity_type": r["entity_type"]} for r in rows]
    # ---------------- Cleanup / TTL helpers ----------------

    def cleanup_short_term_context(self, user_id: str | None = None, keep_seconds: int = 3600) -> int:
        """Remove recent_context rows older than keep_seconds. If user_id provided, scope to that user."""
        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=keep_seconds)).isoformat()
        with self._lock:
            with self._connection() as conn:
                if user_id is None:
                    cur = conn.execute("DELETE FROM recent_context WHERE timestamp < ?", (cutoff,))
                else:
                    cur = conn.execute("DELETE FROM recent_context WHERE user_id = ? AND timestamp < ?", (user_id, cutoff))
                conn.commit()
                self.logger.info("cleanup_short_term_context user=%s removed=%d cutoff=%s", user_id, cur.rowcount, cutoff)
                return cur.rowcount

