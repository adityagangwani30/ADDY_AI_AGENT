from __future__ import annotations

import logging
import os
import sqlite3
import time
from typing import Dict, List, Optional

LOGGER = logging.getLogger("memory.file_index")


class FileIndex:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or os.path.join(os.getcwd(), "file_index.sqlite")
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True) if os.path.dirname(self.db_path) else None
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY,
                file_id TEXT UNIQUE,
                filename TEXT,
                extracted_text TEXT,
                keywords TEXT,
                upload_ts INTEGER,
                source TEXT,
                category TEXT,
                recent_usage_ts INTEGER
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS aliases (
                alias TEXT,
                file_id TEXT
            )
            """
        )
        self.conn.commit()

    def add_file(self, file_id: str, filename: str, extracted_text: str | None = None, keywords: Optional[str] = None, source: Optional[str] = None, category: Optional[str] = None, upload_ts: Optional[int] = None, aliases: Optional[List[str]] = None) -> None:
        upload_ts = upload_ts or int(time.time())
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO files (file_id, filename, extracted_text, keywords, upload_ts, source, category, recent_usage_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (file_id, filename, extracted_text or "", keywords or "", upload_ts, source or "", category or "", upload_ts),
        )
        if aliases:
            for a in aliases:
                try:
                    cur.execute("INSERT INTO aliases (alias, file_id) VALUES (?, ?)", (a.lower(), file_id))
                except Exception:
                    pass
        self.conn.commit()

    def add_alias(self, alias: str, file_id: str) -> None:
        cur = self.conn.cursor()
        cur.execute("INSERT INTO aliases (alias, file_id) VALUES (?, ?)", (alias.lower(), file_id))
        self.conn.commit()

    def get_by_file_id(self, file_id: str) -> Optional[Dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM files WHERE file_id = ?", (file_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def touch_usage(self, file_id: str) -> None:
        cur = self.conn.cursor()
        cur.execute("UPDATE files SET recent_usage_ts = ? WHERE file_id = ?", (int(time.time()), file_id))
        self.conn.commit()

    def search(self, query: str, limit: int = 10) -> List[Dict]:
        q = query.strip().lower()
        cur = self.conn.cursor()

        # 1) exact alias match
        cur.execute("SELECT file_id FROM aliases WHERE alias = ?", (q,))
        row = cur.fetchone()
        results: List[Dict] = []
        seen = set()
        if row:
            file_id = row[0]
            rec = self.get_by_file_id(file_id)
            if rec:
                rec["score"] = 100
                results.append(rec)
                seen.add(rec["file_id"])

        # 2) filename match
        cur.execute("SELECT *, 0 as score FROM files WHERE lower(filename) LIKE ? LIMIT ?", (f"%{q}%", limit))
        for r in cur.fetchall():
            d = dict(r)
            if d["file_id"] in seen:
                continue
            d["score"] = 80
            results.append(d)
            seen.add(d["file_id"])

        # 3) keyword match
        cur.execute("SELECT *, 0 as score FROM files WHERE lower(keywords) LIKE ? LIMIT ?", (f"%{q}%", limit))
        for r in cur.fetchall():
            d = dict(r)
            if d["file_id"] in seen:
                continue
            d["score"] = 60
            results.append(d)
            seen.add(d["file_id"])

        # 4) extracted text match (partial)
        cur.execute("SELECT *, 0 as score FROM files WHERE lower(extracted_text) LIKE ? LIMIT ?", (f"%{q}%", limit))
        for r in cur.fetchall():
            d = dict(r)
            if d["file_id"] in seen:
                continue
            # rough scoring based on recency
            recency = int(time.time()) - (d.get("recent_usage_ts") or d.get("upload_ts") or 0)
            recency_bonus = max(0, 20 - int(recency / 86400))
            d["score"] = 30 + recency_bonus
            results.append(d)
            seen.add(d["file_id"])

        # Basic ranking
        results = sorted(results, key=lambda x: x.get("score", 0), reverse=True)
        return results[:limit]

    def cleanup_stale(self, keep_days: int = 365) -> int:
        cutoff = int(time.time()) - (keep_days * 86400)
        cur = self.conn.cursor()
        cur.execute("DELETE FROM files WHERE recent_usage_ts < ? AND upload_ts < ?", (cutoff, cutoff))
        removed = cur.rowcount
        self.conn.commit()
        LOGGER.info("cleanup_stale removed=%s", removed)
        return removed


__all__ = ["FileIndex"]
