import sqlite3
import json
from contextlib import closing
from datetime import datetime


def get_conn(db_path: str):
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_db(db_path: str):
    with closing(get_conn(db_path)) as conn:
        cur = conn.cursor()
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS scheduled_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            task_type TEXT NOT NULL,
            task_payload TEXT,
            scheduled_time TEXT,
            recurrence TEXT,
            status TEXT DEFAULT 'scheduled',
            created_at TEXT,
            last_run_at TEXT,
            next_run_at TEXT,
            attempts INTEGER DEFAULT 0
        );
        """
        )
        conn.commit()


def rows_to_dicts(rows):
    out = []
    for r in rows:
        d = dict(r)
        # decode payload if present
        if d.get("task_payload"):
            try:
                d["task_payload"] = json.loads(d["task_payload"])
            except Exception:
                pass
        out.append(d)
    return out
