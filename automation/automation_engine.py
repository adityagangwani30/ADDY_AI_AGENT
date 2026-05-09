import json
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from . import db


ISO_FMT = "%Y-%m-%dT%H:%M:%S"


def _now_iso():
    return datetime.utcnow().strftime(ISO_FMT)


def _parse_time(dt):
    if isinstance(dt, datetime):
        return dt
    # assume iso string
    return datetime.strptime(dt, ISO_FMT)


def _compute_next_run(base: datetime, recurrence: Optional[str]):
    if not recurrence:
        return None
    r = recurrence.lower()
    if r == "daily":
        return (base + timedelta(days=1)).strftime(ISO_FMT)
    if r == "weekly":
        return (base + timedelta(weeks=1)).strftime(ISO_FMT)
    if r == "hourly":
        return (base + timedelta(hours=1)).strftime(ISO_FMT)
    # fallback: treat recurrence as ISO delta seconds
    try:
        seconds = int(r)
        return (base + timedelta(seconds=seconds)).strftime(ISO_FMT)
    except Exception:
        return None


class AutomationEngine:
    def __init__(self, db_path: str):
        self.db_path = db_path
        db.init_db(db_path)

    def schedule_task(self, user_id: str, task_type: str, task_payload: Dict[str, Any], scheduled_time: datetime, recurrence: Optional[str] = None) -> int:
        payload_json = json.dumps(task_payload or {})
        scheduled_iso = scheduled_time.strftime(ISO_FMT)
        created = _now_iso()
        next_run = scheduled_iso
        conn = db.get_conn(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO scheduled_tasks (user_id, task_type, task_payload, scheduled_time, recurrence, status, created_at, next_run_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (user_id, task_type, payload_json, scheduled_iso, recurrence, "scheduled", created, next_run),
            )
            conn.commit()
            return cur.lastrowid
        finally:
            conn.close()

    def list_tasks(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        conn = db.get_conn(self.db_path)
        try:
            cur = conn.cursor()
            if user_id:
                cur.execute("SELECT * FROM scheduled_tasks WHERE user_id = ? ORDER BY created_at DESC", (user_id,))
            else:
                cur.execute("SELECT * FROM scheduled_tasks ORDER BY created_at DESC")
            rows = cur.fetchall()
            return db.rows_to_dicts(rows)
        finally:
            conn.close()

    def cancel_task(self, task_id: int) -> None:
        conn = db.get_conn(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute("UPDATE scheduled_tasks SET status = 'cancelled' WHERE id = ?", (task_id,))
            conn.commit()
        finally:
            conn.close()

    def pause_task(self, task_id: int) -> None:
        conn = db.get_conn(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute("UPDATE scheduled_tasks SET status = 'paused' WHERE id = ?", (task_id,))
            conn.commit()
        finally:
            conn.close()

    def resume_task(self, task_id: int) -> None:
        conn = db.get_conn(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute("UPDATE scheduled_tasks SET status = 'scheduled' WHERE id = ?", (task_id,))
            conn.commit()
        finally:
            conn.close()

    def reschedule_task(self, task_id: int, new_time: datetime) -> None:
        conn = db.get_conn(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute("UPDATE scheduled_tasks SET scheduled_time = ?, next_run_at = ?, status = 'scheduled' WHERE id = ?", (new_time.strftime(ISO_FMT), new_time.strftime(ISO_FMT), task_id))
            conn.commit()
        finally:
            conn.close()
