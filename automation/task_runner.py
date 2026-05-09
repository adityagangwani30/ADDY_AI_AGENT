import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from . import db, handlers

logger = logging.getLogger("automation.runner")


class TaskRunner:
    def __init__(self, db_path: str, polling_interval: float = 30.0):
        self.db_path = db_path
        self.polling_interval = polling_interval
        self._stop_event = asyncio.Event()

    async def start(self):
        logger.info("TaskRunner starting, polling every %ss", self.polling_interval)
        while not self._stop_event.is_set():
            try:
                await self._run_once_async()
            except Exception:
                logger.exception("TaskRunner iteration failed")
            # sleep but wake early if stopped
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=self.polling_interval)
            except asyncio.TimeoutError:
                continue
        logger.info("TaskRunner stopped")

    def stop(self):
        self._stop_event.set()

    async def _run_once_async(self):
        now_iso = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        conn = db.get_conn(self.db_path)
        try:
            cur = conn.cursor()
            cur.execute("BEGIN")
            cur.execute("SELECT * FROM scheduled_tasks WHERE status = 'scheduled' AND next_run_at <= ?", (now_iso,))
            rows = cur.fetchall()
            tasks = db.rows_to_dicts(rows)
            for t in tasks:
                try:
                    logger.info("Executing task id=%s type=%s", t["id"], t["task_type"])
                    # mark running
                    cur.execute("UPDATE scheduled_tasks SET status = 'running' WHERE id = ?", (t["id"],))
                    conn.commit()
                    ok = handlers.run_handler(t["task_type"], t.get("task_payload") or {})
                    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
                    if ok:
                        # update last_run_at and compute next_run
                        next_run = None
                        if t.get("recurrence"):
                            base = datetime.utcnow()
                            # compute next based on recurrence
                            if t["recurrence"].lower() == "daily":
                                next_run = (base + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%S")
                            elif t["recurrence"].lower() == "weekly":
                                next_run = (base + timedelta(weeks=1)).strftime("%Y-%m-%dT%H:%M:%S")
                            else:
                                next_run = None
                        if next_run:
                            cur.execute("UPDATE scheduled_tasks SET last_run_at = ?, next_run_at = ?, status = 'scheduled', attempts = 0 WHERE id = ?", (now, next_run, t["id"]))
                        else:
                            cur.execute("UPDATE scheduled_tasks SET last_run_at = ?, status = 'completed' WHERE id = ?", (now, t["id"]))
                        conn.commit()
                    else:
                        # failure: increment attempts and schedule retry after backoff
                        attempts = (t.get("attempts") or 0) + 1
                        backoff_seconds = min(60 * attempts, 3600)
                        next_run_time = (datetime.utcnow() + timedelta(seconds=backoff_seconds)).strftime("%Y-%m-%dT%H:%M:%S")
                        cur.execute("UPDATE scheduled_tasks SET attempts = ?, next_run_at = ?, status = 'scheduled' WHERE id = ?", (attempts, next_run_time, t["id"]))
                        conn.commit()
                except Exception:
                    logger.exception("Failed to execute task %s", t.get("id"))
                    # on unexpected error, set as failed and schedule retry
                    attempts = (t.get("attempts") or 0) + 1
                    backoff_seconds = min(60 * attempts, 3600)
                    next_run_time = (datetime.utcnow() + timedelta(seconds=backoff_seconds)).strftime("%Y-%m-%dT%H:%M:%S")
                    cur.execute("UPDATE scheduled_tasks SET attempts = ?, next_run_at = ?, status = 'scheduled' WHERE id = ?", (attempts, next_run_time, t.get("id")))
                    conn.commit()
        finally:
            conn.close()

    # helper for tests: run one iteration synchronously
    def run_once(self):
        return asyncio.get_event_loop().run_until_complete(self._run_once_async())
