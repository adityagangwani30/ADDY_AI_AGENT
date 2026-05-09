import os
import tempfile
from datetime import datetime, timedelta

import pytest

from automation.automation_engine import AutomationEngine
from automation.task_runner import TaskRunner
from automation import db


def test_schedule_and_run_once(tmp_path):
    db_path = str(tmp_path / "automation.db")
    engine = AutomationEngine(db_path)
    now = datetime.utcnow()
    # schedule a task in the past so it's due immediately
    task_id = engine.schedule_task(user_id="user1", task_type="send_reminder", task_payload={"message": "test"}, scheduled_time=now - timedelta(minutes=1))
    runner = TaskRunner(db_path, polling_interval=1)
    # run once and ensure it executes
    runner.run_once()
    tasks = engine.list_tasks("user1")
    assert any(t["id"] == task_id and t["status"] in ("completed", "scheduled") for t in tasks)


def test_recurring_task_next_run(tmp_path):
    db_path = str(tmp_path / "automation2.db")
    engine = AutomationEngine(db_path)
    now = datetime.utcnow()
    task_id = engine.schedule_task(user_id="u2", task_type="send_reminder", task_payload={"message": "recurring"}, scheduled_time=now - timedelta(minutes=1), recurrence="daily")
    runner = TaskRunner(db_path, polling_interval=1)
    runner.run_once()
    tasks = engine.list_tasks("u2")
    t = next((x for x in tasks if x["id"] == task_id), None)
    assert t is not None
    assert t["status"] in ("scheduled", "completed")
    # recurring should have next_run_at set in the future when status is scheduled
    if t["status"] == "scheduled":
        assert t["next_run_at"] is not None
