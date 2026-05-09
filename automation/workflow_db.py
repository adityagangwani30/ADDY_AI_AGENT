"""Workflow persistence layer for Phase 5.

Provides database schema and access layer for workflow state, executions,
and recovery metadata. Optimized for crash-safe and resumable workflows.
"""
import sqlite3
import json
from contextlib import closing
from datetime import datetime
from typing import Dict, List, Any, Optional


def get_conn(db_path: str):
    """Get a database connection with WAL and NORMAL settings."""
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def init_workflow_db(db_path: str):
    """Initialize workflow persistence tables."""
    with closing(get_conn(db_path)) as conn:
        cur = conn.cursor()
        
        # Workflow runs: tracks overall workflow execution state
        cur.execute("""
            CREATE TABLE IF NOT EXISTS workflow_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                workflow_type TEXT NOT NULL,
                workflow_state TEXT NOT NULL,
                current_step INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                planned_steps TEXT,
                execution_context TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                cancel_requested BOOLEAN DEFAULT 0,
                error_message TEXT
            );
        """)
        
        # Workflow steps: individual step execution history
        cur.execute("""
            CREATE TABLE IF NOT EXISTS workflow_steps (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_run_id INTEGER NOT NULL,
                step_number INTEGER NOT NULL,
                tool_name TEXT NOT NULL,
                parameters TEXT,
                status TEXT DEFAULT 'pending',
                result TEXT,
                error TEXT,
                attempts INTEGER DEFAULT 0,
                max_retries INTEGER DEFAULT 2,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                FOREIGN KEY (workflow_run_id) REFERENCES workflow_runs(id)
            );
        """)
        
        # Workflow clarifications: when user input is needed
        cur.execute("""
            CREATE TABLE IF NOT EXISTS workflow_clarifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_run_id INTEGER NOT NULL,
                step_number INTEGER NOT NULL,
                clarification_type TEXT NOT NULL,
                question TEXT NOT NULL,
                options TEXT,
                response TEXT,
                responded_at TEXT,
                created_at TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                FOREIGN KEY (workflow_run_id) REFERENCES workflow_runs(id)
            );
        """)
        
        # Workflow confirmations: for risky actions
        cur.execute("""
            CREATE TABLE IF NOT EXISTS workflow_confirmations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                workflow_run_id INTEGER NOT NULL,
                step_number INTEGER NOT NULL,
                action_description TEXT NOT NULL,
                required BOOLEAN DEFAULT 1,
                confirmed BOOLEAN,
                confirmed_at TEXT,
                created_at TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                FOREIGN KEY (workflow_run_id) REFERENCES workflow_runs(id)
            );
        """)
        
        # Create indexes for performance
        cur.execute("CREATE INDEX IF NOT EXISTS idx_workflow_runs_user ON workflow_runs(user_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_workflow_runs_status ON workflow_runs(status);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_workflow_steps_workflow ON workflow_steps(workflow_run_id);")
        
        conn.commit()


def rows_to_dicts(rows) -> List[Dict[str, Any]]:
    """Convert SQLite rows to dictionaries, decoding JSON fields."""
    out = []
    for r in rows:
        d = dict(r)
        # Decode JSON fields
        for key in ["planned_steps", "execution_context", "parameters", "result", "options"]:
            if key in d and d[key]:
                try:
                    d[key] = json.loads(d[key])
                except Exception:
                    pass
        out.append(d)
    return out


class WorkflowDatabase:
    """High-level workflow persistence API."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        init_workflow_db(db_path)

    def create_workflow_run(
        self,
        user_id: str,
        workflow_type: str,
        workflow_state: Dict[str, Any],
        planned_steps: List[Dict[str, Any]],
    ) -> int:
        """Create a new workflow run."""
        now = datetime.utcnow().isoformat()
        with closing(get_conn(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO workflow_runs 
                   (user_id, workflow_type, workflow_state, status, planned_steps, execution_context, created_at, updated_at) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    user_id,
                    workflow_type,
                    json.dumps(workflow_state),
                    "pending",
                    json.dumps(planned_steps),
                    json.dumps({}),
                    now,
                    now,
                ),
            )
            conn.commit()
            return cur.lastrowid

    def get_workflow_run(self, workflow_run_id: int) -> Optional[Dict[str, Any]]:
        """Get a workflow run by ID."""
        with closing(get_conn(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM workflow_runs WHERE id = ?", (workflow_run_id,))
            row = cur.fetchone()
            if row:
                return rows_to_dicts([row])[0]
        return None

    def list_workflow_runs(self, user_id: str, status: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        """List workflow runs for a user."""
        with closing(get_conn(self.db_path)) as conn:
            cur = conn.cursor()
            if status:
                cur.execute(
                    "SELECT * FROM workflow_runs WHERE user_id = ? AND status = ? ORDER BY created_at DESC LIMIT ?",
                    (user_id, status, limit),
                )
            else:
                cur.execute(
                    "SELECT * FROM workflow_runs WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                    (user_id, limit),
                )
            rows = cur.fetchall()
            return rows_to_dicts(rows)

    def update_workflow_status(
        self,
        workflow_run_id: int,
        status: str,
        current_step: int = None,
        error_message: str = None,
    ) -> None:
        """Update workflow execution status."""
        now = datetime.utcnow().isoformat()
        with closing(get_conn(self.db_path)) as conn:
            cur = conn.cursor()
            if status == "running" and current_step is not None:
                cur.execute(
                    """UPDATE workflow_runs SET status = ?, current_step = ?, updated_at = ?, started_at = COALESCE(started_at, ?)
                       WHERE id = ?""",
                    (status, current_step, now, now, workflow_run_id),
                )
            elif status == "completed":
                cur.execute(
                    "UPDATE workflow_runs SET status = ?, updated_at = ?, completed_at = ? WHERE id = ?",
                    (status, now, now, workflow_run_id),
                )
            elif status == "failed":
                cur.execute(
                    "UPDATE workflow_runs SET status = ?, updated_at = ?, error_message = ?, completed_at = ? WHERE id = ?",
                    (status, now, error_message, now, workflow_run_id),
                )
            else:
                cur.execute(
                    "UPDATE workflow_runs SET status = ?, updated_at = ? WHERE id = ?",
                    (status, now, workflow_run_id),
                )
            conn.commit()

    def add_workflow_step(
        self,
        workflow_run_id: int,
        step_number: int,
        tool_name: str,
        parameters: Dict[str, Any],
        max_retries: int = 2,
    ) -> int:
        """Add a workflow step."""
        now = datetime.utcnow().isoformat()
        with closing(get_conn(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO workflow_steps 
                   (workflow_run_id, step_number, tool_name, parameters, status, max_retries, created_at) 
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    workflow_run_id,
                    step_number,
                    tool_name,
                    json.dumps(parameters),
                    "pending",
                    max_retries,
                    now,
                ),
            )
            conn.commit()
            return cur.lastrowid

    def update_workflow_step(
        self,
        step_id: int,
        status: str,
        result: Dict[str, Any] = None,
        error: str = None,
    ) -> None:
        """Update step execution status."""
        now = datetime.utcnow().isoformat()
        with closing(get_conn(self.db_path)) as conn:
            cur = conn.cursor()
            if status == "completed":
                cur.execute(
                    """UPDATE workflow_steps SET status = ?, result = ?, completed_at = ? 
                       WHERE id = ?""",
                    (status, json.dumps(result) if result else None, now, step_id),
                )
            elif status == "failed":
                cur.execute(
                    """UPDATE workflow_steps SET status = ?, error = ?, completed_at = ? 
                       WHERE id = ?""",
                    (status, error, now, step_id),
                )
            elif status == "running":
                cur.execute(
                    """UPDATE workflow_steps SET status = ?, started_at = ? 
                       WHERE id = ?""",
                    (status, now, step_id),
                )
            else:
                cur.execute(
                    "UPDATE workflow_steps SET status = ? WHERE id = ?",
                    (status, step_id),
                )
            conn.commit()

    def increment_step_attempts(self, step_id: int) -> None:
        """Increment retry attempts for a step."""
        with closing(get_conn(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE workflow_steps SET attempts = attempts + 1 WHERE id = ?",
                (step_id,),
            )
            conn.commit()

    def get_workflow_steps(self, workflow_run_id: int) -> List[Dict[str, Any]]:
        """Get all steps for a workflow."""
        with closing(get_conn(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                "SELECT * FROM workflow_steps WHERE workflow_run_id = ? ORDER BY step_number",
                (workflow_run_id,),
            )
            rows = cur.fetchall()
            return rows_to_dicts(rows)

    def add_clarification(
        self,
        workflow_run_id: int,
        step_number: int,
        clarification_type: str,
        question: str,
        options: List[str] = None,
    ) -> int:
        """Add a clarification request."""
        now = datetime.utcnow().isoformat()
        with closing(get_conn(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO workflow_clarifications 
                   (workflow_run_id, step_number, clarification_type, question, options, status, created_at) 
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    workflow_run_id,
                    step_number,
                    clarification_type,
                    question,
                    json.dumps(options) if options else None,
                    "pending",
                    now,
                ),
            )
            conn.commit()
            return cur.lastrowid

    def respond_to_clarification(self, clarification_id: int, response: str) -> None:
        """Record user response to clarification."""
        now = datetime.utcnow().isoformat()
        with closing(get_conn(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE workflow_clarifications SET response = ?, responded_at = ?, status = 'resolved' WHERE id = ?",
                (response, now, clarification_id),
            )
            conn.commit()

    def add_confirmation(
        self,
        workflow_run_id: int,
        step_number: int,
        action_description: str,
        required: bool = True,
    ) -> int:
        """Add a confirmation requirement."""
        now = datetime.utcnow().isoformat()
        with closing(get_conn(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO workflow_confirmations 
                   (workflow_run_id, step_number, action_description, required, status, created_at) 
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    workflow_run_id,
                    step_number,
                    action_description,
                    required,
                    "pending",
                    now,
                ),
            )
            conn.commit()
            return cur.lastrowid

    def confirm_action(self, confirmation_id: int, confirmed: bool) -> None:
        """Record confirmation decision."""
        now = datetime.utcnow().isoformat()
        with closing(get_conn(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE workflow_confirmations SET confirmed = ?, confirmed_at = ?, status = 'resolved' WHERE id = ?",
                (confirmed, now, confirmation_id),
            )
            conn.commit()

    def cancel_workflow(self, workflow_run_id: int) -> None:
        """Mark a workflow for cancellation."""
        now = datetime.utcnow().isoformat()
        with closing(get_conn(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE workflow_runs SET cancel_requested = 1, updated_at = ? WHERE id = ?",
                (now, workflow_run_id),
            )
            conn.commit()

    def cleanup_old_workflows(self, days: int = 30) -> int:
        """Clean up completed workflows older than specified days."""
        import time
        threshold = datetime.utcfromtimestamp(time.time() - (days * 86400)).isoformat()
        with closing(get_conn(self.db_path)) as conn:
            cur = conn.cursor()
            cur.execute(
                "DELETE FROM workflow_runs WHERE status IN ('completed', 'failed', 'cancelled') AND updated_at < ?",
                (threshold,),
            )
            conn.commit()
            return cur.rowcount
