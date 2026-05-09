# Personal AI Assistant

> A Telegram-based personal assistant backend powered by **FastAPI**, **Groq** with **NVIDIA** fallback, and the **Google Workspace APIs** for Gmail, Calendar, and Drive.

---

## Overview

This project is a Telegram bot backend that runs as a **FastAPI** webhook service. It uses a hybrid routing layer so simple requests are handled deterministically first, while ambiguous requests fall back to the LLM provider layer in `brain/llm_provider.py`.

The assistant supports multiple Google accounts via OAuth 2.0, stores conversation state and account preferences in SQLite, records executed actions in an audit table, and verifies Telegram webhook requests with a secret token header.

---

## What It Does

| Feature | Description |
|---|---|
| 📩 **Read Emails** | Fetch inbox messages across connected Gmail accounts |
| 🔍 **Search Emails** | Query Gmail with search syntax such as `is:unread` and `from:` |
| 📤 **Send Emails** | Compose and send email with a confirmation gate |
| 🗑️ **Delete Emails** | Permanently remove messages with confirmation |
| 📅 **Calendar Management** | List, create, edit, and delete Google Calendar events |
| 📁 **Google Drive** | List files, upload documents, search, retrieve, share, and delete |
| 🤖 **AI Replies** | General chat routed through Groq with NVIDIA fallback |
| 📱 **Telegram UI** | Text-first Telegram interaction with typing indicators |
| 🔐 **Multi-Account OAuth** | Connect and switch between multiple Google accounts |
| ⚡ **Hybrid Routing** | Deterministic parsing first, LLM fallback second |
| 🧭 **Intent Router** | Structured intent classification for Gmail, Calendar, Drive, and chat |
| 🛠️ **Tool Executor** | Centralized validated dispatch with retries and failures surfaced cleanly |
| ✅ **Reusable Confirmations** | Confirmation workflow with timeout for risky actions |
| 🧾 **Action Audit** | SQLite audit table records executed actions and confirmations |
| 🩺 **Health Check** | `GET /health` reports database, LLM, Telegram, and Google auth status |

---

## Architecture

```text
Telegram User
     │
     ▼
FastAPI Webhook  (api/routes.py)
     │
     ▼
PhaseOneAssistant  (agent/assistant.py)
     │
     ├─── Heuristic Intent Router ──► Direct Tool Execution
     │         (keyword matching)         (no LLM call when possible)
     │
     └─── LLM Fallback ──────────────► Tool Execution
               (Groq primary / NVIDIA fallback)
     │
     ├─── Memory System  (memory/storage.py + memory/memory_manager.py)
     │    SQLite: conversation history, account preferences, aliases, action audit
     │    Phase 2 additions: `memory_entries`, `recent_context`, `entity_aliases`, `user_preferences`.
     │    See `memory/memory_manager.py` for a lightweight deterministic-first API,
     │    alias resolution, and context-aware retrieval used before LLM prompting.
     │
     └─── Auth Manager  (auth/google_auth_manager.py)
          OAuth 2.0 credentials with auto-refresh
```

Key design choices:

- Hybrid routing keeps latency low and avoids unnecessary LLM calls.
- Risky operations require explicit confirmation before execution.
- `ACCOUNTS_JSON` can replace local `accounts.json` in cloud deployments.
- Request IDs are propagated through middleware and structured logs.

---

## Current Scope

This is a stabilization release, not a feature expansion:

- deterministic extraction before LLM fallback for common email, drive, and calendar inputs
- centralized account resolution in `services/account_manager.py`
- atomic account-token writes in `auth/google_auth_manager.py`
- async Telegram API calls via `httpx.AsyncClient`
- `executed_actions` SQLite audit table
- startup environment validation and OAuth health checks

---

## Project Structure

```text
d:\AI Assistant\
├── api/
│   ├── routes.py          # Telegram webhook endpoint and health route
│   └── middleware.py      # Request ID middleware and error handling
├── auth/
│   ├── google_auth_manager.py  # OAuth credential loading, refresh, and persistence
│   └── token_validator.py      # Token refresh and Gmail/Calendar/Drive validation
├── brain/
│   ├── ai_brain.py        # Backward-compatible assistant entrypoint
│   ├── llm_provider.py    # Groq primary / NVIDIA fallback provider
│   └── tool_registry.py   # Tool allowlist and aliases
├── tools/
│   ├── gmail_tools.py     # Gmail API helpers
│   ├── calendar_tools.py  # Calendar API helpers
│   └── drive_tools.py     # Drive API helpers
├── memory/
│   └── storage.py         # SQLite memory, aliases, confirmations, and action audits
├── services/
│   ├── logger.py          # Structured logging and request tracing
│   └── account_manager.py # Shared account resolution logic
├── config/                # Environment loading and validation
├── main.py                # FastAPI app entry point
├── reauth.py              # OAuth token regeneration CLI
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file and set the required values:

| Variable | Description |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Telegram bot token from BotFather |
| `WEBHOOK_SECRET` | Telegram webhook secret token |
| `GROQ_API_KEY` | Primary LLM provider key |
| `NVIDIA_API_KEY` | Fallback LLM provider key |
| `GOOGLE_CLIENT_ID` | Google OAuth client ID |
| `GOOGLE_CLIENT_SECRET` | Google OAuth client secret |
| `GOOGLE_REFRESH_TOKEN` | Bootstrap refresh token used for validation |
| `ACCOUNTS_JSON` | Optional JSON string of stored account credentials |

### 3. Run the server

```bash
uvicorn main:app --reload
```

Register the Telegram webhook with the secret token header:

```text
https://api.telegram.org/bot<YOUR_TOKEN>/setWebhook?url=https://<your-domain>/telegram-webhook&secret_token=<WEBHOOK_SECRET>
```

The health endpoint is available at `GET /health`.

---

## OAuth Regeneration

Use the CLI when you need fresh tokens or want to validate the current account state:

```bash
python reauth.py --check
python reauth.py --email user@example.com
python reauth.py --all
```

What it does:

- regenerates OAuth credentials through the installed-app flow
- overwrites the local account store atomically
- validates Gmail, Calendar, and Drive access together
- prints a clean success or failure summary

If you rotate Google credentials, update `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, and `GOOGLE_REFRESH_TOKEN`, then rerun `python reauth.py --check`.

---

## Deployment

For Render or another cloud host:

1. Set the environment variables above.
2. Use `pip install -r requirements.txt` as the build command.
3. Use `uvicorn main:app --host 0.0.0.0 --port 10000` as the start command.
4. Keep `ACCOUNTS_JSON` in the environment if you do not want to ship local token files.

---

## Security Notes

- Never commit `accounts.json`, `client_secret.json`, `credentials.json`, `token.json`, or `.env`.
- Telegram webhook requests are rejected unless the secret token header matches `WEBHOOK_SECRET`.
- Startup validation fails fast if critical environment variables are missing.
- OAuth refresh errors are surfaced clearly instead of failing silently.
- Destructive tool calls require confirmation before execution.

---

## Health Check

`GET /health` returns structured JSON for:

- database availability
- LLM provider configuration
- Google OAuth health
- Telegram configuration presence

Use it for deployment checks and lightweight monitoring.

---

## Tests

The current baseline includes safety-focused unit tests for:

- intent routing
- confirmation flow and destructive-action protection
- webhook verification
- shared account resolution

Run them with:

```bash
python -m unittest discover -s tests
```

---

## Phase 3: Document Intelligence (Overview)

This release adds a lightweight document intelligence pipeline aimed at enabling
PDF understanding, OCR extraction, semantic file retrieval, and unified search
without introducing external vector databases or heavy infra.

Key components added:

- `services/document_processor.py`: text extraction, simple metadata inference,
     paragraph/page chunking for TXT, DOCX, and PDF.
- `services/ocr_service.py`: optional OCR powered by `pytesseract` and `pdf2image`
     (best-effort; falls back gracefully when binaries are missing).
- `memory/file_index.py`: SQLite-backed file index and alias table with a
     deterministic-first search and lightweight ranking.
- `services/unified_search.py`: combines file index results with existing
     `MemoryManager` searches for a single interface to query files and memories.
- Telegram integration updated in `api/routes.py` to process and index
     uploaded documents for later retrieval.

Design goals:

- Keep memory footprint small and use SQLite for persistence.
- Deterministic search first (aliases, filenames, keywords), LLM fallback
     only when necessary.
- Chunk-aware retrieval to avoid sending whole documents to LLMs.
- Lightweight, free-tier friendly.

## Phase 4 — Automation (Added)

This project now includes a lightweight automation engine designed for Render free-tier deployments. Key features:

- SQLite-backed `scheduled_tasks` persistence (see `automation/db.py`).
- `AutomationEngine` for scheduling, listing, pausing, rescheduling, and cancelling tasks (`automation/automation_engine.py`).
- `TaskRunner` lightweight async runner with configurable polling and graceful shutdown (`automation/task_runner.py`).
- Handler registry for task actions; built-in `send_reminder` handler (`automation/handlers.py`).
- Tests for scheduling and runner under `tests/test_automation.py`.

Quick start (example usage in Python):

```python
from automation.automation_engine import AutomationEngine
from automation.task_runner import TaskRunner
from datetime import datetime, timedelta

engine = AutomationEngine("automation.db")
engine.schedule_task("me", "send_reminder", {"message": "Submit report"}, datetime.utcnow() + timedelta(minutes=5))
runner = TaskRunner("automation.db")
# in production, run runner.start() inside an asyncio loop; for tests use runner.run_once()
```

See docs and comments in the `automation/` package for more details.

## Phase 5 — Intelligent Autonomous Workflows (Added)

This phase transforms the assistant into an **intelligent multi-step autonomous workflow orchestrator**. It enables the assistant to:

- Break complex tasks into multi-step plans
- Execute workflows with dynamic tool chaining
- Handle clarifications and confirmations gracefully
- Recover from failures with retries and fallbacks
- Persist workflow state for resumable execution
- All while remaining lightweight, controllable, and Render free-tier friendly

### Core Components

#### 1. **Task Planner** (`agent/planner.py`)

Breaks user requests into executable multi-step workflow plans.

```python
from agent.planner import TaskPlanner

planner = TaskPlanner()
plan = planner.plan(
    user_intent="Find my resume and send it to HR",
    user_id="user123",
    plan_id="plan-001"
)

# Returns WorkflowPlan with:
# - Step 1: Search Drive for "resume"
# - Step 2: Get file details
# - Step 3: Draft email to HR
# - Step 4: Send email with attachment
```

**Features:**
- Deterministic-first planning (rule-based heuristics before LLM)
- Ambiguity detection with clarification support
- Step dependency tracking for conditional execution
- Complexity estimation (simple / moderate / complex)

#### 2. **Decision Engine** (`agent/decision_engine.py`)

Provides lightweight AI reasoning for:
- Tool selection based on user intent
- Parameter resolution from context and memory
- Ambiguity detection and handling
- Tool chaining and execution ordering
- Fallback strategies for failures

```python
from agent.decision_engine import DecisionEngine

engine = DecisionEngine()

# Tool selection
tools = engine.select_tools(user_intent, context, user_id)

# Parameter resolution
params = engine.resolve_parameters(tool_name, user_intent, context, user_id)

# Ambiguity detection
ambiguities = engine.detect_ambiguities(user_intent, context, user_id)
```

**Design:**
- Rule-based decisions first (70% of use cases)
- Lightweight LLM reasoning only when needed
- Tool chaining templates for common patterns
- Deterministic failure recovery

#### 3. **Workflow Executor** (`agent/workflow_executor.py`)

Executes workflow plans with full state management.

```python
from agent.workflow_executor import WorkflowExecutor

executor = WorkflowExecutor(db_path="workflows.db")

# Execute a plan
status, result = executor.execute(
    plan=plan,
    account="user123",
    request_id="req-123"
)

# status: "completed" | "confirmation_required" | "clarification_required" | "error"
```

**Capabilities:**
- Sequential step execution with timeout protection
- Confirmation gates for risky operations
- Automatic retries for transient failures
- Context passing between steps
- Crash-safe state persistence
- Resumable workflows after user input

#### 4. **Workflow Database** (`automation/workflow_db.py`)

SQLite-backed persistence for workflow state and history.

```
Tables:
- workflow_runs: overall execution state
- workflow_steps: individual step results
- workflow_clarifications: user clarification requests
- workflow_confirmations: confirmation gates
```

**Features:**
- Crash-safe WAL mode
- Resumable workflow recovery
- Audit trail for compliance
- Automatic cleanup of old workflows

#### 5. **Workflow Manager** (`agent/workflow_manager.py`)

Main orchestration layer that coordinates all components.

```python
from agent.workflow_manager import WorkflowManager

manager = WorkflowManager(db_path="workflows.db")

# Start new workflow
result = manager.start_workflow(
    user_intent="Find my resume and send it to HR",
    user_id="user123",
    account="user123"
)

if result["status"] == "confirmation_required":
    # User must confirm before proceeding
    manager.confirm_action(
        workflow_run_id=result["workflow_run_id"],
        confirmation_id=result["confirmation_id"],
        confirmed=True,
        account="user123"
    )

if result["status"] == "clarification_required":
    # User must answer clarifying questions
    for clarification in result["clarifications"]:
        manager.respond_to_clarification(
            workflow_run_id=result["workflow_run_id"],
            clarification_id=clarification["id"],
            response="john@example.com"
        )
```

### Workflow Lifecycle

```
User Intent
    ↓
[Planner] → WorkflowPlan with steps
    ↓
Detect Ambiguities? → [Clarification Flow] → User Response → Refine Plan
    ↓
[Executor] → Execute Steps Sequentially
    ↓
Step Requires Confirmation? → [Confirmation Flow] → User Decision
    ↓
Step Failed? → [Retry Logic] or [Fallback] or [Abort]
    ↓
All Steps Complete? → [Completed] ✅
```

### Safety & Control Mechanisms

**Confirmation Gates:**
```python
# Risky operations always require explicit confirmation:
- gmail_send
- calendar_delete
- calendar_edit
- drive_share
```

**Automatic Safeguards:**
- 5-minute total workflow timeout
- 30-second per-step timeout
- Non-risky operations auto-retry once on failure
- Risky operations never auto-retry
- Workflow cancellation support

**Graceful Degradation:**
```python
# If gmail_send fails:
# → Suggest drafting instead
# → Preserve all context
# → Allow user to intervene

fallback = engine.fallback_strategy(
    failed_tool="gmail_send",
    user_intent="Send email",
    error="SMTP error"
)
# Returns: {"fallback_tool": "gmail_draft", "reason": "..."}
```

### Example Workflows

#### Send Resume to HR
```
User: "Find my resume and send it to HR"

Plan:
1. Search Drive for "resume"
2. Get latest resume file
3. Draft email to HR@company.com
4. [CONFIRM] Send email with attachment

Result: Email sent, memory updated with HR contact
```

#### Prepare for Interview
```
User: "Prepare me for tomorrow's interview"

Plan:
1. Search Calendar for tomorrow's events
2. Find company info in recent emails
3. Retrieve attached job description
4. Create prep document
5. Set reminder for 30 min before

Result: Prep doc + calendar reminder + summary
```

#### Multi-Step Automation
```
User: "Archive old emails and update my task list"

Plan:
1. Search for emails older than 6 months
2. Move to Archive
3. Parse tasks from recent emails
4. [OPTIONAL] Create calendar events for tasks

Result: Archive cleaned, tasks extracted
```

### Integration with Existing Assistant

The Phase 5 workflow system integrates seamlessly with Phase 1-4:

```python
# In agent/assistant.py
from agent.workflow_manager import WorkflowManager

class PhaseOneAssistant:
    def __init__(self):
        # ... existing initialization ...
        self.workflow_manager = WorkflowManager(db_path)

    def run(self, user_message: str, session_id: str, request_id: str):
        # ... existing routing logic ...
        
        # Check if user wants multi-step autonomy
        if self._is_complex_request(user_message):
            result = self.workflow_manager.start_workflow(
                user_intent=user_message,
                user_id=session_id,
                account=account,
            )
            return self._workflow_result_to_agent_result(result)
        
        # Fall through to existing Phase 1-4 logic
        # ...
```

### Performance & Cost

**Render Free-Tier Optimization:**
- Minimal orchestration overhead (mostly Python dictionaries)
- Single SQLite database (no external services)
- Lazy LLM calls (rules first, LLM only when needed)
- No vector databases or embeddings
- Workflow timeouts prevent runaway processes

**Typical Costs:**
- Simple workflow: 1-2 LLM calls (or 0 with rules-only)
- Complex workflow: 2-4 LLM calls
- Execution overhead: <100ms per step
- State persistence: <1KB per workflow run

### Testing

Comprehensive test suite in `tests/test_workflows.py`:

```bash
# Run all Phase 5 tests
python -m unittest tests.test_workflows -v

# Test categories:
# - DecisionEngine (tool selection, ambiguity detection)
# - TaskPlanner (plan generation, complexity estimation)
# - WorkflowDatabase (persistence, recovery)
# - WorkflowExecutor (step execution, confirmations)
# - WorkflowManager (orchestration, error handling)
```

### Logging & Observability

All Phase 5 operations are fully logged:

```python
# Enable DEBUG logging to see workflow details
import logging
logging.getLogger("agent.workflow_executor").setLevel(logging.DEBUG)

# Example log output:
# INFO: Starting workflow execution: plan_id=plan-001
# INFO: Selected 3 tools: ['drive_search', 'gmail_draft', 'gmail_send']
# INFO: Created plan with 4 steps: ...
# INFO: Executing step 1: drive_search
# INFO: Step 1 completed in 234ms
# INFO: Step 3 requires confirmation
# INFO: Workflow completed: 3 steps executed, 0 failed
```

### Future Enhancements

Potential additions (not included in Phase 5):
- Parallel step execution (currently sequential only)
- Conditional branching with if/then logic
- Loop support for iterative workflows
- Workflow templates and reusable chains
- Analytics on workflow patterns and success rates
- User-defined workflow macros

---

## License

This project is for personal use. Not affiliated with Google or Telegram.