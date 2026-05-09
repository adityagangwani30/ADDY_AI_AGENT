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
     ├─── Memory System  (memory/storage.py)
     │    SQLite: conversation history, account preferences, aliases, action audit
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

## License

This project is for personal use. Not affiliated with Google or Telegram.