# Personal AI Assistant

> A Telegram-based personal AI assistant powered by **Gemini**, **FastAPI**, and the **Google Workspace APIs** (Gmail, Calendar, Drive). Manage your email, calendar, and files through natural conversation — all from your Telegram app.

---

## Overview

This project is a production-ready AI assistant that runs as a **Telegram bot** backed by a **FastAPI** webhook server. It uses a **hybrid routing architecture**: simple, pattern-matched requests are executed directly (fast + free), while complex or ambiguous requests are routed through **Google Gemini** for natural language understanding and tool planning.

The assistant supports **multiple Google accounts** via OAuth 2.0, stores conversation history in SQLite, and is designed to deploy on **Render** (or any cloud platform) with zero local file dependency.

---

## Features

| Feature | Description |
|---|---|
| 📩 **Read Emails** | Fetch inbox messages across multiple Gmail accounts |
| 🔍 **Search Emails** | Query Gmail with full search syntax (`is:unread`, `from:`, etc.) |
| 📤 **Send Emails** | Compose and send emails from any connected account |
| 🗑️ **Delete Emails** | Permanently remove messages (with confirmation gate) |
| 📅 **Calendar Management** | List, create, and delete Google Calendar events |
| 📁 **Google Drive** | List files, upload documents, and delete files |
| 🤖 **AI Q&A** | Answer general knowledge questions via Gemini |
| 📱 **Telegram UI** | Intuitive button-menu interface — no commands to memorize |
| 🔐 **Multi-Account OAuth** | Connect and switch between multiple Google accounts |
| ⚡ **Hybrid Agent** | Fast heuristic routing + Gemini fallback, with daily call limits |
| 🧭 **Intent Router** | Structured JSON intent classification for Gmail, Calendar, Drive, and chat |
| 🛠️ **Tool Executor** | Centralized validated tool dispatch with retries and safe failures |
| ✅ **Reusable Confirmations** | Confirmation workflow with timeout for risky actions |
| 📂 **Telegram File Upload** | Upload Telegram documents directly to Google Drive |

---

## Architecture

```
Telegram User
     │
     ▼
FastAPI Webhook  (api/routes.py)
     │
     ▼
SecureHybridAgent  (brain/ai_brain.py)
     │
     ├─── Heuristic Intent Router ──► Direct Tool Execution
     │         (keyword matching)         (no Gemini call)
     │
     └─── Gemini Planner ──────────► Tool Execution
               (LLM tool selection)    (Gmail / Calendar / Drive)
     │
     ├─── Memory System  (memory/storage.py)
     │    SQLite: conversation history, account preferences, aliases
     │
     └─── Auth Manager  (auth/google_auth_manager.py)
          OAuth 2.0 credentials with auto-refresh
```

**Key design decisions:**

- **Hybrid routing** keeps latency low and Gemini API costs minimal — most repetitive commands never make a Gemini call.
- **Tool confirmation gate** — destructive operations (`delete_email`, `delete_file`, `delete_event`) require an explicit `confirm` reply before execution.
- **Daily Gemini call limit** prevents runaway API spend.
- **ENV-first credential loading** — on cloud deployments, `ACCOUNTS_JSON` env var replaces the local `accounts.json` file entirely.

---

## Phase 1 Upgrade Notes

Phase 1 introduces modular action execution while preserving existing bot behavior:

- Intent classification with structured JSON output and malformed-response fallback
- Centralized tool execution with per-intent validation
- Gmail draft-and-confirm send flow (never auto-sends)
- Calendar create/edit/delete/list flow with duplicate prevention support
- Drive search/retrieve/share/upload actions
- Confirmation timeout handling for risky actions
- Improved Telegram UX with typing indicator and clearer success/error messages

New internal module:

```
agent/
├── assistant.py
├── confirmation.py
├── intent_router.py
└── tool_executor.py
```

Backward compatibility is preserved through `brain/ai_brain.py`, which now proxies to the new assistant implementation.

---

## Migration Step (Important)

Drive actions in Phase 1 require write access. The default Drive scope is now:

- `https://www.googleapis.com/auth/drive`

If your existing tokens were generated with read-only Drive scope, run re-authentication once so Google issues updated tokens.

---

## Project Structure

```
d:\AI Assistant\
├── api/
│   ├── routes.py          # FastAPI webhook endpoint + Telegram menu logic
│   └── middleware.py      # Request logging middleware
├── auth/
│   └── google_auth_manager.py  # OAuth credential loading, refresh, and persistence
├── brain/
│   ├── ai_brain.py        # SecureHybridAgent — core agent logic and Gemini integration
│   ├── tool_registry.py   # Tool allowlist, parameter models, destructive tool set
│   └── ...
├── tools/
│   ├── gmail_tools.py     # Gmail API: list, send, search, delete
│   ├── calendar_tools.py  # Calendar API: list events, create, delete
│   ├── drive_tools.py     # Drive API: list files, upload, delete
│   └── __init__.py
├── memory/
│   ├── storage.py         # SQLiteMemoryRepository: history, preferences, aliases
│   └── __init__.py
├── services/
│   └── logging_service.py # Centralized logging configuration
├── domain/
│   └── schemas.py         # Pydantic models: AgentDecision, AgentResult
├── config.py              # Environment variable loading (dotenv)
├── main.py                # FastAPI app entry point
├── requirements.txt
└── .env.example           # Template for required environment variables
```

---

## Setup — Local Development

### 1. Clone and install dependencies

```bash
git clone https://github.com/adityagangwani30/ADDY_AI_AGENT.git
cd ADDY_AI_AGENT
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `TELEGRAM_BOT_TOKEN` | From [@BotFather](https://t.me/BotFather) on Telegram |
| `GEMINI_API_KEY` | From [Google AI Studio](https://aistudio.google.com/) |
| `ACCOUNTS_JSON` | JSON string of OAuth credentials (see below) |
| `GOOGLE_CLIENT_ID` | From your Google Cloud OAuth 2.0 Client |
| `GOOGLE_CLIENT_SECRET` | From your Google Cloud OAuth 2.0 Client |

### 3. Run the server

```bash
uvicorn main:app --reload
```

Then register your webhook with Telegram:
```
https://api.telegram.org/bot<YOUR_TOKEN>/setWebhook?url=https://<your-domain>/telegram-webhook
```

---

## Deployment — Render

1. Push the repository to GitHub.
2. Create a new **Web Service** on [Render](https://render.com) pointing to your repo.
3. Set **Build Command**: `pip install -r requirements.txt`
4. Set **Start Command**: `uvicorn main:app --host 0.0.0.0 --port 10000`
5. Add all environment variables from the table above in Render's **Environment** tab.
6. Deploy — no local files needed. `ACCOUNTS_JSON` replaces `accounts.json` on the server.

---

## Security Notes

> [!CAUTION]
> **Never commit `accounts.json` or `client_secret.json`.** These files contain OAuth tokens that grant full access to your Google accounts. They are excluded in `.gitignore`.

- All sensitive values live in environment variables — never hardcoded.
- OAuth tokens are refreshed automatically; expired tokens raise a clear error rather than silently failing.
- Destructive tool calls require an explicit `confirm` reply to prevent accidental data loss.
- The `ACCOUNTS_JSON` environment variable is used on cloud deployments so credentials are never written to disk.

---

## Future Improvements

- [ ] **Response caching** — cache repeated read-only queries (e.g. inbox list) to reduce API calls.
- [ ] **Rate limiting** — per-user request throttling to prevent abuse.
- [ ] **Multi-user support** — map Telegram `chat_id` to separate Google account sets.
- [ ] **Richer Telegram UI** — inline keyboards with paginated results.
- [ ] **Email body reading** — fetch and summarize full message content, not just metadata.
- [ ] **Webhook security** — validate Telegram secret token header on every request.

---

## License

This project is for personal use. Not affiliated with Google or Telegram.
