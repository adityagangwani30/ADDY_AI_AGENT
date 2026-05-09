from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
import traceback
import uuid
from pathlib import Path

import requests
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from brain.ai_brain import GLOBAL_FALLBACK_RESPONSE, run_agent
from brain.tool_registry import TOOLS
from auth.google_auth_manager import list_available_accounts
from config import TELEGRAM_BOT_TOKEN
from memory.storage import SQLiteMemoryRepository

router = APIRouter()
LOGGER = logging.getLogger(__name__)
MEMORY_REPO = SQLiteMemoryRepository()

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"


# ────────────────────────────────────────────
# Telegram helper functions
# ────────────────────────────────────────────


def _log(level: int, **payload: object) -> None:
    LOGGER.log(level, json.dumps(payload, default=str))


def send_text(chat_id: int, text: str) -> None:
    """Send a plain text message with keyboard removed."""
    try:
        requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text,
                "reply_markup": {"remove_keyboard": True},
            },
            timeout=10,
        )
    except Exception:
        pass


def send_typing(chat_id: int) -> None:
    try:
        requests.post(
            f"{TELEGRAM_API}/sendChatAction",
            json={"chat_id": chat_id, "action": "typing"},
            timeout=10,
        )
    except Exception:
        pass


def _resolve_account_for_session(session_id: str) -> str | None:
    available = list_available_accounts()
    if not available:
        return None

    preferred = MEMORY_REPO.get_account_preference(session_id)
    if preferred and preferred in available:
        return preferred

    aliases = MEMORY_REPO.list_account_aliases()
    mapped = aliases.get("personal")
    if mapped:
        for acc in available:
            if acc.lower() == mapped.lower():
                MEMORY_REPO.set_account_preference(session_id, acc)
                return acc

    MEMORY_REPO.set_account_preference(session_id, available[0])
    return available[0]


def _download_telegram_file(file_id: str) -> Path:
    info = requests.get(f"{TELEGRAM_API}/getFile", params={"file_id": file_id}, timeout=10).json()
    if not info.get("ok"):
        raise RuntimeError("Telegram getFile failed")

    file_path = str((info.get("result") or {}).get("file_path") or "")
    if not file_path:
        raise RuntimeError("Telegram file path missing")

    download_url = f"https://api.telegram.org/file/bot{TELEGRAM_BOT_TOKEN}/{file_path}"
    response = requests.get(download_url, timeout=30)
    response.raise_for_status()

    folder = Path(tempfile.gettempdir()) / "ai-assistant-telegram"
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / Path(file_path).name
    target.write_bytes(response.content)
    return target


def _upload_telegram_document_to_drive(chat_id: int, session_id: str, document_payload: dict[str, object]) -> str:
    file_id = str(document_payload.get("file_id") or "")
    if not file_id:
        return "❌ Could not read the incoming file."

    account = _resolve_account_for_session(session_id)
    if not account:
        return "❌ No connected Google account found for Drive upload."

    local_path: Path | None = None
    try:
        local_path = _download_telegram_file(file_id)
        drive_upload = TOOLS.get("drive_upload")
        if drive_upload is None:
            return "❌ Drive upload tool is not available."
        result = drive_upload(account, file_path=str(local_path))
        return (
            "📂 File uploaded to Drive\n"
            f"Name: {result.get('name', local_path.name)}"
        )
    except Exception as exc:
        return f"❌ Drive upload failed: {exc}"
    finally:
        if local_path and local_path.exists():
            try:
                os.remove(local_path)
            except Exception:
                pass


def send_welcome(chat_id: int) -> None:
    """Send a clean welcome message — no buttons, no menus."""
    welcome = (
        "Hey! 👋 I'm your personal AI assistant.\n\n"
        "Just type what you need — here are some things I can help with:\n\n"
        "📩 *Email* — \"list my emails\", \"send email to X\", \"any urgent emails?\"\n"
        "📅 *Calendar* — \"what meetings do I have today?\", \"create a meeting\"\n"
        "📁 *Drive* — \"list my drive files\"\n"
        "🤖 *Anything else* — \"summarize this\", ask me a question\n\n"
        "Just type naturally and I'll handle the rest!"
    )
    try:
        requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": welcome,
                "parse_mode": "Markdown",
                "reply_markup": {"remove_keyboard": True},
            },
            timeout=10,
        )
    except Exception:
        pass


# ────────────────────────────────────────────
# Webhook route
# ────────────────────────────────────────────


@router.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    started = time.perf_counter()

    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"ok": True})

    # Ignore non-message updates (edits, joins, etc.)
    if "message" not in data:
        return JSONResponse({"ok": True})

    message_data = data["message"]
    chat = message_data.get("chat", {})
    chat_id = chat.get("id")
    user_text = (message_data.get("text") or "").strip()
    document = message_data.get("document")

    if not chat_id:
        return JSONResponse({"ok": True})

    # Telegram file flow: document upload to Drive.
    if isinstance(document, dict):
        send_typing(chat_id)
        reply = _upload_telegram_document_to_drive(chat_id, str(chat_id), document)
        send_text(chat_id, reply)
        return JSONResponse({"ok": True})

    if not user_text:
        return JSONResponse({"ok": True})

    # ── /start command ──────────────────────
    if user_text == "/start":
        send_welcome(chat_id)
        return JSONResponse({"ok": True})

    # ── All input goes straight to the AI agent ──
    _log(
        logging.INFO,
        event="agent_execution_start",
        request_id=request_id,
        tool_name=None,
        latency_ms=None,
        error_type=None,
    )

    send_typing(chat_id)
    reply_text = await _run_agent_safe(user_text, str(chat_id), request_id, started)
    send_text(chat_id, reply_text)
    return JSONResponse({"ok": True})


# ────────────────────────────────────────────
# Agent execution wrapper
# ────────────────────────────────────────────


async def _run_agent_safe(
    user_message: str,
    session_id: str,
    request_id: str,
    started: float,
) -> str:
    """Run the agent in a thread with timeout, returning the reply string."""
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                run_agent,
                user_message=user_message,
                session_id=session_id,
                request_id=request_id,
            ),
            timeout=25,
        )

        latency_ms = int((time.perf_counter() - started) * 1000)
        _log(
            logging.INFO,
            event="agent_execution_finish",
            request_id=request_id,
            tool_name=result.tool_name,
            latency_ms=latency_ms,
            error_type=result.error_type,
        )
        return result.message

    except asyncio.TimeoutError:
        latency_ms = int((time.perf_counter() - started) * 1000)
        _log(
            logging.ERROR,
            event="agent_execution_timeout",
            request_id=request_id,
            tool_name=None,
            latency_ms=latency_ms,
            error_type="TimeoutError",
        )
        return "⏱ Request timed out. Please try again."

    except Exception as exc:
        print(f"ERROR: {exc}")
        traceback.print_exc()

        latency_ms = int((time.perf_counter() - started) * 1000)
        _log(
            logging.ERROR,
            event="agent_execution_error",
            request_id=request_id,
            tool_name=None,
            latency_ms=latency_ms,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        return GLOBAL_FALLBACK_RESPONSE