from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
import uuid

import requests
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from brain.ai_brain import run_agent
from config import TELEGRAM_BOT_TOKEN

router = APIRouter()
LOGGER = logging.getLogger(__name__)

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

    if not chat_id or not user_text:
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
        return f"Error: {exc}"