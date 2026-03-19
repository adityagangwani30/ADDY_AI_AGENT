from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback
import uuid
from typing import Any

import requests
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from brain.ai_brain import run_agent
from config import TELEGRAM_BOT_TOKEN

router = APIRouter()
LOGGER = logging.getLogger(__name__)

TELEGRAM_API = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

# ────────────────────────────────────────────
# Account display labels → internal aliases
# ────────────────────────────────────────────

_ACCOUNTS = [
    {"emoji": "🎓", "label": "College",  "email": "1ms23ec007@msrit.edu",           "alias": "college"},
    {"emoji": "📧", "label": "Personal", "email": "adityabvbvpn0011@gmail.com",     "alias": "personal"},
    {"emoji": "🧪", "label": "Exam",     "email": "adityagangwaniexam@gmail.com",   "alias": "exam"},
    {"emoji": "🔒", "label": "Private",  "email": "ashgangcr7@gmail.com",           "alias": "private"},
]

# Build button-text → alias map at import time
_ACCOUNT_BUTTON_MAP: dict[str, str] = {}
for _acct in _ACCOUNTS:
    _btn = f"{_acct['emoji']} {_acct['label']} ({_acct['email']})"
    _ACCOUNT_BUTTON_MAP[_btn] = _acct["alias"]

# ────────────────────────────────────────────
# Menu state tracking (in-memory, per chat)
# ────────────────────────────────────────────

_MENU_STATE: dict[int, str] = {}  # chat_id → "email" | "calendar" | "drive" | "main"

# ────────────────────────────────────────────
# Menu button labels (constants)
# ────────────────────────────────────────────

BTN_EMAIL    = "📩 Email"
BTN_CALENDAR = "📅 Calendar"
BTN_DRIVE    = "📁 Drive"
BTN_ASK      = "🤖 Ask Anything"
BTN_BACK     = "⬅️ Back"

# ────────────────────────────────────────────
# Telegram helper functions
# ────────────────────────────────────────────


def _log(level: int, **payload: object) -> None:
    LOGGER.log(level, json.dumps(payload, default=str))


def send_text(chat_id: int, text: str) -> None:
    """Send a plain text message (no keyboard change)."""
    try:
        requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={"chat_id": chat_id, "text": text},
            timeout=10,
        )
    except Exception:
        pass


def _send_keyboard(chat_id: int, text: str, keyboard: list[list[str]]) -> None:
    """Send a message with a ReplyKeyboardMarkup."""
    try:
        requests.post(
            f"{TELEGRAM_API}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text,
                "reply_markup": {
                    "keyboard": keyboard,
                    "resize_keyboard": True,
                    "one_time_keyboard": False,
                },
            },
            timeout=10,
        )
    except Exception:
        pass


def send_main_menu(chat_id: int) -> None:
    """Show the top-level main menu."""
    _MENU_STATE[chat_id] = "main"
    _send_keyboard(
        chat_id,
        "Hey! 👋 What would you like to do?",
        [
            [BTN_EMAIL, BTN_CALENDAR],
            [BTN_DRIVE, BTN_ASK],
        ],
    )


def send_email_menu(chat_id: int) -> None:
    """Show the email account selection menu."""
    _MENU_STATE[chat_id] = "email"
    buttons = [[btn_text] for btn_text in _ACCOUNT_BUTTON_MAP]
    buttons.append([BTN_BACK])
    _send_keyboard(chat_id, "📩 Select an email account:", buttons)


def send_calendar_menu(chat_id: int) -> None:
    """Show the calendar account selection menu."""
    _MENU_STATE[chat_id] = "calendar"
    buttons = [[btn_text] for btn_text in _ACCOUNT_BUTTON_MAP]
    buttons.append([BTN_BACK])
    _send_keyboard(chat_id, "📅 Select a calendar account:", buttons)


def send_drive_menu(chat_id: int) -> None:
    """Show the drive account selection menu."""
    _MENU_STATE[chat_id] = "drive"
    buttons = [[btn_text] for btn_text in _ACCOUNT_BUTTON_MAP]
    buttons.append([BTN_BACK])
    _send_keyboard(chat_id, "📁 Select a Drive account:", buttons)


# ────────────────────────────────────────────
# Agent action per-menu context
# ────────────────────────────────────────────

_MENU_ACTION_PHRASES: dict[str, str] = {
    "email":    "list my emails from {alias}",
    "calendar": "show my calendar events from {alias}",
    "drive":    "list my files from {alias}",
}

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
        send_main_menu(chat_id)
        return JSONResponse({"ok": True})

    # ── Back button ─────────────────────────
    if user_text == BTN_BACK:
        send_main_menu(chat_id)
        return JSONResponse({"ok": True})

    # ── Main menu clicks ────────────────────
    if user_text == BTN_EMAIL:
        send_email_menu(chat_id)
        return JSONResponse({"ok": True})

    if user_text == BTN_CALENDAR:
        send_calendar_menu(chat_id)
        return JSONResponse({"ok": True})

    if user_text == BTN_DRIVE:
        send_drive_menu(chat_id)
        return JSONResponse({"ok": True})

    if user_text == BTN_ASK:
        send_text(chat_id, "🤖 Go ahead — type your question and I'll answer it!")
        _MENU_STATE[chat_id] = "main"
        return JSONResponse({"ok": True})

    # ── Account selection (from sub-menus) ──
    alias = _ACCOUNT_BUTTON_MAP.get(user_text)
    menu_context = _MENU_STATE.get(chat_id, "main")

    if alias and menu_context in _MENU_ACTION_PHRASES:
        # Build a natural-language command for the agent
        agent_message = _MENU_ACTION_PHRASES[menu_context].format(alias=alias)
        reply_text = await _run_agent_safe(agent_message, str(chat_id), request_id, started)
        send_text(chat_id, reply_text)
        return JSONResponse({"ok": True})

    # ── Fallback: pass free-text to AI agent ─
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