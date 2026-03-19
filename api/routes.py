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


def _log(level: int, **payload: object) -> None:
    LOGGER.log(level, json.dumps(payload, default=str))


@router.post("/telegram-webhook")
async def telegram_webhook(request: Request):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    started = time.perf_counter()

    try:
        data = await request.json()
    except Exception:
        return JSONResponse({"ok": True})

    # Ignore non-message updates (like edits, joins, etc.)
    if "message" not in data:
        return JSONResponse({"ok": True})

    message_data = data["message"]
    chat = message_data.get("chat", {})
    chat_id = chat.get("id")
    user_text = message_data.get("text", "")

    if not chat_id or not user_text:
        return JSONResponse({"ok": True})

    _log(
        logging.INFO,
        event="agent_execution_start",
        request_id=request_id,
        tool_name=None,
        latency_ms=None,
        error_type=None,
    )

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                run_agent,
                user_message=user_text,
                session_id=str(chat_id),
                request_id=request_id,
            ),
            timeout=25,
        )

        reply_text = result.message

        latency_ms = int((time.perf_counter() - started) * 1000)

        _log(
            logging.INFO,
            event="agent_execution_finish",
            request_id=request_id,
            tool_name=result.tool_name,
            latency_ms=latency_ms,
            error_type=result.error_type,
        )

    except asyncio.TimeoutError:
        reply_text = "Request timed out. Please try again."
        latency_ms = int((time.perf_counter() - started) * 1000)

        _log(
            logging.ERROR,
            event="agent_execution_timeout",
            request_id=request_id,
            tool_name=None,
            latency_ms=latency_ms,
            error_type="TimeoutError",
        )

    except Exception as exc:
        print(f"ERROR: {exc}")
        traceback.print_exc()

        reply_text = f"Error: {exc}"
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

    # Send reply back to Telegram
    try:
        telegram_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(
            telegram_url,
            json={
                "chat_id": chat_id,
                "text": reply_text,
            },
            timeout=10,
        )
    except Exception:
        pass  # Avoid crashing webhook if Telegram send fails

    return JSONResponse({"ok": True})