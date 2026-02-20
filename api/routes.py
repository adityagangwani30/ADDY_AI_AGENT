from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid

from fastapi import APIRouter, Form, Request
from fastapi.responses import Response
from twilio.twiml.messaging_response import MessagingResponse

from brain.ai_brain import run_agent

router = APIRouter()
LOGGER = logging.getLogger(__name__)


def _log(level: int, **payload: object) -> None:
    LOGGER.log(level, json.dumps(payload, default=str))


@router.post("/webhook")
async def whatsapp_webhook(
    request: Request,
    Body: str = Form(...),
):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    form_data = await request.form()
    sender = form_data.get("From", "unknown")

    started = time.perf_counter()
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
                user_message=Body,
                session_id=sender,
                request_id=request_id,
            ),
            timeout=20,
        )
        reply = result.message
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
        latency_ms = int((time.perf_counter() - started) * 1000)
        _log(
            logging.ERROR,
            event="agent_execution_timeout",
            request_id=request_id,
            tool_name=None,
            latency_ms=latency_ms,
            error_type="TimeoutError",
        )
        reply = "Request timed out after 20 seconds. Please retry."
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        _log(
            logging.ERROR,
            event="agent_execution_error",
            request_id=request_id,
            tool_name=None,
            latency_ms=latency_ms,
            error_type=type(exc).__name__,
        )
        reply = "Request failed due to a temporary error. Please retry."

    twilio_response = MessagingResponse()
    twilio_response.message(reply)

    return Response(
        content=str(twilio_response),
        media_type="application/xml",
        headers={"X-Request-ID": request_id},
    )
