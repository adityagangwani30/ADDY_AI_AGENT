from __future__ import annotations

import logging

from fastapi import APIRouter, Header, HTTPException, Request

from api.security import verify_webhook_signature
from domain.schemas import AgentResult, WebhookRequest
from services.agent_service import process_incoming_message
from services.logging_service import log_event

router = APIRouter()
LOGGER = logging.getLogger(__name__)


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/webhook", response_model=AgentResult)
async def webhook(
    payload: WebhookRequest,
    request: Request,
    x_signature: str | None = Header(default=None, alias="X-Signature-256"),
) -> AgentResult:
    request_id = str(getattr(request.state, "request_id", "unknown"))
    raw_body = getattr(request.state, "raw_body", b"")

    if not verify_webhook_signature(raw_body, x_signature):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")

    result = await process_incoming_message(payload, request_id=request_id)

    log_event(
        LOGGER,
        logging.INFO,
        event="webhook_processed",
        request_id=request_id,
        tool_name=result.tool_name,
        account=result.account,
        latency_ms=result.latency_ms,
        error_type=result.error_type,
    )
    return result
