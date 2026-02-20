from __future__ import annotations

import asyncio

from brain.ai_brain import run_agent
from domain.schemas import AgentResult, WebhookRequest


async def process_incoming_message(payload: WebhookRequest, request_id: str) -> AgentResult:
    return await asyncio.to_thread(
        run_agent,
        payload.message,
        payload.session_id,
        request_id,
    )
