from __future__ import annotations

import json
import re
from typing import Any

from brain.llm_provider import call_llm
from domain.schemas import IntentRouteResult

ROUTER_PROMPT = """You are an intent router for a personal AI assistant.
Classify the user request and return ONLY strict JSON:
{
  \"intent\": \"gmail_read|gmail_summarize|gmail_draft|gmail_send|calendar_create|calendar_edit|calendar_delete|calendar_list|drive_upload|drive_search|drive_retrieve|drive_share|general_chat|unknown\",
  \"confidence\": 0.0,
  \"parameters\": {}
}
Rules:
- Keep confidence between 0 and 1.
- Never output markdown.
- If intent is unknown/general, return empty parameters.
- For gmail_send/gmail_draft extract recipient, subject, message, format (plain/html).
- For calendar actions extract title/start_time/end_time/time_zone when possible.
- For drive search extract filename.
- Keep output compact.
"""


class IntentRouter:
    def __init__(self, timeout_seconds: int = 12) -> None:
        self.timeout_seconds = timeout_seconds

    def route(self, user_message: str, request_id: str) -> IntentRouteResult:
        quick = self._quick_route(user_message)
        if quick is not None:
            return quick

        text = call_llm(
            prompt=user_message,
            system_prompt=ROUTER_PROMPT,
            request_id=request_id,
            phase="intent_router",
            timeout_seconds=self.timeout_seconds,
        )

        if not text:
            return IntentRouteResult(intent="unknown", confidence=0.0, parameters={})

        parsed = self._safe_parse_json(text)
        if parsed is None:
            return IntentRouteResult(intent="unknown", confidence=0.0, parameters={})

        try:
            return IntentRouteResult.model_validate(parsed)
        except Exception:
            return IntentRouteResult(intent="unknown", confidence=0.0, parameters={})

    def _safe_parse_json(self, raw: str) -> dict[str, Any] | None:
        text = raw.strip()
        fenced = re.search(r"```(?:json)?\\s*(\{.*\})\\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            text = fenced.group(1)

        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        candidate = text[start:end + 1]
        try:
            obj = json.loads(candidate)
        except Exception:
            return None

        return obj if isinstance(obj, dict) else None

    def _quick_route(self, message: str) -> IntentRouteResult | None:
        low = (message or "").strip().lower()
        if not low:
            return IntentRouteResult(intent="unknown", confidence=0.0, parameters={})

        if any(x in low for x in ("summarize emails", "summarise emails", "email summary", "summarize my inbox")):
            return IntentRouteResult(intent="gmail_summarize", confidence=0.82, parameters={})

        if any(x in low for x in ("send mail", "send email", "email to", "mail to")):
            params: dict[str, Any] = {"message": message}
            return IntentRouteResult(intent="gmail_send", confidence=0.76, parameters=params)

        if any(x in low for x in ("draft email", "draft mail", "compose email")):
            return IntentRouteResult(intent="gmail_draft", confidence=0.78, parameters={"message": message})

        if any(x in low for x in ("read email", "list emails", "inbox", "unread emails")):
            return IntentRouteResult(intent="gmail_read", confidence=0.8, parameters={})

        if any(x in low for x in ("create event", "add meeting", "schedule", "calendar")):
            if any(x in low for x in ("delete", "remove", "cancel")):
                return IntentRouteResult(intent="calendar_delete", confidence=0.76, parameters={"title": message})
            if any(x in low for x in ("move", "edit", "reschedule", "update")):
                return IntentRouteResult(intent="calendar_edit", confidence=0.76, parameters={"title": message})
            if any(x in low for x in ("list", "upcoming", "today", "tomorrow")):
                return IntentRouteResult(intent="calendar_list", confidence=0.8, parameters={})
            return IntentRouteResult(intent="calendar_create", confidence=0.72, parameters={"title": message})

        if any(x in low for x in ("drive", "file", "pdf", "document", "upload")):
            if any(x in low for x in ("upload", "put this", "save this")):
                return IntentRouteResult(intent="drive_upload", confidence=0.79, parameters={})
            if any(x in low for x in ("share", "link")):
                return IntentRouteResult(intent="drive_share", confidence=0.72, parameters={"filename": message})
            if any(x in low for x in ("find", "search", "locate")):
                return IntentRouteResult(intent="drive_search", confidence=0.82, parameters={"filename": message})
            if any(x in low for x in ("get", "retrieve", "open")):
                return IntentRouteResult(intent="drive_retrieve", confidence=0.72, parameters={"filename": message})

        if any(x in low for x in ("hi", "hello", "how are you", "what is", "who is", "explain", "help")):
            return IntentRouteResult(intent="general_chat", confidence=0.62, parameters={})

        return None
