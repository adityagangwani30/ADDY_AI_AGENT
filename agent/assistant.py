from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from auth.google_auth_manager import list_available_accounts
from brain.llm_provider import FALLBACK_MESSAGE, call_llm
from domain.schemas import AgentResult
from memory.storage import SQLiteMemoryRepository

from agent.confirmation import ConfirmationManager
from agent.intent_router import IntentRouter
from agent.tool_executor import ToolExecutor

LOGGER = logging.getLogger(__name__)

GENERAL_SYSTEM_PROMPT = (
    "You are a helpful personal AI assistant. Answer clearly and keep replies concise unless asked for details."
)


class PhaseOneAssistant:
    def __init__(self) -> None:
        self.memory_repo = SQLiteMemoryRepository()
        self.router = IntentRouter()
        self.executor = ToolExecutor()
        self.confirmation = ConfirmationManager(self.memory_repo)

    def run(self, user_message: str, session_id: str, request_id: str | None = None) -> AgentResult:
        rid = request_id or str(uuid.uuid4())
        text = (user_message or "").strip()
        if not text:
            return AgentResult(request_id=rid, status="error", message="Message cannot be empty.", error_type="ValidationError")

        self.memory_repo.add_conversation(session_id, "user", text)

        # Confirmation and cancellation handling.
        reply_type = self.confirmation.classify_reply(text)
        if reply_type == "confirm":
            return self._execute_pending(session_id, rid)
        if reply_type == "cancel":
            self.confirmation.clear(session_id)
            msg = "Cancelled the pending action."
            self.memory_repo.add_conversation(session_id, "assistant", msg)
            return AgentResult(request_id=rid, status="ok", message=msg)

        account = self._resolve_active_account(session_id, text)
        if not account:
            msg = "No connected Google account found. Please connect an account first."
            self.memory_repo.add_conversation(session_id, "assistant", msg)
            return AgentResult(request_id=rid, status="error", message=msg, error_type="NoAccount")

        route = self.router.route(text, rid)
        if route.intent in {"unknown"}:
            msg = "I could not map that request to a supported action yet."
            self.memory_repo.add_conversation(session_id, "assistant", msg)
            return AgentResult(request_id=rid, status="error", message=msg, error_type="UnknownIntent")

        if route.intent == "general_chat":
            answer = self._answer_general(text, rid)
            self.memory_repo.add_conversation(session_id, "assistant", answer)
            return AgentResult(request_id=rid, status="ok", message=answer)

        routed_params = self._enrich_parameters(route.intent, text, route.parameters, rid)

        try:
            parameters = self.executor.validate(route.intent, routed_params)
        except Exception as exc:
            msg = f"I need more details to execute that action: {exc}"
            self.memory_repo.add_conversation(session_id, "assistant", msg)
            return AgentResult(request_id=rid, status="error", message=msg, error_type=type(exc).__name__)

        # Gmail send must always be a confirmation flow.
        if route.intent == "gmail_send":
            draft_payload = self._prepare_email_draft(parameters, rid)
            self.confirmation.save(session_id, "gmail_send", account, draft_payload)
            preview = self._format_confirmation_preview("gmail_send", draft_payload)
            self.memory_repo.add_conversation(session_id, "assistant", preview)
            return AgentResult(
                request_id=rid,
                status="confirmation_required",
                message=preview,
                tool_name="gmail_send",
                account=account,
            )

        if route.intent in self.executor.risky_intents:
            self.confirmation.save(session_id, route.intent, account, parameters)
            preview = self._format_confirmation_preview(route.intent, parameters)
            self.memory_repo.add_conversation(session_id, "assistant", preview)
            return AgentResult(
                request_id=rid,
                status="confirmation_required",
                message=preview,
                tool_name=route.intent,
                account=account,
            )

        result = self.executor.execute(route.intent, account, parameters, rid)
        message, status = self._format_tool_result(route.intent, result)
        self.memory_repo.add_conversation(session_id, "assistant", message)
        return AgentResult(
            request_id=rid,
            status=status,
            message=message,
            tool_name=route.intent,
            account=account,
            latency_ms=int(result.get("latency_ms") or 0),
            data=result.get("result"),
            error_type=None if result.get("ok") else "ToolError",
        )

    def _resolve_active_account(self, session_id: str, text: str) -> str | None:
        available = list_available_accounts()
        if not available:
            return None

        aliases = self.memory_repo.list_account_aliases()
        low = text.lower()
        for alias, mapped in aliases.items():
            if alias in low:
                for acc in available:
                    if acc.lower() == mapped.lower():
                        self.memory_repo.set_account_preference(session_id, acc)
                        return acc

        preferred = self.memory_repo.get_account_preference(session_id)
        if preferred and preferred in available:
            return preferred

        default_alias = aliases.get("personal")
        if default_alias:
            for acc in available:
                if acc.lower() == default_alias.lower():
                    self.memory_repo.set_account_preference(session_id, acc)
                    return acc

        fallback = available[0]
        self.memory_repo.set_account_preference(session_id, fallback)
        return fallback

    def _prepare_email_draft(self, params: dict[str, Any], request_id: str) -> dict[str, Any]:
        recipient = str(params.get("recipient") or "").strip()
        subject = str(params.get("subject") or "").strip()
        message = str(params.get("message") or "").strip()
        fmt = str(params.get("format") or "plain").lower()
        if fmt not in {"plain", "html"}:
            fmt = "plain"

        if not subject:
            generated = call_llm(
                prompt=f"Generate a short email subject for this message:\n\n{message}",
                system_prompt="Return only a concise email subject line.",
                request_id=request_id,
                phase="email_subject",
                timeout_seconds=8,
            )
            subject = (generated or "No subject").strip().splitlines()[0][:200]

        return {
            "recipient": recipient,
            "subject": subject,
            "message": message,
            "format": fmt,
        }

    def _enrich_parameters(self, intent: str, user_text: str, params: dict[str, Any], request_id: str) -> dict[str, Any]:
        payload = dict(params or {})

        if intent in {"gmail_send", "gmail_draft"}:
            if payload.get("recipient") and payload.get("message"):
                return payload
            prompt = (
                "Extract JSON for email action. Return only JSON with keys "
                "recipient, subject, message, format(plain|html).\n\n"
                f"User: {user_text}\n"
                f"Existing: {json.dumps(payload, default=str)}"
            )
            text = call_llm(
                prompt=prompt,
                system_prompt="Return strict JSON only.",
                request_id=request_id,
                phase="param_extract_email",
                timeout_seconds=10,
            )
            parsed = self._extract_json(text)
            if parsed:
                payload.update(parsed)
            if not payload.get("message"):
                payload["message"] = user_text
            return payload

        if intent in {"calendar_create", "calendar_edit", "calendar_delete"}:
            prompt = (
                "Extract JSON for calendar action. Return only JSON.\n"
                "For create/edit include: title, start_time, end_time, time_zone, event_id, date_hint.\n"
                "Use ISO 8601 datetimes when possible.\n\n"
                f"User: {user_text}\n"
                f"Existing: {json.dumps(payload, default=str)}"
            )
            text = call_llm(
                prompt=prompt,
                system_prompt="Return strict JSON only.",
                request_id=request_id,
                phase="param_extract_calendar",
                timeout_seconds=10,
            )
            parsed = self._extract_json(text)
            if parsed:
                payload.update(parsed)
            return payload

        if intent in {"drive_upload", "drive_search", "drive_retrieve", "drive_share"}:
            prompt = (
                "Extract JSON for drive action. Return only JSON.\n"
                "Possible keys: file_path, filename, file_id, max_results, overwrite.\n\n"
                f"User: {user_text}\n"
                f"Existing: {json.dumps(payload, default=str)}"
            )
            text = call_llm(
                prompt=prompt,
                system_prompt="Return strict JSON only.",
                request_id=request_id,
                phase="param_extract_drive",
                timeout_seconds=10,
            )
            parsed = self._extract_json(text)
            if parsed:
                payload.update(parsed)
            return payload

        return payload

    def _extract_json(self, raw: str | None) -> dict[str, Any] | None:
        if not raw:
            return None
        text = raw.strip()
        if not text:
            return None
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            obj = json.loads(text[start:end + 1])
        except Exception:
            return None
        return obj if isinstance(obj, dict) else None

    def _format_confirmation_preview(self, intent: str, params: dict[str, Any]) -> str:
        if intent == "gmail_send":
            body_preview = str(params.get("message") or "")
            if len(body_preview) > 600:
                body_preview = body_preview[:600] + "..."
            return (
                "📧 Draft ready for review\n\n"
                f"To: {params.get('recipient', '')}\n"
                f"Subject: {params.get('subject', '')}\n"
                f"Format: {params.get('format', 'plain')}\n\n"
                "Body preview:\n"
                f"{body_preview}\n\n"
                "Reply YES/CONFIRM to send, or CANCEL."
            )

        if intent == "calendar_delete":
            return (
                "⚠️ Event deletion requires confirmation\n\n"
                f"Target: {json.dumps(params, default=str)}\n\n"
                "Reply YES/CONFIRM to continue, or CANCEL."
            )

        return (
            "⚠️ Action requires confirmation\n\n"
            f"{intent}: {json.dumps(params, default=str)}\n\n"
            "Reply YES/CONFIRM to continue, or CANCEL."
        )

    def _execute_pending(self, session_id: str, request_id: str) -> AgentResult:
        pending = self.confirmation.get_valid_pending(session_id)
        if not pending:
            return AgentResult(
                request_id=request_id,
                status="error",
                message="No valid pending action found. It may have expired.",
                error_type="NoPendingConfirmation",
            )

        self.confirmation.clear(session_id)
        intent = str(pending.get("tool_name") or "")
        account = str(pending.get("account") or "")
        params = dict(pending.get("parameters") or {})

        result = self.executor.execute(intent, account, params, request_id)
        message, status = self._format_tool_result(intent, result)
        self.memory_repo.add_conversation(session_id, "assistant", message)
        return AgentResult(
            request_id=request_id,
            status=status,
            message=message,
            tool_name=intent,
            account=account,
            latency_ms=int(result.get("latency_ms") or 0),
            data=result.get("result"),
            error_type=None if result.get("ok") else "ToolError",
        )

    def _format_tool_result(self, intent: str, result: dict[str, Any]) -> tuple[str, str]:
        if not result.get("ok"):
            return (f"❌ {result.get('error') or FALLBACK_MESSAGE}", "error")

        payload = result.get("result")
        if intent == "gmail_summarize" and isinstance(payload, dict):
            return (f"📧 Inbox summary\n\n{payload.get('summary', '')}", "ok")

        if intent == "gmail_read" and isinstance(payload, dict):
            msgs = payload.get("messages", [])
            if not msgs:
                return ("📭 No emails found.", "ok")
            lines: list[str] = ["📬 Latest emails"]
            for idx, msg in enumerate(msgs[:8], start=1):
                lines.append(f"{idx}. {msg.get('subject', 'No subject')} - {msg.get('from', 'Unknown')}")
            return ("\n".join(lines), "ok")

        if intent == "gmail_send":
            return ("✅ Email sent successfully.", "ok")
        if intent == "calendar_create":
            return ("✅ Event created successfully.", "ok")
        if intent == "calendar_edit":
            return ("✅ Event updated successfully.", "ok")
        if intent == "calendar_delete":
            return ("✅ Event deleted successfully.", "ok")
        if intent == "calendar_list" and isinstance(payload, dict):
            events = payload.get("events", [])
            if not events:
                return ("📅 No upcoming events found.", "ok")
            lines = ["📅 Upcoming schedule"]
            for idx, event in enumerate(events[:10], start=1):
                start = event.get("start", {}).get("dateTime") or event.get("start", {}).get("date") or ""
                lines.append(f"{idx}. {event.get('summary', 'Untitled')} at {start}")
            return ("\n".join(lines), "ok")

        if intent == "drive_upload":
            return ("📂 File uploaded to Drive.", "ok")
        if intent == "drive_search" and isinstance(payload, dict):
            files = payload.get("files", [])
            if not files:
                return ("📂 No matching files found.", "ok")
            lines = ["📂 Matching files"]
            for idx, item in enumerate(files[:10], start=1):
                lines.append(f"{idx}. {item.get('name', 'Unnamed')}")
            return ("\n".join(lines), "ok")
        if intent == "drive_retrieve" and isinstance(payload, dict):
            file_meta = payload.get("file", {})
            return (
                "📄 File details\n"
                f"Name: {file_meta.get('name', '')}\n"
                f"Link: {file_meta.get('webViewLink', 'N/A')}",
                "ok",
            )
        if intent == "drive_share" and isinstance(payload, dict):
            file_meta = payload.get("file", {})
            return (
                "🔗 Share link ready\n"
                f"Name: {file_meta.get('name', '')}\n"
                f"Link: {file_meta.get('webViewLink', 'N/A')}",
                "ok",
            )

        return ("✅ Action completed successfully.", "ok")

    def _answer_general(self, text: str, request_id: str) -> str:
        response = call_llm(
            prompt=text,
            system_prompt=GENERAL_SYSTEM_PROMPT,
            request_id=request_id,
            phase="general_chat",
            timeout_seconds=12,
        )
        return (response or FALLBACK_MESSAGE).strip()


_AGENT = PhaseOneAssistant()


def run_agent(user_message: str, session_id: str, request_id: str | None = None) -> AgentResult:
    rid = request_id or str(uuid.uuid4())
    try:
        return _AGENT.run(user_message, session_id, rid)
    except Exception as exc:
        LOGGER.exception("agent_run_failed", extra={"request_id": rid})
        return AgentResult(
            request_id=rid,
            status="error",
            message=FALLBACK_MESSAGE,
            error_type=type(exc).__name__,
        )
