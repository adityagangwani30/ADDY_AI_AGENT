from __future__ import annotations

import json
import logging
import re
import uuid
from typing import Any

from brain.llm_provider import FALLBACK_MESSAGE, call_llm
from domain.schemas import AgentResult
from memory.storage import SQLiteMemoryRepository
from services.account_manager import resolve_account_for_session

from agent.confirmation import ConfirmationManager
from agent.intent_router import IntentRouter
from agent.tool_executor import ToolExecutor
from services import document_qa as doc_qa
from services import alias_service

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
            result = self._execute_pending(session_id, rid)
            self._audit_confirmation(session_id, "confirm", result.status == "ok", result.message)
            return result
        if reply_type == "cancel":
            self.confirmation.clear(session_id)
            msg = "Cancelled the pending action."
            self._audit_confirmation(session_id, "cancel", True, msg)
            self.memory_repo.add_conversation(session_id, "assistant", msg)
            return AgentResult(request_id=rid, status="ok", message=msg)

        account = resolve_account_for_session(session_id, text=text, memory_repo=self.memory_repo)
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
            answer = self._answer_general(text, session_id, rid)
            self.memory_repo.add_conversation(session_id, "assistant", answer)
            return AgentResult(request_id=rid, status="ok", message=answer)

        if route.intent == "document_qa":
            # run document QA pipeline (deterministic-first)
            answer = doc_qa.answer_question(user_id=session_id, question=text, request_id=rid)
            self.memory_repo.add_conversation(session_id, "assistant", answer)
            return AgentResult(request_id=rid, status="ok", message=answer)

        routed_params = self._enrich_parameters(route.intent, text, route.parameters, rid)

        # alias learning: simple deterministic detection that requests confirmation
        try:
            candidate = alias_service.learn_alias_from_text(text)
            if candidate:
                alias, file_id = candidate
                # save pending alias mapping for confirmation
                self.confirmation.save(session_id, "alias_map", session_id, {"alias": alias, "file_id": file_id})
                preview = f"I detected a reference to '{alias}' — map it to the most recent file? Reply YES to confirm or CANCEL."
                self.memory_repo.add_conversation(session_id, "assistant", preview)
                return AgentResult(request_id=rid, status="confirmation_required", message=preview, tool_name="alias_map")
        except Exception:
            pass

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
            self._audit_confirmation(session_id, "gmail_send_draft", True, "Confirmation requested")
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
            self._audit_confirmation(session_id, route.intent, True, "Confirmation requested")
            preview = self._format_confirmation_preview(route.intent, parameters)
            self.memory_repo.add_conversation(session_id, "assistant", preview)
            return AgentResult(
                request_id=rid,
                status="confirmation_required",
                message=preview,
                tool_name=route.intent,
                account=account,
            )

        result = self.executor.execute(route.intent, account, parameters, rid, session_id=session_id)
        message, status = self._format_tool_result(route.intent, result)
        if route.intent in {"github_repository_summary", "github_project_dashboard", "github_changelog", "github_commits", "github_issues", "github_pull_requests"} and result.get("ok"):
            repository = str(parameters.get("repository") or f"{parameters.get('owner', '')}/{parameters.get('name', '')}").strip()
            if repository and "/" in repository:
                self.memory_repo.set_project_alias(session_id, repository.replace("/", "_"), repository, metadata={"source": route.intent})
                self.memory_repo.set_active_repository(session_id, repository)
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
            extracted = self._extract_email_parameters(user_text)
            if extracted:
                payload.update({key: value for key, value in extracted.items() if value})
                if payload.get("recipient") and payload.get("message"):
                    return payload

        if intent in {"calendar_create", "calendar_edit", "calendar_delete"}:
            extracted = self._extract_calendar_parameters(user_text)
            if extracted:
                payload.update({key: value for key, value in extracted.items() if value})
                if payload.get("title") or payload.get("event_id"):
                    if intent != "calendar_create" or (payload.get("start_time") and payload.get("end_time")):
                        return payload

        if intent in {"drive_upload", "drive_search", "drive_retrieve", "drive_share"}:
            extracted = self._extract_drive_parameters(user_text)
            if extracted:
                payload.update({key: value for key, value in extracted.items() if value})
                if payload.get("filename") or payload.get("file_id") or payload.get("file_path"):
                    return payload

        if intent.startswith("github_"):
            extracted = self._extract_github_parameters(user_text)
            if extracted:
                payload.update({key: value for key, value in extracted.items() if value})
                if payload.get("repository") or payload.get("owner") or payload.get("name") or payload.get("code") or payload.get("traceback_text") or payload.get("changes"):
                    return payload

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

        if intent.startswith("github_"):
            prompt = (
                "Extract JSON for GitHub action. Return only JSON.\n"
                "Possible keys: repository, owner, name, page_size, per_page, state, changes, code, language, traceback_text.\n\n"
                f"User: {user_text}\n"
                f"Existing: {json.dumps(payload, default=str)}"
            )
            text = call_llm(
                prompt=prompt,
                system_prompt="Return strict JSON only.",
                request_id=request_id,
                phase="param_extract_github",
                timeout_seconds=10,
            )
            parsed = self._extract_json(text)
            if parsed:
                payload.update(parsed)
            return payload

        return payload

    def _extract_email_parameters(self, text: str) -> dict[str, str]:
        low = text.lower()
        recipient_match = re.search(r"\b([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\b", text, flags=re.IGNORECASE)
        subject_match = re.search(r"subject\s*[:=]\s*([^\n;]+)", text, flags=re.IGNORECASE)
        body_match = re.search(r"body\s*[:=]\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)

        payload: dict[str, str] = {}
        if recipient_match:
            payload["recipient"] = recipient_match.group(1).strip()
        if subject_match:
            payload["subject"] = subject_match.group(1).strip().strip('"')
        if body_match:
            payload["message"] = body_match.group(1).strip()

        if not payload.get("message"):
            quoted = re.search(r"(?:email|message|body)\s+['\"]([^'\"]{8,})['\"]", text, flags=re.IGNORECASE)
            if quoted:
                payload["message"] = quoted.group(1).strip()

        if "send" in low and not payload.get("subject"):
            short_subject = re.search(r"(?:about|subject|re)\s+([^\n]+)", text, flags=re.IGNORECASE)
            if short_subject:
                payload["subject"] = short_subject.group(1).strip()[:200]

        return payload

    def _extract_calendar_parameters(self, text: str) -> dict[str, str]:
        payload: dict[str, str] = {}
        title_match = re.search(r"(?:titled?|title)\s*[:=]\s*([^\n;]+)", text, flags=re.IGNORECASE)
        start_match = re.search(r"\b(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?)\b", text)
        end_match = re.search(r"\bto\s+(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2})?)\b", text, flags=re.IGNORECASE)
        event_id_match = re.search(r"\bevent\s+id\s*[:=]\s*([A-Za-z0-9_-]+)", text, flags=re.IGNORECASE)

        if title_match:
            payload["title"] = title_match.group(1).strip()
        if start_match:
            payload["start_time"] = start_match.group(1).replace(" ", "T")
        if end_match:
            payload["end_time"] = end_match.group(1).replace(" ", "T")
        if event_id_match:
            payload["event_id"] = event_id_match.group(1).strip()

        return payload

    def _extract_drive_parameters(self, text: str) -> dict[str, str]:
        payload: dict[str, str] = {}
        path_match = re.search(r"(?:file|path)\s*[:=]\s*([^\n;]+)", text, flags=re.IGNORECASE)
        filename_match = re.search(r"(?:named|name)\s+['\"]?([^'\"\n]+)['\"]?", text, flags=re.IGNORECASE)
        file_id_match = re.search(r"\bfile\s+id\s*[:=]\s*([A-Za-z0-9_-]+)", text, flags=re.IGNORECASE)

        if path_match:
            payload["file_path"] = path_match.group(1).strip()
        if filename_match:
            payload["filename"] = filename_match.group(1).strip()
        if file_id_match:
            payload["file_id"] = file_id_match.group(1).strip()

        return payload

    def _extract_github_parameters(self, text: str) -> dict[str, str]:
        payload: dict[str, str] = {}
        repo_match = re.search(r"(?:repo|repository)\s*[:=]\s*([^\n;]+)", text, flags=re.IGNORECASE)
        slug_match = re.search(r"github\.com/([^/\s]+)/([^/#?\s]+)", text, flags=re.IGNORECASE)
        owner_repo_match = re.search(r"\b([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)\b", text)
        code_match = re.search(r"```(?:\w+)?\s*(.*?)```", text, flags=re.DOTALL)
        traceback_match = re.search(r"Traceback \(most recent call last\):.*", text, flags=re.DOTALL | re.IGNORECASE)
        language_match = re.search(r"\b(python|javascript|typescript|fastapi|react|go|java|rust)\b", text, flags=re.IGNORECASE)
        changes_match = re.search(r"(?:changes|summary|diff)\s*[:=]\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)

        if repo_match:
            payload["repository"] = repo_match.group(1).strip()
        elif slug_match:
            payload["repository"] = f"{slug_match.group(1).strip()}/{slug_match.group(2).strip().removesuffix('.git')}"
        elif owner_repo_match and "." not in owner_repo_match.group(1).lower():
            payload["repository"] = f"{owner_repo_match.group(1).strip()}/{owner_repo_match.group(2).strip().removesuffix('.git')}"

        if code_match:
            payload["code"] = code_match.group(1).strip()
        if traceback_match:
            payload["traceback_text"] = traceback_match.group(0).strip()
        if language_match:
            payload["language"] = language_match.group(1).strip().lower()
        if changes_match:
            payload["changes"] = changes_match.group(1).strip()
        elif code_match:
            payload["changes"] = code_match.group(1).strip()
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

        # special-case alias mapping
        if intent == "alias_map":
            alias = str(params.get("alias") or "")
            file_id = str(params.get("file_id") or "")
            try:
                _id = self.memory_repo.set_entity_alias(session_id, alias, file_id, entity_type="file")
                msg = f"Alias '{alias}' mapped to the file." if _id else "Alias mapping saved."
                self.memory_repo.add_conversation(session_id, "assistant", msg)
                return AgentResult(request_id=request_id, status="ok", message=msg, tool_name="alias_map")
            except Exception as exc:
                msg = f"Failed to save alias: {exc}"
                self.memory_repo.add_conversation(session_id, "assistant", msg)
                return AgentResult(request_id=request_id, status="error", message=msg, error_type=type(exc).__name__)

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

    def _audit_confirmation(self, session_id: str, action_type: str, success: bool, message: str) -> None:
        try:
            self.memory_repo.record_executed_action(
                action_type=action_type,
                parameters={"session_id": session_id, "message": message},
                success=success,
                user_id=session_id,
                execution_time_ms=0,
                error_message=None if success else message,
            )
        except Exception:
            pass

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

    def _answer_general(self, text: str, session_id: str, request_id: str) -> str:
        # Memory-aware prompting: fetch relevant short-term context and top memories
        try:
            recent = self.memory_repo.get_recent_context(session_id, limit=6) if hasattr(self.memory_repo, 'get_recent_context') else []
        except Exception:
            recent = []

        try:
            # lightweight search: look up memories that may match the query
            memories = self.memory_repo.search_memory_entries(user_id=session_id, query=text, limit=6) if hasattr(self.memory_repo, 'search_memory_entries') else []
        except Exception:
            memories = []

        context_lines = []
        if memories:
            for m in memories[:4]:
                context_lines.append(f"MEMORY {m.get('category','')}/{m.get('key','')}: {m.get('value')}")
        if recent:
            context_lines.append("RECENT CONVERSATION:")
            for r in recent:
                role = r.get('role')
                msg = r.get('message')
                context_lines.append(f"{role}: {msg}")

        prefix = "".join([l + "\n" for l in context_lines])
        # Keep prompt compact and deterministic-first; send only a short context prefix
        prompt_text = (prefix + "\nUser: " + text) if prefix else text

        response = call_llm(
            prompt=prompt_text,
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
