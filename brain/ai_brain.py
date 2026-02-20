from __future__ import annotations

import inspect
import json
import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any

from google import genai
from google.genai import types
from pydantic import ValidationError

from auth.google_auth_manager import list_available_accounts
from brain.tool_registry import DESTRUCTIVE_TOOLS, TOOLS, TOOL_PARAMETER_MODELS
from config import GEMINI_API_KEY
from domain.schemas import AgentDecision, AgentResult
from memory.storage import SQLiteMemoryRepository

LOGGER = logging.getLogger(__name__)

_GEMINI_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="agent-gemini")
_TOOL_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="agent-tools")

GEMINI_MODELS = ["gemini-2.5-flash", "gemini-1.5-flash"]
FALLBACK_MODEL_MESSAGE = "I'm having trouble accessing my reasoning model right now. Please try again."

CLASSIFICATION_TIMEOUT_SECONDS = 3
GEMINI_ANSWER_TIMEOUT_SECONDS = 8
TOOL_TIMEOUT_SECONDS = 8
MAX_HISTORY_MESSAGES = 6

INTENTS = {
    "general_knowledge",
    "gmail_action",
    "calendar_action",
    "drive_action",
    "multi_step",
}


def _log(level: int, **payload: Any) -> None:
    LOGGER.log(level, json.dumps(payload, default=str))


def execute_tool(
    tool_name: str,
    account: str,
    parameters: dict[str, Any],
    request_id: str,
) -> dict[str, Any]:
    if tool_name not in TOOLS:
        raise ValueError(f"Tool '{tool_name}' is not registered.")

    started = time.perf_counter()
    tool_fn = TOOLS[tool_name]
    call_kwargs = dict(parameters)

    signature = inspect.signature(tool_fn)
    if "request_id" in signature.parameters:
        call_kwargs["request_id"] = request_id

    _log(
        logging.INFO,
        event="tool_start",
        request_id=request_id,
        tool_name=tool_name,
        latency_ms=None,
        error_type=None,
    )

    future = _TOOL_EXECUTOR.submit(tool_fn, account, **call_kwargs)
    try:
        result = future.result(timeout=TOOL_TIMEOUT_SECONDS)
    except FuturesTimeoutError as exc:
        future.cancel()
        latency_ms = int((time.perf_counter() - started) * 1000)
        _log(
            logging.ERROR,
            event="tool_timeout",
            request_id=request_id,
            tool_name=tool_name,
            latency_ms=latency_ms,
            error_type="TimeoutError",
        )
        raise TimeoutError("That request is taking longer than expected. Please try again.") from exc
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        _log(
            logging.ERROR,
            event="tool_error",
            request_id=request_id,
            tool_name=tool_name,
            latency_ms=latency_ms,
            error_type=type(exc).__name__,
        )
        raise RuntimeError(f"Tool '{tool_name}' failed: {exc}") from exc

    latency_ms = int((time.perf_counter() - started) * 1000)
    _log(
        logging.INFO,
        event="tool_finish",
        request_id=request_id,
        tool_name=tool_name,
        latency_ms=latency_ms,
        error_type=None,
    )
    return {"latency_ms": latency_ms, "result": result}


class SecureHybridAgent:
    """
    Hybrid architecture:
    1) Lightweight intent classification
    2) Direct Gemini answer for general knowledge
    3) Tool planning + execution only when needed
    """

    def __init__(self) -> None:
        self.memory_repo = SQLiteMemoryRepository()
        self._genai_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

    def run(self, user_message: str, session_id: str, request_id: str | None = None) -> AgentResult:
        rid = request_id or str(uuid.uuid4())
        started = time.perf_counter()
        cleaned = user_message.strip()

        _log(
            logging.INFO,
            event="agent_start",
            request_id=rid,
            tool_name=None,
            latency_ms=None,
            error_type=None,
        )

        if not cleaned:
            return AgentResult(request_id=rid, status="error", message="Message cannot be empty.")

        self.memory_repo.add_conversation(session_id, "user", cleaned)
        lower = cleaned.lower()

        if self._is_accounts_question(lower):
            message = self._format_accounts_overview()
            self.memory_repo.add_conversation(session_id, "assistant", message)
            return AgentResult(request_id=rid, status="ok", message=message)

        if lower in {"confirm", "yes confirm", "confirm action"}:
            return self._handle_confirmation(session_id, rid)

        if lower in {"cancel", "deny", "reject"}:
            self.memory_repo.clear_pending_confirmation(session_id)
            return AgentResult(request_id=rid, status="ok", message="Pending action cancelled.")

        history = self.memory_repo.get_conversation(session_id, limit=MAX_HISTORY_MESSAGES)

        alias_in_message = self._detect_account_alias(lower)
        if alias_in_message:
            resolved_alias_account = self._resolve_account_identifier(alias_in_message)
            if resolved_alias_account:
                self.memory_repo.set_account_preference(session_id, resolved_alias_account)

        intent = self._classify_intent(cleaned, history, rid)

        if intent == "general_knowledge":
            answer = self._answer_general(cleaned, history, rid)
            self.memory_repo.add_conversation(session_id, "assistant", answer)
            self._log_agent_finish(rid, started, None, None)
            return AgentResult(request_id=rid, status="ok", message=answer)

        preferred_account = self.memory_repo.get_account_preference(session_id)
        decision = self._plan_tool_decision(cleaned, history, preferred_account, intent, rid)

        if decision is None:
            answer = self._answer_general(cleaned, history, rid)
            self.memory_repo.add_conversation(session_id, "assistant", answer)
            self._log_agent_finish(rid, started, None, None)
            return AgentResult(request_id=rid, status="ok", message=answer)

        if decision.type == "response":
            text = decision.response or self._answer_general(cleaned, history, rid)
            self.memory_repo.add_conversation(session_id, "assistant", text)
            self._log_agent_finish(rid, started, None, None)
            return AgentResult(request_id=rid, status="ok", message=text)

        tool_name = decision.tool or ""
        raw_account = decision.account or preferred_account or ""
        account = self._resolve_account_identifier(raw_account)

        if not account:
            text = "I could not map that request to a connected account. Ask 'what accounts do I have?'"
            self.memory_repo.add_conversation(session_id, "assistant", text)
            self._log_agent_finish(rid, started, tool_name, "AccountResolutionError")
            return AgentResult(request_id=rid, status="error", message=text, error_type="AccountResolutionError")

        try:
            validated = self._validate_tool_call(tool_name, account, decision.parameters)
        except ValueError as exc:
            self._log_agent_finish(rid, started, tool_name, type(exc).__name__)
            return AgentResult(request_id=rid, status="error", message=str(exc), error_type=type(exc).__name__)

        self.memory_repo.set_account_preference(session_id, account)

        if tool_name in DESTRUCTIVE_TOOLS:
            self.memory_repo.save_pending_confirmation(
                session_id=session_id,
                tool_name=tool_name,
                account=account,
                parameters=validated,
            )
            confirmation_message = (
                f"Confirmation required for '{tool_name}' on '{account}'. Reply 'confirm' to continue or 'cancel' to abort."
            )
            self.memory_repo.add_conversation(session_id, "assistant", confirmation_message)
            self._log_agent_finish(rid, started, tool_name, None)
            return AgentResult(
                request_id=rid,
                status="confirmation_required",
                message=confirmation_message,
                tool_name=tool_name,
                account=account,
            )

        result = self._execute_and_build_result(
            session_id=session_id,
            request_id=rid,
            tool_name=tool_name,
            account=account,
            parameters=validated,
            user_message=cleaned,
        )
        self._log_agent_finish(rid, started, tool_name, result.error_type)
        return result

    def _handle_confirmation(self, session_id: str, request_id: str) -> AgentResult:
        pending = self.memory_repo.get_pending_confirmation(session_id)
        if not pending:
            return AgentResult(
                request_id=request_id,
                status="error",
                message="No pending action to confirm.",
                error_type="NoPendingConfirmation",
            )

        self.memory_repo.clear_pending_confirmation(session_id)
        return self._execute_and_build_result(
            session_id=session_id,
            request_id=request_id,
            tool_name=str(pending["tool_name"]),
            account=str(pending["account"]),
            parameters=dict(pending["parameters"]),
            user_message="confirmed action",
        )

    def _execute_and_build_result(
        self,
        session_id: str,
        request_id: str,
        tool_name: str,
        account: str,
        parameters: dict[str, Any],
        user_message: str,
    ) -> AgentResult:
        try:
            tool_response = execute_tool(
                tool_name=tool_name,
                account=account,
                parameters=parameters,
                request_id=request_id,
            )
        except TimeoutError:
            message = "That request is taking longer than expected. Please try again."
            self.memory_repo.add_conversation(session_id, "assistant", message)
            return AgentResult(
                request_id=request_id,
                status="error",
                message=message,
                tool_name=tool_name,
                account=account,
                error_type="TimeoutError",
            )
        except Exception as exc:
            message = str(exc)
            self.memory_repo.add_conversation(session_id, "assistant", message)
            return AgentResult(
                request_id=request_id,
                status="error",
                message=message,
                tool_name=tool_name,
                account=account,
                error_type=type(exc).__name__,
            )

        summary = self._summarize_tool_result(
            user_message=user_message,
            tool_name=tool_name,
            tool_result=tool_response["result"],
            request_id=request_id,
        )

        self.memory_repo.add_conversation(session_id, "assistant", summary)
        return AgentResult(
            request_id=request_id,
            status="ok",
            message=summary,
            tool_name=tool_name,
            account=account,
            latency_ms=tool_response["latency_ms"],
            data=tool_response["result"],
        )

    def _classify_intent(self, user_message: str, history: list[dict[str, str]], request_id: str) -> str:
        lower = user_message.lower()
        if "weather" in lower:
            return "general_knowledge"

        history_tail = "\n".join(f"{h['role']}: {h['content']}" for h in history[-2:])
        prompt = (
            "Classify the user request into exactly one label:\n"
            "general_knowledge | gmail_action | calendar_action | drive_action | multi_step\n"
            "Return only the label."
        )
        llm_input = f"Recent:\n{history_tail}\n\nUser: {user_message}"

        text = self._call_gemini_with_fallback(
            system_instruction=prompt,
            user_message=llm_input,
            request_id=request_id,
            phase="intent_classification",
            timeout_seconds=CLASSIFICATION_TIMEOUT_SECONDS,
        )

        if not text:
            return self._heuristic_intent(lower)

        label = text.strip().lower().splitlines()[0].strip()
        if label in INTENTS:
            return label
        return self._heuristic_intent(lower)

    def _heuristic_intent(self, lowered_message: str) -> str:
        if any(k in lowered_message for k in ("email", "gmail", "inbox", "mail")):
            return "gmail_action"
        if any(k in lowered_message for k in ("calendar", "event", "schedule", "meeting")):
            return "calendar_action"
        if any(k in lowered_message for k in ("drive", "file", "folder", "upload")):
            return "drive_action"
        return "general_knowledge"

    def _answer_general(self, user_message: str, history: list[dict[str, str]], request_id: str) -> str:
        history_tail = "\n".join(f"{h['role']}: {h['content']}" for h in history[-4:])
        system = (
            "You are a helpful AI assistant. Answer directly and clearly. "
            "Do not mention tool limitations. Keep responses concise unless detail is requested."
        )
        answer = self._call_gemini_with_fallback(
            system_instruction=system,
            user_message=f"Recent context:\n{history_tail}\n\nUser question: {user_message}",
            request_id=request_id,
            phase="general_answer",
            timeout_seconds=GEMINI_ANSWER_TIMEOUT_SECONDS,
        )
        return answer or FALLBACK_MODEL_MESSAGE

    def _plan_tool_decision(
        self,
        user_message: str,
        history: list[dict[str, str]],
        preferred_account: str | None,
        intent: str,
        request_id: str,
    ) -> AgentDecision | None:
        history_tail = "\n".join(f"{h['role']}: {h['content']}" for h in history[-4:])
        available_accounts = sorted(list_available_accounts())
        aliases = self.memory_repo.list_account_aliases()

        planner_prompt = (
            "Return strict JSON only.\n"
            "If tool needed return: {\"type\":\"tool_call\",\"tool\":\"...\",\"account\":\"...\",\"parameters\":{...}}\n"
            "If tool not needed return: {\"type\":\"response\",\"response\":\"...\"}.\n"
            "Use only these tools: send_email,list_emails,search_email,delete_email,create_event,list_events,delete_event,list_files,upload_file,delete_file"
        )

        planner_input = (
            f"Intent category: {intent}\n"
            f"Preferred account: {preferred_account or 'none'}\n"
            f"Available accounts: {available_accounts}\n"
            f"Alias map: {aliases}\n"
            f"Recent context:\n{history_tail}\n\n"
            f"User request: {user_message}"
        )

        text = self._call_gemini_with_fallback(
            system_instruction=planner_prompt,
            user_message=planner_input,
            request_id=request_id,
            phase="tool_planning",
            timeout_seconds=GEMINI_ANSWER_TIMEOUT_SECONDS,
        )
        if not text:
            return None

        try:
            return self._parse_decision(text)
        except ValueError:
            return None

    def _summarize_tool_result(
        self,
        user_message: str,
        tool_name: str,
        tool_result: Any,
        request_id: str,
    ) -> str:
        result_preview = json.dumps(tool_result, default=str)
        if len(result_preview) > 1200:
            result_preview = result_preview[:1200] + "..."

        summary_prompt = (
            "You are an assistant summarizing a completed action. "
            "Use plain language. Include key outcome only."
        )

        text = self._call_gemini_with_fallback(
            system_instruction=summary_prompt,
            user_message=(
                f"User request: {user_message}\n"
                f"Tool used: {tool_name}\n"
                f"Tool result JSON: {result_preview}"
            ),
            request_id=request_id,
            phase="tool_summary",
            timeout_seconds=CLASSIFICATION_TIMEOUT_SECONDS,
        )

        if text:
            return text
        return f"Action completed using '{tool_name}'."

    def _call_gemini_with_fallback(
        self,
        system_instruction: str,
        user_message: str,
        request_id: str,
        phase: str,
        timeout_seconds: int,
    ) -> str | None:
        if self._genai_client is None:
            _log(
                logging.ERROR,
                event="gemini_unavailable",
                request_id=request_id,
                tool_name=phase,
                latency_ms=0,
                error_type="MissingAPIKey",
            )
            return None

        last_error: Exception | None = None

        for model_name in GEMINI_MODELS:
            started = time.perf_counter()
            _log(
                logging.INFO,
                event="gemini_start",
                request_id=request_id,
                tool_name=phase,
                model_name=model_name,
                latency_ms=None,
                error_type=None,
            )

            future = _GEMINI_EXECUTOR.submit(
                self._genai_client.models.generate_content,
                model=model_name,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1,
                ),
                request_options={"timeout": timeout_seconds},
            )

            try:
                response = future.result(timeout=timeout_seconds)
            except TypeError as exc:
                if "request_options" not in str(exc):
                    last_error = exc
                    continue

                fallback_future = _GEMINI_EXECUTOR.submit(
                    self._genai_client.models.generate_content,
                    model=model_name,
                    contents=user_message,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.1,
                    ),
                )
                try:
                    response = fallback_future.result(timeout=timeout_seconds)
                except Exception as inner_exc:
                    fallback_future.cancel()
                    last_error = inner_exc
                    latency_ms = int((time.perf_counter() - started) * 1000)
                    _log(
                        logging.ERROR,
                        event="gemini_error",
                        request_id=request_id,
                        tool_name=phase,
                        model_name=model_name,
                        latency_ms=latency_ms,
                        error_type=type(inner_exc).__name__,
                    )
                    continue
            except FuturesTimeoutError as exc:
                future.cancel()
                last_error = exc
                latency_ms = int((time.perf_counter() - started) * 1000)
                _log(
                    logging.ERROR,
                    event="gemini_timeout",
                    request_id=request_id,
                    tool_name=phase,
                    model_name=model_name,
                    latency_ms=latency_ms,
                    error_type="TimeoutError",
                )
                continue
            except Exception as exc:
                last_error = exc
                latency_ms = int((time.perf_counter() - started) * 1000)
                _log(
                    logging.ERROR,
                    event="gemini_error",
                    request_id=request_id,
                    tool_name=phase,
                    model_name=model_name,
                    latency_ms=latency_ms,
                    error_type=type(exc).__name__,
                )
                continue

            text = self._extract_text(response)
            latency_ms = int((time.perf_counter() - started) * 1000)
            _log(
                logging.INFO,
                event="gemini_finish",
                request_id=request_id,
                tool_name=phase,
                model_name=model_name,
                latency_ms=latency_ms,
                error_type=None,
            )
            if text:
                return text.strip()

        _log(
            logging.ERROR,
            event="gemini_fallback_failed",
            request_id=request_id,
            tool_name=phase,
            latency_ms=None,
            error_type=type(last_error).__name__ if last_error else "UnknownError",
        )
        return None

    @staticmethod
    def _extract_text(response: Any) -> str:
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()

        candidates = getattr(response, "candidates", None)
        if not candidates:
            return ""

        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if not parts:
                continue
            for part in parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    return part_text.strip()

        return ""

    def _parse_decision(self, llm_text: str) -> AgentDecision:
        raw = llm_text.strip()

        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            raw = fenced.group(1)

        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise ValueError("Model output not valid JSON.")

        payload = json.loads(raw[start : end + 1])
        return AgentDecision.model_validate(payload)

    def _validate_tool_call(self, tool_name: str, account: str, parameters: dict[str, Any]) -> dict[str, Any]:
        if tool_name not in TOOLS:
            raise ValueError(f"Tool '{tool_name}' is not allowlisted.")

        available_accounts = list_available_accounts()
        if account not in available_accounts:
            raise ValueError(f"Account '{account}' is invalid.")

        model_cls = TOOL_PARAMETER_MODELS.get(tool_name)
        if model_cls is None:
            raise ValueError(f"No schema configured for '{tool_name}'.")

        try:
            validated = model_cls.model_validate(parameters or {})
        except ValidationError as exc:
            raise ValueError(f"Invalid parameters: {exc}") from exc

        return validated.model_dump(exclude_none=True)

    def _is_accounts_question(self, lowered_message: str) -> bool:
        return any(
            phrase in lowered_message
            for phrase in (
                "what accounts do i have",
                "show my accounts",
                "list my accounts",
                "which accounts do i have",
            )
        )

    def _format_accounts_overview(self) -> str:
        aliases = self.memory_repo.list_account_aliases()
        ordered = ["exam", "college", "personal", "private"]
        lines: list[str] = []

        for key in ordered:
            value = aliases.get(key)
            if value:
                lines.append(f"{key.title()} - {value}")

        for key in sorted(aliases.keys()):
            if key not in ordered:
                lines.append(f"{key.title()} - {aliases[key]}")

        if not lines:
            return "No accounts configured."
        return "\n".join(lines)

    def _detect_account_alias(self, lowered_message: str) -> str | None:
        aliases = self.memory_repo.list_account_aliases()
        for alias in aliases:
            if alias in lowered_message:
                return alias
        return None

    def _resolve_account_identifier(self, raw_identifier: str) -> str | None:
        identifier = raw_identifier.strip().lower()
        if not identifier:
            return None

        available = list_available_accounts()

        if identifier in available:
            return identifier

        for account in available:
            if account.lower() == identifier:
                return account

        mapped = self.memory_repo.get_account_by_alias(identifier)
        if mapped:
            if mapped in available:
                return mapped
            for account in available:
                if account.lower() == mapped.lower():
                    return account

        return None

    def _log_agent_finish(self, request_id: str, started: float, tool_name: str | None, error_type: str | None) -> None:
        latency_ms = int((time.perf_counter() - started) * 1000)
        _log(
            logging.INFO,
            event="agent_finish",
            request_id=request_id,
            tool_name=tool_name,
            latency_ms=latency_ms,
            error_type=error_type,
        )


_AGENT = SecureHybridAgent()


def run_agent(user_message: str, session_id: str, request_id: str | None = None) -> AgentResult:
    return _AGENT.run(user_message, session_id, request_id)
