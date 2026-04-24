"""
Core AI agent for the personal assistant.

Contains ``SecureHybridAgent``, which routes user messages either to direct
tool execution (fast heuristic path) or to a dual-provider LLM layer
(Groq primary → NVIDIA fallback) for natural-language tool planning.
Also exposes ``execute_tool`` and the module-level ``run_agent``
entry point used by the FastAPI route handler.
"""
from __future__ import annotations

import inspect
import json
import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import datetime, timezone
from threading import Lock
from typing import Any

from pydantic import ValidationError

from auth.google_auth_manager import list_available_accounts
from brain.llm_provider import FALLBACK_MESSAGE, call_llm
from brain.system_prompt import GENERAL_ANSWER_SYSTEM_PROMPT, SUMMARIZATION_SYSTEM_PROMPT
from brain.tool_registry import DESTRUCTIVE_TOOLS, TOOLS, TOOL_PARAMETER_MODELS
from config import LLM_TIMEOUT_SECONDS
from domain.schemas import AgentDecision, AgentResult
from memory.storage import SQLiteMemoryRepository

LOGGER = logging.getLogger(__name__)

_TOOL_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="agent-tools")

FALLBACK_MODEL_MESSAGE = FALLBACK_MESSAGE
DAILY_LIMIT_MESSAGE = "Daily AI limit reached. Please try again tomorrow."
DAILY_LLM_CALL_LIMIT = 100

TOOL_TIMEOUT_SECONDS = 8
MAX_HISTORY_MESSAGES = 6
MAX_LLM_CALLS_PER_REQUEST = 2


def _log(level: int, **payload: Any) -> None:
    LOGGER.log(level, json.dumps(payload, default=str))


class _DailyLLMLimiter:
    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._lock = Lock()
        self._day_utc = datetime.now(timezone.utc).date()
        self._count = 0

    def try_consume(self) -> bool:
        today_utc = datetime.now(timezone.utc).date()
        with self._lock:
            if today_utc != self._day_utc:
                self._day_utc = today_utc
                self._count = 0

            if self._count >= self._limit:
                return False

            self._count += 1
            return True

    def snapshot(self) -> tuple[str, int, int]:
        with self._lock:
            return self._day_utc.isoformat(), self._count, self._limit


_DAILY_LLM_LIMITER = _DailyLLMLimiter(DAILY_LLM_CALL_LIMIT)


def execute_tool(
    tool_name: str,
    account: str,
    parameters: dict[str, Any],
    request_id: str,
) -> dict[str, Any]:
    """
    Dispatch a registered tool function in a thread pool with timeout enforcement.

    Args:
        tool_name: Name of the tool to execute (must be in the tool registry).
        account: The account identifier to pass as the first argument to the tool.
        parameters: Keyword arguments forwarded to the tool function.
        request_id: Unique request identifier for structured logging.

    Returns:
        A dict with ``latency_ms`` (int) and ``result`` (tool's return value).

    Raises:
        ValueError: If the tool is not registered.
        TimeoutError: If the tool exceeds ``TOOL_TIMEOUT_SECONDS``.
        RuntimeError: If the tool raises any other exception.
    """
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
    Smart hybrid architecture:
    1) Heuristic intent routing (no classification call)
    2) Conditional multi-step LLM pipeline:
       - Fast path: direct tool execution for simple queries (no LLM)
       - Smart path: tool execution → LLM summarization for reasoning queries
    3) Max 2 LLM calls per user message (Groq primary → NVIDIA fallback)
    4) Deterministic fallback when LLM is unavailable
    """

    _REASONING_KEYWORDS = (
        "summarize", "summarise", "summary", "analyze", "analyse", "analysis",
        "compare", "explain", "insights", "insight", "important", "urgent",
        "priority", "highlight", "overview", "brief", "digest", "recap",
        "what's new", "what happened", "anything important", "any urgent",
        "key takeaways", "action items", "today",
    )

    def __init__(self) -> None:
        self.memory_repo = SQLiteMemoryRepository()
        self._daily_limiter = _DAILY_LLM_LIMITER

    def run(self, user_message: str, session_id: str, request_id: str | None = None) -> AgentResult:
        """
        Process a user message and return an ``AgentResult``.

        Routing order:
        1. Account overview shortcut (no tool / no LLM call).
        2. Confirmation / cancellation of pending destructive actions.
        3. Heuristic intent detection → direct tool execution (no LLM call).
        4. LLM tool planner → validated tool execution.

        Args:
            user_message: Raw text message from the user.
            session_id: Telegram ``chat_id`` used to scope memory.
            request_id: Optional trace ID; auto-generated if omitted.

        Returns:
            An ``AgentResult`` with ``status``, ``message``, and optional metadata.
        """
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
        llm_calls_used = 0

        alias_in_message = self._detect_account_alias(lower)
        if alias_in_message:
            resolved_alias_account = self._resolve_account_identifier(alias_in_message)
            if resolved_alias_account:
                self.memory_repo.set_account_preference(session_id, resolved_alias_account)

        intent = self._heuristic_intent(lower)
        preferred_account = self.memory_repo.get_account_preference(session_id)

        if intent == "general_knowledge":
            if not self._reserve_llm_call(rid, "general_answer"):
                return self._daily_limit_result(session_id, rid, started, None)
            llm_calls_used += 1

            answer = self._answer_general(cleaned, history, rid)
            self.memory_repo.add_conversation(session_id, "assistant", answer)
            self._log_agent_finish(rid, started, None, None)
            return AgentResult(request_id=rid, status="ok", message=answer)

        decision = self._build_direct_tool_decision(lower, preferred_account)
        if decision is None:
            if not self._reserve_llm_call(rid, "tool_planning"):
                return self._daily_limit_result(session_id, rid, started, None)
            llm_calls_used += 1
            decision = self._plan_tool_decision(cleaned, history, preferred_account, intent, rid)

        if decision is None:
            message = "I could not determine the required tool action. Please rephrase with the exact action."
            self.memory_repo.add_conversation(session_id, "assistant", message)
            self._log_agent_finish(rid, started, None, "ToolDecisionError")
            return AgentResult(request_id=rid, status="error", message=message, error_type="ToolDecisionError")

        if decision.type == "response":
            text = decision.response or "I could not determine the requested tool action."
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
            llm_calls_used=llm_calls_used,
        )
        self._log_agent_finish(rid, started, tool_name, result.error_type)
        return result

    def _reserve_llm_call(self, request_id: str, phase: str) -> bool:
        """
        Attempt to consume one LLM quota unit for the current UTC day.

        Returns:
            ``True`` if the call is allowed, ``False`` if the daily limit is reached.
        """
        allowed = self._daily_limiter.try_consume()
        day_utc, count, limit = self._daily_limiter.snapshot()
        _log(
            logging.INFO,
            event="llm_quota_check",
            request_id=request_id,
            tool_name=phase,
            day_utc=day_utc,
            daily_count=count,
            daily_limit=limit,
            allowed=allowed,
            latency_ms=0,
            error_type=None if allowed else "DailyLimitExceeded",
        )
        return allowed

    def _daily_limit_result(
        self,
        session_id: str,
        request_id: str,
        started: float,
        tool_name: str | None,
    ) -> AgentResult:
        self.memory_repo.add_conversation(session_id, "assistant", DAILY_LIMIT_MESSAGE)
        self._log_agent_finish(request_id, started, tool_name, "DailyLimitExceeded")
        return AgentResult(
            request_id=request_id,
            status="error",
            message=DAILY_LIMIT_MESSAGE,
            tool_name=tool_name,
            error_type="DailyLimitExceeded",
        )

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
        )

    def _execute_and_build_result(
        self,
        session_id: str,
        request_id: str,
        tool_name: str,
        account: str,
        parameters: dict[str, Any],
        user_message: str = "",
        llm_calls_used: int = 0,
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

        raw_result = tool_response["result"]

        # --- Smart path: LLM summarization when reasoning is needed ---
        if (
            user_message
            and self._needs_reasoning(user_message.lower())
            and llm_calls_used < MAX_LLM_CALLS_PER_REQUEST
        ):
            summary = self._summarize_tool_result(
                user_message=user_message,
                tool_name=tool_name,
                tool_result=raw_result,
                request_id=request_id,
            )
            if summary:
                self.memory_repo.add_conversation(session_id, "assistant", summary)
                return AgentResult(
                    request_id=request_id,
                    status="ok",
                    message=summary,
                    tool_name=tool_name,
                    account=account,
                    latency_ms=tool_response["latency_ms"],
                    data=raw_result,
                )

        # --- Fast path: deterministic formatting ---
        summary = self._format_tool_result(tool_name, account, raw_result)
        self.memory_repo.add_conversation(session_id, "assistant", summary)
        return AgentResult(
            request_id=request_id,
            status="ok",
            message=summary,
            tool_name=tool_name,
            account=account,
            latency_ms=tool_response["latency_ms"],
            data=raw_result,
        )

    _GMAIL_KEYWORDS = ("email", "emails", "mail", "mails", "gmail", "inbox", "unread", "message", "messages")
    _CALENDAR_KEYWORDS = ("calendar", "event", "events", "schedule", "meeting", "agenda", "appointments")
    _DRIVE_KEYWORDS = ("drive", "file", "files", "folder", "upload", "document", "documents", "storage")

    def _heuristic_intent(self, lowered_message: str) -> str:
        """
        Classify the user's intent using keyword matching (no LLM call).

        Returns:
            One of ``"gmail_action"``, ``"calendar_action"``, ``"drive_action"``,
            or ``"general_knowledge"``.
        """
        if any(k in lowered_message for k in self._GMAIL_KEYWORDS):
            return "gmail_action"
        if any(k in lowered_message for k in self._CALENDAR_KEYWORDS):
            return "calendar_action"
        if any(k in lowered_message for k in self._DRIVE_KEYWORDS):
            return "drive_action"
        return "general_knowledge"

    def _answer_general(self, user_message: str, history: list[dict[str, str]], request_id: str) -> str:
        history_tail = "\n".join(f"{h['role']}: {h['content']}" for h in history[-4:])
        answer = call_llm(
            prompt=f"Recent context:\n{history_tail}\n\nUser question: {user_message}",
            system_prompt=GENERAL_ANSWER_SYSTEM_PROMPT,
            request_id=request_id,
            phase="general_answer",
            timeout_seconds=LLM_TIMEOUT_SECONDS,
        )
        return answer or FALLBACK_MODEL_MESSAGE

    _SEARCH_EMAIL_PHRASES = ("search email", "find email", "emails from", "emails about", "mail from", "mail about")

    def _build_direct_tool_decision(
        self, lowered_message: str, preferred_account: str | None
    ) -> AgentDecision | None:
        """
        Attempt to build a tool decision purely from keyword heuristics.

        Skips the LLM planner entirely when a known pattern is found.
        Requires a ``preferred_account`` to be set (from previous session context).

        Returns:
            An ``AgentDecision`` with ``type="tool_call"``, or ``None`` if no
            heuristic matched.
        """
        if not preferred_account:
            return None

        # Search-email detection (must come before generic email detection)
        if any(phrase in lowered_message for phrase in self._SEARCH_EMAIL_PHRASES):
            # Extract a rough query from the message by stripping common prefixes
            query = lowered_message
            for prefix in ("search email", "find email", "search emails", "find emails"):
                query = query.replace(prefix, "")
            query = query.strip() or "is:unread"
            return AgentDecision(
                type="tool_call",
                tool="search_email",
                account=preferred_account,
                parameters={"query": query, "max_results": 10},
            )

        # Flexible email keyword matching
        if any(k in lowered_message for k in self._GMAIL_KEYWORDS):
            return AgentDecision(
                type="tool_call",
                tool="list_emails",
                account=preferred_account,
                parameters={"max_results": 5},
            )

        # Flexible calendar keyword matching
        if any(k in lowered_message for k in self._CALENDAR_KEYWORDS):
            return AgentDecision(
                type="tool_call",
                tool="list_events",
                account=preferred_account,
                parameters={"max_results": 10},
            )

        # Flexible drive keyword matching
        if any(k in lowered_message for k in self._DRIVE_KEYWORDS):
            return AgentDecision(
                type="tool_call",
                tool="list_files",
                account=preferred_account,
                parameters={"page_size": 10},
            )

        return None

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
            "Use only these tools: send_email,list_emails,search_email,delete_email,create_event,list_events,delete_event,list_files,upload_file,delete_file.\n"
            "Do not return markdown."
        )

        planner_input = (
            f"Intent category: {intent}\n"
            f"Preferred account: {preferred_account or 'none'}\n"
            f"Available accounts: {available_accounts}\n"
            f"Alias map: {aliases}\n"
            f"Recent context:\n{history_tail}\n\n"
            f"User request: {user_message}"
        )

        text = call_llm(
            prompt=planner_input,
            system_prompt=planner_prompt,
            request_id=request_id,
            phase="tool_planning",
            timeout_seconds=LLM_TIMEOUT_SECONDS,
        )
        if not text:
            return None

        try:
            return self._parse_decision(text)
        except ValueError:
            return None

    def _parse_decision(self, llm_text: str) -> AgentDecision:
        """
        Parse a JSON tool-decision string returned by the LLM planner.

        Strips optional markdown fences and extracts the first JSON object
        found in the response.

        Raises:
            ValueError: If no valid JSON object can be extracted.
        """
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

    def _validate_tool_call(
        self, tool_name: str, account: str, parameters: dict[str, Any]
    ) -> dict[str, Any]:
        """
        Validate that a tool call is safe to execute.

        Checks that the tool is allowlisted, the account is available, and
        the parameters pass Pydantic schema validation.

        Returns:
            A clean parameter dict (``None`` values excluded).

        Raises:
            ValueError: On any validation failure.
        """
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

    # ── Smart reasoning detection ──────────────────────────────────────

    def _needs_reasoning(self, lowered_message: str) -> bool:
        """
        Determine if the user's query requires LLM reasoning/summarization.

        Uses keyword heuristics only (no LLM call). Returns True when the
        message contains words implying summarization, analysis, or insight.
        """
        return any(kw in lowered_message for kw in self._REASONING_KEYWORDS)

    # ── Data formatting for LLM consumption ────────────────────────────

    def _format_data_for_llm(self, tool_name: str, tool_result: Any) -> str:
        """
        Convert raw tool result JSON into structured, human-readable text
        suitable for passing into an LLM summarization prompt.

        This avoids sending raw JSON blobs to the model, improving output
        quality significantly.
        """
        if not isinstance(tool_result, dict):
            return json.dumps(tool_result, default=str, indent=2)

        if tool_name in ("list_emails", "search_email"):
            messages = tool_result.get("messages", [])
            if not messages:
                return "No emails found."
            lines = []
            for i, msg in enumerate(messages, 1):
                subject = msg.get("subject", msg.get("Subject", "No subject"))
                sender = msg.get("from", msg.get("From", msg.get("sender", "Unknown")))
                snippet = msg.get("snippet", msg.get("Snippet", ""))
                date = msg.get("date", msg.get("Date", msg.get("internalDate", "")))
                labels = msg.get("labelIds", [])
                lines.append(
                    f"Email {i}:\n"
                    f"  - Subject: {subject}\n"
                    f"  - From: {sender}\n"
                    f"  - Date: {date}\n"
                    f"  - Labels: {', '.join(labels) if labels else 'none'}\n"
                    f"  - Snippet: {snippet}"
                )
            return f"Total: {tool_result.get('count', len(messages))} email(s)\n\n" + "\n\n".join(lines)

        if tool_name == "list_events":
            events = tool_result.get("events", [])
            if not events:
                return "No upcoming events found."
            lines = []
            for i, evt in enumerate(events, 1):
                summary = evt.get("summary", "No title")
                start = evt.get("start", {})
                start_time = start.get("dateTime", start.get("date", "Unknown"))
                end = evt.get("end", {})
                end_time = end.get("dateTime", end.get("date", ""))
                location = evt.get("location", "")
                lines.append(
                    f"Event {i}:\n"
                    f"  - Title: {summary}\n"
                    f"  - Start: {start_time}\n"
                    f"  - End: {end_time}\n"
                    + (f"  - Location: {location}\n" if location else "")
                )
            return f"Total: {tool_result.get('count', len(events))} event(s)\n\n" + "\n\n".join(lines)

        if tool_name == "list_files":
            files = tool_result.get("files", [])
            if not files:
                return "No files found."
            lines = []
            for i, f in enumerate(files, 1):
                lines.append(f"File {i}: {f.get('name', 'Unknown')} (ID: {f.get('id', 'N/A')})")
            return f"Total: {tool_result.get('count', len(files))} file(s)\n\n" + "\n".join(lines)

        # Fallback: compact JSON
        formatted = json.dumps(tool_result, default=str, indent=2)
        if len(formatted) > 3000:
            formatted = formatted[:3000] + "\n... (truncated)"
        return formatted

    # ── LLM summarization pipeline ─────────────────────────────────────

    def _summarize_tool_result(
        self,
        user_message: str,
        tool_name: str,
        tool_result: Any,
        request_id: str,
    ) -> str | None:
        """
        Pass tool results through the LLM for intelligent summarization.

        Only called when _needs_reasoning() returns True and the LLM call
        budget has not been exhausted.  Returns None on failure so the
        caller can fall back to deterministic formatting.
        """
        if not self._reserve_llm_call(request_id, "summarization"):
            _log(
                logging.WARNING,
                event="summarization_skipped",
                request_id=request_id,
                tool_name=tool_name,
                latency_ms=0,
                error_type="DailyLimitExceeded",
            )
            return None

        formatted_data = self._format_data_for_llm(tool_name, tool_result)

        user_prompt = (
            f"User asked: {user_message}\n\n"
            f"Here is the data retrieved:\n{formatted_data}\n\n"
            "Your task:\n"
            "- Identify the most important items\n"
            "- Summarize clearly and concisely\n"
            "- Highlight key insights, deadlines, or action items\n"
            "- Use bullet points for readability\n"
            "- Be informative but not verbose"
        )

        summary = call_llm(
            prompt=user_prompt,
            system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
            request_id=request_id,
            phase="summarization",
            timeout_seconds=LLM_TIMEOUT_SECONDS,
        )
        return summary or None

    # ── Deterministic tool-result formatting (fast path) ───────────────

    def _format_tool_result(self, tool_name: str, account: str, tool_result: Any) -> str:
        if tool_name == "list_emails":
            count = self._safe_int(tool_result, "count")
            return f"Fetched {count} email(s) for {account}."
        if tool_name == "search_email":
            count = self._safe_int(tool_result, "count")
            return f"Found {count} email(s) matching your query in {account}."
        if tool_name == "send_email":
            message_id = self._safe_str(tool_result, "id")
            return f"Email sent from {account}. Message ID: {message_id}."
        if tool_name == "delete_email":
            message_id = self._safe_str(tool_result, "message_id")
            return f"Deleted email {message_id} from {account}."
        if tool_name == "list_events":
            count = self._safe_int(tool_result, "count")
            return f"Fetched {count} event(s) for {account}."
        if tool_name == "create_event":
            event_id = self._safe_str(tool_result, "id")
            return f"Created event {event_id} for {account}."
        if tool_name == "delete_event":
            event_id = self._safe_str(tool_result, "event_id")
            return f"Deleted event {event_id} for {account}."
        if tool_name == "list_files":
            count = self._safe_int(tool_result, "count")
            return f"Fetched {count} file(s) for {account}."
        if tool_name == "upload_file":
            file_name = self._safe_str(tool_result, "name")
            file_id = self._safe_str(tool_result, "id")
            return f"Uploaded file '{file_name}' to {account}. File ID: {file_id}."
        if tool_name == "delete_file":
            file_id = self._safe_str(tool_result, "file_id")
            return f"Deleted file {file_id} from {account}."

        preview = json.dumps(tool_result, default=str)
        if len(preview) > 240:
            preview = preview[:240] + "..."
        return f"Action '{tool_name}' completed for {account}. Result: {preview}"

    @staticmethod
    def _safe_int(payload: Any, key: str) -> int:
        if isinstance(payload, dict):
            value = payload.get(key, 0)
            if isinstance(value, int):
                return value
            try:
                return int(value)
            except (TypeError, ValueError):
                return 0
        return 0

    @staticmethod
    def _safe_str(payload: Any, key: str) -> str:
        if isinstance(payload, dict):
            value = payload.get(key)
            if value is None:
                return "unknown"
            return str(value)
        return "unknown"

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
        """
        Resolve a raw account string to a verified account key.

        Resolution order:
        1. Exact match against available accounts.
        2. Case-insensitive match.
        3. Alias lookup (DB + defaults), then re-matched against available accounts.

        Returns:
            The resolved account key, or ``None`` if unresolvable.
        """
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
    """
    Module-level entry point for running the singleton agent.

    Called by the FastAPI route handler via ``asyncio.to_thread``.

    Args:
        user_message: The user's Telegram message text.
        session_id: Telegram ``chat_id`` cast to string.
        request_id: Optional trace ID propagated from the HTTP request.

    Returns:
        An ``AgentResult`` containing the reply and metadata.
    """
    return _AGENT.run(user_message, session_id, request_id)
