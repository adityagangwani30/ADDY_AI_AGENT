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
from brain.system_prompt import (
    GENERAL_ANSWER_SYSTEM_PROMPT,
    PLANNER_SYSTEM_PROMPT,
    REASONING_SYSTEM_PROMPT,
    REFINEMENT_SYSTEM_PROMPT,
    RESPONSE_BUILDER_SYSTEM_PROMPT,
)
from brain.tool_registry import DESTRUCTIVE_TOOLS, TOOLS, TOOL_PARAMETER_MODELS
from config import LLM_TIMEOUT_SECONDS
from domain.schemas import AgentResult
from memory.storage import SQLiteMemoryRepository

LOGGER = logging.getLogger(__name__)

_TOOL_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="agent-tools")

FALLBACK_MODEL_MESSAGE = FALLBACK_MESSAGE
DAILY_LIMIT_MESSAGE = "Daily AI limit reached. Please try again tomorrow."
DAILY_LLM_CALL_LIMIT = 100

TOOL_TIMEOUT_SECONDS = 8
MAX_HISTORY_MESSAGES = 6
MAX_LLM_CALLS_PER_REQUEST = 4


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
    LLM-first multi-step personal assistant.

    Pipeline:
    1) LLM Planner   — understand intent, select tools (MANDATORY)
    2) Tool Execution — deterministic, validated (if needed)
    3) LLM Response   — natural-language formatting (ALWAYS for tools)
    4) LLM Reasoning  — summarization / insights (CONDITIONAL)
    5) LLM Refinement — polish verbose output (RARE)

    LLM budget: min 1, typical 2-3, max 4 calls per request.
    """

    def __init__(self) -> None:
        self.memory_repo = SQLiteMemoryRepository()
        self._daily_limiter = _DAILY_LLM_LIMITER

    def run(self, user_message: str, session_id: str, request_id: str | None = None) -> AgentResult:
        rid = request_id or str(uuid.uuid4())
        started = time.perf_counter()
        cleaned = user_message.strip()
        _log(logging.INFO, event="agent_start", request_id=rid,
             tool_name=None, latency_ms=None, error_type=None)

        if not cleaned:
            return AgentResult(request_id=rid, status="error",
                               message="Message cannot be empty.")

        self.memory_repo.add_conversation(session_id, "user", cleaned)
        lower = cleaned.lower()

        # --- Quick exits (no LLM) ---
        if self._is_accounts_question(lower):
            msg = self._format_accounts_overview()
            self.memory_repo.add_conversation(session_id, "assistant", msg)
            return AgentResult(request_id=rid, status="ok", message=msg)

        if lower in {"confirm", "yes confirm", "confirm action"}:
            return self._handle_confirmation(session_id, rid)

        if lower in {"cancel", "deny", "reject"}:
            self.memory_repo.clear_pending_confirmation(session_id)
            return AgentResult(request_id=rid, status="ok",
                               message="Pending action cancelled.")

        # --- Resolve account context ---
        alias = self._detect_account_alias(lower)
        if alias:
            resolved = self._resolve_account_identifier(alias)
            if resolved:
                self.memory_repo.set_account_preference(session_id, resolved)

        preferred = self.memory_repo.get_account_preference(session_id)
        history = self.memory_repo.get_conversation(
            session_id, limit=MAX_HISTORY_MESSAGES)
        llm_calls = 0

        # ═══ STEP 1: LLM PLANNER (mandatory) ═══
        if not self._reserve_llm_call(rid, "planner"):
            return self._daily_limit_result(session_id, rid, started, None)
        llm_calls += 1
        plan = self._plan_with_llm(cleaned, history, preferred, rid)

        task_type = plan.get("task_type", "direct_answer")
        needs_reasoning = plan.get("requires_reasoning", False)
        needs_refinement = plan.get("requires_refinement", False)
        style = plan.get("response_style", "concise")

        # ═══ DIRECT ANSWER (no tools) ═══
        if task_type == "direct_answer":
            direct = plan.get("direct_response", "")
            if direct and direct.strip():
                self.memory_repo.add_conversation(
                    session_id, "assistant", direct)
                self._log_agent_finish(rid, started, None, None)
                return AgentResult(
                    request_id=rid, status="ok", message=direct)

            if (llm_calls < MAX_LLM_CALLS_PER_REQUEST
                    and self._reserve_llm_call(rid, "general_answer")):
                llm_calls += 1
                answer = self._answer_general(cleaned, history, rid)
                self.memory_repo.add_conversation(
                    session_id, "assistant", answer)
                self._log_agent_finish(rid, started, None, None)
                return AgentResult(
                    request_id=rid, status="ok", message=answer)

            self.memory_repo.add_conversation(
                session_id, "assistant", FALLBACK_MODEL_MESSAGE)
            self._log_agent_finish(rid, started, None, None)
            return AgentResult(
                request_id=rid, status="ok", message=FALLBACK_MODEL_MESSAGE)

        # ═══ TOOL EXECUTION PATH ═══
        tool_name = plan.get("tool") or ""
        raw_account = plan.get("account") or preferred or ""
        parameters = plan.get("parameters") or {}
        account = self._resolve_account_identifier(raw_account)

        if not tool_name or tool_name not in TOOLS:
            msg = ("I could not determine the required action. "
                   "Please rephrase your request.")
            self.memory_repo.add_conversation(session_id, "assistant", msg)
            self._log_agent_finish(rid, started, None, "PlannerError")
            return AgentResult(request_id=rid, status="error",
                               message=msg, error_type="PlannerError")

        if not account:
            msg = ("I could not map that request to a connected account. "
                   "Ask 'what accounts do I have?'")
            self.memory_repo.add_conversation(session_id, "assistant", msg)
            self._log_agent_finish(
                rid, started, tool_name, "AccountResolutionError")
            return AgentResult(request_id=rid, status="error",
                               message=msg,
                               error_type="AccountResolutionError")

        try:
            validated = self._validate_tool_call(
                tool_name, account, parameters)
        except ValueError as exc:
            self._log_agent_finish(
                rid, started, tool_name, type(exc).__name__)
            return AgentResult(request_id=rid, status="error",
                               message=str(exc),
                               error_type=type(exc).__name__)

        self.memory_repo.set_account_preference(session_id, account)

        # Destructive action confirmation
        if tool_name in DESTRUCTIVE_TOOLS:
            self.memory_repo.save_pending_confirmation(
                session_id=session_id, tool_name=tool_name,
                account=account, parameters=validated)
            cmsg = (f"⚠️ Confirmation required for '{tool_name}' on "
                    f"'{account}'. Reply 'confirm' or 'cancel'.")
            self.memory_repo.add_conversation(
                session_id, "assistant", cmsg)
            self._log_agent_finish(rid, started, tool_name, None)
            return AgentResult(
                request_id=rid, status="confirmation_required",
                message=cmsg, tool_name=tool_name, account=account)

        # ═══ STEP 2: TOOL EXECUTION (deterministic) ═══
        result, latency, err = self._execute_tool_safely(
            tool_name, account, validated, rid)
        if err:
            self.memory_repo.add_conversation(session_id, "assistant", err)
            self._log_agent_finish(rid, started, tool_name, "ToolError")
            return AgentResult(
                request_id=rid, status="error", message=err,
                tool_name=tool_name, account=account,
                error_type="ToolError")

        data_text = self._format_data_for_llm(tool_name, result)

        # ═══ STEP 3: LLM RESPONSE BUILDER (always) ═══
        response = None
        if (llm_calls < MAX_LLM_CALLS_PER_REQUEST
                and self._reserve_llm_call(rid, "response_builder")):
            llm_calls += 1
            response = self._build_response_with_llm(
                cleaned, tool_name, data_text, style, rid)

        # ═══ STEP 4: LLM REASONING (conditional) ═══
        if (needs_reasoning
                and llm_calls < MAX_LLM_CALLS_PER_REQUEST
                and self._reserve_llm_call(rid, "reasoning")):
            llm_calls += 1
            reasoning = self._reason_with_llm(data_text, cleaned, rid)
            if reasoning:
                response = reasoning

        # ═══ STEP 5: LLM REFINEMENT (rare) ═══
        if (needs_refinement and response
                and llm_calls < MAX_LLM_CALLS_PER_REQUEST
                and self._reserve_llm_call(rid, "refinement")):
            llm_calls += 1
            refined = self._refine_response_with_llm(response, rid)
            if refined:
                response = refined

        # Fallback if all LLM calls failed
        if not response:
            response = self._fallback_format(tool_name, account, result)

        self.memory_repo.add_conversation(session_id, "assistant", response)
        self._log_agent_finish(rid, started, tool_name, None)
        return AgentResult(
            request_id=rid, status="ok", message=response,
            tool_name=tool_name, account=account,
            latency_ms=latency, data=result)

    # ── STEP 1 impl ───────────────────────────────────────────────────

    def _plan_with_llm(self, user_message: str,
                       history: list[dict[str, str]],
                       preferred_account: str | None,
                       request_id: str) -> dict:
        htail = "\n".join(
            f"{h['role']}: {h['content']}" for h in history[-4:])
        accounts = sorted(list_available_accounts())
        aliases = self.memory_repo.list_account_aliases()

        prompt = (
            f"Preferred account: {preferred_account or 'none'}\n"
            f"Available accounts: {accounts}\n"
            f"Alias map: {aliases}\n"
            f"Recent conversation:\n{htail}\n\n"
            f"User message: {user_message}")

        text = call_llm(
            prompt=prompt, system_prompt=PLANNER_SYSTEM_PROMPT,
            request_id=request_id, phase="planner",
            timeout_seconds=LLM_TIMEOUT_SECONDS)

        if not text:
            return {"task_type": "direct_answer",
                    "direct_response": FALLBACK_MODEL_MESSAGE}
        try:
            return self._extract_json(text)
        except ValueError:
            return {"task_type": "direct_answer",
                    "direct_response": text.strip()}

    # ── STEP 3 impl ───────────────────────────────────────────────────

    def _build_response_with_llm(self, user_message: str,
                                 tool_name: str, data: str,
                                 style: str,
                                 request_id: str) -> str | None:
        prompt = (
            f"User asked: {user_message}\n"
            f"Tool executed: {tool_name}\n"
            f"Response style: {style}\n\n"
            f"Raw data:\n{data}\n\n"
            "Convert into a clear, natural response. "
            "Never show raw JSON or IDs.")
        return call_llm(
            prompt=prompt,
            system_prompt=RESPONSE_BUILDER_SYSTEM_PROMPT,
            request_id=request_id, phase="response_builder",
            timeout_seconds=LLM_TIMEOUT_SECONDS)

    # ── STEP 4 impl ───────────────────────────────────────────────────

    def _reason_with_llm(self, data: str, user_message: str,
                         request_id: str) -> str | None:
        prompt = (
            f"User asked: {user_message}\n\n"
            f"Data to analyze:\n{data}\n\n"
            "Provide intelligent analysis: identify important items, "
            "extract action items, highlight urgency, summarize insights.")
        return call_llm(
            prompt=prompt, system_prompt=REASONING_SYSTEM_PROMPT,
            request_id=request_id, phase="reasoning",
            timeout_seconds=LLM_TIMEOUT_SECONDS)

    # ── STEP 5 impl ───────────────────────────────────────────────────

    def _refine_response_with_llm(self, response: str,
                                  request_id: str) -> str | None:
        prompt = ("Polish this response for clarity and brevity. "
                  f"Return only the improved text:\n\n{response}")
        return call_llm(
            prompt=prompt, system_prompt=REFINEMENT_SYSTEM_PROMPT,
            request_id=request_id, phase="refinement",
            timeout_seconds=LLM_TIMEOUT_SECONDS)

    # ── General answer ─────────────────────────────────────────────────

    def _answer_general(self, user_message: str,
                        history: list[dict[str, str]],
                        request_id: str) -> str:
        htail = "\n".join(
            f"{h['role']}: {h['content']}" for h in history[-4:])
        answer = call_llm(
            prompt=(f"Recent context:\n{htail}\n\n"
                    f"User question: {user_message}"),
            system_prompt=GENERAL_ANSWER_SYSTEM_PROMPT,
            request_id=request_id, phase="general_answer",
            timeout_seconds=LLM_TIMEOUT_SECONDS)
        return answer or FALLBACK_MODEL_MESSAGE

    # ── Tool execution ─────────────────────────────────────────────────

    def _execute_tool_safely(self, tool_name: str, account: str,
                             parameters: dict[str, Any],
                             request_id: str
                             ) -> tuple[Any, int, str | None]:
        try:
            r = execute_tool(tool_name=tool_name, account=account,
                             parameters=parameters, request_id=request_id)
            return r["result"], r["latency_ms"], None
        except TimeoutError:
            return None, 0, ("That request is taking longer than expected."
                             " Please try again.")
        except Exception as exc:
            return None, 0, str(exc)

    # ── Confirmation ───────────────────────────────────────────────────

    def _handle_confirmation(self, session_id: str,
                             request_id: str) -> AgentResult:
        pending = self.memory_repo.get_pending_confirmation(session_id)
        if not pending:
            return AgentResult(
                request_id=request_id, status="error",
                message="No pending action to confirm.",
                error_type="NoPendingConfirmation")

        self.memory_repo.clear_pending_confirmation(session_id)
        tn = str(pending["tool_name"])
        acc = str(pending["account"])
        params = dict(pending["parameters"])

        result, lat, err = self._execute_tool_safely(
            tn, acc, params, request_id)
        if err:
            self.memory_repo.add_conversation(
                session_id, "assistant", err)
            return AgentResult(
                request_id=request_id, status="error", message=err,
                tool_name=tn, account=acc, error_type="ToolError")

        resp = None
        if self._reserve_llm_call(request_id, "confirm_response"):
            fmt = self._format_data_for_llm(tn, result)
            resp = self._build_response_with_llm(
                f"confirmed {tn}", tn, fmt, "concise", request_id)
        if not resp:
            resp = self._fallback_format(tn, acc, result)

        self.memory_repo.add_conversation(session_id, "assistant", resp)
        return AgentResult(
            request_id=request_id, status="ok", message=resp,
            tool_name=tn, account=acc, latency_ms=lat, data=result)

    # ── Quota ──────────────────────────────────────────────────────────

    def _reserve_llm_call(self, request_id: str, phase: str) -> bool:
        allowed = self._daily_limiter.try_consume()
        day, count, limit = self._daily_limiter.snapshot()
        _log(logging.INFO, event="llm_quota_check",
             request_id=request_id, tool_name=phase,
             day_utc=day, daily_count=count, daily_limit=limit,
             allowed=allowed, latency_ms=0,
             error_type=None if allowed else "DailyLimitExceeded")
        return allowed

    def _daily_limit_result(self, session_id: str, request_id: str,
                            started: float,
                            tool_name: str | None) -> AgentResult:
        self.memory_repo.add_conversation(
            session_id, "assistant", DAILY_LIMIT_MESSAGE)
        self._log_agent_finish(
            request_id, started, tool_name, "DailyLimitExceeded")
        return AgentResult(
            request_id=request_id, status="error",
            message=DAILY_LIMIT_MESSAGE, tool_name=tool_name,
            error_type="DailyLimitExceeded")

    # ── Data formatting ────────────────────────────────────────────────

    def _format_data_for_llm(self, tool_name: str,
                             tool_result: Any) -> str:
        if not isinstance(tool_result, dict):
            return json.dumps(tool_result, default=str, indent=2)

        if tool_name in ("list_emails", "search_email"):
            msgs = tool_result.get("messages", [])
            if not msgs:
                return "No emails found."
            lines = []
            for i, m in enumerate(msgs, 1):
                lines.append(
                    f"Email {i}:\n"
                    f"  Subject: {m.get('subject', m.get('Subject', 'N/A'))}\n"
                    f"  From: {m.get('from', m.get('From', 'Unknown'))}\n"
                    f"  Date: {m.get('date', m.get('internalDate', ''))}\n"
                    f"  Labels: {', '.join(m.get('labelIds', [])) or 'none'}\n"
                    f"  Snippet: {m.get('snippet', '')}")
            return (f"Total: {tool_result.get('count', len(msgs))}"
                    f" email(s)\n\n" + "\n\n".join(lines))

        if tool_name == "list_events":
            evts = tool_result.get("events", [])
            if not evts:
                return "No upcoming events found."
            lines = []
            for i, e in enumerate(evts, 1):
                s = e.get("start", {})
                ed = e.get("end", {})
                loc = e.get("location", "")
                lines.append(
                    f"Event {i}:\n"
                    f"  Title: {e.get('summary', 'No title')}\n"
                    f"  Start: {s.get('dateTime', s.get('date', '?'))}\n"
                    f"  End: {ed.get('dateTime', ed.get('date', ''))}\n"
                    + (f"  Location: {loc}\n" if loc else ""))
            return (f"Total: {tool_result.get('count', len(evts))}"
                    f" event(s)\n\n" + "\n\n".join(lines))

        if tool_name == "list_files":
            files = tool_result.get("files", [])
            if not files:
                return "No files found."
            lines = [f"File {i}: {f.get('name', '?')} "
                     f"(ID: {f.get('id', 'N/A')})"
                     for i, f in enumerate(files, 1)]
            return (f"Total: {tool_result.get('count', len(files))}"
                    f" file(s)\n\n" + "\n".join(lines))

        if tool_name == "send_email":
            return (f"Email sent. ID: {tool_result.get('id', '?')}, "
                    f"Thread: {tool_result.get('thread_id', '?')}")
        if tool_name == "create_event":
            return (f"Event created. ID: {tool_result.get('id', '?')}, "
                    f"Link: {tool_result.get('html_link', 'N/A')}")
        if tool_name == "upload_file":
            return (f"File uploaded. Name: {tool_result.get('name', '?')},"
                    f" ID: {tool_result.get('id', '?')}")

        out = json.dumps(tool_result, default=str, indent=2)
        return out[:3000] + "\n...(truncated)" if len(out) > 3000 else out

    # ── Deterministic fallback ─────────────────────────────────────────

    def _fallback_format(self, tool_name: str, account: str,
                         tool_result: Any) -> str:
        si = self._safe_int
        m = {
            "list_emails": f"📧 Fetched {si(tool_result,'count')} email(s).",
            "search_email": f"🔍 Found {si(tool_result,'count')} email(s).",
            "send_email": "✅ Email sent.",
            "delete_email": "🗑️ Email deleted.",
            "list_events": f"📅 {si(tool_result,'count')} event(s) found.",
            "create_event": "✅ Event created.",
            "delete_event": "🗑️ Event deleted.",
            "list_files": f"📁 {si(tool_result,'count')} file(s) found.",
            "upload_file": "✅ File uploaded.",
            "delete_file": "🗑️ File deleted.",
        }
        return m.get(tool_name,
                     f"Action '{tool_name}' completed for {account}.")

    # ── JSON extraction ────────────────────────────────────────────────

    def _extract_json(self, llm_text: str) -> dict:
        raw = llm_text.strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```",
                           raw, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            raw = fenced.group(1)
        s, e = raw.find("{"), raw.rfind("}")
        if s == -1 or e == -1 or e <= s:
            raise ValueError("No JSON found.")
        return json.loads(raw[s:e + 1])

    # ── Validation ─────────────────────────────────────────────────────

    def _validate_tool_call(self, tool_name: str, account: str,
                            parameters: dict[str, Any]) -> dict[str, Any]:
        if tool_name not in TOOLS:
            raise ValueError(f"Tool '{tool_name}' not allowlisted.")
        if account not in list_available_accounts():
            raise ValueError(f"Account '{account}' is invalid.")
        model_cls = TOOL_PARAMETER_MODELS.get(tool_name)
        if not model_cls:
            raise ValueError(f"No schema for '{tool_name}'.")
        try:
            v = model_cls.model_validate(parameters or {})
        except ValidationError as exc:
            raise ValueError(f"Invalid parameters: {exc}") from exc
        return v.model_dump(exclude_none=True)

    # ── Utility ────────────────────────────────────────────────────────

    @staticmethod
    def _safe_int(payload: Any, key: str) -> int:
        if isinstance(payload, dict):
            try:
                return int(payload.get(key, 0))
            except (TypeError, ValueError):
                return 0
        return 0

    def _is_accounts_question(self, low: str) -> bool:
        return any(p in low for p in (
            "what accounts do i have", "show my accounts",
            "list my accounts", "which accounts do i have"))

    def _format_accounts_overview(self) -> str:
        aliases = self.memory_repo.list_account_aliases()
        ordered = ["exam", "college", "personal", "private"]
        lines: list[str] = []
        for k in ordered:
            v = aliases.get(k)
            if v:
                lines.append(f"{k.title()} - {v}")
        for k in sorted(aliases.keys()):
            if k not in ordered:
                lines.append(f"{k.title()} - {aliases[k]}")
        return "\n".join(lines) if lines else "No accounts configured."

    def _detect_account_alias(self, low: str) -> str | None:
        for alias in self.memory_repo.list_account_aliases():
            if alias in low:
                return alias
        return None

    def _resolve_account_identifier(self, raw: str) -> str | None:
        ident = raw.strip().lower()
        if not ident:
            return None
        available = list_available_accounts()
        if ident in available:
            return ident
        for a in available:
            if a.lower() == ident:
                return a
        mapped = self.memory_repo.get_account_by_alias(ident)
        if mapped:
            if mapped in available:
                return mapped
            for a in available:
                if a.lower() == mapped.lower():
                    return a
        return None

    def _log_agent_finish(self, rid: str, started: float,
                          tool_name: str | None,
                          error_type: str | None) -> None:
        ms = int((time.perf_counter() - started) * 1000)
        _log(logging.INFO, event="agent_finish", request_id=rid,
             tool_name=tool_name, latency_ms=ms, error_type=error_type)


_AGENT = SecureHybridAgent()


def run_agent(user_message: str, session_id: str,
              request_id: str | None = None) -> AgentResult:
    """Module-level entry point. Called by FastAPI via asyncio.to_thread."""
    return _AGENT.run(user_message, session_id, request_id)

