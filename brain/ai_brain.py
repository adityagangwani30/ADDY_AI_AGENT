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
from pydantic_core import PydanticUndefined

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

# Primary account used when no account is specified or resolved.
# Falls back to the first available account from accounts.json.
DEFAULT_ACCOUNT_ALIAS = "personal"

LOGGER = logging.getLogger(__name__)

_TOOL_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="agent-tools")

FALLBACK_MODEL_MESSAGE = FALLBACK_MESSAGE
DAILY_LIMIT_MESSAGE = "Daily AI limit reached. Please try again tomorrow."
DAILY_LLM_CALL_LIMIT = 100

TOOL_TIMEOUT_SECONDS = 8
MAX_HISTORY_MESSAGES = 6
MAX_LLM_CALLS_PER_REQUEST = 4
MAX_GMAIL_RESULTS = 50
DEFAULT_EMAIL_PARAMS = {
    "query": "in:inbox",
    "max_results": 5,
}
GLOBAL_FALLBACK_RESPONSE = (
    "⚠️ I couldn’t process that request properly. Please try again."
)


def _log(level: int, **payload: Any) -> None:
    LOGGER.log(level, json.dumps(payload, default=str))


def normalize_email_params(params: dict[str, Any] | None) -> dict[str, Any]:
    normalized = dict(DEFAULT_EMAIL_PARAMS)
    incoming = dict(params or {})

    query = str(incoming.get("query", "") or "").strip()
    normalized["query"] = query or DEFAULT_EMAIL_PARAMS["query"]

    max_results = incoming.get("max_results", DEFAULT_EMAIL_PARAMS["max_results"])
    try:
        normalized["max_results"] = max(1, min(int(max_results), MAX_GMAIL_RESULTS))
    except (TypeError, ValueError):
        normalized["max_results"] = DEFAULT_EMAIL_PARAMS["max_results"]

    return normalized


def _schema_default_data(schema: Any) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    model_fields = getattr(schema, "model_fields", {}) or {}
    for name, field in model_fields.items():
        default = getattr(field, "default", PydanticUndefined)
        default_factory = getattr(field, "default_factory", None)
        if default is not PydanticUndefined:
            defaults[name] = default
        elif callable(default_factory):
            try:
                defaults[name] = default_factory()
            except Exception:
                continue
    return defaults


def safe_validate(schema: Any, data: dict[str, Any] | None, default_data: dict[str, Any] | None) -> dict[str, Any]:
    candidate = dict(data or {})
    fallback = dict(default_data or _schema_default_data(schema))

    try:
        validated = schema.model_validate(candidate)
        normalized = validated.model_dump(exclude_none=True)
        _log(
            logging.INFO,
            event="validation_success",
            schema=getattr(schema, "__name__", str(schema)),
            normalized=normalized,
            fallback_used=False,
        )
        return normalized
    except Exception as exc:
        _log(
            logging.WARNING,
            event="validation_failed",
            schema=getattr(schema, "__name__", str(schema)),
            candidate=candidate,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

    try:
        validated = schema.model_validate(fallback)
        normalized = validated.model_dump(exclude_none=True)
        _log(
            logging.INFO,
            event="validation_fallback_used",
            schema=getattr(schema, "__name__", str(schema)),
            normalized=normalized,
            fallback_used=True,
        )
        return normalized
    except Exception as exc:
        _log(
            logging.ERROR,
            event="validation_fallback_failed",
            schema=getattr(schema, "__name__", str(schema)),
            fallback=fallback,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        return fallback


def _tool_default_params(tool_name: str) -> dict[str, Any]:
    if tool_name in {"list_emails", "search_email"}:
        return dict(DEFAULT_EMAIL_PARAMS)
    schema = TOOL_PARAMETER_MODELS.get(tool_name)
    return _schema_default_data(schema) if schema else {}


def _simplified_email_params(params: dict[str, Any] | None) -> dict[str, Any]:
    simplified = normalize_email_params(params)
    simplified["query"] = DEFAULT_EMAIL_PARAMS["query"]
    simplified["max_results"] = min(simplified.get("max_results", 1), 1)
    return simplified


def _normalize_tool_params(tool_name: str, params: dict[str, Any] | None) -> dict[str, Any]:
    if tool_name in {"list_emails", "search_email"}:
        return normalize_email_params(params)
    return dict(params or {})


def safe_tool_call(
    tool_fn: Any,
    params: dict[str, Any] | None,
    default_params: dict[str, Any] | None,
    *,
    account: str,
    request_id: str = "unknown",
    tool_name: str | None = None,
) -> dict[str, Any]:
    name = tool_name or getattr(tool_fn, "__name__", "unknown_tool")
    primary = dict(params or {})
    defaults = dict(default_params or {})
    attempts: list[dict[str, Any]] = [primary]
    if defaults and defaults != primary:
        attempts.append(defaults)
    if name in {"list_emails", "search_email"}:
        simplified = _simplified_email_params(primary)
        if simplified not in attempts:
            attempts.append(simplified)

    last_error: str | None = None
    started = time.perf_counter()
    signature = inspect.signature(tool_fn)

    for attempt_index, attempt_params in enumerate(attempts, start=1):
        _log(
            logging.INFO,
            event="tool_input",
            request_id=request_id,
            tool_name=name,
            attempt=attempt_index,
            params=attempt_params,
        )
        try:
            call_kwargs = dict(attempt_params)
            if "request_id" in signature.parameters:
                call_kwargs["request_id"] = request_id
            future = _TOOL_EXECUTOR.submit(tool_fn, account, **call_kwargs)
            result = future.result(timeout=TOOL_TIMEOUT_SECONDS)
            latency_ms = int((time.perf_counter() - started) * 1000)
            _log(
                logging.INFO,
                event="tool_call_success",
                request_id=request_id,
                tool_name=name,
                attempt=attempt_index,
                latency_ms=latency_ms,
                error_type=None,
            )
            return {
                "ok": True,
                "result": result,
                "error": None,
                "attempt": attempt_index,
                "params_used": attempt_params,
                "latency_ms": latency_ms,
            }
        except FuturesTimeoutError as exc:
            try:
                future.cancel()
            except Exception:
                pass
            last_error = str(exc)
            _log(
                logging.ERROR,
                event="tool_call_failed",
                request_id=request_id,
                tool_name=name,
                attempt=attempt_index,
                error_type="TimeoutError",
                error_message=f"Tool '{name}' timed out after {TOOL_TIMEOUT_SECONDS} seconds.",
                params=attempt_params,
            )
        except Exception as exc:
            last_error = str(exc)
            _log(
                logging.ERROR,
                event="tool_call_failed",
                request_id=request_id,
                tool_name=name,
                attempt=attempt_index,
                error_type=type(exc).__name__,
                error_message=str(exc),
                params=attempt_params,
            )

    latency_ms = int((time.perf_counter() - started) * 1000)
    return {
        "ok": False,
        "result": None,
        "error": last_error or GLOBAL_FALLBACK_RESPONSE,
        "attempt": len(attempts),
        "params_used": attempts[-1] if attempts else {},
        "latency_ms": latency_ms,
    }


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

    Returns a structured result and never raises for validation or tool errors.
    """
    started = time.perf_counter()
    if tool_name not in TOOLS:
        latency_ms = int((time.perf_counter() - started) * 1000)
        _log(
            logging.ERROR,
            event="tool_error",
            request_id=request_id,
            tool_name=tool_name,
            latency_ms=latency_ms,
            error_type="ToolNotRegistered",
            error_message=f"Tool '{tool_name}' is not registered.",
        )
        return {
            "latency_ms": latency_ms,
            "ok": False,
            "result": None,
            "error": GLOBAL_FALLBACK_RESPONSE,
            "attempt": 0,
            "params_used": {},
        }

    tool_fn = TOOLS[tool_name]
    schema = TOOL_PARAMETER_MODELS.get(tool_name)
    normalized_params = _normalize_tool_params(tool_name, parameters)
    default_params = _tool_default_params(tool_name)
    validated_params = safe_validate(
        schema,
        normalized_params,
        default_params,
    ) if schema else normalized_params

    _log(
        logging.INFO,
        event="tool_normalized_params",
        request_id=request_id,
        tool_name=tool_name,
        latency_ms=None,
        error_type=None,
        normalized_params=validated_params,
        default_params=default_params,
    )

    _log(
        logging.INFO,
        event="tool_start",
        request_id=request_id,
        tool_name=tool_name,
        latency_ms=None,
        error_type=None,
    )

    try:
        call_result = safe_tool_call(
            tool_fn,
            validated_params,
            default_params,
            account=account,
            request_id=request_id,
            tool_name=tool_name,
        )
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        _log(
            logging.ERROR,
            event="tool_error",
            request_id=request_id,
            tool_name=tool_name,
            latency_ms=latency_ms,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        return {
            "latency_ms": latency_ms,
            "ok": False,
            "result": None,
            "error": GLOBAL_FALLBACK_RESPONSE,
            "attempt": 0,
            "params_used": validated_params,
        }

    latency_ms = int((time.perf_counter() - started) * 1000)
    _log(
        logging.INFO,
        event="tool_finish",
        request_id=request_id,
        tool_name=tool_name,
        latency_ms=latency_ms,
        error_type=None if call_result.get("ok") else "ToolRetryFailed",
    )
    call_result["latency_ms"] = latency_ms
    return call_result


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
        cleaned = (user_message or "").strip()
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

        # Auto-resolve: fall back to default account if none resolved
        if not account:
            account = self._get_default_account()
            if account:
                _log(logging.INFO, event="auto_default_account",
                     request_id=rid, tool_name=tool_name,
                     latency_ms=None, error_type=None,
                     resolved_account=account)
            else:
                msg = ("No connected accounts found. "
                       "Please add an account first.")
                self.memory_repo.add_conversation(
                    session_id, "assistant", msg)
                self._log_agent_finish(
                    rid, started, tool_name, "NoAccountsAvailable")
                return AgentResult(request_id=rid, status="error",
                                   message=msg,
                                   error_type="NoAccountsAvailable")

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

        _log(
            logging.INFO,
            event="planner_llm_output",
            request_id=request_id,
            tool_name="planner",
            latency_ms=None,
            error_type=None,
            output=text,
        )

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
            r = execute_tool(
                tool_name=tool_name,
                account=account,
                parameters=parameters,
                request_id=request_id,
            )
            if r.get("ok"):
                return r.get("result"), int(r.get("latency_ms", 0)), None
            return None, int(r.get("latency_ms", 0)), str(r.get("error") or GLOBAL_FALLBACK_RESPONSE)
        except Exception as exc:
            _log(
                logging.ERROR,
                event="tool_error",
                request_id=request_id,
                tool_name=tool_name,
                latency_ms=None,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
            return None, 0, GLOBAL_FALLBACK_RESPONSE

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
                return "📧 Email Details:\n\nNo emails found."
            lines = []
            for i, m in enumerate(msgs, 1):
                body = str(m.get("body", "") or "").strip()
                body_preview = body[:200] + ("..." if len(body) > 200 else "") if body else ""
                lines.append(
                    f"• Email {i}:\n"
                    f"  Subject: {m.get('subject', m.get('Subject', 'No Subject'))}\n"
                    f"  From: {m.get('from', m.get('From', 'Unknown Sender'))}\n"
                    f"  Snippet: {m.get('snippet', '') or 'No snippet available'}"
                    + (f"\n  Body: {body_preview}" if body_preview else ""))
            return ("📧 Email Details:\n\n"
                    f"Total: {tool_result.get('count', len(msgs))} email(s)\n\n"
                    + "\n\n".join(lines))

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
        normalized = dict(parameters or {})
        default_data = _tool_default_params(tool_name)

        if tool_name not in TOOLS:
            _log(
                logging.ERROR,
                event="validation_failed",
                request_id="unknown",
                tool_name=tool_name,
                candidate=normalized,
                error_type="ToolNotRegistered",
                error_message=f"Tool '{tool_name}' not allowlisted.",
            )
            return default_data

        if account not in list_available_accounts():
            _log(
                logging.ERROR,
                event="validation_failed",
                request_id="unknown",
                tool_name=tool_name,
                candidate=normalized,
                error_type="InvalidAccount",
                error_message=f"Account '{account}' is invalid.",
            )
            return default_data

        schema = TOOL_PARAMETER_MODELS.get(tool_name)
        if not schema:
            _log(
                logging.WARNING,
                event="validation_no_schema",
                request_id="unknown",
                tool_name=tool_name,
                candidate=normalized,
            )
            return default_data

        if tool_name in {"list_emails", "search_email"}:
            normalized = normalize_email_params(normalized)

        validated = safe_validate(schema, normalized, default_data)
        _log(
            logging.INFO,
            event="validation_normalized",
            request_id="unknown",
            tool_name=tool_name,
            normalized=validated,
            default_data=default_data,
        )
        return validated

    # ── Utility ────────────────────────────────────────────────────────

    @staticmethod
    def _safe_int(payload: Any, key: str) -> int:
        if isinstance(payload, dict):
            try:
                return int(payload.get(key, 0))
            except (TypeError, ValueError):
                return 0
        return 0

    @staticmethod
    def _clamp_max_results(value: Any) -> int:
        try:
            return max(1, min(int(value), MAX_GMAIL_RESULTS))
        except (TypeError, ValueError):
            return 1

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

    def _get_default_account(self) -> str | None:
        """Return the best default account, preferring the configured alias."""
        # Try the configured default alias first
        mapped = self.memory_repo.get_account_by_alias(DEFAULT_ACCOUNT_ALIAS)
        if mapped:
            resolved = self._resolve_account_identifier(mapped)
            if resolved:
                return resolved
        # Fall back to the first available account
        available = list_available_accounts()
        return available[0] if available else None

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
    rid = request_id or str(uuid.uuid4())
    try:
        result = _AGENT.run(user_message, session_id, request_id)
        if isinstance(result, AgentResult):
            return result
        _log(
            logging.ERROR,
            event="agent_run_failed",
            request_id=rid,
            tool_name=None,
            latency_ms=None,
            error_type="InvalidAgentResult",
        )
        return AgentResult(
            request_id=rid,
            status="error",
            message=GLOBAL_FALLBACK_RESPONSE,
            error_type="InvalidAgentResult",
        )
    except Exception as exc:
        _log(
            logging.ERROR,
            event="agent_run_failed",
            request_id=rid,
            tool_name=None,
            latency_ms=None,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
        return AgentResult(
            request_id=rid,
            status="error",
            message=GLOBAL_FALLBACK_RESPONSE,
            error_type=type(exc).__name__,
        )

