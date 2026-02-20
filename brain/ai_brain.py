from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any

from pydantic import ValidationError

from auth.google_auth_manager import list_available_accounts
from brain.system_prompt import SYSTEM_PROMPT
from brain.tool_registry import DESTRUCTIVE_TOOLS, TOOLS, TOOL_PARAMETER_MODELS
from domain.schemas import AgentDecision, AgentResult
from integrations.gemini_client import GeminiClient, GeminiIntegrationError
from memory.storage import MemoryRepository, SQLiteMemoryRepository

LOGGER = logging.getLogger(__name__)


def _log(level: int, **payload: Any) -> None:
    LOGGER.log(level, json.dumps(payload, default=str))


def execute_tool(
    tool_name: str,
    account: str,
    parameters: dict[str, Any],
    request_id: str,
) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        tool_fn = TOOLS[tool_name]
    except KeyError as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        _log(
            logging.ERROR,
            event="tool_execution_error",
            request_id=request_id,
            tool_name=tool_name,
            account=account,
            latency_ms=latency_ms,
            error_type="UnknownTool",
        )
        raise ValueError(f"Tool '{tool_name}' is not registered.") from exc

    try:
        result = tool_fn(account, **parameters)
    except Exception as exc:
        latency_ms = int((time.perf_counter() - start) * 1000)
        _log(
            logging.ERROR,
            event="tool_execution_error",
            request_id=request_id,
            tool_name=tool_name,
            account=account,
            latency_ms=latency_ms,
            error_type=type(exc).__name__,
        )
        raise RuntimeError(f"Tool '{tool_name}' failed: {exc}") from exc

    latency_ms = int((time.perf_counter() - start) * 1000)
    _log(
        logging.INFO,
        event="tool_execution_success",
        request_id=request_id,
        tool_name=tool_name,
        account=account,
        latency_ms=latency_ms,
        error_type=None,
    )
    return {"latency_ms": latency_ms, "result": result}


class SecureHybridAgent:
    def __init__(
        self,
        gemini_client: GeminiClient | None = None,
        memory_repo: MemoryRepository | None = None,
    ) -> None:
        self.gemini_client = gemini_client or GeminiClient()
        self.memory_repo = memory_repo or SQLiteMemoryRepository()

    def run(self, user_message: str, session_id: str, request_id: str | None = None) -> AgentResult:
        rid = request_id or str(uuid.uuid4())
        cleaned = user_message.strip()
        if not cleaned:
            return AgentResult(
                request_id=rid,
                status="error",
                message="Message cannot be empty.",
                error_type="ValidationError",
            )

        self.memory_repo.add_conversation(session_id, "user", cleaned)

        lower = cleaned.lower()
        if lower in {"confirm", "confirm action", "yes confirm"}:
            return self._handle_confirmation(session_id, rid)
        if lower in {"cancel", "deny", "reject"}:
            return self._handle_cancellation(session_id, rid)

        try:
            llm_text = self._generate_decision(cleaned, session_id)
            decision = self._parse_decision(llm_text)
        except (GeminiIntegrationError, ValueError) as exc:
            result = AgentResult(
                request_id=rid,
                status="error",
                message=str(exc),
                error_type=type(exc).__name__,
            )
            self.memory_repo.add_conversation(session_id, "assistant", result.message)
            return result

        if decision.type == "response":
            response_text = decision.response or "I could not process that request."
            result = AgentResult(request_id=rid, status="ok", message=response_text)
            self.memory_repo.add_conversation(session_id, "assistant", response_text)
            return result

        try:
            tool_name = decision.tool or ""
            account = decision.account or ""
            validated_parameters = self._validate_tool_call(
                tool_name=tool_name,
                account=account,
                parameters=decision.parameters,
            )
        except ValueError as exc:
            result = AgentResult(
                request_id=rid,
                status="error",
                message=str(exc),
                error_type="ToolValidationError",
            )
            self.memory_repo.add_conversation(session_id, "assistant", result.message)
            return result

        self.memory_repo.set_account_preference(session_id, account)

        if tool_name in DESTRUCTIVE_TOOLS:
            self.memory_repo.save_pending_confirmation(
                session_id=session_id,
                tool_name=tool_name,
                account=account,
                parameters=validated_parameters,
            )
            confirmation_message = (
                f"Confirmation required for '{tool_name}' on account '{account}'. "
                "Reply with 'confirm' to execute or 'cancel' to abort."
            )
            result = AgentResult(
                request_id=rid,
                status="confirmation_required",
                message=confirmation_message,
                tool_name=tool_name,
                account=account,
                data={"parameters": validated_parameters},
            )
            self.memory_repo.add_conversation(session_id, "assistant", confirmation_message)
            return result

        return self._execute_and_build_result(
            session_id=session_id,
            request_id=rid,
            tool_name=tool_name,
            account=account,
            parameters=validated_parameters,
        )

    def _handle_confirmation(self, session_id: str, request_id: str) -> AgentResult:
        pending = self.memory_repo.get_pending_confirmation(session_id)
        if not pending:
            return AgentResult(
                request_id=request_id,
                status="error",
                message="No pending destructive action to confirm.",
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

    def _handle_cancellation(self, session_id: str, request_id: str) -> AgentResult:
        pending = self.memory_repo.get_pending_confirmation(session_id)
        if not pending:
            return AgentResult(
                request_id=request_id,
                status="ok",
                message="No pending destructive action. Nothing was cancelled.",
            )

        self.memory_repo.clear_pending_confirmation(session_id)
        response = "Pending destructive action cancelled."
        self.memory_repo.add_conversation(session_id, "assistant", response)
        return AgentResult(request_id=request_id, status="ok", message=response)

    def _execute_and_build_result(
        self,
        session_id: str,
        request_id: str,
        tool_name: str,
        account: str,
        parameters: dict[str, Any],
    ) -> AgentResult:
        try:
            tool_response = execute_tool(
                tool_name=tool_name,
                account=account,
                parameters=parameters,
                request_id=request_id,
            )
        except (RuntimeError, ValueError) as exc:
            result = AgentResult(
                request_id=request_id,
                status="error",
                message=str(exc),
                tool_name=tool_name,
                account=account,
                error_type=type(exc).__name__,
            )
            self.memory_repo.add_conversation(session_id, "assistant", result.message)
            return result

        message = f"Action completed using '{tool_name}'."
        result = AgentResult(
            request_id=request_id,
            status="ok",
            message=message,
            tool_name=tool_name,
            account=account,
            latency_ms=int(tool_response["latency_ms"]),
            data=tool_response["result"],
        )
        self.memory_repo.add_conversation(session_id, "assistant", message)
        return result

    def _generate_decision(self, user_message: str, session_id: str) -> str:
        history = self.memory_repo.get_conversation(session_id=session_id, limit=8)
        history_text = "\n".join(f"{entry['role']}: {entry['content']}" for entry in history)
        preferred_account = self.memory_repo.get_account_preference(session_id)

        composed_user_message = (
            "Conversation context:\n"
            f"{history_text}\n\n"
            f"Preferred account (if available): {preferred_account or 'none'}\n"
            f"Current user message: {user_message}"
        )

        return self.gemini_client.generate_json_decision(
            system_instruction=SYSTEM_PROMPT,
            user_message=composed_user_message,
        )

    def _parse_decision(self, llm_text: str) -> AgentDecision:
        raw = llm_text.strip()
        payload_candidate = self._extract_json(raw)

        try:
            payload = json.loads(payload_candidate)
        except json.JSONDecodeError as exc:
            raise ValueError("Model output was not valid JSON; tool execution blocked.") from exc

        try:
            return AgentDecision.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(f"Model JSON failed schema validation: {exc}") from exc

    @staticmethod
    def _extract_json(text: str) -> str:
        fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            return fenced.group(1)

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return text

    def _validate_tool_call(
        self,
        tool_name: str,
        account: str,
        parameters: dict[str, Any],
    ) -> dict[str, Any]:
        if tool_name not in TOOLS:
            raise ValueError(f"Tool '{tool_name}' is not allowlisted.")

        available_accounts = list_available_accounts()
        if account not in available_accounts:
            raise ValueError(
                f"Account '{account}' is invalid. Allowed accounts: {sorted(available_accounts)}"
            )

        model_cls = TOOL_PARAMETER_MODELS.get(tool_name)
        if model_cls is None:
            raise ValueError(f"No parameter schema configured for tool '{tool_name}'.")

        try:
            validated = model_cls.model_validate(parameters)
        except ValidationError as exc:
            raise ValueError(f"Invalid parameters for tool '{tool_name}': {exc}") from exc

        return validated.model_dump(exclude_none=True)


_AGENT = SecureHybridAgent()


def run_agent(user_message: str, session_id: str, request_id: str | None = None) -> AgentResult:
    return _AGENT.run(user_message=user_message, session_id=session_id, request_id=request_id)
