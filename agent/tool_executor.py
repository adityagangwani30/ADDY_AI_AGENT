from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any

from brain.llm_provider import call_llm
from brain.tool_registry import TOOLS, normalize_tool_name
from domain.schemas import (
    CalendarCreateActionParams,
    CalendarDeleteActionParams,
    CalendarEditActionParams,
    CalendarListActionParams,
    DriveRetrieveActionParams,
    DriveSearchActionParams,
    DriveShareActionParams,
    DriveUploadActionParams,
    GitHubChangelogParams,
    GitHubCodeSummaryParams,
    GitHubCommitSummaryParams,
    GitHubDraftCommitParams,
    GitHubIssuesParams,
    GitHubListRepositoriesParams,
    GitHubPullRequestsParams,
    GitHubRepoRefParams,
    GitHubTracebackParams,
    GmailDraftParams,
    GmailReadParams,
    GmailSendParams,
    GmailSummarizeParams,
)
from memory.storage import SQLiteMemoryRepository

LOGGER = logging.getLogger(__name__)
_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="agent-exec")
_TOOL_TIMEOUT_SECONDS = 12

INTENT_TO_TOOL = {
    "gmail_read": "gmail_read",
    "gmail_summarize": "gmail_read",
    "gmail_draft": "gmail_draft",
    "gmail_send": "gmail_send",
    "calendar_create": "calendar_create",
    "calendar_edit": "calendar_edit",
    "calendar_delete": "calendar_delete",
    "calendar_list": "calendar_list",
    "drive_upload": "drive_upload",
    "drive_search": "drive_search",
    "drive_retrieve": "drive_retrieve",
    "drive_share": "drive_share",
    "github_list_repositories": "github_list_repositories",
    "github_repository_summary": "github_repository_summary",
    "github_commits": "github_commits",
    "github_issues": "github_issues",
    "github_pull_requests": "github_pull_requests",
    "github_changelog": "github_changelog",
    "github_draft_commit_message": "github_draft_commit_message",
    "github_code_summary": "github_code_summary",
    "github_traceback_explain": "github_traceback_explain",
    "github_project_dashboard": "github_project_dashboard",
}

PARAM_MODELS = {
    "gmail_read": GmailReadParams,
    "gmail_summarize": GmailSummarizeParams,
    "gmail_draft": GmailDraftParams,
    "gmail_send": GmailSendParams,
    "calendar_create": CalendarCreateActionParams,
    "calendar_edit": CalendarEditActionParams,
    "calendar_delete": CalendarDeleteActionParams,
    "calendar_list": CalendarListActionParams,
    "drive_upload": DriveUploadActionParams,
    "drive_search": DriveSearchActionParams,
    "drive_retrieve": DriveRetrieveActionParams,
    "drive_share": DriveShareActionParams,
    "github_list_repositories": GitHubListRepositoriesParams,
    "github_repository_summary": GitHubRepoRefParams,
    "github_commits": GitHubCommitSummaryParams,
    "github_issues": GitHubIssuesParams,
    "github_pull_requests": GitHubPullRequestsParams,
    "github_changelog": GitHubChangelogParams,
    "github_draft_commit_message": GitHubDraftCommitParams,
    "github_code_summary": GitHubCodeSummaryParams,
    "github_traceback_explain": GitHubTracebackParams,
    "github_project_dashboard": GitHubRepoRefParams,
}


class ToolExecutor:
    risky_intents = {"gmail_send", "calendar_delete"}

    def __init__(self, audit_repo: SQLiteMemoryRepository | None = None) -> None:
        self.audit_repo = audit_repo or SQLiteMemoryRepository()

    def validate(self, intent: str, parameters: dict[str, Any]) -> dict[str, Any]:
        model = PARAM_MODELS.get(intent)
        if model is None:
            return dict(parameters or {})
        validated = model.model_validate(parameters or {})
        payload = validated.model_dump(exclude_none=True)
        if intent.startswith("github_") and not payload.get("repository"):
            repository = str(payload.get("owner") or "").strip()
            name = str(payload.get("name") or "").strip()
            if repository and name:
                payload["repository"] = f"{repository}/{name}"
        return payload

    def execute(self, intent: str, account: str, parameters: dict[str, Any], request_id: str, session_id: str | None = None) -> dict[str, Any]:
        tool_name = normalize_tool_name(INTENT_TO_TOOL.get(intent, intent))
        if not tool_name:
            return {"ok": False, "error": f"Unsupported intent: {intent}", "result": None, "latency_ms": 0}

        if intent == "gmail_summarize":
            read_result = self._dispatch(
                tool_name="gmail_read",
                account=account,
                parameters=parameters,
                request_id=request_id,
                session_id=session_id,
            )
            if not read_result.get("ok"):
                return read_result
            return self._summarize_emails(read_result, request_id)

        return self._dispatch(
            tool_name=tool_name,
            account=account,
            parameters=parameters,
            request_id=request_id,
            session_id=session_id,
        )

    def _dispatch(self, tool_name: str, account: str, parameters: dict[str, Any], request_id: str, session_id: str | None = None) -> dict[str, Any]:
        normalized_tool_name = normalize_tool_name(tool_name)
        fn = TOOLS.get(normalized_tool_name) or TOOLS.get(tool_name)
        if fn is None:
            return {"ok": False, "error": f"Tool not registered: {tool_name}", "result": None, "latency_ms": 0}

        started = time.perf_counter()
        kwargs = dict(parameters or {})
        try:
            code_vars = fn.__code__.co_varnames
            if "request_id" in code_vars:
                kwargs["request_id"] = request_id
            if session_id is not None and "session_id" in code_vars:
                kwargs["session_id"] = session_id
        except Exception:
            pass

        future = _EXECUTOR.submit(fn, account, **kwargs)
        try:
            result = future.result(timeout=_TOOL_TIMEOUT_SECONDS)
            payload = {
                "ok": True,
                "result": result,
                "error": None,
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }
            self.audit_repo.record_executed_action(
                action_type=normalized_tool_name,
                parameters=kwargs,
                success=True,
                user_id=account,
                execution_time_ms=payload["latency_ms"],
            )
            return payload
        except FuturesTimeoutError:
            future.cancel()
            payload = {
                "ok": False,
                "result": None,
                "error": f"Tool timeout after {_TOOL_TIMEOUT_SECONDS}s",
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }
            self.audit_repo.record_executed_action(
                action_type=normalized_tool_name,
                parameters=kwargs,
                success=False,
                error_message=payload["error"],
                user_id=account,
                execution_time_ms=payload["latency_ms"],
            )
            return payload
        except Exception as exc:
            payload = {
                "ok": False,
                "result": None,
                "error": str(exc),
                "latency_ms": int((time.perf_counter() - started) * 1000),
            }
            self.audit_repo.record_executed_action(
                action_type=normalized_tool_name,
                parameters=kwargs,
                success=False,
                error_message=payload["error"],
                user_id=account,
                execution_time_ms=payload["latency_ms"],
            )
            return payload

    def _summarize_emails(self, read_result: dict[str, Any], request_id: str) -> dict[str, Any]:
        payload = read_result.get("result") or {}
        messages = payload.get("messages", []) if isinstance(payload, dict) else []
        if not messages:
            return {
                "ok": True,
                "result": {"summary": "No emails found.", "count": 0, "messages": []},
                "error": None,
                "latency_ms": int(read_result.get("latency_ms") or 0),
            }

        lines: list[str] = []
        for msg in messages[:12]:
            lines.append(
                f"From: {msg.get('from', '')}\\n"
                f"Subject: {msg.get('subject', '')}\\n"
                f"Snippet: {msg.get('snippet', '')}"
            )

        prompt = "Summarize these emails with priorities and action items:\\n\\n" + "\\n\\n".join(lines)
        summary = call_llm(
            prompt=prompt,
            system_prompt="You summarize emails for a personal assistant. Keep it concise and actionable.",
            request_id=request_id,
            phase="gmail_summarize",
            timeout_seconds=12,
        )
        if not summary:
            summary = "I fetched your emails but could not summarize them right now."

        return {
            "ok": True,
            "result": {"summary": summary, "count": len(messages), "messages": messages},
            "error": None,
            "latency_ms": int(read_result.get("latency_ms") or 0),
        }
