from __future__ import annotations

import base64
import json
import logging
import socket
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from email.mime.text import MIMEText
from typing import Any, Callable

from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from auth.google_auth_manager import get_credentials

LOGGER = logging.getLogger(__name__)
_GMAIL_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="gmail-exec")
_GMAIL_TIMEOUT_SECONDS = 15
_GMAIL_MAX_RETRIES = 2


def _log(level: int, **payload: Any) -> None:
    LOGGER.log(level, json.dumps(payload, default=str))


def _build_service(account: str):
    creds = get_credentials(account)
    return build("gmail", "v1", credentials=creds, cache_discovery=False)


def _error_payload(tool_name: str, request_id: str, error_type: str, message: str) -> dict[str, Any]:
    return {
        "component": "gmail_tool",
        "request_id": request_id,
        "tool_name": tool_name,
        "error_type": error_type,
        "message": message,
    }


def _execute_with_timeout(
    execute_fn: Callable[[], Any],
    *,
    tool_name: str,
    request_id: str,
) -> Any:
    started = time.perf_counter()
    _log(
        logging.INFO,
        event="gmail_api_call_start",
        request_id=request_id,
        tool_name=tool_name,
        latency_ms=None,
        error_type=None,
    )

    future = _GMAIL_EXECUTOR.submit(execute_fn)
    try:
        response = future.result(timeout=_GMAIL_TIMEOUT_SECONDS)
        latency_ms = int((time.perf_counter() - started) * 1000)
        _log(
            logging.INFO,
            event="gmail_api_call_finish",
            request_id=request_id,
            tool_name=tool_name,
            latency_ms=latency_ms,
            error_type=None,
        )
        return response
    except FuturesTimeoutError as exc:
        future.cancel()
        latency_ms = int((time.perf_counter() - started) * 1000)
        _log(
            logging.ERROR,
            event="gmail_api_call_timeout",
            request_id=request_id,
            tool_name=tool_name,
            latency_ms=latency_ms,
            error_type="TimeoutError",
        )
        raise TimeoutError(
            json.dumps(
                _error_payload(
                    tool_name=tool_name,
                    request_id=request_id,
                    error_type="TimeoutError",
                    message=f"Gmail API call timed out after {_GMAIL_TIMEOUT_SECONDS} seconds.",
                )
            )
        ) from exc
    except HttpError as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        _log(
            logging.ERROR,
            event="gmail_api_call_error",
            request_id=request_id,
            tool_name=tool_name,
            latency_ms=latency_ms,
            error_type="HttpError",
        )
        raise RuntimeError(
            json.dumps(
                _error_payload(
                    tool_name=tool_name,
                    request_id=request_id,
                    error_type="HttpError",
                    message=str(exc),
                )
            )
        ) from exc
    except (ConnectionError, socket.timeout, OSError, TimeoutError) as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        _log(
            logging.ERROR,
            event="gmail_api_call_error",
            request_id=request_id,
            tool_name=tool_name,
            latency_ms=latency_ms,
            error_type=type(exc).__name__,
        )
        raise RuntimeError(
            json.dumps(
                _error_payload(
                    tool_name=tool_name,
                    request_id=request_id,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            )
        ) from exc


def list_emails(account: str, max_results: int = 5, request_id: str = "unknown") -> dict:
    if max_results < 1:
        raise ValueError("max_results must be >= 1")

    tool_name = "list_emails"
    started = time.perf_counter()
    _log(
        logging.INFO,
        event="gmail_tool_start",
        request_id=request_id,
        tool_name=tool_name,
        latency_ms=None,
        error_type=None,
    )

    service = _build_service(account)
    request = service.users().messages().list(userId="me", maxResults=max_results)
    results = _execute_with_timeout(
        lambda: request.execute(num_retries=_GMAIL_MAX_RETRIES),
        tool_name=tool_name,
        request_id=request_id,
    )

    messages = results.get("messages", [])
    latency_ms = int((time.perf_counter() - started) * 1000)
    _log(
        logging.INFO,
        event="gmail_tool_finish",
        request_id=request_id,
        tool_name=tool_name,
        latency_ms=latency_ms,
        error_type=None,
    )
    return {"count": len(messages), "messages": messages}


def send_email(account: str, to: str, subject: str, body: str, request_id: str = "unknown") -> dict:
    if not to or not subject:
        raise ValueError("Both 'to' and 'subject' are required.")

    tool_name = "send_email"
    started = time.perf_counter()
    _log(
        logging.INFO,
        event="gmail_tool_start",
        request_id=request_id,
        tool_name=tool_name,
        latency_ms=None,
        error_type=None,
    )

    service = _build_service(account)

    message = MIMEText(body)
    message["to"] = to
    message["subject"] = subject

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    request = service.users().messages().send(userId="me", body={"raw": raw})
    sent = _execute_with_timeout(
        lambda: request.execute(num_retries=_GMAIL_MAX_RETRIES),
        tool_name=tool_name,
        request_id=request_id,
    )

    latency_ms = int((time.perf_counter() - started) * 1000)
    _log(
        logging.INFO,
        event="gmail_tool_finish",
        request_id=request_id,
        tool_name=tool_name,
        latency_ms=latency_ms,
        error_type=None,
    )
    return {"id": sent.get("id"), "thread_id": sent.get("threadId")}


def search_email(
    account: str,
    query: str,
    max_results: int = 10,
    request_id: str = "unknown",
) -> dict:
    if not query:
        raise ValueError("query is required")

    tool_name = "search_email"
    started = time.perf_counter()
    _log(
        logging.INFO,
        event="gmail_tool_start",
        request_id=request_id,
        tool_name=tool_name,
        latency_ms=None,
        error_type=None,
    )

    service = _build_service(account)
    request = service.users().messages().list(userId="me", q=query, maxResults=max_results)
    results = _execute_with_timeout(
        lambda: request.execute(num_retries=_GMAIL_MAX_RETRIES),
        tool_name=tool_name,
        request_id=request_id,
    )

    messages = results.get("messages", [])
    latency_ms = int((time.perf_counter() - started) * 1000)
    _log(
        logging.INFO,
        event="gmail_tool_finish",
        request_id=request_id,
        tool_name=tool_name,
        latency_ms=latency_ms,
        error_type=None,
    )
    return {"count": len(messages), "messages": messages}


def delete_email(account: str, message_id: str, request_id: str = "unknown") -> dict:
    if not message_id:
        raise ValueError("message_id is required")

    tool_name = "delete_email"
    started = time.perf_counter()
    _log(
        logging.INFO,
        event="gmail_tool_start",
        request_id=request_id,
        tool_name=tool_name,
        latency_ms=None,
        error_type=None,
    )

    service = _build_service(account)
    request = service.users().messages().delete(userId="me", id=message_id)
    _execute_with_timeout(
        lambda: request.execute(num_retries=_GMAIL_MAX_RETRIES),
        tool_name=tool_name,
        request_id=request_id,
    )

    latency_ms = int((time.perf_counter() - started) * 1000)
    _log(
        logging.INFO,
        event="gmail_tool_finish",
        request_id=request_id,
        tool_name=tool_name,
        latency_ms=latency_ms,
        error_type=None,
    )
    return {"deleted": True, "message_id": message_id}
