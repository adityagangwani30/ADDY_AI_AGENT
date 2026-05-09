"""
Unified LLM provider abstraction layer.

Routes all LLM calls through Groq (primary) with automatic fallback to
NVIDIA AI when the primary provider fails, times out, or returns empty output.

Usage::

    from brain.llm_provider import call_llm

    result = call_llm(
        prompt="Summarize these emails",
        system_prompt="You are a helpful assistant.",
        request_id="abc-123",
        phase="summarization",
    )
"""
from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any

import requests

from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    LLM_TIMEOUT_SECONDS,
    NVIDIA_API_KEY,
    NVIDIA_MODEL,
)

LOGGER = logging.getLogger(__name__)

_LLM_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="llm-provider")

# Provider registry: ordered list of (name, caller_fn)
_PROVIDER_ORDER = ["groq", "nvidia"]

FALLBACK_MESSAGE = (
    "I'm having trouble accessing my reasoning model right now. Please try again."
)


def _log(level: int, **payload: Any) -> None:
    LOGGER.log(level, json.dumps(payload, default=str))


# ── Provider adapters ──────────────────────────────────────────────────


def _call_groq(prompt: str, system_prompt: str, timeout: int) -> str:
    """
    Call the Groq API via its OpenAI-compatible chat completions endpoint.

    Args:
        prompt: The user message content.
        system_prompt: System instruction for the model.
        timeout: Request timeout in seconds.

    Returns:
        The model's response text.

    Raises:
        ValueError: If ``GROQ_API_KEY`` is not configured.
        RuntimeError: On HTTP errors or empty responses.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not configured.")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": GROQ_MODEL,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 2048,
        },
        timeout=timeout,
    )
    response.raise_for_status()

    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("Groq returned no choices.")

    text = choices[0].get("message", {}).get("content", "")
    if not text or not text.strip():
        raise RuntimeError("Groq returned empty content.")

    return text.strip()


def _call_nvidia(prompt: str, system_prompt: str, timeout: int) -> str:
    """
    Call the NVIDIA AI API via its OpenAI-compatible chat completions endpoint.

    Args:
        prompt: The user message content.
        system_prompt: System instruction for the model.
        timeout: Request timeout in seconds.

    Returns:
        The model's response text.

    Raises:
        ValueError: If ``NVIDIA_API_KEY`` is not configured.
        RuntimeError: On HTTP errors or empty responses.
    """
    if not NVIDIA_API_KEY:
        raise ValueError("NVIDIA_API_KEY is not configured.")

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = requests.post(
        "https://integrate.api.nvidia.com/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": NVIDIA_MODEL,
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 2048,
        },
        timeout=timeout,
    )
    response.raise_for_status()

    data = response.json()
    choices = data.get("choices", [])
    if not choices:
        raise RuntimeError("NVIDIA returned no choices.")

    text = choices[0].get("message", {}).get("content", "")
    if not text or not text.strip():
        raise RuntimeError("NVIDIA returned empty content.")

    return text.strip()


_PROVIDERS = {
    "groq": _call_groq,
    "nvidia": _call_nvidia,
}


def get_provider_health() -> dict[str, str]:
    groq_status = "configured" if GROQ_API_KEY else "missing"
    nvidia_status = "configured" if NVIDIA_API_KEY else "missing"
    status = "healthy" if groq_status == "configured" else "degraded"
    if groq_status != "configured" and nvidia_status == "configured":
        status = "degraded"
    elif groq_status != "configured" and nvidia_status != "configured":
        status = "unavailable"
    return {
        "status": status,
        "groq": groq_status,
        "nvidia": nvidia_status,
    }


# ── Unified caller with automatic fallback ─────────────────────────────


def call_llm(
    prompt: str,
    system_prompt: str = "",
    request_id: str = "unknown",
    phase: str = "general",
    timeout_seconds: int | None = None,
) -> str | None:
    """
    Route an LLM call through Groq (primary) → NVIDIA (fallback).

    Tries each provider in order. Falls back on:
    - Exception during request
    - Timeout
    - Empty / None response

    Args:
        prompt: The user-facing query or formatted input.
        system_prompt: System instruction guiding model behaviour.
        request_id: Trace ID for structured logging.
        phase: Label for the call phase (e.g. ``"tool_planning"``).
        timeout_seconds: Wall-clock timeout per provider attempt.

    Returns:
        The model's text response, or ``None`` if all providers fail.
    """
    timeout = timeout_seconds or LLM_TIMEOUT_SECONDS
    last_error: Exception | None = None

    for provider_name in _PROVIDER_ORDER:
        caller_fn = _PROVIDERS.get(provider_name)
        if caller_fn is None:
            continue

        started = time.perf_counter()
        _log(
            logging.INFO,
            event="llm_start",
            request_id=request_id,
            tool_name=phase,
            provider=provider_name,
            latency_ms=None,
            error_type=None,
        )

        future = _LLM_EXECUTOR.submit(caller_fn, prompt, system_prompt, timeout)

        try:
            text = future.result(timeout=timeout)
        except FuturesTimeoutError as exc:
            future.cancel()
            last_error = exc
            latency_ms = int((time.perf_counter() - started) * 1000)
            _log(
                logging.ERROR,
                event="llm_timeout",
                request_id=request_id,
                tool_name=phase,
                provider=provider_name,
                latency_ms=latency_ms,
                error_type="TimeoutError",
            )
            continue
        except Exception as exc:
            last_error = exc
            latency_ms = int((time.perf_counter() - started) * 1000)
            _log(
                logging.ERROR,
                event="llm_error",
                request_id=request_id,
                tool_name=phase,
                provider=provider_name,
                latency_ms=latency_ms,
                error_type=type(exc).__name__,
            )
            continue

        if not text or not text.strip():
            latency_ms = int((time.perf_counter() - started) * 1000)
            _log(
                logging.WARNING,
                event="llm_empty_response",
                request_id=request_id,
                tool_name=phase,
                provider=provider_name,
                latency_ms=latency_ms,
                error_type="EmptyResponse",
            )
            continue

        latency_ms = int((time.perf_counter() - started) * 1000)
        _log(
            logging.INFO,
            event="llm_finish",
            request_id=request_id,
            tool_name=phase,
            provider=provider_name,
            latency_ms=latency_ms,
            error_type=None,
        )
        return text.strip()

    _log(
        logging.ERROR,
        event="llm_all_providers_failed",
        request_id=request_id,
        tool_name=phase,
        latency_ms=None,
        error_type=type(last_error).__name__ if last_error else "UnknownError",
    )
    return None
