from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Any

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL_FALLBACKS, GEMINI_MODEL_PRIMARY

LOGGER = logging.getLogger(__name__)
_GEMINI_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gemini-call")
_GEMINI_TIMEOUT_SECONDS = 10
_MAX_MODEL_ATTEMPTS = 2


def _log(level: int, **payload: Any) -> None:
    LOGGER.log(level, json.dumps(payload, default=str))


def _error_payload(model_name: str, error_type: str, message: str) -> str:
    return json.dumps(
        {
            "component": "gemini",
            "model_name": model_name,
            "error_type": error_type,
            "message": message,
        }
    )


class GeminiIntegrationError(RuntimeError):
    """Raised when Gemini generation fails across all configured models."""


class GeminiClient:
    def __init__(self) -> None:
        if not GEMINI_API_KEY:
            raise GeminiIntegrationError("Missing GEMINI_API_KEY in environment variables.")

        self._client = genai.Client(api_key=GEMINI_API_KEY)
        self._models = self._build_model_priority()

    def _build_model_priority(self) -> list[str]:
        ordered: list[str] = [GEMINI_MODEL_PRIMARY]
        for model_name in GEMINI_MODEL_FALLBACKS:
            if model_name and model_name not in ordered:
                ordered.append(model_name)
        return ordered[:_MAX_MODEL_ATTEMPTS]

    def generate_json_decision(
        self,
        system_instruction: str,
        user_message: str,
        request_id: str = "unknown",
    ) -> str:
        last_error: Exception | None = None

        for model_name in self._models:
            started = time.perf_counter()
            _log(
                logging.INFO,
                event="gemini_call_start",
                request_id=request_id,
                tool_name="gemini_generate_content",
                model_name=model_name,
                latency_ms=None,
                error_type=None,
            )

            try:
                response = self._generate_with_timeout(
                    model_name=model_name,
                    system_instruction=system_instruction,
                    user_message=user_message,
                )
                text = self._extract_text(response)
                latency_ms = int((time.perf_counter() - started) * 1000)

                if not text:
                    raise GeminiIntegrationError(
                        _error_payload(
                            model_name=model_name,
                            error_type="EmptyResponse",
                            message="Model returned an empty response.",
                        )
                    )

                _log(
                    logging.INFO,
                    event="gemini_call_finish",
                    request_id=request_id,
                    tool_name="gemini_generate_content",
                    model_name=model_name,
                    latency_ms=latency_ms,
                    error_type=None,
                )
                return text
            except GeminiIntegrationError as exc:
                latency_ms = int((time.perf_counter() - started) * 1000)
                _log(
                    logging.ERROR,
                    event="gemini_call_error",
                    request_id=request_id,
                    tool_name="gemini_generate_content",
                    model_name=model_name,
                    latency_ms=latency_ms,
                    error_type=type(exc).__name__,
                )
                last_error = exc

        raise GeminiIntegrationError(
            f"Gemini generation failed for configured models {self._models}: {last_error}"
        ) from last_error

    def _generate_with_timeout(
        self,
        model_name: str,
        system_instruction: str,
        user_message: str,
    ) -> Any:
        future = _GEMINI_EXECUTOR.submit(
            self._client.models.generate_content,
            model=model_name,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,
            ),
            request_options={"timeout": _GEMINI_TIMEOUT_SECONDS},
        )

        try:
            return future.result(timeout=_GEMINI_TIMEOUT_SECONDS)
        except FuturesTimeoutError as exc:
            future.cancel()
            raise GeminiIntegrationError(
                _error_payload(
                    model_name=model_name,
                    error_type="TimeoutError",
                    message=f"Gemini call timed out after {_GEMINI_TIMEOUT_SECONDS} seconds.",
                )
            ) from exc
        except TypeError as exc:
            if "request_options" not in str(exc):
                raise GeminiIntegrationError(
                    _error_payload(
                        model_name=model_name,
                        error_type=type(exc).__name__,
                        message=str(exc),
                    )
                ) from exc

            fallback_future = _GEMINI_EXECUTOR.submit(
                self._client.models.generate_content,
                model=model_name,
                contents=user_message,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.1,
                ),
            )
            try:
                return fallback_future.result(timeout=_GEMINI_TIMEOUT_SECONDS)
            except FuturesTimeoutError as timeout_exc:
                fallback_future.cancel()
                raise GeminiIntegrationError(
                    _error_payload(
                        model_name=model_name,
                        error_type="TimeoutError",
                        message=f"Gemini fallback call timed out after {_GEMINI_TIMEOUT_SECONDS} seconds.",
                    )
                ) from timeout_exc
            except Exception as fallback_exc:
                raise GeminiIntegrationError(
                    _error_payload(
                        model_name=model_name,
                        error_type=type(fallback_exc).__name__,
                        message=str(fallback_exc),
                    )
                ) from fallback_exc
        except Exception as exc:
            raise GeminiIntegrationError(
                _error_payload(
                    model_name=model_name,
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
            ) from exc

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
            if content is None:
                continue
            parts = getattr(content, "parts", None)
            if not parts:
                continue
            for part in parts:
                part_text = getattr(part, "text", None)
                if isinstance(part_text, str) and part_text.strip():
                    return part_text.strip()

        return ""
