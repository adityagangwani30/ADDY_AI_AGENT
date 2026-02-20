from __future__ import annotations

from typing import Any

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_MODEL_FALLBACKS, GEMINI_MODEL_PRIMARY


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
        return ordered

    def generate_json_decision(self, system_instruction: str, user_message: str) -> str:
        last_error: Exception | None = None

        for model_name in self._models:
            try:
                response = self._client.models.generate_content(
                    model=model_name,
                    contents=user_message,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.1,
                    ),
                )
                text = self._extract_text(response)
                if text:
                    return text
                raise GeminiIntegrationError(f"Model '{model_name}' returned an empty response.")
            except Exception as exc:
                last_error = exc

        raise GeminiIntegrationError(
            f"Gemini generation failed for all configured models: {self._models}"
        ) from last_error

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
