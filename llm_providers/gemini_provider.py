from __future__ import annotations

from typing import Any

import requests

from constants import GEMINI_DEFAULT_BASE_URL, GEMINI_MODEL_OPTIONS
from llm_providers.openai_provider import OpenAICompatibleProvider


class GeminiProvider(OpenAICompatibleProvider):
    """Adapter for Google Gemini via OpenAI-compatible endpoint."""

    @staticmethod
    def normalize_base_url(raw_base_url: str) -> str:
        base_url = (raw_base_url or GEMINI_DEFAULT_BASE_URL).strip().rstrip("/")
        if not base_url:
            return GEMINI_DEFAULT_BASE_URL

        low = base_url.lower()
        if "generativelanguage.googleapis.com" in low and "/openai" not in low:
            base_url = f"{base_url}/openai"

        return base_url.rstrip("/")

    def __init__(self, session: requests.Session, base_url: str, api_key: str) -> None:
        super().__init__(
            session=session,
            base_url=self.normalize_base_url(base_url),
            api_key=api_key,
        )

    def list_models(self) -> list[str]:
        try:
            models = super().list_models()
            if models:
                return models
        except requests.HTTPError as exc:
            response = exc.response
            if response is None or response.status_code != 404:
                raise

        return list(GEMINI_MODEL_OPTIONS)

    def check_connection(self, model_name: str) -> None:
        try:
            super().check_connection(model_name)
            return
        except RuntimeError:
            # Fallback verification in case /models is unavailable.
            pass

        try:
            self.send_chat(
                model_name=model_name,
                messages=[{"role": "user", "content": "ping"}],
                options={"max_tokens": 1, "temperature": 0.0},
                timeout=(8, 25),
            )
        except requests.HTTPError as exc:
            response = exc.response
            if response is not None and response.status_code == 404:
                raise RuntimeError(
                    "Gemini Endpoint/Modell nicht gefunden (404). "
                    "Nutze als Base URL z.B. https://generativelanguage.googleapis.com/v1beta/openai "
                    "und ein gueltiges Modell wie gemini-2.0-flash."
                ) from exc
            raise
