from __future__ import annotations

import json
import threading
from typing import Any, Callable

import requests


class OpenAICompatibleProvider:
    """Adapter for OpenAI-compatible chat APIs (OpenAI, Groq, many gateways)."""

    def __init__(self, session: requests.Session, base_url: str, api_key: str) -> None:
        self.session = session
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("API-Key fehlt fuer OpenAI-kompatiblen Provider")
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def list_models(self) -> list[str]:
        response = self.session.get(
            f"{self.base_url}/models",
            headers=self._headers(),
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
        models: list[str] = []
        for item in data.get("data", []):
            if not isinstance(item, dict):
                continue
            model_id = str(item.get("id", "")).strip()
            if model_id:
                models.append(model_id)
        return sorted(set(models))

    def check_connection(self, model_name: str) -> None:
        try:
            models = self.list_models()
        except requests.RequestException as exc:
            raise RuntimeError("Provider nicht erreichbar oder API-Key ungueltig") from exc
        except ValueError as exc:
            raise RuntimeError("Ungueltige Antwort vom Provider /models") from exc

        if model_name and model_name not in models:
            raise RuntimeError(
                f"Modell '{model_name}' nicht verfuegbar. Verfuegbar: {', '.join(models[:8])}"
            )

    def send_chat(
        self,
        *,
        model_name: str,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
        timeout: tuple[float, float] = (10, 60),
        keep_alive: str | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "stream": False,
        }
        if options:
            payload.update(options)

        response = self.session.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
        return str(message.get("content", "")).strip()

    def stream_chat(
        self,
        *,
        model_name: str,
        messages: list[dict[str, str]],
        on_chunk: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
        options: dict[str, Any] | None = None,
        timeout: tuple[float, float] = (10, 300),
        keep_alive: str | None = None,
        active_response_setter: Callable[[Any | None], None] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "stream": True,
        }
        if options:
            payload.update(options)

        answer_parts: list[str] = []
        response: requests.Response | None = None
        try:
            response = self.session.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
                stream=True,
                timeout=timeout,
            )
            if active_response_setter is not None:
                active_response_setter(response)

            response.raise_for_status()
            for raw_line in response.iter_lines(decode_unicode=True):
                if cancel_event is not None and cancel_event.is_set():
                    break
                if not raw_line:
                    continue

                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data_payload = line[len("data:") :].strip()
                if data_payload == "[DONE]":
                    break

                try:
                    data = json.loads(data_payload)
                except json.JSONDecodeError:
                    continue

                choices = data.get("choices", [])
                if not choices or not isinstance(choices[0], dict):
                    continue
                delta = choices[0].get("delta", {})
                chunk = str(delta.get("content", ""))
                if chunk:
                    answer_parts.append(chunk)
                    if on_chunk is not None:
                        on_chunk(chunk)
        except Exception as exc:
            if cancel_event is not None and cancel_event.is_set():
                return "".join(answer_parts).strip()
            raise exc
        finally:
            if active_response_setter is not None:
                active_response_setter(None)
            if response is not None:
                response.close()

        return "".join(answer_parts).strip()
