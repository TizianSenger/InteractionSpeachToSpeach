from __future__ import annotations

import json
import threading
from typing import Any, Callable

import requests


class OllamaProvider:
    def __init__(self, session: requests.Session, base_url: str) -> None:
        self.session = session
        self.base_url = base_url.rstrip("/")

    def list_models(self) -> list[str]:
        response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
        response.raise_for_status()
        data = response.json()
        models: list[str] = []
        for item in data.get("models", []):
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if name:
                models.append(name)
        return models

    def check_connection(self, model_name: str) -> None:
        try:
            available_models = self.list_models()
        except requests.RequestException as exc:
            raise RuntimeError(
                "Ollama nicht erreichbar. Starte Ollama (App oder 'ollama serve') und versuche es erneut."
            ) from exc
        except ValueError as exc:
            raise RuntimeError("Ungültige Antwort von Ollama /api/tags") from exc

        has_model = any(name == model_name or name.startswith(f"{model_name}:") for name in available_models)
        if not has_model:
            raise RuntimeError(
                f"Modell '{model_name}' nicht gefunden. Bitte zuerst: ollama pull {model_name}"
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
        if keep_alive:
            payload["keep_alive"] = keep_alive
        if options:
            payload["options"] = options

        response = self.session.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        return str(data.get("message", {}).get("content", "")).strip()

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
        if keep_alive:
            payload["keep_alive"] = keep_alive
        if options:
            payload["options"] = options

        answer_parts: list[str] = []
        response: requests.Response | None = None
        try:
            response = self.session.post(
                f"{self.base_url}/api/chat",
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

                try:
                    data = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue

                chunk = str(data.get("message", {}).get("content", ""))
                if chunk:
                    answer_parts.append(chunk)
                    if on_chunk is not None:
                        on_chunk(chunk)

                if bool(data.get("done", False)):
                    break
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
