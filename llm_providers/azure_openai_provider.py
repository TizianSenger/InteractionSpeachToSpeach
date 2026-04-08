from __future__ import annotations

import json
import threading
from typing import Any, Callable

import requests


class AzureOpenAIProvider:
    def __init__(
        self,
        session: requests.Session,
        endpoint: str,
        deployment_name: str,
        api_key: str,
        api_version: str,
    ) -> None:
        self.session = session
        self.endpoint = endpoint.rstrip("/")
        self.deployment_name = deployment_name.strip()
        self.api_key = api_key.strip()
        self.api_version = api_version.strip() or "2024-10-21"

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("Azure OpenAI API-Key fehlt")
        return {
            "api-key": self.api_key,
            "Content-Type": "application/json",
        }

    def _chat_url(self) -> str:
        if not self.endpoint:
            raise RuntimeError("Azure OpenAI Endpoint fehlt")
        if not self.deployment_name:
            raise RuntimeError("Azure OpenAI Deployment-Name fehlt")
        return (
            f"{self.endpoint}/openai/deployments/{self.deployment_name}/chat/completions"
            f"?api-version={self.api_version}"
        )

    def list_models(self) -> list[str]:
        # Azure deploys models as deployments; return deployment as selectable runtime name.
        if self.deployment_name:
            return [self.deployment_name]

        url = f"{self.endpoint}/openai/models?api-version={self.api_version}"
        response = self.session.get(url, headers=self._headers(), timeout=8)
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
        # Lightweight check using list endpoint where available.
        models = self.list_models()
        if models and model_name and model_name not in models:
            raise RuntimeError(
                f"Azure OpenAI Modell/Deployment '{model_name}' nicht gefunden"
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
            "messages": messages,
            "stream": False,
        }
        if options:
            payload.update(options)

        response = self.session.post(
            self._chat_url(),
            headers=self._headers(),
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        choices = data.get("choices", [])
        if not choices or not isinstance(choices[0], dict):
            return ""
        message = choices[0].get("message", {})
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
            "messages": messages,
            "stream": True,
        }
        if options:
            payload.update(options)

        answer_parts: list[str] = []
        response: requests.Response | None = None
        try:
            response = self.session.post(
                self._chat_url(),
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
