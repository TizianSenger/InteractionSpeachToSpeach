from __future__ import annotations

import json
import threading
from typing import Any, Callable

import requests


class AnthropicProvider:
    def __init__(
        self,
        session: requests.Session,
        base_url: str,
        api_key: str,
        api_version: str = "2023-06-01",
    ) -> None:
        self.session = session
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key.strip()
        self.api_version = api_version.strip() or "2023-06-01"

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise RuntimeError("Anthropic API-Key fehlt")
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.api_version,
            "content-type": "application/json",
        }

    def _split_messages(self, messages: list[dict[str, str]]) -> tuple[str | None, list[dict[str, str]]]:
        system_prompt: str | None = None
        converted: list[dict[str, str]] = []
        for msg in messages:
            role = str(msg.get("role", "")).strip().lower()
            content = str(msg.get("content", ""))
            if role == "system":
                if system_prompt:
                    system_prompt = f"{system_prompt}\n{content}"
                else:
                    system_prompt = content
                continue
            if role not in {"user", "assistant"}:
                role = "user"
            converted.append({"role": role, "content": content})
        return system_prompt, converted

    def list_models(self) -> list[str]:
        response = self.session.get(
            f"{self.base_url}/v1/models",
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
        models = self.list_models()
        if model_name and model_name not in models:
            raise RuntimeError(f"Anthropic Modell '{model_name}' nicht verfuegbar")

    def send_chat(
        self,
        *,
        model_name: str,
        messages: list[dict[str, str]],
        options: dict[str, Any] | None = None,
        timeout: tuple[float, float] = (10, 60),
        keep_alive: str | None = None,
    ) -> str:
        system_prompt, converted_messages = self._split_messages(messages)
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": converted_messages,
            "stream": False,
            "max_tokens": 256,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if options:
            payload.update(options)

        response = self.session.post(
            f"{self.base_url}/v1/messages",
            headers=self._headers(),
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        blocks = data.get("content", [])
        text_parts: list[str] = []
        for block in blocks:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
        return "".join(text_parts).strip()

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
        system_prompt, converted_messages = self._split_messages(messages)
        payload: dict[str, Any] = {
            "model": model_name,
            "messages": converted_messages,
            "stream": True,
            "max_tokens": 256,
        }
        if system_prompt:
            payload["system"] = system_prompt
        if options:
            payload.update(options)

        answer_parts: list[str] = []
        response: requests.Response | None = None
        try:
            response = self.session.post(
                f"{self.base_url}/v1/messages",
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
                if line.startswith("event:"):
                    continue
                if not line.startswith("data:"):
                    continue

                data_payload = line[len("data:") :].strip()
                if not data_payload:
                    continue

                try:
                    data = json.loads(data_payload)
                except json.JSONDecodeError:
                    continue

                event_type = str(data.get("type", ""))
                if event_type == "content_block_delta":
                    delta = data.get("delta", {})
                    chunk = str(delta.get("text", ""))
                    if chunk:
                        answer_parts.append(chunk)
                        if on_chunk is not None:
                            on_chunk(chunk)
                elif event_type == "message_stop":
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
