"""Ollama API client – mixed into VoiceAssistantUI via multiple inheritance."""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable

import requests

from constants import OLLAMA_TOOL_ROUTER_PROMPT, OLLAMA_VOICE_SYSTEM_PROMPT


class OllamaMixin:
    """Provides all Ollama streaming + tool-routing methods."""

    def check_ollama(self, force_refresh: bool = False) -> tuple[str, str]:
        model_name = self.ollama_model_var.get().strip() or "phi4-mini"
        ollama_url = self.ollama_url_var.get().strip() or "http://localhost:11434"
        cache_key = f"{ollama_url.rstrip('/')}::{model_name}"
        now = time.time()

        if not force_refresh and self.ollama_check_cache.get("cache_key") == cache_key:
            checked_at = float(self.ollama_check_cache.get("checked_at", 0.0))
            if (now - checked_at) <= self.ollama_cache_ttl_seconds:
                return model_name, ollama_url

        try:
            tags_resp = self.http_session.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=5)
            tags_resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                "Ollama nicht erreichbar. Starte Ollama (App oder 'ollama serve') und versuche es erneut."
            ) from exc

        try:
            tags_data = tags_resp.json()
            available_models = {
                item.get("name", "")
                for item in tags_data.get("models", [])
                if isinstance(item, dict)
            }
            has_model = any(name == model_name or name.startswith(f"{model_name}:") for name in available_models)
            if not has_model:
                raise RuntimeError(
                    f"Modell '{model_name}' nicht gefunden. Bitte zuerst: ollama pull {model_name}"
                )
        except ValueError as exc:
            raise RuntimeError("Ungültige Antwort von Ollama /api/tags") from exc

        self.ollama_check_cache.update(
            {
                "cache_key": cache_key,
                "checked_at": now,
                "model_name": model_name,
                "ollama_url": ollama_url,
            }
        )

        return model_name, ollama_url

    def _get_reply_max_tokens(self) -> int:
        try:
            value = int(self.reply_max_tokens_var.get().strip())
        except ValueError:
            value = 120
        return max(32, min(1024, value))

    def _get_reply_temperature(self) -> float:
        try:
            value = float(self.reply_temperature_var.get().strip())
        except ValueError:
            value = 0.3
        return max(0.0, min(1.2, value))

    def _build_chat_messages(self, user_text: str) -> list[dict[str, str]]:
        system_prompt = self._build_persona_system_prompt()
        if self.concise_reply_var.get():
            system_prompt = f"{system_prompt} {OLLAMA_VOICE_SYSTEM_PROMPT}"
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": user_text})
        return messages

    def ask_ollama(
        self,
        user_text: str,
        on_chunk: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        model_name, ollama_url = self.check_ollama()

        payload = {
            "model": model_name,
            "messages": self._build_chat_messages(user_text),
            "stream": True,
            "keep_alive": "30m",
            "options": {
                "num_predict": self._get_reply_max_tokens(),
                "temperature": self._get_reply_temperature(),
                "top_p": 0.9,
                "repeat_penalty": 1.05,
            },
        }

        answer_parts: list[str] = []
        try:
            with self.http_session.post(
                f"{ollama_url.rstrip('/')}/api/chat",
                json=payload,
                stream=True,
                timeout=(10, 300),
            ) as response:
                with self.active_ollama_response_lock:
                    self.active_ollama_response = response

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

                    message = data.get("message", {})
                    chunk = str(message.get("content", ""))
                    if chunk:
                        answer_parts.append(chunk)
                        if on_chunk is not None:
                            on_chunk(chunk)

                    if bool(data.get("done", False)):
                        break
        except Exception as exc:
            # When a streaming response is force-closed during cancel, urllib can raise
            # transport errors like "NoneType has no attribute read"; treat these as a normal abort.
            if cancel_event is not None and cancel_event.is_set():
                return "".join(answer_parts).strip()
            raise exc
        finally:
            with self.active_ollama_response_lock:
                self.active_ollama_response = None

        return "".join(answer_parts).strip()

    def test_ollama(self) -> None:
        self.test_btn.configure(state="disabled")
        worker = threading.Thread(target=self._run_ollama_test, daemon=True)
        worker.start()

    def _run_ollama_test(self) -> None:
        try:
            self.set_status("Prüfe Ollama...")
            model_name, _ = self.check_ollama()
            self.set_status(f"Ollama OK. Modell erreichbar: {model_name}")
        except Exception as exc:
            self.set_status(f"Ollama Test fehlgeschlagen: {exc}")
        finally:
            self.after(0, lambda: self.test_btn.configure(state="normal"))

    def _extract_json_object(self, text: str) -> dict[str, Any] | None:
        candidate = text.strip()
        if not candidate:
            return None

        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None

        try:
            parsed = json.loads(candidate[start : end + 1])
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            return None
        return None

    def _decide_tool_action_with_ollama(self, user_text: str) -> str:
        model_name, ollama_url = self.check_ollama()
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": OLLAMA_TOOL_ROUTER_PROMPT},
                {"role": "user", "content": user_text},
            ],
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "num_predict": 120,
                "temperature": 0.0,
                "top_p": 0.9,
            },
        }

        response = self.http_session.post(
            f"{ollama_url.rstrip('/')}/api/chat",
            json=payload,
            timeout=(10, 60),
        )
        response.raise_for_status()
        data = response.json()
        content = str(data.get("message", {}).get("content", "")).strip()
        decision = self._extract_json_object(content)
        if decision is None:
            self.logger.warning("Tool-Router gab kein gueltiges JSON zurueck: %s", content[:200])
            return "none"

        tool = str(decision.get("tool", "none")).strip().lower()
        if tool in {"light_on", "light_off", "none"}:
            self.logger.info("Tool-Router Entscheidung: %s", tool)
            return tool

        self.logger.warning("Unbekanntes Tool vom Router: %s", tool)
        return "none"
