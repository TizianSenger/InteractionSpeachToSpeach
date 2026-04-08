"""Generic LLM provider client mixin for VoiceAssistantUI."""
from __future__ import annotations

import json
import threading
import time
from typing import Any, Callable

import requests

from constants import (
    ANTHROPIC_DEFAULT_API_VERSION,
    ANTHROPIC_DEFAULT_BASE_URL,
    GEMINI_DEFAULT_BASE_URL,
    GEMINI_DEFAULT_MODEL,
    OLLAMA_TOOL_ROUTER_PROMPT,
    OLLAMA_VOICE_SYSTEM_PROMPT,
)
from llm_providers.anthropic_provider import AnthropicProvider
from llm_providers.azure_openai_provider import AzureOpenAIProvider
from llm_providers.gemini_provider import GeminiProvider
from llm_providers.openai_provider import OpenAICompatibleProvider
from llm_providers.ollama_provider import OllamaProvider


class LlmProviderMixin:
    """Provides multi-provider LLM streaming + tool-routing methods."""

    _PROVIDER_DISPLAY_NAMES: dict[str, str] = {
        "ollama": "Ollama",
        "openai": "OpenAI",
        "azure openai": "Azure OpenAI",
        "anthropic": "Anthropic",
        "groq": "Groq",
        "google gemini": "Google Gemini",
    }

    _PROVIDER_ALIASES: dict[str, str] = {
        "azure": "azure openai",
        "azureopenai": "azure openai",
        "azure_openai": "azure openai",
        "gemini": "google gemini",
        "google": "google gemini",
        "google_gemini": "google gemini",
    }

    def _normalize_provider_key(self, raw_value: str) -> str:
        normalized = (raw_value or "ollama").strip().lower()
        return self._PROVIDER_ALIASES.get(normalized, normalized)

    def _provider_display_name(self, provider_key: str) -> str:
        key = self._normalize_provider_key(provider_key)
        return self._PROVIDER_DISPLAY_NAMES.get(key, key.title())

    def _validate_provider_configuration(self, provider_key: str) -> None:
        if provider_key == "ollama":
            if not self.ollama_url_var.get().strip():
                raise RuntimeError("Ollama URL fehlt")
            if not self.ollama_model_var.get().strip():
                raise RuntimeError("Ollama Modell fehlt")
            return

        if provider_key == "openai":
            if not self.openai_base_url_var.get().strip():
                raise RuntimeError("OpenAI Base URL fehlt")
            if not self.openai_api_key_var.get().strip():
                raise RuntimeError("OpenAI API-Key fehlt")
            if not self.openai_model_var.get().strip():
                raise RuntimeError("OpenAI Modell fehlt")
            return

        if provider_key == "azure openai":
            if not self.azure_openai_endpoint_var.get().strip():
                raise RuntimeError("Azure OpenAI Endpoint fehlt")
            if not self.azure_openai_api_key_var.get().strip():
                raise RuntimeError("Azure OpenAI API-Key fehlt")
            if not self.azure_openai_deployment_var.get().strip():
                raise RuntimeError("Azure OpenAI Deployment fehlt")
            return

        if provider_key == "anthropic":
            if not self.anthropic_base_url_var.get().strip():
                raise RuntimeError("Anthropic Base URL fehlt")
            if not self.anthropic_api_key_var.get().strip():
                raise RuntimeError("Anthropic API-Key fehlt")
            if not self.anthropic_model_var.get().strip():
                raise RuntimeError("Anthropic Modell fehlt")
            return

        if provider_key == "groq":
            if not self.groq_base_url_var.get().strip():
                raise RuntimeError("Groq Base URL fehlt")
            if not self.groq_api_key_var.get().strip():
                raise RuntimeError("Groq API-Key fehlt")
            if not self.groq_model_var.get().strip():
                raise RuntimeError("Groq Modell fehlt")
            return

        if provider_key == "google gemini":
            if not self.gemini_base_url_var.get().strip():
                raise RuntimeError("Gemini Base URL fehlt")
            if not self.gemini_api_key_var.get().strip():
                raise RuntimeError("Gemini API-Key fehlt")
            if not self.gemini_model_var.get().strip():
                raise RuntimeError("Gemini Modell fehlt")
            return

    def _format_provider_exception(self, provider_key: str, exc: Exception) -> RuntimeError:
        provider_name = self._provider_display_name(provider_key)
        msg = str(exc).strip() or exc.__class__.__name__

        if isinstance(exc, requests.Timeout):
            return RuntimeError(f"{provider_name}: Timeout bei der API-Anfrage")

        if isinstance(exc, requests.HTTPError):
            response = exc.response
            if response is not None:
                status = response.status_code
                if status == 401:
                    return RuntimeError(f"{provider_name}: Auth fehlgeschlagen (401). API-Key pruefen")
                if status == 403:
                    return RuntimeError(f"{provider_name}: Zugriff verweigert (403)")
                if status == 404:
                    if provider_key == "google gemini":
                        return RuntimeError(
                            "Google Gemini: Endpoint oder Modell nicht gefunden (404). "
                            "Pruefe Base URL (…/v1beta/openai) und Modellname (z.B. gemini-2.0-flash)."
                        )
                    return RuntimeError(f"{provider_name}: Endpoint oder Modell nicht gefunden (404)")
                if status == 429:
                    detail = ""
                    try:
                        payload = response.json()
                        if isinstance(payload, dict):
                            err_obj = payload.get("error")
                            if isinstance(err_obj, dict):
                                detail = str(err_obj.get("message", "")).strip()
                            elif "message" in payload:
                                detail = str(payload.get("message", "")).strip()
                    except Exception:
                        detail = ""

                    if detail:
                        return RuntimeError(
                            f"{provider_name}: Rate-Limit/Quota (429): {detail}"
                        )

                    return RuntimeError(
                        f"{provider_name}: Rate-Limit/Quota (429). "
                        "API-Limits/Billing/Quota im Provider pruefen"
                    )
                if 500 <= status <= 599:
                    return RuntimeError(f"{provider_name}: Serverfehler ({status})")
                return RuntimeError(f"{provider_name}: HTTP-Fehler ({status})")

        if isinstance(exc, requests.RequestException):
            return RuntimeError(f"{provider_name}: Netzwerk-/Verbindungsfehler ({msg})")

        return RuntimeError(f"{provider_name}: {msg}")

    def _get_ollama_provider(self, ollama_url: str) -> OllamaProvider:
        return OllamaProvider(session=self.http_session, base_url=ollama_url)

    def _provider_key(self) -> str:
        selected = getattr(self, "llm_provider_var", None)
        if selected is None:
            return "ollama"
        return self._normalize_provider_key(selected.get())

    def _get_openai_provider(self) -> OpenAICompatibleProvider:
        base_url = self.openai_base_url_var.get().strip() or "https://api.openai.com/v1"
        api_key = self.openai_api_key_var.get().strip()
        return OpenAICompatibleProvider(
            session=self.http_session,
            base_url=base_url,
            api_key=api_key,
        )

    def _get_groq_provider(self) -> OpenAICompatibleProvider:
        base_url = self.groq_base_url_var.get().strip() or "https://api.groq.com/openai/v1"
        api_key = self.groq_api_key_var.get().strip()
        return OpenAICompatibleProvider(
            session=self.http_session,
            base_url=base_url,
            api_key=api_key,
        )

    def _get_gemini_provider(self) -> GeminiProvider:
        base_url = self.gemini_base_url_var.get().strip() or GEMINI_DEFAULT_BASE_URL
        api_key = self.gemini_api_key_var.get().strip()
        normalized = GeminiProvider.normalize_base_url(base_url)
        if normalized != base_url:
            self.gemini_base_url_var.set(normalized)
        return GeminiProvider(
            session=self.http_session,
            base_url=normalized,
            api_key=api_key,
        )

    def _get_azure_openai_provider(self) -> AzureOpenAIProvider:
        endpoint = self.azure_openai_endpoint_var.get().strip()
        deployment_name = self.azure_openai_deployment_var.get().strip()
        api_key = self.azure_openai_api_key_var.get().strip()
        api_version = self.azure_openai_api_version_var.get().strip() or "2024-10-21"
        return AzureOpenAIProvider(
            session=self.http_session,
            endpoint=endpoint,
            deployment_name=deployment_name,
            api_key=api_key,
            api_version=api_version,
        )

    def _get_anthropic_provider(self) -> AnthropicProvider:
        base_url = getattr(self, "anthropic_base_url_var", None)
        if base_url is None:
            resolved_base_url = ANTHROPIC_DEFAULT_BASE_URL
        else:
            resolved_base_url = base_url.get().strip() or ANTHROPIC_DEFAULT_BASE_URL

        api_version_var = getattr(self, "anthropic_api_version_var", None)
        if api_version_var is None:
            resolved_api_version = ANTHROPIC_DEFAULT_API_VERSION
        else:
            resolved_api_version = api_version_var.get().strip() or ANTHROPIC_DEFAULT_API_VERSION

        api_key = self.anthropic_api_key_var.get().strip()
        return AnthropicProvider(
            session=self.http_session,
            base_url=resolved_base_url,
            api_key=api_key,
            api_version=resolved_api_version,
        )

    def _get_provider_context_for_key(self, provider_key: str) -> tuple[str, str, str, Any]:
        provider_key = self._normalize_provider_key(provider_key)
        if provider_key == "ollama":
            model_name = self.ollama_model_var.get().strip() or "phi4-mini"
            endpoint = self.ollama_url_var.get().strip() or "http://localhost:11434"
            return provider_key, model_name, endpoint, self._get_ollama_provider(endpoint)

        if provider_key == "openai":
            model_name = self.openai_model_var.get().strip() or "gpt-4o-mini"
            endpoint = self.openai_base_url_var.get().strip() or "https://api.openai.com/v1"
            return provider_key, model_name, endpoint, self._get_openai_provider()

        if provider_key == "azure openai":
            model_name = self.azure_openai_deployment_var.get().strip() or ""
            endpoint = self.azure_openai_endpoint_var.get().strip() or ""
            return provider_key, model_name, endpoint, self._get_azure_openai_provider()

        if provider_key == "anthropic":
            model_name = self.anthropic_model_var.get().strip() or "claude-3-5-sonnet-latest"
            endpoint = (
                self.anthropic_base_url_var.get().strip()
                if hasattr(self, "anthropic_base_url_var")
                else ANTHROPIC_DEFAULT_BASE_URL
            )
            return provider_key, model_name, endpoint, self._get_anthropic_provider()

        if provider_key == "groq":
            model_name = self.groq_model_var.get().strip() or "llama-3.3-70b-versatile"
            endpoint = self.groq_base_url_var.get().strip() or "https://api.groq.com/openai/v1"
            return provider_key, model_name, endpoint, self._get_groq_provider()

        if provider_key == "google gemini":
            model_name = self.gemini_model_var.get().strip() or GEMINI_DEFAULT_MODEL
            endpoint = self.gemini_base_url_var.get().strip() or GEMINI_DEFAULT_BASE_URL
            return provider_key, model_name, endpoint, self._get_gemini_provider()

        raise RuntimeError(
            "Provider aktuell noch nicht verdrahtet. Bitte nutze einen verfuegbaren Provider."
        )

    def _get_active_provider_context(self) -> tuple[str, str, str, Any]:
        return self._get_provider_context_for_key(self._provider_key())

    def test_provider_api_key(self, provider_key: str) -> tuple[str, str]:
        normalized_key = self._normalize_provider_key(provider_key)
        _provider_key, model_name, endpoint, provider = self._get_provider_context_for_key(normalized_key)
        self._validate_provider_configuration(normalized_key)
        try:
            self._call_with_retry(lambda: provider.check_connection(model_name), retries=1)
        except Exception as exc:
            raise self._format_provider_exception(normalized_key, exc) from exc
        return model_name, endpoint

    def _build_provider_options(self, provider_key: str) -> dict[str, Any]:
        if provider_key == "ollama":
            return {
                "num_predict": self._get_reply_max_tokens(),
                "temperature": self._get_reply_temperature(),
                "top_p": 0.9,
                "repeat_penalty": 1.05,
            }

        if provider_key == "anthropic":
            return {
                "max_tokens": self._get_reply_max_tokens(),
                "temperature": self._get_reply_temperature(),
                "top_p": 0.9,
            }

        if provider_key == "google gemini":
            budget = max(512, self._get_reply_max_tokens())
            return {
                # Keep payload conservative for Gemini OpenAI-compat endpoints.
                "max_tokens": budget,
                "temperature": self._get_reply_temperature(),
                "top_p": 0.9,
            }

        # OpenAI-compatible payload keys
        return {
            "max_tokens": self._get_reply_max_tokens(),
            "temperature": self._get_reply_temperature(),
            "top_p": 0.9,
        }

    def _call_with_retry(self, fn: Callable[[], Any], retries: int = 2) -> Any:
        delay = 0.4
        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return fn()
            except Exception as exc:
                last_exc = exc
                if attempt >= retries:
                    break
                msg = str(exc).lower()
                transient = any(
                    token in msg for token in ["rate", "429", "timeout", "timed out", "tempor", "503", "502"]
                )
                if not transient:
                    break
                time.sleep(delay)
                delay *= 1.8

        if last_exc is not None:
            raise last_exc
        raise RuntimeError("Unbekannter Provider-Fehler")

    def list_provider_models(self) -> list[str]:
        provider_key, _model_name, _endpoint, provider = self._get_active_provider_context()
        self._validate_provider_configuration(provider_key)

        def _run() -> list[str]:
            names = provider.list_models()
            if provider_key == "ollama":
                return [n[: -len(":latest")] if n.endswith(":latest") else n for n in names]
            return names

        try:
            names = self._call_with_retry(_run, retries=1)
        except Exception as exc:
            raise self._format_provider_exception(provider_key, exc) from exc
        return sorted(set([n for n in names if n]))

    def check_llm_connection(self, force_refresh: bool = False) -> tuple[str, str]:
        provider_key, model_name, endpoint, provider = self._get_active_provider_context()
        self._validate_provider_configuration(provider_key)
        cache_key = f"{provider_key}::{endpoint.rstrip('/')}::{model_name}"
        now = time.time()

        if not force_refresh and self.ollama_check_cache.get("cache_key") == cache_key:
            checked_at = float(self.ollama_check_cache.get("checked_at", 0.0))
            if (now - checked_at) <= self.ollama_cache_ttl_seconds:
                return model_name, endpoint

        try:
            self._call_with_retry(lambda: provider.check_connection(model_name), retries=1)
        except Exception as exc:
            raise self._format_provider_exception(provider_key, exc) from exc

        self.ollama_check_cache.update(
            {
                "cache_key": cache_key,
                "checked_at": now,
                "model_name": model_name,
                "ollama_url": endpoint,
            }
        )

        return model_name, endpoint

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

    def ask_llm(
        self,
        user_text: str,
        on_chunk: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        model_name, _endpoint = self.check_llm_connection()
        provider_key, _active_model_name, _active_endpoint, provider = self._get_active_provider_context()
        options = self._build_provider_options(provider_key)
        keep_alive = "30m" if provider_key == "ollama" else None

        def set_active_response(response: Any | None) -> None:
            with self.active_ollama_response_lock:
                self.active_ollama_response = response

        try:
            answer = self._call_with_retry(
                lambda: provider.stream_chat(
                    model_name=model_name,
                    messages=self._build_chat_messages(user_text),
                    on_chunk=on_chunk,
                    cancel_event=cancel_event,
                    options=options,
                    timeout=(10, 300),
                    keep_alive=keep_alive,
                    active_response_setter=set_active_response,
                ),
                retries=1,
            )

            # Gemini can occasionally end mid-sentence despite generous token budgets.
            # If response looks cut off, request one continuation chunk once.
            if provider_key == "google gemini" and answer and not (cancel_event and cancel_event.is_set()):
                tail = answer.rstrip()
                if tail and tail[-1] not in ".!?\n\"'":
                    continuation_messages = self._build_chat_messages(user_text)
                    continuation_messages.append({"role": "assistant", "content": answer})
                    continuation_messages.append(
                        {
                            "role": "user",
                            "content": "Bitte fuehre die letzte Antwort exakt ab der letzten Stelle ohne Wiederholung fort.",
                        }
                    )
                    try:
                        cont = self._call_with_retry(
                            lambda: provider.send_chat(
                                model_name=model_name,
                                messages=continuation_messages,
                                options=options,
                                timeout=(10, 120),
                                keep_alive=keep_alive,
                            ),
                            retries=0,
                        )
                    except Exception:
                        cont = ""
                    if cont:
                        answer = f"{answer.rstrip()} {cont.lstrip()}"

            return answer
        except Exception as exc:
            raise self._format_provider_exception(provider_key, exc) from exc

    def check_ollama(self, force_refresh: bool = False) -> tuple[str, str]:
        """Backward-compatible alias for legacy call sites."""
        return self.check_llm_connection(force_refresh=force_refresh)

    def ask_ollama(
        self,
        user_text: str,
        on_chunk: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        """Backward-compatible alias for legacy call sites."""
        return self.ask_llm(user_text=user_text, on_chunk=on_chunk, cancel_event=cancel_event)

    def test_ollama(self) -> None:
        self.test_btn.configure(state="disabled")
        worker = threading.Thread(target=self._run_ollama_test, daemon=True)
        worker.start()

    def _run_ollama_test(self) -> None:
        try:
            provider_name = self._provider_display_name(self._provider_key())
            self.set_status(f"Pruefe Provider: {provider_name}...")
            model_name, _ = self.check_llm_connection()
            self.set_status(f"Provider OK. Modell erreichbar: {model_name}")
        except Exception as exc:
            self.set_status(f"Provider-Test fehlgeschlagen: {exc}")
        finally:
            self.after(0, lambda: self.test_btn.configure(state="normal"))

    def run_provider_diagnostics(self) -> dict[str, Any]:
        provider_key, model_name, endpoint, _provider = self._get_active_provider_context()
        started = time.perf_counter()
        self.check_llm_connection(force_refresh=True)
        models = self.list_provider_models()
        elapsed = time.perf_counter() - started
        return {
            "provider": provider_key,
            "model": model_name,
            "endpoint": endpoint,
            "models_count": len(models),
            "models_preview": models[:5],
            "latency_seconds": elapsed,
        }

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
        if self._provider_key() != "ollama":
            return "none"

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

        provider = self._get_ollama_provider(ollama_url)
        content = self._call_with_retry(
            lambda: provider.send_chat(
                model_name=payload["model"],
                messages=payload["messages"],
                options=payload.get("options"),
                timeout=(10, 60),
                keep_alive=str(payload.get("keep_alive", "")) or None,
            ),
            retries=1,
        )
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
