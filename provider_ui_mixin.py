"""Provider-specific UI handling for VoiceAssistantUI."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from constants import LLM_PROVIDER_OPTIONS


class ProviderUiMixin:
    def _canonical_provider_label(self, value: str) -> str:
        normalized = (value or "Ollama").strip().lower()
        aliases = {
            "azure": "Azure OpenAI",
            "azureopenai": "Azure OpenAI",
            "azure_openai": "Azure OpenAI",
        }
        if normalized in aliases:
            return aliases[normalized]
        for option in LLM_PROVIDER_OPTIONS:
            if option.lower() == normalized:
                return option
        return "Ollama"

    def on_llm_provider_changed(self, selected_provider: str) -> None:
        normalized = self._canonical_provider_label(selected_provider)

        self.llm_provider_var.set(normalized)
        provider_frames = getattr(self, "provider_frames", {})
        if not provider_frames:
            return

        for key, frame in provider_frames.items():
            if key == normalized:
                frame.pack(fill="x", padx=14, pady=(0, 8))
            else:
                frame.pack_forget()

        # Keep the existing model dropdown linked to active provider selection.
        if normalized == "OpenAI":
            self.ollama_model_var.set(self.openai_model_var.get().strip())
        elif normalized == "Azure OpenAI":
            self.ollama_model_var.set(self.azure_openai_deployment_var.get().strip())
        elif normalized == "Anthropic":
            self.ollama_model_var.set(self.anthropic_model_var.get().strip())
        elif normalized == "Groq":
            self.ollama_model_var.set(self.groq_model_var.get().strip())
        else:
            self.ollama_model_var.set(self.ollama_model_var.get().strip() or "phi4-mini")

        self.set_status(f"Provider aktiv: {normalized}")

    def on_active_provider_model_changed(self, selected_model: str) -> None:
        model_name = selected_model.strip()
        if not model_name:
            return

        self.ollama_model_var.set(model_name)
        provider = self.llm_provider_var.get().strip()
        if provider == "OpenAI":
            self.openai_model_var.set(model_name)
        elif provider == "Azure OpenAI":
            self.azure_openai_deployment_var.set(model_name)
        elif provider == "Anthropic":
            self.anthropic_model_var.set(model_name)
        elif provider == "Groq":
            self.groq_model_var.set(model_name)
        else:
            self.ollama_model_var.set(model_name)

    def run_provider_api_key_test_async(self, provider_label: str) -> None:
        normalized_label = self._canonical_provider_label(provider_label)
        provider_key = self._normalize_provider_key(normalized_label)

        button_attr_map = {
            "OpenAI": "openai_api_test_btn",
            "Groq": "groq_api_test_btn",
            "Azure OpenAI": "azure_api_test_btn",
            "Anthropic": "anthropic_api_test_btn",
        }
        btn = getattr(self, button_attr_map.get(normalized_label, ""), None)
        if btn is not None:
            btn.configure(state="disabled", text="Teste...")

        self.provider_diagnostics_var.set(f"API-Key Test: pruefe {normalized_label}...")

        def worker() -> None:
            try:
                model_name, _endpoint = self.test_provider_api_key(provider_key)
                provider_name = self._provider_display_name(provider_key)
                msg = f"API-Key Test: OK | {provider_name} | Modell: {model_name}"
                self.after(0, lambda: self.provider_diagnostics_var.set(msg))
                self.after(0, lambda: self.set_status(f"API-Key OK ({provider_name})"))
            except Exception as exc:
                self.after(0, lambda: self.provider_diagnostics_var.set(f"API-Key Test: Fehler - {exc}"))
                self.after(0, lambda: self.set_status(f"API-Key Test fehlgeschlagen: {exc}"))
            finally:
                if btn is not None:
                    self.after(0, lambda: btn.configure(state="normal", text="API-Key testen"))

        threading.Thread(target=worker, daemon=True).start()

    def run_provider_diagnostics_async(self) -> None:
        self.provider_diagnostics_var.set("Provider-Diagnostik: pruefe...")

        btn = getattr(self, "provider_diagnostics_btn", None)
        if btn is not None:
            btn.configure(state="disabled", text="Pruefe...")

        def worker() -> None:
            try:
                result = self.run_provider_diagnostics()
                provider_key = str(result.get("provider", "")).strip()
                provider_name = self._provider_display_name(provider_key)
                model_name = str(result.get("model", "-"))
                latency = float(result.get("latency_seconds", 0.0))
                models_count = int(result.get("models_count", 0))
                preview = result.get("models_preview", [])
                preview_text = ", ".join(preview) if preview else "-"
                text = (
                    f"Provider-Diagnostik: OK | {provider_name} | Modell: {model_name} | "
                    f"Modelle: {models_count} | RTT: {latency:.2f}s | Preview: {preview_text}"
                )
                self.after(0, lambda: self.provider_diagnostics_var.set(text))
                self.after(0, lambda: self.set_status(f"Provider-Diagnostik OK ({provider_name})"))
            except Exception as exc:
                self.after(0, lambda: self.provider_diagnostics_var.set(f"Provider-Diagnostik: Fehler - {exc}"))
                self.after(0, lambda: self.set_status(f"Provider-Diagnostik fehlgeschlagen: {exc}"))
            finally:
                if btn is not None:
                    self.after(0, lambda: btn.configure(state="normal", text="Provider-Diagnostik"))

        threading.Thread(target=worker, daemon=True).start()

    def _provider_config_export_payload(self) -> dict[str, Any]:
        return {
            "provider": self.llm_provider_var.get().strip(),
            "model": {
                "ollama_model": self.ollama_model_var.get().strip(),
                "ollama_url": self.ollama_url_var.get().strip(),
                "openai_model": self.openai_model_var.get().strip(),
                "openai_base_url": self.openai_base_url_var.get().strip(),
                "azure_openai_endpoint": self.azure_openai_endpoint_var.get().strip(),
                "azure_openai_deployment": self.azure_openai_deployment_var.get().strip(),
                "azure_openai_api_version": self.azure_openai_api_version_var.get().strip(),
                "anthropic_model": self.anthropic_model_var.get().strip(),
                "anthropic_base_url": self.anthropic_base_url_var.get().strip(),
                "anthropic_api_version": self.anthropic_api_version_var.get().strip(),
                "groq_model": self.groq_model_var.get().strip(),
                "groq_base_url": self.groq_base_url_var.get().strip(),
                "reply_max_tokens": self.reply_max_tokens_var.get().strip(),
                "reply_temperature": self.reply_temperature_var.get().strip(),
                "concise_reply": bool(self.concise_reply_var.get()),
            },
            "meta": {
                "version": 1,
                "contains_api_keys": False,
            },
        }

    def export_provider_config(self) -> None:
        from tkinter import filedialog

        default_name = "provider_config.json"
        target = filedialog.asksaveasfilename(
            title="Provider-Konfiguration exportieren",
            defaultextension=".json",
            initialfile=default_name,
            filetypes=[("JSON-Dateien", "*.json"), ("Alle Dateien", "*.*")],
        )
        if not target:
            return

        try:
            payload = self._provider_config_export_payload()
            Path(target).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self.set_status(f"Provider-Konfiguration exportiert: {Path(target).name}")
        except Exception as exc:
            self.set_status(f"Export fehlgeschlagen: {exc}")

    def import_provider_config(self) -> None:
        from tkinter import filedialog

        source = filedialog.askopenfilename(
            title="Provider-Konfiguration importieren",
            filetypes=[("JSON-Dateien", "*.json"), ("Alle Dateien", "*.*")],
        )
        if not source:
            return

        try:
            data = json.loads(Path(source).read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("Datei ist kein gueltiges JSON-Objekt")

            model = data.get("model", {}) if isinstance(data.get("model", {}), dict) else {}

            def _str(d: dict[str, Any], key: str, fallback: str) -> str:
                value = str(d.get(key, "")).strip()
                return value if value else fallback

            provider_label = self._canonical_provider_label(str(data.get("provider", self.llm_provider_var.get())))
            self.llm_provider_var.set(provider_label)

            self.ollama_model_var.set(_str(model, "ollama_model", self.ollama_model_var.get()))
            self.ollama_url_var.set(_str(model, "ollama_url", self.ollama_url_var.get()))
            self.openai_model_var.set(_str(model, "openai_model", self.openai_model_var.get()))
            self.openai_base_url_var.set(_str(model, "openai_base_url", self.openai_base_url_var.get()))
            self.azure_openai_endpoint_var.set(_str(model, "azure_openai_endpoint", self.azure_openai_endpoint_var.get()))
            self.azure_openai_deployment_var.set(_str(model, "azure_openai_deployment", self.azure_openai_deployment_var.get()))
            self.azure_openai_api_version_var.set(_str(model, "azure_openai_api_version", self.azure_openai_api_version_var.get()))
            self.anthropic_model_var.set(_str(model, "anthropic_model", self.anthropic_model_var.get()))
            self.anthropic_base_url_var.set(_str(model, "anthropic_base_url", self.anthropic_base_url_var.get()))
            self.anthropic_api_version_var.set(_str(model, "anthropic_api_version", self.anthropic_api_version_var.get()))
            self.groq_model_var.set(_str(model, "groq_model", self.groq_model_var.get()))
            self.groq_base_url_var.set(_str(model, "groq_base_url", self.groq_base_url_var.get()))
            self.reply_max_tokens_var.set(_str(model, "reply_max_tokens", self.reply_max_tokens_var.get()))
            self.reply_temperature_var.set(_str(model, "reply_temperature", self.reply_temperature_var.get()))
            if "concise_reply" in model:
                self.concise_reply_var.set(bool(model["concise_reply"]))

            self.on_llm_provider_changed(provider_label)
            self.set_status(f"Provider-Konfiguration importiert: {Path(source).name}")
        except Exception as exc:
            self.set_status(f"Import fehlgeschlagen: {exc}")
