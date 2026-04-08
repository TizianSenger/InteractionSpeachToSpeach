"""Profile persistence and persona state helpers for VoiceAssistantUI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from constants import PROFILE_FILE_NAME


class ProfilePersistenceMixin:
    def _profile_path(self) -> Path:
        return Path(__file__).resolve().parent / PROFILE_FILE_NAME

    def _set_persona_label_from_var(self, label_var: Any, value: float) -> None:
        label_var.set(str(int(max(0.0, min(100.0, value)))))

    def _refresh_persona_labels(self) -> None:
        self._set_persona_label_from_var(self.persona_flirty_label_var, self.persona_flirty_var.get())
        self._set_persona_label_from_var(self.persona_humor_label_var, self.persona_humor_var.get())
        self._set_persona_label_from_var(self.persona_serious_label_var, self.persona_serious_var.get())
        self._set_persona_label_from_var(self.persona_dominance_label_var, self.persona_dominance_var.get())
        self._set_persona_label_from_var(self.persona_empathy_label_var, self.persona_empathy_var.get())
        self._set_persona_label_from_var(self.persona_temperament_label_var, self.persona_temperament_var.get())

    def _on_persona_slider_changed(self, _value: float) -> None:
        self._refresh_persona_labels()

    def _collect_profile_data(self) -> dict[str, Any]:
        store_secrets = bool(self.store_api_keys_var.get())

        openai_key = self.openai_api_key_var.get().strip() if store_secrets else ""
        azure_key = self.azure_openai_api_key_var.get().strip() if store_secrets else ""
        anthropic_key = self.anthropic_api_key_var.get().strip() if store_secrets else ""
        groq_key = self.groq_api_key_var.get().strip() if store_secrets else ""
        gemini_key = self.gemini_api_key_var.get().strip() if store_secrets else ""

        return {
            "persona": {
                "flirty": float(self.persona_flirty_var.get()),
                "humor": float(self.persona_humor_var.get()),
                "serious": float(self.persona_serious_var.get()),
                "dominance": float(self.persona_dominance_var.get()),
                "empathy": float(self.persona_empathy_var.get()),
                "temperament": float(self.persona_temperament_var.get()),
            },
            "preferences": {
                "mic_device_label": self.mic_device_var.get().strip(),
                "tts_engine": self.tts_engine_var.get().strip(),
                "tts_voice": self.tts_voice_var.get().strip(),
                "tts_emotion": self.tts_emotion_var.get().strip(),
                "tts_rate": self.tts_rate_var.get().strip(),
                "piper_model_path": self.piper_model_path_var.get().strip(),
                "piper_config_path": self.piper_config_path_var.get().strip(),
                "appearance_mode": self.appearance_mode_var.get().strip(),
            },
            "stt": {
                "whisper_model": self.whisper_model_var.get().strip(),
                "whisper_language": self.whisper_language_var.get().strip(),
                "whisper_speed": self.whisper_speed_var.get().strip(),
            },
            "model": {
                "provider": self.llm_provider_var.get().strip(),
                "ollama_model": self.ollama_model_var.get().strip(),
                "ollama_url": self.ollama_url_var.get().strip(),
                "openai_model": self.openai_model_var.get().strip(),
                "openai_base_url": self.openai_base_url_var.get().strip(),
                "openai_api_key": openai_key,
                "azure_openai_endpoint": self.azure_openai_endpoint_var.get().strip(),
                "azure_openai_deployment": self.azure_openai_deployment_var.get().strip(),
                "azure_openai_api_key": azure_key,
                "azure_openai_api_version": self.azure_openai_api_version_var.get().strip(),
                "anthropic_model": self.anthropic_model_var.get().strip(),
                "anthropic_api_key": anthropic_key,
                "anthropic_base_url": self.anthropic_base_url_var.get().strip(),
                "anthropic_api_version": self.anthropic_api_version_var.get().strip(),
                "groq_model": self.groq_model_var.get().strip(),
                "groq_base_url": self.groq_base_url_var.get().strip(),
                "groq_api_key": groq_key,
                "gemini_model": self.gemini_model_var.get().strip(),
                "gemini_base_url": self.gemini_base_url_var.get().strip(),
                "gemini_api_key": gemini_key,
                "store_api_keys": store_secrets,
                "concise_reply": bool(self.concise_reply_var.get()),
                "reply_max_tokens": self.reply_max_tokens_var.get().strip(),
                "reply_temperature": self.reply_temperature_var.get().strip(),
            },
            "workflow": {
                "auto_speak": bool(self.auto_speak_var.get()),
                "auto_pipeline": bool(self.auto_pipeline_var.get()),
                "avatar_lipsync": bool(self.avatar_lipsync_var.get()),
                "vrm_model": self.vrm_model_var.get().strip(),
            },
            "audio": {
                "sample_rate": self.sample_rate_var.get().strip(),
                "vad_enabled": bool(self.vad_enabled_var.get()),
                "vad_aggressiveness": self.vad_aggressiveness_var.get().strip(),
                "vad_silence_timeout": self.vad_silence_timeout_var.get().strip(),
                "wake_word_enabled": bool(self.wake_word_enabled_var.get()),
                "wake_word_model": self.wake_word_model_var.get().strip(),
                "realtime_mode": self.realtime_mode_var.get().strip(),
            },
        }

    def save_profile(self, notify: bool = True) -> None:
        profile_path = self._profile_path()
        payload = self._collect_profile_data()
        profile_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self.logger.info("Profil gespeichert: %s", profile_path)
        if notify:
            self.set_status("Profil gespeichert")

    def _load_profile(self) -> None:
        profile_path = self._profile_path()
        if not profile_path.exists():
            self._refresh_persona_labels()
            return

        try:
            data = json.loads(profile_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
            persona = data.get("persona", {})
            preferences = data.get("preferences", {})
            stt = data.get("stt", {})
            model = data.get("model", {})
            workflow = data.get("workflow", {})
            audio = data.get("audio", {})

            # Persona
            self.persona_flirty_var.set(float(persona.get("flirty", self.persona_flirty_var.get())))
            self.persona_humor_var.set(float(persona.get("humor", self.persona_humor_var.get())))
            self.persona_serious_var.set(float(persona.get("serious", self.persona_serious_var.get())))
            self.persona_dominance_var.set(float(persona.get("dominance", self.persona_dominance_var.get())))
            self.persona_empathy_var.set(float(persona.get("empathy", self.persona_empathy_var.get())))
            self.persona_temperament_var.set(float(persona.get("temperament", self.persona_temperament_var.get())))

            # Preferences (TTS / Mikrofon / Darstellung)
            def _str(d: dict, key: str, fallback: str) -> str:
                v = str(d.get(key, "")).strip()
                return v if v else fallback

            self.mic_device_var.set(_str(preferences, "mic_device_label", self.mic_device_var.get()))
            self.tts_engine_var.set(_str(preferences, "tts_engine", self.tts_engine_var.get()))
            self.tts_voice_var.set(_str(preferences, "tts_voice", self.tts_voice_var.get()))
            self.tts_emotion_var.set(_str(preferences, "tts_emotion", self.tts_emotion_var.get()))
            self.tts_rate_var.set(_str(preferences, "tts_rate", self.tts_rate_var.get()))
            self.piper_model_path_var.set(_str(preferences, "piper_model_path", self.piper_model_path_var.get()))
            self.piper_config_path_var.set(_str(preferences, "piper_config_path", self.piper_config_path_var.get()))
            self.appearance_mode_var.set(_str(preferences, "appearance_mode", self.appearance_mode_var.get()))

            # STT
            self.whisper_model_var.set(_str(stt, "whisper_model", self.whisper_model_var.get()))
            self.whisper_language_var.set(_str(stt, "whisper_language", self.whisper_language_var.get()))
            self.whisper_speed_var.set(_str(stt, "whisper_speed", self.whisper_speed_var.get()))

            # Modell
            loaded_provider = _str(model, "provider", self.llm_provider_var.get())
            self.llm_provider_var.set(self._canonical_provider_label(loaded_provider))
            if "store_api_keys" in model:
                self.store_api_keys_var.set(bool(model["store_api_keys"]))
            self.ollama_model_var.set(_str(model, "ollama_model", self.ollama_model_var.get()))
            self.ollama_url_var.set(_str(model, "ollama_url", self.ollama_url_var.get()))
            self.openai_model_var.set(_str(model, "openai_model", self.openai_model_var.get()))
            self.openai_base_url_var.set(_str(model, "openai_base_url", self.openai_base_url_var.get()))
            self.openai_api_key_var.set(_str(model, "openai_api_key", self.openai_api_key_var.get()))
            self.azure_openai_endpoint_var.set(_str(model, "azure_openai_endpoint", self.azure_openai_endpoint_var.get()))
            self.azure_openai_deployment_var.set(_str(model, "azure_openai_deployment", self.azure_openai_deployment_var.get()))
            self.azure_openai_api_key_var.set(_str(model, "azure_openai_api_key", self.azure_openai_api_key_var.get()))
            self.azure_openai_api_version_var.set(_str(model, "azure_openai_api_version", self.azure_openai_api_version_var.get()))
            self.anthropic_model_var.set(_str(model, "anthropic_model", self.anthropic_model_var.get()))
            self.anthropic_api_key_var.set(_str(model, "anthropic_api_key", self.anthropic_api_key_var.get()))
            self.anthropic_base_url_var.set(_str(model, "anthropic_base_url", self.anthropic_base_url_var.get()))
            self.anthropic_api_version_var.set(_str(model, "anthropic_api_version", self.anthropic_api_version_var.get()))
            self.groq_model_var.set(_str(model, "groq_model", self.groq_model_var.get()))
            self.groq_base_url_var.set(_str(model, "groq_base_url", self.groq_base_url_var.get()))
            self.groq_api_key_var.set(_str(model, "groq_api_key", self.groq_api_key_var.get()))
            self.gemini_model_var.set(_str(model, "gemini_model", self.gemini_model_var.get()))
            self.gemini_base_url_var.set(_str(model, "gemini_base_url", self.gemini_base_url_var.get()))
            self.gemini_api_key_var.set(_str(model, "gemini_api_key", self.gemini_api_key_var.get()))
            if not self.store_api_keys_var.get():
                self.openai_api_key_var.set("")
                self.azure_openai_api_key_var.set("")
                self.anthropic_api_key_var.set("")
                self.groq_api_key_var.set("")
                self.gemini_api_key_var.set("")
            if "concise_reply" in model:
                self.concise_reply_var.set(bool(model["concise_reply"]))
            self.reply_max_tokens_var.set(_str(model, "reply_max_tokens", self.reply_max_tokens_var.get()))
            self.reply_temperature_var.set(_str(model, "reply_temperature", self.reply_temperature_var.get()))

            # Workflow
            if "auto_speak" in workflow:
                self.auto_speak_var.set(bool(workflow["auto_speak"]))
            if "auto_pipeline" in workflow:
                self.auto_pipeline_var.set(bool(workflow["auto_pipeline"]))
            if "avatar_lipsync" in workflow:
                self.avatar_lipsync_var.set(bool(workflow["avatar_lipsync"]))
            vrm_name = _str(workflow, "vrm_model", self.vrm_model_var.get())
            self.vrm_model_var.set(vrm_name)
            # Apply immediately so the auto-start viewer uses the saved model
            vrm_abs = Path(__file__).resolve().parent / "runtime_assets" / "model" / vrm_name
            if vrm_abs.exists():
                self.avatar_bridge.vrm_relative_path = f"runtime_assets/model/{vrm_name}"

            # Audio
            self.sample_rate_var.set(_str(audio, "sample_rate", self.sample_rate_var.get()))
            if "vad_enabled" in audio:
                self.vad_enabled_var.set(bool(audio["vad_enabled"]))
            self.vad_aggressiveness_var.set(_str(audio, "vad_aggressiveness", self.vad_aggressiveness_var.get()))
            self.vad_silence_timeout_var.set(_str(audio, "vad_silence_timeout", self.vad_silence_timeout_var.get()))
            if "wake_word_enabled" in audio:
                self.wake_word_enabled_var.set(bool(audio["wake_word_enabled"]))
            self.wake_word_model_var.set(_str(audio, "wake_word_model", self.wake_word_model_var.get()))
            self.realtime_mode_var.set(_str(audio, "realtime_mode", self.realtime_mode_var.get()))
        except Exception as exc:
            self.logger.warning("Profil konnte nicht geladen werden: %s", exc)

        self._refresh_persona_labels()
