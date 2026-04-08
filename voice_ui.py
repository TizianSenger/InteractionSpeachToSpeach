import asyncio
from collections import deque
import ctypes
import ctypes.wintypes
from datetime import datetime
import importlib
import json
import logging
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Any, Callable

import customtkinter as ctk
import numpy as np
import pyttsx3
import requests
import sounddevice as sd

from avatar_bridge import AvatarBridge
from constants import *
from ollama_mixin import OllamaMixin
from tts_mixin import TtsMixin
from wake_word_mixin import WakeWordMixin

_FFMPEG_CHECKED = False
_FFMPEG_AVAILABLE = False


def ensure_ffmpeg_available() -> bool:
    global _FFMPEG_CHECKED
    global _FFMPEG_AVAILABLE

    if _FFMPEG_CHECKED:
        return _FFMPEG_AVAILABLE

    if shutil.which("ffmpeg"):
        _FFMPEG_CHECKED = True
        _FFMPEG_AVAILABLE = True
        return True

    packages_root = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    if not packages_root.exists():
        _FFMPEG_CHECKED = True
        _FFMPEG_AVAILABLE = False
        return False

    candidates = list(packages_root.rglob("ffmpeg.exe"))
    if not candidates:
        _FFMPEG_CHECKED = True
        _FFMPEG_AVAILABLE = False
        return False

    ffmpeg_dir = str(candidates[0].parent)
    current_path = os.environ.get("PATH", "")
    if ffmpeg_dir not in current_path:
        os.environ["PATH"] = f"{ffmpeg_dir};{current_path}" if current_path else ffmpeg_dir

    _FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
    _FFMPEG_CHECKED = True
    return _FFMPEG_AVAILABLE


class UILogQueueHandler(logging.Handler):
    def __init__(self, target_queue: queue.Queue[str]) -> None:
        super().__init__()
        self.target_queue = target_queue

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            self.target_queue.put(message)
        except Exception:
            pass


class VoiceAssistantUI(OllamaMixin, TtsMixin, WakeWordMixin, ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        self.withdraw()               # Hauptfenster ausblenden
        self.attributes("-alpha", 0.0)  # zusätzlich komplett transparent – kein kurzes Aufblitzen

        self.title("Voice Assistant")
        self.geometry("1160x760")

        self.recording_stream: sd.InputStream | None = None  # kept for compat. checks
        self.monitor_stream: sd.InputStream | None = None     # kept for compat. checks
        self._bg_stream: sd.InputStream | None = None         # unified permanent stream
        self.recording_wave: wave.Wave_write | None = None
        self._recording_wave_lock = threading.Lock()
        self.recording_path: str | None = None
        self.is_recording = False
        self._closing: bool = False
        # Splash screen state
        self._splash: Any = None
        self._splash_canvas: Any = None
        self._splash_angle: int = 0
        self._splash_spinner_job: Any = None
        self._splash_status_var: Any = None
        # VAD runtime state (reset at each recording start)
        self.vad_speech_detected: bool = False
        self.vad_last_speech_time: float = 0.0
        self.last_transcript: str = ""
        self.whisper_model_cache: dict[str, Any] = {}
        self.whisper_module: Any | None = None
        self.mic_devices_map: dict[str, int] = {}
        self.piper_models_map: dict[str, str] = {}
        self.http_session = requests.Session()
        self.ollama_cache_ttl_seconds = 300.0
        self.ollama_check_cache: dict[str, Any] = {
            "cache_key": "",
            "checked_at": 0.0,
            "model_name": "",
            "ollama_url": "",
        }
        self.cancel_ollama_event = threading.Event()
        self.cancel_tts_event = threading.Event()
        self.active_ollama_response_lock = threading.Lock()
        self.active_ollama_response: Any | None = None
        self.current_tts_queue: queue.Queue[str | None] | None = None
        self.pyttsx3_engine: Any | None = None
        self.pyttsx3_engine_lock = threading.Lock()
        self.avatar_lipsync_var = ctk.BooleanVar(value=True)
        self.vrm_model_var = ctk.StringVar(value="vrm_AvatarSample_S.vrm")
        self.column_visible: dict[str, bool] = {"left": False, "middle": True, "right": True}
        self.column_frames: dict[str, ctk.CTkFrame] = {}
        self.body_frame: ctk.CTkFrame | None = None
        self.column_weights: dict[str, int] = {"left": 5, "middle": 2, "right": 5}
        self.settings_popup: ctk.CTkToplevel | None = None
        self.viewer_host_frame: ctk.CTkFrame | None = None
        self.embedded_viewer_hwnd: int | None = None
        self._viewer_titlebar_h: int = 33  # Chromium-rendered title bar height (pixels)
        self.light_popup: ctk.CTkToplevel | None = None
        self.light_state_label: ctk.CTkLabel | None = None
        self.light_indicator: ctk.CTkFrame | None = None
        self.light_state = False
        self.light_state_var = ctk.StringVar(value="Lichtstatus: AUS")
        self.whisper_backend = ""
        self.stt_progress_var = ctk.StringVar(value="STT Laden: 0%")
        self.stt_loading_active = False
        self.stt_loading_value = 0.0
        self.stt_loading_job_id: str | None = None
        self.stt_loading_model_name = ""
        self.recording_started_at: float | None = None
        self.pipeline_phase: str = "idle"   # idle | mic | stt | ollama | tts
        self._pipeline_anim_job: str | None = None
        self._pipeline_anim_tick: int = 0
        self._cursor_active: bool = False
        self._cursor_job: str | None = None
        self._cursor_visible: bool = False
        self._thinking_active: bool = False
        self._thinking_job: str | None = None
        self._thinking_tick: int = 0
        self._active_response_started_at: float | None = None
        self._first_audio_chunk_recorded: bool = False
        self.realtime_mode_var = ctk.StringVar(value="Balanced")
        self.store_api_keys_var = ctk.BooleanVar(value=True)
        self._tts_stream_first_chars = TTS_STREAM_FIRST_CHARS
        self._tts_stream_min_chars = TTS_STREAM_MIN_CHARS
        self._tts_stream_max_buffer_chars = TTS_STREAM_MAX_BUFFER_CHARS
        self._tts_stream_max_wait_seconds = TTS_STREAM_MAX_WAIT_SECONDS
        self.thinking_canvas: Any | None = None
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.log_pump_job_id: str | None = None
        self.stats_refresh_job_id: str | None = None
        self.max_ui_log_lines = 1500
        self.debug_log_history: deque[str] = deque(maxlen=2500)
        self.event_history: deque[str] = deque(maxlen=400)
        # Stores {"role": "user"|"assistant", "content": str} for multi-turn context.
        # Capped at 20 turns (40 messages) to keep prompt size manageable.
        self.conversation_history: deque[dict[str, str]] = deque(maxlen=40)
        # Waveform visualiser – ring buffer of amplitude samples drawn in the middle column.
        self.waveform_samples: deque[float] = deque(maxlen=120)
        self.waveform_canvas: Any | None = None
        self._ww_flash_frames: int = 0   # >0 while wake-word flash is running
        self._ww_pulse_tick: int = 0     # increments each waveform frame for pulse effect
        self.waveform_animate_job: str | None = None
        self.metric_samples: dict[str, list[float]] = {
            "recording_seconds": [],
            "transcription_seconds": [],
            "ollama_first_token_seconds": [],
            "ollama_total_seconds": [],
            "tts_chunk_seconds": [],
            "first_audio_seconds": [],
        }
        self.counters: dict[str, int] = {
            "recordings_started": 0,
            "recordings_finished": 0,
            "transcriptions": 0,
            "ollama_requests": 0,
            "ollama_cancels": 0,
            "tts_chunks": 0,
            "errors": 0,
        }
        self.app_started_at = time.time()
        self.logger = self._setup_logger()
        self.avatar_bridge = AvatarBridge(
            base_dir=Path(__file__).resolve().parent,
            http_session=self.http_session,
            lipsync_enabled_getter=lambda: bool(self.avatar_lipsync_var.get()),
            set_status=self.set_status,
            log_exception=self._log_exception,
            logger=self.logger,
        )

        self.status_var = ctk.StringVar(value="Bereit")
        self.provider_diagnostics_var = ctk.StringVar(value="Provider-Diagnostik: -")
        self.debug_log_level_var = ctk.StringVar(value="INFO")
        self.stats_summary_var = ctk.StringVar(value="Keine Daten")
        self.stats_latency_var = ctk.StringVar(value="Latenzen: -")
        self.whisper_model_var = ctk.StringVar(value="small")
        self.llm_provider_var = ctk.StringVar(value="Ollama")
        self.ollama_model_var = ctk.StringVar(value="phi4-mini")
        self.ollama_url_var = ctk.StringVar(value="http://localhost:11434")
        self.openai_model_var = ctk.StringVar(value="gpt-4o-mini")
        self.openai_base_url_var = ctk.StringVar(value=OPENAI_DEFAULT_BASE_URL)
        self.openai_api_key_var = ctk.StringVar(value="")
        self.azure_openai_endpoint_var = ctk.StringVar(value="")
        self.azure_openai_deployment_var = ctk.StringVar(value="")
        self.azure_openai_api_key_var = ctk.StringVar(value="")
        self.azure_openai_api_version_var = ctk.StringVar(value=AZURE_OPENAI_DEFAULT_API_VERSION)
        self.anthropic_model_var = ctk.StringVar(value="claude-3-5-sonnet-latest")
        self.anthropic_api_key_var = ctk.StringVar(value="")
        self.anthropic_base_url_var = ctk.StringVar(value=ANTHROPIC_DEFAULT_BASE_URL)
        self.anthropic_api_version_var = ctk.StringVar(value=ANTHROPIC_DEFAULT_API_VERSION)
        self.groq_model_var = ctk.StringVar(value="llama-3.3-70b-versatile")
        self.groq_base_url_var = ctk.StringVar(value=GROQ_DEFAULT_BASE_URL)
        self.groq_api_key_var = ctk.StringVar(value="")
        self.whisper_language_var = ctk.StringVar(value="Deutsch")
        self.whisper_speed_var = ctk.StringVar(value="Genau")
        self.mic_device_var = ctk.StringVar(value="")
        self.sample_rate_var = ctk.StringVar(value="16000")
        self.tts_rate_var = ctk.StringVar(value="170")
        self.tts_engine_var = ctk.StringVar(value="edge-tts (natürlich)")
        self.tts_voice_var = ctk.StringVar(value="Deutsch (männlich, tief) - Killian")
        self.tts_emotion_var = ctk.StringVar(value="freundlich")
        self.appearance_mode_var = ctk.StringVar(value="Dark")
        self.piper_model_path_var = ctk.StringVar(value="models/de_DE-karlsson-medium.onnx")
        self.piper_config_path_var = ctk.StringVar(value="")
        self.mic_level_text_var = ctk.StringVar(value="Pegel: 0%")
        self.auto_speak_var = ctk.BooleanVar(value=True)
        self.auto_pipeline_var = ctk.BooleanVar(value=True)
        self.reply_max_tokens_var = ctk.StringVar(value="120")
        self.reply_temperature_var = ctk.StringVar(value="0.3")
        self.concise_reply_var = ctk.BooleanVar(value=True)

        self.persona_flirty_var = ctk.DoubleVar(value=35.0)
        self.persona_humor_var = ctk.DoubleVar(value=55.0)
        self.persona_serious_var = ctk.DoubleVar(value=40.0)
        self.persona_dominance_var = ctk.DoubleVar(value=35.0)
        self.persona_empathy_var = ctk.DoubleVar(value=70.0)
        self.persona_temperament_var = ctk.DoubleVar(value=30.0)
        self.persona_flirty_label_var = ctk.StringVar(value="35")
        self.persona_humor_label_var = ctk.StringVar(value="55")
        self.persona_serious_label_var = ctk.StringVar(value="40")
        self.persona_dominance_label_var = ctk.StringVar(value="35")
        self.persona_empathy_label_var = ctk.StringVar(value="70")
        self.persona_temperament_label_var = ctk.StringVar(value="30")
        self.vad_enabled_var = ctk.BooleanVar(value=True)
        self.vad_aggressiveness_var = ctk.StringVar(value="1")
        self.vad_silence_timeout_var = ctk.StringVar(value="0.8")
        # Wake-word listener state
        self.wake_word_enabled_var = ctk.BooleanVar(value=True)
        self.wake_word_model_var = ctk.StringVar(value="Hey Jarvis")
        self.wake_word_status_var = ctk.StringVar(value="Inaktiv")
        self._ww_thread: threading.Thread | None = None
        self._ww_stop_event: threading.Event = threading.Event()
        self._ww_whisper_model: Any | None = None

        self._load_profile()
        self.apply_realtime_preset(self.realtime_mode_var.get(), announce=False)
        # Remember the theme that was active at startup for restart-guard
        self._active_appearance_mode = self.appearance_mode_var.get().strip().capitalize() or "Dark"
        self.on_appearance_mode_changed(self.appearance_mode_var.get())

        self._build_layout()
        self.refresh_piper_model_options()
        self.on_tts_engine_changed(self.tts_engine_var.get())
        self.refresh_input_devices()
        self._start_bg_stream()
        self.start_wake_word_listener()  # no-op if wake_word_enabled_var is False
        self._show_splash()              # Splash einblenden bevor Warmup startet
        self._start_background_warmup()
        self._schedule_log_pump()
        self._schedule_stats_refresh()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        # Avatar wird erst nach dem Splash-Schließen gestartet (nicht mehr hier)

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("voice_studio_ui")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False

        if not logger.handlers:
            log_dir = Path(__file__).resolve().parent / LOG_DIR_NAME
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / LOG_FILE_NAME
            formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(threadName)s | %(message)s")

            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            ui_handler = UILogQueueHandler(self.log_queue)
            ui_handler.setLevel(logging.DEBUG)
            ui_handler.setFormatter(formatter)
            logger.addHandler(ui_handler)

        logger.info("Voice Studio gestartet")
        return logger

    def _schedule_log_pump(self) -> None:
        self._pump_logs_into_ui()
        self.log_pump_job_id = self.after(200, self._schedule_log_pump)

    def _extract_level_token(self, line: str) -> str:
        parts = line.split("|")
        if len(parts) >= 2:
            return parts[1].strip().upper()
        return "INFO"

    def _matches_selected_log_level(self, line: str) -> bool:
        log_level = self.debug_log_level_var.get().strip().upper() or "INFO"
        level_order = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}
        selected_level = level_order.get(log_level, 20)
        line_level = self._extract_level_token(line)
        return level_order.get(line_level, 20) >= selected_level

    def _insert_debug_log_line(self, line: str) -> None:
        if not hasattr(self, "debug_log_box"):
            return

        level_token = self._extract_level_token(line)
        tag_name = {
            "DEBUG": "log_debug",
            "INFO": "log_info",
            "WARNING": "log_warning",
            "ERROR": "log_error",
            "CRITICAL": "log_critical",
        }.get(level_token, "log_info")
        self.debug_log_box.insert("end", f"{line}\n", tag_name)

    def _pump_logs_into_ui(self) -> None:
        if not hasattr(self, "debug_log_box"):
            return

        inserted_any = False
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break

            self.debug_log_history.append(line)
            if self._matches_selected_log_level(line):
                self._insert_debug_log_line(line)
                inserted_any = True

        if inserted_any:
            self.debug_log_box.see("end")

            current_end = self.debug_log_box.index("end-1c")
            try:
                line_count = int(float(current_end.split(".")[0]))
            except Exception:
                line_count = 0
            overflow = max(0, line_count - self.max_ui_log_lines)
            if overflow > 0:
                self.debug_log_box.delete("1.0", f"{overflow + 1}.0")

    def _track_event(self, label: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.event_history.append(f"[{timestamp}] {label}")
        self._refresh_event_box()

    def _increment_counter(self, name: str, amount: int = 1) -> None:
        current = self.counters.get(name, 0)
        self.counters[name] = current + amount

    def _add_metric_sample(self, name: str, value: float) -> None:
        samples = self.metric_samples.get(name)
        if samples is None:
            return
        samples.append(value)
        if len(samples) > 300:
            del samples[: len(samples) - 300]

    def _format_metric(self, name: str) -> str:
        values = self.metric_samples.get(name, [])
        if not values:
            return "-"
        avg = sum(values) / len(values)
        return f"{avg:.2f}s (n={len(values)})"

    def _schedule_stats_refresh(self) -> None:
        self._refresh_stats_view()
        self.stats_refresh_job_id = self.after(1200, self._schedule_stats_refresh)

    def _refresh_event_box(self) -> None:
        if not hasattr(self, "events_box"):
            return
        history = list(self.event_history)
        current_len = len(history)
        prev_len = getattr(self, "_events_displayed", 0)
        if current_len == prev_len:
            return
        if current_len < prev_len or prev_len == 0:
            # Erster Aufruf oder Deque rotiert: kompletter Neuaufbau
            self.events_box.delete("1.0", "end")
            self.events_box.insert("1.0", "\n".join(history) if history else "Noch keine Events")
        else:
            # Nur neue Einträge anhängen (O(1) statt O(n))
            for entry in history[prev_len:]:
                self.events_box.insert("end", f"\n{entry}")
        self._events_displayed = current_len
        self.events_box.see("end")

    def _refresh_stats_view(self) -> None:
        uptime_seconds = max(0, int(time.time() - self.app_started_at))
        minutes, seconds = divmod(uptime_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        uptime = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        summary = (
            f"Uptime: {uptime}\n"
            f"Recordings: {self.counters.get('recordings_started', 0)} gestartet / "
            f"{self.counters.get('recordings_finished', 0)} beendet\n"
            f"Transkriptionen: {self.counters.get('transcriptions', 0)}\n"
            f"Ollama Requests: {self.counters.get('ollama_requests', 0)}\n"
            f"Abbrueche: {self.counters.get('ollama_cancels', 0)}\n"
            f"TTS Chunks: {self.counters.get('tts_chunks', 0)}\n"
            f"Fehler: {self.counters.get('errors', 0)}"
        )
        self.stats_summary_var.set(summary)

        latency_text = (
            f"Transkription: {self._format_metric('transcription_seconds')} | "
            f"First token: {self._format_metric('ollama_first_token_seconds')} | "
            f"First audio: {self._format_metric('first_audio_seconds')} | "
            f"Ollama gesamt: {self._format_metric('ollama_total_seconds')} | "
            f"TTS Chunk: {self._format_metric('tts_chunk_seconds')}"
        )
        self.stats_latency_var.set(latency_text)

    def apply_realtime_preset(self, mode: str, announce: bool = True) -> None:
        normalized = (mode or "Balanced").strip().capitalize()
        if normalized not in set(REALTIME_MODE_OPTIONS):
            normalized = "Balanced"

        if normalized == "Aggressiv":
            self._tts_stream_first_chars = 16
            self._tts_stream_min_chars = 28
            self._tts_stream_max_buffer_chars = 70
            self._tts_stream_max_wait_seconds = 0.22
        elif normalized == "Stabil":
            self._tts_stream_first_chars = 32
            self._tts_stream_min_chars = 55
            self._tts_stream_max_buffer_chars = 120
            self._tts_stream_max_wait_seconds = 0.50
        else:
            self._tts_stream_first_chars = TTS_STREAM_FIRST_CHARS
            self._tts_stream_min_chars = TTS_STREAM_MIN_CHARS
            self._tts_stream_max_buffer_chars = TTS_STREAM_MAX_BUFFER_CHARS
            self._tts_stream_max_wait_seconds = TTS_STREAM_MAX_WAIT_SECONDS

        self.realtime_mode_var.set(normalized)
        if announce:
            self.set_status(f"Realtime-Modus aktiv: {normalized}")

    def on_realtime_mode_changed(self, selected_mode: str) -> None:
        self.apply_realtime_preset(selected_mode, announce=True)

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

    def clear_debug_logs(self) -> None:
        if hasattr(self, "debug_log_box"):
            self.debug_log_box.delete("1.0", "end")

    def on_debug_log_level_changed(self, _selected: str) -> None:
        self.clear_debug_logs()
        for line in self.debug_log_history:
            if self._matches_selected_log_level(line):
                self._insert_debug_log_line(line)
        if hasattr(self, "debug_log_box"):
            self.debug_log_box.see("end")

    def _log_exception(self, context: str, exc: Exception) -> None:
        self._increment_counter("errors")
        self._track_event(f"Fehler in {context}: {exc}")
        self.logger.exception("%s: %s", context, exc)

    def _profile_path(self) -> Path:
        return Path(__file__).resolve().parent / PROFILE_FILE_NAME

    def _set_persona_label_from_var(self, label_var: ctk.StringVar, value: float) -> None:
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
            if not self.store_api_keys_var.get():
                self.openai_api_key_var.set("")
                self.azure_openai_api_key_var.set("")
                self.anthropic_api_key_var.set("")
                self.groq_api_key_var.set("")
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

    def on_appearance_mode_changed(self, selected_mode: str) -> None:
        normalized = selected_mode.strip().capitalize() if selected_mode else "Dark"
        if normalized not in {"Dark", "Light", "System"}:
            normalized = "Dark"

        self.appearance_mode_var.set(normalized)
        ctk.set_appearance_mode(normalized)

        # Only restart when the user changed the theme after startup
        active = getattr(self, "_active_appearance_mode", None)
        if active is not None and normalized != active:
            try:
                self.save_profile(notify=False)
            except Exception:
                pass
            self.after(80, self._restart_process)

    def _restart_process(self) -> None:
        try:
            self._closing = True
            self._stop_bg_stream()
            self.stop_wake_word_listener()
        except Exception:
            pass
        subprocess.Popen([sys.executable] + sys.argv)
        self.destroy()

    def _persona_instruction(self, key: str, value: float) -> str:
        if key == "flirty":
            if value >= 70:
                return "dezent flirtend und charmant"
            if value >= 40:
                return "warm, spielerisch und freundlich"
            return "neutral-professionell"

        if key == "humor":
            if value >= 70:
                return "mit leichtem, intelligentem Humor"
            if value >= 40:
                return "locker und auflockernd"
            return "ohne Witze"

        if key == "serious":
            if value >= 70:
                return "sachlich, praezise und fokussiert"
            if value >= 40:
                return "ausgewogen zwischen locker und sachlich"
            return "locker-konversationell"

        if key == "dominance":
            if value >= 70:
                return "klar fuehrend mit konkreten Vorschlaegen"
            if value >= 40:
                return "leitend, aber kooperativ"
            return "zurueckhaltend und fragend"

        if key == "empathy":
            if value >= 70:
                return "sehr empathisch und validierend"
            if value >= 40:
                return "freundlich und verstaendnisvoll"
            return "direkt ohne viel Emotionalisierung"

        if value >= 70:
            return "dynamisch und energisch"
        if value >= 40:
            return "ausgeglichen im Tempo"
        return "ruhig und kontrolliert"

    def _build_persona_system_prompt(self) -> str:
        flirty = self._persona_instruction("flirty", float(self.persona_flirty_var.get()))
        humor = self._persona_instruction("humor", float(self.persona_humor_var.get()))
        serious = self._persona_instruction("serious", float(self.persona_serious_var.get()))
        dominance = self._persona_instruction("dominance", float(self.persona_dominance_var.get()))
        empathy = self._persona_instruction("empathy", float(self.persona_empathy_var.get()))
        temperament = self._persona_instruction("temperament", float(self.persona_temperament_var.get()))
        concise_instruction = "Antworte in 1-3 Saetzen." if self.concise_reply_var.get() else "Antworte so detailliert wie noetig."
        return (
            "Du bist ein persoenlicher Sprachassistent auf Deutsch. "
            "Sei menschlich, charmant und kontextbewusst."
            f"Stil: {flirty}; {humor}; {serious}; {dominance}; {empathy}; {temperament}. "
            "Verwende KEINE Emojis in deinen Antworten. "
            f"{concise_instruction}"
        )

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)

        topbar = ctk.CTkFrame(self)
        topbar.grid(row=0, column=0, padx=12, pady=(10, 6), sticky="ew")
        topbar.grid_columnconfigure(5, weight=1)

        ctk.CTkLabel(topbar, text="Voice Studio", font=(FONT_FAMILY, 20, "bold")).grid(
            row=0, column=0, padx=(10, 12), pady=8, sticky="w"
        )
        ctk.CTkButton(topbar, text="Links", width=72, command=lambda: self._toggle_column("left")).grid(
            row=0, column=1, padx=4, pady=8
        )
        ctk.CTkButton(topbar, text="Mitte", width=72, command=lambda: self._toggle_column("middle")).grid(
            row=0, column=2, padx=4, pady=8
        )
        ctk.CTkButton(topbar, text="Rechts", width=72, command=lambda: self._toggle_column("right")).grid(
            row=0, column=3, padx=4, pady=8
        )
        ctk.CTkButton(topbar, text="Einstellungen", width=120, command=self.open_settings_popup).grid(
            row=0, column=4, padx=8, pady=8
        )

        # ── Status-Anzeige (streckt sich, Echtzeit-Feedback) ───────────
        self.status_label = ctk.CTkLabel(
            topbar, textvariable=self.status_var,
            font=(FONT_FAMILY, 13), text_color="#94a3b8", anchor="w",
        )
        self.status_label.grid(row=0, column=5, padx=(16, 8), pady=8, sticky="ew")

        # ── Pipeline-Indikator (rechts) ──────────────────────────────────
        import tkinter as tk
        self.pipeline_canvas = tk.Canvas(
            topbar, bg="#1a2236", highlightthickness=0, height=36, width=340
        )
        self.pipeline_canvas.grid(row=0, column=6, padx=(0, 8), pady=6, sticky="e")

        body = ctk.CTkFrame(self)
        body.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")
        self.body_frame = body
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=5)
        body.grid_columnconfigure(1, weight=2)
        body.grid_columnconfigure(2, weight=5)

        left_col = ctk.CTkFrame(body)
        left_col.grid(row=0, column=0, padx=(0, 6), pady=0, sticky="nsew")
        left_col.grid_rowconfigure(1, weight=1)
        left_col.grid_columnconfigure(0, weight=1)

        middle_col = ctk.CTkFrame(body)
        middle_col.grid(row=0, column=1, padx=6, pady=0, sticky="nsew")
        middle_col.grid_rowconfigure(1, weight=1)
        middle_col.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            middle_col,
            text="Waveform",
            font=(FONT_FAMILY, 14, "bold"),
        ).grid(row=0, column=0, padx=12, pady=(10, 4), sticky="w")

        waveform_host = ctk.CTkFrame(middle_col, fg_color="#0d1117")
        waveform_host.grid(row=1, column=0, padx=8, pady=(0, 8), sticky="nsew")
        waveform_host.grid_columnconfigure(0, weight=1)
        waveform_host.grid_rowconfigure(0, weight=1)
        self.waveform_canvas = tk.Canvas(
            waveform_host,
            bg="#0d1117",
            highlightthickness=0,
        )
        self.waveform_canvas.grid(row=0, column=0, sticky="nsew")
        self._schedule_waveform_draw()

        right_col = ctk.CTkFrame(body)
        right_col.grid(row=0, column=2, padx=(6, 0), pady=0, sticky="nsew")
        right_col.grid_rowconfigure(1, weight=1)
        right_col.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(right_col, text="Model Viewer", font=(FONT_FAMILY, 16, "bold")).grid(
            row=0, column=0, padx=12, pady=(10, 6), sticky="w"
        )

        self.viewer_host_frame = ctk.CTkFrame(right_col, fg_color="#101524")
        self.viewer_host_frame.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        ctk.CTkLabel(
            self.viewer_host_frame,
            text="Avatar starten, um den Viewer hier einzubetten",
            text_color="gray70",
        ).place(relx=0.5, rely=0.5, anchor="center")
        self.viewer_host_frame.bind("<Configure>", lambda _evt: self._resize_docked_viewer())

        self.column_frames = {"left": left_col, "middle": middle_col, "right": right_col}
        self._refresh_column_layout()

        workflow = ctk.CTkFrame(left_col)
        workflow.grid(row=0, column=0, padx=8, pady=(8, 6), sticky="ew")
        for i in range(6):
            workflow.grid_columnconfigure(i, weight=1)

        self.start_btn = ctk.CTkButton(workflow, text="Mic Start", command=self.start_recording)
        self.start_btn.grid(row=0, column=0, padx=4, pady=6, sticky="ew")

        self.stop_btn = ctk.CTkButton(workflow, text="Mic Stop", command=self.stop_recording, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=4, pady=6, sticky="ew")

        self.transcribe_btn = ctk.CTkButton(
            workflow,
            text="Transkribieren",
            command=self.transcribe_recording,
            state="disabled",
            fg_color="#1f6aa5",
        )
        self.transcribe_btn.grid(row=0, column=2, padx=4, pady=6, sticky="ew")

        self.send_btn = ctk.CTkButton(
            workflow,
            text="Senden",
            command=self.send_to_ollama,
            state="disabled",
            fg_color="#0f766e",
        )
        self.send_btn.grid(row=0, column=3, padx=4, pady=6, sticky="ew")

        self.cancel_btn = ctk.CTkButton(
            workflow,
            text="Abbrechen",
            command=self.cancel_current_response,
            state="disabled",
            fg_color="#b42318",
            hover_color="#8f1d15",
        )
        self.cancel_btn.grid(row=0, column=4, padx=4, pady=6, sticky="ew")

        self.avatar_btn = ctk.CTkButton(
            workflow,
            text="Avatar starten",
            command=self.toggle_avatar_viewer,
            fg_color="#1d4ed8",
            hover_color="#1e40af",
        )
        self.avatar_btn.grid(row=0, column=5, padx=4, pady=6, sticky="ew")

        self.text_send_btn = ctk.CTkButton(
            workflow,
            text="Text senden",
            command=self.send_to_ollama,
            fg_color="#7c3aed",
        )
        self.text_send_btn.grid(row=1, column=0, padx=4, pady=6, sticky="ew")

        tabs = ctk.CTkTabview(left_col)
        tabs.grid(row=1, column=0, padx=8, pady=(0, 8), sticky="nsew")

        chat_tab = tabs.add("Chat")
        chat_tab.grid_columnconfigure(0, weight=1)
        chat_tab.grid_rowconfigure(1, weight=1)
        chat_tab.grid_rowconfigure(4, weight=1)

        ctk.CTkLabel(chat_tab, text="Transkript", font=(FONT_FAMILY, 13, "bold")).grid(
            row=0, column=0, padx=10, pady=(10, 4), sticky="w"
        )
        self.transcript_box = ctk.CTkTextbox(chat_tab, wrap="word")
        self.transcript_box.grid(row=1, column=0, padx=10, pady=(0, 8), sticky="nsew")
        self.transcript_box.insert("1.0", "Du kannst hier auch direkt Text eintippen und dann auf 'Text senden' klicken.")

        ctk.CTkLabel(chat_tab, text="Antwort", font=(FONT_FAMILY, 13, "bold")).grid(
            row=2, column=0, padx=10, pady=(4, 2), sticky="w"
        )
        # Thinking indicator (hidden by default, shown while waiting for first token)
        self.thinking_canvas = tk.Canvas(
            chat_tab, bg="#0d1117", highlightthickness=0, height=18
        )
        self.thinking_canvas.grid(row=3, column=0, padx=10, pady=(0, 2), sticky="ew")
        self.thinking_canvas.grid_remove()   # hidden until needed
        self.answer_box = ctk.CTkTextbox(chat_tab, wrap="word")
        self.answer_box.grid(row=4, column=0, padx=10, pady=(0, 10), sticky="nsew")

        history_tab = tabs.add("Verlauf")
        history_tab.grid_rowconfigure(1, weight=1)
        history_tab.grid_columnconfigure(0, weight=1)

        history_header = ctk.CTkFrame(history_tab, fg_color="transparent")
        history_header.grid(row=0, column=0, padx=10, pady=(8, 4), sticky="ew")
        history_header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(history_header, text="Gesprächsverlauf", font=(FONT_FAMILY, 16, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        self.clear_history_btn = ctk.CTkButton(
            history_header,
            text="Verlauf leeren",
            width=130,
            command=self.clear_history,
        )
        self.clear_history_btn.grid(row=0, column=1, sticky="e")

        self.history_box = ctk.CTkTextbox(history_tab, wrap="word")
        self.history_box.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.history_box.insert("1.0", "Der Verlauf wird hier mit Zeitstempel angezeigt.\n")

        stats_tab = tabs.add("Statistik")
        stats_tab.grid_rowconfigure(2, weight=1)
        stats_tab.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(stats_tab, text="Laufzeit und Kennzahlen", font=(FONT_FAMILY, 16, "bold")).grid(
            row=0, column=0, padx=10, pady=(10, 4), sticky="w"
        )
        ctk.CTkLabel(stats_tab, textvariable=self.stats_summary_var, justify="left").grid(
            row=1, column=0, padx=10, pady=(0, 8), sticky="w"
        )
        ctk.CTkLabel(stats_tab, textvariable=self.stats_latency_var, justify="left", text_color="gray70").grid(
            row=2, column=0, padx=10, pady=(0, 8), sticky="w"
        )

        self.events_box = ctk.CTkTextbox(stats_tab, wrap="word", height=220)
        self.events_box.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.events_box.insert("1.0", "Noch keine Events")

        debug_tab = tabs.add("Debug Logs")
        debug_tab.grid_rowconfigure(1, weight=1)
        debug_tab.grid_columnconfigure(0, weight=1)

        debug_controls = ctk.CTkFrame(debug_tab, fg_color="transparent")
        debug_controls.grid(row=0, column=0, padx=10, pady=(8, 4), sticky="ew")
        debug_controls.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(debug_controls, text="Log-Level").grid(row=0, column=0, padx=(0, 6), pady=0, sticky="w")
        self.debug_level_menu = ctk.CTkOptionMenu(
            debug_controls,
            values=["DEBUG", "INFO", "WARNING", "ERROR"],
            variable=self.debug_log_level_var,
            command=self.on_debug_log_level_changed,
            width=120,
        )
        self.debug_level_menu.grid(row=0, column=1, padx=(0, 8), pady=0, sticky="w")

        self.clear_debug_btn = ctk.CTkButton(debug_controls, text="Logs leeren", command=self.clear_debug_logs, width=120)
        self.clear_debug_btn.grid(row=0, column=2, padx=(0, 8), pady=0, sticky="w")

        self.debug_log_box = ctk.CTkTextbox(debug_tab, wrap="none")
        self.debug_log_box.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        self.debug_log_box.tag_config("log_debug", foreground="#8f9bb3")
        self.debug_log_box.tag_config("log_info", foreground="#d7dce2")
        self.debug_log_box.tag_config("log_warning", foreground="#f4c86a")
        self.debug_log_box.tag_config("log_error", foreground="#ff6b6b")
        self.debug_log_box.tag_config("log_critical", foreground="#ff4d4d")

        self._build_settings_popup()

    def _build_settings_popup(self) -> None:
        popup = ctk.CTkToplevel(self)
        popup.title("Einstellungen")
        popup.geometry("660x820")
        popup.minsize(560, 600)
        popup.withdraw()
        popup.protocol("WM_DELETE_WINDOW", popup.withdraw)
        self.settings_popup = popup

        # ── Titelzeile ────────────────────────────────────────────────────
        header = ctk.CTkFrame(popup, fg_color="transparent")
        header.pack(fill="x", padx=20, pady=(18, 4))
        ctk.CTkLabel(
            header, text="⚙  Einstellungen",
            font=(FONT_FAMILY, 22, "bold"),
        ).pack(anchor="w")
        ctk.CTkLabel(
            header,
            text="Konfiguriere Voice Studio nach deinen Wünschen.",
            text_color="gray60",
            font=(FONT_FAMILY, 12),
        ).pack(anchor="w", pady=(2, 0))

        # ── Tab-Leiste ────────────────────────────────────────────────────
        tabs = ctk.CTkTabview(popup, corner_radius=10, border_width=0)
        tabs.pack(fill="both", expand=True, padx=12, pady=(8, 12))

        for tab_name in ("Workflow", "STT", "Modell", "Persona", "Audio", "TTS", "Avatar"):
            tabs.add(tab_name)

        # Helper: section header + separator line
        def section(parent: ctk.CTkScrollableFrame, title: str) -> None:
            ctk.CTkLabel(
                parent, text=title.upper(),
                font=(FONT_FAMILY, 10, "bold"),
                text_color="#64748b",
                anchor="w",
            ).pack(fill="x", padx=14, pady=(16, 0))
            ctk.CTkFrame(parent, height=1, fg_color="#334155").pack(
                fill="x", padx=14, pady=(3, 8)
            )

        # Helper: label + widget pair
        def lbl(parent: ctk.CTkScrollableFrame, text: str) -> None:
            ctk.CTkLabel(parent, text=text, anchor="w", font=(FONT_FAMILY, 13)).pack(
                fill="x", padx=14, pady=(0, 2)
            )

        # ── Tab: Workflow ─────────────────────────────────────────────────
        wf = ctk.CTkScrollableFrame(tabs.tab("Workflow"), fg_color="transparent")
        wf.pack(fill="both", expand=True)

        section(wf, "Darstellung")
        lbl(wf, "Theme-Modus")
        self.appearance_mode_menu = ctk.CTkOptionMenu(
            wf, values=["Dark", "Light", "System"],
            variable=self.appearance_mode_var,
            command=self.on_appearance_mode_changed,
        )
        self.appearance_mode_menu.pack(fill="x", padx=14, pady=(0, 6))

        section(wf, "Automatisierung")
        self.speak_switch = ctk.CTkSwitch(
            wf, text="Antwort automatisch vorlesen",
            variable=self.auto_speak_var,
        )
        self.speak_switch.pack(anchor="w", padx=14, pady=(0, 8))
        self.auto_pipeline_switch = ctk.CTkSwitch(
            wf, text="Auto-Workflow  (Aufnahme → STT → Ollama)",
            variable=self.auto_pipeline_var,
        )
        self.auto_pipeline_switch.pack(anchor="w", padx=14, pady=(0, 8))
        self.avatar_lipsync_switch = ctk.CTkSwitch(
            wf, text="LipSync (Avatar-Lippenbewegung)",
            variable=self.avatar_lipsync_var,
        )
        self.avatar_lipsync_switch.pack(anchor="w", padx=14, pady=(0, 10))

        section(wf, "Tests & Diagnose")
        self.test_btn = ctk.CTkButton(
            wf, text="  Ollama-Verbindung testen",
            command=self.test_ollama,
            height=38, anchor="w",
        )
        self.test_btn.pack(fill="x", padx=14, pady=(0, 8))
        self.light_test_btn = ctk.CTkButton(
            wf, text="  Licht-Controller testen",
            command=self.open_light_test_popup,
            fg_color="#a16207", hover_color="#854d0e",
            height=38, anchor="w",
        )
        self.light_test_btn.pack(fill="x", padx=14, pady=(0, 10))

        # ── Tab: STT ──────────────────────────────────────────────────────
        stt = ctk.CTkScrollableFrame(tabs.tab("STT"), fg_color="transparent")
        stt.pack(fill="both", expand=True)

        section(stt, "Whisper-Modell")
        lbl(stt, "Modell")
        self.whisper_menu = ctk.CTkOptionMenu(
            stt, values=WHISPER_MODEL_OPTIONS,
            variable=self.whisper_model_var,
            command=self.on_whisper_model_changed,
        )
        self.whisper_menu.pack(fill="x", padx=14, pady=(0, 8))
        lbl(stt, "Erkennungssprache")
        self.whisper_language_menu = ctk.CTkOptionMenu(
            stt, values=list(WHISPER_LANGUAGE_OPTIONS.keys()),
            variable=self.whisper_language_var,
        )
        self.whisper_language_menu.pack(fill="x", padx=14, pady=(0, 8))
        lbl(stt, "Geschwindigkeitsmodus")
        self.whisper_speed_menu = ctk.CTkOptionMenu(
            stt, values=WHISPER_SPEED_OPTIONS,
            variable=self.whisper_speed_var,
        )
        self.whisper_speed_menu.pack(fill="x", padx=14, pady=(0, 10))

        section(stt, "Ladefortschritt")
        self.stt_progress_bar = ctk.CTkProgressBar(stt, height=14)
        self.stt_progress_bar.pack(fill="x", padx=14, pady=(0, 6))
        self.stt_progress_bar.set(0)
        self.stt_progress_label = ctk.CTkLabel(
            stt, textvariable=self.stt_progress_var,
            text_color="gray60", anchor="w",
        )
        self.stt_progress_label.pack(fill="x", padx=14, pady=(0, 10))

        # ── Tab: Modell ───────────────────────────────────────────────────
        mdl = ctk.CTkScrollableFrame(tabs.tab("Modell"), fg_color="transparent")
        mdl.pack(fill="both", expand=True)

        section(mdl, "LLM Provider")
        lbl(mdl, "Provider")
        self.llm_provider_menu = ctk.CTkOptionMenu(
            mdl,
            values=LLM_PROVIDER_OPTIONS,
            variable=self.llm_provider_var,
            command=self.on_llm_provider_changed,
        )
        self.llm_provider_menu.pack(fill="x", padx=14, pady=(0, 8))

        self.provider_frames: dict[str, ctk.CTkFrame] = {}

        ollama_frame = ctk.CTkFrame(mdl, fg_color="transparent")
        self.provider_frames["Ollama"] = ollama_frame
        lbl(ollama_frame, "Modell")
        self.ollama_model_menu = ctk.CTkOptionMenu(
            ollama_frame,
            values=OLLAMA_MODEL_OPTIONS,
            variable=self.ollama_model_var,
            command=self.on_active_provider_model_changed,
        )
        self.ollama_model_menu.pack(fill="x", padx=0, pady=(0, 6))
        self.refresh_ollama_btn = ctk.CTkButton(
            ollama_frame,
            text="Modelle vom Provider laden",
            command=self.refresh_ollama_models,
            height=36,
        )
        self.refresh_ollama_btn.pack(fill="x", padx=0, pady=(0, 8))
        lbl(ollama_frame, "API-URL")
        self.ollama_url_entry = ctk.CTkEntry(ollama_frame, textvariable=self.ollama_url_var, height=36)
        self.ollama_url_entry.pack(fill="x", padx=0, pady=(0, 10))

        openai_frame = ctk.CTkFrame(mdl, fg_color="transparent")
        self.provider_frames["OpenAI"] = openai_frame
        lbl(openai_frame, "Modell")
        self.openai_model_menu = ctk.CTkOptionMenu(
            openai_frame,
            values=[self.openai_model_var.get().strip() or "gpt-4o-mini"],
            variable=self.openai_model_var,
            command=self.on_active_provider_model_changed,
        )
        self.openai_model_menu.pack(fill="x", padx=0, pady=(0, 6))
        self.refresh_openai_btn = ctk.CTkButton(
            openai_frame,
            text="Modelle vom Provider laden",
            command=self.refresh_ollama_models,
            height=36,
        )
        self.refresh_openai_btn.pack(fill="x", padx=0, pady=(0, 8))
        lbl(openai_frame, "Base URL")
        self.openai_base_url_entry = ctk.CTkEntry(openai_frame, textvariable=self.openai_base_url_var, height=36)
        self.openai_base_url_entry.pack(fill="x", padx=0, pady=(0, 6))
        lbl(openai_frame, "API Key")
        self.openai_api_key_entry = ctk.CTkEntry(openai_frame, textvariable=self.openai_api_key_var, show="*", height=36)
        self.openai_api_key_entry.pack(fill="x", padx=0, pady=(0, 10))

        groq_frame = ctk.CTkFrame(mdl, fg_color="transparent")
        self.provider_frames["Groq"] = groq_frame
        lbl(groq_frame, "Modell")
        self.groq_model_menu = ctk.CTkOptionMenu(
            groq_frame,
            values=[self.groq_model_var.get().strip() or "llama-3.3-70b-versatile"],
            variable=self.groq_model_var,
            command=self.on_active_provider_model_changed,
        )
        self.groq_model_menu.pack(fill="x", padx=0, pady=(0, 6))
        self.refresh_groq_btn = ctk.CTkButton(
            groq_frame,
            text="Modelle vom Provider laden",
            command=self.refresh_ollama_models,
            height=36,
        )
        self.refresh_groq_btn.pack(fill="x", padx=0, pady=(0, 8))
        lbl(groq_frame, "Base URL")
        self.groq_base_url_entry = ctk.CTkEntry(groq_frame, textvariable=self.groq_base_url_var, height=36)
        self.groq_base_url_entry.pack(fill="x", padx=0, pady=(0, 6))
        lbl(groq_frame, "API Key")
        self.groq_api_key_entry = ctk.CTkEntry(groq_frame, textvariable=self.groq_api_key_var, show="*", height=36)
        self.groq_api_key_entry.pack(fill="x", padx=0, pady=(0, 10))

        azure_frame = ctk.CTkFrame(mdl, fg_color="transparent")
        self.provider_frames["Azure OpenAI"] = azure_frame
        lbl(azure_frame, "Deployment")
        self.azure_openai_model_menu = ctk.CTkOptionMenu(
            azure_frame,
            values=[self.azure_openai_deployment_var.get().strip() or "my-deployment"],
            variable=self.azure_openai_deployment_var,
            command=self.on_active_provider_model_changed,
        )
        self.azure_openai_model_menu.pack(fill="x", padx=0, pady=(0, 6))
        self.refresh_azure_btn = ctk.CTkButton(
            azure_frame,
            text="Modelle vom Provider laden",
            command=self.refresh_ollama_models,
            height=36,
        )
        self.refresh_azure_btn.pack(fill="x", padx=0, pady=(0, 8))
        lbl(azure_frame, "Endpoint")
        self.azure_openai_endpoint_entry = ctk.CTkEntry(
            azure_frame,
            textvariable=self.azure_openai_endpoint_var,
            height=36,
        )
        self.azure_openai_endpoint_entry.pack(fill="x", padx=0, pady=(0, 6))
        lbl(azure_frame, "API Version")
        self.azure_openai_api_version_entry = ctk.CTkEntry(
            azure_frame,
            textvariable=self.azure_openai_api_version_var,
            height=36,
        )
        self.azure_openai_api_version_entry.pack(fill="x", padx=0, pady=(0, 6))
        lbl(azure_frame, "API Key")
        self.azure_openai_api_key_entry = ctk.CTkEntry(
            azure_frame,
            textvariable=self.azure_openai_api_key_var,
            show="*",
            height=36,
        )
        self.azure_openai_api_key_entry.pack(fill="x", padx=0, pady=(0, 10))

        anthropic_frame = ctk.CTkFrame(mdl, fg_color="transparent")
        self.provider_frames["Anthropic"] = anthropic_frame
        lbl(anthropic_frame, "Modell")
        self.anthropic_model_menu = ctk.CTkOptionMenu(
            anthropic_frame,
            values=[self.anthropic_model_var.get().strip() or "claude-3-5-sonnet-latest"],
            variable=self.anthropic_model_var,
            command=self.on_active_provider_model_changed,
        )
        self.anthropic_model_menu.pack(fill="x", padx=0, pady=(0, 6))
        self.refresh_anthropic_btn = ctk.CTkButton(
            anthropic_frame,
            text="Modelle vom Provider laden",
            command=self.refresh_ollama_models,
            height=36,
        )
        self.refresh_anthropic_btn.pack(fill="x", padx=0, pady=(0, 8))
        lbl(anthropic_frame, "Base URL")
        self.anthropic_base_url_entry = ctk.CTkEntry(
            anthropic_frame,
            textvariable=self.anthropic_base_url_var,
            height=36,
        )
        self.anthropic_base_url_entry.pack(fill="x", padx=0, pady=(0, 6))
        lbl(anthropic_frame, "API Version")
        self.anthropic_api_version_entry = ctk.CTkEntry(
            anthropic_frame,
            textvariable=self.anthropic_api_version_var,
            height=36,
        )
        self.anthropic_api_version_entry.pack(fill="x", padx=0, pady=(0, 6))
        lbl(anthropic_frame, "API Key")
        self.anthropic_api_key_entry = ctk.CTkEntry(
            anthropic_frame,
            textvariable=self.anthropic_api_key_var,
            show="*",
            height=36,
        )
        self.anthropic_api_key_entry.pack(fill="x", padx=0, pady=(0, 10))

        self.on_llm_provider_changed(self.llm_provider_var.get())

        section(mdl, "Diagnostik")
        self.provider_diagnostics_btn = ctk.CTkButton(
            mdl,
            text="Provider-Diagnostik",
            command=self.run_provider_diagnostics_async,
            height=36,
        )
        self.provider_diagnostics_btn.pack(fill="x", padx=14, pady=(0, 6))
        transfer_row = ctk.CTkFrame(mdl, fg_color="transparent")
        transfer_row.pack(fill="x", padx=14, pady=(0, 6))
        transfer_row.grid_columnconfigure((0, 1), weight=1)
        self.export_provider_cfg_btn = ctk.CTkButton(
            transfer_row,
            text="Provider-Config export",
            command=self.export_provider_config,
            height=34,
        )
        self.export_provider_cfg_btn.grid(row=0, column=0, padx=(0, 4), sticky="ew")
        self.import_provider_cfg_btn = ctk.CTkButton(
            transfer_row,
            text="Provider-Config import",
            command=self.import_provider_config,
            height=34,
        )
        self.import_provider_cfg_btn.grid(row=0, column=1, padx=(4, 0), sticky="ew")
        self.store_api_keys_switch = ctk.CTkSwitch(
            mdl,
            text="API-Keys im Profil speichern",
            variable=self.store_api_keys_var,
        )
        self.store_api_keys_switch.pack(anchor="w", padx=14, pady=(0, 6))
        ctk.CTkLabel(
            mdl,
            textvariable=self.provider_diagnostics_var,
            text_color="gray60",
            wraplength=560,
            justify="left",
            anchor="w",
        ).pack(fill="x", padx=14, pady=(0, 10))

        section(mdl, "Generierungs-Parameter")
        self.concise_reply_switch = ctk.CTkSwitch(
            mdl, text="Kurze Voice-Antworten bevorzugen",
            variable=self.concise_reply_var,
        )
        self.concise_reply_switch.pack(anchor="w", padx=14, pady=(0, 10))
        lbl(mdl, "Max Tokens  (num_predict)")
        self.reply_max_tokens_entry = ctk.CTkEntry(mdl, textvariable=self.reply_max_tokens_var, height=36)
        self.reply_max_tokens_entry.pack(fill="x", padx=14, pady=(0, 8))
        lbl(mdl, "Temperatur")
        self.reply_temperature_entry = ctk.CTkEntry(mdl, textvariable=self.reply_temperature_var, height=36)
        self.reply_temperature_entry.pack(fill="x", padx=14, pady=(0, 10))

        # ── Tab: Persona ──────────────────────────────────────────────────
        per = ctk.CTkScrollableFrame(tabs.tab("Persona"), fg_color="transparent")
        per.pack(fill="both", expand=True)

        section(per, "Persönlichkeit der KI")
        ctk.CTkLabel(
            per,
            text="Schieberegler bestimmen den Charakter der Assistenz-Antworten.",
            text_color="gray60", wraplength=560, justify="left", anchor="w",
            font=(FONT_FAMILY, 12),
        ).pack(fill="x", padx=14, pady=(0, 10))

        def add_persona_slider(label: str, variable: ctk.DoubleVar, label_var: ctk.StringVar) -> None:
            row = ctk.CTkFrame(per, fg_color="transparent")
            row.pack(fill="x", padx=14, pady=(6, 0))
            row.grid_columnconfigure(0, weight=1)
            ctk.CTkLabel(row, text=label, anchor="w", font=(FONT_FAMILY, 13)).grid(
                row=0, column=0, sticky="w"
            )
            ctk.CTkLabel(row, textvariable=label_var, width=42, anchor="e", font=(FONT_FAMILY, 13, "bold")).grid(
                row=0, column=1, sticky="e"
            )
            ctk.CTkSlider(
                per, from_=0, to=100, number_of_steps=100,
                variable=variable,
                command=self._on_persona_slider_changed,
            ).pack(fill="x", padx=14, pady=(4, 2))

        add_persona_slider("Flirty", self.persona_flirty_var, self.persona_flirty_label_var)
        add_persona_slider("Humor / Sarkasmus", self.persona_humor_var, self.persona_humor_label_var)
        add_persona_slider("Ernsthaftigkeit", self.persona_serious_var, self.persona_serious_label_var)
        add_persona_slider("Dominanz", self.persona_dominance_var, self.persona_dominance_label_var)
        add_persona_slider("Empathie / Wärme", self.persona_empathy_var, self.persona_empathy_label_var)
        add_persona_slider("Temperament", self.persona_temperament_var, self.persona_temperament_label_var)

        self.save_profile_btn = ctk.CTkButton(
            per, text="  Profil speichern",
            command=self.save_profile,
            fg_color="#1d4ed8", hover_color="#1e40af",
            height=40, anchor="w",
        )
        self.save_profile_btn.pack(fill="x", padx=14, pady=(16, 10))
        self._refresh_persona_labels()

        # ── Tab: Audio ────────────────────────────────────────────────────
        aud = ctk.CTkScrollableFrame(tabs.tab("Audio"), fg_color="transparent")
        aud.pack(fill="both", expand=True)

        section(aud, "Mikrofon")
        lbl(aud, "Gerät")
        self.mic_menu = ctk.CTkOptionMenu(
            aud, values=[NO_MIC_DEVICES_LABEL],
            variable=self.mic_device_var,
            command=self.on_mic_selection_changed,
        )
        self.mic_menu.pack(fill="x", padx=14, pady=(0, 6))
        self.refresh_mic_btn = ctk.CTkButton(
            aud, text="Geräteliste aktualisieren",
            command=self.refresh_input_devices, height=36,
        )
        self.refresh_mic_btn.pack(fill="x", padx=14, pady=(0, 8))
        lbl(aud, "Sample Rate (Hz)")
        self.sample_rate_entry = ctk.CTkEntry(aud, textvariable=self.sample_rate_var, height=36)
        self.sample_rate_entry.pack(fill="x", padx=14, pady=(0, 8))
        self.mic_level_bar = ctk.CTkProgressBar(aud, height=14)
        self.mic_level_bar.pack(fill="x", padx=14, pady=(2, 4))
        self.mic_level_bar.set(0)
        self.mic_level_label = ctk.CTkLabel(
            aud, textvariable=self.mic_level_text_var,
            text_color="gray60", anchor="w",
        )
        self.mic_level_label.pack(fill="x", padx=14, pady=(0, 6))

        section(aud, "VAD – Automatischer Stopp")
        self.vad_switch = ctk.CTkSwitch(
            aud, text="Auto-Stop aktivieren  (energiebasiert)",
            variable=self.vad_enabled_var,
        )
        self.vad_switch.pack(anchor="w", padx=14, pady=(0, 10))
        vad_row = ctk.CTkFrame(aud, fg_color="transparent")
        vad_row.pack(fill="x", padx=14, pady=(0, 4))
        vad_row.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(vad_row, text="Aggressivität  (0–3)", font=(FONT_FAMILY, 13)).grid(
            row=0, column=0, sticky="w", padx=(0, 12)
        )
        self.vad_aggressiveness_menu = ctk.CTkOptionMenu(
            vad_row, values=["0", "1", "2", "3"],
            variable=self.vad_aggressiveness_var, width=90,
        )
        self.vad_aggressiveness_menu.grid(row=0, column=1, sticky="w")
        lbl(aud, "Stille-Timeout (Sekunden)")
        self.vad_silence_entry = ctk.CTkEntry(aud, textvariable=self.vad_silence_timeout_var, height=36)
        self.vad_silence_entry.pack(fill="x", padx=14, pady=(0, 10))

        section(aud, "Wake-Word")
        ctk.CTkLabel(
            aud,
            text="Sprich das Aktivierungswort – das Mikrofon startet automatisch.",
            text_color="gray60", wraplength=560, justify="left", anchor="w",
            font=(FONT_FAMILY, 12),
        ).pack(fill="x", padx=14, pady=(0, 8))
        self.ww_switch = ctk.CTkSwitch(
            aud, text="Wake-Word aktivieren",
            variable=self.wake_word_enabled_var,
            command=self.on_wake_word_toggle,
        )
        self.ww_switch.pack(anchor="w", padx=14, pady=(0, 8))
        lbl(aud, "Aktivierungswort")
        ww_model_row = ctk.CTkFrame(aud, fg_color="transparent")
        ww_model_row.pack(fill="x", padx=14, pady=(0, 6))
        ww_model_row.grid_columnconfigure(0, weight=1)
        from wake_word_mixin import OWW_MODEL_DISPLAY_NAMES
        self.ww_model_menu = ctk.CTkOptionMenu(
            ww_model_row, values=OWW_MODEL_DISPLAY_NAMES,
            variable=self.wake_word_model_var,
        )
        self.ww_model_menu.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        ctk.CTkButton(
            ww_model_row, text="Neu starten", width=110,
            command=self.start_wake_word_listener,
        ).grid(row=0, column=1)
        lbl(aud, "Status")
        ctk.CTkLabel(
            aud, textvariable=self.wake_word_status_var,
            text_color="#22d3ee", anchor="w",
        ).pack(fill="x", padx=14, pady=(0, 10))

        section(aud, "Realtime")
        lbl(aud, "Latenz-/Stabilitäts-Profil")
        self.realtime_mode_menu = ctk.CTkOptionMenu(
            aud,
            values=REALTIME_MODE_OPTIONS,
            variable=self.realtime_mode_var,
            command=self.on_realtime_mode_changed,
        )
        self.realtime_mode_menu.pack(fill="x", padx=14, pady=(0, 6))
        ctk.CTkLabel(
            aud,
            text="Aggressiv = schnellste Reaktion, Stabil = weniger Fehltrigger.",
            text_color="gray60", wraplength=560, justify="left", anchor="w",
            font=(FONT_FAMILY, 12),
        ).pack(fill="x", padx=14, pady=(0, 10))

        # ── Tab: TTS ──────────────────────────────────────────────────────
        tts = ctk.CTkScrollableFrame(tabs.tab("TTS"), fg_color="transparent")
        tts.pack(fill="both", expand=True)

        section(tts, "Engine & Stimme")
        lbl(tts, "TTS-Engine")
        tts_engines = ["edge-tts (natürlich)", "piper (lokal, natürlich)", "pyttsx3 (lokal)"]
        self.tts_engine_menu = ctk.CTkOptionMenu(
            tts, values=tts_engines,
            variable=self.tts_engine_var,
            command=self.on_tts_engine_changed,
        )
        self.tts_engine_menu.pack(fill="x", padx=14, pady=(0, 8))
        lbl(tts, "Stimme")
        self.tts_voice_menu = ctk.CTkOptionMenu(
            tts, values=list(EDGE_VOICE_OPTIONS.keys()),
            variable=self.tts_voice_var,
            command=self.on_tts_voice_changed,
        )
        self.tts_voice_menu.pack(fill="x", padx=14, pady=(0, 8))
        lbl(tts, "Emotion")
        self.tts_emotion_menu = ctk.CTkOptionMenu(
            tts, values=list(EMOTION_PRESETS.keys()),
            variable=self.tts_emotion_var,
        )
        self.tts_emotion_menu.pack(fill="x", padx=14, pady=(0, 8))
        lbl(tts, "Sprechgeschwindigkeit (Rate)")
        self.tts_rate_entry = ctk.CTkEntry(tts, textvariable=self.tts_rate_var, height=36)
        self.tts_rate_entry.pack(fill="x", padx=14, pady=(0, 10))

        section(tts, "Piper – Lokale Stimme")
        lbl(tts, "Modell-Pfad  (.onnx)")
        self.piper_model_entry = ctk.CTkEntry(tts, textvariable=self.piper_model_path_var, height=36)
        self.piper_model_entry.pack(fill="x", padx=14, pady=(0, 8))
        lbl(tts, "Config-Pfad  (.json, optional)")
        self.piper_config_entry = ctk.CTkEntry(tts, textvariable=self.piper_config_path_var, height=36)
        self.piper_config_entry.pack(fill="x", padx=14, pady=(0, 6))
        ctk.CTkLabel(
            tts,
            text="Tipp: Lade eine .onnx-Stimmdatei herunter und trage den Pfad ein.",
            text_color="gray60", wraplength=560, justify="left", anchor="w",
            font=(FONT_FAMILY, 12),
        ).pack(fill="x", padx=14, pady=(0, 10))

        # ── Tab: Avatar ───────────────────────────────────────────────────
        av = ctk.CTkScrollableFrame(tabs.tab("Avatar"), fg_color="transparent")
        av.pack(fill="both", expand=True)

        section(av, "VRM-Modell auswählen")
        lbl(av, "Installiertes Modell")
        vrm_names = self._scan_vrm_models()
        self.vrm_model_menu = ctk.CTkOptionMenu(
            av, values=vrm_names if vrm_names else ["(Kein Modell gefunden)"],
            variable=self.vrm_model_var,
        )
        self.vrm_model_menu.pack(fill="x", padx=14, pady=(0, 6))

        # Refresh + Browse row
        btn_row = ctk.CTkFrame(av, fg_color="transparent")
        btn_row.pack(fill="x", padx=14, pady=(0, 8))
        btn_row.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(
            btn_row, text="↻  Ordner neu einlesen",
            command=self._refresh_vrm_model_list,
            height=36,
        ).grid(row=0, column=0, padx=(0, 4), sticky="ew")
        ctk.CTkButton(
            btn_row, text="＋  VRM-Datei importieren",
            command=self._browse_vrm_model,
            height=36,
        ).grid(row=0, column=1, padx=(4, 0), sticky="ew")

        ctk.CTkLabel(
            av,
            text="Importierte Dateien werden in den Modell-Ordner kopiert.",
            text_color="gray60", wraplength=560, justify="left", anchor="w",
            font=(FONT_FAMILY, 12),
        ).pack(fill="x", padx=14, pady=(0, 10))

        section(av, "Modell anwenden")
        ctk.CTkButton(
            av, text="Viewer neu starten mit gewähltem Modell",
            command=self._apply_vrm_model,
            height=40, fg_color="#1d4ed8", hover_color="#1e40af",
        ).pack(fill="x", padx=14, pady=(0, 6))
        ctk.CTkLabel(
            av,
            text="Der Viewer wird gestoppt und sofort mit dem neuen Modell neu gestartet.",
            text_color="gray60", wraplength=560, justify="left", anchor="w",
            font=(FONT_FAMILY, 12),
        ).pack(fill="x", padx=14, pady=(0, 10))

    def open_settings_popup(self) -> None:
        if self.settings_popup is None:
            return
        self.settings_popup.deiconify()
        self.settings_popup.lift()
        self.settings_popup.focus()

    def _toggle_column(self, column_name: str) -> None:
        frame = self.column_frames.get(column_name)
        if frame is None:
            return
        visible = self.column_visible.get(column_name, True)
        if visible and sum(1 for value in self.column_visible.values() if value) == 1:
            return
        self.column_visible[column_name] = not visible
        self._refresh_column_layout()

    def _refresh_column_layout(self) -> None:
        body = self.body_frame
        if body is None:
            return

        order = ["left", "middle", "right"]
        visible_columns = [name for name in order if self.column_visible.get(name, True)]
        if not visible_columns:
            self.column_visible["left"] = True
            visible_columns = ["left"]

        for slot in range(3):
            body.grid_columnconfigure(slot, weight=0)

        for name, frame in self.column_frames.items():
            if name not in visible_columns:
                frame.grid_remove()

        for slot, name in enumerate(visible_columns):
            frame = self.column_frames[name]
            left_pad = 0 if slot == 0 else 6
            right_pad = 0 if slot == (len(visible_columns) - 1) else 6
            frame.grid(row=0, column=slot, padx=(left_pad, right_pad), pady=0, sticky="nsew")
            body.grid_columnconfigure(slot, weight=self.column_weights.get(name, 1))

    def set_status(self, message: str) -> None:
        self.logger.info("Status: %s", message)
        self._track_event(f"Status: {message}")
        self.after(0, lambda: self.status_var.set(message))

    # ── Pipeline phase indicator ──────────────────────────────────────────

    # Phases:  idle | mic | stt | ollama | tts
    _PIPELINE_STEPS: list[tuple[str, str]] = [
        ("mic",    "Mikrofon"),
        ("stt",    "STT"),
        ("ollama", "Ollama"),
        ("tts",    "Sprechen"),
    ]
    _PHASE_COLORS: dict[str, str] = {
        "idle":   "#334155",
        "mic":    "#22d3ee",
        "stt":    "#a78bfa",
        "ollama": "#f59e0b",
        "tts":    "#4ade80",
    }
    # Status-Label-Farben je Phase (besser lesbar als Canvas-Farben)
    _PHASE_DISPLAY_COLORS: dict[str, str] = {
        "idle":   "#94a3b8",
        "mic":    "#22d3ee",
        "stt":    "#a78bfa",
        "ollama": "#f59e0b",
        "tts":    "#4ade80",
    }

    def set_pipeline_phase(self, phase: str) -> None:
        """Switch the pipeline indicator to a new phase (thread-safe)."""
        if phase == self.pipeline_phase:
            return
        self.pipeline_phase = phase
        # Notify avatar viewer for animation switching
        self.avatar_bridge.post_phase(phase)
        # Cancel running animation
        if self._pipeline_anim_job is not None:
            try:
                self.after_cancel(self._pipeline_anim_job)
            except Exception:
                pass
            self._pipeline_anim_job = None
        self._pipeline_anim_tick = 0
        self.after(0, self._draw_pipeline)
        # Status-Label-Farbe an aktive Phase anpassen
        color = self._PHASE_DISPLAY_COLORS.get(phase, "#94a3b8")
        if hasattr(self, "status_label"):
            self.after(0, lambda c=color: self.status_label.configure(text_color=c))
        if phase != "idle":
            self._pipeline_schedule_anim()

    def _pipeline_schedule_anim(self) -> None:
        self._pipeline_anim_job = self.after(500, self._pipeline_anim_step)

    def _pipeline_anim_step(self) -> None:
        if self.pipeline_phase == "idle":
            return
        self._pipeline_anim_tick += 1
        self._draw_pipeline()
        self._pipeline_schedule_anim()

    def _draw_pipeline(self) -> None:
        cv = getattr(self, "pipeline_canvas", None)
        if cv is None:
            return
        try:
            cv.delete("all")
        except Exception:
            return

        # Theme-aware palette
        is_light = ctk.get_appearance_mode().lower() == "light"
        if is_light:
            canvas_bg    = "#e8edf4"
            inactive_col = "#b0bdd0"
            done_col     = "#bfcfe4"
            done_fill    = "#dce8f5"
            done_outline = "#0284c7"
            done_check   = "#0284c7"
            text_done    = "#64748b"
            text_pending = "#94a3b8"
        else:
            canvas_bg    = "#1a2236"
            inactive_col = "#334155"
            done_col     = "#1e3a4a"
            done_fill    = "#1e3a4a"
            done_outline = "#22d3ee"
            done_check   = "#22d3ee"
            text_done    = "#64748b"
            text_pending = "#475569"

        try:
            cv.configure(bg=canvas_bg)
        except Exception:
            pass

        active = self.pipeline_phase
        steps = self._PIPELINE_STEPS
        n = len(steps)                # 4
        w = 340
        h = 36
        dot_r = 7
        text_y = h - 7
        # x positions for dots: evenly spaced
        xs = [int(w * (i + 0.5) / n) for i in range(n)]
        active_color = self._PHASE_COLORS.get(active, "#22d3ee")

        # Determine which step index is active (-1 = idle)
        active_idx = next(
            (i for i, (k, _) in enumerate(steps) if k == active), -1
        )

        # Draw connector lines between dots
        for i in range(n - 1):
            x1, x2 = xs[i], xs[i + 1]
            y = h // 2 - 4
            col = done_col if i < active_idx else inactive_col
            cv.create_line(x1 + dot_r, y, x2 - dot_r, y, fill=col, width=2)

        for i, (key, label) in enumerate(steps):
            x, y = xs[i], h // 2 - 4
            if i < active_idx:
                # completed
                cv.create_oval(x - dot_r, y - dot_r, x + dot_r, y + dot_r,
                               fill=done_fill, outline=done_outline, width=1)
                cv.create_text(x, y, text="✓", fill=done_check,
                               font=("Segoe UI", 8, "bold"))
                cv.create_text(x, text_y, text=label, fill=text_done,
                               font=("Segoe UI", 8))
            elif i == active_idx:
                # active — pulsing glow
                pulse = self._pipeline_anim_tick % 2 == 0
                glow_r = dot_r + (3 if pulse else 1)
                cv.create_oval(x - glow_r, y - glow_r, x + glow_r, y + glow_r,
                               fill="", outline=active_color, width=1)
                cv.create_oval(x - dot_r, y - dot_r, x + dot_r, y + dot_r,
                               fill=active_color, outline="")
                cv.create_text(x, text_y, text=label, fill=active_color,
                               font=("Segoe UI", 8, "bold"))
            else:
                # pending
                cv.create_oval(x - dot_r, y - dot_r, x + dot_r, y + dot_r,
                               fill=inactive_col, outline="")
                cv.create_text(x, text_y, text=label, fill=text_pending,
                               font=("Segoe UI", 8))

    def set_textbox(self, textbox: ctk.CTkTextbox, content: str) -> None:
        def updater() -> None:
            textbox.delete("1.0", "end")
            textbox.insert("1.0", content)
            textbox.see("1.0")

        self.after(0, updater)

    def append_textbox(self, textbox: ctk.CTkTextbox, content: str) -> None:
        def updater() -> None:
            # Remove cursor before inserting new content so it stays at the end
            self._remove_cursor(textbox)
            textbox.insert("end", content)
            textbox.see("end")
        self.after(0, updater)

    # ── Streaming cursor ──────────────────────────────────────────────

    def _start_cursor(self, textbox: ctk.CTkTextbox) -> None:
        """Start blinking ▌ cursor at end of textbox."""
        self._cursor_active = True
        self._cursor_visible = False
        self._cancel_cursor_job()
        # configure the tag once (color matches the ollama phase color)
        try:
            textbox._textbox.tag_configure("stream_cursor", foreground="#f59e0b")
        except Exception:
            pass
        self._blink_cursor(textbox)

    def _stop_cursor(self, textbox: ctk.CTkTextbox) -> None:
        """Stop blinking and remove cursor character."""
        self._cursor_active = False
        self._cancel_cursor_job()
        self.after(0, lambda: self._remove_cursor(textbox))

    def _cancel_cursor_job(self) -> None:
        if self._cursor_job is not None:
            try:
                self.after_cancel(self._cursor_job)
            except Exception:
                pass
            self._cursor_job = None

    def _remove_cursor(self, textbox: ctk.CTkTextbox) -> None:
        try:
            widget = textbox._textbox
            # find and delete all tagged cursor characters
            ranges = widget.tag_ranges("stream_cursor")
            for i in range(len(ranges) - 1, -1, -2):
                widget.delete(str(ranges[i - 1]), str(ranges[i]))
        except Exception:
            pass

    def _blink_cursor(self, textbox: ctk.CTkTextbox) -> None:
        if not self._cursor_active:
            return
        try:
            widget = textbox._textbox
            self._remove_cursor(textbox)
            if self._cursor_visible:
                # insert cursor glyph with tag at end
                widget.insert("end", "▌", "stream_cursor")
                widget.see("end")
            self._cursor_visible = not self._cursor_visible
        except Exception:
            pass
        self._cursor_job = self.after(530, lambda: self._blink_cursor(textbox))

    # ── Ollama "thinking" indicator ──────────────────────────────────────────

    def _start_thinking(self) -> None:
        """Show animated dots below the Antwort label while waiting for first token."""
        cv = getattr(self, "thinking_canvas", None)
        if cv is None:
            return
        self._thinking_active = True
        self._thinking_tick = 0
        try:
            cv.grid()          # make visible
        except Exception:
            pass
        self._thinking_step()

    def _stop_thinking(self) -> None:
        self._thinking_active = False
        if self._thinking_job is not None:
            try:
                self.after_cancel(self._thinking_job)
            except Exception:
                pass
            self._thinking_job = None
        cv = getattr(self, "thinking_canvas", None)
        if cv is not None:
            try:
                cv.grid_remove()   # hide
                cv.delete("all")
            except Exception:
                pass

    def _thinking_step(self) -> None:
        if not self._thinking_active:
            return
        self._thinking_tick += 1
        self._draw_thinking()
        self._thinking_job = self.after(120, self._thinking_step)

    def _draw_thinking(self) -> None:
        import math
        cv = getattr(self, "thinking_canvas", None)
        if cv is None:
            return
        try:
            cv.update_idletasks()
            w = cv.winfo_width()
            h = cv.winfo_height()
        except Exception:
            return
        if w < 4 or h < 4:
            return
        cv.delete("all")

        # Three dots that light up in a travelling wave
        n_dots = 3
        dot_r = 4
        spacing = 18
        total = (n_dots - 1) * spacing
        start_x = w // 2 - total // 2
        cy = h // 2
        t = self._thinking_tick

        # label
        cv.create_text(
            start_x - 28, cy,
            text="denkt",
            fill="#475569",
            font=("Segoe UI", 8),
            anchor="e",
        )

        for i in range(n_dots):
            # wave phase offset per dot
            phase = (t - i * 3) % 18
            brightness = max(0.0, math.sin(phase / 18 * math.pi))
            r = int(0x0e + (0xf5 - 0x0e) * brightness)
            g = int(0x74 + (0x9e - 0x74) * brightness)
            b = int(0x90 + (0x0b - 0x90) * brightness)
            color = f"#{r:02x}{g:02x}{b:02x}"
            x = start_x + i * spacing
            cv.create_oval(
                x - dot_r, cy - dot_r, x + dot_r, cy + dot_r,
                fill=color, outline="",
            )

    def _extract_complete_sentences(self, text_buffer: str) -> tuple[list[str], str]:
        completed: list[str] = []
        last_split_index = 0
        sentence_end_chars = ".!?\n"

        for index, char in enumerate(text_buffer):
            if char in sentence_end_chars:
                sentence = text_buffer[last_split_index : index + 1].strip()
                if sentence:
                    completed.append(sentence)
                last_split_index = index + 1

        remaining = text_buffer[last_split_index:]
        return completed, remaining

    def _extract_complete_phrases(self, text_buffer: str) -> tuple[list[str], str]:
        completed: list[str] = []
        last_split_index = 0
        phrase_end_chars = ",;:\n"

        for index, char in enumerate(text_buffer):
            if char in phrase_end_chars:
                phrase = text_buffer[last_split_index : index + 1].strip()
                if phrase:
                    completed.append(phrase)
                last_split_index = index + 1

        remaining = text_buffer[last_split_index:]
        return completed, remaining

    def _start_streaming_tts_worker(self) -> tuple[queue.Queue[str | None], threading.Thread]:
        tts_queue: queue.Queue[str | None] = queue.Queue()
        self.current_tts_queue = tts_queue

        def worker() -> None:
            while True:
                item = tts_queue.get()
                if item is None:
                    tts_queue.task_done()
                    break
                try:
                    if self.cancel_tts_event.is_set():
                        continue
                    if not self._first_audio_chunk_recorded and self._active_response_started_at is not None:
                        self._first_audio_chunk_recorded = True
                        self._add_metric_sample(
                            "first_audio_seconds",
                            max(0.0, time.perf_counter() - self._active_response_started_at),
                        )
                    started = time.perf_counter()
                    self.speak_text(item)
                    elapsed = time.perf_counter() - started
                    self._increment_counter("tts_chunks")
                    self._add_metric_sample("tts_chunk_seconds", elapsed)
                except Exception as exc:
                    # Ignore per-chunk TTS failures so the rest of the response can continue.
                    self.logger.warning("TTS-Chunk fehlgeschlagen: %s", exc)
                    self._increment_counter("errors")
                finally:
                    tts_queue.task_done()

            if self.current_tts_queue is tts_queue:
                self.current_tts_queue = None
            self.after(0, lambda: self.set_pipeline_phase("idle"))

        tts_thread = threading.Thread(target=worker, daemon=True)
        tts_thread.start()
        return tts_queue, tts_thread

    def cancel_current_response(self) -> None:
        self.cancel_ollama_event.set()
        self.cancel_tts_event.set()
        self._increment_counter("ollama_cancels")
        self.logger.warning("Aktive Antwort wurde abgebrochen")
        self._track_event("Antwortabbruch angefordert")
        self._stop_cursor(self.answer_box)
        self._stop_thinking()

        queue_ref = self.current_tts_queue
        if queue_ref is not None:
            queue_ref.put(None)

        with self.active_ollama_response_lock:
            active_response = self.active_ollama_response
        if active_response is not None:
            try:
                active_response.close()
            except Exception:
                pass

        self._stop_active_audio_playback()

        self.set_status(DEFAULT_ABORTED_STATUS)
        self.after(0, lambda: self.cancel_btn.configure(state="disabled"))
        self.after(0, lambda: self.send_btn.configure(state="normal"))
        self.after(0, lambda: self.transcribe_btn.configure(state="normal"))
        self.after(0, lambda: self.text_send_btn.configure(state="normal"))

    def _stop_active_audio_playback(self) -> None:
        self._reset_avatar_lipsync()
        try:
            pygame_module = importlib.import_module("pygame")
            if pygame_module.mixer.get_init():
                pygame_module.mixer.music.stop()
                try:
                    pygame_module.mixer.music.unload()
                except Exception:
                    pass
        except Exception:
            pass

        try:
            with self.pyttsx3_engine_lock:
                engine = self.pyttsx3_engine
            if engine is not None:
                engine.stop()
        except Exception:
            pass

    def add_history_entry(self, role: str, content: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logger.debug("History %s: %s", role, content[:140])

        # Keep conversation_history in sync for multi-turn Ollama context.
        ollama_role = "user" if role.lower() in {"du", "user"} else "assistant"
        self.conversation_history.append({"role": ollama_role, "content": content})

        def updater() -> None:
            self.history_box.insert("end", f"[{timestamp}] {role}:\n{content}\n\n")
            self.history_box.see("end")

        self.after(0, updater)

    def clear_history(self) -> None:
        self.conversation_history.clear()
        self.history_box.delete("1.0", "end")
        self._track_event("Verlauf geloescht")

    def _refresh_light_popup_visuals(self) -> None:
        label_widget = self.light_state_label
        indicator_widget = self.light_indicator

        if self.light_state:
            self.light_state_var.set("Lichtstatus: AN")
            if label_widget is not None and label_widget.winfo_exists():
                label_widget.configure(text_color="#ffd166")
            if indicator_widget is not None and indicator_widget.winfo_exists():
                indicator_widget.configure(fg_color="#facc15")
            return

        self.light_state_var.set("Lichtstatus: AUS")
        if label_widget is not None and label_widget.winfo_exists():
            label_widget.configure(text_color="gray70")
        if indicator_widget is not None and indicator_widget.winfo_exists():
            indicator_widget.configure(fg_color="#334155")

    def set_light_state(self, is_on: bool, source: str = "manual") -> None:
        self.light_state = is_on
        self._refresh_light_popup_visuals()
        human_state = "AN" if is_on else "AUS"
        self.logger.info("LightController (%s): %s", source, human_state)
        self._track_event(f"Licht auf {human_state} gesetzt ({source})")

    def open_light_test_popup(self) -> None:
        if self.light_popup is not None and self.light_popup.winfo_exists():
            self.light_popup.focus()
            self.light_popup.lift()
            return

        popup = ctk.CTkToplevel(self)
        popup.title("Light Controller Test")
        popup.geometry("360x240")
        popup.transient(self)
        popup.grab_set()
        popup.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(popup, text="Licht-Teststeuerung", font=(FONT_FAMILY, 18, "bold")).grid(
            row=0, column=0, padx=16, pady=(16, 10), sticky="w"
        )

        self.light_indicator = ctk.CTkFrame(popup, height=70, fg_color="#334155")
        self.light_indicator.grid(row=1, column=0, padx=16, pady=(0, 12), sticky="ew")

        self.light_state_label = ctk.CTkLabel(popup, textvariable=self.light_state_var, font=(FONT_FAMILY, 15, "bold"))
        self.light_state_label.grid(row=2, column=0, padx=16, pady=(0, 12), sticky="w")

        controls = ctk.CTkFrame(popup, fg_color="transparent")
        controls.grid(row=3, column=0, padx=16, pady=(0, 16), sticky="ew")
        controls.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkButton(
            controls,
            text="Licht AN",
            fg_color="#15803d",
            hover_color="#166534",
            command=lambda: self.set_light_state(True, source="popup"),
        ).grid(row=0, column=0, padx=(0, 6), pady=0, sticky="ew")

        ctk.CTkButton(
            controls,
            text="Licht AUS",
            fg_color="#b91c1c",
            hover_color="#991b1b",
            command=lambda: self.set_light_state(False, source="popup"),
        ).grid(row=0, column=1, padx=(6, 0), pady=0, sticky="ew")

        self.light_popup = popup
        self._refresh_light_popup_visuals()
        self._track_event("Light-Test-Popup geoeffnet")

        def on_popup_close() -> None:
            self.light_popup = None
            self.light_state_label = None
            self.light_indicator = None
            popup.destroy()

        popup.protocol("WM_DELETE_WINDOW", on_popup_close)

    def _parse_light_command(self, text: str) -> str | None:
        lowered = text.lower()
        has_light_keyword = "licht" in lowered or "lichter" in lowered or "light" in lowered
        if not has_light_keyword:
            return None

        if " an" in lowered or lowered.endswith("an") or "einschalten" in lowered:
            return "on"
        if " aus" in lowered or lowered.endswith("aus") or "ausschalten" in lowered:
            return "off"
        return None

    def refresh_piper_model_options(self) -> None:
        project_root = Path(__file__).resolve().parent
        candidate_dirs = [
            project_root / "piperVoices",
            project_root / "models",
        ]

        configured_model_input = self.piper_model_path_var.get().strip()
        if configured_model_input:
            configured_candidate = Path(configured_model_input).expanduser()
            if not configured_candidate.is_absolute():
                configured_candidate = (project_root / configured_candidate).resolve()
            if configured_candidate.is_dir():
                candidate_dirs.append(configured_candidate)

        model_paths: list[Path] = []
        seen: set[Path] = set()
        for directory in candidate_dirs:
            if directory.exists() and directory.is_dir():
                for model_path in sorted(directory.rglob(MODEL_GLOB_PATTERN)):
                    resolved = model_path.resolve()
                    if resolved not in seen:
                        seen.add(resolved)
                        model_paths.append(resolved)

        new_map: dict[str, str] = {}
        for model_path in model_paths:
            label = model_path.stem
            if label in new_map:
                label = str(model_path.relative_to(project_root)) if model_path.is_relative_to(project_root) else str(model_path)
            new_map[label] = str(model_path)

        self.piper_models_map = new_map

    def on_tts_voice_changed(self, selected: str) -> None:
        selected_engine = self.tts_engine_var.get().strip().lower()
        if "piper" in selected_engine:
            model_path = self.piper_models_map.get(selected)
            if model_path:
                self.piper_model_path_var.set(model_path)

    def on_tts_engine_changed(self, _selected: str) -> None:
        selected_engine = self.tts_engine_var.get().strip().lower()

        if "piper" in selected_engine:
            self.refresh_piper_model_options()
            piper_labels = list(self.piper_models_map.keys())
            if not piper_labels:
                piper_labels = [NO_PIPER_MODELS_LABEL]

            self.tts_voice_menu.configure(values=piper_labels)
            current_voice = self.tts_voice_var.get().strip()
            if current_voice not in piper_labels:
                self.tts_voice_var.set(piper_labels[0])

            selected_label = self.tts_voice_var.get().strip()
            model_path = self.piper_models_map.get(selected_label)
            if model_path:
                self.piper_model_path_var.set(model_path)

            self.tts_emotion_menu.configure(state="disabled")
            self.piper_model_entry.configure(state="normal")
            self.piper_config_entry.configure(state="normal")
            return

        if "pyttsx3" in selected_engine:
            self.tts_voice_menu.configure(values=PYTTSX3_VOICE_OPTIONS)
            current_voice = self.tts_voice_var.get().strip()
            if current_voice not in PYTTSX3_VOICE_OPTIONS:
                self.tts_voice_var.set(PYTTSX3_VOICE_OPTIONS[0])

            self.tts_emotion_menu.configure(state="disabled")
            self.piper_model_entry.configure(state="disabled")
            self.piper_config_entry.configure(state="disabled")
            return

        edge_labels = list(EDGE_VOICE_OPTIONS.keys())
        self.tts_voice_menu.configure(values=edge_labels)
        current_voice = self.tts_voice_var.get().strip()
        if current_voice not in edge_labels:
            self.tts_voice_var.set(edge_labels[0])

        self.tts_emotion_menu.configure(state="normal")
        self.piper_model_entry.configure(state="disabled")
        self.piper_config_entry.configure(state="disabled")

    def set_mic_level(self, level: float) -> None:
        clamped = max(0.0, min(1.0, level))
        self.waveform_samples.append(clamped)

        def updater() -> None:
            self.mic_level_bar.set(clamped)
            self.mic_level_text_var.set(f"Pegel: {int(clamped * 100)}%")

        self.after(0, updater)

    def _schedule_waveform_draw(self) -> None:
        self._draw_waveform()
        self.waveform_animate_job = self.after(20, self._schedule_waveform_draw)  # 20 fps

    def trigger_waveform_flash(self) -> None:
        """Called when wake-word fires: bright flash for ~15 frames (600 ms)."""
        self._ww_flash_frames = 15

    def _draw_waveform(self) -> None:
        canvas = self.waveform_canvas
        if canvas is None:
            return
        # Kein Zeichnen wenn die mittlere Spalte ausgeblendet ist
        if not self.column_visible.get("middle", True):
            return
        try:
            canvas.update_idletasks()
            w = canvas.winfo_width()
            h = canvas.winfo_height()
        except Exception:
            return
        if w < 4 or h < 4:
            return

        self._ww_pulse_tick += 1

        # Theme-aware palette
        is_light = ctk.get_appearance_mode().lower() == "light"
        if is_light:
            canvas_bg  = "#f0f4f8"
            grid_mid   = "#c5cdd9"
            grid_dash  = "#dce3ec"
            idle_bar   = "#94a3b8"
            idle_glow  = "#e2e8f0"
            rec_bar    = "#0284c7"
            rec_glow   = "#7dd3fc"
            ww_glow    = "#93c5fd"
            flash_glow = "#67e8f9"
        else:
            canvas_bg  = "#0d1117"
            grid_mid   = "#1e2a3a"
            grid_dash  = "#151f2b"
            idle_bar   = "#334155"
            idle_glow  = "#1e2a3a"
            rec_bar    = "#22d3ee"
            rec_glow   = "#0e7490"
            ww_glow    = "#0a4a5a"
            flash_glow = "#67e8f9"

        try:
            canvas.configure(bg=canvas_bg)
        except Exception:
            pass

        # Determine visual mode
        is_recording = self.is_recording
        ww_listening = (
            not is_recording
            and getattr(self, "wake_word_enabled_var", None) is not None
            and self.wake_word_enabled_var.get()
            and getattr(self, "_oww_model", None) is not None
        )
        flash = self._ww_flash_frames > 0
        if flash:
            self._ww_flash_frames -= 1

        # Colour scheme per mode
        import math
        if flash:
            # Bright white burst, fades over the 15 frames
            fade = self._ww_flash_frames / 15          # 1.0 → 0.0
            br, bg_, bb = (0x02, 0x84, 0xc7) if is_light else (0x22, 0xd3, 0xee)
            r = int(br + (0xff - br) * fade)
            g = int(bg_ + (0xff - bg_) * fade)
            b = int(bb + (0xff - bb) * fade)
            bar_color = f"#{r:02x}{g:02x}{b:02x}"
            glow_color = flash_glow
            height_boost = 1.0 + 0.6 * fade
        elif is_recording:
            bar_color = rec_bar
            glow_color = rec_glow
            height_boost = 1.0
        elif ww_listening:
            # Gentle idle pulse: amplitude 0.08, period ~2 s
            pulse = 0.92 + 0.08 * math.sin(self._ww_pulse_tick * 0.08)
            if is_light:
                # Pulse between rec_bar (#0284c7) and a lighter sky-blue (#7dd3fc)
                r = int(0x02 + (0x7d - 0x02) * (1.0 - pulse))
                g = int(0x84 + (0xd3 - 0x84) * (1.0 - pulse))
                b = int(0xc7 + (0xfc - 0xc7) * (1.0 - pulse))
            else:
                r = int(0x22 * pulse)
                g = int(0xd3 * pulse)
                b = int(0xee * pulse)
            bar_color = f"#{r:02x}{g:02x}{b:02x}"
            glow_color = ww_glow
            height_boost = pulse
        else:
            bar_color = idle_bar
            glow_color = idle_glow
            height_boost = 1.0

        canvas.delete("all")

        # Background grid lines
        mid_y = h // 2
        canvas.create_line(0, mid_y, w, mid_y, fill=grid_mid, width=1)
        for frac in (0.25, 0.75):
            y = int(h * frac)
            canvas.create_line(0, y, w, y, fill=grid_dash, width=1, dash=(4, 6))

        # Wake-word listening ring (thin outer circle hint)
        if ww_listening and not is_recording:
            cx, cy = w // 2, mid_y
            ring_r = min(cx, mid_y) - 4
            alpha_pulse = 0.4 + 0.3 * math.sin(self._ww_pulse_tick * 0.08)
            if is_light:
                rr = int(0x02 + (0x7d - 0x02) * (1.0 - alpha_pulse))
                gg = int(0x84 + (0xd3 - 0x84) * (1.0 - alpha_pulse))
                bb = int(0xc7 + (0xfc - 0xc7) * (1.0 - alpha_pulse))
            else:
                rr = int(0x22 * alpha_pulse)
                gg = int(0xd3 * alpha_pulse)
                bb = int(0xee * alpha_pulse)
            ring_col = f"#{rr:02x}{gg:02x}{bb:02x}"
            canvas.create_oval(
                cx - ring_r, cy - ring_r, cx + ring_r, cy + ring_r,
                outline=ring_col, width=1,
            )

        samples = list(self.waveform_samples)
        if not samples:
            return

        n = len(samples)
        step = w / max(n, 1)

        for i, amp in enumerate(samples):
            x = int(i * step)
            bar_h = max(2, int(amp * (h * 0.85) / 2 * height_boost))
            # Glow (wider, dimmer)
            canvas.create_line(x, mid_y - bar_h - 2, x, mid_y + bar_h + 2,
                               fill=glow_color, width=3)
            # Main bar
            canvas.create_line(x, mid_y - bar_h, x, mid_y + bar_h,
                               fill=bar_color, width=1)

    def _monitor_callback(self, indata: Any, frames: int, callback_time: Any, status: Any) -> None:
        # Legacy: not used when unified stream is active
        peak = float(abs(indata).max()) / 32767.0
        self.set_mic_level(peak)

    def stop_level_monitor(self) -> None:
        # Unified stream stays alive – nothing to stop.
        # Legacy monitor_stream fallback kept for safety.
        if self.monitor_stream is not None:
            self.monitor_stream.stop()
            self.monitor_stream.close()
            self.monitor_stream = None

    def start_level_monitor(self) -> None:
        # With the unified background stream, level monitoring is always on.
        # Only start the legacy monitor if the bg stream is somehow not running.
        if self._bg_stream is not None and self._bg_stream.active:
            return
        self._start_bg_stream()

    # ------------------------------------------------------------------ #
    #  Unified permanent audio stream                                      #
    # ------------------------------------------------------------------ #

    def _bg_audio_callback(self, indata: np.ndarray, frames: int, _time: Any, _status: Any) -> None:
        """Single callback for ALL audio work: level, wake-word, recording, VAD."""
        # indata dtype=float32, shape (frames, 1), values –1..1
        peak = float(np.abs(indata).max())
        self.set_mic_level(peak)

        if self.is_recording:
            # Write int16 PCM to the open wave file
            with self._recording_wave_lock:
                if self.recording_wave is not None:
                    pcm = (indata[:, 0] * 32767.0).astype(np.int16)
                    self.recording_wave.writeframes(pcm.tobytes())
            # Energy-based VAD
            self._vad_check_energy(peak)
        else:
            # Feed openwakeword (only when not already recording)
            self._ww_feed_audio(indata[:, 0])

    def _vad_check_energy(self, peak: float) -> None:
        if not self.vad_enabled_var.get():
            return
        elapsed_s = time.perf_counter() - (self.recording_started_at or 0.0)
        if elapsed_s > 45.0:
            self.logger.info("VAD: maximale Aufnahmedauer – stoppe")
            self.after(0, self.stop_recording)
            return
        if elapsed_s < 0.6:
            return

        _SPEECH = 0.015
        _SILENCE = 0.010
        now = time.perf_counter()
        if peak >= _SPEECH:
            self.vad_speech_detected = True
            self.vad_last_speech_time = now

        if not self.vad_speech_detected:
            return
        try:
            silence_timeout = float(self.vad_silence_timeout_var.get().strip())
        except ValueError:
            silence_timeout = 1.5

        if peak < _SILENCE and (now - self.vad_last_speech_time) >= silence_timeout:
            self.logger.info("VAD: Stille – stoppe Aufnahme")
            self.after(0, self.stop_recording)

    def _start_bg_stream(self) -> None:
        self._stop_bg_stream()
        selected_label = self.mic_device_var.get().strip()
        if selected_label not in self.mic_devices_map:
            self.set_mic_level(0)
            return
        input_device = self.get_selected_input_device()
        try:
            sample_rate = int(self.sample_rate_var.get().strip())
        except ValueError:
            sample_rate = 16000
        try:
            self._bg_stream = sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                device=input_device,
                blocksize=1280,   # 80 ms – required by openwakeword
                callback=self._bg_audio_callback,
            )
            self._bg_stream.start()
            self.logger.info("Unified audio stream gestartet (%d Hz, device=%s)", sample_rate, input_device)
        except Exception as exc:
            self._bg_stream = None
            self.set_mic_level(0)
            self.logger.warning("Unified stream Fehler: %s", exc)

    def _stop_bg_stream(self) -> None:
        if self._bg_stream is not None:
            try:
                self._bg_stream.stop()
                self._bg_stream.close()
            except Exception:
                pass
            self._bg_stream = None

    def on_mic_selection_changed(self, _selected: str) -> None:
        self._start_bg_stream()

    def _avatar_button_or_none(self) -> Any | None:
        return self.avatar_btn if hasattr(self, "avatar_btn") else None

    def _is_avatar_viewer_running(self) -> bool:
        return self.avatar_bridge.is_running()

    def _update_avatar_button_state(self) -> None:
        avatar_btn = self._avatar_button_or_none()
        if avatar_btn is not None:
            self.avatar_bridge.update_button_state(avatar_btn)

    def _on_avatar_hwnd_ready(self, hwnd: int) -> None:
        self.after(0, lambda: self._dock_viewer_window(hwnd))

    def _dock_viewer_window(self, hwnd: int) -> None:
        host = self.viewer_host_frame
        if host is None or not host.winfo_exists():
            return

        host.update_idletasks()
        host_hwnd = host.winfo_id()

        user32 = ctypes.windll.user32

        # Measure Chromium title-bar height BEFORE touching any styles.
        # ClientToScreen(0,0) gives the top of the client area in screen coords;
        # GetWindowRect gives the top of the window frame.  The difference is the
        # non-client top (= title bar height when WS_CAPTION is still present).
        _pt = ctypes.wintypes.POINT(0, 0)
        _win_rect = ctypes.wintypes.RECT()
        user32.ClientToScreen(hwnd, ctypes.byref(_pt))
        user32.GetWindowRect(hwnd, ctypes.byref(_win_rect))
        _nc_top = _pt.y - _win_rect.top
        if _nc_top >= 4:
            self._viewer_titlebar_h = _nc_top
        else:
            # Chromium uses DwmExtendFrameIntoClientArea so the OS non-client area
            # is effectively 0; the title bar is drawn in the client area at a height
            # matching the standard caption metrics.
            self._viewer_titlebar_h = (
                user32.GetSystemMetrics(4)   # SM_CYCAPTION
                + user32.GetSystemMetrics(33)  # SM_CYFRAME
                + user32.GetSystemMetrics(92)  # SM_CXPADDEDBORDER
            ) or 33

        GWL_STYLE = -16
        GWL_EXSTYLE = -20
        WS_CHILD = 0x40000000
        WS_VISIBLE = 0x10000000
        WS_CAPTION = 0x00C00000
        WS_THICKFRAME = 0x00040000
        WS_SYSMENU = 0x00080000
        WS_MINIMIZEBOX = 0x00020000
        WS_MAXIMIZEBOX = 0x00010000
        WS_EX_APPWINDOW = 0x00040000
        WS_EX_TOOLWINDOW = 0x00000080

        style = user32.GetWindowLongW(hwnd, GWL_STYLE)
        style = (style | WS_CHILD | WS_VISIBLE) & ~(
            WS_CAPTION | WS_THICKFRAME | WS_SYSMENU | WS_MINIMIZEBOX | WS_MAXIMIZEBOX
        )
        user32.SetWindowLongW(hwnd, GWL_STYLE, style)

        # Remove app-window appearance; use tool-window so it doesn't show in taskbar
        ex_style = user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
        ex_style = (ex_style | WS_EX_TOOLWINDOW) & ~WS_EX_APPWINDOW
        user32.SetWindowLongW(hwnd, GWL_EXSTYLE, ex_style)

        user32.SetParent(hwnd, host_hwnd)

        # Flush the style change so the frame actually disappears
        SWP_NOMOVE = 0x0002
        SWP_NOSIZE = 0x0001
        SWP_NOZORDER = 0x0004
        SWP_FRAMECHANGED = 0x0020
        user32.SetWindowPos(hwnd, 0, 0, 0, 0, 0,
                            SWP_NOMOVE | SWP_NOSIZE | SWP_NOZORDER | SWP_FRAMECHANGED)

        self.embedded_viewer_hwnd = hwnd
        self._resize_docked_viewer()
        # SetParent() kann den maximierten Zustand des Hauptfensters zurücksetzen – wiederherstellen.
        self.after(50, lambda: self.state("zoomed"))

    def _resize_docked_viewer(self) -> None:
        hwnd = self.embedded_viewer_hwnd
        host = self.viewer_host_frame
        if hwnd is None or host is None or not host.winfo_exists():
            return

        user32 = ctypes.windll.user32
        width = max(1, host.winfo_width())
        height = max(1, host.winfo_height())
        # Offset the window upward by the title-bar height so that Chromium's
        # client-drawn title bar is pushed above the host frame's clip boundary
        # and becomes invisible.  The extra height compensates so content fills
        # the full host area.
        titlebar_h = getattr(self, "_viewer_titlebar_h", 33)
        user32.MoveWindow(hwnd, 0, -titlebar_h, width, height + titlebar_h, True)

    def _start_avatar_viewer(self) -> bool:
        return self.avatar_bridge.start(
            self._avatar_button_or_none(),
            on_viewer_hwnd=self._on_avatar_hwnd_ready,
        )

    def _stop_avatar_viewer(self) -> None:
        self.avatar_bridge.stop(self._avatar_button_or_none())
        self.embedded_viewer_hwnd = None

    def toggle_avatar_viewer(self) -> None:
        if self._is_avatar_viewer_running():
            self._stop_avatar_viewer()
            return
        self._start_avatar_viewer()

    def _post_avatar_lipsync(self, active: bool, energy: float = 0.0, force: bool = False) -> None:
        self.avatar_bridge.post_lipsync(active=active, energy=energy, force=force)

    def _reset_avatar_lipsync(self) -> None:
        self.avatar_bridge.reset_lipsync()

    def _estimate_lipsync_energy(self, text: str, elapsed_seconds: float) -> float:
        return self.avatar_bridge.estimate_lipsync_energy(text, elapsed_seconds)

    def _start_avatar_lipsync_background(self, text: str) -> tuple[threading.Event, threading.Thread | None]:
        return self.avatar_bridge.start_lipsync_background(text)

    def _ensure_avatar_for_lipsync(self) -> None:
        self.avatar_bridge.ensure_for_lipsync(
            self._avatar_button_or_none(),
            on_viewer_hwnd=self._on_avatar_hwnd_ready,
        )

    def on_close(self) -> None:
        self.stt_loading_active = False
        self.logger.info("Voice Studio wird beendet")
        try:
            self.save_profile(notify=False)
        except Exception as exc:
            self.logger.warning("Profil konnte beim Beenden nicht gespeichert werden: %s", exc)
        if self.stt_loading_job_id is not None:
            try:
                self.after_cancel(self.stt_loading_job_id)
            except Exception:
                pass
            self.stt_loading_job_id = None

        if self.log_pump_job_id is not None:
            try:
                self.after_cancel(self.log_pump_job_id)
            except Exception:
                pass
            self.log_pump_job_id = None

        if self.stats_refresh_job_id is not None:
            try:
                self.after_cancel(self.stats_refresh_job_id)
            except Exception:
                pass
            self.stats_refresh_job_id = None

        # Splash aufräumen falls noch offen (z.B. App während Ladevorgang geschlossen)
        if self._splash is not None:
            try:
                self._splash.destroy()
            except Exception:
                pass
            self._splash = None
        self._closing = True
        self.cancel_current_response()
        self._stop_active_audio_playback()
        self._stop_avatar_viewer()
        self.stop_wake_word_listener()
        self._stop_bg_stream()
        with self._recording_wave_lock:
            if self.recording_wave is not None:
                self.recording_wave.close()
                self.recording_wave = None
        self.http_session.close()
        self.destroy()

    def _start_background_warmup(self) -> None:
        worker = threading.Thread(target=self._run_background_warmup, daemon=True)
        worker.start()

    def _run_background_warmup(self) -> None:
        # Warm up expensive dependencies once so first interaction feels immediate.
        try:
            ensure_ffmpeg_available()
            self.after(0, lambda: self._set_splash_status("Lade Spracherkennungsmodell..."))
            self.load_whisper_model(self.whisper_model_var.get().strip() or "small", announce=False)
            self.after(0, lambda: self._set_splash_status("Verbinde mit Ollama..."))
            self.check_ollama(force_refresh=True)
            self.after(0, lambda: self._set_splash_status("Bereit."))
        except Exception as exc:
            # Warmup should never block or break normal interaction.
            self.logger.warning("Hintergrundstart fehlgeschlagen: %s", exc)
        finally:
            self.after(300, self._close_splash)  # kurzes Delay damit "Bereit." sichtbar ist

    def _show_splash(self) -> None:
        import tkinter as tk
        splash = tk.Toplevel(self)
        splash.overrideredirect(True)
        sw = splash.winfo_screenwidth()
        sh = splash.winfo_screenheight()
        w, h = 460, 290
        x = (sw - w) // 2
        y = (sh - h) // 2
        splash.geometry(f"{w}x{h}+{x}+{y}")
        splash.configure(bg="#0d1117")
        splash.attributes("-topmost", True)

        # Rahmen
        border = tk.Frame(splash, bg="#1e293b", bd=0)
        border.place(x=1, y=1, width=w - 2, height=h - 2)

        tk.Label(
            border, text="Voice Studio", bg="#1e293b", fg="#e2e8f0",
            font=("Segoe UI", 32, "bold"),
        ).place(relx=0.5, y=72, anchor="center")

        tk.Label(
            border, text="Sprachassistent wird geladen", bg="#1e293b", fg="#64748b",
            font=("Segoe UI", 12),
        ).place(relx=0.5, y=114, anchor="center")

        canvas = tk.Canvas(border, width=52, height=52, bg="#1e293b", highlightthickness=0)
        canvas.place(relx=0.5, y=185, anchor="center")

        self._splash_status_var = tk.StringVar(value="Initialisiere...")
        tk.Label(
            border, textvariable=self._splash_status_var, bg="#1e293b",
            fg="#475569", font=("Segoe UI", 10),
        ).place(relx=0.5, y=248, anchor="center")

        self._splash = splash
        self._splash_canvas = canvas
        self._splash_angle = 0
        self._animate_splash_spinner()

    def _set_splash_status(self, text: str) -> None:
        if self._splash_status_var is not None:
            try:
                self._splash_status_var.set(text)
            except Exception:
                pass

    def _animate_splash_spinner(self) -> None:
        if self._splash is None:
            return
        canvas = self._splash_canvas
        if canvas is None:
            return
        try:
            canvas.delete("all")
            canvas.create_arc(
                4, 4, 48, 48,
                start=self._splash_angle, extent=270,
                outline="#22d3ee", style="arc", width=3,
            )
            self._splash_angle = (self._splash_angle + 10) % 360
            self._splash_spinner_job = self.after(33, self._animate_splash_spinner)
        except Exception:
            pass

    def _close_splash(self) -> None:
        if self._splash_spinner_job is not None:
            try:
                self.after_cancel(self._splash_spinner_job)
            except Exception:
                pass
            self._splash_spinner_job = None
        if self._splash is not None:
            try:
                self._splash.destroy()
            except Exception:
                pass
            self._splash = None
        if not self._closing:
            self.attributes("-alpha", 1.0)  # Transparenz aufheben bevor Fenster erscheint
            self.deiconify()
            self.state("zoomed")
            # Avatar erst nach dem Hauptfenster starten, damit Docking korrekt funktioniert
            self.after(500, self._start_avatar_viewer)

    def _set_stt_progress(self, value: float, label: str | None = None) -> None:
        clamped = max(0.0, min(1.0, value))

        def updater() -> None:
            self.stt_progress_bar.set(clamped)
            if label is not None:
                self.stt_progress_var.set(label)
            else:
                self.stt_progress_var.set(f"STT Laden: {int(clamped * 100)}%")

        self.after(0, updater)

    def _tick_stt_progress(self) -> None:
        if not self.stt_loading_active:
            self.stt_loading_job_id = None
            return

        # Approximate progress: backend download/init has no reliable percentage callback.
        self.stt_loading_value = min(0.95, self.stt_loading_value + 0.02)
        self._set_stt_progress(
            self.stt_loading_value,
            f"STT Laden ({self.stt_loading_model_name}): {int(self.stt_loading_value * 100)}%",
        )
        self.stt_loading_job_id = self.after(350, self._tick_stt_progress)

    def _start_stt_progress(self, model_name: str) -> None:
        self.stt_loading_active = True
        self.stt_loading_model_name = model_name
        self.stt_loading_value = 0.05
        self._set_stt_progress(self.stt_loading_value, f"STT Laden ({model_name}): 5%")
        if self.stt_loading_job_id is None:
            self.stt_loading_job_id = self.after(350, self._tick_stt_progress)

    def _finish_stt_progress(self, model_name: str, success: bool) -> None:
        self.stt_loading_active = False
        if self.stt_loading_job_id is not None:
            try:
                self.after_cancel(self.stt_loading_job_id)
            except Exception:
                pass
            self.stt_loading_job_id = None

        if success:
            self.stt_loading_value = 1.0
            self._set_stt_progress(1.0, f"STT Modell bereit ({model_name}): 100%")
        else:
            self.stt_loading_value = 0.0
            self._set_stt_progress(0.0, f"STT Laden fehlgeschlagen ({model_name})")

    def on_whisper_model_changed(self, selected_model: str) -> None:
        if selected_model in self.whisper_model_cache:
            self._set_stt_progress(1.0, f"STT Modell bereits geladen ({selected_model}): 100%")
            return

        self._start_stt_progress(selected_model)
        self.set_status(f"Lade STT-Modell im Hintergrund: {selected_model}")
        worker = threading.Thread(
            target=self._preload_whisper_model_safe,
            kwargs={"model_name": selected_model},
            daemon=True,
        )
        worker.start()

    def _preload_whisper_model_safe(self, model_name: str) -> None:
        try:
            self.load_whisper_model(model_name=model_name, announce=False)
            self._finish_stt_progress(model_name, success=True)
            self.set_status(f"STT-Modell bereit: {model_name}")
        except Exception as exc:
            self._finish_stt_progress(model_name, success=False)
            self.set_status(f"STT-Modell konnte nicht geladen werden: {exc}")

    def _resolve_whisper_model_name(self, model_name: str) -> str:
        normalized = model_name.strip().lower()
        if self.whisper_backend == "faster-whisper":
            if normalized == "large":
                return "large-v3"
            return normalized

        if normalized == "large-v3":
            return "large"
        return normalized

    def refresh_input_devices(self) -> None:
        try:
            devices = sd.query_devices()
            host_apis = sd.query_hostapis()
        except Exception as exc:
            self.mic_devices_map = {}
            self.mic_menu.configure(values=[NO_MIC_DEVICES_LABEL])
            self.mic_device_var.set(NO_MIC_DEVICES_LABEL)
            self.set_status(f"Mikrofone konnten nicht geladen werden: {exc}")
            return

        input_candidates: list[tuple[int, str, str]] = []
        for index, device in enumerate(devices):
            max_input_channels = int(device.get("max_input_channels", 0))
            if max_input_channels > 0:
                name = str(device.get("name", f"Input {index}"))
                host_index = int(device.get("hostapi", -1))
                host_name = ""
                if 0 <= host_index < len(host_apis):
                    host_name = str(host_apis[host_index].get("name", ""))
                input_candidates.append((index, name, host_name))

        wasapi_candidates = [item for item in input_candidates if "wasapi" in item[2].lower()]
        selected_candidates = wasapi_candidates if wasapi_candidates else input_candidates

        input_options: list[str] = [WINDOWS_DEFAULT_MIC_LABEL]
        input_map: dict[str, int] = {WINDOWS_DEFAULT_MIC_LABEL: -1}
        name_counts: dict[str, int] = {}
        for index, name, _host_name in selected_candidates:
            base_name = " ".join(name.split())
            count = name_counts.get(base_name, 0) + 1
            name_counts[base_name] = count
            label = base_name if count == 1 else f"{base_name} ({count})"
            input_options.append(label)
            input_map[label] = index

        if len(input_options) == 1:
            self.mic_devices_map = {}
            self.mic_menu.configure(values=[NO_MIC_DEVICES_LABEL])
            self.mic_device_var.set(NO_MIC_DEVICES_LABEL)
            self.set_status("Kein Mikrofon gefunden")
            return

        self.mic_devices_map = input_map
        self.mic_menu.configure(values=input_options)

        current_selection = self.mic_device_var.get().strip()
        if current_selection not in self.mic_devices_map:
            self.mic_device_var.set(input_options[0])

        if not self.is_recording:
            self.start_level_monitor()

        self.set_status(f"Mikrofone geladen: {len(input_options)}")

    def get_selected_input_device(self) -> int | None:
        selected = self.mic_device_var.get().strip()
        device = self.mic_devices_map.get(selected)
        if device == -1:
            return None
        return device

    def refresh_provider_models(self) -> None:
        provider_name = self.llm_provider_var.get().strip() or "Ollama"
        button_name = {
            "OpenAI": "refresh_openai_btn",
            "Azure OpenAI": "refresh_azure_btn",
            "Anthropic": "refresh_anthropic_btn",
            "Groq": "refresh_groq_btn",
        }.get(provider_name, "refresh_ollama_btn")
        btn = getattr(self, button_name, None)
        if btn is not None:
            btn.configure(state="disabled", text="Lädt...")

        def _fetch() -> None:
            try:
                provider_name = self.llm_provider_var.get().strip() or "Ollama"
                names = self.list_provider_models()
                if not names:
                    self.after(0, lambda: self.set_status("Keine Modelle fuer den aktiven Provider gefunden"))
                    return

                provider_key = provider_name.lower()
                if provider_key == "openai":
                    current = self.openai_model_var.get().strip()
                elif provider_key == "azure openai":
                    current = self.azure_openai_deployment_var.get().strip()
                elif provider_key == "anthropic":
                    current = self.anthropic_model_var.get().strip()
                elif provider_key == "groq":
                    current = self.groq_model_var.get().strip()
                else:
                    current = self.ollama_model_var.get().strip()

                def _apply() -> None:
                    self.ollama_model_menu.configure(values=names)
                    if hasattr(self, "openai_model_menu"):
                        self.openai_model_menu.configure(values=names)
                    if hasattr(self, "azure_openai_model_menu"):
                        self.azure_openai_model_menu.configure(values=names)
                    if hasattr(self, "anthropic_model_menu"):
                        self.anthropic_model_menu.configure(values=names)
                    if hasattr(self, "groq_model_menu"):
                        self.groq_model_menu.configure(values=names)
                    if current not in names:
                        selected = names[0]
                    else:
                        selected = current

                    if provider_key == "openai":
                        self.openai_model_var.set(selected)
                    elif provider_key == "azure openai":
                        self.azure_openai_deployment_var.set(selected)
                    elif provider_key == "anthropic":
                        self.anthropic_model_var.set(selected)
                    elif provider_key == "groq":
                        self.groq_model_var.set(selected)
                    else:
                        self.ollama_model_var.set(selected)

                    # Keep existing model dropdown readable by mirroring active provider model.
                    self.ollama_model_var.set(selected)
                    self.set_status(
                        f"{provider_name}: {len(names)} Modell(e) geladen: {', '.join(names[:4])}"
                        + (" …" if len(names) > 4 else "")
                    )

                self.after(0, _apply)
            except Exception as exc:
                self.after(
                    0,
                    lambda: self.set_status(f"Provider-Refresh fehlgeschlagen: {exc}"),
                )
            finally:
                def _reset_btn() -> None:
                    if btn is not None:
                        btn.configure(state="normal", text="Modelle vom Provider laden")
                self.after(0, _reset_btn)

        import threading
        threading.Thread(target=_fetch, daemon=True).start()

    def refresh_ollama_models(self) -> None:
        """Backward-compatible alias."""
        self.refresh_provider_models()

    # ── VRM model helpers ─────────────────────────────────────────────────

    def _vrm_model_dir(self) -> Path:
        return Path(__file__).resolve().parent / "runtime_assets" / "model"

    def _scan_vrm_models(self) -> list[str]:
        """Return sorted list of .vrm filenames in the model directory."""
        model_dir = self._vrm_model_dir()
        if not model_dir.exists():
            return []
        return sorted(p.name for p in model_dir.glob("*.vrm"))

    def _refresh_vrm_model_list(self) -> None:
        names = self._scan_vrm_models()
        menu = getattr(self, "vrm_model_menu", None)
        if menu is None:
            return
        if not names:
            menu.configure(values=["(Kein Modell gefunden)"])
            return
        menu.configure(values=names)
        if self.vrm_model_var.get() not in names:
            self.vrm_model_var.set(names[0])
        self.set_status(f"{len(names)} VRM-Modell(e) gefunden")

    def _browse_vrm_model(self) -> None:
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="VRM-Datei auswählen",
            filetypes=[("VRM-Dateien", "*.vrm"), ("Alle Dateien", "*.*")],
        )
        if not path:
            return
        src = Path(path)
        dest = self._vrm_model_dir() / src.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest != src:
            import shutil
            shutil.copy2(src, dest)
        self._refresh_vrm_model_list()
        self.vrm_model_var.set(src.name)
        self.set_status(f"Modell importiert: {src.name}")

    def _apply_vrm_model(self) -> None:
        selected = self.vrm_model_var.get().strip()
        if not selected or selected == "(Kein Modell gefunden)":
            self.set_status("Kein gültiges Modell ausgewählt")
            return
        vrm_rel = f"runtime_assets/model/{selected}"
        vrm_abs = Path(__file__).resolve().parent / vrm_rel
        if not vrm_abs.exists():
            self.set_status(f"Datei nicht gefunden: {vrm_rel}")
            return
        self.avatar_bridge.vrm_relative_path = vrm_rel
        if self.avatar_bridge.is_running():
            self._stop_avatar_viewer()
        self.after(300, self._start_avatar_viewer)
        self.set_status(f"Avatar-Viewer startet mit: {selected}")

    def _audio_callback(self, indata: Any, frames: int, callback_time: Any, status: Any) -> None:
        if self.recording_wave is None:
            return

        raw = indata.copy().tobytes()
        peak = float(abs(indata).max()) / 32767.0
        self.set_mic_level(peak)
        self.recording_wave.writeframes(raw)

        # Energy-based VAD – no external library required.
        if not self.vad_enabled_var.get():
            return

        elapsed_s = time.perf_counter() - (self.recording_started_at or 0.0)

        # Safety net: hard stop after 45 s
        if elapsed_s > 45.0:
            self.logger.info("VAD: maximale Aufnahmedauer – stoppe")
            self.after(0, self.stop_recording)
            return

        # Grace period: first 0.6 s are ignored so user has time to start
        if elapsed_s < 0.6:
            return

        # RMS energy (robust against int16 overflow by using peak already computed above)
        rms = peak  # peak = abs(indata).max() / 32767, good enough proxy for energy

        _SPEECH_RMS = 0.015   # above → speaking
        _SILENCE_RMS = 0.010  # below → silence

        now = time.perf_counter()
        if rms >= _SPEECH_RMS:
            self.vad_speech_detected = True
            self.vad_last_speech_time = now

        if not self.vad_speech_detected:
            return

        try:
            silence_timeout = float(self.vad_silence_timeout_var.get().strip())
        except ValueError:
            silence_timeout = 1.5

        silence_duration = now - self.vad_last_speech_time
        if rms < _SILENCE_RMS and silence_duration >= silence_timeout:
            self.logger.info(
                "VAD: %.1fs Stille nach Sprache (rms=%.4f) – stoppe",
                silence_duration, rms,
            )
            self.after(0, self.stop_recording)

    def start_recording(self) -> None:
        if self.is_recording:
            return

        self.cancel_current_response()

        try:
            sample_rate = int(self.sample_rate_var.get().strip())
        except ValueError:
            self.set_status("Ungültige Sample Rate")
            return

        if self.recording_path is not None:
            try:
                Path(self.recording_path).unlink(missing_ok=True)
            except OSError:
                pass
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_file.close()
        self.recording_path = temp_file.name

        self.recording_wave = wave.open(self.recording_path, "wb")
        self.recording_wave.setnchannels(1)
        self.recording_wave.setsampwidth(2)
        self.recording_wave.setframerate(sample_rate)

        selected_label = self.mic_device_var.get().strip()
        if selected_label not in self.mic_devices_map:
            self.set_status("Kein gültiges Mikrofon ausgewählt")
            self.recording_wave.close()
            self.recording_wave = None
            return

        input_device = self.get_selected_input_device()

        # Ensure unified stream is running on the correct device (no stop needed).
        if self._bg_stream is None or not self._bg_stream.active:
            self._start_bg_stream()

        # Reset energy-based VAD state
        self.vad_speech_detected = False
        self.vad_last_speech_time = 0.0
        if self.vad_enabled_var.get():
            self.logger.info("VAD aktiv (energie-basiert, Timeout %.1fs)",
                             float(self.vad_silence_timeout_var.get() or 1.5))

        self.is_recording = True
        self.recording_started_at = time.perf_counter()
        self._increment_counter("recordings_started")
        self.last_transcript = ""
        self.set_mic_level(0)
        vad_hint = " | VAD AN" if self.vad_enabled_var.get() else ""
        self.set_status(f"Mikrofon läuft...{vad_hint} ({self.mic_device_var.get()})")
        self.set_pipeline_phase("mic")
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.transcribe_btn.configure(state="disabled")
        self.send_btn.configure(state="disabled")

    def stop_recording(self) -> None:
        if not self.is_recording:
            return

        # Close the wav file – the unified bg_stream keeps running.
        if self.recording_wave is not None:
            self.recording_wave.close()
            self.recording_wave = None

        self.is_recording = False
        self.recording_stream = None  # nothing to clean up
        self._increment_counter("recordings_finished")
        if self.recording_started_at is not None:
            duration = time.perf_counter() - self.recording_started_at
            self._add_metric_sample("recording_seconds", duration)
            self.logger.info("Aufnahmedauer: %.2fs", duration)
            self.recording_started_at = None
        self.set_mic_level(0)
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        # bg_stream continues – no start_level_monitor() call needed

        if self.auto_pipeline_var.get():
            self.set_status("Aufnahme beendet. Starte Auto-Workflow...")
            self.transcribe_btn.configure(state="disabled")
            self.send_btn.configure(state="disabled")
            self._start_transcription_worker(auto_send=True)
        else:
            self.set_status("Aufnahme beendet. Bereit zum Verarbeiten.")
            self.set_pipeline_phase("idle")
            self.transcribe_btn.configure(state="normal")
            self.send_btn.configure(state="disabled")

    def load_whisper_model(self, model_name: str, announce: bool = True) -> Any:
        if self.whisper_module is None:
            try:
                self.whisper_module = importlib.import_module("faster_whisper")
                self.whisper_backend = "faster-whisper"
            except ModuleNotFoundError:
                self.whisper_module = importlib.import_module("whisper")
                self.whisper_backend = "openai-whisper"

        resolved_model_name = self._resolve_whisper_model_name(model_name)

        if model_name not in self.whisper_model_cache:
            if announce:
                self.set_status(
                    f"Lade STT-Modell: {resolved_model_name} ({self.whisper_backend}) - "
                    "beim ersten Mal kann Download dauern"
                )

            if self.whisper_backend == "faster-whisper":
                device = "cpu"
                compute_type = "int8"
                try:
                    torch_module = importlib.import_module("torch")
                    if bool(torch_module.cuda.is_available()):
                        device = "cuda"
                        compute_type = "float16"
                except Exception:
                    pass

                self.whisper_model_cache[model_name] = self.whisper_module.WhisperModel(
                    resolved_model_name,
                    device=device,
                    compute_type=compute_type,
                )
            else:
                self.whisper_model_cache[model_name] = self.whisper_module.load_model(resolved_model_name)

        return self.whisper_model_cache[model_name]

    def transcribe_recording(self) -> None:
        if self.is_recording:
            self.set_status("Bitte zuerst Aufnahme stoppen.")
            return

        if not self.recording_path or not Path(self.recording_path).exists():
            self.set_status("Keine Aufnahme gefunden.")
            return

        self.transcribe_btn.configure(state="disabled")
        self.send_btn.configure(state="disabled")
        self._start_transcription_worker(auto_send=False)

    def _start_transcription_worker(self, auto_send: bool) -> None:
        worker = threading.Thread(target=self._run_transcription, args=(auto_send,), daemon=True)
        worker.start()

    def _run_transcription(self, auto_send: bool = False) -> None:
        started = time.perf_counter()
        try:
            model_name = self.whisper_model_var.get().strip() or "small"
            audio_path = self.recording_path
            if not audio_path:
                self.set_status("Keine Aufnahme vorhanden.")
                return

            self.set_status("Transkribiere Audio...")
            self.after(0, lambda: self.set_pipeline_phase("stt"))
            if not ensure_ffmpeg_available():
                raise FileNotFoundError(
                    "ffmpeg nicht gefunden. Bitte ffmpeg installieren oder Terminal/VS Code neu starten."
                )

            model = self.load_whisper_model(model_name)
            language_label = self.whisper_language_var.get().strip()
            language_code = WHISPER_LANGUAGE_OPTIONS.get(language_label, "")
            speed_mode = self.whisper_speed_var.get().strip()

            if self.whisper_backend == "faster-whisper":
                transcribe_options: dict[str, Any] = {
                    "beam_size": 1 if speed_mode == "Schnell" else 3,
                    "condition_on_previous_text": False if speed_mode == "Schnell" else True,
                    "vad_filter": True,
                }
                if language_code:
                    transcribe_options["language"] = language_code

                segments, _info = model.transcribe(audio_path, **transcribe_options)
                transcript = "".join(segment.text for segment in segments).strip()
            else:
                use_fp16 = str(model.device).lower().startswith("cuda")
                transcribe_options = {
                    "fp16": use_fp16,
                }
                if language_code:
                    transcribe_options["language"] = language_code

                if speed_mode == "Schnell":
                    transcribe_options.update(
                        {
                            "beam_size": 1,
                            "best_of": 1,
                            "temperature": 0.0,
                            "condition_on_previous_text": False,
                        }
                    )

                result = model.transcribe(audio_path, **transcribe_options)
                transcript = str(result.get("text", "")).strip()

            if not transcript:
                self.last_transcript = ""
                self.set_textbox(self.transcript_box, "[Kein Text erkannt. Bitte deutlicher/länger sprechen und erneut versuchen.]")
                self.set_textbox(self.answer_box, "")
                self.set_status("Kein Text erkannt")
                return

            self.last_transcript = transcript
            self._increment_counter("transcriptions")
            self._add_metric_sample("transcription_seconds", time.perf_counter() - started)
            self.logger.info("Transkription fertig (%d Zeichen)", len(transcript))
            self.set_textbox(self.transcript_box, transcript)
            if auto_send:
                self.set_status("Transkript erkannt. Sende an Ollama...")
                self._run_ollama_only(transcript, manage_buttons=False)
            else:
                self.set_status("Transkript erkannt")
                self.after(0, lambda: self.send_btn.configure(state="normal"))
        except Exception as exc:
            self._log_exception("Transkription", exc)
            self.set_status(f"Fehler: {exc}")
        finally:
            self.after(0, lambda: self.transcribe_btn.configure(state="normal"))
            if auto_send:
                self.after(0, lambda: self.send_btn.configure(state="normal"))

    def get_transcript_text(self) -> str:
        return self.transcript_box.get("1.0", "end").strip()

    def send_to_ollama(self) -> None:
        if self.is_recording:
            self.set_status("Bitte zuerst Aufnahme stoppen.")
            return

        transcript = self.get_transcript_text()
        if not transcript or transcript.startswith("[Kein Text erkannt"):
            self.set_status("Bitte zuerst transkribieren oder Text eingeben.")
            return

        self.send_btn.configure(state="disabled")
        self.transcribe_btn.configure(state="disabled")
        self.text_send_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        worker = threading.Thread(target=self._run_ollama_only, args=(transcript,), daemon=True)
        worker.start()

    def _run_ollama_only(self, transcript: str, manage_buttons: bool = True) -> None:
        request_started = time.perf_counter()
        try:
            self.cancel_ollama_event.clear()
            self.cancel_tts_event.clear()
            self._active_response_started_at = request_started
            self._first_audio_chunk_recorded = False
            self.after(0, lambda: self.cancel_btn.configure(state="normal"))
            self._increment_counter("ollama_requests")
            self.logger.info("Ollama Anfrage gestartet (%d Zeichen Input)", len(transcript))

            self.set_status("Frage Ollama... (erster Lauf kann etwas dauern)")
            self.after(0, lambda: self.set_pipeline_phase("ollama"))
            self.after(0, self._start_thinking)
            self.add_history_entry("Du", transcript)
            self.set_textbox(self.answer_box, "")

            # Fast pre-filter: only call the LLM router when light keywords are present.
            # This avoids an extra 1-3s round-trip for every normal conversation turn.
            local_hint = self._parse_light_command(transcript)
            if local_hint is not None:
                tool_action = self._decide_tool_action_with_ollama(transcript)
                # If the LLM router is unsure, trust the regex hint as fallback.
                if tool_action == "none":
                    tool_action = "light_on" if local_hint == "on" else "light_off"
            else:
                tool_action = "none"

            # tts_queue muss vor diesem Branch definiert sein (wird in finally geprüft)
            tts_queue = None

            if tool_action in {"light_on", "light_off"}:
                self.after(0, self._stop_thinking)
                self.after(0, lambda: self._stop_cursor(self.answer_box))
                self.after(0, self.open_light_test_popup)
                if tool_action == "light_on":
                    self.after(0, lambda: self.set_light_state(True, source="ollama-tool-router"))
                    answer = "Alles klar, ich habe das Licht fuer dich eingeschaltet."
                else:
                    self.after(0, lambda: self.set_light_state(False, source="ollama-tool-router"))
                    answer = "Alles klar, ich habe das Licht fuer dich ausgeschaltet."

                self.set_textbox(self.answer_box, answer)
                self.add_history_entry("Assistent", answer)
                self.after(0, lambda: self.set_pipeline_phase("idle"))

                if self.auto_speak_var.get() and not self.cancel_tts_event.is_set():
                    self._speak_text_background(answer, done_status="Lichtbefehl ueber Ollama-Toolrouting ausgefuehrt")
                else:
                    self.set_status("Lichtbefehl ueber Ollama-Toolrouting ausgefuehrt")

                return

            chunk_buffer: list[str] = []
            first_token_received = False
            first_token_time: float | None = None
            tts_queue = None
            tts_text_buffer = ""
            tts_phrase_buffer = ""
            tts_phrase_count = 0
            tts_last_emit_at = request_started
            tts_first_emitted = False  # erste Phrase sofort senden, unabhängig von der Mindestlänge

            if self.auto_speak_var.get():
                tts_queue, _ = self._start_streaming_tts_worker()

            def flush_chunks() -> None:
                if not chunk_buffer:
                    return
                chunk_text = "".join(chunk_buffer)
                chunk_buffer.clear()
                self.append_textbox(self.answer_box, chunk_text)

            def on_chunk(chunk: str) -> None:
                nonlocal first_token_received
                nonlocal first_token_time
                nonlocal tts_text_buffer
                nonlocal tts_phrase_buffer
                nonlocal tts_phrase_count
                nonlocal tts_last_emit_at
                nonlocal tts_first_emitted

                if self.cancel_ollama_event.is_set():
                    return

                chunk_buffer.append(chunk)
                if not first_token_received:
                    first_token_received = True
                    first_token_time = time.perf_counter()
                    self.set_status("Ollama antwortet...")
                    self.after(0, lambda: self.set_pipeline_phase("ollama"))
                    self.after(0, self._stop_thinking)
                    self.after(0, lambda: self._start_cursor(self.answer_box))

                if tts_queue is not None:
                    tts_text_buffer += chunk

                    def enqueue_phrase(phrase: str) -> None:
                        nonlocal tts_phrase_buffer
                        nonlocal tts_phrase_count
                        nonlocal tts_last_emit_at
                        nonlocal tts_first_emitted

                        normalized_phrase = phrase.strip()
                        if not normalized_phrase:
                            return

                        if tts_phrase_buffer:
                            tts_phrase_buffer = f"{tts_phrase_buffer} {normalized_phrase}"
                        else:
                            tts_phrase_buffer = normalized_phrase
                        tts_phrase_count += 1

                        should_emit_phrase = (
                            (not tts_first_emitted and len(tts_phrase_buffer) >= self._tts_stream_first_chars)
                            or len(tts_phrase_buffer) >= self._tts_stream_min_chars
                            or tts_phrase_count >= TTS_STREAM_MAX_SENTENCES
                            or normalized_phrase.endswith("\n")
                        )
                        if should_emit_phrase:
                            tts_queue.put(tts_phrase_buffer)
                            tts_last_emit_at = time.perf_counter()
                            tts_first_emitted = True
                            tts_phrase_buffer = ""
                            tts_phrase_count = 0

                    ready_sentences, tts_text_buffer = self._extract_complete_sentences(tts_text_buffer)
                    for sentence in ready_sentences:
                        enqueue_phrase(sentence)

                    ready_phrases, tts_text_buffer = self._extract_complete_phrases(tts_text_buffer)
                    for phrase in ready_phrases:
                        enqueue_phrase(phrase)

                    if len(tts_text_buffer) >= self._tts_stream_max_buffer_chars:
                        split_idx = tts_text_buffer.rfind(" ", 0, self._tts_stream_max_buffer_chars)
                        if split_idx <= 0:
                            split_idx = self._tts_stream_max_buffer_chars
                        chunked_phrase = tts_text_buffer[:split_idx].strip()
                        tts_text_buffer = tts_text_buffer[split_idx:].lstrip()
                        enqueue_phrase(chunked_phrase)

                    if (
                        tts_phrase_buffer
                        and (time.perf_counter() - tts_last_emit_at) >= self._tts_stream_max_wait_seconds
                        and len(tts_phrase_buffer) >= self._tts_stream_first_chars
                    ):
                        tts_queue.put(tts_phrase_buffer)
                        tts_last_emit_at = time.perf_counter()
                        tts_first_emitted = True
                        tts_phrase_buffer = ""
                        tts_phrase_count = 0

                should_flush = len(chunk_buffer) >= 3 or chunk.endswith((".", "!", "?", "\n"))
                if should_flush:
                    flush_chunks()

            answer = self.ask_ollama(
                transcript,
                on_chunk=on_chunk,
                cancel_event=self.cancel_ollama_event,
            )
            flush_chunks()
            # Stream done — remove cursor
            self.after(0, lambda: self._stop_cursor(self.answer_box))

            total_elapsed = time.perf_counter() - request_started
            self._add_metric_sample("ollama_total_seconds", total_elapsed)
            if first_token_time is not None:
                self._add_metric_sample("ollama_first_token_seconds", first_token_time - request_started)
            self.logger.info("Ollama Antwort fertig (%d Zeichen, %.2fs)", len(answer), total_elapsed)

            if tts_queue is not None:
                if tts_phrase_buffer.strip():
                    tts_queue.put(tts_phrase_buffer.strip())
                trailing = tts_text_buffer.strip()
                if trailing:
                    tts_queue.put(trailing)
                tts_queue.put(None)

            if self.cancel_ollama_event.is_set():
                self.set_status(DEFAULT_ABORTED_STATUS)
                return

            if not answer:
                self.set_textbox(self.answer_box, "[Keine Antwort erhalten]")

            self.add_history_entry("Assistent", answer)

            if tts_queue is not None:
                self.set_status("Antwort fertig. Audio streamt...")
                self.after(0, lambda: self.set_pipeline_phase("tts"))
                return

            self.set_status("Fertig")
            self.after(0, lambda: self.set_pipeline_phase("idle"))
        except Exception as exc:
            if self.cancel_ollama_event.is_set():
                self.set_status(DEFAULT_ABORTED_STATUS)
                self.after(0, lambda: self.set_pipeline_phase("idle"))
                return
            self._log_exception("Ollama", exc)
            self.set_status(f"Fehler: {exc}")
        finally:
            self._active_response_started_at = None
            self._first_audio_chunk_recorded = False
            # Ensure TTS worker is always terminated even on exception.
            if tts_queue is not None and self.current_tts_queue is tts_queue:
                try:
                    tts_queue.put_nowait(None)
                except Exception:
                    pass
            if not self._closing:
                self.after(0, lambda: self.cancel_btn.configure(state="disabled"))
                if manage_buttons:
                    self.after(0, lambda: self.send_btn.configure(state="normal"))
                    self.after(0, lambda: self.transcribe_btn.configure(state="normal"))
                    self.after(0, lambda: self.text_send_btn.configure(state="normal"))


def main() -> None:
    app = VoiceAssistantUI()
    app.mainloop()


if __name__ == "__main__":
    main()
