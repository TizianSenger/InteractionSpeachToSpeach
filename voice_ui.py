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
from app_behavior_mixin import AppBehaviorMixin
from constants import *
from llm_provider_mixin import LlmProviderMixin
from profile_persistence_mixin import ProfilePersistenceMixin
from provider_ui_mixin import ProviderUiMixin
from stats_logging_mixin import StatsLoggingMixin
from tts_mixin import TtsMixin
from ui_layout_mixin import UiLayoutMixin
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


class VoiceAssistantUI(
    UiLayoutMixin,
    LlmProviderMixin,
    ProviderUiMixin,
    ProfilePersistenceMixin,
    AppBehaviorMixin,
    StatsLoggingMixin,
    TtsMixin,
    WakeWordMixin,
    ctk.CTk,
):
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
        self._oww_model: Any | None = None

        # Core runtime/UI state that must exist before profile loading.
        self.whisper_model_var = ctk.StringVar(value=WHISPER_MODEL_OPTIONS[0])
        self.realtime_mode_var = ctk.StringVar(value="Balanced")
        self.llm_provider_var = ctk.StringVar(value="Ollama")
        self.ollama_model_var = ctk.StringVar(value=OLLAMA_MODEL_OPTIONS[0])
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
        self.store_api_keys_var = ctk.BooleanVar(value=False)
        self.avatar_lipsync_var = ctk.BooleanVar(value=True)
        self.vrm_model_var = ctk.StringVar(value="Sakura.vrm")

        self.status_var = ctk.StringVar(value="Bereit")
        self.stats_summary_var = ctk.StringVar(value="")
        self.stats_latency_var = ctk.StringVar(value="")
        self.stt_progress_var = ctk.StringVar(value="")
        self.provider_diagnostics_var = ctk.StringVar(value="")
        self.debug_log_level_var = ctk.StringVar(value="INFO")
        self.light_state_var = ctk.StringVar(value="Lichtstatus: AUS")

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.debug_log_history: deque[str] = deque(maxlen=4000)
        self.max_ui_log_lines = 1200
        self.logger = self._setup_logger()

        self.http_session = requests.Session()
        self.ollama_check_cache: dict[str, Any] = {}
        self.ollama_cache_ttl_seconds = 10.0
        self.active_ollama_response_lock = threading.Lock()
        self.active_ollama_response = None
        self.cancel_ollama_event = threading.Event()
        self.cancel_tts_event = threading.Event()
        self.current_tts_queue: queue.Queue[str | None] | None = None
        self.conversation_history: list[dict[str, str]] = []

        self.app_started_at = time.time()
        self.event_history: deque[str] = deque(maxlen=300)
        self._events_displayed = 0
        self.counters: dict[str, int] = {
            "recordings_started": 0,
            "recordings_finished": 0,
            "transcriptions": 0,
            "ollama_requests": 0,
            "ollama_cancels": 0,
            "tts_chunks": 0,
            "errors": 0,
        }
        self.metric_samples: dict[str, list[float]] = {
            "transcription_seconds": [],
            "ollama_first_token_seconds": [],
            "first_audio_seconds": [],
            "ollama_total_seconds": [],
            "tts_chunk_seconds": [],
        }
        self.stats_refresh_job_id = None
        self.log_pump_job_id = None

        self.pipeline_phase = "idle"
        self._pipeline_anim_job = None
        self._pipeline_anim_tick = 0
        self._thinking_active = False
        self._thinking_job = None
        self._thinking_tick = 0
        self._cursor_active = False
        self._cursor_visible = False
        self._cursor_job = None

        self.settings_popup = None
        self.body_frame = None
        self.viewer_host_frame = None
        self.column_frames: dict[str, Any] = {}
        self.column_visible: dict[str, bool] = {"left": True, "middle": True, "right": True}
        self.column_weights: dict[str, int] = {"left": 5, "middle": 2, "right": 5}

        self.viewer_process = None
        self.viewer_process_lock = threading.Lock()
        self.embedded_viewer_hwnd = None
        self.avatar_bridge = AvatarBridge(
            base_dir=Path(__file__).resolve().parent,
            http_session=self.http_session,
            lipsync_enabled_getter=lambda: bool(self.avatar_lipsync_var.get()),
            set_status=self.set_status,
            log_exception=self._log_exception,
            logger=self.logger,
        )
        self.light_popup = None
        self.light_state_label = None
        self.light_indicator = None
        self.light_state = False
        self._first_audio_chunk_recorded = False
        self._active_response_started_at: float | None = None
        self.waveform_samples: deque[float] = deque(maxlen=240)
        self.waveform_animate_job = None
        self._ww_pulse_tick = 0
        self._ww_flash_frames = 0
        self.stt_loading_job_id = None
        self.whisper_model_cache: dict[str, Any] = {}
        self.whisper_module: Any | None = None
        self.whisper_backend = "faster-whisper"
        self.piper_models_map: dict[str, str] = {}
        self.mic_devices_map: dict[str, int] = {}
        self.pyttsx3_engine = None
        self.pyttsx3_engine_lock = threading.Lock()
        self.recording_started_at: float | None = None
        self.last_transcript = ""

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
