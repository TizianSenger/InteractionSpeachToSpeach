import asyncio
from collections import deque
from datetime import datetime
import importlib
import json
import logging
import math
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import wave
from pathlib import Path
from typing import Any, Callable

import customtkinter as ctk
import pyttsx3
import requests
import sounddevice as sd

FONT_FAMILY = "Segoe UI"
NO_MIC_DEVICES_LABEL = "Keine Geräte gefunden"
WINDOWS_DEFAULT_MIC_LABEL = "Standard (Windows)"
NO_PIPER_MODELS_LABEL = "Keine Piper-Stimmen gefunden"
OLLAMA_MODEL_OPTIONS = ["phi4-mini", "dolphin-mistral"]
PYTTSX3_VOICE_OPTIONS = ["Deutsch (weiblich) - Lokal", "Deutsch (männlich) - Lokal"]
EDGE_VOICE_OPTIONS: dict[str, str] = {
    "Deutsch (weiblich) - Katja": "de-DE-KatjaNeural",
    "Deutsch (männlich) - Conrad": "de-DE-ConradNeural",
    "Deutsch (männlich, tief) - Killian": "de-DE-KillianNeural",
    "Deutsch (weiblich) - Amala": "de-DE-AmalaNeural",
    "Deutsch (männlich) - Florian": "de-DE-FlorianMultilingualNeural",
}
EMOTION_PRESETS: dict[str, tuple[str, str]] = {
    "neutral": ("+0%", "+0Hz"),
    "freundlich": ("+8%", "+10Hz"),
    "fröhlich": ("+15%", "+30Hz"),
    "ruhig": ("-10%", "-10Hz"),
    "ernst": ("-5%", "-20Hz"),
}
WHISPER_LANGUAGE_OPTIONS: dict[str, str] = {
    "Auto": "",
    "Deutsch": "de",
    "Englisch": "en",
}
WHISPER_SPEED_OPTIONS = ["Schnell", "Genau"]
WHISPER_MODEL_OPTIONS = ["small", "medium"]
TTS_STREAM_MIN_CHARS = 120
TTS_STREAM_MAX_SENTENCES = 2
DEFAULT_ABORTED_STATUS = "Antwort abgebrochen"
OLLAMA_VOICE_SYSTEM_PROMPT = (
    "Du bist ein Sprachassistent. Antworte kurz, klar und direkt auf Deutsch. "
    "Standard: 1-3 Saetze, keine langen Ausfuehrungen."
)
OLLAMA_TOOL_ROUTER_PROMPT = (
    "Du bist ein Tool-Router. Entscheide nur, ob ein Licht-Tool ausgefuehrt werden soll. "
    "Antworte NUR als JSON ohne Markdown. Format: "
    '{"tool":"none|light_on|light_off","confidence":0.0,"reason":"kurz"}. '
    "Nutze light_on nur bei klarer Absicht Licht einzuschalten, light_off nur bei klarer Absicht Licht auszuschalten. "
    "Wenn unklar oder kein Lichtbezug, nutze none."
)
MODEL_GLOB_PATTERN = "*.onnx"
LOG_DIR_NAME = "logs"
LOG_FILE_NAME = "voice_ui.log"
PROFILE_FILE_NAME = "assistant_profile.json"
DEFAULT_VRM_RELATIVE_PATH = "runtime_assets/model/vrm_AvatarSample_S.vrm"

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


def safe_remove_file(path: str, retries: int = 6, delay_seconds: float = 0.15) -> None:
    for _ in range(retries):
        try:
            if os.path.exists(path):
                os.remove(path)
            return
        except PermissionError:
            time.sleep(delay_seconds)
        except OSError:
            time.sleep(delay_seconds)


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


class VoiceAssistantUI(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Voice Studio")
        self.geometry("1160x760")

        self.recording_stream: sd.InputStream | None = None
        self.monitor_stream: sd.InputStream | None = None
        self.recording_wave: wave.Wave_write | None = None
        self.recording_path: str | None = None
        self.is_recording = False
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
        self.avatar_http_port: int | None = None
        self.avatar_server_module: Any | None = None
        self.avatar_viewer_process: subprocess.Popen[str] | None = None
        self.avatar_last_push_at = 0.0
        self.avatar_auto_start_attempted = False
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
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.log_pump_job_id: str | None = None
        self.stats_refresh_job_id: str | None = None
        self.max_ui_log_lines = 1500
        self.debug_log_history: deque[str] = deque(maxlen=2500)
        self.event_history: deque[str] = deque(maxlen=400)
        self.metric_samples: dict[str, list[float]] = {
            "recording_seconds": [],
            "transcription_seconds": [],
            "ollama_first_token_seconds": [],
            "ollama_total_seconds": [],
            "tts_chunk_seconds": [],
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

        self.status_var = ctk.StringVar(value="Bereit")
        self.debug_log_level_var = ctk.StringVar(value="INFO")
        self.stats_summary_var = ctk.StringVar(value="Keine Daten")
        self.stats_latency_var = ctk.StringVar(value="Latenzen: -")
        self.whisper_model_var = ctk.StringVar(value="small")
        self.ollama_model_var = ctk.StringVar(value="phi4-mini")
        self.ollama_url_var = ctk.StringVar(value="http://localhost:11434")
        self.whisper_language_var = ctk.StringVar(value="Deutsch")
        self.whisper_speed_var = ctk.StringVar(value="Schnell")
        self.mic_device_var = ctk.StringVar(value="")
        self.sample_rate_var = ctk.StringVar(value="16000")
        self.tts_rate_var = ctk.StringVar(value="170")
        self.tts_engine_var = ctk.StringVar(value="edge-tts (natürlich)")
        self.tts_voice_var = ctk.StringVar(value="Deutsch (männlich, tief) - Killian")
        self.tts_emotion_var = ctk.StringVar(value="freundlich")
        self.piper_model_path_var = ctk.StringVar(value="models/de_DE-karlsson-medium.onnx")
        self.piper_config_path_var = ctk.StringVar(value="")
        self.mic_level_text_var = ctk.StringVar(value="Pegel: 0%")
        self.auto_speak_var = ctk.BooleanVar(value=True)
        self.auto_pipeline_var = ctk.BooleanVar(value=False)
        self.reply_max_tokens_var = ctk.StringVar(value="120")
        self.reply_temperature_var = ctk.StringVar(value="0.3")
        self.concise_reply_var = ctk.BooleanVar(value=True)
        self.favorite_light_scene_var = ctk.StringVar(value="Abendlicht")
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

        self._load_profile()

        self._build_layout()
        self.refresh_piper_model_options()
        self.on_tts_engine_changed(self.tts_engine_var.get())
        self.refresh_input_devices()
        self.start_level_monitor()
        self._start_background_warmup()
        self._schedule_log_pump()
        self._schedule_stats_refresh()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

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
        content = "\n".join(self.event_history)
        self.events_box.delete("1.0", "end")
        self.events_box.insert("1.0", content if content else "Noch keine Events")
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
            f"Ollama gesamt: {self._format_metric('ollama_total_seconds')} | "
            f"TTS Chunk: {self._format_metric('tts_chunk_seconds')}"
        )
        self.stats_latency_var.set(latency_text)

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
                "favorite_light_scene": self.favorite_light_scene_var.get().strip() or "Abendlicht",
                "mic_device_label": self.mic_device_var.get().strip(),
                "tts_engine": self.tts_engine_var.get().strip(),
                "tts_voice": self.tts_voice_var.get().strip(),
            },
        }

    def save_profile(self, notify: bool = True) -> None:
        profile_path = self._profile_path()
        payload = self._collect_profile_data()
        profile_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")
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
            persona = data.get("persona", {}) if isinstance(data, dict) else {}
            preferences = data.get("preferences", {}) if isinstance(data, dict) else {}

            self.persona_flirty_var.set(float(persona.get("flirty", self.persona_flirty_var.get())))
            self.persona_humor_var.set(float(persona.get("humor", self.persona_humor_var.get())))
            self.persona_serious_var.set(float(persona.get("serious", self.persona_serious_var.get())))
            self.persona_dominance_var.set(float(persona.get("dominance", self.persona_dominance_var.get())))
            self.persona_empathy_var.set(float(persona.get("empathy", self.persona_empathy_var.get())))
            self.persona_temperament_var.set(float(persona.get("temperament", self.persona_temperament_var.get())))

            fav_scene = str(preferences.get("favorite_light_scene", "")).strip()
            if fav_scene:
                self.favorite_light_scene_var.set(fav_scene)

            stored_mic = str(preferences.get("mic_device_label", "")).strip()
            if stored_mic:
                self.mic_device_var.set(stored_mic)

            stored_engine = str(preferences.get("tts_engine", "")).strip()
            if stored_engine:
                self.tts_engine_var.set(stored_engine)

            stored_voice = str(preferences.get("tts_voice", "")).strip()
            if stored_voice:
                self.tts_voice_var.set(stored_voice)
        except Exception as exc:
            self.logger.warning("Profil konnte nicht geladen werden: %s", exc)

        self._refresh_persona_labels()

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
        favorite_scene = self.favorite_light_scene_var.get().strip() or "Abendlicht"

        concise_instruction = "Antworte in 1-3 Saetzen." if self.concise_reply_var.get() else "Antworte so detailliert wie noetig."
        return (
            "Du bist ein persoenlicher Sprachassistent auf Deutsch. "
            "Sei menschlich, charmant und kontextbewusst, aber respektvoll und nicht beleidigend. "
            f"Stil: {flirty}; {humor}; {serious}; {dominance}; {empathy}; {temperament}. "
            f"Merke als bevorzugte Lichtszene des Users: {favorite_scene}. "
            f"{concise_instruction}"
        )

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(3, weight=1)
        self.grid_rowconfigure(4, weight=1)

        header = ctk.CTkFrame(self)
        header.grid(row=0, column=0, columnspan=3, padx=16, pady=(16, 10), sticky="ew")
        header.grid_columnconfigure(1, weight=1)

        title = ctk.CTkLabel(header, text="Voice Studio", font=(FONT_FAMILY, 22, "bold"))
        title.grid(row=0, column=0, padx=12, pady=(10, 2), sticky="w")

        subtitle = ctk.CTkLabel(
            header,
            text="Aufnehmen → Transkribieren → An Modell senden (optional automatisch)",
            font=(FONT_FAMILY, 12),
            text_color="gray70",
        )
        subtitle.grid(row=1, column=0, padx=12, pady=(0, 10), sticky="w")

        status = ctk.CTkLabel(header, textvariable=self.status_var, font=(FONT_FAMILY, 14))
        status.grid(row=0, column=1, padx=12, pady=12, sticky="e")

        workflow = ctk.CTkFrame(self)
        workflow.grid(row=1, column=0, columnspan=3, padx=16, pady=(0, 10), sticky="ew")
        for i in range(12):
            workflow.grid_columnconfigure(i, weight=1)

        self.start_btn = ctk.CTkButton(workflow, text="🎙️ Mic Start", command=self.start_recording)
        self.start_btn.grid(row=0, column=0, padx=6, pady=10, sticky="ew")

        self.stop_btn = ctk.CTkButton(workflow, text="⏹ Mic Stop", command=self.stop_recording, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=6, pady=10, sticky="ew")

        self.transcribe_btn = ctk.CTkButton(
            workflow,
            text="📝 Transkribieren",
            command=self.transcribe_recording,
            state="disabled",
            fg_color="#1f6aa5",
        )
        self.transcribe_btn.grid(row=0, column=2, padx=6, pady=10, sticky="ew")

        self.send_btn = ctk.CTkButton(
            workflow,
            text="📤 An Ollama senden",
            command=self.send_to_ollama,
            state="disabled",
            fg_color="#0f766e",
        )
        self.send_btn.grid(row=0, column=3, padx=6, pady=10, sticky="ew")

        self.cancel_btn = ctk.CTkButton(
            workflow,
            text="⛔ Antwort abbrechen",
            command=self.cancel_current_response,
            state="disabled",
            fg_color="#b42318",
            hover_color="#8f1d15",
        )
        self.cancel_btn.grid(row=0, column=4, padx=6, pady=10, sticky="ew")

        self.test_btn = ctk.CTkButton(workflow, text="🔎 Ollama Test", command=self.test_ollama)
        self.test_btn.grid(row=0, column=5, padx=6, pady=10, sticky="ew")

        self.text_send_btn = ctk.CTkButton(
            workflow,
            text="⌨️ Text direkt senden",
            command=self.send_to_ollama,
            fg_color="#7c3aed",
        )
        self.text_send_btn.grid(row=0, column=6, padx=6, pady=10, sticky="ew")

        self.speak_switch = ctk.CTkSwitch(workflow, text="Antwort vorlesen", variable=self.auto_speak_var)
        self.speak_switch.grid(row=0, column=7, padx=6, pady=10, sticky="w")

        self.light_test_btn = ctk.CTkButton(
            workflow,
            text="💡 Licht Test",
            command=self.open_light_test_popup,
            fg_color="#a16207",
            hover_color="#854d0e",
        )
        self.light_test_btn.grid(row=0, column=8, padx=6, pady=10, sticky="ew")

        self.auto_pipeline_switch = ctk.CTkSwitch(
            workflow,
            text="Auto: Stop -> Transkribieren -> Senden",
            variable=self.auto_pipeline_var,
        )
        self.auto_pipeline_switch.grid(row=0, column=9, columnspan=1, padx=6, pady=10, sticky="w")

        self.avatar_btn = ctk.CTkButton(
            workflow,
            text="Avatar starten",
            command=self.toggle_avatar_viewer,
            fg_color="#1d4ed8",
            hover_color="#1e40af",
        )
        self.avatar_btn.grid(row=0, column=10, padx=6, pady=10, sticky="ew")

        self.avatar_lipsync_switch = ctk.CTkSwitch(
            workflow,
            text="Avatar LipSync",
            variable=self.avatar_lipsync_var,
        )
        self.avatar_lipsync_switch.grid(row=0, column=11, padx=6, pady=10, sticky="w")

        sidebar = ctk.CTkScrollableFrame(self, label_text="Einstellungen")
        sidebar.grid(row=2, column=0, rowspan=3, padx=(16, 8), pady=(0, 16), sticky="nsew")

        stt_frame = ctk.CTkFrame(sidebar)
        stt_frame.pack(fill="x", padx=8, pady=(8, 6))
        ctk.CTkLabel(stt_frame, text="STT", font=(FONT_FAMILY, 14, "bold")).pack(anchor="w", padx=10, pady=(8, 2))

        ctk.CTkLabel(stt_frame, text="Whisper Modell").pack(anchor="w", padx=10, pady=(4, 2))
        self.whisper_menu = ctk.CTkOptionMenu(
            stt_frame,
            values=WHISPER_MODEL_OPTIONS,
            variable=self.whisper_model_var,
            command=self.on_whisper_model_changed,
        )
        self.whisper_menu.pack(fill="x", padx=10, pady=(0, 8))

        ctk.CTkLabel(stt_frame, text="Sprache").pack(anchor="w", padx=10, pady=(4, 2))
        self.whisper_language_menu = ctk.CTkOptionMenu(
            stt_frame,
            values=list(WHISPER_LANGUAGE_OPTIONS.keys()),
            variable=self.whisper_language_var,
        )
        self.whisper_language_menu.pack(fill="x", padx=10, pady=(0, 8))

        ctk.CTkLabel(stt_frame, text="Modus").pack(anchor="w", padx=10, pady=(4, 2))
        self.whisper_speed_menu = ctk.CTkOptionMenu(
            stt_frame,
            values=WHISPER_SPEED_OPTIONS,
            variable=self.whisper_speed_var,
        )
        self.whisper_speed_menu.pack(fill="x", padx=10, pady=(0, 10))

        self.stt_progress_bar = ctk.CTkProgressBar(stt_frame)
        self.stt_progress_bar.pack(fill="x", padx=10, pady=(0, 4))
        self.stt_progress_bar.set(0)

        self.stt_progress_label = ctk.CTkLabel(stt_frame, textvariable=self.stt_progress_var, text_color="gray70")
        self.stt_progress_label.pack(anchor="w", padx=10, pady=(0, 10))

        model_frame = ctk.CTkFrame(sidebar)
        model_frame.pack(fill="x", padx=8, pady=6)
        ctk.CTkLabel(model_frame, text="Modell", font=(FONT_FAMILY, 14, "bold")).pack(anchor="w", padx=10, pady=(8, 2))

        ctk.CTkLabel(model_frame, text="Ollama Modell").pack(anchor="w", padx=10, pady=(4, 2))
        self.ollama_model_menu = ctk.CTkOptionMenu(
            model_frame,
            values=OLLAMA_MODEL_OPTIONS,
            variable=self.ollama_model_var,
        )
        self.ollama_model_menu.pack(fill="x", padx=10, pady=(0, 8))

        ctk.CTkLabel(model_frame, text="Ollama URL").pack(anchor="w", padx=10, pady=(4, 2))
        self.ollama_url_entry = ctk.CTkEntry(model_frame, textvariable=self.ollama_url_var)
        self.ollama_url_entry.pack(fill="x", padx=10, pady=(0, 10))

        self.concise_reply_switch = ctk.CTkSwitch(
            model_frame,
            text="Kurze Voice-Antworten",
            variable=self.concise_reply_var,
        )
        self.concise_reply_switch.pack(anchor="w", padx=10, pady=(0, 8))

        ctk.CTkLabel(model_frame, text="Max Tokens (num_predict)").pack(anchor="w", padx=10, pady=(2, 2))
        self.reply_max_tokens_entry = ctk.CTkEntry(model_frame, textvariable=self.reply_max_tokens_var)
        self.reply_max_tokens_entry.pack(fill="x", padx=10, pady=(0, 8))

        ctk.CTkLabel(model_frame, text="Temperatur").pack(anchor="w", padx=10, pady=(2, 2))
        self.reply_temperature_entry = ctk.CTkEntry(model_frame, textvariable=self.reply_temperature_var)
        self.reply_temperature_entry.pack(fill="x", padx=10, pady=(0, 10))

        persona_frame = ctk.CTkFrame(sidebar)
        persona_frame.pack(fill="x", padx=8, pady=6)
        ctk.CTkLabel(persona_frame, text="Persona", font=(FONT_FAMILY, 14, "bold")).pack(anchor="w", padx=10, pady=(8, 2))

        def add_persona_slider(label: str, variable: ctk.DoubleVar, label_var: ctk.StringVar) -> None:
            row = ctk.CTkFrame(persona_frame, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=(3, 2))
            row.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(row, text=label).grid(row=0, column=0, sticky="w")
            ctk.CTkLabel(row, textvariable=label_var, width=34, anchor="e").grid(row=0, column=1, sticky="e")

            slider = ctk.CTkSlider(
                persona_frame,
                from_=0,
                to=100,
                number_of_steps=100,
                variable=variable,
                command=self._on_persona_slider_changed,
            )
            slider.pack(fill="x", padx=10, pady=(0, 6))

        add_persona_slider("Flirty", self.persona_flirty_var, self.persona_flirty_label_var)
        add_persona_slider("Humor/Sarkasmus", self.persona_humor_var, self.persona_humor_label_var)
        add_persona_slider("Ernsthaftigkeit", self.persona_serious_var, self.persona_serious_label_var)
        add_persona_slider("Dominanz", self.persona_dominance_var, self.persona_dominance_label_var)
        add_persona_slider("Empathie/Waerme", self.persona_empathy_var, self.persona_empathy_label_var)
        add_persona_slider("Temperament", self.persona_temperament_var, self.persona_temperament_label_var)

        ctk.CTkLabel(persona_frame, text="Lieblingslichtszene").pack(anchor="w", padx=10, pady=(2, 2))
        self.favorite_light_scene_entry = ctk.CTkEntry(persona_frame, textvariable=self.favorite_light_scene_var)
        self.favorite_light_scene_entry.pack(fill="x", padx=10, pady=(0, 8))

        self.save_profile_btn = ctk.CTkButton(
            persona_frame,
            text="Profil speichern",
            command=self.save_profile,
        )
        self.save_profile_btn.pack(fill="x", padx=10, pady=(0, 10))

        self._refresh_persona_labels()

        audio_frame = ctk.CTkFrame(sidebar)
        audio_frame.pack(fill="x", padx=8, pady=6)
        ctk.CTkLabel(audio_frame, text="Audio", font=(FONT_FAMILY, 14, "bold")).pack(anchor="w", padx=10, pady=(8, 2))

        ctk.CTkLabel(audio_frame, text="Mikrofon").pack(anchor="w", padx=10, pady=(4, 2))
        self.mic_menu = ctk.CTkOptionMenu(
            audio_frame,
            values=[NO_MIC_DEVICES_LABEL],
            variable=self.mic_device_var,
            command=self.on_mic_selection_changed,
        )
        self.mic_menu.pack(fill="x", padx=10, pady=(0, 6))

        self.refresh_mic_btn = ctk.CTkButton(audio_frame, text="🔄 Geräte aktualisieren", command=self.refresh_input_devices)
        self.refresh_mic_btn.pack(fill="x", padx=10, pady=(0, 6))

        ctk.CTkLabel(audio_frame, text="Sample Rate").pack(anchor="w", padx=10, pady=(2, 2))
        self.sample_rate_entry = ctk.CTkEntry(audio_frame, textvariable=self.sample_rate_var)
        self.sample_rate_entry.pack(fill="x", padx=10, pady=(0, 8))

        self.mic_level_bar = ctk.CTkProgressBar(audio_frame)
        self.mic_level_bar.pack(fill="x", padx=10, pady=(2, 4))
        self.mic_level_bar.set(0)

        self.mic_level_label = ctk.CTkLabel(audio_frame, textvariable=self.mic_level_text_var)
        self.mic_level_label.pack(anchor="w", padx=10, pady=(0, 10))

        tts_frame = ctk.CTkFrame(sidebar)
        tts_frame.pack(fill="x", padx=8, pady=6)
        ctk.CTkLabel(tts_frame, text="TTS", font=(FONT_FAMILY, 14, "bold")).pack(anchor="w", padx=10, pady=(8, 2))

        ctk.CTkLabel(tts_frame, text="TTS Engine").pack(anchor="w", padx=10, pady=(4, 2))
        tts_engines = ["edge-tts (natürlich)", "piper (lokal, natürlich)", "pyttsx3 (lokal)"]
        self.tts_engine_menu = ctk.CTkOptionMenu(
            tts_frame,
            values=tts_engines,
            variable=self.tts_engine_var,
            command=self.on_tts_engine_changed,
        )
        self.tts_engine_menu.pack(fill="x", padx=10, pady=(0, 8))

        ctk.CTkLabel(tts_frame, text="Stimme").pack(anchor="w", padx=10, pady=(4, 2))
        self.tts_voice_menu = ctk.CTkOptionMenu(
            tts_frame,
            values=list(EDGE_VOICE_OPTIONS.keys()),
            variable=self.tts_voice_var,
            command=self.on_tts_voice_changed,
        )
        self.tts_voice_menu.pack(fill="x", padx=10, pady=(0, 8))

        ctk.CTkLabel(tts_frame, text="Emotion").pack(anchor="w", padx=10, pady=(4, 2))
        self.tts_emotion_menu = ctk.CTkOptionMenu(
            tts_frame,
            values=list(EMOTION_PRESETS.keys()),
            variable=self.tts_emotion_var,
        )
        self.tts_emotion_menu.pack(fill="x", padx=10, pady=(0, 8))

        ctk.CTkLabel(tts_frame, text="TTS Rate").pack(anchor="w", padx=10, pady=(2, 2))
        self.tts_rate_entry = ctk.CTkEntry(tts_frame, textvariable=self.tts_rate_var)
        self.tts_rate_entry.pack(fill="x", padx=10, pady=(0, 8))

        ctk.CTkLabel(tts_frame, text="Piper Modell (.onnx)").pack(anchor="w", padx=10, pady=(2, 2))
        self.piper_model_entry = ctk.CTkEntry(tts_frame, textvariable=self.piper_model_path_var)
        self.piper_model_entry.pack(fill="x", padx=10, pady=(0, 8))

        ctk.CTkLabel(tts_frame, text="Piper Config (.json, optional)").pack(anchor="w", padx=10, pady=(2, 2))
        self.piper_config_entry = ctk.CTkEntry(tts_frame, textvariable=self.piper_config_path_var)
        self.piper_config_entry.pack(fill="x", padx=10, pady=(0, 4))

        ctk.CTkLabel(tts_frame, text="Hinweis: Piper braucht eine lokale .onnx Stimme", text_color="gray70").pack(
            anchor="w", padx=10, pady=(0, 10)
        )

        tabs = ctk.CTkTabview(self)
        tabs.grid(row=2, column=1, rowspan=3, columnspan=2, padx=(8, 16), pady=(0, 16), sticky="nsew")

        transcript_tab = tabs.add("Transkript")
        transcript_tab.grid_rowconfigure(0, weight=1)
        transcript_tab.grid_columnconfigure(0, weight=1)
        self.transcript_box = ctk.CTkTextbox(transcript_tab, wrap="word")
        self.transcript_box.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.transcript_box.insert("1.0", "Du kannst hier auch direkt Text eintippen und dann auf 'Text direkt senden' klicken.")

        answer_tab = tabs.add("Antwort")
        answer_tab.grid_rowconfigure(0, weight=1)
        answer_tab.grid_columnconfigure(0, weight=1)
        self.answer_box = ctk.CTkTextbox(answer_tab, wrap="word")
        self.answer_box.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

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

    def set_status(self, message: str) -> None:
        self.logger.info("Status: %s", message)
        self._track_event(f"Status: {message}")
        self.after(0, lambda: self.status_var.set(message))

    def set_textbox(self, textbox: ctk.CTkTextbox, content: str) -> None:
        def updater() -> None:
            textbox.delete("1.0", "end")
            textbox.insert("1.0", content)
            textbox.see("1.0")

        self.after(0, updater)

    def append_textbox(self, textbox: ctk.CTkTextbox, content: str) -> None:
        def updater() -> None:
            textbox.insert("end", content)
            textbox.see("end")

        self.after(0, updater)

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
                    started = time.perf_counter()
                    self.speak_text(item)
                    elapsed = time.perf_counter() - started
                    self._increment_counter("tts_chunks")
                    self._add_metric_sample("tts_chunk_seconds", elapsed)
                except Exception:
                    # Ignore per-chunk TTS failures so the rest of the response can continue.
                    self._increment_counter("errors")
                finally:
                    tts_queue.task_done()

            if self.current_tts_queue is tts_queue:
                self.current_tts_queue = None

        tts_thread = threading.Thread(target=worker, daemon=True)
        tts_thread.start()
        return tts_queue, tts_thread

    def cancel_current_response(self) -> None:
        self.cancel_ollama_event.set()
        self.cancel_tts_event.set()
        self._increment_counter("ollama_cancels")
        self.logger.warning("Aktive Antwort wurde abgebrochen")
        self._track_event("Antwortabbruch angefordert")

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

    def _get_pyttsx3_engine(self) -> Any:
        with self.pyttsx3_engine_lock:
            if self.pyttsx3_engine is None:
                self.pyttsx3_engine = pyttsx3.init()
            return self.pyttsx3_engine

    def add_history_entry(self, role: str, content: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logger.debug("History %s: %s", role, content[:140])

        def updater() -> None:
            self.history_box.insert("end", f"[{timestamp}] {role}:\n{content}\n\n")
            self.history_box.see("end")

        self.after(0, updater)

    def clear_history(self) -> None:
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

        def updater() -> None:
            self.mic_level_bar.set(clamped)
            self.mic_level_text_var.set(f"Pegel: {int(clamped * 100)}%")

        self.after(0, updater)

    def _monitor_callback(self, indata: Any, frames: int, callback_time: Any, status: Any) -> None:
        peak = float(abs(indata).max()) / 32767.0
        self.set_mic_level(peak)

    def stop_level_monitor(self) -> None:
        if self.monitor_stream is not None:
            self.monitor_stream.stop()
            self.monitor_stream.close()
            self.monitor_stream = None

    def start_level_monitor(self) -> None:
        if self.is_recording:
            return

        self.stop_level_monitor()

        selected_label = self.mic_device_var.get().strip()
        if selected_label not in self.mic_devices_map:
            self.set_mic_level(0)
            return

        try:
            sample_rate = int(self.sample_rate_var.get().strip())
        except ValueError:
            sample_rate = 16000

        input_device = self.get_selected_input_device()

        try:
            self.monitor_stream = sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
                device=input_device,
                callback=self._monitor_callback,
            )
            self.monitor_stream.start()
        except Exception:
            self.monitor_stream = None
            self.set_mic_level(0)

    def on_mic_selection_changed(self, _selected: str) -> None:
        if not self.is_recording:
            self.start_level_monitor()

    def _character_model_base_dir(self) -> Path:
        return Path(__file__).resolve().parent

    def _is_avatar_viewer_running(self) -> bool:
        return self.avatar_viewer_process is not None and self.avatar_viewer_process.poll() is None

    def _update_avatar_button_state(self) -> None:
        if not hasattr(self, "avatar_btn"):
            return

        if self._is_avatar_viewer_running():
            self.avatar_btn.configure(text="Avatar stoppen", fg_color="#991b1b", hover_color="#7f1d1d")
        else:
            self.avatar_btn.configure(text="Avatar starten", fg_color="#1d4ed8", hover_color="#1e40af")

    def _build_avatar_viewer_url(self, port: int, base_dir: Path) -> str:
        vrm_file = base_dir / DEFAULT_VRM_RELATIVE_PATH
        if not vrm_file.exists():
            raise FileNotFoundError(f"VRM-Datei nicht gefunden: {vrm_file}")

        rel_vrm = DEFAULT_VRM_RELATIVE_PATH.replace("\\", "/")
        vrm_url = f"http://127.0.0.1:{port}/{urllib.parse.quote(rel_vrm, safe='/')}"
        query = urllib.parse.urlencode({"vrm": vrm_url})
        return f"http://127.0.0.1:{port}/web/vrm_viewer.html?{query}"

    def _start_avatar_viewer(self) -> bool:
        if self._is_avatar_viewer_running():
            self._update_avatar_button_state()
            return True

        base_dir = self._character_model_base_dir()
        if not base_dir.exists():
            self.set_status(f"Avatar-Ordner fehlt: {base_dir}")
            return False

        try:
            if self.avatar_server_module is None:
                self.avatar_server_module = importlib.import_module("local_http_server")

            self.avatar_http_port = int(self.avatar_server_module.start(str(base_dir)))
            viewer_url = self._build_avatar_viewer_url(self.avatar_http_port, base_dir)
            viewer_script = base_dir / "viewer_process.py"
            self.avatar_viewer_process = subprocess.Popen(
                [sys.executable, str(viewer_script), viewer_url],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            self.avatar_auto_start_attempted = True
            self._update_avatar_button_state()
            self.set_status("Avatar-Viewer gestartet")
            self.logger.info("Avatar-Viewer gestartet (Port %s)", self.avatar_http_port)
            return True
        except Exception as exc:
            self._log_exception("Avatar-Viewer Start", exc)
            self.avatar_viewer_process = None
            self._update_avatar_button_state()
            self.set_status(f"Avatar-Viewer Start fehlgeschlagen: {exc}")
            return False

    def _stop_avatar_viewer(self) -> None:
        self._reset_avatar_lipsync()
        proc = self.avatar_viewer_process
        self.avatar_viewer_process = None
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except Exception:
                pass

        if self.avatar_server_module is not None:
            try:
                self.avatar_server_module.stop()
            except Exception:
                pass

        self.avatar_http_port = None
        self._update_avatar_button_state()
        self.set_status("Avatar-Viewer gestoppt")

    def toggle_avatar_viewer(self) -> None:
        if self._is_avatar_viewer_running():
            self._stop_avatar_viewer()
            return
        self._start_avatar_viewer()

    def _post_avatar_lipsync(self, active: bool, energy: float = 0.0, force: bool = False) -> None:
        if not self.avatar_lipsync_var.get():
            return

        if self.avatar_http_port is None or not self._is_avatar_viewer_running():
            return

        now = time.perf_counter()
        if not force and (now - self.avatar_last_push_at) < 0.045:
            return

        self.avatar_last_push_at = now
        try:
            self.http_session.post(
                f"http://127.0.0.1:{self.avatar_http_port}/api/lipsync",
                json={"active": bool(active), "energy": max(0.0, min(1.0, float(energy)))},
                timeout=(0.5, 0.5),
            )
        except Exception:
            # Do not interrupt voice playback when avatar bridge is unavailable.
            pass

    def _reset_avatar_lipsync(self) -> None:
        if self.avatar_http_port is None or not self._is_avatar_viewer_running():
            return
        try:
            self.http_session.post(
                f"http://127.0.0.1:{self.avatar_http_port}/api/lipsync",
                json={"active": False, "energy": 0.0},
                timeout=(0.5, 0.5),
            )
        except Exception:
            pass

    def _estimate_lipsync_energy(self, text: str, elapsed_seconds: float) -> float:
        vowel_count = sum(1 for ch in text.lower() if ch in "aeiouäöüy")
        density = vowel_count / max(1, len(text))
        text_factor = max(0.35, min(0.95, 0.45 + density * 2.6))
        pulse = 0.55 + 0.45 * abs(math.sin(elapsed_seconds * 11.5))
        return max(0.12, min(1.0, text_factor * pulse))

    def _start_avatar_lipsync_background(self, text: str) -> tuple[threading.Event, threading.Thread | None]:
        stop_event = threading.Event()

        if not self.avatar_lipsync_var.get() or self.avatar_http_port is None:
            return stop_event, None

        def worker() -> None:
            started_at = time.perf_counter()
            while not stop_event.is_set():
                elapsed = time.perf_counter() - started_at
                self._post_avatar_lipsync(True, self._estimate_lipsync_energy(text, elapsed))
                time.sleep(0.05)

            self._reset_avatar_lipsync()

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        return stop_event, thread

    def _ensure_avatar_for_lipsync(self) -> None:
        if not self.avatar_lipsync_var.get():
            return
        if self._is_avatar_viewer_running() and self.avatar_http_port is not None:
            return
        if self.avatar_auto_start_attempted:
            return
        self._start_avatar_viewer()

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

        self.cancel_current_response()
        self._stop_active_audio_playback()
        self._stop_avatar_viewer()
        self.stop_level_monitor()
        if self.recording_stream is not None:
            self.recording_stream.stop()
            self.recording_stream.close()
            self.recording_stream = None
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
            self.load_whisper_model(self.whisper_model_var.get().strip() or "small", announce=False)
            self.check_ollama(force_refresh=True)
        except Exception:
            # Warmup should never block or break normal interaction.
            pass

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

    def _audio_callback(self, indata: Any, frames: int, callback_time: Any, status: Any) -> None:
        if self.recording_wave is None:
            return

        peak = float(abs(indata).max()) / 32767.0
        self.set_mic_level(peak)
        self.recording_wave.writeframes(indata.copy().tobytes())

    def start_recording(self) -> None:
        if self.is_recording:
            return

        self.cancel_current_response()

        try:
            sample_rate = int(self.sample_rate_var.get().strip())
        except ValueError:
            self.set_status("Ungültige Sample Rate")
            return

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

        self.stop_level_monitor()

        try:
            self.recording_stream = sd.InputStream(
                samplerate=sample_rate,
                channels=1,
                dtype="int16",
                device=input_device,
                callback=self._audio_callback,
            )
            self.recording_stream.start()
        except Exception as exc:
            self.recording_stream = None
            self.recording_wave.close()
            self.recording_wave = None
            self.set_status(f"Mikrofon-Start fehlgeschlagen: {exc}")
            self.start_level_monitor()
            return

        self.is_recording = True
        self.recording_started_at = time.perf_counter()
        self._increment_counter("recordings_started")
        self.last_transcript = ""
        self.set_mic_level(0)
        self.set_status(f"Mikrofon läuft... ({self.mic_device_var.get()})")
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.transcribe_btn.configure(state="disabled")
        self.send_btn.configure(state="disabled")

    def stop_recording(self) -> None:
        if not self.is_recording:
            return

        if self.recording_stream is not None:
            self.recording_stream.stop()
            self.recording_stream.close()
            self.recording_stream = None

        if self.recording_wave is not None:
            self.recording_wave.close()
            self.recording_wave = None

        self.is_recording = False
        self._increment_counter("recordings_finished")
        if self.recording_started_at is not None:
            duration = time.perf_counter() - self.recording_started_at
            self._add_metric_sample("recording_seconds", duration)
            self.logger.info("Aufnahmedauer: %.2fs", duration)
            self.recording_started_at = None
        self.set_mic_level(0)
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.start_level_monitor()

        if self.auto_pipeline_var.get():
            self.set_status("Aufnahme beendet. Starte Auto-Workflow...")
            self.transcribe_btn.configure(state="disabled")
            self.send_btn.configure(state="disabled")
            self._start_transcription_worker(auto_send=True)
        else:
            self.set_status("Aufnahme beendet. Bereit zum Verarbeiten.")
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
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ]

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

    def speak_text(self, text: str) -> None:
        self._ensure_avatar_for_lipsync()
        selected_engine = self.tts_engine_var.get().strip().lower()
        if "edge-tts" in selected_engine:
            self.speak_text_edge_tts(text)
            return

        if "piper" in selected_engine:
            self.speak_text_piper(text)
            return

        self.speak_text_pyttsx3(text)

    def _speak_text_background(self, text: str, done_status: str = "Fertig") -> None:
        def worker() -> None:
            try:
                if self.cancel_tts_event.is_set():
                    return

                self.set_status("Antwort wird vorgelesen...")
                self.logger.info("TTS gestartet (%d Zeichen)", len(text))

                try:
                    self.speak_text(text)
                except Exception as first_exc:
                    # Fallback to local engine when network/engine specific TTS fails.
                    self.logger.warning("Primaere TTS fehlgeschlagen, fallback pyttsx3: %s", first_exc)
                    self.speak_text_pyttsx3(text)

                self.logger.info("TTS abgeschlossen")
            except Exception as exc:
                self._log_exception("TTS Hintergrund", exc)
            finally:
                if not self.cancel_tts_event.is_set():
                    self.set_status(done_status)

        tts_worker = threading.Thread(target=worker, daemon=True)
        tts_worker.start()

    def _resolve_piper_config_path(self, model_path: Path, configured_config_path: str) -> Path | None:
        if configured_config_path:
            config_path = Path(configured_config_path).expanduser()
            if not config_path.is_absolute():
                config_path = (Path(__file__).resolve().parent / config_path).resolve()
            return config_path if config_path.exists() else None

        model_str = str(model_path)
        candidates = [
            Path(f"{model_str}.json"),
            Path(model_str.replace(".onnx", ".onnx.json")),
            model_path.with_suffix(".json"),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _resolve_piper_model_path(self, configured_model_path: str) -> Path | None:
        project_root = Path(__file__).resolve().parent
        normalized_input = configured_model_path.strip()

        if normalized_input:
            candidate = Path(normalized_input).expanduser()
            if not candidate.is_absolute():
                candidate = (project_root / candidate).resolve()

            if candidate.is_file() and candidate.suffix.lower() == ".onnx":
                return candidate

            if candidate.is_dir():
                found = sorted(candidate.rglob(MODEL_GLOB_PATTERN))
                if found:
                    return found[0]

        fallback_dirs = [project_root / "piperVoices", project_root / "models"]
        for fallback_dir in fallback_dirs:
            if fallback_dir.exists():
                found = sorted(fallback_dir.rglob(MODEL_GLOB_PATTERN))
                if found:
                    return found[0]

        return None

    def speak_text_piper(self, text: str) -> None:
        try:
            pygame_module = importlib.import_module("pygame")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "pygame fehlt im aktiven Python. Bitte im gleichen Interpreter installieren: pip install pygame"
            ) from exc

        piper_executable = shutil.which("piper")
        python_cmd: list[str] | None = None

        try:
            importlib.import_module("piper")
            python_cmd = [sys.executable, "-m", "piper"]
        except ModuleNotFoundError:
            python_cmd = None

        if piper_executable is None and python_cmd is None:
            raise RuntimeError(
                "Piper nicht gefunden. Installiere piper-tts im gleichen Python wie die App oder waehle einen Interpreter, in dem piper verfuegbar ist."
            )

        configured_model_input = self.piper_model_path_var.get().strip()
        model_path = self._resolve_piper_model_path(configured_model_input)
        if model_path is None:
            raise RuntimeError(
                "Piper Modell nicht gefunden. Bitte gueltigen .onnx Pfad eintragen oder eine .onnx Datei in piperVoices/models ablegen."
            )

        if str(model_path) != configured_model_input:
            self.after(0, lambda: self.piper_model_path_var.set(str(model_path)))

        config_path = self._resolve_piper_config_path(model_path, self.piper_config_path_var.get().strip())
        if config_path is None:
            raise RuntimeError(
                "Piper Config fehlt. Neben der .onnx wird meist eine .onnx.json benoetigt. "
                f"Modell: {model_path}"
            )

        try:
            tts_rate = int(self.tts_rate_var.get().strip())
        except ValueError:
            tts_rate = 170

        safe_rate = max(80, min(300, tts_rate))
        length_scale = max(0.7, min(1.6, 170.0 / float(safe_rate)))

        fd, temp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

        cmd = [
            *(python_cmd if python_cmd is not None else [piper_executable]),
            "--model",
            str(model_path),
            "--output_file",
            temp_path,
            "--length_scale",
            f"{length_scale:.3f}",
        ]
        if config_path is not None:
            cmd.extend(["--config", str(config_path)])

        try:
            completed = subprocess.run(
                cmd,
                input=text,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                error_output = (completed.stderr or completed.stdout or "").strip()
                raise RuntimeError(
                    f"Piper fehlgeschlagen: {error_output or 'Unbekannter Fehler'} | Modell: {model_path} | Config: {config_path}"
                )

            if not pygame_module.mixer.get_init():
                pygame_module.mixer.init()
            pygame_module.mixer.music.load(temp_path)
            pygame_module.mixer.music.play()
            started_at = time.perf_counter()
            while pygame_module.mixer.music.get_busy():
                if self.cancel_tts_event.is_set():
                    pygame_module.mixer.music.stop()
                    break
                elapsed = time.perf_counter() - started_at
                self._post_avatar_lipsync(True, self._estimate_lipsync_energy(text, elapsed))
                time.sleep(0.05)
            pygame_module.mixer.music.stop()
            try:
                pygame_module.mixer.music.unload()
            except Exception:
                pass
            self._reset_avatar_lipsync()
        finally:
            safe_remove_file(temp_path)

    def speak_text_pyttsx3(self, text: str) -> None:
        if self.cancel_tts_event.is_set():
            return

        try:
            tts_rate = int(self.tts_rate_var.get().strip())
        except ValueError:
            tts_rate = 170

        engine = self._get_pyttsx3_engine()
        engine.setProperty("rate", tts_rate)

        selected_voice = self.tts_voice_var.get().strip().lower()
        voices = engine.getProperty("voices")
        want_female = "weiblich" in selected_voice
        want_male = "männlich" in selected_voice

        chosen_voice_id = None
        for voice in voices:
            voice_desc = f"{voice.id} {voice.name}".lower()
            if want_female and any(token in voice_desc for token in ["female", "weib", "zira", "heda"]):
                chosen_voice_id = voice.id
                break
            if want_male and any(token in voice_desc for token in ["male", "männ", "david", "mark"]):
                chosen_voice_id = voice.id
                break

        if chosen_voice_id:
            engine.setProperty("voice", chosen_voice_id)

        stop_lipsync_event, lipsync_thread = self._start_avatar_lipsync_background(text)
        engine.stop()
        try:
            engine.say(text)
            engine.runAndWait()
        finally:
            stop_lipsync_event.set()
            if lipsync_thread is not None:
                lipsync_thread.join(timeout=0.3)
            self._reset_avatar_lipsync()

    def speak_text_edge_tts(self, text: str) -> None:
        try:
            edge_tts_module = importlib.import_module("edge_tts")
            pygame_module = importlib.import_module("pygame")
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "edge-tts/pygame fehlen im aktiven Python. Bitte im gleichen Interpreter installieren: pip install edge-tts pygame"
            ) from exc

        voice_label = self.tts_voice_var.get().strip()
        voice = EDGE_VOICE_OPTIONS.get(voice_label, "de-DE-KatjaNeural")
        emotion = self.tts_emotion_var.get().strip().lower()
        rate, pitch = EMOTION_PRESETS.get(emotion, EMOTION_PRESETS["neutral"])

        async def synthesize(target_file: str) -> None:
            communicate = edge_tts_module.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
            await communicate.save(target_file)

        fd, temp_path = tempfile.mkstemp(suffix=".mp3")
        os.close(fd)

        try:
            asyncio.run(synthesize(temp_path))
            if not pygame_module.mixer.get_init():
                pygame_module.mixer.init()
            pygame_module.mixer.music.load(temp_path)
            pygame_module.mixer.music.play()
            started_at = time.perf_counter()
            while pygame_module.mixer.music.get_busy():
                if self.cancel_tts_event.is_set():
                    pygame_module.mixer.music.stop()
                    break
                elapsed = time.perf_counter() - started_at
                self._post_avatar_lipsync(True, self._estimate_lipsync_energy(text, elapsed))
                time.sleep(0.05)
            pygame_module.mixer.music.stop()
            try:
                pygame_module.mixer.music.unload()
            except Exception:
                pass
            self._reset_avatar_lipsync()
        finally:
            safe_remove_file(temp_path)

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
                    "beam_size": 1 if speed_mode == "Schnell" else 5,
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
            self.after(0, lambda: self.cancel_btn.configure(state="normal"))
            self._increment_counter("ollama_requests")
            self.logger.info("Ollama Anfrage gestartet (%d Zeichen Input)", len(transcript))

            self.set_status("Frage Ollama... (erster Lauf kann etwas dauern)")
            self.add_history_entry("Du", transcript)
            self.set_textbox(self.answer_box, "")

            tool_action = self._decide_tool_action_with_ollama(transcript)

            # Fallback for malformed router output: keep prototype responsive.
            if tool_action == "none":
                local_fallback = self._parse_light_command(transcript)
                if local_fallback == "on":
                    tool_action = "light_on"
                elif local_fallback == "off":
                    tool_action = "light_off"

            if tool_action in {"light_on", "light_off"}:
                self.after(0, self.open_light_test_popup)
                if tool_action == "light_on":
                    self.after(0, lambda: self.set_light_state(True, source="ollama-tool-router"))
                    answer = "Alles klar, ich habe das Licht fuer dich eingeschaltet."
                else:
                    self.after(0, lambda: self.set_light_state(False, source="ollama-tool-router"))
                    answer = "Alles klar, ich habe das Licht fuer dich ausgeschaltet."

                self.set_textbox(self.answer_box, answer)
                self.add_history_entry("Assistent", answer)

                if self.auto_speak_var.get() and not self.cancel_tts_event.is_set():
                    self._speak_text_background(answer, done_status="Lichtbefehl ueber Ollama-Toolrouting ausgefuehrt")
                else:
                    self.set_status("Lichtbefehl ueber Ollama-Toolrouting ausgefuehrt")

                return

            chunk_buffer: list[str] = []
            first_token_received = False
            first_token_time: float | None = None
            tts_queue: queue.Queue[str | None] | None = None
            tts_text_buffer = ""
            tts_phrase_buffer = ""
            tts_phrase_count = 0

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

                if self.cancel_ollama_event.is_set():
                    return

                chunk_buffer.append(chunk)
                if not first_token_received:
                    first_token_received = True
                    first_token_time = time.perf_counter()
                    self.set_status("Ollama antwortet...")

                if tts_queue is not None:
                    tts_text_buffer += chunk
                    ready_sentences, tts_text_buffer = self._extract_complete_sentences(tts_text_buffer)
                    for sentence in ready_sentences:
                        normalized_sentence = sentence.strip()
                        if not normalized_sentence:
                            continue

                        if tts_phrase_buffer:
                            tts_phrase_buffer = f"{tts_phrase_buffer} {normalized_sentence}"
                        else:
                            tts_phrase_buffer = normalized_sentence
                        tts_phrase_count += 1

                        should_emit_phrase = (
                            len(tts_phrase_buffer) >= TTS_STREAM_MIN_CHARS
                            or tts_phrase_count >= TTS_STREAM_MAX_SENTENCES
                            or normalized_sentence.endswith("\n")
                        )
                        if should_emit_phrase:
                            tts_queue.put(tts_phrase_buffer)
                            tts_phrase_buffer = ""
                            tts_phrase_count = 0

                should_flush = len(chunk_buffer) >= 8 or chunk.endswith((".", "!", "?", "\n"))
                if should_flush:
                    flush_chunks()

            answer = self.ask_ollama(
                transcript,
                on_chunk=on_chunk,
                cancel_event=self.cancel_ollama_event,
            )
            flush_chunks()

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
                return

            self.set_status("Fertig")
        except Exception as exc:
            if self.cancel_ollama_event.is_set():
                self.set_status(DEFAULT_ABORTED_STATUS)
                return
            self._log_exception("Ollama", exc)
            self.set_status(f"Fehler: {exc}")
        finally:
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
