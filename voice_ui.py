import asyncio
from datetime import datetime
import importlib
import json
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
TTS_STREAM_MIN_CHARS = 120
TTS_STREAM_MAX_SENTENCES = 2

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

        self.status_var = ctk.StringVar(value="Bereit")
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

        self._build_layout()
        self.refresh_piper_model_options()
        self.on_tts_engine_changed(self.tts_engine_var.get())
        self.refresh_input_devices()
        self.start_level_monitor()
        self._start_background_warmup()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

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
        for i in range(10):
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

        self.auto_pipeline_switch = ctk.CTkSwitch(
            workflow,
            text="Auto: Stop -> Transkribieren -> Senden",
            variable=self.auto_pipeline_var,
        )
        self.auto_pipeline_switch.grid(row=0, column=8, columnspan=2, padx=6, pady=10, sticky="w")

        sidebar = ctk.CTkScrollableFrame(self, label_text="Einstellungen")
        sidebar.grid(row=2, column=0, rowspan=3, padx=(16, 8), pady=(0, 16), sticky="nsew")

        stt_frame = ctk.CTkFrame(sidebar)
        stt_frame.pack(fill="x", padx=8, pady=(8, 6))
        ctk.CTkLabel(stt_frame, text="STT", font=(FONT_FAMILY, 14, "bold")).pack(anchor="w", padx=10, pady=(8, 2))

        ctk.CTkLabel(stt_frame, text="Whisper Modell").pack(anchor="w", padx=10, pady=(4, 2))
        self.whisper_menu = ctk.CTkOptionMenu(
            stt_frame,
            values=["tiny", "base", "small", "medium", "large"],
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

        transcript_frame = ctk.CTkFrame(self)
        transcript_frame.grid(row=2, column=1, columnspan=2, padx=(8, 16), pady=(0, 8), sticky="nsew")
        transcript_frame.grid_rowconfigure(1, weight=1)
        transcript_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(transcript_frame, text="Transkript", font=(FONT_FAMILY, 16, "bold")).grid(
            row=0, column=0, padx=12, pady=(10, 6), sticky="w"
        )
        self.transcript_box = ctk.CTkTextbox(transcript_frame, wrap="word")
        self.transcript_box.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")
        self.transcript_box.insert("1.0", "Du kannst hier auch direkt Text eintippen und dann auf 'Text direkt senden' klicken.")

        answer_frame = ctk.CTkFrame(self)
        answer_frame.grid(row=3, column=1, columnspan=2, padx=(8, 16), pady=(0, 8), sticky="nsew")
        answer_frame.grid_rowconfigure(1, weight=1)
        answer_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(answer_frame, text="Antwort", font=(FONT_FAMILY, 16, "bold")).grid(
            row=0, column=0, padx=12, pady=(10, 6), sticky="w"
        )
        self.answer_box = ctk.CTkTextbox(answer_frame, wrap="word")
        self.answer_box.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")

        history_frame = ctk.CTkFrame(self)
        history_frame.grid(row=4, column=1, columnspan=2, padx=(8, 16), pady=(0, 16), sticky="nsew")
        history_frame.grid_rowconfigure(1, weight=1)
        history_frame.grid_columnconfigure(0, weight=1)

        history_header = ctk.CTkFrame(history_frame, fg_color="transparent")
        history_header.grid(row=0, column=0, padx=12, pady=(8, 4), sticky="ew")
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

        self.history_box = ctk.CTkTextbox(history_frame, wrap="word")
        self.history_box.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")
        self.history_box.insert("1.0", "Der Verlauf wird hier mit Zeitstempel angezeigt.\n")

    def set_status(self, message: str) -> None:
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
                    self.speak_text(item)
                except Exception:
                    # Ignore per-chunk TTS failures so the rest of the response can continue.
                    pass
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

        self.set_status("Antwort abgebrochen")
        self.after(0, lambda: self.cancel_btn.configure(state="disabled"))
        self.after(0, lambda: self.send_btn.configure(state="normal"))
        self.after(0, lambda: self.transcribe_btn.configure(state="normal"))
        self.after(0, lambda: self.text_send_btn.configure(state="normal"))

    def _stop_active_audio_playback(self) -> None:
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

        def updater() -> None:
            self.history_box.insert("end", f"[{timestamp}] {role}:\n{content}\n\n")
            self.history_box.see("end")

        self.after(0, updater)

    def clear_history(self) -> None:
        self.history_box.delete("1.0", "end")

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
                for model_path in sorted(directory.rglob("*.onnx")):
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

    def _monitor_callback(self, indata: Any, frames: int, time: Any, status: Any) -> None:
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

    def on_close(self) -> None:
        self.cancel_current_response()
        self._stop_active_audio_playback()
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

    def on_whisper_model_changed(self, selected_model: str) -> None:
        if selected_model in self.whisper_model_cache:
            return

        worker = threading.Thread(
            target=self.load_whisper_model,
            kwargs={"model_name": selected_model, "announce": False},
            daemon=True,
        )
        worker.start()

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

    def _audio_callback(self, indata: Any, frames: int, time: Any, status: Any) -> None:
        if self.recording_wave is None:
            return

        peak = float(abs(indata).max()) / 32767.0
        self.set_mic_level(peak)
        self.recording_wave.writeframes(indata.copy().tobytes())

    def start_recording(self) -> None:
        if self.is_recording:
            return

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
            self.whisper_module = importlib.import_module("whisper")

        if model_name not in self.whisper_model_cache:
            if announce:
                self.set_status(f"Lade Whisper-Modell: {model_name}")
            self.whisper_model_cache[model_name] = self.whisper_module.load_model(model_name)
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

    def ask_ollama(
        self,
        user_text: str,
        on_chunk: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> str:
        model_name, ollama_url = self.check_ollama()

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": user_text}],
            "stream": True,
            "keep_alive": "30m",
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
        selected_engine = self.tts_engine_var.get().strip().lower()
        if "edge-tts" in selected_engine:
            self.speak_text_edge_tts(text)
            return

        if "piper" in selected_engine:
            self.speak_text_piper(text)
            return

        self.speak_text_pyttsx3(text)

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
                found = sorted(candidate.rglob("*.onnx"))
                if found:
                    return found[0]

        fallback_dirs = [project_root / "piperVoices", project_root / "models"]
        for fallback_dir in fallback_dirs:
            if fallback_dir.exists():
                found = sorted(fallback_dir.rglob("*.onnx"))
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
            while pygame_module.mixer.music.get_busy():
                if self.cancel_tts_event.is_set():
                    pygame_module.mixer.music.stop()
                    break
                time.sleep(0.05)
            pygame_module.mixer.music.stop()
            try:
                pygame_module.mixer.music.unload()
            except Exception:
                pass
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

        engine.stop()
        engine.say(text)
        engine.runAndWait()

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
            while pygame_module.mixer.music.get_busy():
                if self.cancel_tts_event.is_set():
                    pygame_module.mixer.music.stop()
                    break
                time.sleep(0.05)
            pygame_module.mixer.music.stop()
            try:
                pygame_module.mixer.music.unload()
            except Exception:
                pass
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
            use_fp16 = str(model.device).lower().startswith("cuda")
            language_label = self.whisper_language_var.get().strip()
            language_code = WHISPER_LANGUAGE_OPTIONS.get(language_label, "")
            speed_mode = self.whisper_speed_var.get().strip()

            transcribe_options: dict[str, Any] = {
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
            self.set_textbox(self.transcript_box, transcript)
            if auto_send:
                self.set_status("Transkript erkannt. Sende an Ollama...")
                self._run_ollama_only(transcript, manage_buttons=False)
            else:
                self.set_status("Transkript erkannt")
                self.after(0, lambda: self.send_btn.configure(state="normal"))
        except Exception as exc:
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
        try:
            self.cancel_ollama_event.clear()
            self.cancel_tts_event.clear()
            self.after(0, lambda: self.cancel_btn.configure(state="normal"))

            self.set_status("Frage Ollama... (erster Lauf kann etwas dauern)")
            self.add_history_entry("Du", transcript)
            self.set_textbox(self.answer_box, "")

            chunk_buffer: list[str] = []
            first_token_received = False
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
                nonlocal tts_text_buffer
                nonlocal tts_phrase_buffer
                nonlocal tts_phrase_count

                if self.cancel_ollama_event.is_set():
                    return

                chunk_buffer.append(chunk)
                if not first_token_received:
                    first_token_received = True
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

            if tts_queue is not None:
                if tts_phrase_buffer.strip():
                    tts_queue.put(tts_phrase_buffer.strip())
                trailing = tts_text_buffer.strip()
                if trailing:
                    tts_queue.put(trailing)
                tts_queue.put(None)

            if self.cancel_ollama_event.is_set():
                self.set_status("Antwort abgebrochen")
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
                self.set_status("Antwort abgebrochen")
                return
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
