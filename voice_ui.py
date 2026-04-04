import os
import shutil
import tempfile
import threading
import wave
from pathlib import Path
from typing import Any

import customtkinter as ctk
import pyttsx3
import requests
import sounddevice as sd
import whisper

FONT_FAMILY = "Segoe UI"
NO_MIC_DEVICES_LABEL = "Keine Geräte gefunden"
WINDOWS_DEFAULT_MIC_LABEL = "Standard (Windows)"


def ensure_ffmpeg_available() -> bool:
    if shutil.which("ffmpeg"):
        return True

    packages_root = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    if not packages_root.exists():
        return False

    candidates = list(packages_root.rglob("ffmpeg.exe"))
    if not candidates:
        return False

    ffmpeg_dir = str(candidates[0].parent)
    current_path = os.environ.get("PATH", "")
    if ffmpeg_dir not in current_path:
        os.environ["PATH"] = f"{ffmpeg_dir};{current_path}" if current_path else ffmpeg_dir

    return shutil.which("ffmpeg") is not None


class VoiceAssistantUI(ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.title("Local Voice Assistant")
        self.geometry("980x700")

        self.recording_stream: sd.InputStream | None = None
        self.monitor_stream: sd.InputStream | None = None
        self.recording_wave: wave.Wave_write | None = None
        self.recording_path: str | None = None
        self.is_recording = False
        self.whisper_model_cache: dict[str, Any] = {}
        self.mic_devices_map: dict[str, int] = {}

        self.status_var = ctk.StringVar(value="Bereit")
        self.whisper_model_var = ctk.StringVar(value="small")
        self.ollama_model_var = ctk.StringVar(value="phi4-mini")
        self.ollama_url_var = ctk.StringVar(value="http://localhost:11434")
        self.mic_device_var = ctk.StringVar(value="")
        self.sample_rate_var = ctk.StringVar(value="16000")
        self.tts_rate_var = ctk.StringVar(value="170")
        self.mic_level_text_var = ctk.StringVar(value="Pegel: 0%")
        self.auto_speak_var = ctk.BooleanVar(value=True)

        self._build_layout()
        self.refresh_input_devices()
        self.start_level_monitor()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_layout(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        self.grid_rowconfigure(3, weight=1)

        header = ctk.CTkFrame(self)
        header.grid(row=0, column=0, padx=16, pady=(16, 10), sticky="ew")
        header.grid_columnconfigure(1, weight=1)

        title = ctk.CTkLabel(header, text="Whisper → Ollama → TTS", font=(FONT_FAMILY, 20, "bold"))
        title.grid(row=0, column=0, padx=12, pady=12, sticky="w")

        status = ctk.CTkLabel(header, textvariable=self.status_var, font=(FONT_FAMILY, 14))
        status.grid(row=0, column=1, padx=12, pady=12, sticky="e")

        controls = ctk.CTkFrame(self)
        controls.grid(row=1, column=0, padx=16, pady=(0, 10), sticky="ew")
        for i in range(8):
            controls.grid_columnconfigure(i, weight=1)

        self.start_btn = ctk.CTkButton(controls, text="🎙️ Mic Start", command=self.start_recording)
        self.start_btn.grid(row=0, column=0, padx=6, pady=10, sticky="ew")

        self.stop_btn = ctk.CTkButton(controls, text="⏹ Mic Stop", command=self.stop_recording, state="disabled")
        self.stop_btn.grid(row=0, column=1, padx=6, pady=10, sticky="ew")

        self.process_btn = ctk.CTkButton(controls, text="▶️ Verarbeiten", command=self.process_recording, state="disabled")
        self.process_btn.grid(row=0, column=2, padx=6, pady=10, sticky="ew")

        self.test_btn = ctk.CTkButton(controls, text="🔎 Ollama Test", command=self.test_ollama)
        self.test_btn.grid(row=0, column=3, padx=6, pady=10, sticky="ew")

        self.speak_switch = ctk.CTkSwitch(controls, text="Antwort vorlesen", variable=self.auto_speak_var)
        self.speak_switch.grid(row=0, column=4, padx=6, pady=10, sticky="w")

        ctk.CTkLabel(controls, text="Whisper").grid(row=1, column=0, padx=6, pady=6, sticky="w")
        self.whisper_menu = ctk.CTkOptionMenu(
            controls,
            values=["tiny", "base", "small", "medium", "large"],
            variable=self.whisper_model_var,
        )
        self.whisper_menu.grid(row=1, column=1, padx=6, pady=6, sticky="ew")

        ctk.CTkLabel(controls, text="Ollama Modell").grid(row=1, column=2, padx=6, pady=6, sticky="w")
        self.ollama_model_entry = ctk.CTkEntry(controls, textvariable=self.ollama_model_var)
        self.ollama_model_entry.grid(row=1, column=3, padx=6, pady=6, sticky="ew")

        ctk.CTkLabel(controls, text="Ollama URL").grid(row=1, column=4, padx=6, pady=6, sticky="w")
        self.ollama_url_entry = ctk.CTkEntry(controls, textvariable=self.ollama_url_var)
        self.ollama_url_entry.grid(row=1, column=5, padx=6, pady=6, sticky="ew")

        ctk.CTkLabel(controls, text="Sample Rate").grid(row=1, column=6, padx=6, pady=6, sticky="w")
        self.sample_rate_entry = ctk.CTkEntry(controls, textvariable=self.sample_rate_var)
        self.sample_rate_entry.grid(row=1, column=7, padx=6, pady=6, sticky="ew")

        ctk.CTkLabel(controls, text="TTS Rate").grid(row=2, column=6, padx=6, pady=6, sticky="w")
        self.tts_rate_entry = ctk.CTkEntry(controls, textvariable=self.tts_rate_var)
        self.tts_rate_entry.grid(row=2, column=7, padx=6, pady=6, sticky="ew")

        ctk.CTkLabel(controls, text="Mikrofon").grid(row=2, column=0, padx=6, pady=6, sticky="w")
        self.mic_menu = ctk.CTkOptionMenu(
            controls,
            values=[NO_MIC_DEVICES_LABEL],
            variable=self.mic_device_var,
            command=self.on_mic_selection_changed,
        )
        self.mic_menu.grid(row=2, column=1, columnspan=2, padx=6, pady=6, sticky="ew")

        self.refresh_mic_btn = ctk.CTkButton(controls, text="🔄 Geräte", command=self.refresh_input_devices)
        self.refresh_mic_btn.grid(row=2, column=3, padx=6, pady=6, sticky="ew")

        self.mic_level_bar = ctk.CTkProgressBar(controls)
        self.mic_level_bar.grid(row=2, column=4, columnspan=2, padx=6, pady=6, sticky="ew")
        self.mic_level_bar.set(0)

        self.mic_level_label = ctk.CTkLabel(controls, textvariable=self.mic_level_text_var)
        self.mic_level_label.grid(row=2, column=6, padx=6, pady=6, sticky="w")

        transcript_frame = ctk.CTkFrame(self)
        transcript_frame.grid(row=2, column=0, padx=16, pady=(0, 10), sticky="nsew")
        transcript_frame.grid_rowconfigure(1, weight=1)
        transcript_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(transcript_frame, text="Transkript", font=(FONT_FAMILY, 16, "bold")).grid(
            row=0, column=0, padx=12, pady=(10, 6), sticky="w"
        )
        self.transcript_box = ctk.CTkTextbox(transcript_frame, wrap="word")
        self.transcript_box.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")

        answer_frame = ctk.CTkFrame(self)
        answer_frame.grid(row=3, column=0, padx=16, pady=(0, 16), sticky="nsew")
        answer_frame.grid_rowconfigure(1, weight=1)
        answer_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(answer_frame, text="Antwort", font=(FONT_FAMILY, 16, "bold")).grid(
            row=0, column=0, padx=12, pady=(10, 6), sticky="w"
        )
        self.answer_box = ctk.CTkTextbox(answer_frame, wrap="word")
        self.answer_box.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")

    def set_status(self, message: str) -> None:
        self.after(0, lambda: self.status_var.set(message))

    def set_textbox(self, textbox: ctk.CTkTextbox, content: str) -> None:
        def updater() -> None:
            textbox.delete("1.0", "end")
            textbox.insert("1.0", content)
            textbox.see("1.0")

        self.after(0, updater)

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
        self.stop_level_monitor()
        if self.recording_stream is not None:
            self.recording_stream.stop()
            self.recording_stream.close()
            self.recording_stream = None
        if self.recording_wave is not None:
            self.recording_wave.close()
            self.recording_wave = None
        self.destroy()

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
        self.set_mic_level(0)
        self.set_status(f"Mikrofon läuft... ({self.mic_device_var.get()})")
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.process_btn.configure(state="disabled")

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
        self.set_status("Aufnahme beendet. Bereit zum Verarbeiten.")
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.process_btn.configure(state="normal")
        self.start_level_monitor()

    def load_whisper_model(self, model_name: str) -> Any:
        if model_name not in self.whisper_model_cache:
            self.set_status(f"Lade Whisper-Modell: {model_name}")
            self.whisper_model_cache[model_name] = whisper.load_model(model_name)
        return self.whisper_model_cache[model_name]

    def check_ollama(self) -> tuple[str, str]:
        model_name = self.ollama_model_var.get().strip() or "phi4-mini"
        ollama_url = self.ollama_url_var.get().strip() or "http://localhost:11434"
        try:
            tags_resp = requests.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=5)
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

        return model_name, ollama_url

    def ask_ollama(self, user_text: str) -> str:
        model_name, ollama_url = self.check_ollama()

        payload = {
            "model": model_name,
            "messages": [{"role": "user", "content": user_text}],
            "stream": False,
        }
        response = requests.post(
            f"{ollama_url.rstrip('/')}/api/chat",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"].strip()

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
        try:
            tts_rate = int(self.tts_rate_var.get().strip())
        except ValueError:
            tts_rate = 170
        engine = pyttsx3.init()
        engine.setProperty("rate", tts_rate)
        engine.say(text)
        engine.runAndWait()

    def process_recording(self) -> None:
        if self.is_recording:
            self.set_status("Bitte zuerst Aufnahme stoppen.")
            return

        if not self.recording_path or not Path(self.recording_path).exists():
            self.set_status("Keine Aufnahme gefunden.")
            return

        self.process_btn.configure(state="disabled")
        worker = threading.Thread(target=self._run_pipeline, daemon=True)
        worker.start()

    def _run_pipeline(self) -> None:
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
            result = model.transcribe(audio_path, fp16=use_fp16)
            transcript = str(result.get("text", "")).strip()
            if not transcript:
                self.set_textbox(self.transcript_box, "[Kein Text erkannt. Bitte deutlicher/länger sprechen und erneut versuchen.]")
                self.set_textbox(self.answer_box, "")
                self.set_status("Kein Text erkannt")
                return

            self.set_textbox(self.transcript_box, transcript)
            self.set_status("Transkript erkannt")

            self.set_status("Frage Ollama... (erster Lauf kann etwas dauern)")
            answer = self.ask_ollama(transcript)
            self.set_textbox(self.answer_box, answer)

            if self.auto_speak_var.get():
                self.set_status("Lese Antwort vor...")
                self.speak_text(answer)

            self.set_status("Fertig")
        except Exception as exc:
            self.set_status(f"Fehler: {exc}")
        finally:
            self.after(0, lambda: self.process_btn.configure(state="normal"))


def main() -> None:
    app = VoiceAssistantUI()
    app.mainloop()


if __name__ == "__main__":
    main()
