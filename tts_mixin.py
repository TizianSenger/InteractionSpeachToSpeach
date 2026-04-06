"""TTS synthesis methods – mixed into VoiceAssistantUI via multiple inheritance."""
from __future__ import annotations

import asyncio
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import pyttsx3

from constants import EDGE_VOICE_OPTIONS, EMOTION_PRESETS, MODEL_GLOB_PATTERN, PYTTSX3_VOICE_OPTIONS


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


class TtsMixin:
    """Provides all TTS synthesis backends (edge-tts, piper, pyttsx3)."""

    def _get_pyttsx3_engine(self) -> Any:
        with self.pyttsx3_engine_lock:
            if self.pyttsx3_engine is None:
                self.pyttsx3_engine = pyttsx3.init()
            return self.pyttsx3_engine

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
