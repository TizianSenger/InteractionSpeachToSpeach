"""Wake-word detection using openwakeword (ONNX, CPU-only, no training required).

Architecture: no separate audio stream or thread.
The unified background stream in voice_ui.py calls _ww_feed_audio() on every
80 ms block.  The OWW model runs inline in that callback - negligible latency,
zero gap between wake-word and recording start.

Install:  pip install openwakeword
"""
from __future__ import annotations

import importlib
import threading
from typing import Any

import numpy as np

_DEFAULT_THRESHOLD = 0.5

# Display label -> internal model name used by openwakeword
OWW_MODELS: dict[str, str] = {
    "Hey Jarvis":  "hey_jarvis",
    "Alexa":       "alexa",
    "Hey Mycroft": "hey_mycroft",
    "Ok Nabu":     "ok_nabu",
    "Timer":       "timer",
    "Wetter":      "weather",
}
OWW_MODEL_DISPLAY_NAMES = list(OWW_MODELS.keys())


class WakeWordMixin:
    """Mixin for VoiceAssistantUI: always-on wake-word via openwakeword.

    Expected attributes from host:
        wake_word_enabled_var  - ctk.BooleanVar
        wake_word_model_var    - ctk.StringVar
        wake_word_status_var   - ctk.StringVar
        is_recording           - bool
        logger                 - logging.Logger
        after()                - Tk after()
        start_recording()      - starts recording pipeline
    """

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def start_wake_word_listener(self) -> None:
        """Load the OWW model in a background thread; enable detection."""
        if not self.wake_word_enabled_var.get():
            return
        self._oww_model: Any | None = None
        self._oww_cooldown_until: float = 0.0
        self.wake_word_status_var.set("Lade Modell ...")
        t = threading.Thread(target=self._ww_load_model_bg, daemon=True, name="OWW-Loader")
        t.start()

    def stop_wake_word_listener(self) -> None:
        """Disable detection (model stays cached for fast re-enable)."""
        self._oww_model = None
        self.wake_word_status_var.set("Inaktiv")

    def on_wake_word_toggle(self) -> None:
        if self.wake_word_enabled_var.get():
            self.start_wake_word_listener()
        else:
            self.stop_wake_word_listener()

    # ------------------------------------------------------------------ #
    #  Called from the unified bg_audio_callback (voice_ui.py)            #
    # ------------------------------------------------------------------ #

    def _ww_feed_audio(self, audio_float32: np.ndarray) -> None:
        """Process one 80 ms float32 block. Called from sounddevice callback."""
        import time
        model = getattr(self, "_oww_model", None)
        if model is None:
            return
        if not self.wake_word_enabled_var.get():
            return
        if time.monotonic() < getattr(self, "_oww_cooldown_until", 0.0):
            return

        audio_int16 = (audio_float32 * 32767.0).astype(np.int16)
        try:
            predictions = model.predict(audio_int16)
        except Exception:
            return

        score = max(predictions.values(), default=0.0)
        if score >= _DEFAULT_THRESHOLD:
            display = self.wake_word_model_var.get()
            self.logger.info("Wake-word erkannt: >>%s<< score=%.3f", display, score)
            import time as _t
            self._oww_cooldown_until = _t.monotonic() + 1.5
            model.reset()
            self.after(0, self._ww_trigger)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _ww_load_model_bg(self) -> None:
        """Background thread: load OWW model, then signal ready."""
        try:
            oww_utils = importlib.import_module("openwakeword.utils")
            oww_model_mod = importlib.import_module("openwakeword.model")
        except ModuleNotFoundError:
            self.logger.error("Wake-word: openwakeword nicht installiert")
            self.after(0, lambda: self.wake_word_status_var.set("Fehler: nicht installiert"))
            return

        try:
            oww_utils.download_models()
        except Exception as exc:
            self.logger.warning("Wake-word: Download-Fehler (ignoriert): %s", exc)

        display_name = self.wake_word_model_var.get()
        internal_name = OWW_MODELS.get(display_name, "hey_jarvis")
        try:
            model = oww_model_mod.Model(
                wakeword_models=[internal_name],
                inference_framework="onnx",
            )
            self._oww_model = model
            self._oww_cooldown_until = 0.0
            self.logger.info("Wake-word: >>%s<< bereit", internal_name)
            self.after(0, lambda: self.wake_word_status_var.set("Hoere zu ..."))
        except Exception as exc:
            self.logger.error("Wake-word: Ladefehler: %s", exc)
            self.after(0, lambda: self.wake_word_status_var.set(f"Fehler: {exc}"))

    def _ww_trigger(self) -> None:
        """Fired on the UI thread when wake word detected."""
        if self.is_recording:
            return
        self.wake_word_status_var.set("Aktiviert!")
        # Visual flash on the waveform canvas
        if hasattr(self, "trigger_waveform_flash"):
            self.trigger_waveform_flash()
        self.start_recording()
        self.after(1500, lambda: self.wake_word_status_var.set("Hoere zu ..."))
