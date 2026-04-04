import argparse
import os
import shutil
import tempfile
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pyttsx3
import requests
import sounddevice as sd
import whisper


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


def transcribe_audio(audio_path: str, model_name: str) -> str:
    if not ensure_ffmpeg_available():
        raise SystemExit(
            "ffmpeg nicht gefunden. Bitte ffmpeg installieren oder Terminal/VS Code neu starten."
        )
    model = whisper.load_model(model_name)
    use_fp16 = str(model.device).lower().startswith("cuda")
    result = model.transcribe(audio_path, fp16=use_fp16)
    return result["text"].strip()


def ask_ollama(user_text: str, model_name: str, ollama_url: str) -> str:
    try:
        tags_resp = requests.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=5)
        tags_resp.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(
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
            raise SystemExit(
                f"Modell '{model_name}' nicht gefunden. Bitte zuerst: ollama pull {model_name}"
            )
    except ValueError as exc:
        raise SystemExit("Ungültige Antwort von Ollama /api/tags") from exc

    payload: dict[str, Any] = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": user_text,
            }
        ],
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


def speak_text(text: str, rate: int) -> None:
    engine = pyttsx3.init()
    engine.setProperty("rate", rate)
    engine.say(text)
    engine.runAndWait()


def record_microphone(seconds: float, sample_rate: int) -> str:
    print(f"Starte Mikrofonaufnahme für {seconds} Sekunden...")
    recording = sd.rec(
        int(seconds * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="int16",
    )
    sd.wait()
    print("Aufnahme beendet.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_path = temp_file.name

    mono_audio = np.squeeze(recording)
    with wave.open(temp_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(mono_audio.tobytes())

    return temp_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Whisper (STT) -> Ollama (phi4-mini)"
    )
    parser.add_argument(
        "--audio",
        help="Pfad zur Audiodatei (z. B. aufnahme.wav)",
    )
    parser.add_argument(
        "--whisper-model",
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper-Modellgröße (Standard: small)",
    )
    parser.add_argument(
        "--ollama-model",
        default="phi4-mini",
        help="Lokales Ollama-Modell (Standard: phi4-mini)",
    )
    parser.add_argument(
        "--ollama-url",
        default="http://localhost:11434",
        help="Ollama Server URL (Standard: http://localhost:11434)",
    )
    parser.add_argument(
        "--tts-rate",
        type=int,
        default=170,
        help="Sprechgeschwindigkeit für pyttsx3 (Standard: 170)",
    )
    parser.add_argument(
        "--mic",
        action="store_true",
        help="Audio direkt vom Mikrofon aufnehmen",
    )
    parser.add_argument(
        "--mic-seconds",
        type=float,
        default=6.0,
        help="Aufnahmedauer in Sekunden für Mikrofonmodus (Standard: 6)",
    )
    parser.add_argument(
        "--mic-sample-rate",
        type=int,
        default=16000,
        help="Sample-Rate für Mikrofonaufnahme (Standard: 16000)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    audio_path = args.audio

    should_use_mic = args.mic or not audio_path
    if should_use_mic:
        audio_path = record_microphone(args.mic_seconds, args.mic_sample_rate)
    elif not Path(audio_path).exists():
        raise SystemExit(f"Audio-Datei nicht gefunden: {audio_path}")

    text = transcribe_audio(audio_path, args.whisper_model)
    answer = ask_ollama(text, args.ollama_model, args.ollama_url)
    print("Transkript:")
    print(text)
    print("\nAntwort:")
    print(answer)
    speak_text(answer, args.tts_rate)


if __name__ == "__main__":
    main()
