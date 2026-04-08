"""Shared constants for the Voice Studio application."""
from __future__ import annotations

FONT_FAMILY = "Segoe UI"
NO_MIC_DEVICES_LABEL = "Keine Geräte gefunden"
WINDOWS_DEFAULT_MIC_LABEL = "Standard (Windows)"
NO_PIPER_MODELS_LABEL = "Keine Piper-Stimmen gefunden"
OLLAMA_MODEL_OPTIONS = ["phi4-mini", "dolphin-mistral"]
LLM_PROVIDER_OPTIONS = ["Ollama", "OpenAI", "Azure OpenAI", "Anthropic", "Groq"]
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
AZURE_OPENAI_DEFAULT_API_VERSION = "2024-10-21"
GROQ_DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_DEFAULT_API_VERSION = "2023-06-01"
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
REALTIME_MODE_OPTIONS = ["Balanced", "Aggressiv", "Stabil"]
TTS_STREAM_MIN_CHARS = 40
TTS_STREAM_FIRST_CHARS = 24
TTS_STREAM_MAX_BUFFER_CHARS = 90
TTS_STREAM_MAX_WAIT_SECONDS = 0.35
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
