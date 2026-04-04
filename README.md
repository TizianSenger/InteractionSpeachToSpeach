# Local Voice Assistant (Whisper + Ollama + pyttsx3)

Dieses Projekt baut genau den gewünschten lokalen Stack:

Mikrofon/Audio-Datei -> Whisper (STT) -> Ollama (Phi4-mini) -> pyttsx3 (TTS)

## Status

- [x] Grundstruktur erstellt
- [x] STT mit Whisper integriert
- [x] Ollama-Anbindung integriert
- [x] TTS mit pyttsx3 integriert
- [x] End-to-End Test und Feinschliff

## Setup (Windows, lokal)

1. Optional: Virtuelle Umgebung erstellen und aktivieren

	python -m venv .venv
	.\.venv\Scripts\Activate.ps1

2. Pakete installieren (explizit)

	pip install openai-whisper
	pip install pyttsx3
	pip install requests
	pip install sounddevice

3. Oder alles auf einmal über Datei

	pip install -r requirements.txt

4. Ollama Modell bereitstellen

	ollama pull phi4-mini

Hinweis: Für Whisper muss `ffmpeg` auf dem System verfügbar sein.

Schnellinstallation unter Windows (mit `winget`):

	winget install --id Gyan.FFmpeg -e --accept-package-agreements --accept-source-agreements
	winget install --id Ollama.Ollama -e --accept-package-agreements --accept-source-agreements

Danach Terminal neu öffnen und prüfen:

	ffmpeg -version
	ollama --version

## STT testen

1. Abhängigkeiten installieren:

	pip install -r requirements.txt

2. Transkription starten (Standardmodell: `small`):

	python main.py --audio aufnahme.wav

3. Optional anderes Modell:

	python main.py --audio aufnahme.wav --whisper-model tiny

## STT + Ollama testen

Voraussetzung: Ollama läuft lokal und `phi4-mini` ist verfügbar.

1. Modell laden (einmalig):

	ollama pull phi4-mini

2. Pipeline starten:

	python main.py --audio aufnahme.wav --ollama-model phi4-mini

3. Optional Sprechgeschwindigkeit anpassen:

	python main.py --audio aufnahme.wav --ollama-model phi4-mini --tts-rate 170

## End-to-End (dein finaler Stack)

Mikrofon/Audio-Datei -> Whisper -> Ollama (`phi4-mini`) -> pyttsx3 -> Lautsprecher

Beispielstart:

	python main.py --audio aufnahme.wav --whisper-model small --ollama-model phi4-mini --tts-rate 170

Direkt über Mikrofon (ohne WAV-Datei):

	python main.py --mic --mic-seconds 6 --whisper-model small --ollama-model phi4-mini --tts-rate 170

Oder ganz ohne Parameter (nimmt automatisch Mikrofon):

	python main.py

Optional ohne Argumente (z. B. über "Run Python File"):

	python main.py

Das Skript nimmt dann automatisch über dein Mikrofon auf.

## CustomTkinter UI

Starten:

	python voice_ui.py

Funktionen in der UI:

- Mic Start / Mic Stop
- Ollama Test Button (Server + Modell-Check)
- Mikrofon-Auswahl (Input Device) + Geräte-Refresh
- Live-Mikrofonpegel (auch ohne aktive Aufnahme)
- Statusanzeige (Aufnahme, Transkription, Ollama, TTS)
- Anzeige von Transkript und Modellantwort
- Mikrofon-Sample-Rate, Whisper-Modell, Ollama-Modell/URL, TTS-Rate einstellbar
