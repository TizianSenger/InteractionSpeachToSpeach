# ! piper voice files need to be downloaded those are not provided in this project



# Local Voice Assistant (Whisper + Ollama + pyttsx3)

Dieses Projekt baut genau den gewünschten lokalen Stack:

Mikrofon/Audio-Datei -> Whisper (STT) -> Ollama (Phi4-mini) -> pyttsx3 (TTS)

## Status

- [x] Grundstruktur erstellt
- [x] STT mit Faster-Whisper (Fallback: Whisper) integriert
- [x] Ollama-Anbindung integriert
- [x] TTS mit pyttsx3 integriert
- [x] End-to-End Test und Feinschliff

## Setup (Windows, lokal)

1. Optional: Virtuelle Umgebung erstellen und aktivieren

	python -m venv .venv
	.\.venv\Scripts\Activate.ps1

2. Pakete installieren (explizit)

	pip install faster-whisper
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
- Getrennter Ablauf: 1) Aufnehmen 2) Transkribieren 3) An Ollama senden
- Optionaler Auto-Workflow: nach `Mic Stop` automatisch transkribieren und an Modell senden
- Mikrofon kann umgangen werden: Text direkt ins Transkriptfeld schreiben und `Text direkt senden` nutzen
- Ollama Test Button (Server + Modell-Check)
- Mikrofon-Auswahl (Input Device) + Geräte-Refresh
- Live-Mikrofonpegel (auch ohne aktive Aufnahme)
- Gruppierte Einstellungs-Panels: STT, Modell, Audio, TTS
- Gesprächsverlauf mit Zeitstempeln (inkl. Button zum Leeren)
- Statistik-Tab mit Laufzeit, Countern und durchschnittlichen Latenzen
- Debug-Logs-Tab mit Live-Logstream, Log-Level-Filter und "Logs leeren"
- Light Controller Test-Popup mit Buttons `Licht AN` / `Licht AUS`
- Sprachbefehle fuer Lichtsteuerung (z. B. `Licht an`, `Licht aus`, `Licht einschalten`, `Licht ausschalten`)
- Lichtbefehle werden ueber ein Ollama-Tool-Routing (JSON-Entscheidung) erkannt und dann lokal ausgefuehrt
- Persona-System mit Slidern: `Flirty`, `Humor/Sarkasmus`, `Ernsthaftigkeit`, `Dominanz`, `Empathie/Waerme`, `Temperament`
- Profil-Memory mit persistenter Datei `assistant_profile.json` (merkt Stilpraeferenzen, Lieblingslichtszene, Standardgeraete)
- Statusanzeige (Aufnahme, Transkription, Ollama, TTS)
- Anzeige von Transkript und Modellantwort
- Mikrofon-Sample-Rate, Whisper-Modell, Ollama-Modell/URL, TTS-Rate einstellbar
- TTS Engine-Auswahl: `edge-tts (natürlich)`, `piper (lokal, natürlich)` oder `pyttsx3 (lokal)`
- Mehrere Stimmen (u. a. weiblich/männlich) + Emotions-Presets
- Integrierter Avatar-Button: startet den VRM-Viewer direkt aus dem Projekt
- `Avatar LipSync`-Schalter: synchronisiert Mundbewegung waehrend TTS-Wiedergabe
- Whisper ist standardmäßig auf `large` gesetzt
- Ollama-Modell in der UI umschaltbar: `phi4-mini` oder `dolphin-mistral`
- STT Sprache wählbar: `Deutsch`, `Englisch` oder `Auto`
- STT Modus wählbar: `Schnell` (niedrigere Latenz) oder `Genau`
- Fast-Reply Controls fuer Ollama: `Max Tokens`, `Temperatur`, `Kurze Voice-Antworten`

Hinweis: `edge-tts` klingt natürlicher, benötigt aber eine Internetverbindung.
Für die Wiedergabe in der UI wird dafür `pygame` genutzt.

## Avatar + LipSync (neu)

In der Workflow-Leiste gibt es jetzt:

- `Avatar starten` / `Avatar stoppen`
- `Avatar LipSync` (ein/aus)

Ablauf:

1. `Avatar starten` klicken.
2. Eine Antwort erzeugen und vorlesen lassen (`Antwort vorlesen` aktiv).
3. Waehrend der Wiedergabe wird der Mund des VRM-Avatars automatisch animiert.

Hinweis:

- Benoetigte Datei: `runtime_assets/model/vrm_AvatarSample_S.vrm`
- LipSync arbeitet lokal ueber eine interne HTTP-Bridge (`/api/lipsync`) zwischen Voice-UI und Viewer.

Debug-Logdatei:

- Laufende Logs werden zusätzlich in `logs/voice_ui.log` gespeichert.

Piper (offline, natürlich):

1. In der UI bei TTS Engine `piper (lokal, natürlich)` wählen.
2. Bei `Piper Modell (.onnx)` den Pfad zu einer lokalen Stimme eintragen (z. B. `models/de_DE-karlsson-medium.onnx`).
3. `Piper Config (.json)` optional leer lassen, wenn die passende JSON neben der ONNX-Datei liegt.

Hinweis: Piper ist komplett offline, benötigt aber lokal heruntergeladene Stimmen.
