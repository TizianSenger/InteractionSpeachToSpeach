# Portable Offline Bundle

Diese Anleitung erstellt ein transportierbares Offline-Paket mit:

- App-Dateien
- lokaler `.venv`
- Ollama-Binaries
- lokal vorhandenen Ollama-Modellen

## 1. Bundle bauen

Im Projektordner ausfuehren:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\create_portable_bundle.ps1 -BundleName VoiceStudio-Portable
```

Optional:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\create_portable_bundle.ps1 -BundleName VoiceStudio-Portable -IncludeLogs $true
```

Ergebnis liegt unter:

- `dist_portable\VoiceStudio-Portable`

## 2. Auf Zielrechner kopieren

Den kompletten Ordner `dist_portable\VoiceStudio-Portable` auf den isolierten Rechner kopieren.

## 3. Starten auf Zielrechner

Im Bundle-Ordner:

- `start_portable_voice_studio.bat`

Das Skript startet zuerst Ollama lokal und dann die UI.

## Wichtige Hinweise

- Zielsystem sollte Windows x64 sein.
- Die Bundle-Groesse kann sehr gross sein (Modelle + `.venv`).
- Wenn auf dem Zielrechner Audio- oder Laufzeitkomponenten fehlen, kann TTS/STT eingeschraenkt sein.
- Falls du spaeter weitere Modelle brauchst, muessen diese vor dem Bundling auf dem Build-Rechner vorhanden sein.
