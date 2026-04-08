# Provider Smoke-Test Checklist

Ziel: schneller End-to-End Check nach Aenderungen an Provider-Logik, UI oder Streaming.

## Vorbedingungen

- App startet ohne Traceback.
- Mikrofon ist waehlbar.
- Mindestens ein TTS-Backend funktioniert.
- Fuer Cloud-Provider sind gueltige API-Keys vorhanden.

## Globaler Basis-Check

1. App starten: python voice_ui.py
2. Einstellungen -> Modell oeffnen.
3. Provider wechseln: Ollama, OpenAI, Azure OpenAI, Anthropic, Groq.
4. Erwartung:
- Nur der passende Provider-Block ist sichtbar.
- Keine UI-Fehler oder eingefrorene Buttons.

## Provider-Diagnostik Check

Pro Provider:

1. Felder ausfuellen (Model/Deployment, URL/Endpoint, API-Key).
2. Provider-Diagnostik starten.
3. Erwartung:
- Status zeigt OK mit RTT.
- Modellanzahl > 0 (falls Listing unterstuetzt).
- Fehlertexte sind klar (401/404/429/Timeout).

## Modell-Refresh Check

Pro Provider:

1. Modelle vom Provider laden klicken.
2. Erwartung:
- Model-Dropdown wird aktualisiert.
- Aktives Modell bleibt erhalten oder faellt auf erstes Modell.
- Status meldet Anzahl + Preview.

## Chat-Streaming Check

Pro Provider:

1. Kurze Eingabe senden (z. B. "Sag hallo in 1 Satz").
2. Erwartung:
- Streaming-Text erscheint ohne Freeze.
- Cancel funktioniert.
- Antwort landet im Verlauf.

## Audio/Interrupt Check

1. Auto-Speak aktivieren.
2. Frage senden, waehrend TTS laeuft hineinsprechen.
3. Erwartung:
- Barge-In stoppt laufende Antwort.
- Neue Aufnahme startet.
- App bleibt responsive.

## Config Export/Import (ohne API-Keys)

1. Provider-Config export.
2. Datei pruefen:
- Enthalten: Provider, Modelle, Endpoints, Parameter.
- Nicht enthalten: API-Keys.
3. Provider-Config import.
4. Erwartung:
- Felder werden wiederhergestellt.
- API-Key-Felder bleiben leer.

## Security-Flag Check

1. API-Keys im Profil speichern deaktivieren.
2. Profil speichern, App neu starten.
3. Erwartung:
- Nicht-sensitive Felder sind erhalten.
- API-Key-Felder sind leer.

## Fehlerfall-Checkliste

- 401: ungueltiger API-Key
- 404: falscher Endpoint oder Modell/Deployment
- 429: Rate-Limit
- Timeout: Netzwerk/Latenzproblem

Erwartung:
- Fehlertext ist providerbezogen und verstaendlich.
- Kein Crash der UI.

## Release-Gate (schnell)

Ein Release gilt als "smoke-test passed", wenn:

- Globaler Basis-Check: OK
- Mindestens 2 Provider inklusive 1 Cloud-Provider: OK
- Chat-Streaming + Cancel: OK
- Export/Import ohne Keys: OK
- Security-Flag: OK
