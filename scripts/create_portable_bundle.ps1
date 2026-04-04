param(
    [string]$BundleName = "VoiceStudio-Portable",
    [bool]$IncludeVenv = $true,
    [bool]$IncludeLogs = $false,
    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"

$ScriptsRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptsRoot
$ResolvedOutputRoot = $OutputRoot.Trim()
if (-not $ResolvedOutputRoot) {
    $ResolvedOutputRoot = [Environment]::GetFolderPath("Desktop")
}
$DistRoot = Join-Path $ResolvedOutputRoot "dist_portable"
$BundleRoot = Join-Path $DistRoot $BundleName

function Write-Step {
    param([string]$Message)
    Write-Host "[portable] $Message"
}

function Copy-IfExists {
    param(
        [string]$Source,
        [string]$Destination
    )

    if (Test-Path $Source) {
        Copy-Item -Path $Source -Destination $Destination -Recurse -Force
        return $true
    }
    return $false
}

Write-Step "Bereite Zielordner vor"
New-Item -Path $DistRoot -ItemType Directory -Force | Out-Null
if (Test-Path $BundleRoot) {
    Remove-Item -Path $BundleRoot -Recurse -Force
}
New-Item -Path $BundleRoot -ItemType Directory -Force | Out-Null

Write-Step "Kopiere App-Dateien"
$rootFiles = @(
    "assistant_profile.json",
    "avatar_bridge.py",
    "local_http_server.py",
    "main.py",
    "requirements.txt",
    "run_vrm_viewer.bat",
    "standalone_viewer.py",
    "viewer_process.py",
    "voice_ui.py"
)

foreach ($file in $rootFiles) {
    $src = Join-Path $ProjectRoot $file
    if (Test-Path $src) {
        Copy-Item -Path $src -Destination $BundleRoot -Force
    }
}

$rootDirs = @(
    "web",
    "runtime_assets",
    "piperVoices"
)

if ($IncludeLogs) {
    $rootDirs += "logs"
}

foreach ($dir in $rootDirs) {
    $src = Join-Path $ProjectRoot $dir
    if (Test-Path $src) {
        Copy-Item -Path $src -Destination (Join-Path $BundleRoot $dir) -Recurse -Force
    }
}

if ($IncludeVenv) {
    $venvPath = Join-Path $ProjectRoot ".venv"
    if (-not (Test-Path $venvPath)) {
        throw "Die lokale .venv wurde nicht gefunden. Erstelle sie zuerst oder setze -IncludeVenv `$false."
    }
    Write-Step "Kopiere .venv (kann dauern)"
    Copy-Item -Path $venvPath -Destination (Join-Path $BundleRoot ".venv") -Recurse -Force
}

Write-Step "Suche Ollama Installation"
$ollamaCandidatePaths = @(
    (Join-Path $env:LOCALAPPDATA "Programs\Ollama"),
    (Join-Path $env:ProgramFiles "Ollama")
)

$ollamaInstallPath = $null
foreach ($candidate in $ollamaCandidatePaths) {
    if (Test-Path (Join-Path $candidate "ollama.exe")) {
        $ollamaInstallPath = $candidate
        break
    }
}

if (-not $ollamaInstallPath) {
    throw "Ollama Installation nicht gefunden. Installiere Ollama lokal, dann Script erneut ausfuehren."
}

Write-Step "Kopiere Ollama Binaries"
Copy-Item -Path $ollamaInstallPath -Destination (Join-Path $BundleRoot "ollama") -Recurse -Force

Write-Step "Kopiere lokale Ollama Modelle"
$localOllamaModels = Join-Path $env:USERPROFILE ".ollama\models"
$bundleOllamaData = Join-Path $BundleRoot "ollama-data"
New-Item -Path $bundleOllamaData -ItemType Directory -Force | Out-Null

if (Test-Path $localOllamaModels) {
    Copy-Item -Path $localOllamaModels -Destination (Join-Path $bundleOllamaData "models") -Recurse -Force
}
else {
    Write-Warning "Kein lokaler .ollama/models Ordner gefunden. Das Bundle enthaelt dann keine Modelle."
}

Write-Step "Erzeuge Startskripte"
$startBatPath = Join-Path $BundleRoot "start_portable_voice_studio.bat"
$startBat = @"
@echo off
setlocal
cd /d "%~dp0"

set "ROOT=%~dp0"
set "OLLAMA_MODELS=%ROOT%ollama-data\models"
set "OLLAMA_HOST=127.0.0.1:11434"

if not exist "%ROOT%ollama\ollama.exe" (
  echo [Fehler] ollama.exe fehlt im Unterordner ollama.
  pause
  exit /b 1
)

start "Ollama Portable" /min "%ROOT%ollama\ollama.exe" serve

timeout /t 3 /nobreak >nul

if exist "%ROOT%.venv\Scripts\python.exe" (
  "%ROOT%.venv\Scripts\python.exe" "%ROOT%voice_ui.py"
) else (
  python "%ROOT%voice_ui.py"
)

endlocal
"@
Set-Content -Path $startBatPath -Value $startBat -Encoding ASCII

$stopBatPath = Join-Path $BundleRoot "stop_portable_ollama.bat"
$stopBat = @"
@echo off
for /f "tokens=2" %%a in ('tasklist /fi "imagename eq ollama.exe" /fo csv ^| find /i "ollama.exe"') do (
  taskkill /pid %%~a /f >nul 2>&1
)
echo Ollama Prozesse wurden beendet.
"@
Set-Content -Path $stopBatPath -Value $stopBat -Encoding ASCII

$readmePath = Join-Path $BundleRoot "PORTABLE_README.txt"
$readme = @"
Voice Studio Portable (Offline)

1) Kopiere den gesamten Ordner auf den Zielrechner.
2) Starte start_portable_voice_studio.bat.
3) Beenden: stop_portable_ollama.bat (optional).

Hinweise:
- Zielrechner muss Windows x64 sein (moeglichst gleiche Architektur wie Build-Rechner).
- Bei fehlenden Audio-Treibern/VC-Runtimes kann die App nicht starten.
- Modelle liegen in ollama-data\models.
"@
Set-Content -Path $readmePath -Value $readme -Encoding ASCII

Write-Step "Fertig: $BundleRoot"
Write-Host ""
Write-Host "Portable Bundle erstellt:" -ForegroundColor Green
Write-Host $BundleRoot
