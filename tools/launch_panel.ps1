$ErrorActionPreference = "Stop"

$RepoRoot = "C:\Users\Usuario\Documents\whisper_live"
$VenvPy = "C:\Users\Usuario\fish-speech\fishspeech_env\Scripts\python.exe"

Set-Location $RepoRoot
$env:FISH_CHECKPOINT_DIR = "C:\Users\Usuario\fish-speech\checkpoints"
$env:PYTHONIOENCODING = "utf-8"

if (-not (Test-Path $VenvPy)) {
    Write-Host "ERROR: No se encontro el venv en:" -ForegroundColor Red
    Write-Host $VenvPy -ForegroundColor Red
    Read-Host "Presiona Enter para salir"
    exit 1
}

Write-Host "Iniciando Agente Smith Control Panel (CPU)..." -ForegroundColor Green
& $VenvPy "tools\run_webui.py" --device cpu

Read-Host "WebUI finalizo. Presiona Enter para cerrar"
