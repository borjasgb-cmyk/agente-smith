$ErrorActionPreference = "Stop"

$RepoRoot = "C:\Users\Usuario\Documents\whisper_live"
$VenvPy = "C:\Users\Usuario\fish-speech\fishspeech_env\Scripts\python.exe"
$CheckpointBase = $env:FISH_CHECKPOINT_DIR
if (-not $CheckpointBase) {
    $CheckpointBase = "C:\Users\Usuario\fish-speech\checkpoints"
}
$CheckpointDir = Join-Path $CheckpointBase "openaudio-s1-mini"
$RequiredFiles = @("config.json", "model.pth", "codec.pth", "tokenizer.tiktoken")

Set-Location $RepoRoot
$env:FISH_CHECKPOINT_DIR = $CheckpointBase
$env:PYTHONIOENCODING = "utf-8"
$env:CUDA_VISIBLE_DEVICES = "-1"
$env:FISH_SKIP_WARMUP = "1"

if (-not (Test-Path $VenvPy)) {
    Write-Host "ERROR: No se encontro el venv en:" -ForegroundColor Red
    Write-Host $VenvPy -ForegroundColor Red
    Read-Host "Presiona Enter para salir"
    exit 1
}

if (-not (Test-Path $CheckpointDir)) {
    Write-Host "ERROR: checkpoints no existen: $CheckpointDir" -ForegroundColor Red
    Write-Host "Configura FISH_CHECKPOINT_DIR o crea un symlink:" -ForegroundColor Yellow
    Write-Host "  mklink /D checkpoints C:\Users\Usuario\fish-speech\checkpoints"
    Read-Host "Presiona Enter para salir"
    exit 1
}

$missing = @()
foreach ($f in $RequiredFiles) {
    if (-not (Test-Path (Join-Path $CheckpointDir $f))) {
        $missing += $f
    }
}
if ($missing.Count -gt 0) {
    Write-Host "ERROR: faltan archivos: $($missing -join ', ')" -ForegroundColor Red
    Read-Host "Presiona Enter para salir"
    exit 1
}

Write-Host "Iniciando AgenteSmith Panel..." -ForegroundColor Green
& $VenvPy "tools\launch_panel.py"

Read-Host "WebUI finalizo. Presiona Enter para cerrar"
