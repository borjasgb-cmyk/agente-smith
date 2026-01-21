param(
    [int]$Port = 7865
)

$ErrorActionPreference = "Stop"

$RepoRoot = "C:\Users\Usuario\Documents\whisper_live"
$VenvPy = "C:\Users\Usuario\fish-speech\fishspeech_env\Scripts\python.exe"

Write-Host "== Smoke Test Panel ==" -ForegroundColor Cyan

if (-not (Test-Path $VenvPy)) {
    Write-Host "FAIL: venv python no existe: $VenvPy" -ForegroundColor Red
    exit 1
}

$refTest = @'
from fish_speech.inference_engine.reference_loader import ReferenceLoader
ReferenceLoader()
print("ReferenceLoader OK")
'@
$refTest | & $VenvPy -

$env:FISH_CHECKPOINT_DIR = "C:\Users\Usuario\fish-speech\checkpoints"
$env:FISH_SKIP_WARMUP = "1"
$env:FISH_SMOKE_UI = "1"
$env:PANEL_METER_LOG = "1"

$log = Join-Path $RepoRoot "tools\panel_smoke.log"
$logErr = Join-Path $RepoRoot "tools\panel_smoke.err.log"
if (Test-Path $log) { Remove-Item $log -Force }
if (Test-Path $logErr) { Remove-Item $logErr -Force }

$proc = Start-Process -FilePath $VenvPy `
    -ArgumentList @("-u", "tools\run_webui.py", "--server-port", "$Port") `
    -WorkingDirectory $RepoRoot `
    -NoNewWindow -PassThru `
    -RedirectStandardOutput $log -RedirectStandardError $logErr

Start-Sleep -Seconds 5

$meterFile = Join-Path $RepoRoot "logs\panel_meters.json"
if (-not (Test-Path $meterFile)) {
    Write-Host "FAIL: meters no actualizan (panel_meters.json missing)" -ForegroundColor Red
    if (-not $proc.HasExited) { $proc.Kill() }
    exit 1
}

Write-Host "OK: meters updated"

Write-Host "== TTS test ==" -ForegroundColor Cyan
& $VenvPy "tools\chat_tts_test.py"
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAIL: TTS test" -ForegroundColor Red
    if (-not $proc.HasExited) { $proc.Kill() }
    exit 1
}

Write-Host "OK: TTS file generated"
if (-not $proc.HasExited) { $proc.Kill() }
exit 0
