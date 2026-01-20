$ErrorActionPreference = "Stop"

$VenvPy = "C:\Users\Usuario\fish-speech\fishspeech_env\Scripts\python.exe"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")

Write-Host "== Dispositivos de audio ==" -ForegroundColor Cyan
$pyList = @'
import sounddevice as sd
hostapis = sd.query_hostapis()
for i, dev in enumerate(sd.query_devices()):
    host = hostapis[dev["hostapi"]]["name"]
    ins = dev.get("max_input_channels", 0)
    outs = dev.get("max_output_channels", 0)
    print("{:>2} | {:<8} | in={} out={} | {}".format(i, host, ins, outs, dev["name"]))
'@
$pyList | & $VenvPy -

if (-not $env:SPK_DEV_INDEX) {
    $env:SPK_DEV_INDEX = Read-Host "Indice de salida (SPK) para beep"
}

Write-Host "== Beep de prueba ==" -ForegroundColor Cyan
$pyBeep = @'
import math
import sys
import numpy as np
import sounddevice as sd

idx = int(sys.argv[1])
dev = sd.query_devices(idx)
if dev.get("max_output_channels", 0) < 1:
    raise SystemExit("Device {} no tiene salida de audio".format(idx))
sr = 48000
duration = 0.5
freq = 880.0
t = np.linspace(0, duration, int(sr * duration), endpoint=False)
tone = 0.2 * np.sin(2 * math.pi * freq * t)
sd.play(tone, samplerate=sr, device=idx)
sd.wait()
print("Beep OK en device={}".format(idx))
'@
$pyBeep | & $VenvPy - $env:SPK_DEV_INDEX

if (-not $env:MIC_DEV_INDEX) { $env:MIC_DEV_INDEX = "18" }
if (-not $env:SYS_DEV_INDEX) { $env:SYS_DEV_INDEX = "17" }
if (-not $env:CAPTURE_SYS) { $env:CAPTURE_SYS = "1" }
if (-not $env:PYTHONIOENCODING) { $env:PYTHONIOENCODING = "utf-8" }

Write-Host "== Arranque whisper_live (5s) ==" -ForegroundColor Cyan
$log = Join-Path $PSScriptRoot "selftest-whisper_live.log"
$logErr = Join-Path $PSScriptRoot "selftest-whisper_live.err.log"
if (Test-Path $log) { Remove-Item $log -Force }
if (Test-Path $logErr) { Remove-Item $logErr -Force }

$whisperPath = Join-Path $RepoRoot "whisper_live.py"
$proc = Start-Process -FilePath $VenvPy `
    -ArgumentList @("-u", $whisperPath) `
    -NoNewWindow -PassThru `
    -RedirectStandardOutput $log -RedirectStandardError $logErr

Start-Sleep -Seconds 5
if ($proc.HasExited) {
    Write-Host "whisper_live termino antes de 5s" -ForegroundColor Red
    if (Test-Path $log) { Get-Content $log -Tail 60 }
    if (Test-Path $logErr) { Get-Content $logErr -Tail 60 }
    exit 1
}

Write-Host "whisper_live OK, deteniendo..." -ForegroundColor Green
$proc.Kill()
