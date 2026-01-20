#Requires -Version 5.1
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Status {
    param(
        [string]$Message,
        [ValidateSet("OK", "FAIL", "WARN")][string]$Level = "OK"
    )
    $prefix = switch ($Level) {
        "OK" { "✅" }
        "FAIL" { "❌" }
        "WARN" { "⚠️" }
    }
    Write-Host "$prefix $Message"
}

$Paths = [ordered]@{
    AgenteSmith = "C:\Users\Usuario\Documents\whisper_live\AgenteSmith.py"
    WhisperLive = "C:\Users\Usuario\Documents\whisper_live\whisper_live.py"
    VenvRoot    = "C:\Users\Usuario\fish-speech\fishspeech_env"
    FishModel   = "C:\Users\Usuario\fish-speech\checkpoints\openaudio-s1-mini\model.pth"
    WhisperDir  = "C:\Users\Usuario\Documents\whisper_live"
}

Write-Host "Agente Smith setup starting..."

# 1) Verifica git/python/docker
foreach ($cmd in @("git", "python", "docker")) {
    if (Get-Command $cmd -ErrorAction SilentlyContinue) {
        Write-Status "$cmd encontrado."
    } else {
        Write-Status "$cmd NO encontrado en PATH." "FAIL"
    }
}

try {
    $dockerVersion = docker version --format '{{.Server.Version}}' 2>$null
    if ($dockerVersion) {
        Write-Status "Docker OK (Server $dockerVersion)."
    } else {
        Write-Status "Docker no respondió. No se tocarán contenedores." "WARN"
    }
} catch {
    Write-Status "Docker no respondió. No se tocarán contenedores." "WARN"
}

# 2) Crea/usa venv
if (-not (Test-Path $Paths.VenvRoot)) {
    Write-Status "Creando venv en $($Paths.VenvRoot)..."
    python -m venv $Paths.VenvRoot
} else {
    Write-Status "Venv existente en $($Paths.VenvRoot)."
}

$VenvPython = Join-Path $Paths.VenvRoot "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    Write-Status "No se encontró python del venv en $VenvPython" "FAIL"
    exit 1
}
Write-Status "Python del venv: $VenvPython"

# 3) Instala dependencias
Write-Status "Actualizando pip/setuptools/wheel..."
& $VenvPython -m pip install --upgrade pip setuptools wheel | Out-Host

$packages = @(
    "requests",
    "numpy",
    "sounddevice",
    "soundfile"
)
Write-Status "Instalando dependencias base: $($packages -join ', ')"
& $VenvPython -m pip install @packages | Out-Host

try {
    Write-Status "Intentando instalar pyaudio (opcional)..."
    & $VenvPython -m pip install pyaudio | Out-Host
} catch {
    Write-Status "No se pudo instalar pyaudio. Continúa si no es necesario." "WARN"
}

# 4) Verifica whisper_live.py con python del venv
if (Test-Path $Paths.WhisperLive) {
    Write-Status "whisper_live.py encontrado: $($Paths.WhisperLive)"
    try {
        $exe = & $VenvPython -c "import sys; print(sys.executable)"
        Write-Status "whisper_live.py se ejecutará con: $exe"
        & $VenvPython $Paths.WhisperLive --help 2>$null | Out-Host
        Write-Status "whisper_live.py ejecutado con el python del venv (ver salida arriba)."
    } catch {
        Write-Status "No se pudo ejecutar whisper_live.py con el venv. Revisa dependencias." "WARN"
    }
} else {
    Write-Status "No se encontró whisper_live.py en $($Paths.WhisperLive)" "FAIL"
}

# 5) Corrección de encoding (AgenteSmith.py)
if (Test-Path $Paths.AgenteSmith) {
    Write-Status "AgenteSmith.py encontrado para aplicar cambios de encoding (ver diff adjunto)."
} else {
    Write-Status "AgenteSmith.py no encontrado en $($Paths.AgenteSmith)" "WARN"
}

# 6) Configuración de dispositivos
$deviceInfoJson = @'
import json
try:
    import sounddevice as sd
except Exception as exc:
    print(json.dumps({"error": str(exc)}))
    raise SystemExit(0)

mic_pref = "Micrófono (Razer Barracuda X 2.4)"
sys_pref = "CABLE Output (VB-Audio Virtual Cable)"
spk_pref = "Auriculares (Razer Barracuda X (BT))"
spk_fallback = "Altavoces (Razer Barracuda X 2.4)"

def find_device(name):
    for dev in sd.query_devices():
        if name.lower() in dev["name"].lower():
            return dev["name"]
    return None

mic = find_device(mic_pref) or "default"
sys = find_device(sys_pref) or "default"
spk = find_device(spk_pref) or find_device(spk_fallback) or "default"

payload = {
    "mic": mic,
    "sys": sys,
    "spk": spk,
    "mic_pref": mic_pref,
    "sys_pref": sys_pref,
    "spk_pref": spk_pref,
    "spk_fallback": spk_fallback,
}
print(json.dumps(payload))
'@ | & $VenvPython -

if ($deviceInfoJson) {
    $deviceInfo = $deviceInfoJson | ConvertFrom-Json
    if ($deviceInfo.error) {
        Write-Status "Error consultando dispositivos de audio: $($deviceInfo.error)" "WARN"
    } else {
        $env:MIC_DEV = $deviceInfo.mic
        $env:SYS_DEV = $deviceInfo.sys
        Write-Status "MIC_DEV configurado: $($env:MIC_DEV)"
        Write-Status "SYS_DEV configurado: $($env:SYS_DEV)"

        if ($deviceInfo.mic -eq $deviceInfo.sys) {
            Write-Status "SYS_DEV y MIC_DEV apuntan al mismo dispositivo. SYS podría capturar el micro." "WARN"
        }

        $env:SPK_DEV = $deviceInfo.spk
        Write-Status "Dispositivo de salida para test: $($env:SPK_DEV)"
    }
}

# 7) Evitar duplicado MIC+SYS (aviso ya gestionado arriba)

# 8) Test de audio: pitido
try {
    @'
import os
import numpy as np
import sounddevice as sd

dev = os.environ.get("TEST_SPK") or os.environ.get("SPK_DEV")

fs = 48000
seconds = 0.5
freq = 880

samples = (np.sin(2 * np.pi * np.arange(fs * seconds) * freq / fs)).astype(np.float32)

sd.play(samples, fs, device=dev)

sd.wait()
'@ | & $VenvPython -
    Write-Status "Test de audio ejecutado (pitido)."
} catch {
    Write-Status "No se pudo ejecutar el test de audio." "WARN"
}

# 9) Logs claros con rutas
foreach ($key in $Paths.Keys) {
    Write-Status "$key => $($Paths[$key])"
}

# Paths ejecutables desde cualquier ruta (System32)
$existingPath = [Environment]::GetEnvironmentVariable("PATH", "User")
if ($existingPath -and ($existingPath -notmatch [Regex]::Escape($Paths.WhisperDir))) {
    [Environment]::SetEnvironmentVariable("PATH", "$existingPath;$($Paths.WhisperDir)", "User")
    Write-Status "Se añadió $($Paths.WhisperDir) al PATH del usuario (sesión nueva requerida)."
} else {
    Write-Status "PATH ya contiene $($Paths.WhisperDir)."
}

Write-Host "Setup finalizado."
