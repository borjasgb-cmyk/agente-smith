$ErrorActionPreference = "Stop"

$RepoRoot = "C:\Users\Usuario\Documents\whisper_live"
$VenvPy = "C:\Users\Usuario\fish-speech\fishspeech_env\Scripts\python.exe"
$CheckpointBase = $env:FISH_CHECKPOINT_DIR
if (-not $CheckpointBase) {
    $CheckpointBase = Join-Path $RepoRoot "checkpoints"
}
$CheckpointDir = Join-Path $CheckpointBase "openaudio-s1-mini"
$RequiredFiles = @("config.json", "model.pth", "codec.pth", "tokenizer.tiktoken")

Write-Host "== Smoke Test UI ==" -ForegroundColor Cyan

if (-not (Test-Path $VenvPy)) {
    Write-Host "FAIL: venv python no existe: $VenvPy" -ForegroundColor Red
    exit 1
}
Write-Host "OK: venv python encontrado"

if (-not (Test-Path $CheckpointDir)) {
    Write-Host "FAIL: checkpoints no existen: $CheckpointDir" -ForegroundColor Red
    exit 1
}

$missing = @()
foreach ($f in $RequiredFiles) {
    if (-not (Test-Path (Join-Path $CheckpointDir $f))) {
        $missing += $f
    }
}
if ($missing.Count -gt 0) {
    Write-Host "FAIL: faltan archivos: $($missing -join ', ')" -ForegroundColor Red
    exit 1
}
Write-Host "OK: checkpoints completos"

Write-Host "== Import test ==" -ForegroundColor Cyan
& $VenvPy -c "import tools.webui, tools.agent_supervisor, tools.n8n_client; print('imports OK')"

function Get-FreePort {
    param([int]$Preferred)
    try {
        $listener = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, $Preferred)
        $listener.Start()
        $listener.Stop()
        return $Preferred
    } catch {
        $tcp = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, 0)
        $tcp.Start()
        $port = ($tcp.LocalEndpoint).Port
        $tcp.Stop()
        return $port
    }
}

$port = Get-FreePort -Preferred 7863
Write-Host "== Start WebUI (port $port) ==" -ForegroundColor Cyan
$env:FISH_CHECKPOINT_DIR = $CheckpointBase
$env:FISH_SKIP_WARMUP = "1"
$env:FISH_SMOKE_UI = "1"
$env:PYTHONUNBUFFERED = "1"
$log = Join-Path $RepoRoot "tools\smoke_webui.log"
$logErr = Join-Path $RepoRoot "tools\smoke_webui.err.log"
if (Test-Path $log) { Remove-Item $log -Force }
if (Test-Path $logErr) { Remove-Item $logErr -Force }

$proc = Start-Process -FilePath $VenvPy `
    -ArgumentList @("-u", "tools\run_webui.py", "--device", "cpu", "--server-port", "$port") `
    -WorkingDirectory $RepoRoot `
    -NoNewWindow -PassThru `
    -RedirectStandardOutput $log -RedirectStandardError $logErr

Start-Sleep -Seconds 10

if (-not (Test-Path $log)) {
    Write-Host "FAIL: no se genero log" -ForegroundColor Red
    $proc.Kill()
    exit 1
}

$content = ""
if (Test-Path $log) { $content += Get-Content $log -Raw }
if (Test-Path $logErr) { $content += Get-Content $logErr -Raw }
if ($content -match "Running on local URL") {
    Write-Host "OK: WebUI arranco en http://127.0.0.1:$port"
    $proc.Kill()
    exit 0
}

Write-Host "FAIL: no se encontro 'Running on local URL' en log" -ForegroundColor Red
Write-Host ($content | Select-Object -Last 40)
if (-not $proc.HasExited) { $proc.Kill() }
exit 1
