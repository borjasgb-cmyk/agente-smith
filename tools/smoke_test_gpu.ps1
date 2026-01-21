param(
    [switch]$RequireCuda
)

$ErrorActionPreference = "Stop"

$RepoRoot = "C:\Users\Usuario\Documents\whisper_live"
$VenvPy = "C:\Users\Usuario\fish-speech\fishspeech_env\Scripts\python.exe"

Write-Host "== Smoke Test GPU ==" -ForegroundColor Cyan

if (-not (Test-Path $VenvPy)) {
    Write-Host "FAIL: venv python no existe: $VenvPy" -ForegroundColor Red
    exit 1
}

$script = @'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("torch_cuda", torch.version.cuda)
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("cc", torch.cuda.get_device_capability(0))
    try:
        x = torch.randn(1024, device="cuda")
        print("cuda_test", x.sum().item())
        print("cuda_ok", True)
    except Exception as exc:
        print("cuda_ok", False)
        print("cuda_error", exc)
else:
    print("cuda_ok", False)
'@

$output = $script | & $VenvPy -
Write-Host $output

$cudaOk = $false
if ($output -match "cuda_ok True") { $cudaOk = $true }

if ($RequireCuda -and -not $cudaOk) {
    Write-Host "FAIL: CUDA requested but not usable." -ForegroundColor Red
    exit 1
}

Write-Host "OK: GPU check complete" -ForegroundColor Green
exit 0
