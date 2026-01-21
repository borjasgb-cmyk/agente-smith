from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class GPUInfo:
    torch_version: str
    torch_cuda: str | None
    cuda_available: bool
    device_name: str | None
    compute_capability: tuple[int, int] | None
    vram_total_gb: float | None
    vram_free_gb: float | None


def get_gpu_info() -> GPUInfo:
    device_name = None
    cc = None
    vram_total = None
    vram_free = None
    if torch.cuda.is_available():
        try:
            device_name = torch.cuda.get_device_name(0)
        except Exception:
            device_name = None
        try:
            cc = torch.cuda.get_device_capability(0)
        except Exception:
            cc = None
        try:
            free, total = torch.cuda.mem_get_info(0)
            vram_total = float(total) / (1024**3)
            vram_free = float(free) / (1024**3)
        except Exception:
            vram_total = None
            vram_free = None

    return GPUInfo(
        torch_version=torch.__version__,
        torch_cuda=torch.version.cuda,
        cuda_available=torch.cuda.is_available(),
        device_name=device_name,
        compute_capability=cc,
        vram_total_gb=vram_total,
        vram_free_gb=vram_free,
    )


def cuda_kernel_test() -> tuple[bool, Optional[str]]:
    if not torch.cuda.is_available():
        return False, "cuda not available"
    try:
        x = torch.randn(1024, device="cuda")
        _ = x.sum().item()
        return True, None
    except Exception as exc:
        return False, str(exc)
