from __future__ import annotations

import math
from typing import Any

import numpy as np
import sounddevice as sd


def list_devices() -> list[dict[str, Any]]:
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    items = []
    for i, dev in enumerate(devices):
        host = "Unknown"
        try:
            host = hostapis[dev.get("hostapi")]["name"]
        except Exception:
            pass
        items.append(
            {
                "index": i,
                "name": dev.get("name", "Unknown"),
                "hostapi": host,
                "max_input_channels": dev.get("max_input_channels", 0),
                "max_output_channels": dev.get("max_output_channels", 0),
            }
        )
    return items


def build_device_choices(devices: list[dict[str, Any]], kind: str) -> list[tuple[str, int]]:
    choices: list[tuple[str, int]] = []
    for dev in devices:
        if kind == "input" and dev["max_input_channels"] < 1:
            continue
        if kind == "output" and dev["max_output_channels"] < 1:
            continue
        label = f'{dev["index"]} | {dev["hostapi"]} | {dev["name"]}'
        choices.append((label, dev["index"]))
    return choices


def sys_quick_check(sys_idx: int) -> dict[str, float | str]:
    duration = 3.0
    sample_rate = None
    for rate in (48000, 44100, 32000, 16000):
        try:
            sd.check_input_settings(device=sys_idx, samplerate=rate, channels=1)
            sample_rate = rate
            break
        except Exception:
            continue
    if sample_rate is None:
        return {"rms": 0.0, "peak": 0.0, "verdict": "SILENT"}

    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        device=sys_idx,
        blocking=True,
    ).astype(np.float32).squeeze()
    rms = float(math.sqrt(float(np.mean(audio**2)))) if audio.size else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    verdict = "OK" if (rms >= 0.002 or peak >= 0.01) else "SILENT"
    return {"rms": rms, "peak": peak, "verdict": verdict}
