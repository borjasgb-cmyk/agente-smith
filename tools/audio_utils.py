from __future__ import annotations

import math
import time
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


def pick_device_index(
    devices: list[dict[str, Any]],
    kind: str,
    name_keywords: list[str],
    hostapi_prefer: str | None = None,
) -> int | None:
    matches = []
    for dev in devices:
        if kind == "input" and dev["max_input_channels"] < 1:
            continue
        if kind == "output" and dev["max_output_channels"] < 1:
            continue
        name = dev["name"].lower()
        if all(keyword.lower() in name for keyword in name_keywords):
            matches.append(dev)
    if not matches:
        return None
    if hostapi_prefer:
        for dev in matches:
            if hostapi_prefer.lower() in dev["hostapi"].lower():
                return int(dev["index"])
    return int(matches[0]["index"])


def rms_meter(device_idx: int, duration: float = 0.2) -> dict[str, float | str]:
    sample_rate = _pick_input_rate([device_idx])
    if sample_rate is None:
        return {"rms": 0.0, "peak": 0.0, "status": "UNAVAILABLE"}
    try:
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            device=device_idx,
            blocking=True,
        ).astype(np.float32).squeeze()
        rms = float(math.sqrt(float(np.mean(audio**2)))) if audio.size else 0.0
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        return {"rms": rms, "peak": peak, "status": "OK"}
    except Exception:
        return {"rms": 0.0, "peak": 0.0, "status": "ERROR"}


def play_audio(audio: np.ndarray, sample_rate: int, device_idx: int | None) -> None:
    if audio.size == 0:
        return
    sd.play(audio, samplerate=sample_rate, device=device_idx)
    sd.wait()


def sys_quick_check(sys_idx: int) -> dict[str, float | str]:
    duration = 3.0
    sample_rate = _pick_input_rate([sys_idx])
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


def mic_sys_bleed_check(mic_idx: int, sys_idx: int) -> dict[str, float | str]:
    duration = 3.0
    sample_rate = _pick_input_rate([mic_idx, sys_idx])
    if sample_rate is None:
        return {"mic_rms": 0.0, "sys_rms": 0.0, "ratio": 0.0, "corr": 0.0, "verdict": "SILENT"}

    try:
        mic_audio = _record_stream(mic_idx, duration, sample_rate)
        sys_audio = _record_stream(sys_idx, duration, sample_rate)
    except Exception:
        return {"mic_rms": 0.0, "sys_rms": 0.0, "ratio": 0.0, "corr": 0.0, "verdict": "SILENT"}
    mic_audio = mic_audio.astype(np.float32)
    sys_audio = sys_audio.astype(np.float32)
    if mic_audio.size == 0 or sys_audio.size == 0:
        return {"mic_rms": 0.0, "sys_rms": 0.0, "ratio": 0.0, "corr": 0.0, "verdict": "SILENT"}

    mic_rms = float(math.sqrt(float(np.mean(mic_audio**2))))
    sys_rms = float(math.sqrt(float(np.mean(sys_audio**2))))
    ratio = float(sys_rms / mic_rms) if mic_rms > 0 else 0.0

    corr = _correlation(mic_audio, sys_audio)
    verdict = "BLEEDING" if (corr >= 0.25 and ratio >= 0.2) else "OK"
    return {
        "mic_rms": mic_rms,
        "sys_rms": sys_rms,
        "ratio": ratio,
        "corr": corr,
        "verdict": verdict,
    }


def _pick_input_rate(devices: list[int]) -> int | None:
    for rate in (48000, 44100, 32000, 16000):
        try:
            for dev in devices:
                sd.check_input_settings(device=dev, samplerate=rate, channels=1)
            return rate
        except Exception:
            continue
    return None


def _record_stream(device: int, duration: float, sample_rate: int) -> np.ndarray:
    frames: list[np.ndarray] = []

    def callback(indata, _frames, _time, _status):
        frames.append(indata.copy())

    with sd.InputStream(
        device=device, channels=1, samplerate=sample_rate, callback=callback
    ):
        time.sleep(duration)

    if not frames:
        return np.array([], dtype=np.float32)
    return np.concatenate(frames, axis=0).squeeze()


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    size = min(a.shape[0], b.shape[0])
    if size <= 0:
        return 0.0
    a = a[:size] - float(np.mean(a[:size]))
    b = b[:size] - float(np.mean(b[:size]))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 0.0
    return float(np.dot(a, b) / denom)
