from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
import sounddevice as sd


class ChatRecorder:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._recording = False
        self._buffer: list[np.ndarray] = []
        self._sample_rate: int | None = None
        self._mic_idx: int | None = None
        self._last_error: Optional[str] = None

    def start(self, mic_idx: int | None) -> tuple[bool, str]:
        if mic_idx is None:
            return False, "MIC no seleccionado"
        with self._lock:
            if self._recording:
                return True, "Grabando..."
            self._recording = True
            self._mic_idx = mic_idx
            self._buffer = []
            self._last_error = None
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return True, "Grabando..."

    def stop(self) -> tuple[bool, str, Optional[np.ndarray], Optional[int]]:
        with self._lock:
            if not self._recording:
                return True, "No estaba grabando", None, None
            self._stop.set()
            thread = self._thread
        if thread:
            thread.join(timeout=2)
        with self._lock:
            self._recording = False
            if not self._buffer:
                return False, self._last_error or "Sin audio", None, None
            audio = np.concatenate(self._buffer, axis=0).squeeze()
            return True, "OK", audio, self._sample_rate

    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    def _pick_rate(self, mic_idx: int) -> int | None:
        for rate in (48000, 44100, 32000, 16000):
            try:
                sd.check_input_settings(device=mic_idx, samplerate=rate, channels=1)
                return rate
            except Exception:
                continue
        return None

    def _run(self) -> None:
        mic_idx = self._mic_idx
        if mic_idx is None:
            self._set_error("MIC no seleccionado")
            return
        sample_rate = self._pick_rate(mic_idx)
        if sample_rate is None:
            self._set_error("MIC no disponible")
            return
        self._sample_rate = sample_rate

        def callback(indata, _frames, _time, _status):
            self._buffer.append(indata.copy())

        try:
            with sd.InputStream(
                device=mic_idx,
                channels=1,
                samplerate=sample_rate,
                callback=callback,
            ):
                while not self._stop.is_set():
                    time.sleep(0.05)
        except Exception as exc:
            self._set_error(f"Error MIC: {exc}")

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._last_error = message
            self._recording = False


RECORDER = ChatRecorder()


def start_recording(mic_idx: int | None) -> tuple[bool, str]:
    return RECORDER.start(mic_idx)


def stop_recording() -> tuple[bool, str, Optional[np.ndarray], Optional[int]]:
    return RECORDER.stop()


def is_recording() -> bool:
    return RECORDER.is_recording()
