import json
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
import torchaudio

MAX_CLIPS = 200
MAX_SECONDS = 30 * 60
TARGET_SR = 16000
MIN_CLIP_SECONDS = 5.0
MAX_CLIP_SECONDS = 10.0
VAD_RMS_THRESHOLD = 0.008
SILENCE_HOLD_SECONDS = 0.6
PREBUFFER_SECONDS = 0.3


def find_repo_root(start: str | None = None) -> Path:
    base = Path(start or __file__).resolve()
    for parent in [base] + list(base.parents):
        if (parent / ".project-root").exists():
            return parent
    return Path.cwd()


def data_paths() -> tuple[Path, Path]:
    root = find_repo_root()
    raw_dir = root / "data" / "voice_user" / "raw"
    manifest = root / "data" / "voice_user" / "manifest.jsonl"
    return raw_dir, manifest


def ensure_dirs() -> tuple[Path, Path]:
    raw_dir, manifest = data_paths()
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    return raw_dir, manifest


def load_manifest_stats() -> tuple[int, float]:
    _, manifest = data_paths()
    if not manifest.exists():
        return 0, 0.0
    count = 0
    seconds = 0.0
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except Exception:
                continue
            count += 1
            seconds += float(entry.get("duration", 0.0) or 0.0)
    return count, seconds


def clear_dataset() -> None:
    raw_dir, manifest = data_paths()
    if raw_dir.exists():
        for path in raw_dir.glob("*.wav"):
            try:
                path.unlink()
            except Exception:
                pass
    if manifest.exists():
        try:
            manifest.unlink()
        except Exception:
            pass


def _normalize_audio(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def _resample_audio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if sample_rate == TARGET_SR:
        return audio
    waveform = torch.from_numpy(audio).unsqueeze(0)
    resampler = torchaudio.transforms.Resample(
        orig_freq=sample_rate, new_freq=TARGET_SR
    )
    with torch.no_grad():
        resampled = resampler(waveform)
    return resampled.squeeze(0).cpu().numpy()


def save_clip(audio: np.ndarray, sample_rate: int, transcript: str | None) -> dict | None:
    raw_dir, manifest = ensure_dirs()
    ok, _message = can_accept_more()
    if not ok:
        return None
    audio = _normalize_audio(audio)
    audio = _resample_audio(audio, sample_rate)
    duration = float(audio.shape[0]) / TARGET_SR if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0

    if duration < MIN_CLIP_SECONDS or _is_silent(rms, peak):
        return None

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"clip_{timestamp}_{int(time.time() * 1000) % 100000}.wav"
    path = raw_dir / filename

    waveform = torch.from_numpy(audio).unsqueeze(0)
    torchaudio.save(str(path), waveform, TARGET_SR)

    entry = {
        "path": str(path.relative_to(find_repo_root())),
        "duration": duration,
        "rms": rms,
        "peak": peak,
    }
    if transcript:
        entry["transcript"] = transcript

    with (raw_dir.parent / "manifest.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return entry


def can_accept_more() -> tuple[bool, str]:
    count, seconds = load_manifest_stats()
    if count >= MAX_CLIPS:
        return False, f"Limite alcanzado: {count}/{MAX_CLIPS} clips"
    if seconds >= MAX_SECONDS:
        return False, f"Limite alcanzado: {int(seconds)}/{MAX_SECONDS} segundos"
    return True, f"OK: {count}/{MAX_CLIPS} clips, {int(seconds)}/{MAX_SECONDS} s"


def _is_silent(rms: float, peak: float) -> bool:
    return rms < VAD_RMS_THRESHOLD and peak < 0.02


def _pick_input_rate(device: int) -> int | None:
    for rate in (48000, 44100, 32000, 16000):
        try:
            sd.check_input_settings(device=device, samplerate=rate, channels=1)
            return rate
        except Exception:
            continue
    return None


class VoiceCollector:
    def __init__(self) -> None:
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._running = False
        self._mic_idx: int | None = None
        self._last_error: str | None = None
        self._last_saved: dict | None = None

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def start(self, mic_idx: int | None) -> tuple[bool, str]:
        if mic_idx is None:
            return False, "MIC no seleccionado"
        with self._lock:
            if self._running:
                return True, "Ya esta recolectando"
            self._running = True
            self._mic_idx = mic_idx
            self._last_error = None
            self._last_saved = None
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return True, "Recolectando"

    def stop(self) -> tuple[bool, str]:
        with self._lock:
            if not self._running:
                return True, "Recolecta detenida"
            self._stop.set()
            thread = self._thread
        if thread:
            thread.join(timeout=2)
        with self._lock:
            self._running = False
        return True, "Recolecta detenida"

    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def last_saved(self) -> dict | None:
        with self._lock:
            return self._last_saved

    def _run(self) -> None:
        mic_idx = self._mic_idx
        if mic_idx is None:
            self._set_error("MIC no seleccionado")
            return
        sample_rate = _pick_input_rate(mic_idx)
        if sample_rate is None:
            self._set_error("MIC no disponible")
            return

        blocksize = int(0.1 * sample_rate)
        max_len = int(MAX_CLIP_SECONDS * sample_rate)
        min_len = int(MIN_CLIP_SECONDS * sample_rate)
        prebuffer = deque(maxlen=int(PREBUFFER_SECONDS * sample_rate))
        clip_parts: list[np.ndarray] = []
        silence_time = 0.0
        voice_active = False

        def finalize_clip() -> None:
            nonlocal clip_parts, voice_active, silence_time
            if not clip_parts:
                return
            audio = np.concatenate(clip_parts, axis=0)
            clip_parts = []
            voice_active = False
            silence_time = 0.0
            if audio.shape[0] < min_len:
                return
            ok, _message = can_accept_more()
            if not ok:
                self._stop.set()
                return
            entry = save_clip(audio, sample_rate, None)
            if entry:
                with self._lock:
                    self._last_saved = entry

        def on_audio(indata, _frames, _time, _status) -> None:
            nonlocal clip_parts, voice_active, silence_time
            if self._stop.is_set():
                return
            audio = indata.copy().squeeze().astype(np.float32)
            if audio.ndim == 0:
                audio = np.expand_dims(audio, 0)
            rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
            if voice_active:
                clip_parts.append(audio)
                if rms < VAD_RMS_THRESHOLD:
                    silence_time += audio.shape[0] / sample_rate
                else:
                    silence_time = 0.0
                total_len = sum(part.shape[0] for part in clip_parts)
                if total_len >= max_len or silence_time >= SILENCE_HOLD_SECONDS:
                    finalize_clip()
            else:
                prebuffer.extend(audio.tolist())
                if rms >= VAD_RMS_THRESHOLD:
                    voice_active = True
                    silence_time = 0.0
                    if prebuffer:
                        clip_parts.append(np.array(prebuffer, dtype=np.float32))
                    clip_parts.append(audio)
                    prebuffer.clear()

        try:
            with sd.InputStream(
                device=mic_idx,
                channels=1,
                samplerate=sample_rate,
                blocksize=blocksize,
                callback=on_audio,
            ):
                while not self._stop.is_set():
                    time.sleep(0.1)
        except Exception as exc:
            self._set_error(f"Error MIC: {exc}")
        finally:
            finalize_clip()
            with self._lock:
                self._running = False

    def _set_error(self, message: str) -> None:
        with self._lock:
            self._last_error = message
            self._running = False


COLLECTOR = VoiceCollector()


def start_collection(mic_idx: int | None) -> tuple[bool, str]:
    return COLLECTOR.start(mic_idx)


def stop_collection() -> tuple[bool, str]:
    return COLLECTOR.stop()


def is_collecting() -> bool:
    return COLLECTOR.is_running()


def last_collection_error() -> str | None:
    return COLLECTOR.last_error()


def last_saved_entry() -> dict | None:
    return COLLECTOR.last_saved()
