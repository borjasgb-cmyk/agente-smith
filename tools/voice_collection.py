import json
import time
from pathlib import Path

import numpy as np
import torch
import torchaudio

MAX_CLIPS = 200
MAX_SECONDS = 30 * 60
TARGET_SR = 16000


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


def save_clip(audio: np.ndarray, sample_rate: int, transcript: str | None) -> dict:
    raw_dir, manifest = ensure_dirs()
    audio = _normalize_audio(audio)
    audio = _resample_audio(audio, sample_rate)
    duration = float(audio.shape[0]) / TARGET_SR if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0

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

