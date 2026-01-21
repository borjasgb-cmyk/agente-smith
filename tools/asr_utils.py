from __future__ import annotations

from functools import lru_cache
from typing import Optional

import numpy as np
from faster_whisper import WhisperModel


@lru_cache(maxsize=1)
def _get_model(model_name: str) -> WhisperModel:
    return WhisperModel(model_name, device="cpu", compute_type="int8")


def transcribe_audio(audio: np.ndarray, sample_rate: int, model_name: str) -> str:
    if audio.size == 0 or sample_rate <= 0:
        return ""
    model = _get_model(model_name)
    segments, _info = model.transcribe(
        audio,
        language="es",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 400},
        beam_size=5,
    )
    text_parts = [seg.text.strip() for seg in segments if seg.text.strip()]
    return " ".join(text_parts)
