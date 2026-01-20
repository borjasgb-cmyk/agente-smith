import sounddevice as sd
from faster_whisper import WhisperModel
import sys

MIC_DEV = 18
SAMPLE_RATE = 44100

print(f"MIC dev={MIC_DEV} sr={SAMPLE_RATE} ch=1 (FORZADO)")

model = WhisperModel("base", device="cuda", compute_type="float16")

with sd.InputStream(device=MIC_DEV, channels=1, samplerate=SAMPLE_RATE):
    print("Escuchando... Ctrl+C para salir")
    while True:
        pass
