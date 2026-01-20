# Contexto (Windows)

Objetivo: ejecutar AgenteSmith.py en Windows usando el venv:
C:\Users\Usuario\fish-speech\fishspeech_env\Scripts\python.exe

Arreglos requeridos:
- UTF-8/acentos (PYTHONIOENCODING=utf-8 + normalizar NFC).
- Fijar micro (MIC index 18) y sistema (SYS = CABLE Output) y evitar duplicado MIC+SYS.
- WhisperLive debe ejecutarse con el Python del venv (evitar ModuleNotFoundError: requests).
- Selección de dispositivos por ÍNDICE (no por nombre ambiguo DirectSound/WASAPI/WDM-KS).

Rutas actuales:
- AgenteSmith.py: C:\Users\Usuario\Documents\whisper_live\AgenteSmith.py
- whisper_live.py: C:\Users\Usuario\Documents\whisper_live\whisper_live.py
- Python venv: C:\Users\Usuario\fish-speech\fishspeech_env\Scripts\python.exe
- Modelo Fish: C:\Users\Usuario\fish-speech\checkpoints\openaudio-s1-mini\model.pth

Variables de entorno esperadas:
- MIC_DEV_INDEX (default 18)
- SYS_DEV_INDEX (default: índice del CABLE Output)
- CAPTURE_SYS (default 1)
