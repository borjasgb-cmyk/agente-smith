from __future__ import annotations

from pathlib import Path
import sys

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

from tools.tts_sapi import sapi_speak_to_wav


def main() -> int:
    out = Path("data") / "chat_tts_test.wav"
    ok, msg = sapi_speak_to_wav("Prueba de texto a voz.", str(out))
    if not ok:
        print("ERROR:", msg)
        return 1
    if not out.exists() or out.stat().st_size == 0:
        print("ERROR: output file missing")
        return 1
    print(f"OK: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
