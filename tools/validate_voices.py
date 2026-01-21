import sys
from pathlib import Path

EXPECTED_PYTHON = r"C:\Users\Usuario\fish-speech\fishspeech_env\Scripts\python.exe"


def _require_venv():
    exe = Path(sys.executable).resolve()
    expected = Path(EXPECTED_PYTHON).resolve()
    if exe != expected:
        print("ERROR: validate_voices.py must be executed with the venv python.")
        print(f"Expected: {expected}")
        print(f"Current:  {exe}")
        print(
            f"Run: {expected} tools\\validate_voices.py",
        )
        raise SystemExit(1)


def main() -> int:
    _require_venv()
    try:
        from tools.voices import load_voices
    except Exception as exc:
        print("ERROR: voices loader not available.")
        print(f"Details: {exc}")
        return 1

    valid, invalid = load_voices()
    if invalid:
        print("INVALID")
        for entry in invalid:
            name = entry.get("name") or entry.get("id") or "<unknown>"
            reason = entry.get("reason", "unknown")
            print(f"- {name}: {reason}")
        return 1
    print("OK")
    print(f"voices={len(valid)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
