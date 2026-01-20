import os
import subprocess
import threading
import unicodedata

import sounddevice as sd

PYTHON_EXE = r"C:\Users\Usuario\fish-speech\fishspeech_env\Scripts\python.exe"
BASE = os.path.dirname(__file__)
WHISPER = os.path.join(BASE, "whisper_live.py")


def _parse_env_int(name):
    value = os.environ.get(name)
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        print(f"WARN: {name}={value} no es entero; usando default")
        return None


def _parse_env_bool(name, default):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _hostapi_name(hostapis, idx):
    try:
        return hostapis[idx]["name"]
    except (IndexError, KeyError, TypeError):
        return "Unknown"


def _print_device_table(devices, hostapis):
    print("Dispositivos disponibles:")
    for i, dev in enumerate(devices):
        host = _hostapi_name(hostapis, dev.get("hostapi"))
        name = dev.get("name", "Unknown")
        print(f"{i:>2} | {host:<8} | {name}")


def _find_sys_default(devices, hostapis):
    matches = []
    for i, dev in enumerate(devices):
        name = dev.get("name", "")
        if "cable output" in name.lower():
            matches.append(i)
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    for idx in matches:
        host = _hostapi_name(hostapis, devices[idx].get("hostapi"))
        if "WASAPI" in host.upper():
            return idx
    return matches[0]


def _validate_index(idx, devices, label):
    if idx is None:
        return None
    if 0 <= idx < len(devices):
        return idx
    print(f"WARN: {label} index {idx} no existe; usando default")
    return None


def _device_name(devices, idx):
    if idx is None:
        return "default"
    try:
        return devices[idx].get("name", "Unknown")
    except (IndexError, TypeError):
        return "Unknown"


def _reader_loop(proc, capture_sys):
    if proc.stdout is None:
        return
    for line in proc.stdout:
        if not line:
            break
        line = unicodedata.normalize("NFC", line.rstrip("\r\n"))
        if line.startswith("TRANSCRIPCION[mic]"):
            msg = line.split("]", 1)[1].strip()
            print(f"[mic] {msg}")
            continue
        if line.startswith("TRANSCRIPCION[sys]"):
            if not capture_sys:
                continue
            msg = line.split("]", 1)[1].strip()
            print(f"[sys] {msg}")
            continue
        print(line)


def main():
    print("Arrancando AgenteSmith...")

    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    _print_device_table(devices, hostapis)

    mic_idx = _parse_env_int("MIC_DEV_INDEX")
    if mic_idx is None:
        mic_idx = 18
    mic_idx = _validate_index(mic_idx, devices, "MIC_DEV_INDEX")

    sys_idx = _parse_env_int("SYS_DEV_INDEX")
    if sys_idx is None:
        sys_idx = _find_sys_default(devices, hostapis)
    sys_idx = _validate_index(sys_idx, devices, "SYS_DEV_INDEX")

    capture_sys = _parse_env_bool("CAPTURE_SYS", True)

    mic_name = _device_name(devices, mic_idx)
    sys_name = _device_name(devices, sys_idx)
    mic_label = mic_idx if mic_idx is not None else "default"
    sys_label = sys_idx if sys_idx is not None else "default"

    print(f"MIC index={mic_label} name={mic_name}")
    if capture_sys:
        print(f"SYS index={sys_label} name={sys_name}")
    else:
        print(f"SYS captura desactivada (CAPTURE_SYS=0), index={sys_label} name={sys_name}")

    if capture_sys and mic_idx is not None and sys_idx is not None and mic_idx == sys_idx:
        print("WARN: MIC y SYS usan el mismo dispositivo; duplicado probable")
    if capture_sys and "cable output" in sys_name.lower():
        print("WARN: SYS es cable virtual; si Windows enruta el micro al cable, habra duplicado")

    child_env = os.environ.copy()
    child_env["PYTHONIOENCODING"] = "utf-8"
    child_env["MIC_DEV_INDEX"] = str(mic_label)
    child_env["SYS_DEV_INDEX"] = str(sys_label)
    child_env["CAPTURE_SYS"] = "1" if capture_sys else "0"

    proc = subprocess.Popen(
        [PYTHON_EXE, WHISPER],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=child_env,
    )

    reader = threading.Thread(target=_reader_loop, args=(proc, capture_sys), daemon=True)
    reader.start()

    try:
        input("Ctrl+C para salir")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        if proc.poll() is None:
            proc.terminate()


if __name__ == "__main__":
    main()
