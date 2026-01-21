from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def _wait_port(host: str, port: int, timeout: float = 180.0) -> bool:
    deadline = time.time() + timeout
    delay = 0.2
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(delay)
            delay = min(delay * 1.5, 2.0)
    return False


def _pick_free_port(preferred: int) -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", preferred))
            return preferred
    except OSError:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]


def _port_responding(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1):
            return True
    except OSError:
        return False


def _load_lock(lock_path: Path) -> dict | None:
    if not lock_path.exists():
        return None
    try:
        return json.loads(lock_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_lock(lock_path: Path, payload: dict) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    preferred = int(sys.argv[1]) if len(sys.argv) > 1 else 7862
    data_dir = repo_root / "data"
    port_file = data_dir / "panel_port.txt"
    lock_file = data_dir / "panel.lock"

    lock = _load_lock(lock_file)
    if lock and "port" in lock:
        existing_port = int(lock["port"])
        if _port_responding(existing_port):
            url = f"http://127.0.0.1:{existing_port}"
            print(f"Panel already running on {url}")
            _open_once(url)
            return 0
        else:
            try:
                lock_file.unlink()
            except Exception:
                pass

    port = _pick_free_port(preferred)
    url = f"http://127.0.0.1:{port}"

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    proc = subprocess.Popen(
        [sys.executable, "tools/run_webui.py", "--device", "cpu", "--server-port", str(port)],
        cwd=str(repo_root),
    )
    _write_lock(lock_file, {"pid": proc.pid, "port": port})
    port_file.parent.mkdir(parents=True, exist_ok=True)
    port_file.write_text(str(port), encoding="utf-8")
    print("Warming up... waiting for panel URL.")
    if _wait_port("127.0.0.1", port):
        _open_once(url)
    else:
        print("WARNING: panel did not start in time. Opening anyway.")
        _open_once(url)
    return proc.wait()


_OPENED = False


def _open_once(url: str) -> None:
    global _OPENED
    if _OPENED:
        return
    _OPENED = True
    try:
        import webview  # type: ignore

        webview.create_window("AgenteSmith Panel", url)
        webview.start()
    except Exception:
        webbrowser.open(url)


if __name__ == "__main__":
    raise SystemExit(main())
