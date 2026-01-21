from __future__ import annotations

import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path


def _wait_port(host: str, port: int, timeout: float = 120.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(0.2)
    return False


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 7862
    url = f"http://127.0.0.1:{port}"

    proc = subprocess.Popen(
        [sys.executable, "tools/run_webui.py", "--device", "auto", "--server-port", str(port)],
        cwd=str(repo_root),
    )
    if _wait_port("127.0.0.1", port):
        try:
            import webview  # type: ignore

            webview.create_window("AgenteSmith Panel", url)
            webview.start()
        except Exception:
            webbrowser.open(url)
    else:
        print("WARNING: panel did not start in time. Opening anyway.")
        webbrowser.open(url)
    return proc.wait()


if __name__ == "__main__":
    raise SystemExit(main())
