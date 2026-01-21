import os
import threading
import time
from collections import deque
from pathlib import Path
from subprocess import PIPE, STDOUT, Popen
from typing import Deque

import logging
from logging.handlers import RotatingFileHandler


REPO_ROOT = Path(__file__).resolve().parents[1]
VENV_PY = r"C:\Users\Usuario\fish-speech\fishspeech_env\Scripts\python.exe"
AGENT_SCRIPT = str(REPO_ROOT / "AgenteSmith.py")


def _setup_logger() -> logging.Logger:
    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "agent.log"

    logger = logging.getLogger("agent_supervisor")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler(
        log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8"
    )
    formatter = logging.Formatter("%(asctime)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class AgentSupervisor:
    def __init__(self) -> None:
        self._process: Popen | None = None
        self._reader: threading.Thread | None = None
        self._lock = threading.Lock()
        self._lines: Deque[str] = deque(maxlen=200)
        self._mic_lines: Deque[str] = deque(maxlen=10)
        self._sys_lines: Deque[str] = deque(maxlen=10)
        self._warnings: Deque[str] = deque(maxlen=10)
        self._last_config: dict = {}
        self._logger = _setup_logger()

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self, mic_idx: int | None, sys_idx: int | None, mode: str) -> str:
        with self._lock:
            if self.is_running():
                return "Agente ya esta en ejecucion"

            env = os.environ.copy()
            if mic_idx is not None:
                env["MIC_DEV_INDEX"] = str(mic_idx)
            if sys_idx is not None:
                env["SYS_DEV_INDEX"] = str(sys_idx)

            capture_sys = "1"
            if mode == "MIC only":
                capture_sys = "0"
            env["CAPTURE_SYS"] = capture_sys
            env["PYTHONIOENCODING"] = "utf-8"

            self._last_config = {
                "mic_idx": mic_idx,
                "sys_idx": sys_idx,
                "mode": mode,
                "capture_sys": capture_sys,
            }

            self._process = Popen(
                [VENV_PY, AGENT_SCRIPT],
                cwd=str(REPO_ROOT),
                env=env,
                stdout=PIPE,
                stderr=STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            self._reader = threading.Thread(target=self._reader_loop, daemon=True)
            self._reader.start()
            return "Agente iniciado"

    def stop(self) -> str:
        with self._lock:
            if not self.is_running():
                return "Agente no esta en ejecucion"
            assert self._process is not None
            self._process.terminate()
        time.sleep(0.5)
        with self._lock:
            if self.is_running():
                assert self._process is not None
                self._process.kill()
        return "Agente detenido"

    def _reader_loop(self) -> None:
        if not self._process or self._process.stdout is None:
            return
        for line in self._process.stdout:
            if not line:
                break
            clean = line.rstrip("\r\n")
            self._logger.info(clean)
            self._lines.append(clean)
            lower = clean.lower()
            if "[mic]" in lower or "transcripcion[mic]" in lower:
                self._mic_lines.append(clean)
            if "[sys]" in lower or "transcripcion[sys]" in lower:
                self._sys_lines.append(clean)
            if "warn" in lower:
                self._warnings.append(clean)

    def status(self) -> dict:
        return {
            "running": self.is_running(),
            "pid": self._process.pid if self._process else None,
            "lines": list(self._lines),
            "mic_lines": list(self._mic_lines),
            "sys_lines": list(self._sys_lines),
            "warnings": list(self._warnings),
            "config": dict(self._last_config),
        }


SUPERVISOR = AgentSupervisor()
