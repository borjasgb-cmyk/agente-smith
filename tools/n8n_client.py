import json
import time
from dataclasses import dataclass
from typing import Any
from urllib import request, error


@dataclass
class N8NConfig:
    enabled: bool = False
    base_url: str | None = None
    webhook_url: str | None = None
    api_key: str | None = None
    open_url: str | None = None


_CONFIG = N8NConfig()


def set_config(**kwargs) -> None:
    for key, value in kwargs.items():
        if hasattr(_CONFIG, key):
            setattr(_CONFIG, key, value or None)


def get_config() -> N8NConfig:
    return _CONFIG


def _post_json(url: str, payload: dict, headers: dict[str, str] | None = None) -> tuple[bool, str]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        with request.urlopen(req, timeout=5) as resp:
            return 200 <= resp.status < 300, f"HTTP {resp.status}"
    except error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except Exception as exc:
        return False, f"ERROR {exc}"


def _get(url: str, headers: dict[str, str] | None = None) -> tuple[bool, str]:
    req = request.Request(url, method="GET")
    if headers:
        for k, v in headers.items():
            req.add_header(k, v)
    try:
        with request.urlopen(req, timeout=5) as resp:
            return 200 <= resp.status < 300, f"HTTP {resp.status}"
    except error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except Exception as exc:
        return False, f"ERROR {exc}"


def test_connection() -> tuple[bool, str]:
    cfg = _CONFIG
    if cfg.webhook_url:
        payload = {"event": "test", "timestamp": time.time()}
        return _post_json(cfg.webhook_url, payload)
    if cfg.base_url and cfg.api_key:
        url = cfg.base_url.rstrip("/") + "/rest/settings"
        return _get(url, headers={"X-N8N-API-KEY": cfg.api_key})
    return False, "No webhook URL. Set N8N_WEBHOOK_URL."


def emit_event(event: str, payload: dict[str, Any]) -> tuple[bool, str]:
    cfg = _CONFIG
    if not cfg.enabled:
        return False, "disabled"
    enriched = {
        "event": event,
        "timestamp": time.time(),
        "payload": payload,
    }
    if cfg.webhook_url:
        return _post_json(cfg.webhook_url, enriched)
    return False, "No webhook URL. Use N8N_WEBHOOK_URL."
