from __future__ import annotations

import json
from urllib import request


def generate_reply(prompt: str, endpoint: str | None, api_key: str | None) -> str:
    prompt = prompt.strip()
    if not prompt:
        return ""
    if not endpoint:
        return f"Echo: {prompt}"

    payload = {
        "prompt": prompt,
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(endpoint, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8", errors="ignore")
        try:
            parsed = json.loads(raw)
        except Exception:
            return raw.strip()
        for key in ("text", "reply", "response", "output"):
            if key in parsed and isinstance(parsed[key], str):
                return parsed[key].strip()
        return raw.strip()
    except Exception as exc:
        return f"Echo: {prompt} (LLM error: {exc})"
