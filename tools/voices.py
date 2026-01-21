import json
from pathlib import Path

REQUIRED_FIELDS = {
    "id",
    "name",
    "gender",
    "accent",
    "type",
    "model_path",
    "ref_audio",
    "enabled",
}

ALLOWED_GENDERS = {"male", "female"}
ALLOWED_TYPES = {"cloned", "preset"}


def find_repo_root(start: str | None = None) -> Path:
    base = Path(start or __file__).resolve()
    for parent in [base] + list(base.parents):
        if (parent / ".project-root").exists():
            return parent
    return Path.cwd()


def _invalid_entry(id_value: str, name: str, reason: str) -> dict:
    return {"id": id_value, "name": name, "reason": reason}


def load_voices(path: str | None = None) -> tuple[list[dict], list[dict]]:
    root = find_repo_root()
    voices_path = Path(path) if path else (root / "voices.json")

    if not voices_path.exists():
        return [], [_invalid_entry("voices.json", "voices.json", "not found")]

    try:
        data = json.loads(voices_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [], [_invalid_entry("voices.json", "voices.json", f"invalid json: {exc}")]

    if not isinstance(data, list):
        return [], [_invalid_entry("voices.json", "voices.json", "root must be a list")]

    seen_ids: set[str] = set()
    valid: list[dict] = []
    invalid: list[dict] = []

    for entry in data:
        if not isinstance(entry, dict):
            invalid.append(_invalid_entry("<unknown>", "<unknown>", "entry is not a dict"))
            continue

        entry_id = str(entry.get("id", "")).strip()
        entry_name = str(entry.get("name", "")).strip() or entry_id or "<unknown>"
        reasons: list[str] = []

        missing = REQUIRED_FIELDS - entry.keys()
        if missing:
            reasons.append("missing fields: " + ", ".join(sorted(missing)))

        if not entry_id:
            reasons.append("id is required")
        elif entry_id in seen_ids:
            reasons.append("duplicate id")
        else:
            seen_ids.add(entry_id)

        gender = entry.get("gender")
        if gender not in ALLOWED_GENDERS:
            reasons.append("gender must be male or female")

        voice_type = entry.get("type")
        if voice_type not in ALLOWED_TYPES:
            reasons.append("type must be cloned or preset")

        accent = entry.get("accent")
        if not isinstance(accent, str) or not accent.strip():
            reasons.append("accent is required")

        model_path = entry.get("model_path")
        ref_audio = entry.get("ref_audio")
        enabled = entry.get("enabled")

        if not isinstance(model_path, str) or not model_path.strip():
            reasons.append("model_path is required")
        if not isinstance(ref_audio, str) or not ref_audio.strip():
            reasons.append("ref_audio is required")

        if enabled is not True:
            reasons.append("disabled")

        if isinstance(model_path, str) and model_path.strip():
            model_p = Path(model_path)
            if model_p.is_absolute():
                reasons.append("model_path must be relative")
            elif not (root / model_p).exists():
                reasons.append("model_path not found")

        if isinstance(ref_audio, str) and ref_audio.strip():
            ref_p = Path(ref_audio)
            if ref_p.is_absolute():
                reasons.append("ref_audio must be relative")
            elif not (root / ref_p).exists():
                reasons.append("ref_audio not found")

        if reasons:
            invalid.append(_invalid_entry(entry_id or "<unknown>", entry_name, "; ".join(reasons)))
        else:
            valid.append(entry)

    return valid, invalid


def load_config(path: str | None = None) -> dict:
    root = find_repo_root()
    config_path = Path(path) if path else (root / "config.json")
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_config(voice_id: str | None, path: str | None = None) -> None:
    root = find_repo_root()
    config_path = Path(path) if path else (root / "config.json")
    payload = {"voice_id": voice_id}
    config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
