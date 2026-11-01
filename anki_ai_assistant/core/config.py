import json
from pathlib import Path
from typing import Any

DEFAULTS_PATH = Path(__file__).resolve().parent.parent / "config.json"


class ConfigError(Exception):
    """A setting is missing or invalid. The message is shown to the user as is."""


def _merge(defaults: dict, overrides: dict) -> dict:
    merged = dict(defaults)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(user_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the user's config on top of the defaults from config.json.

    Anki returns the user's config whole and ignores keys added to config.json later,
    so merge to keep new settings working after an update."""
    defaults = json.loads(DEFAULTS_PATH.read_text(encoding="utf-8"))
    return _merge(defaults, user_config or {})
