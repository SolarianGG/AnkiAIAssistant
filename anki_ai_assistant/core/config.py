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


ROLES = ("manager", "searcher", "validator")


def check_config(config: dict[str, Any]) -> list[str]:
    """Problems that would stop the assistant, in words for the settings dialog. Empty if the config is usable.

    A missing API key is not one: it may come from the environment, and build_model reports it when used."""
    problems = []
    providers = config.get("providers") or {}
    for name, provider in providers.items():
        if not name.strip():
            problems.append("A provider has no name.")
        elif not str(provider.get("api_base", "")).startswith(("http://", "https://")):
            problems.append(f"Provider '{name}': the API base must be an http(s) URL ending with /v1.")
    for role in ROLES:
        role_config = (config.get("roles") or {}).get(role) or {}
        if role_config.get("provider") not in providers:
            problems.append(f"Role '{role}': choose one of the providers.")
        if not str(role_config.get("model", "")).strip():
            problems.append(f"Role '{role}': the model is empty.")
        if "extra_body" in role_config and not isinstance(role_config["extra_body"], dict):
            problems.append(f"Role '{role}': extra_body must be a JSON object, e.g. {{\"provider\": {{...}}}}.")
    for preset in config.get("manager_presets") or []:
        if preset.get("provider") not in providers:
            problems.append(f"Chat model '{preset.get('model', '')}': the provider "
                            f"'{preset.get('provider', '')}' is not in the providers list.")
    return problems
