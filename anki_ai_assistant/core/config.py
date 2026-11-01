import copy
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
    return _from_manager_presets(_merge(defaults, user_config or {}))


def _from_manager_presets(config: dict) -> dict:
    """Older versions saved other manager models alone ("manager_presets"): each becomes a profile
    with the current searcher and validator."""
    presets = config.pop("manager_presets", None) or []
    manager = config["roles"].get("manager") or {}
    if "name" in manager:
        manager = dict(manager)
        config["profile_name"] = config.get("profile_name") or str(manager.pop("name") or "")
        config["roles"] = {**config["roles"], "manager": manager}
    if presets and not config.get("profiles"):
        profiles = []
        for preset in presets:
            if not isinstance(preset, dict):
                continue
            preset = dict(preset)
            name = str(preset.pop("name", "") or "")
            new_manager = {**{k: v for k, v in manager.items() if k == "max_steps"}, **preset}
            profiles.append({"name": name, "roles": {**copy.deepcopy(config["roles"]), "manager": new_manager}})
        config["profiles"] = profiles
    return config


ROLES = ("manager", "searcher", "validator")


# Profiles: saved sets of the three roles' models. config["roles"] is the profile in use (named
# config["profile_name"]), config["profiles"] the others: {"name": "...", "roles": {"manager": {...}, ...}}

def model_label(provider: str, model: str) -> str:
    return f"{provider} / {model}"


def profile_label(profile: dict) -> str:
    """How a profile is shown: its name, or the manager's "provider / model" without one."""
    name = str(profile.get("name") or "").strip()
    if name:
        return name
    manager = (profile.get("roles") or {}).get("manager") or {}
    return model_label(manager.get("provider", ""), manager.get("model", ""))


def current_profile(config: dict) -> dict:
    return {"name": config.get("profile_name", ""), "roles": copy.deepcopy(config["roles"])}


def profile_choices(config: dict) -> list[dict]:
    """The profiles to pick from: the one in use first, then the saved ones, no two with the same label."""
    choices = [current_profile(config)]
    for profile in config.get("profiles") or []:
        if isinstance(profile, dict) and isinstance(profile.get("roles"), dict) \
                and all(profile_label(c) != profile_label(profile) for c in choices):
            choices.append(copy.deepcopy(profile))
    return choices


def with_profiles(config: dict, profiles: list[dict], active: int) -> dict:
    """The config with these profiles: profiles[active] is used, the others are saved for later.

    A role the profile has no settings for keeps the current ones."""
    profile = profiles[active]
    return {**config,
            "roles": {**config["roles"], **copy.deepcopy(profile["roles"])},
            "profile_name": str(profile.get("name") or "").strip(),
            "profiles": [copy.deepcopy(p) for index, p in enumerate(profiles) if index != active]}


def with_profile(config: dict, choice: dict) -> dict:
    """The config with the chosen profile in use. The one used before is saved with all its settings,
    so switching back restores them."""
    choices = profile_choices(config)
    labels = [profile_label(c) for c in choices]
    if profile_label(choice) not in labels:
        return config
    return with_profiles(config, choices, labels.index(profile_label(choice)))


def duplicate_labels(profiles: list[dict]) -> list[str]:
    """Labels shown for more than one profile, which the chat could not tell apart."""
    labels = [profile_label(p) for p in profiles]
    return sorted({label for label in labels if labels.count(label) > 1})


def _role_problems(roles: dict, providers: dict, where: str) -> list[str]:
    problems = []
    for role in ROLES:
        role_config = roles.get(role) or {}
        if role_config.get("provider") not in providers:
            problems.append(f"{where}role '{role}': choose one of the providers.")
        if not str(role_config.get("model", "")).strip():
            problems.append(f"{where}role '{role}': the model is empty.")
        if "extra_body" in role_config and not isinstance(role_config["extra_body"], dict):
            problems.append(f"{where}role '{role}': extra_body must be a JSON object, "
                            f"e.g. {{\"provider\": {{...}}}}.")
    return problems


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
    for profile in [current_profile(config)] + list(config.get("profiles") or []):
        problems += _role_problems(profile.get("roles") or {}, providers, f"Profile '{profile_label(profile)}', ")
    return problems
