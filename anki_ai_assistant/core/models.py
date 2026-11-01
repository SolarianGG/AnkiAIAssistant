"""LLMs for the agent roles. Every provider (OpenRouter, Ollama, ...) is reached through its OpenAI-compatible API."""

import os
from typing import Any

from smolagents import OpenAIServerModel

from .config import ConfigError


def build_model(config: dict[str, Any], role: str) -> OpenAIServerModel:
    role_config = config["roles"].get(role)
    if not role_config:
        raise ConfigError(f"No settings for the '{role}' role in the add-on config.")
    provider_name = role_config.get("provider", "")
    provider = config["providers"].get(provider_name)
    if provider is None:
        raise ConfigError(f"Role '{role}' uses the provider '{provider_name}', which is not in 'providers'.")
    model_id = role_config.get("model")
    if not model_id:
        raise ConfigError(f"No model is set for the '{role}' role.")
    api_key = provider.get("api_key") or os.environ.get(f"{provider_name.upper()}_API_KEY")
    if not api_key:
        raise ConfigError(f"No API key for the provider '{provider_name}'. Set it in the add-on config.")

    kwargs = {}
    if role_config.get("reasoning_effort"):
        kwargs["reasoning_effort"] = role_config["reasoning_effort"]
    return OpenAIServerModel(
        model_id=model_id,
        api_base=provider["api_base"],
        api_key=api_key,
        # A hung request once blocked a run for 14 minutes: time out and resend instead
        client_kwargs={"timeout": float(config["llm_timeout"]), "max_retries": int(config["llm_max_retries"])},
        **kwargs,
    )
