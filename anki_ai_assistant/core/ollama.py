"""A context window size for Ollama models.

Ollama's OpenAI-compatible endpoint ignores num_ctx in a request and loads every model with the server default
(OLLAMA_CONTEXT_LENGTH). A model whose Modelfile sets num_ctx keeps it there, though. So for a role with
context_length the add-on creates a derived model "<model>-ctx<N>" once: it shares the weights of the original
(nothing is copied or downloaded) and only adds the parameter. Then requests go to that model."""

import json
import urllib.error
import urllib.request
from typing import Any

from .config import ConfigError

TIMEOUT = 60


def _native_base(api_base: str) -> str:
    base = api_base.rstrip("/")
    return base[:-3] if base.endswith("/v1") else base


def _post(url: str, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        return json.loads(response.read() or b"{}")


def _show(base: str, model: str) -> dict[str, Any] | None:
    """The model's details, or None if Ollama has no such model."""
    try:
        return _post(f"{base}/api/show", {"model": model})
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def _max_context(details: dict[str, Any]) -> int | None:
    for key, value in (details.get("model_info") or {}).items():
        if key.endswith(".context_length") and isinstance(value, int):
            return value
    return None


def model_with_context(api_base: str, model: str, context_length: int) -> str:
    """The name of a model like `model` with the given context size, created in Ollama if needed."""
    base = _native_base(api_base)
    derived = f"{model}-ctx{context_length}"
    try:
        if _show(base, derived) is not None:
            return derived
        details = _show(base, model)
        if details is None:
            raise ConfigError(f"Ollama has no model '{model}'. Download it with: ollama pull {model}")
        limit = _max_context(details)
        if limit is not None and context_length > limit:
            raise ConfigError(f"context_length {context_length} is more than '{model}' supports ({limit}).")
        _post(f"{base}/api/create", {"model": derived, "from": model,
                                     "parameters": {"num_ctx": context_length}, "stream": False})
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise ConfigError(f"context_length works only with Ollama, and Ollama at {base} did not answer as expected "
                          f"({e}). Set context_length to 0 for other providers, or check that Ollama is running.")
    return derived
