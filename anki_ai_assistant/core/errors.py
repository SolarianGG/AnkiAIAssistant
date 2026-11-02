"""Errors of a request in words for the user: what failed and what to do about it.

smolagents wraps a failed model request in AgentGenerationError ("Error while generating output: ..."),
so the original exception is looked up in the __cause__ / __context__ chain."""

from typing import Any
from urllib.parse import urlsplit

from .config import ConfigError

OLLAMA_PORT = 11434


def _chain(error: BaseException) -> list[BaseException]:
    seen: list[BaseException] = []
    while error is not None and error not in seen:
        seen.append(error)
        error = error.__cause__ or error.__context__
    return seen


def _server(error: Any) -> str:
    """The host the failed request went to, e.g. "openrouter.ai" or "localhost:11434"."""
    try:
        return urlsplit(str(error.request.url)).netloc
    except Exception:
        return ""


def _is_ollama(server: str) -> bool:
    return server.endswith(f":{OLLAMA_PORT}")


def _is_local(server: str) -> bool:
    return server.rsplit(":", 1)[0] in ("localhost", "127.0.0.1", "[::1]", "0.0.0.0")


def _api_message(error: Any) -> str:
    """The provider's own message: OpenAI-style bodies hold it in {"error": {"message": ...}}."""
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        inner = body.get("error", body)
        if isinstance(inner, dict) and inner.get("message"):
            return str(inner["message"])
        if isinstance(inner, str):
            return inner
    return str(getattr(error, "message", "") or error)


def _looks_like_context_limit(message: str) -> bool:
    message = message.lower()
    return any(marker in message for marker in (
        "context length", "context_length", "context window", "maximum context", "too many tokens",
        "prompt is too long", "reduce the length", "exceeds the context"))


def error_message(error: BaseException) -> str:
    """One or two sentences for the chat. Unknown errors keep their type and text."""
    from .models import ContextOverflowError, ProviderError

    for cause in _chain(error):
        if isinstance(cause, (ConfigError, ContextOverflowError)):
            return str(cause)
        if isinstance(cause, ProviderError):
            return f"{cause} Try again, or pick another provider for the model (extra_body in the role settings)."

    try:
        import openai
    except ImportError:
        openai = None

    if openai is not None:
        for cause in _chain(error):
            server = _server(cause)
            where = f" ({server})" if server else ""
            if isinstance(cause, openai.APITimeoutError):
                return (f"The model server{where} did not answer in time. Try again; a slow local model may need "
                        f"a larger llm_timeout.")
            if isinstance(cause, openai.APIConnectionError):
                if _is_ollama(server):
                    return (f"Ollama is not reachable at {server}. Start it (the Ollama app or `ollama serve`) "
                            f"and try again.")
                if _is_local(server):
                    return (f"Nothing answers at {server}. If it is a local server like Ollama, start it; "
                            f"otherwise check the API base in the settings.")
                return f"Cannot connect to the model server{where}. Check the internet connection and the API base."
            if isinstance(cause, openai.AuthenticationError):
                return f"The model server{where} rejected the API key. Check the key in the add-on settings (⚙)."
            if isinstance(cause, openai.PermissionDeniedError):
                return f"The model server{where} refused the request: {_api_message(cause)}"
            if isinstance(cause, openai.RateLimitError):
                return (f"The model server{where} hit a rate or credit limit: {_api_message(cause)} "
                        f"Wait a bit, or switch to another model.")
            if isinstance(cause, openai.NotFoundError):
                hint = " Download it with `ollama pull <model>`." if _is_ollama(server) else \
                    " Check the model name in the settings."
                return f"The model server{where} does not know this model: {_api_message(cause)}.{hint}"
            if isinstance(cause, openai.BadRequestError):
                message = _api_message(cause)
                if _looks_like_context_limit(message):
                    return (f"The conversation no longer fits into the model's context: {message} "
                            f"Start a new dialog, or use a model with a larger context.")
                return f"The model server{where} rejected the request: {message}"
            if isinstance(cause, openai.APIStatusError):
                return (f"The model server{where} failed (HTTP {cause.status_code}): {_api_message(cause)} "
                        f"Try again later.")

    return f"{type(error).__name__}: {error}"
