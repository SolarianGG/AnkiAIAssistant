import httpx
import openai
from smolagents.monitoring import AgentLogger, LogLevel
from smolagents.utils import AgentGenerationError

from anki_ai_assistant.core.config import ConfigError
from anki_ai_assistant.core.errors import error_message
from anki_ai_assistant.core.logs import setup_logging, short

OLLAMA = "http://localhost:11434/v1/chat/completions"
OPENROUTER = "https://openrouter.ai/api/v1/chat/completions"


def wrapped(cause: Exception) -> AgentGenerationError:
    """What smolagents raises out of run() when the model request fails."""
    try:
        try:
            raise cause
        except Exception as e:
            raise AgentGenerationError(f"Error while generating output:\n{e}", AgentLogger(LogLevel.OFF)) from e
    except AgentGenerationError as error:
        return error


def status_error(cls, url: str, status: int, message: str):
    request = httpx.Request("POST", url)
    return cls(message, response=httpx.Response(status, request=request), body={"error": {"message": message}})


def test_ollama_not_running():
    error = wrapped(openai.APIConnectionError(request=httpx.Request("POST", OLLAMA)))
    text = error_message(error)
    assert "Ollama is not reachable at localhost:11434" in text and "ollama serve" in text


def test_other_local_server_not_running():
    text = error_message(wrapped(openai.APIConnectionError(request=httpx.Request("POST", "http://127.0.0.1:8080/v1"))))
    assert "Nothing answers at 127.0.0.1:8080" in text


def test_cloud_connection_and_timeout():
    assert "Cannot connect" in error_message(wrapped(openai.APIConnectionError(request=httpx.Request("POST", OPENROUTER))))
    assert "did not answer in time" in error_message(
        wrapped(openai.APITimeoutError(request=httpx.Request("POST", OPENROUTER))))


def test_bad_key():
    text = error_message(wrapped(status_error(openai.AuthenticationError, OPENROUTER, 401, "No auth credentials")))
    assert "openrouter.ai" in text and "API key" in text


def test_rate_limit_keeps_provider_message():
    text = error_message(wrapped(status_error(openai.RateLimitError, OPENROUTER, 429, "Rate limit exceeded: free-models-per-day")))
    assert "free-models-per-day" in text and "another model" in text


def test_unknown_ollama_model():
    text = error_message(wrapped(status_error(openai.NotFoundError, OLLAMA, 404, 'model "qwen9" not found')))
    assert "ollama pull" in text


def test_context_limit():
    text = error_message(wrapped(status_error(
        openai.BadRequestError, OPENROUTER, 400, "This endpoint's maximum context length is 32768 tokens")))
    assert "new dialog" in text


def test_config_error_as_is_and_unknown_error_with_type():
    assert error_message(ConfigError("No API key for the provider 'x'.")) == "No API key for the provider 'x'."
    assert error_message(ValueError("boom")) == "ValueError: boom"


def test_log_file(tmp_path):
    path = setup_logging(tmp_path / "logs")
    assert setup_logging(tmp_path / "logs") == path
    from anki_ai_assistant.core.logs import log

    log.info("hello")
    for handler in log.handlers:
        handler.flush()
    assert "hello" in path.read_text(encoding="utf-8")
    assert short("x" * 10, 4) == "xxxx… [6 more characters]"
    for handler in list(log.handlers):
        log.removeHandler(handler)
        handler.close()
