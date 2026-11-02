"""The add-on's log: requests, agent steps, answers and errors with tracebacks, for finding out what went wrong.

Anki: user_files/logs/assistant.log (user_files survives add-on updates). The file rotates at 1 MB, 3 old files kept."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

LOGGER_NAME = "anki_ai_assistant"
LOG_FILE_NAME = "assistant.log"
# Tool results and model texts can be long (a whole web page): the log keeps the start
MAX_TEXT = 2000

log = logging.getLogger(LOGGER_NAME)


def setup_logging(log_dir: Path) -> Path:
    """Write the add-on's log into log_dir. Calling it again does not add a second handler."""
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / LOG_FILE_NAME
    for handler in log.handlers:
        if isinstance(handler, RotatingFileHandler) and Path(handler.baseFilename) == path.resolve():
            return path
    handler = RotatingFileHandler(path, maxBytes=2**20, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(threadName)s: %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    # Not into Anki's own logging: its console would show every step
    log.propagate = False
    return path


def short(value: Any, limit: int = MAX_TEXT) -> str:
    text = str(value)
    return text if len(text) <= limit else f"{text[:limit]}… [{len(text) - limit} more characters]"


def log_step(step: Any) -> None:
    """One line per agent step: its number, the tokens, the tool calls and their results, the error."""
    from smolagents.memory import ActionStep

    if not isinstance(step, ActionStep):
        return
    parts = [f"step {step.step_number}"]
    usage = getattr(step, "token_usage", None)
    if usage is not None:
        parts.append(f"tokens in {usage.input_tokens} out {usage.output_tokens}")
    timing = getattr(step, "timing", None)
    if timing is not None and timing.duration is not None:
        parts.append(f"{timing.duration:.1f} s")
    lines = [", ".join(parts)]
    for call in step.tool_calls or []:
        lines.append(f"  call {call.name}({short(call.arguments, 500)})")
    if step.observations:
        lines.append(f"  result {short(step.observations)}")
    if not step.tool_calls and step.model_output:
        lines.append(f"  model {short(step.model_output)}")
    if step.error is not None:
        lines.append(f"  error {short(step.error)}")
    log.info("\n".join(lines))
