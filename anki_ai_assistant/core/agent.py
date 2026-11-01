import importlib.resources
import json
from pathlib import Path
from typing import Any

import yaml
from smolagents import LogLevel, Tool, ToolCallingAgent
from smolagents.utils import AgentGenerationError, AgentToolCallError, AgentToolExecutionError

from .history import NativeToolHistory
from .checks import WRITE_RISKS, check_answer_has_text, check_changes_made
from .models import build_model
from .prompts import ASSISTANT_INSTRUCTIONS
from .transcript import TOOL_ERROR_PREFIX


def describe_inputs(tool: Any) -> str:
    """What a tool accepts, for an error about a wrong argument."""
    inputs = getattr(tool, "inputs", None) or {}
    if not inputs:
        return f"{tool.name} takes no arguments: call it with {{}}."
    names = [name + ("" if not spec.get("nullable") else " (optional)") for name, spec in inputs.items()]
    return f"{tool.name} takes only these arguments: {', '.join(names)}."


# The same failing call this many times in one request: the model is stuck, stop instead of burning all steps
MAX_SAME_FAILED_CALLS = 3
# The same call with the same result this many times: also stuck (e.g. a search that keeps finding nothing).
# From the second time on the result tells the model it repeats itself
MAX_SAME_RESULTS = 4
REPEAT_NOTE = ("\n(You already made this exact call in this request and got this same result. Do not repeat it: "
               "change the call, use another tool, or answer the user.)")


class RepeatedCallError(AgentGenerationError):
    """The model keeps repeating a call that fails the same way."""


class AssistantAgent(NativeToolHistory, ToolCallingAgent):
    """ToolCallingAgent whose failed tool calls are results of those calls, not errors of the whole step.

    smolagents records a step's tool calls only after all of them succeed. When one fails, e.g. on an argument
    the tool does not take, the model gets the bare error without its own call, and loses the results of
    the calls that ran next to it. Then models repeat the same wrong call over and over. As a result of the call,
    the error stays next to the call in the history.

    Also counts the write tools that succeeded in the current request, for check_changes_made,
    and stops the request when the model repeats one failing call MAX_SAME_FAILED_CALLS times."""

    request_writes = 0

    def run(self, *args, **kwargs):
        self.request_writes = 0
        self._failed_calls: dict[str, int] = {}
        self._same_results: dict[str, int] = {}
        return super().run(*args, **kwargs)

    @staticmethod
    def _call_key(tool_name: str, arguments: Any) -> str:
        try:
            return tool_name + json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            return tool_name + repr(arguments)

    def _repeated(self, tool_name: str, arguments: Any, result: Any) -> Any:
        key = self._call_key(tool_name, arguments) + "\0" + str(result)
        seen = self.__dict__.setdefault("_same_results", {})
        seen[key] = seen.get(key, 0) + 1
        if seen[key] >= MAX_SAME_RESULTS:
            raise RepeatedCallError(
                f"The model made the same call of {tool_name} {seen[key]} times with the same result "
                "and was stopped.", self.logger)
        if seen[key] > 1 and isinstance(result, str):
            return result + REPEAT_NOTE
        return result

    def _failed(self, tool_name: str, arguments: Any, error: str) -> str:
        key = self._call_key(tool_name, arguments)
        failed = self.__dict__.setdefault("_failed_calls", {})
        failed[key] = failed.get(key, 0) + 1
        if failed[key] >= MAX_SAME_FAILED_CALLS:
            raise RepeatedCallError(
                f"The model made the same failing call of {tool_name} {failed[key]} times and was stopped. "
                f"The error was: {error.removeprefix(TOOL_ERROR_PREFIX)}", self.logger)
        return error

    def execute_tool_call(self, tool_name: str, arguments: Any) -> Any:
        try:
            result = super().execute_tool_call(tool_name, arguments)
        except AgentToolCallError as e:
            # A wrong final_answer must stay an error: its result would become the answer to the user
            if tool_name == "final_answer":
                raise
            tool = self.tools.get(tool_name) or self.managed_agents.get(tool_name)
            hint = f" {describe_inputs(tool)}" if tool is not None else ""
            return self._failed(tool_name, arguments,
                                f"{TOOL_ERROR_PREFIX}the call was not run: {e}.{hint} Fix the call before trying again.")
        except AgentToolExecutionError as e:
            if tool_name == "final_answer":
                raise
            return self._failed(tool_name, arguments, f"{TOOL_ERROR_PREFIX}{e}")
        if getattr(self.tools.get(tool_name), "risk", "read") in WRITE_RISKS:
            self.request_writes += 1
        if tool_name == "final_answer":
            return result
        return self._repeated(tool_name, arguments, result)


def build_agent(config: dict[str, Any], tools: list[Tool], download_dir: Path, cache_dir: Path | None = None,
                verbosity_level: int = LogLevel.OFF) -> ToolCallingAgent:
    """Build the assistant: a ToolCallingAgent with the collection tools, the web tools and image_searcher
    as a managed agent. cache_dir keeps answers of web APIs (none: no cache).

    verbosity_level controls smolagents' console log: off inside Anki, where nobody reads stdout."""
    from ..web.images import build_image_searcher
    from ..web.net import WebCache
    from ..web.tools import web_tools

    cache = WebCache(cache_dir)
    managed_agents = [build_image_searcher(
        model=build_model(config, "searcher"),
        validator_model=build_model(config, "validator"),
        download_dir=download_dir,
        cache=cache,
        max_steps=int(config["roles"]["searcher"].get("max_steps", 20)),
        verbosity_level=verbosity_level,
        openserp_base_url=config.get("openserp_base_url", ""),
    )]

    return AssistantAgent(
        model=build_model(config, "manager"),
        tools=tools + web_tools(cache),
        managed_agents=managed_agents,
        prompt_templates=_prompt_templates(),
        final_answer_checks=[check_answer_has_text, check_changes_made],
        max_steps=int(config["roles"]["manager"].get("max_steps", 20)),
        verbosity_level=verbosity_level,
    )


def _prompt_templates() -> dict:
    """smolagents' templates with our own system prompt.

    The default one lists every tool again as text (the API already gets them as schemas) and shows examples
    of calls written as text: ~4k tokens, which overflowed an 8k context and taught models to write calls as text."""
    templates = yaml.safe_load(
        importlib.resources.files("smolagents.prompts").joinpath("toolcalling_agent.yaml").read_text(encoding="utf-8"))
    # raw: the instructions are plain text, not a Jinja template
    templates["system_prompt"] = "{% raw %}" + ASSISTANT_INSTRUCTIONS + "{% endraw %}"
    return templates
