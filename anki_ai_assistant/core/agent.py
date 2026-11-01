import importlib.resources
from pathlib import Path
from typing import Any

import yaml
from smolagents import LogLevel, Tool, ToolCallingAgent
from smolagents.utils import AgentToolCallError, AgentToolExecutionError

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


class AssistantAgent(ToolCallingAgent):
    """ToolCallingAgent whose failed tool calls are results of those calls, not errors of the whole step.

    smolagents records a step's tool calls only after all of them succeed. When one fails, e.g. on an argument
    the tool does not take, the model gets the bare error without its own call, and loses the results of
    the calls that ran next to it. Then models repeat the same wrong call over and over. As a result of the call,
    the error stays next to the call in the history.

    Also counts the write tools that succeeded in the current request, for check_changes_made."""

    request_writes = 0

    def run(self, *args, **kwargs):
        self.request_writes = 0
        return super().run(*args, **kwargs)

    def execute_tool_call(self, tool_name: str, arguments: Any) -> Any:
        try:
            result = super().execute_tool_call(tool_name, arguments)
        except AgentToolCallError as e:
            # A wrong final_answer must stay an error: its result would become the answer to the user
            if tool_name == "final_answer":
                raise
            tool = self.tools.get(tool_name) or self.managed_agents.get(tool_name)
            hint = f" {describe_inputs(tool)}" if tool is not None else ""
            return f"{TOOL_ERROR_PREFIX}the call was not run: {e}.{hint} Fix the call before trying again."
        except AgentToolExecutionError as e:
            if tool_name == "final_answer":
                raise
            return f"{TOOL_ERROR_PREFIX}{e}"
        if getattr(self.tools.get(tool_name), "risk", "read") in WRITE_RISKS:
            self.request_writes += 1
        return result


def build_agent(config: dict[str, Any], tools: list[Tool], download_dir: Path,
                verbosity_level: int = LogLevel.OFF) -> ToolCallingAgent:
    """Build the assistant: a ToolCallingAgent with the collection tools and image_searcher as a managed agent.

    verbosity_level controls smolagents' console log: off inside Anki, where nobody reads stdout."""
    managed_agents = []
    if config.get("openserp_base_url"):
        from ..web.images import build_image_searcher

        managed_agents.append(build_image_searcher(
            model=build_model(config, "searcher"),
            validator_model=build_model(config, "validator"),
            openserp_base_url=config["openserp_base_url"],
            download_dir=download_dir,
            max_steps=int(config["roles"]["searcher"].get("max_steps", 20)),
            verbosity_level=verbosity_level,
        ))

    return AssistantAgent(
        model=build_model(config, "manager"),
        tools=tools,
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
