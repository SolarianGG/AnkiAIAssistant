import importlib.resources
from pathlib import Path
from typing import Any

import yaml
from smolagents import LogLevel, Tool, ToolCallingAgent

from .checks import check_changes_made
from .models import build_model
from .prompts import ASSISTANT_INSTRUCTIONS


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

    return ToolCallingAgent(
        model=build_model(config, "manager"),
        tools=tools,
        managed_agents=managed_agents,
        prompt_templates=_prompt_templates(),
        final_answer_checks=[check_changes_made],
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
