from pathlib import Path
from typing import Any

from smolagents import LogLevel, Tool, ToolCallingAgent

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
        instructions=ASSISTANT_INSTRUCTIONS,
        max_steps=int(config["roles"]["manager"].get("max_steps", 20)),
        verbosity_level=verbosity_level,
    )
