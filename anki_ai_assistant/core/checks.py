"""Checks of the agent's final answer against what its tools really did."""

from typing import Any

from smolagents.memory import ActionStep, AgentMemory, TaskStep

WRITE_RISKS = ("write", "destructive")

NO_CHANGES_MESSAGE = (
    "No tool changed the collection during this request: nothing was added, changed or deleted. "
    "If the user asked for a change, make it now by calling the tools. "
    "If the request needed no change, call final_answer again with your answer, "
    "and do not claim changes you did not make.")


def _current_run_steps(memory: AgentMemory) -> list:
    """The steps of the current request: a dialog keeps the earlier ones in memory too."""
    steps = memory.steps
    for index in range(len(steps) - 1, -1, -1):
        if isinstance(steps[index], TaskStep):
            return steps[index + 1:]
    return list(steps)


def check_changes_made(final_answer: Any, memory: AgentMemory, agent=None) -> bool:
    """Send the first final answer of a request back once if no write tool was called in it.

    Models sometimes answer "I added the card" after only reading the collection. Whether the request needed
    a change, only the model can tell, so the check does not reject the answer, it gives the model the fact:
    it then makes the change or repeats its answer."""
    if agent is None:
        return True
    write_tools = {name for name, tool in agent.tools.items() if getattr(tool, "risk", "read") in WRITE_RISKS}
    steps = [step for step in _current_run_steps(memory) if isinstance(step, ActionStep)]
    if any(call.name in write_tools for step in steps for call in step.tool_calls or []):
        return True
    if any(step.error is not None and NO_CHANGES_MESSAGE in str(step.error) for step in steps):
        return True
    raise AssertionError(NO_CHANGES_MESSAGE)
