"""Checks of the agent's final answer against what its tools really did."""

from typing import Any

from smolagents.memory import ActionStep, AgentMemory, TaskStep

from .models import is_empty_reply

WRITE_RISKS = ("write", "destructive")

NO_CHANGES_MESSAGE = (
    "No tool changed the collection during this request: nothing was added, changed or deleted. "
    "This is only a check that your answer matches what was done, not a request to change anything. "
    "If your answer does not claim a change (you answered a question, asked the user, or explained why "
    "the request cannot be done as asked), call final_answer again with the same answer. "
    "Only if your answer says something was added, changed or deleted: correct the answer, "
    "or make that change now if the user asked for it. Do not make changes the user did not ask for.")


EMPTY_ANSWER_MESSAGE = (
    "Your final answer has no reply for the user: it is empty or only tool call syntax. "
    "Call final_answer again with your reply to the user as plain text.")


def check_answer_has_text(final_answer: Any, memory: AgentMemory, agent=None) -> bool:
    """Reject a final answer the user could not read, e.g. "</tool_call>" from a model that leaked call syntax
    into the answer argument. Runs before check_changes_made, which would otherwise let it through after a write."""
    if final_answer is None or is_empty_reply(str(final_answer)):
        raise AssertionError(EMPTY_ANSWER_MESSAGE)
    return True


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
    steps = [step for step in _current_run_steps(memory) if isinstance(step, ActionStep)]
    writes = getattr(agent, "request_writes", None)
    if writes is None:
        # A plain ToolCallingAgent: its memory has only the calls of steps without errors
        write_tools = {name for name, tool in agent.tools.items() if getattr(tool, "risk", "read") in WRITE_RISKS}
        writes = sum(call.name in write_tools for step in steps for call in step.tool_calls or [])
    if writes:
        return True
    if any(step.error is not None and NO_CHANGES_MESSAGE in str(step.error) for step in steps):
        return True
    raise AssertionError(NO_CHANGES_MESSAGE)
