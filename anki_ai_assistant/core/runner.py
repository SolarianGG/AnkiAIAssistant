"""Runs the agent in a background thread, so Anki's UI stays responsive while the model thinks.

Callbacks are called from the worker thread: the UI must pass them to the main thread itself."""

import threading
from collections.abc import Callable
from typing import Any

from smolagents import ToolCallingAgent
from smolagents.memory import FinalAnswerStep
from smolagents.utils import AgentMaxStepsError

from .logs import log, log_step, short


class RunnerBusy(Exception):
    pass


MAX_STEPS_NOTE = "⚠️ The assistant used all its steps (max_steps) and stopped. What it managed to tell:\n\n"


def answer_text(output: Any, max_steps_reached: bool = False) -> Any:
    """The answer as text. After the last step smolagents asks the model for an answer without tools and returns
    the message content as is: a list of parts like [{"type": "text", "text": ...}]."""
    if isinstance(output, list) and all(isinstance(part, dict) for part in output):
        output = "\n".join(str(part.get("text", "")) for part in output if part.get("type", "text") == "text")
    if max_steps_reached:
        output = MAX_STEPS_NOTE + ("" if output is None else str(output))
    return output


class AgentRunner:
    """Holds one dialog with the agent. Consecutive requests continue it until new_dialog()."""

    def __init__(self, agent_factory: Callable[[], ToolCallingAgent], *,
                 on_event: Callable[[Any], None],
                 on_done: Callable[[Any], None],
                 on_error: Callable[[Exception], None],
                 on_stopped: Callable[[], None]):
        self._agent_factory = agent_factory
        self._on_event, self._on_done, self._on_error, self._on_stopped = on_event, on_done, on_error, on_stopped
        self._agent: ToolCallingAgent | None = None
        self._has_history = False
        self._thread: threading.Thread | None = None
        self._stop_requested = threading.Event()
        # The saved models changed: the current dialog takes them on the next request
        self._models_changed = False

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def submit(self, task: str) -> None:
        if self.busy:
            raise RunnerBusy("The assistant is still working on the previous request.")
        self._stop_requested.clear()
        self._thread = threading.Thread(target=self._run, args=(task,), name="AIAssistantAgent", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Ask the agent to stop. It stops after the current step: a running model request is not cut off."""
        if not self.busy:
            return
        self._stop_requested.set()
        if self._agent is not None:
            self._agent.interrupt()
            # A managed agent runs its own loop for minutes: stop it too, or the manager waits for it
            for managed_agent in self._agent.managed_agents.values():
                managed_agent.interrupt()

    def change_models(self) -> None:
        """Use the models of the saved config (another profile) from the next request on, keeping the dialog.

        The agent factory builds them in the worker thread: building a model may need the network
        (Ollama's context_length)."""
        if self.busy:
            raise RunnerBusy("Stop the assistant before changing the models.")
        if self._agent is not None:
            self._models_changed = True

    def new_dialog(self) -> None:
        """Forget the conversation. The agent is rebuilt on the next request, so config changes apply too."""
        if self.busy:
            raise RunnerBusy("Stop the assistant before starting a new dialog.")
        self._agent = None
        self._has_history = False
        self._models_changed = False

    def _max_steps_reached(self) -> bool:
        steps = getattr(getattr(self._agent, "memory", None), "steps", None) or []
        return bool(steps) and isinstance(getattr(steps[-1], "error", None), AgentMaxStepsError)

    def _run(self, task: str) -> None:
        # In full: the request is what the whole run is checked against
        log.info("request (%s dialog): %s", "continued" if self._has_history else "new", task)
        try:
            if self._agent is None:
                self._agent = self._agent_factory()
            elif self._models_changed:
                # A fresh agent has the new models; the dialog's memory stays in the current one.
                # The image searcher is a new managed agent: each of its calls starts from scratch anyway
                fresh = self._agent_factory()
                self._agent.model = fresh.model
                self._agent.managed_agents = fresh.managed_agents
                self._agent.max_steps = fresh.max_steps
                self._models_changed = False
            if self._stop_requested.is_set():
                self._on_stopped()
                return
            answer = None
            for event in self._agent.run(task, stream=True, reset=not self._has_history):
                # The history counts from the first step on, also when the run fails or is stopped later
                self._has_history = True
                if isinstance(event, FinalAnswerStep):
                    answer = answer_text(event.output, self._max_steps_reached())
                else:
                    log_step(event)
                    self._on_event(event)
            log.info("answer: %s", short(answer))
            self._on_done(answer)
        except Exception as e:
            # smolagents raises a plain AgentError on interrupt, so tell it apart by our own flag
            if self._stop_requested.is_set():
                log.info("stopped by the user")
                self._on_stopped()
            else:
                log.exception("request failed")
                self._on_error(e)

