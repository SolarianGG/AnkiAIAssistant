"""Runs the agent in a background thread, so Anki's UI stays responsive while the model thinks.

Callbacks are called from the worker thread: the UI must pass them to the main thread itself."""

import threading
from collections.abc import Callable
from typing import Any

from smolagents import Model, ToolCallingAgent
from smolagents.memory import FinalAnswerStep


class RunnerBusy(Exception):
    pass


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
        # A new manager model for the current dialog, built in the worker on the next request
        self._model_factory: Callable[[], Model] | None = None

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

    def change_model(self, model_factory: Callable[[], Model]) -> None:
        """Use another manager model from the next request on, keeping the dialog.

        The factory runs in the worker thread: building a model may need the network (Ollama's context_length)."""
        if self.busy:
            raise RunnerBusy("Stop the assistant before changing the model.")
        if self._agent is not None:
            self._model_factory = model_factory

    def new_dialog(self) -> None:
        """Forget the conversation. The agent is rebuilt on the next request, so config changes apply too."""
        if self.busy:
            raise RunnerBusy("Stop the assistant before starting a new dialog.")
        self._agent = None
        self._has_history = False
        self._model_factory = None

    def _run(self, task: str) -> None:
        try:
            if self._agent is None:
                self._agent = self._agent_factory()
            elif self._model_factory is not None:
                self._agent.model = self._model_factory()
                self._model_factory = None
            if self._stop_requested.is_set():
                self._on_stopped()
                return
            answer = None
            for event in self._agent.run(task, stream=True, reset=not self._has_history):
                # The history counts from the first step on, also when the run fails or is stopped later
                self._has_history = True
                if isinstance(event, FinalAnswerStep):
                    answer = event.output
                else:
                    self._on_event(event)
            self._on_done(answer)
        except Exception as e:
            # smolagents raises a plain AgentError on interrupt, so tell it apart by our own flag
            if self._stop_requested.is_set():
                self._on_stopped()
            else:
                self._on_error(e)

