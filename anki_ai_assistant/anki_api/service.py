"""AnkiService: the only way the agent's tools reach the collection.

The agent runs in a worker thread, but the collection may only be used from Anki's main thread.
So every call is passed to the main thread and the worker waits for its result.
Writes also get one undo step per agent action and a backup before the first write of a session."""

import threading
from collections.abc import Callable
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from typing import TypeVar

from anki.collection import Collection, OpChanges

T = TypeVar("T")

UNDO_PREFIX = "AI: "


class CollectionUnavailable(Exception):
    """No collection is open, e.g. while the profile is being switched."""


class AnkiService:
    """Runs functions of the collection on the main thread.

    get_col returns the open collection or None. run_on_main schedules a callable on the main thread
    (mw.taskman.run_on_main in Anki); without it everything runs in the calling thread, as in the tests.
    on_changes gets the OpChanges of every write, so the UI can refresh.
    backup is called once before the first write, e.g. to make a collection backup."""

    def __init__(self, get_col: Callable[[], Collection | None], *,
                 run_on_main: Callable[[Callable[[], None]], None] | None = None,
                 on_changes: Callable[[OpChanges], None] | None = None,
                 backup: Callable[[Collection], None] | None = None,
                 timeout: float = 120.0):
        self._get_col = get_col
        self._run_on_main = run_on_main
        self._on_changes = on_changes
        self._backup = backup
        self._timeout = timeout
        self._backed_up = False
        # The agent runs several tool calls of one step in parallel threads. In Anki the main thread runs them
        # one by one; without run_on_main this lock does, so the undo steps of two writes cannot interleave
        self._direct_lock = threading.RLock()

    def read(self, fn: Callable[[Collection], T]) -> T:
        """Run fn(col) on the main thread and return its result, without an undo step.

        Also for media writes: Anki keeps media outside the undo history anyway."""
        return self._call_on_main(lambda: fn(self._col()))

    def write(self, description: str, fn: Callable[[Collection], T]) -> T:
        """Run fn(col) on the main thread as one undo step named "AI: <description>"."""
        def run() -> T:
            col = self._col()
            if not self._backed_up and self._backup is not None:
                self._backup(col)
            self._backed_up = True
            step = col.add_custom_undo_entry(UNDO_PREFIX + description)
            try:
                return fn(col)
            finally:
                # Also after a failure halfway: what fn did change becomes undoable as one step
                changes = col.merge_undo_entries(step)
                if self._on_changes is not None:
                    self._on_changes(changes)

        return self._call_on_main(run)

    def _col(self) -> Collection:
        col = self._get_col()
        if col is None:
            raise CollectionUnavailable("The Anki collection is not open right now. Try again in a moment.")
        return col

    def _call_on_main(self, fn: Callable[[], T]) -> T:
        if self._run_on_main is None:
            with self._direct_lock:
                return fn()
        if threading.current_thread() is threading.main_thread():
            return fn()

        future: Future = Future()

        def task() -> None:
            # False when the worker gave up waiting: then the call must not run late
            if not future.set_running_or_notify_cancel():
                return
            try:
                future.set_result(fn())
            except BaseException as e:
                future.set_exception(e)

        self._run_on_main(task)
        try:
            return future.result(timeout=self._timeout)
        except FutureTimeoutError:
            if future.cancel():
                raise TimeoutError("Anki did not respond: its window is probably busy. Nothing was changed.")
            # Already running: it will finish, so wait for it rather than report a wrong outcome
            return future.result()


def describe_error(error: Exception) -> str:
    """The message of a collection error without the Unicode isolation marks Anki puts around values."""
    return str(error).replace("⁨", "").replace("⁩", "")
