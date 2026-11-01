import queue
import threading

import pytest

from anki_ai_assistant.anki_api.service import AnkiService, CollectionUnavailable
from conftest import add_note


def test_write_is_one_undo_step(col):
    changes_seen = []
    service = AnkiService(lambda: col, on_changes=changes_seen.append)

    def add_two(c):
        add_note(c, "New deck", Front="a", Back="b")
        add_note(c, "New deck", Front="c", Back="d")

    service.write("add 2 notes", add_two)
    assert col.note_count() == 2
    assert col.undo_status().undo == "AI: add 2 notes"
    assert changes_seen and changes_seen[0].note_text

    col.undo()
    assert col.note_count() == 0
    assert col.decks.id_for_name("New deck") is None


def test_failed_write_stays_undoable(col):
    service = AnkiService(lambda: col)

    def add_then_fail(c):
        add_note(c, "Default", Front="a", Back="b")
        raise ValueError("boom")

    with pytest.raises(ValueError):
        service.write("half done", add_then_fail)
    assert col.undo_status().undo == "AI: half done"
    col.undo()
    assert col.note_count() == 0


def test_backup_once_before_first_write(col):
    backups = []
    service = AnkiService(lambda: col, backup=backups.append)
    service.read(lambda c: c.note_count())
    assert backups == []
    service.write("one", lambda c: add_note(c, "Default", Front="1"))
    service.write("two", lambda c: add_note(c, "Default", Front="2"))
    assert backups == [col]


def test_closed_collection():
    service = AnkiService(lambda: None)
    with pytest.raises(CollectionUnavailable):
        service.read(lambda c: c.note_count())


def test_calls_from_worker_run_on_main_thread(col):
    """In Anki the agent runs in a worker thread: the collection must only be touched on the main thread."""
    tasks = queue.Queue()
    service = AnkiService(lambda: col, run_on_main=tasks.put)
    seen_threads, results = [], []

    def worker():
        results.append(service.read(lambda c: (seen_threads.append(threading.current_thread()), c.note_count())[1]))
        results.append(service.write("add", lambda c: add_note(c, "Default", Front="x")))
        tasks.put(None)

    threading.Thread(target=worker).start()
    while (task := tasks.get(timeout=10)) is not None:
        task()

    assert seen_threads == [threading.main_thread()]
    assert results[0] == 0 and col.note_count() == 1


def test_timeout_cancels_the_call(col):
    tasks = queue.Queue()
    service = AnkiService(lambda: col, run_on_main=tasks.put, timeout=0.1)
    errors = []

    def worker():
        try:
            service.write("late", lambda c: add_note(c, "Default", Front="late"))
        except TimeoutError as e:
            errors.append(e)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(timeout=10)
    # The main thread only gets to the task after the worker gave up: it must not run any more
    tasks.get_nowait()()
    assert errors and col.note_count() == 0
