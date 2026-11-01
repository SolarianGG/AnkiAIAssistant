import pytest
from anki.collection import Collection

from anki_ai_assistant.anki_api.service import AnkiService


@pytest.fixture
def col(tmp_path):
    collection = Collection(str(tmp_path / "test.anki2"))
    yield collection
    collection.close()


@pytest.fixture
def service(col):
    return AnkiService(lambda: col)


def add_note(col: Collection, deck: str, note_type: str = "Basic", tags: tuple[str, ...] = (), **fields) -> int:
    note = col.new_note(col.models.by_name(note_type))
    for name, value in fields.items():
        note[name] = value
    note.tags = list(tags)
    col.add_note(note, col.decks.id(deck))
    return note.id


@pytest.fixture
def german(col):
    """A small collection: a deck with a subdeck, Basic and Cloze notes, tags."""
    ids = {
        "hund": add_note(col, "German::Nouns", Front="der Hund", Back="the dog", tags=("nouns", "animals")),
        "katze": add_note(col, "German::Nouns", Front="die Katze", Back="the cat", tags=("nouns", "animals")),
        "laufen": add_note(col, "German", Front="laufen", Back="to run " + "x" * 500, tags=("verbs",)),
        "cloze": add_note(col, "German", "Cloze", Text="{{c1::Berlin}} ist die Hauptstadt"),
    }
    return ids
