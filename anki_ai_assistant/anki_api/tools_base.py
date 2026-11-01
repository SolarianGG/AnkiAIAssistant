"""What all collection tools share: the base class, errors for the model and name lookups."""

import difflib
import json
from collections.abc import Callable
from typing import Any, TypeVar

from anki.collection import Collection
from anki.errors import AnkiException
from smolagents import Tool

from .service import AnkiService, describe_error

T = TypeVar("T")


class AnkiToolError(Exception):
    """A wrong tool call, e.g. an unknown deck. The message tells the model how to fix the call."""


def to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _suggest(name: str, known: list[str], what: str) -> AnkiToolError:
    close = difflib.get_close_matches(name, known, n=5, cutoff=0.5)
    hint = f" Did you mean: {', '.join(close)}?" if close else f" Call list_{what}s to see them."
    return AnkiToolError(f"No {what.replace('_', ' ')} named '{name}'.{hint}")


def find_deck(col: Collection, name: str) -> dict:
    deck_id = col.decks.id_for_name(name)
    if deck_id is None:
        raise _suggest(name, [d.name for d in col.decks.all_names_and_ids()], "deck")
    return col.decks.get(deck_id)


def find_note_type(col: Collection, name: str) -> dict:
    note_type = col.models.by_name(name)
    if note_type is None:
        raise _suggest(name, [m.name for m in col.models.all_names_and_ids()], "note_type")
    return note_type


def check_field_names(note_type: dict, names) -> None:
    known = [f["name"] for f in note_type["flds"]]
    unknown = [n for n in names if n not in known]
    if unknown:
        raise AnkiToolError(f"Note type '{note_type['name']}' has no field(s) {', '.join(map(repr, unknown))}. "
                            f"Its fields are: {', '.join(known)}.")


class AnkiTool(Tool):
    """Base of the collection tools. risk is for the confirmation layer: read, write or destructive."""

    risk = "read"

    def __init__(self, service: AnkiService, **kwargs):
        super().__init__(**kwargs)
        self.service = service

    def read(self, fn: Callable[[Collection], T]) -> T:
        try:
            return self.service.read(fn)
        except AnkiException as e:
            raise AnkiToolError(describe_error(e)) from e

    def write(self, description: str, fn: Callable[[Collection], T]) -> T:
        try:
            return self.service.write(description, fn)
        except AnkiException as e:
            raise AnkiToolError(describe_error(e)) from e
