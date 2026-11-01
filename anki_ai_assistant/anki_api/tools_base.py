"""What all collection tools share: the base class, errors for the model and name lookups."""

import difflib
import html
import json
import re
from collections.abc import Callable
from typing import Any, TypeVar

from anki.collection import Collection
from anki.errors import AnkiException
from smolagents import Tool

from .confirm import ChangePlan
from .service import AnkiService, describe_error

T = TypeVar("T")


class AnkiToolError(Exception):
    """A wrong tool call, e.g. an unknown deck. The message tells the model how to fix the call."""


def to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def short_text(value: str, limit: int = 120) -> str:
    """A field as one line of plain text for a preview: no HTML tags, at most limit characters.

    Not anki.utils.strip_html: it needs Anki's translations loaded, which the tests and the CLI do not have."""
    value = " ".join(html.unescape(re.sub(r"<[^>]*>", " ", value)).split())
    return value if len(value) <= limit else value[:limit] + "…"


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
                            f"Its fields are: {', '.join(known)}. Use exactly these names, or to have other fields "
                            "create a note type with them (create_note_type) and pass it as note_type.")


def select_notes(col: Collection, note_ids: list | None, query: str | None, limit: int) -> list[int]:
    """The notes a tool works on: given by id or by an Anki search, at most limit of them."""
    if bool(note_ids) == bool(query and query.strip()):
        raise AnkiToolError("Pass either note_ids or query, not both.")
    if note_ids:
        ids = list(dict.fromkeys(int(i) for i in note_ids))
        existing = set(col.find_notes("nid:" + ",".join(map(str, ids))))
        missing = [i for i in ids if i not in existing]
        if missing:
            raise AnkiToolError(f"No notes with ids {missing}. Nothing was changed.")
    else:
        ids = list(col.find_notes(query))
        if not ids:
            raise AnkiToolError(f"No notes match the search {query!r}. Nothing was changed.")
    if len(ids) > limit:
        raise AnkiToolError(f"{len(ids)} notes: at most {limit} per call. Narrow the selection.")
    return ids


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

    def ask(self, plan: ChangePlan) -> str | None:
        """Ask the user to allow the change. None if allowed, else the tool's result for the model."""
        decision = self.service.ask(plan)
        if decision.approved:
            return None
        result = {"done": False, "declined_by_user": True,
                  "note": "The user declined this change, nothing was changed. Do not repeat the same call: "
                          "adjust the plan to the user's comment, or ask the user what they want."}
        if decision.comment:
            result["user_comment"] = decision.comment
        return to_json(result)

    def write(self, description: str, fn: Callable[[Collection], T]) -> T:
        try:
            return self.service.write(description, fn)
        except AnkiException as e:
            raise AnkiToolError(describe_error(e)) from e
