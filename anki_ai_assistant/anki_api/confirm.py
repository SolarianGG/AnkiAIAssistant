"""What a destructive tool shows the user before it acts, and the user's answer.

A tool builds a ChangePlan from the collection, AnkiService.ask() shows it (a dialog in Anki, a prompt in the CLI)
and returns a Decision. Nothing here depends on Qt, so the tools and their tests run without a GUI."""

from dataclasses import dataclass

MAX_DETAIL_ITEMS = 30


@dataclass
class ChangePlan:
    """kind: the type of action, e.g. "delete_notes"; the user may trust a kind for the rest of the dialog.
    title: one line, e.g. "Delete 143 notes". summary: what happens, in a sentence or two.
    details: plain text, a list of what is affected or a diff (lines starting with "+" / "-" are highlighted).
    full_sync: the change modifies the collection schema, so the next sync uploads the whole collection."""

    kind: str
    title: str
    summary: str
    details: str = ""
    full_sync: bool = False


@dataclass
class Decision:
    approved: bool
    comment: str = ""
    # Do not ask again for this kind of action in this dialog
    trust_kind: bool = False


FULL_SYNC_WARNING = ("This changes the collection schema: the next sync will upload the whole collection to AnkiWeb, "
                     "and changes made on other devices since the last sync will be lost.")


def item_list(lines: list[str], total: int | None = None, limit: int = MAX_DETAIL_ITEMS) -> str:
    """The first lines of a list and how many more there are."""
    total = len(lines) if total is None else total
    shown = lines[:limit]
    if total > len(shown):
        shown.append(f"… and {total - len(shown)} more")
    return "\n".join(shown)
