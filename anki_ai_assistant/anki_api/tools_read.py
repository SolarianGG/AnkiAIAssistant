"""Read-only tools: the agent looks at the collection before it changes anything. They run without confirmation."""

import difflib
import re
from datetime import date, datetime, timedelta

from anki.collection import Collection
from anki.errors import AnkiException
from smolagents import Tool

from .service import AnkiService
from .tools_base import AnkiTool, AnkiToolError, find_deck, find_note_type, to_json as _json

DEFAULT_SEARCH_LIMIT = 20
MAX_SEARCH_LIMIT = 100
FIELD_PREVIEW_CHARS = 200
MAX_CARD_INFO_NOTES = 50
MAX_TAGS = 500

CARD_TYPES = {0: "new", 1: "learning", 2: "review", 3: "relearning"}
CARD_QUEUES = {-3: "buried", -2: "buried", -1: "suspended"}


def _kind(note_type: dict) -> str:
    return "cloze" if note_type["type"] == 1 else "standard"


class AnkiReadTool(AnkiTool):
    risk = "read"


class ListDecksTool(AnkiReadTool):
    name = "list_decks"
    description = (
        "List all decks as JSON: full name ('Parent::Child'), id, number of cards in the deck itself and with subdecks, "
        "cards due today (new / learn / review) and whether it is a filtered deck."
    )
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        def run(col: Collection) -> list[dict]:
            decks = []

            def walk(node, parent: str) -> None:
                for child in node.children:
                    name = f"{parent}::{child.name}" if parent else child.name
                    decks.append({
                        "name": name,
                        "cards": child.total_in_deck, "cards_with_subdecks": child.total_including_children,
                        "due_today": {"new": child.new_count, "learn": child.learn_count,
                                      "review": child.review_count},
                        **({"filtered": True} if child.filtered else {}),
                    })
                    walk(child, name)

            walk(col.sched.deck_due_tree(), "")
            return decks

        return _json(self.read(run))


class GetDeckOptionsTool(AnkiReadTool):
    name = "get_deck_options"
    description = (
        "Get the study options of a deck as JSON: its options preset, daily limits, learning steps, intervals, "
        "leech settings and FSRS desired retention. For a filtered deck: its search and limit."
    )
    inputs = {"deck": {"type": "string", "description": "Full deck name, e.g. 'German::Nouns'."}}
    output_type = "string"

    def forward(self, deck: str) -> str:
        def run(col: Collection) -> dict:
            deck_dict = find_deck(col, deck)
            if deck_dict["dyn"]:
                return {"deck": deck_dict["name"], "filtered": True,
                        "searches": [{"search": t[0], "limit": t[1]} for t in deck_dict["terms"]],
                        "reschedule_based_on_answers": bool(deck_dict["resched"])}
            conf = col.decks.config_dict_for_deck_id(deck_dict["id"])
            leech_action = "suspend" if conf["lapse"]["leechAction"] == 0 else "tag only"
            return {
                "deck": deck_dict["name"],
                "preset": conf["name"],
                "preset_used_by_decks": len(col.decks.decks_using_config(conf)),
                "new_cards_per_day": conf["new"]["perDay"],
                "reviews_per_day": conf["rev"]["perDay"],
                "learning_steps_minutes": conf["new"]["delays"],
                "graduating_interval_days": conf["new"]["ints"][0],
                "easy_interval_days": conf["new"]["ints"][1],
                "relearning_steps_minutes": conf["lapse"]["delays"],
                "maximum_interval_days": conf["rev"]["maxIvl"],
                "starting_ease": conf["new"]["initialFactor"] / 1000,
                "leech_threshold": conf["lapse"]["leechFails"],
                "leech_action": leech_action,
                "fsrs_enabled": bool(col.get_config("fsrs", False)),
                "desired_retention": conf.get("desiredRetention"),
            }

        return _json(self.read(run))


class ListNoteTypesTool(AnkiReadTool):
    name = "list_note_types"
    description = (
        "List note types as JSON: name, kind (standard or cloze), number of notes, field names in order "
        "and card template names. Use the exact field names when adding or editing notes."
    )
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        def run(col: Collection) -> list[dict]:
            counts = {m.id: m.use_count for m in col.models.all_use_counts()}
            return [{
                "name": m["name"], "kind": _kind(m), "notes": counts.get(m["id"], 0),
                "fields": [f["name"] for f in m["flds"]],
                "templates": [t["name"] for t in m["tmpls"]],
            } for m in col.models.all()]

        return _json(self.read(run))


class GetNoteTypeTool(AnkiReadTool):
    name = "get_note_type"
    description = (
        "Get one note type in full as JSON: fields, the HTML of every card template (front and back) and the CSS. "
        "Call it before changing templates or styling."
    )
    inputs = {"name": {"type": "string", "description": "Note type name, e.g. 'Basic'."}}
    output_type = "string"

    def forward(self, name: str) -> str:
        def run(col: Collection) -> dict:
            m = find_note_type(col, name)
            return {
                "name": m["name"], "kind": _kind(m),
                "sort_field": m["flds"][m["sortf"]]["name"],
                "fields": [{"name": f["name"], **({"description": f["description"]} if f.get("description") else {})}
                           for f in m["flds"]],
                "templates": [{"name": t["name"], "front": t["qfmt"], "back": t["afmt"]} for t in m["tmpls"]],
                "css": m["css"],
            }

        return _json(self.read(run))


def _preview(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + f"…[{len(value) - limit} more chars]"


class SearchNotesTool(AnkiReadTool):
    name = "search_notes"
    description = (
        "Find notes with Anki search syntax and return them as JSON with their fields, tags and decks. "
        "Examples: 'deck:German' (with subdecks), '\"deck:German::Nouns\"', 'Front:Hund', 'tag:verbs', "
        "'note:Basic', 'added:7' (added in the last 7 days), 'is:due', 'is:suspended', '\"dog house\"', 'Hund or Katze'. "
        "Results are paged: pass the returned next_offset to get more. Long fields are cut unless full_fields is true."
    )
    inputs = {
        "query": {"type": "string", "description": "Anki search query. An empty string matches all notes."},
        "limit": {"type": "integer", "nullable": True,
                  "description": f"Notes per page (default {DEFAULT_SEARCH_LIMIT}, at most {MAX_SEARCH_LIMIT})."},
        "offset": {"type": "integer", "nullable": True, "description": "How many matching notes to skip (default 0)."},
        "full_fields": {"type": "boolean", "nullable": True,
                        "description": "Return whole field contents, e.g. before editing a note (default false)."},
    }
    output_type = "string"

    def forward(self, query: str, limit: int | None = None, offset: int | None = None,
                full_fields: bool | None = None) -> str:
        limit = max(1, min(int(limit or DEFAULT_SEARCH_LIMIT), MAX_SEARCH_LIMIT))
        offset = max(0, int(offset or 0))
        field_limit = None if full_fields else FIELD_PREVIEW_CHARS

        def run(col: Collection) -> dict:
            used = quote_names(col, query)
            note_ids = sorted(col.find_notes(used))
            page = note_ids[offset:offset + limit]
            notes = []
            for note_id in page:
                note = col.get_note(note_id)
                deck_ids = dict.fromkeys(card.did for card in note.cards())
                notes.append({
                    "id": note.id,
                    "note_type": note.note_type()["name"],
                    "decks": [col.decks.name(did) for did in deck_ids],
                    "tags": note.tags,
                    "fields": {name: value if field_limit is None else _preview(value, field_limit)
                               for name, value in note.items()},
                })
            result = {"total": len(note_ids), "offset": offset, "notes": notes}
            if used != query:
                result["query_used"] = used
                result["note"] = ("Names with spaces must be quoted in Anki search, so the query was run as "
                                  f"{used}. Quote such names yourself next time.")
            if offset + len(page) < len(note_ids):
                result["next_offset"] = offset + len(page)
            if not note_ids:
                hint = unknown_deck_hint(col, query)
                if hint:
                    result["hint"] = hint
            return result

        return _json(self.read(run))


# An unquoted deck: or note: term and the words after it, which may be the rest of a name with spaces
UNQUOTED_NAME = re.compile(r'(?<![\w":])(deck|note):([^\s"():]+(?:[ \t]+[^\s":-][^\s":]*)*)', re.IGNORECASE)


def quote_names(col: Collection, query: str) -> str:
    """Quote deck and note type names with spaces that the model left unquoted.

    Anki reads deck:Test French as the deck 'Test' and the word 'French': the search found nothing, and models
    repeated it many times despite the hint. When the words after deck: (note:) start with the full name of
    an existing deck (note type), the longest such name is quoted."""
    names = {"deck": {n.lower() for n in (d.name for d in col.decks.all_names_and_ids())},
             "note": {n.lower() for n in (m.name for m in col.models.all_names_and_ids())}}

    def fix(match: re.Match) -> str:
        kind, words = match.group(1), match.group(2).split()
        for count in range(len(words), 1, -1):
            name = " ".join(words[:count])
            if name.lower() in names[kind.lower()]:
                rest = " ".join(words[count:])
                return f'"{kind}:{name}"' + (f" {rest}" if rest else "")
        return match.group(0)

    return UNQUOTED_NAME.sub(fix, query)


# deck:name in a query: "deck:A B", deck:"A B" or deck:A (up to the next space)
DECK_TERM = re.compile(r'"deck:((?:[^"\\]|\\.)*)"|deck:"((?:[^"\\]|\\.)*)"|deck:(\S+)', re.IGNORECASE)


def unknown_deck_hint(col: Collection, query: str) -> str | None:
    """Why a search found nothing, when it names a deck that does not exist.

    Models write deck:Test German for the deck 'Test German', which Anki reads as deck 'Test' and the word
    'German', or use a part of a deck name; the empty result then looks like an empty deck."""
    names = [d.name for d in col.decks.all_names_and_ids()]
    lower = {n.lower() for n in names}
    hints = []
    for match in DECK_TERM.finditer(query):
        name = next(g for g in match.groups() if g is not None).replace('\\"', '"')
        # * and _ are wildcards in Anki search: such a term is not a deck name to check
        if not name or "*" in name or "_" in name or name.lower() in lower:
            continue
        leaf = name.lower().split("::")[-1]
        similar = [n for n in names if leaf in n.lower()]
        similar += difflib.get_close_matches(name, names, n=5, cutoff=0.5)
        similar = list(dict.fromkeys(similar))[:5]
        hint = f"No deck is named '{name}'."
        if similar:
            hint += " Decks with similar names: " + ", ".join(f'"deck:{n}"' for n in similar) + "."
        hints.append(hint)
    if not hints:
        return None
    return " ".join(hints) + (' Put a deck name with spaces or quotes in quotes: "deck:Test German"; '
                              "a deck name includes its parents, e.g. Parent::Child.")


def _due(col: Collection, card) -> str | int | None:
    """When the card is due, in a form the model can read: a date, a time or the position of a new card."""
    due = card.odue if card.odid else card.due
    if card.queue == 0 or card.type == 0:
        return f"new card #{due}"
    if card.queue == 1:
        return datetime.fromtimestamp(due).isoformat(timespec="minutes")
    # Review cards and day-learning cards count days since the collection was created
    return (date.today() + timedelta(days=due - col.sched.today)).isoformat()


class GetCardInfoTool(AnkiReadTool):
    name = "get_card_info"
    description = (
        "Get the scheduling state of every card of the given notes as JSON: template, deck, state "
        "(new / learning / review / relearning, suspended, buried), due date, interval, ease or FSRS memory state, "
        "reviews, lapses and flag. Find the note ids with search_notes."
    )
    inputs = {"note_ids": {"type": "array", "items": {"type": "integer"},
                           "description": f"Note ids, at most {MAX_CARD_INFO_NOTES}."}}
    output_type = "string"

    def forward(self, note_ids: list) -> str:
        if len(note_ids) > MAX_CARD_INFO_NOTES:
            raise AnkiToolError(f"Too many notes: pass at most {MAX_CARD_INFO_NOTES} ids per call.")

        def run(col: Collection) -> list[dict]:
            cards = []
            for note_id in note_ids:
                try:
                    note = col.get_note(int(note_id))
                except AnkiException:
                    raise AnkiToolError(f"No note with id {note_id}.")
                for card in note.cards():
                    info = {
                        "card_id": card.id, "note_id": note.id,
                        "template": card.template()["name"],
                        "deck": col.decks.name(card.did),
                        "state": CARD_QUEUES.get(card.queue, CARD_TYPES.get(card.type, str(card.type))),
                        "due": _due(col, card),
                        "interval_days": card.ivl,
                        "reviews": card.reps,
                        "lapses": card.lapses,
                    }
                    if card.memory_state:
                        info["fsrs"] = {"stability_days": round(card.memory_state.stability, 2),
                                        "difficulty": round(card.memory_state.difficulty, 2)}
                    elif card.factor:
                        info["ease"] = card.factor / 1000
                    if card.user_flag():
                        info["flag"] = card.user_flag()
                    cards.append(info)
            return cards

        return _json(self.read(run))


class ListTagsTool(AnkiReadTool):
    name = "list_tags"
    description = "List the tags used in the collection as JSON, optionally only those containing a text."
    inputs = {"contains": {"type": "string", "nullable": True,
                           "description": "Return only tags containing this text (case-insensitive)."}}
    output_type = "string"

    def forward(self, contains: str | None = None) -> str:
        def run(col: Collection) -> dict:
            tags = sorted(col.tags.all(), key=str.lower)
            if contains:
                tags = [t for t in tags if contains.lower() in t.lower()]
            result = {"total": len(tags), "tags": tags[:MAX_TAGS]}
            if len(tags) > MAX_TAGS:
                result["note"] = f"Only the first {MAX_TAGS} tags are listed: narrow them with 'contains'."
            return result

        return _json(self.read(run))


class CollectionStatsTool(AnkiReadTool):
    name = "collection_stats"
    description = (
        "Get an overview of the collection as JSON: numbers of notes, cards, decks, note types and tags, "
        "cards due today, cards studied today, suspended and new cards."
    )
    inputs = {}
    output_type = "string"

    def forward(self) -> str:
        def run(col: Collection) -> dict:
            tree = col.sched.deck_due_tree()
            return {
                "notes": col.note_count(),
                "cards": col.card_count(),
                "decks": len(col.decks.all_names_and_ids()),
                "note_types": len(col.models.all_names_and_ids()),
                "tags": len(col.tags.all()),
                "due_today": {"new": tree.new_count, "learn": tree.learn_count, "review": tree.review_count},
                "studied_today": len(col.find_cards("rated:1")),
                "new_cards": len(col.find_cards("is:new")),
                "suspended_cards": len(col.find_cards("is:suspended")),
            }

        return _json(self.read(run))


READ_TOOLS = (ListDecksTool, GetDeckOptionsTool, ListNoteTypesTool, GetNoteTypeTool, SearchNotesTool,
              GetCardInfoTool, ListTagsTool, CollectionStatsTool)


def read_tools(service: AnkiService) -> list[Tool]:
    return [tool(service) for tool in READ_TOOLS]
