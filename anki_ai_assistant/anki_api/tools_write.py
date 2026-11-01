"""Write tools: they add to the collection or change a few notes. They run without confirmation
(update_notes asks only when it changes many notes), but every call is one undo step "AI: ..." (Edit > Undo),
and the collection is backed up before the first one.

Each tool checks its input before it calls service.write(), so a wrong call leaves no empty undo step."""

from pathlib import Path
from urllib.parse import unquote, urlparse

from anki.collection import Collection
from anki.errors import AbortSchemaModification, NotFoundError
from anki.notes import NoteFieldsCheckResult
from smolagents import Tool

from ..web.download import fetch, safe_filename
from .confirm import ChangePlan
from .service import AnkiService
from .tools_base import (AnkiTool, AnkiToolError, check_field_names, find_deck, find_note_type, short_text,
                         to_json)

MAX_ADD_NOTES = 100
# More notes than this need the user's confirmation
MAX_UPDATE_NOTES = 20
MAX_BULK_UPDATE_NOTES = 200
MAX_TAG_NOTES = 1000

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg"}
SOUND_EXTENSIONS = {".mp3", ".ogg", ".oga", ".opus", ".wav", ".m4a", ".flac", ".webm", ".mp4"}

FIELD_CHECK_ERRORS = {
    NoteFieldsCheckResult.EMPTY: "the first field is empty",
    NoteFieldsCheckResult.DUPLICATE: "duplicate: a note of this type with the same first field already exists",
    NoteFieldsCheckResult.MISSING_CLOZE: "no cloze deletion like {{c1::...}} in the cloze field",
    NoteFieldsCheckResult.NOTETYPE_NOT_CLOZE: "cloze deletions are only allowed in a cloze note type",
    NoteFieldsCheckResult.FIELD_NOT_CLOZE: "cloze deletions are in a field that is not a cloze field",
}

# Objects are described in full: some providers turn {"type": "object"} without properties into an empty object
NOTE_FIELDS_SCHEMA = {"type": "object", "additionalProperties": {"type": "string"}}

TEMPLATE_INPUT = {
    "type": "array",
    "items": {"type": "object", "required": ["name", "front", "back"],
              "properties": {"name": {"type": "string"}, "front": {"type": "string"}, "back": {"type": "string"}}},
    "description": "Card templates, each {\"name\": \"Card 1\", \"front\": \"{{Front}}\", "
                   "\"back\": \"{{FrontSide}}<hr id=answer>{{Back}}\"}. Front and back are HTML with {{Field}} references.",
}


class AnkiWriteTool(AnkiTool):
    risk = "write"


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" + ("" if count == 1 else "s")


def _check_deck_name(name: str) -> str:
    name = name.strip()
    if not name or any(not part.strip() for part in name.split("::")):
        raise AnkiToolError(f"Invalid deck name {name!r}: use a name like 'German' or 'German::Nouns'.")
    return name


class CreateDeckTool(AnkiWriteTool):
    name = "create_deck"
    description = "Create a deck. Subdecks are written as 'Parent::Child'; missing parents are created too."
    inputs = {"name": {"type": "string", "description": "Full deck name, e.g. 'German::Nouns'."}}
    output_type = "string"

    def forward(self, name: str) -> str:
        name = _check_deck_name(name)
        existing = self.read(lambda col: col.decks.id_for_name(name))
        if existing is not None:
            return to_json({"deck": name, "id": existing, "created": False, "note": "The deck already exists."})
        deck_id = self.write(f"create deck {name}", lambda col: col.decks.id(name))
        return to_json({"deck": name, "id": deck_id, "created": True})


class AddNotesTool(AnkiWriteTool):
    name = "add_notes"
    description = (
        "Add notes to an existing deck (create a new one with create_deck first); Anki makes the cards from them. "
        "Each note is an object {field name: value} with the exact field names of the note type "
        "(see list_note_types), e.g. [{\"Front\": \"der Hund\", \"Back\": \"the dog\"}]. Values may contain HTML. "
        "A note is skipped if its first field is empty or the same note already exists, unless allow_duplicates is true. "
        f"At most {MAX_ADD_NOTES} notes per call."
    )
    inputs = {
        "deck": {"type": "string", "description": "Exact full name of an existing deck, as list_decks shows it."},
        "notes": {"type": "array", "items": NOTE_FIELDS_SCHEMA,
                  "description": "Notes, each an object {field name: value}."},
        "note_type": {"type": "string", "nullable": True, "description": "Note type name (default 'Basic')."},
        "tags": {"type": "array", "items": {"type": "string"}, "nullable": True,
                 "description": "Tags for all the notes, without spaces, e.g. [\"german\", \"nouns\"]."},
        "allow_duplicates": {"type": "boolean", "nullable": True,
                             "description": "Add notes even if the same note already exists (default false)."},
    }
    output_type = "string"

    def forward(self, deck: str, notes: list, note_type: str | None = None, tags: list | None = None,
                allow_duplicates: bool | None = None) -> str:
        deck = deck.strip()
        if not notes:
            raise AnkiToolError("Pass at least one note.")
        if len(notes) > MAX_ADD_NOTES:
            raise AnkiToolError(f"Too many notes: add at most {MAX_ADD_NOTES} per call.")
        if any(not isinstance(note, dict) for note in notes):
            raise AnkiToolError("Each note must be an object {field name: value}.")
        tags = _check_tags(tags or [])

        def prepare(col: Collection):
            # The deck must exist: creating it here turned a misspelt name into a new deck
            deck_id = find_deck(col, deck)["id"]
            model = find_note_type(col, note_type or "Basic")
            check_field_names(model, {name for note in notes for name in note})
            ready, skipped, first_fields = [], [], set()
            for index, fields in enumerate(notes):
                note = col.new_note(model)
                for name, value in fields.items():
                    note[name] = "" if value is None else str(value)
                note.tags = list(tags)
                state = note.fields_check()
                if state == NoteFieldsCheckResult.DUPLICATE and allow_duplicates:
                    state = NoteFieldsCheckResult.NORMAL
                first = note.fields[0].strip()
                if state == NoteFieldsCheckResult.NORMAL and first in first_fields and not allow_duplicates:
                    skipped.append({"index": index, "reason": "duplicate of an earlier note in this call"})
                elif state != NoteFieldsCheckResult.NORMAL:
                    skipped.append({"index": index, "reason": FIELD_CHECK_ERRORS.get(state, f"check failed ({state})")})
                else:
                    first_fields.add(first)
                    ready.append((index, note))
            return deck_id, model["name"], ready, skipped

        deck_id, model_name, ready, skipped = self.read(prepare)
        added = []
        if ready:
            def add(col: Collection) -> None:
                for index, note in ready:
                    try:
                        col.add_note(note, deck_id)
                    except Exception as e:  # e.g. a template that makes no card from these fields
                        skipped.append({"index": index, "reason": str(e)})
                    else:
                        added.append({"index": index, "id": note.id, "cards": len(note.card_ids())})

            self.write(f"add {_plural(len(ready), 'note')} to {deck}", add)
        return to_json({"deck": deck, "note_type": model_name, "added": added,
                        "skipped": sorted(skipped, key=lambda s: s["index"])})


class UpdateNotesTool(AnkiWriteTool):
    name = "update_notes"
    description = (
        "Change fields of existing notes. Only the given fields change; the others keep their content. "
        "Get the note ids and the current field contents with search_notes(full_fields=true) first. "
        f"Changes of more than {MAX_UPDATE_NOTES} notes are shown to the user to confirm; "
        f"at most {MAX_BULK_UPDATE_NOTES} notes per call. To replace the same text in many notes use find_replace."
    )
    inputs = {"notes": {"type": "array",
                        "items": {"type": "object", "required": ["id", "fields"],
                                  "properties": {"id": {"type": "integer"}, "fields": NOTE_FIELDS_SCHEMA}},
                        "description": "Changes, each {\"id\": note id, \"fields\": {field name: new value}}."}}
    output_type = "string"

    def forward(self, notes: list) -> str:
        if not notes:
            raise AnkiToolError("Pass at least one note.")
        if len(notes) > MAX_BULK_UPDATE_NOTES:
            raise AnkiToolError(f"Too many notes: update at most {MAX_BULK_UPDATE_NOTES} per call.")
        changes: dict[int, dict] = {}
        for item in notes:
            if not isinstance(item, dict) or "id" not in item or not isinstance(item.get("fields"), dict):
                raise AnkiToolError('Each change must be {"id": note id, "fields": {field name: new value}}.')
            changes.setdefault(int(item["id"]), {}).update(
                {k: "" if v is None else str(v) for k, v in item["fields"].items()})

        def check(col: Collection) -> tuple[dict[int, list[str]], list[str]]:
            changed, diff = {}, []
            for note_id, fields in changes.items():
                note = _get_note(col, note_id)
                check_field_names(note.note_type(), fields)
                changed_names = [name for name, value in fields.items() if note[name] != value]
                if changed_names:
                    changed[note_id] = changed_names
                    if len(diff) < 90:
                        for name in changed_names:
                            diff += [f"{note_id} {name}:", f"- {short_text(note[name])}", f"+ {short_text(fields[name])}"]
            return changed, diff

        changed, diff = self.read(check)
        if len(changed) > MAX_UPDATE_NOTES:
            declined = self.ask(ChangePlan(
                kind="bulk_update_notes", title=f"Edit {_plural(len(changed), 'note')}",
                summary="Only the listed fields change; the others keep their content.", details="\n".join(diff)))
            if declined:
                return declined
        if changed:
            def update(col: Collection) -> None:
                # Read the notes again: the user may have edited one meanwhile
                updated = []
                for note_id, names in changed.items():
                    note = col.get_note(note_id)
                    for name in names:
                        note[name] = changes[note_id][name]
                    updated.append(note)
                col.update_notes(updated)

            self.write(f"edit {_plural(len(changed), 'note')}", update)
        return to_json({"updated": [{"id": i, "fields": names} for i, names in changed.items()],
                        "unchanged": [i for i in changes if i not in changed]})


def _get_note(col: Collection, note_id: int):
    try:
        return col.get_note(note_id)
    except NotFoundError:
        raise AnkiToolError(f"No note with id {note_id}.")


def _check_tags(tags: list) -> list[str]:
    tags = [str(t).strip() for t in tags if str(t).strip()]
    with_spaces = [t for t in tags if any(c.isspace() for c in t)]
    if with_spaces:
        raise AnkiToolError(f"Tags cannot contain spaces: {with_spaces}. Use '_' or '::' instead.")
    return tags


class AddTagsTool(AnkiWriteTool):
    name = "add_tags"
    description = ("Add tags to notes; their existing tags stay. Hierarchical tags are written as 'parent::child'. "
                   f"At most {MAX_TAG_NOTES} notes per call.")
    inputs = {
        "note_ids": {"type": "array", "items": {"type": "integer"}, "description": "Note ids from search_notes."},
        "tags": {"type": "array", "items": {"type": "string"}, "description": "Tags without spaces."},
    }
    output_type = "string"

    def forward(self, note_ids: list, tags: list) -> str:
        tags = _check_tags(tags)
        if not tags or not note_ids:
            raise AnkiToolError("Pass at least one note id and one tag.")
        if len(note_ids) > MAX_TAG_NOTES:
            raise AnkiToolError(f"Too many notes: tag at most {MAX_TAG_NOTES} per call.")
        ids = list(dict.fromkeys(int(i) for i in note_ids))
        existing = set(self.read(lambda col: col.find_notes("nid:" + ",".join(map(str, ids)))))
        missing = [i for i in ids if i not in existing]
        if missing:
            raise AnkiToolError(f"No notes with ids {missing}. Nothing was tagged.")
        changes = self.write(f"add tags {' '.join(tags)}",
                             lambda col: col.tags.bulk_add(ids, " ".join(tags)))
        return to_json({"tags": tags, "notes_changed": changes.count})


class StoreMediaTool(AnkiWriteTool):
    name = "store_media"
    description = (
        "Store an image or audio file in the collection media, from a file found by image_searcher or from a URL. "
        "Returns the stored filename (Anki may rename it) and the HTML to put into a note field: "
        "<img src=\"file.jpg\"> for an image, [sound:file.mp3] for audio. "
        "Media is not undone by Edit > Undo, but files no note uses are removed by Tools > Check Media."
    )
    inputs = {
        "source": {"type": "string", "description": "Local path returned by image_searcher, or an http(s) URL."},
        "filename": {"type": "string", "nullable": True,
                     "description": "Name to store it under, e.g. 'hund.jpg' (default: the source's name)."},
    }
    output_type = "string"

    def __init__(self, service: AnkiService, download_dir: Path, **kwargs):
        super().__init__(service, **kwargs)
        self.download_dir = Path(download_dir).resolve()

    def forward(self, source: str, filename: str | None = None) -> str:
        is_url = source.startswith(("http://", "https://"))
        if is_url:
            default_name = unquote(Path(urlparse(source).path).name)
        else:
            path = Path(source).resolve()
            # Only files the agent downloaded itself: not anything from the user's disk
            if not path.is_relative_to(self.download_dir):
                raise AnkiToolError(f"Only files from the download folder {self.download_dir} or URLs can be stored.")
            if not path.is_file():
                raise AnkiToolError(f"No file at {path}.")
            default_name = path.name
        try:
            name = safe_filename(filename or default_name)
        except ValueError as e:
            raise AnkiToolError(str(e))
        extension = Path(name).suffix.lower()
        if extension not in IMAGE_EXTENSIONS | SOUND_EXTENSIONS:
            raise AnkiToolError(f"Unsupported file type {extension or '(none)'!r}: pass a filename ending in one of "
                                f"{', '.join(sorted(IMAGE_EXTENSIONS | SOUND_EXTENSIONS))}.")
        # Downloads and disk reads happen here in the worker thread: only the media write needs the main thread
        data = fetch(source) if is_url else path.read_bytes()
        if not data:
            raise AnkiToolError("The file is empty.")
        # Not an undoable operation, so no undo step: Anki keeps media outside the undo history
        stored = self.read(lambda col: col.media.write_data(name, data))
        html = f'<img src="{stored}">' if extension in IMAGE_EXTENSIONS else f"[sound:{stored}]"
        return to_json({"filename": stored, "html": html, "bytes": len(data)})


def _template_specs(templates: list) -> list[dict]:
    specs = []
    for t in templates:
        if not isinstance(t, dict) or not t.get("name") or not t.get("front") or not t.get("back"):
            raise AnkiToolError('Each template must be {"name": ..., "front": HTML, "back": HTML}, none empty.')
        specs.append({"name": str(t["name"]), "front": str(t["front"]), "back": str(t["back"])})
    return specs


def _add_template(col: Collection, model: dict, spec: dict) -> None:
    template = col.models.new_template(spec["name"])
    template["qfmt"], template["afmt"] = spec["front"], spec["back"]
    col.models.add_template(model, template)


class CreateNoteTypeTool(AnkiWriteTool):
    name = "create_note_type"
    description = (
        "Create a note type: either from scratch (fields + templates, optional CSS) or as a copy of an existing one "
        "(clone_from, to change it afterwards without touching the original). "
        "Templates and CSS must work in AnkiDroid too: plain HTML and CSS, no JavaScript. "
        "For a cloze note type: kind='cloze', one template using {{cloze:FieldName}} on both sides."
    )
    inputs = {
        "name": {"type": "string", "description": "Name of the new note type."},
        "clone_from": {"type": "string", "nullable": True,
                       "description": "Copy this existing note type with its fields, templates and CSS."},
        "fields": {"type": "array", "items": {"type": "string"}, "nullable": True,
                   "description": "Field names in order, for a note type from scratch. The first one is checked for duplicates."},
        "templates": {**TEMPLATE_INPUT, "nullable": True},
        "css": {"type": "string", "nullable": True, "description": "CSS for the cards (default: Anki's basic style)."},
        "kind": {"type": "string", "nullable": True, "enum": ["standard", "cloze"],
                 "description": "standard (default) or cloze."},
    }
    output_type = "string"

    def forward(self, name: str, clone_from: str | None = None, fields: list | None = None,
                templates: list | None = None, css: str | None = None, kind: str | None = None) -> str:
        name = name.strip()
        if not name:
            raise AnkiToolError("Pass a name for the note type.")
        if clone_from and (fields or templates or kind):
            raise AnkiToolError("Pass either clone_from or fields and templates, not both. "
                                "Change the copy afterwards with extend_note_type.")
        if not clone_from:
            fields = [str(f).strip() for f in fields or []]
            if not fields or not all(fields) or len(set(fields)) != len(fields):
                raise AnkiToolError("Pass the field names: at least one, none empty, no repeats.")
            templates = _template_specs(templates or [])
            if not templates:
                raise AnkiToolError("Pass at least one template.")
            if kind == "cloze" and len(templates) != 1:
                raise AnkiToolError("A cloze note type has exactly one template.")

        def check(col: Collection) -> None:
            if col.models.by_name(name) is not None:
                raise AnkiToolError(f"A note type named '{name}' already exists.")
            if clone_from:
                find_note_type(col, clone_from)

        self.read(check)

        def create(col: Collection) -> int:
            if clone_from:
                model = col.models.copy(find_note_type(col, clone_from), add=False)
                model["name"] = name
            else:
                model = col.models.new(name)
                for field in fields:
                    col.models.add_field(model, col.models.new_field(field))
                for spec in templates:
                    _add_template(col, model, spec)
                if kind == "cloze":
                    model["type"] = 1
            if css is not None:
                model["css"] = css
            return col.models.add_dict(model).id

        model_id = self.write(f"create note type {name}", create)
        return self.read(lambda col: _describe_model(col, model_id))


def _describe_model(col: Collection, model_id: int) -> str:
    model = col.models.get(model_id)
    return to_json({"name": model["name"], "id": model["id"],
                    "fields": [f["name"] for f in model["flds"]],
                    "templates": [t["name"] for t in model["tmpls"]]})


class ExtendNoteTypeTool(AnkiWriteTool):
    name = "extend_note_type"
    description = (
        "Add fields and/or card templates to an existing note type. Existing notes get the new fields empty; "
        "new templates make cards only for notes whose fields the template's front uses. "
        "This changes the collection schema, so the next sync must upload the whole collection: "
        "Anki asks the user to allow it. To experiment, copy the note type with create_note_type(clone_from=...) first."
    )
    inputs = {
        "note_type": {"type": "string", "description": "Name of the note type to extend."},
        "add_fields": {"type": "array", "items": {"type": "string"}, "nullable": True,
                       "description": "Names of fields to add at the end."},
        "add_templates": {**TEMPLATE_INPUT, "nullable": True},
    }
    output_type = "string"

    def forward(self, note_type: str, add_fields: list | None = None, add_templates: list | None = None) -> str:
        add_fields = [str(f).strip() for f in add_fields or []]
        add_templates = _template_specs(add_templates or [])
        if not add_fields and not add_templates:
            raise AnkiToolError("Pass add_fields, add_templates or both.")
        if not all(add_fields) or len(set(add_fields)) != len(add_fields):
            raise AnkiToolError("Field names must not be empty or repeated.")

        def check(col: Collection) -> int | None:
            model = find_note_type(col, note_type)
            taken = {f["name"] for f in model["flds"]} & set(add_fields)
            if taken:
                raise AnkiToolError(f"Note type '{model['name']}' already has field(s) {sorted(taken)}.")
            taken = {t["name"] for t in model["tmpls"]} & {t["name"] for t in add_templates}
            if taken:
                raise AnkiToolError(f"Note type '{model['name']}' already has template(s) {sorted(taken)}.")
            if add_templates and model["type"] == 1:
                raise AnkiToolError("A cloze note type has exactly one template: no templates can be added.")
            # Inside Anki this asks the user to allow a full sync, unless the schema already changed since the last one
            try:
                col.mod_schema(check=True)
            except AbortSchemaModification:
                return None
            return model["id"]

        model_id = self.read(check)
        if model_id is None:
            return to_json({"changed": False,
                            "note": "The user declined the change: it would need a full sync. Nothing was changed."})

        def extend(col: Collection) -> None:
            model = col.models.get(model_id)
            for field in add_fields:
                col.models.add_field(model, col.models.new_field(field))
            for spec in add_templates:
                _add_template(col, model, spec)
            col.models.update_dict(model)

        what = ", ".join(add_fields + [t["name"] for t in add_templates])
        self.write(f"extend note type {note_type} with {what}", extend)
        return self.read(lambda col: _describe_model(col, model_id))


def write_tools(service: AnkiService, download_dir: Path) -> list[Tool]:
    return [CreateDeckTool(service), AddNotesTool(service), UpdateNotesTool(service), AddTagsTool(service),
            StoreMediaTool(service, download_dir), CreateNoteTypeTool(service), ExtendNoteTypeTool(service)]
