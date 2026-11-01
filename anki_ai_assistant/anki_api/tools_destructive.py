"""Destructive tools: they delete, change many notes at once, or change decks, note types and scheduling.

Each one first collects what would change into a ChangePlan and asks the user (AnkiService.ask). Only then does it act,
as one undo step "AI: ...". A refusal is the tool's result, so the model can adjust instead of failing.
Similar actions share one tool with an `action` parameter, to keep the tool list short for the model."""

import copy
import difflib
import re

from anki.collection import Collection
from anki.deck_config_pb2 import DeckConfig, UpdateDeckConfigsMode, UpdateDeckConfigsRequest as UpdateDeckConfigs
from anki.utils import ids2str
from smolagents import Tool

from .confirm import ChangePlan, item_list
from .service import AnkiService
from .tools_base import AnkiTool, AnkiToolError, find_deck, find_note_type, select_notes, short_text, to_json

MAX_NOTES = 5000

NOTE_SELECTION = {
    "note_ids": {"type": "array", "items": {"type": "integer"}, "nullable": True,
                 "description": "Note ids from search_notes. Pass either note_ids or query."},
    "query": {"type": "string", "nullable": True,
              "description": "Anki search selecting the notes, e.g. 'deck:German tag:verbs'. Pass either note_ids or query."},
}


class AnkiDestructiveTool(AnkiTool):
    risk = "destructive"


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" + ("" if count == 1 else "s")


def _note_lines(col: Collection, note_ids: list[int]) -> list[str]:
    lines = []
    for note_id in note_ids[:30]:
        note = col.get_note(note_id)
        cards = note.cards()
        deck = col.decks.name(cards[0].odid or cards[0].did) if cards else "no cards"
        lines.append(f"{note_id}  {short_text(note.fields[0], 60)}  ({note.note_type()['name']}, {deck})")
    return lines


def _deck_counts(col: Collection, note_ids: list[int]) -> str:
    rows = col.db.all("select case when odid then odid else did end as deck, count() from cards "
                      f"where nid in {ids2str(note_ids)} group by deck")
    return ", ".join(f"{col.decks.name(deck)}: {_plural(count, 'card')}" for deck, count in rows)


def _diff(old: str, new: str) -> str:
    return "\n".join(difflib.unified_diff(old.splitlines(), new.splitlines(), "before", "after", lineterm=""))


class DeleteNotesTool(AnkiDestructiveTool):
    name = "delete_notes"
    description = ("Delete notes together with their cards and review history. The user sees what will be deleted "
                   f"and must confirm. At most {MAX_NOTES} notes per call.")
    inputs = dict(NOTE_SELECTION)
    output_type = "string"

    def forward(self, note_ids: list | None = None, query: str | None = None) -> str:
        def plan(col: Collection):
            ids = select_notes(col, note_ids, query, MAX_NOTES)
            card_count = col.db.scalar(f"select count() from cards where nid in {ids2str(ids)}")
            return ids, ChangePlan(
                kind="delete_notes",
                title=f"Delete {_plural(len(ids), 'note')} ({_plural(card_count, 'card')})",
                summary=f"The notes, their cards and review history are deleted. {_deck_counts(col, ids)}.",
                details=item_list(_note_lines(col, ids), len(ids)))

        ids, change = self.read(plan)
        if declined := self.ask(change):
            return declined
        self.write(f"delete {_plural(len(ids), 'note')}", lambda col: col.remove_notes(ids))
        return to_json({"done": True, "deleted_notes": len(ids)})


class ModifyDeckTool(AnkiDestructiveTool):
    name = "modify_deck"
    description = (
        "Rename or move a deck (action='rename', new_name is the full new name: 'German::Verbs' moves it under "
        "German), or delete a deck with its subdecks and all their cards (action='delete'). The user must confirm."
    )
    inputs = {
        "action": {"type": "string", "enum": ["rename", "delete"], "description": "rename or delete."},
        "deck": {"type": "string", "description": "Full name of the deck, as list_decks shows it."},
        "new_name": {"type": "string", "nullable": True, "description": "For rename: the full new deck name."},
    }
    output_type = "string"

    def forward(self, action: str, deck: str, new_name: str | None = None) -> str:
        if action == "rename":
            return self._rename(deck, new_name)
        if action == "delete":
            return self._delete(deck)
        raise AnkiToolError("action must be 'rename' or 'delete'.")

    def _rename(self, deck: str, new_name: str | None) -> str:
        new_name = (new_name or "").strip()
        if not new_name or any(not part.strip() for part in new_name.split("::")):
            raise AnkiToolError("Pass new_name, a full deck name like 'German' or 'German::Verbs'.")

        def plan(col: Collection):
            deck_dict = find_deck(col, deck)
            if col.decks.id_for_name(new_name) is not None:
                raise AnkiToolError(f"A deck named '{new_name}' already exists.")
            if new_name.startswith(deck_dict["name"] + "::"):
                raise AnkiToolError("A deck cannot be moved into its own subdeck.")
            subdecks = [name for name, _ in col.decks.deck_and_child_name_ids(deck_dict["id"])][1:]
            details = item_list([f"{name}  →  {new_name}{name[len(deck_dict['name']):]}" for name in subdecks])
            return deck_dict["id"], ChangePlan(
                kind="rename_deck", title=f"Rename deck '{deck_dict['name']}' to '{new_name}'",
                summary="The cards stay in the deck." + (" Its subdecks move with it:" if subdecks else ""),
                details=details)

        deck_id, change = self.read(plan)
        if declined := self.ask(change):
            return declined
        self.write(f"rename deck {deck} to {new_name}", lambda col: col.decks.rename(deck_id, new_name))
        return to_json({"done": True, "deck": new_name})

    def _delete(self, deck: str) -> str:
        def plan(col: Collection):
            deck_dict = find_deck(col, deck)
            if deck_dict["id"] == 1:
                raise AnkiToolError("The Default deck cannot be deleted.")
            decks = list(col.decks.deck_and_child_name_ids(deck_dict["id"]))
            deck_ids = [deck_id for _, deck_id in decks]
            if deck_dict["dyn"]:
                summary = "This is a filtered deck: its cards go back to their home decks, nothing else is deleted."
                cards = 0
            else:
                cards = col.db.scalar(f"select count() from cards where did in {ids2str(deck_ids)} "
                                      f"or odid in {ids2str(deck_ids)}")
                summary = (f"{_plural(cards, 'card')} and their review history are deleted with the deck"
                           + (" and its subdecks." if len(decks) > 1 else "."))
            return deck_dict["id"], cards, ChangePlan(
                kind="delete_deck", title=f"Delete deck '{deck_dict['name']}'", summary=summary,
                details=item_list([name for name, _ in decks]))

        deck_id, cards, change = self.read(plan)
        if declined := self.ask(change):
            return declined
        self.write(f"delete deck {deck}", lambda col: col.decks.remove([deck_id]))
        return to_json({"done": True, "deleted_deck": deck, "deleted_cards": cards})


# Anki's replacement syntax (Rust regex): $1, ${1}, ${name}, $name; $$ is a literal $
_REPLACEMENT_REF = re.compile(r"\$(?:(\$)|\{(\w+)\}|(\w+))")


def _expand(match: re.Match, template: str) -> str:
    def ref(r: re.Match) -> str:
        if r.group(1):
            return "$"
        key = r.group(2) or r.group(3)
        try:
            return match.group(int(key) if key.isdigit() else key) or ""
        except (IndexError, re.error):
            return ""

    return _REPLACEMENT_REF.sub(ref, template)


class FindReplaceTool(AnkiDestructiveTool):
    name = "find_replace"
    description = (
        "Find and replace text in the fields of many notes at once. Without regex the text is found literally, "
        "ignoring case unless match_case is true. With regex=true, find is a regular expression and the "
        "replacement refers to groups as $1 or ${name}. Fields contain HTML: the text is replaced in it as is. "
        "The user sees the changes and must confirm."
    )
    inputs = {
        **NOTE_SELECTION,
        "find": {"type": "string", "description": "Text or regular expression to find."},
        "replace": {"type": "string", "description": "Replacement; may be empty."},
        "field": {"type": "string", "nullable": True, "description": "Only this field (default: all fields)."},
        "regex": {"type": "boolean", "nullable": True, "description": "find is a regular expression (default false)."},
        "match_case": {"type": "boolean", "nullable": True, "description": "Case-sensitive (default false)."},
    }
    output_type = "string"

    def forward(self, find: str, replace: str, note_ids: list | None = None, query: str | None = None,
                field: str | None = None, regex: bool | None = None, match_case: bool | None = None) -> str:
        if not find:
            raise AnkiToolError("Pass the text to find.")
        try:
            pattern = re.compile(find if regex else re.escape(find), 0 if match_case else re.IGNORECASE)
        except re.error as e:
            raise AnkiToolError(f"Invalid regular expression: {e}.")
        # Anki expands $ in the replacement even without regex: escape it so the text is inserted literally
        replacement = replace if regex else replace.replace("$", "$$")

        def plan(col: Collection):
            ids = select_notes(col, note_ids, query, MAX_NOTES)
            changed, lines, field_found = [], [], field is None
            for note_id in ids:
                note = col.get_note(note_id)
                names = [field] if field is not None else list(note.keys())
                note_changed = False
                for name in names:
                    if name not in note:
                        continue
                    field_found = True
                    new = pattern.sub(lambda m: _expand(m, replacement), note[name])
                    if new != note[name]:
                        note_changed = True
                        if len(lines) < 60:
                            lines += [f"{note_id} {name}:", f"- {short_text(note[name])}", f"+ {short_text(new)}"]
                if note_changed:
                    changed.append(note_id)
            if not field_found:
                raise AnkiToolError(f"None of the selected notes has a field named '{field}'.")
            if len(lines) >= 60 and len(changed) > 20:
                lines.append(f"… and more, {_plural(len(changed), 'note')} in all")
            return changed, ChangePlan(
                kind="find_replace", title=f"Replace text in {_plural(len(changed), 'note')}",
                summary=f"'{find}' → '{replace}'" + (f" in the field {field}" if field else " in all fields")
                        + (" (regular expression)" if regex else "") + ".",
                details="\n".join(lines))

        changed, change = self.read(plan)
        if not changed:
            return to_json({"done": False, "changed_notes": 0, "note": "Nothing matched: no note was changed."})
        if declined := self.ask(change):
            return declined
        result = self.write(f"replace '{find}' in {_plural(len(changed), 'note')}", lambda col: col.find_and_replace(
            note_ids=changed, search=find, replacement=replacement, regex=bool(regex), field_name=field,
            match_case=bool(match_case)))
        return to_json({"done": True, "changed_notes": result.count})


class ChangeNoteTypeTool(AnkiDestructiveTool):
    name = "change_note_type"
    description = (
        "Convert notes to another note type, e.g. from 'Basic' to 'Basic (and reversed card)'. All the notes must "
        "have the same note type now. Fields are matched by name, then by position, unless field_map says otherwise; "
        "content of old fields that are not mapped is lost. Needs a full sync. The user must confirm."
    )
    inputs = {
        **NOTE_SELECTION,
        "new_note_type": {"type": "string", "description": "Name of the note type to convert to."},
        "field_map": {"type": "object", "nullable": True, "additionalProperties": {"type": "string"},
                      "description": "{new field: old field} to take each new field's content from; "
                                     "an empty string leaves that new field empty. Unlisted fields use the default."},
    }
    output_type = "string"

    def forward(self, new_note_type: str, note_ids: list | None = None, query: str | None = None,
                field_map: dict | None = None) -> str:
        def mapping(col: Collection, ids: list[int]):
            type_ids = col.db.list(f"select distinct mid from notes where id in {ids2str(ids)}")
            if len(type_ids) > 1:
                names = ", ".join(col.models.get(i)["name"] for i in type_ids)
                raise AnkiToolError(f"The notes have different note types ({names}): change one note type per call.")
            old, new = col.models.get(type_ids[0]), find_note_type(col, new_note_type)
            if old["id"] == new["id"]:
                raise AnkiToolError(f"The notes already have the note type '{new['name']}'.")
            info = col.models.change_notetype_info(old_notetype_id=old["id"], new_notetype_id=new["id"])
            new_fields = list(info.input.new_fields)
            old_names, new_names = list(info.old_field_names), list(info.new_field_names)
            for new_name, old_name in (field_map or {}).items():
                if new_name not in new_names:
                    raise AnkiToolError(f"'{new['name']}' has no field '{new_name}'. Its fields: {', '.join(new_names)}.")
                if old_name and old_name not in old_names:
                    raise AnkiToolError(f"'{old['name']}' has no field '{old_name}'. Its fields: {', '.join(old_names)}.")
                new_fields[new_names.index(new_name)] = old_names.index(old_name) if old_name else -1
            return info, new_fields

        def plan(col: Collection):
            ids = select_notes(col, note_ids, query, MAX_NOTES)
            info, new_fields = mapping(col, ids)
            old_names, new_names = list(info.old_field_names), list(info.new_field_names)
            lines = ["Fields:"] + [f"  {name}  ←  {old_names[i] if i >= 0 else '(empty)'}"
                                   for name, i in zip(new_names, new_fields)]
            lost = [name for i, name in enumerate(old_names) if i not in new_fields]
            if lost:
                lines.append(f"- Content lost: {', '.join(lost)}")
            old_templates, new_templates = list(info.old_template_names), list(info.new_template_names)
            templates = list(info.input.new_templates)
            lines.append("Cards:")
            lines += [f"  {name}  ←  {old_templates[i] if i >= 0 else '(new cards where the fields allow)'}"
                      for name, i in zip(new_templates, templates)]
            removed = [name for i, name in enumerate(old_templates) if i not in templates]
            if removed:
                lines.append(f"- Cards deleted with their review history: {', '.join(removed)}")
            return ids, ChangePlan(
                kind="change_note_type", full_sync=not col.schema_changed(),
                title=f"Change {_plural(len(ids), 'note')} from '{info.old_notetype_name}' to '{new_note_type}'",
                summary="The notes keep their ids and decks; fields and cards are moved as listed.",
                details="\n".join(lines))

        ids, change = self.read(plan)
        if declined := self.ask(change):
            return declined

        def convert(col: Collection) -> None:
            # Built again here: the request carries the schema time, which must be current
            info, new_fields = mapping(col, ids)
            request = info.input
            request.note_ids.extend(ids)
            del request.new_fields[:]
            request.new_fields.extend(new_fields)
            col.models.change_notetype_of_notes(request)

        self.write(f"change {_plural(len(ids), 'note')} to {new_note_type}", convert)
        return to_json({"done": True, "changed_notes": len(ids), "note_type": new_note_type})


class ModifyNoteTypeTool(AnkiDestructiveTool):
    name = "modify_note_type"
    description = (
        "Change an existing note type; it affects every note of this type. Actions: "
        "rename_field (field, new_name), remove_field (field; its content is lost), "
        "reposition_field (field, position starting at 1), "
        "edit_template (template, front and/or back: the full new HTML), "
        "remove_template (template; its cards are deleted), edit_css (css: the full new CSS). "
        "Field and template removal, renames and moves need a full sync. Get the current HTML and CSS with "
        "get_note_type first. Templates must work in AnkiDroid: no JavaScript. The user must confirm."
    )
    inputs = {
        "note_type": {"type": "string", "description": "Name of the note type."},
        "action": {"type": "string", "enum": ["rename_field", "remove_field", "reposition_field", "edit_template",
                                              "remove_template", "edit_css"], "description": "What to change."},
        "field": {"type": "string", "nullable": True, "description": "Field name, for the field actions."},
        "template": {"type": "string", "nullable": True, "description": "Template name, for the template actions."},
        "new_name": {"type": "string", "nullable": True, "description": "For rename_field."},
        "position": {"type": "integer", "nullable": True, "description": "For reposition_field: 1 is first."},
        "front": {"type": "string", "nullable": True, "description": "For edit_template: new front HTML."},
        "back": {"type": "string", "nullable": True, "description": "For edit_template: new back HTML."},
        "css": {"type": "string", "nullable": True, "description": "For edit_css: the new CSS."},
    }
    output_type = "string"

    def forward(self, note_type: str, action: str, field: str | None = None, template: str | None = None,
                new_name: str | None = None, position: int | None = None, front: str | None = None,
                back: str | None = None, css: str | None = None) -> str:
        def item(model: dict, key: str, name: str | None, what: str) -> dict:
            if not name:
                raise AnkiToolError(f"Pass the {what} name for {action}.")
            for entry in model[key]:
                if entry["name"] == name:
                    return entry
            known = ", ".join(entry["name"] for entry in model[key])
            raise AnkiToolError(f"Note type '{model['name']}' has no {what} '{name}'. Its {what}s: {known}.")

        def change(col: Collection, model: dict) -> tuple[str, str, bool]:
            """Apply the action to the note type dict; return the plan's title, details and whether the schema changes."""
            if action == "rename_field":
                entry = item(model, "flds", field, "field")
                if not (new_name or "").strip():
                    raise AnkiToolError("Pass new_name for rename_field.")
                if any(f["name"] == new_name.strip() for f in model["flds"]):
                    raise AnkiToolError(f"The note type already has a field '{new_name.strip()}'.")
                col.models.rename_field(model, entry, new_name.strip())
                return f"Rename field '{field}' to '{new_name.strip()}'", "Templates using it are updated too.", True
            if action == "remove_field":
                entry = item(model, "flds", field, "field")
                if len(model["flds"]) == 1:
                    raise AnkiToolError("A note type needs at least one field.")
                index = model["flds"].index(entry)
                filled = sum(1 for (flds,) in col.db.all("select flds from notes where mid = ?", model["id"])
                             if flds.split("\x1f")[index].strip())
                col.models.remove_field(model, entry)
                return (f"Remove field '{field}'",
                        f"- Its content is lost in {_plural(filled, 'note')}.", True)
            if action == "reposition_field":
                entry = item(model, "flds", field, "field")
                if position is None or not 1 <= int(position) <= len(model["flds"]):
                    raise AnkiToolError(f"Pass position from 1 to {len(model['flds'])}.")
                col.models.reposition_field(model, entry, int(position) - 1)
                order = " | ".join(f["name"] for f in model["flds"])
                return f"Move field '{field}' to position {position}", f"New order: {order}", True
            if action == "edit_template":
                entry = item(model, "tmpls", template, "template")
                if front is None and back is None:
                    raise AnkiToolError("Pass front, back or both for edit_template.")
                parts = []
                for key, value, label in (("qfmt", front, "Front"), ("afmt", back, "Back")):
                    if value is not None and value != entry[key]:
                        parts.append(f"{label}:\n{_diff(entry[key], value)}")
                        entry[key] = value
                return f"Edit template '{template}'", "\n\n".join(parts), False
            if action == "remove_template":
                entry = item(model, "tmpls", template, "template")
                if len(model["tmpls"]) == 1:
                    raise AnkiToolError("A note type needs at least one template.")
                cards = col.models.template_use_count(model["id"], entry["ord"])
                col.models.remove_template(model, entry)
                return (f"Remove template '{template}'",
                        f"- {_plural(cards, 'card')} made from it are deleted with their review history.", True)
            if action == "edit_css":
                if css is None:
                    raise AnkiToolError("Pass css for edit_css.")
                details = _diff(model["css"], css)
                model["css"] = css
                return "Edit the card styling (CSS)", details, False
            raise AnkiToolError(f"Unknown action {action!r}.")

        def plan(col: Collection):
            # A copy: Anki caches note type dicts, and a declined change must not stay in the cache
            model = copy.deepcopy(find_note_type(col, note_type))
            title, details, schema = change(col, model)
            if not details:
                return model["id"], None
            notes = col.models.use_count(model)
            return model["id"], ChangePlan(
                kind=f"modify_note_type:{action}", full_sync=schema and not col.schema_changed(),
                title=f"{title} in note type '{model['name']}'",
                summary=f"Affects all {_plural(notes, 'note')} of this note type.", details=details)

        model_id, plan_ = self.read(plan)
        if plan_ is None:
            return to_json({"done": False, "note": "Nothing to change: the new content is the same as the current."})
        if declined := self.ask(plan_):
            return declined

        def apply(col: Collection) -> None:
            model = copy.deepcopy(col.models.get(model_id))
            change(col, model)
            col.models.update_dict(model)

        self.write(f"{action.replace('_', ' ')} in note type {note_type}", apply)
        return to_json({"done": True, "action": action, "note_type": note_type})


# get_deck_options names → fields of the preset (DeckConfig.Config) and their type
DECK_OPTIONS = {
    "new_cards_per_day": ("new_per_day", int),
    "reviews_per_day": ("reviews_per_day", int),
    "learning_steps_minutes": ("learn_steps", list),
    "relearning_steps_minutes": ("relearn_steps", list),
    "graduating_interval_days": ("graduating_interval_good", int),
    "easy_interval_days": ("graduating_interval_easy", int),
    "maximum_interval_days": ("maximum_review_interval", int),
    "leech_threshold": ("leech_threshold", int),
    "desired_retention": ("desired_retention", float),
}


def _option_value(config, name: str):
    value = getattr(config, DECK_OPTIONS[name][0])
    # Steps and the retention are stored as 32-bit floats: 0.9 comes back as 0.8999999761
    if DECK_OPTIONS[name][1] is list:
        return [round(v, 4) for v in value]
    return round(value, 4) if isinstance(value, float) else value


def _set_option(config, name: str, value) -> None:
    field = DECK_OPTIONS[name][0]
    if DECK_OPTIONS[name][1] is list:
        del getattr(config, field)[:]
        getattr(config, field).extend(value)
    else:
        setattr(config, field, value)


def _convert_option(name: str, value):
    kind = DECK_OPTIONS[name][1]
    try:
        if kind is list:
            if not isinstance(value, list) or not value or any(float(v) <= 0 for v in value):
                raise ValueError
            return [float(v) for v in value]
        converted = kind(value)
    except (TypeError, ValueError):
        raise AnkiToolError(f"Invalid value {value!r} for {name}.")
    if name == "desired_retention" and not 0.7 <= converted <= 0.99:
        raise AnkiToolError("desired_retention must be between 0.70 and 0.99.")
    if kind is int and converted < 0:
        raise AnkiToolError(f"{name} cannot be negative.")
    return converted


def _current_preset(col: Collection, deck_id: int):
    """The deck's options as the deck options screen gets them, and its preset."""
    for_update = col.decks.get_deck_configs_for_update(deck_id)
    preset = next(c for c in for_update.all_config if c.config.id == for_update.current_deck.config_id)
    return for_update, preset


class SetDeckOptionsTool(AnkiDestructiveTool):
    name = "set_deck_options"
    description = (
        "Change study options of a deck (see get_deck_options for the current ones). Options live in a preset that "
        "several decks may share: the change applies to all of them, unless new_preset is given, which moves this "
        "deck to a new preset copied from its current one and changes only that. The user must confirm."
    )
    inputs = {
        "deck": {"type": "string", "description": "Full deck name."},
        "options": {"type": "object", "additionalProperties": False,
                    "properties": {
                        "new_cards_per_day": {"type": "integer"}, "reviews_per_day": {"type": "integer"},
                        "learning_steps_minutes": {"type": "array", "items": {"type": "number"}},
                        "relearning_steps_minutes": {"type": "array", "items": {"type": "number"}},
                        "graduating_interval_days": {"type": "integer"}, "easy_interval_days": {"type": "integer"},
                        "maximum_interval_days": {"type": "integer"}, "leech_threshold": {"type": "integer"},
                        "desired_retention": {"type": "number"}},
                    "description": "Only the options to change, with their new values."},
        "new_preset": {"type": "string", "nullable": True,
                       "description": "Name for a new preset just for this deck."},
    }
    output_type = "string"

    def forward(self, deck: str, options: dict, new_preset: str | None = None) -> str:
        unknown = [name for name in options or {} if name not in DECK_OPTIONS]
        if unknown or not options:
            raise AnkiToolError(f"Pass options to change; known ones: {', '.join(DECK_OPTIONS)}."
                                + (f" Unknown: {', '.join(unknown)}." if unknown else ""))
        values = {name: _convert_option(name, value) for name, value in options.items()}
        new_preset = (new_preset or "").strip() or None

        def plan(col: Collection):
            deck_dict = find_deck(col, deck)
            if deck_dict["dyn"]:
                raise AnkiToolError("This is a filtered deck: it has no options preset.")
            for_update, preset = _current_preset(col, deck_dict["id"])
            if new_preset and any(c.config.name == new_preset for c in for_update.all_config):
                raise AnkiToolError(f"A preset named '{new_preset}' already exists.")
            config = preset.config.config
            lines = [f"{name}: {_option_value(config, name)} → {value}"
                     for name, value in values.items() if _option_value(config, name) != value]
            name = preset.config.name
            sharing = [col.decks.name(i) for i in col.decks.decks_using_config(col.decks.get_config(preset.config.id))
                       if i != deck_dict["id"]]
            if new_preset:
                summary = f"The deck moves from preset '{name}' to the new preset '{new_preset}'."
            elif sharing:
                summary = (f"Preset '{name}' is shared: the change also applies to "
                           f"{_plural(len(sharing), 'other deck')}: {', '.join(sharing[:10])}"
                           + ("…" if len(sharing) > 10 else "") + ".")
            else:
                summary = f"Preset '{name}' is used only by this deck."
            return deck_dict["id"], lines, ChangePlan(
                kind="set_deck_options", title=f"Change the options of deck '{deck_dict['name']}'",
                summary=summary, details="\n".join(lines))

        deck_id, lines, change = self.read(plan)
        if not lines and not new_preset:
            return to_json({"done": False, "note": "The options already have these values: nothing changed."})
        if declined := self.ask(change):
            return declined

        def apply(col: Collection) -> None:
            # What the deck options screen saves: unlike the legacy update_config, it is undoable
            for_update, preset = _current_preset(col, deck_id)
            conf = DeckConfig()
            conf.CopyFrom(preset.config)
            if new_preset:
                conf.id, conf.name = 0, new_preset
            for name, value in values.items():
                _set_option(conf.config, name, value)
            col.decks.update_deck_configs(UpdateDeckConfigs(
                target_deck_id=deck_id, configs=[conf], mode=UpdateDeckConfigsMode.UPDATE_DECK_CONFIGS_MODE_NORMAL,
                card_state_customizer=for_update.card_state_customizer, limits=for_update.current_deck.limits,
                new_cards_ignore_review_limit=for_update.new_cards_ignore_review_limit, fsrs=for_update.fsrs,
                apply_all_parent_limits=for_update.apply_all_parent_limits,
                fsrs_health_check=for_update.fsrs_health_check))

        self.write(f"change options of {deck}", apply)
        return to_json({"done": True, "deck": deck, "changed": lines,
                        **({"preset": new_preset} if new_preset else {})})


SET_DUE_DAYS = re.compile(r"^\d+(-\d+)?!?$")


class RescheduleCardsTool(AnkiDestructiveTool):
    name = "reschedule_cards"
    description = (
        "Change the scheduling of cards: suspend or unsuspend them (runs at once), forget them (they become new; "
        "with reset_counts the review and lapse counts are reset too), or set_due (days: '0' today, '7' in a week, "
        "'1-7' spread over a range; a trailing '!' also sets the interval). forget and set_due need the user's "
        "confirmation. query is a card search here, e.g. 'deck:German card:2' for the reverse cards only."
    )
    inputs = {
        "action": {"type": "string", "enum": ["suspend", "unsuspend", "forget", "set_due"],
                   "description": "What to do with the cards."},
        "note_ids": {"type": "array", "items": {"type": "integer"}, "nullable": True,
                     "description": "All cards of these notes. Pass either note_ids or query."},
        "query": {"type": "string", "nullable": True, "description": "Anki search selecting the cards."},
        "days": {"type": "string", "nullable": True, "description": "For set_due, e.g. '0', '3', '1-7'."},
        "reset_counts": {"type": "boolean", "nullable": True,
                         "description": "For forget: also reset the review and lapse counts (default false)."},
    }
    output_type = "string"

    def forward(self, action: str, note_ids: list | None = None, query: str | None = None, days: str | None = None,
                reset_counts: bool | None = None) -> str:
        if action not in ("suspend", "unsuspend", "forget", "set_due"):
            raise AnkiToolError("action must be suspend, unsuspend, forget or set_due.")
        days = (days or "").strip()
        if action == "set_due" and not SET_DUE_DAYS.match(days):
            raise AnkiToolError("Pass days for set_due like '0', '3', '1-7' or '3!'.")

        def select(col: Collection) -> list[int]:
            if bool(note_ids) == bool(query and query.strip()):
                raise AnkiToolError("Pass either note_ids or query, not both.")
            if note_ids:
                ids = list(col.find_cards(f"nid:{','.join(str(int(i)) for i in note_ids)}"))
            else:
                ids = list(col.find_cards(query))
            if not ids:
                raise AnkiToolError("No cards match: nothing was changed.")
            if len(ids) > MAX_NOTES:
                raise AnkiToolError(f"{len(ids)} cards: at most {MAX_NOTES} per call. Narrow the selection.")
            return ids

        card_ids = self.read(select)
        what = _plural(len(card_ids), "card")
        if action in ("forget", "set_due"):
            def plan(col: Collection) -> ChangePlan:
                note_ids_ = col.db.list(f"select distinct nid from cards where id in {ids2str(card_ids)}")
                if action == "forget":
                    title = f"Forget {what}"
                    summary = ("The cards become new and are shown again as new cards"
                               + (", and their review and lapse counts are reset." if reset_counts else
                                  ". Their review history is kept."))
                else:
                    title, summary = f"Set the due date of {what}", f"The cards become due in {days} day(s)."
                return ChangePlan(kind=f"reschedule:{action}", title=title,
                                  summary=f"{summary} {_deck_counts(col, note_ids_)}.",
                                  details=item_list(_note_lines(col, note_ids_), len(note_ids_)))

            if declined := self.ask(self.read(plan)):
                return declined

        def apply(col: Collection) -> None:
            if action == "suspend":
                col.sched.suspend_cards(card_ids)
            elif action == "unsuspend":
                col.sched.unsuspend_cards(card_ids)
            elif action == "forget":
                col.sched.schedule_cards_as_new(card_ids, restore_position=True, reset_counts=bool(reset_counts))
            else:
                col.sched.set_due_date(card_ids, days)

        self.write(f"{action.replace('_', ' ')} {what}", apply)
        return to_json({"done": True, "action": action, "cards": len(card_ids)})


def _tag_search(tag: str) -> str:
    """A search for the tag and its child tags, with Anki's wildcards in the name taken literally."""
    escaped = re.sub(r'([\\*_"])', r"\\\1", tag)
    return f'"tag:{escaped}"'


class ModifyTagsTool(AnkiDestructiveTool):
    name = "modify_tags"
    description = (
        "Rename a tag on all notes (action='rename', with new_name; child tags 'tag::child' are renamed too), "
        "or remove a tag (action='remove'): from the selected notes if note_ids or query is given, "
        "otherwise from every note. The user must confirm. To add tags use add_tags."
    )
    inputs = {
        "action": {"type": "string", "enum": ["rename", "remove"], "description": "rename or remove."},
        "tag": {"type": "string", "description": "The tag, as list_tags shows it."},
        "new_name": {"type": "string", "nullable": True, "description": "For rename: the new tag, without spaces."},
        **NOTE_SELECTION,
    }
    output_type = "string"

    def forward(self, action: str, tag: str, new_name: str | None = None, note_ids: list | None = None,
                query: str | None = None) -> str:
        tag = tag.strip()
        if not tag or any(c.isspace() for c in tag):
            raise AnkiToolError("Pass one tag, without spaces.")
        if action == "rename":
            new_name = (new_name or "").strip()
            if not new_name or any(c.isspace() for c in new_name):
                raise AnkiToolError("Pass new_name, a tag without spaces.")
            if note_ids or query:
                raise AnkiToolError("rename applies to all notes: do not pass note_ids or query.")
        elif action != "remove":
            raise AnkiToolError("action must be 'rename' or 'remove'.")

        def plan(col: Collection):
            tagged = set(col.find_notes(_tag_search(tag)))
            if not tagged:
                raise AnkiToolError(f"No note has the tag '{tag}'. See list_tags.")
            ids = sorted(tagged & set(select_notes(col, note_ids, query, MAX_NOTES))) if (note_ids or query) \
                else sorted(tagged)
            if not ids:
                raise AnkiToolError(f"None of the selected notes has the tag '{tag}'.")
            title = (f"Rename tag '{tag}' to '{new_name}'" if action == "rename"
                     else f"Remove tag '{tag}' from {_plural(len(ids), 'note')}")
            return ids, ChangePlan(kind=f"modify_tags:{action}", title=title,
                                   summary=f"Affects {_plural(len(ids), 'note')}; their other tags stay.",
                                   details=item_list(_note_lines(col, ids), len(ids)))

        ids, change = self.read(plan)
        if declined := self.ask(change):
            return declined
        if action == "rename":
            result = self.write(f"rename tag {tag} to {new_name}", lambda col: col.tags.rename(tag, new_name))
        else:
            result = self.write(f"remove tag {tag}", lambda col: col.tags.bulk_remove(ids, tag))
        return to_json({"done": True, "action": action, "notes_changed": result.count})


def destructive_tools(service: AnkiService) -> list[Tool]:
    return [DeleteNotesTool(service), ModifyDeckTool(service), FindReplaceTool(service),
            ChangeNoteTypeTool(service), ModifyNoteTypeTool(service), SetDeckOptionsTool(service),
            RescheduleCardsTool(service), ModifyTagsTool(service)]
