import json

import pytest
from smolagents.models import get_tool_json_schema

from anki_ai_assistant.anki_api.confirm import Decision
from anki_ai_assistant.anki_api.service import AnkiService
from anki_ai_assistant.anki_api.tools_destructive import (ChangeNoteTypeTool, DeleteNotesTool, FindReplaceTool,
                                                          ModifyDeckTool, ModifyNoteTypeTool, ModifyTagsTool,
                                                          RescheduleCardsTool, SetDeckOptionsTool, destructive_tools)
from anki_ai_assistant.anki_api.tools_write import UpdateNotesTool
from conftest import add_note


class Asker:
    """Answers confirmations like the user would and remembers what was asked."""

    def __init__(self, approve: bool = True, comment: str = "", trust: bool = False):
        self.approve, self.comment, self.trust = approve, comment, trust
        self.plans = []

    def __call__(self, plan):
        self.plans.append(plan)
        return Decision(self.approve, self.comment, trust_kind=self.trust)


@pytest.fixture
def asker():
    return Asker()


@pytest.fixture
def service(col, asker):
    return AnkiService(lambda: col, confirm=asker)


def call(tool, **kwargs):
    return json.loads(tool(**kwargs))


def mark_synced(col):
    col.db.execute("update col set ls = scm + 1")


def test_all_destructive_tools_are_valid_smolagents_tools(service):
    for tool in destructive_tools(service):
        assert tool.risk == "destructive"
        assert get_tool_json_schema(tool)["function"]["name"] == tool.name


def test_declined_change_changes_nothing(service, asker, col, german):
    asker.approve, asker.comment = False, "keep the cat"
    result = call(DeleteNotesTool(service), query="deck:German::Nouns")
    assert result["declined_by_user"] is True and result["user_comment"] == "keep the cat"
    assert col.note_count() == 4
    assert not col.undo_status().undo.startswith("AI: ")


def test_without_a_way_to_ask_destructive_changes_are_declined(col, german):
    result = call(DeleteNotesTool(AnkiService(lambda: col)), note_ids=[german["hund"]])
    assert result["declined_by_user"] is True
    assert col.note_count() == 4


def test_trusted_kind_is_not_asked_again(service, asker, col, german):
    asker.trust = True
    call(DeleteNotesTool(service), note_ids=[german["hund"]])
    call(DeleteNotesTool(service), note_ids=[german["katze"]])
    assert len(asker.plans) == 1
    assert col.note_count() == 2
    # Another kind of action is still asked
    call(ModifyTagsTool(service), action="remove", tag="verbs")
    assert len(asker.plans) == 2


def test_delete_notes(service, asker, col, german):
    result = call(DeleteNotesTool(service), query="deck:German::Nouns")
    assert result == {"done": True, "deleted_notes": 2}
    plan = asker.plans[0]
    assert plan.title == "Delete 2 notes (2 cards)"
    assert "German::Nouns: 2 cards" in plan.summary and "der Hund" in plan.details
    assert col.undo_status().undo == "AI: delete 2 notes"
    col.undo()
    assert col.note_count() == 4


def test_delete_notes_selection_errors(service, german):
    with pytest.raises(Exception, match="either note_ids or query"):
        DeleteNotesTool(service)(note_ids=[german["hund"]], query="deck:German")
    with pytest.raises(Exception, match="No notes with ids"):
        DeleteNotesTool(service)(note_ids=[123])
    with pytest.raises(Exception, match="No notes match"):
        DeleteNotesTool(service)(query="deck:Nothing")


def test_rename_deck_moves_subdecks(service, asker, col, german):
    assert call(ModifyDeckTool(service), action="rename", deck="German", new_name="Languages::German")["done"]
    assert col.decks.id_for_name("Languages::German::Nouns") is not None
    assert "German::Nouns  →  Languages::German::Nouns" in asker.plans[0].details
    with pytest.raises(Exception, match="own subdeck"):
        ModifyDeckTool(service)(action="rename", deck="Languages", new_name="Languages::Sub")


def test_delete_deck(service, asker, col, german):
    result = call(ModifyDeckTool(service), action="delete", deck="German")
    assert result["deleted_cards"] == 4
    assert col.note_count() == 0 and col.decks.id_for_name("German::Nouns") is None
    assert "4 cards" in asker.plans[0].summary
    col.undo()
    assert col.note_count() == 4
    with pytest.raises(Exception, match="Default deck"):
        ModifyDeckTool(service)(action="delete", deck="Default")


def test_find_replace_literal(service, asker, col, german):
    # Without regex: case-insensitive, and $ is inserted literally
    result = call(FindReplaceTool(service), query="deck:German::Nouns", find="THE", replace="a $1", field="Back")
    assert result == {"done": True, "changed_notes": 2}
    assert col.get_note(german["hund"])["Back"] == "a $1 dog"
    assert "- the dog" in asker.plans[0].details and "+ a $1 dog" in asker.plans[0].details


def test_find_replace_regex_preview_matches_anki(service, asker, col, german):
    call(FindReplaceTool(service), query="deck:German::Nouns", find=r"^(der|die) (\w+)", replace="$2, ${1}",
         regex=True, match_case=True)
    assert col.get_note(german["hund"])["Front"] == "Hund, der"
    assert "+ Hund, der" in asker.plans[0].details and "+ Katze, die" in asker.plans[0].details


def test_find_replace_without_matches_asks_nothing(service, asker, german):
    assert call(FindReplaceTool(service), query="deck:German", find="zebra", replace="x")["changed_notes"] == 0
    assert asker.plans == []
    with pytest.raises(Exception, match="field named 'Nope'"):
        FindReplaceTool(service)(query="deck:German::Nouns", find="a", replace="b", field="Nope")


def test_change_note_type(service, asker, col, german):
    mark_synced(col)
    result = call(ChangeNoteTypeTool(service), note_ids=[german["hund"], german["katze"]],
                  new_note_type="Basic (and reversed card)", field_map={"Back": ""})
    assert result["done"]
    note = col.get_note(german["hund"])
    assert note.note_type()["name"] == "Basic (and reversed card)" and note["Back"] == ""
    plan = asker.plans[0]
    assert plan.full_sync and "Content lost: Back" in plan.details
    with pytest.raises(Exception, match="different note types"):
        ChangeNoteTypeTool(service)(query="deck:German", new_note_type="Basic")


def test_modify_note_type_edit_template_shows_a_diff(service, asker, col, german):
    call(ModifyNoteTypeTool(service), note_type="Basic", action="edit_template", template="Card 1",
         front="<b>{{Front}}</b>")
    plan = asker.plans[0]
    assert "-{{Front}}" in plan.details and "+<b>{{Front}}</b>" in plan.details and not plan.full_sync
    assert col.models.by_name("Basic")["tmpls"][0]["qfmt"] == "<b>{{Front}}</b>"


def test_modify_note_type_declined_leaves_the_cached_note_type_alone(service, asker, col, german):
    asker.approve = False
    call(ModifyNoteTypeTool(service), note_type="Basic", action="edit_css", css="x {}")
    assert col.models.by_name("Basic")["css"] != "x {}"


def test_modify_note_type_field_actions(service, asker, col, german):
    mark_synced(col)
    call(ModifyNoteTypeTool(service), note_type="Basic", action="rename_field", field="Back", new_name="Meaning")
    assert asker.plans[0].full_sync
    assert col.get_note(german["hund"])["Meaning"] == "the dog"
    call(ModifyNoteTypeTool(service), note_type="Basic", action="reposition_field", field="Meaning", position=1)
    assert [f["name"] for f in col.models.by_name("Basic")["flds"]] == ["Meaning", "Front"]
    call(ModifyNoteTypeTool(service), note_type="Basic", action="remove_field", field="Meaning")
    assert "lost in 3 notes" in asker.plans[2].details
    assert col.get_note(german["hund"]).keys() == ["Front"]
    col.undo()
    assert col.get_note(german["hund"])["Meaning"] == "the dog"


def test_modify_note_type_remove_template(service, asker, col):
    note_id = add_note(col, "Default", "Basic (and reversed card)", Front="a", Back="b")
    call(ModifyNoteTypeTool(service), note_type="Basic (and reversed card)", action="remove_template",
         template="Card 2")
    assert "1 card made" in asker.plans[0].details
    assert len(col.get_note(note_id).cards()) == 1


def test_set_deck_options_on_a_shared_preset(service, asker, col, german):
    result = call(SetDeckOptionsTool(service), deck="German::Nouns", options={"new_cards_per_day": 33})
    assert result["changed"] == ["new_cards_per_day: 20 → 33"]
    assert "shared" in asker.plans[0].summary and "German" in asker.plans[0].summary
    assert col.decks.config_dict_for_deck_id(col.decks.id_for_name("German"))["new"]["perDay"] == 33


def test_set_deck_options_in_a_new_preset(service, col, german):
    call(SetDeckOptionsTool(service), deck="German::Nouns", options={"learning_steps_minutes": [1, 10, 60]},
         new_preset="Nouns")
    conf = col.decks.config_dict_for_deck_id(col.decks.id_for_name("German::Nouns"))
    assert conf["name"] == "Nouns" and conf["new"]["delays"] == [1, 10, 60]
    assert col.decks.config_dict_for_deck_id(col.decks.id_for_name("German"))["new"]["delays"] != [1, 10, 60]
    with pytest.raises(Exception, match="Unknown: colour"):
        SetDeckOptionsTool(service)(deck="German", options={"colour": 1})


def test_reschedule_suspend_runs_without_asking(service, asker, col, german):
    assert call(RescheduleCardsTool(service), action="suspend", query="deck:German::Nouns")["cards"] == 2
    assert asker.plans == []
    assert len(col.find_cards("is:suspended")) == 2
    call(RescheduleCardsTool(service), action="unsuspend", note_ids=[german["hund"]])
    assert len(col.find_cards("is:suspended")) == 1


def test_reschedule_set_due_and_forget_ask(service, asker, col, german):
    call(RescheduleCardsTool(service), action="set_due", note_ids=[german["hund"]], days="3")
    assert len(col.find_cards("prop:due=3")) == 1
    call(RescheduleCardsTool(service), action="forget", note_ids=[german["hund"]])
    assert len(col.find_cards("is:new")) == 4
    assert [p.kind for p in asker.plans] == ["reschedule:set_due", "reschedule:forget"]
    with pytest.raises(Exception, match="days for set_due"):
        RescheduleCardsTool(service)(action="set_due", note_ids=[german["hund"]], days="soon")


def test_modify_tags(service, asker, col, german):
    assert call(ModifyTagsTool(service), action="rename", tag="animals", new_name="zoo")["notes_changed"] == 2
    assert sorted(col.get_note(german["hund"]).tags) == ["nouns", "zoo"]
    call(ModifyTagsTool(service), action="remove", tag="nouns", note_ids=[german["hund"]])
    assert col.get_note(german["hund"]).tags == ["zoo"] and "nouns" in col.get_note(german["katze"]).tags
    with pytest.raises(Exception, match="No note has the tag"):
        ModifyTagsTool(service)(action="remove", tag="nope")


def test_update_notes_asks_above_the_limit(service, asker, col):
    ids = [add_note(col, "Default", Front=f"w{i}", Back="old") for i in range(21)]
    UpdateNotesTool(service)(notes=[{"id": i, "fields": {"Back": "new"}} for i in ids[:20]])
    assert asker.plans == []
    asker.approve = False
    result = call(UpdateNotesTool(service), notes=[{"id": i, "fields": {"Back": "newer"}} for i in ids])
    assert result["declined_by_user"] and asker.plans[0].title == "Edit 21 notes"
    assert col.get_note(ids[0])["Back"] == "new"
