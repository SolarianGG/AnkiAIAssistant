import json

import pytest
from smolagents.models import get_tool_json_schema

from anki_ai_assistant.anki_api.tools_read import (CollectionStatsTool, GetCardInfoTool, GetDeckOptionsTool,
                                                   GetNoteTypeTool, ListDecksTool, ListNoteTypesTool, ListTagsTool,
                                                   SearchNotesTool, read_tools)


def call(tool_class, service, **kwargs):
    return json.loads(tool_class(service)(**kwargs))


def test_all_read_tools_are_valid_smolagents_tools(service):
    tools = read_tools(service)
    assert len({t.name for t in tools}) == len(tools)
    for tool in tools:
        assert tool.risk == "read"
        # The schema the model gets must build
        assert get_tool_json_schema(tool)["function"]["name"] == tool.name


def test_list_decks(service, german):
    decks = {d["name"]: d for d in call(ListDecksTool, service)}
    assert decks["German"]["cards"] == 2
    assert decks["German"]["cards_with_subdecks"] == 4
    assert decks["German::Nouns"]["cards"] == 2
    assert decks["German::Nouns"]["due_today"]["new"] == 2


def test_deck_options(service, german):
    options = call(GetDeckOptionsTool, service, deck="German::Nouns")
    assert options["preset"] == "Default"
    assert options["new_cards_per_day"] == 20
    assert options["learning_steps_minutes"] == [1.0, 10.0]


def test_filtered_deck_options(service, col, german):
    col.decks.new_filtered("Cram")
    options = call(GetDeckOptionsTool, service, deck="Cram")
    assert options["filtered"] is True and options["searches"]


def test_unknown_deck_suggests_names(service, german):
    with pytest.raises(Exception, match="Did you mean: German::Nouns"):
        GetDeckOptionsTool(service)(deck="German::Noun")


def test_list_note_types(service, german):
    types = {m["name"]: m for m in call(ListNoteTypesTool, service)}
    assert types["Basic"]["fields"] == ["Front", "Back"]
    assert types["Basic"]["notes"] == 3
    assert types["Cloze"]["kind"] == "cloze"
    assert types["Basic (and reversed card)"]["templates"] == ["Card 1", "Card 2"]


def test_get_note_type(service):
    basic = call(GetNoteTypeTool, service, name="Basic")
    assert basic["sort_field"] == "Front"
    assert "{{Front}}" in basic["templates"][0]["front"]
    assert ".card" in basic["css"]


def test_search_notes_and_paging(service, german):
    found = call(SearchNotesTool, service, query="deck:German tag:nouns", limit=1)
    assert found["total"] == 2 and len(found["notes"]) == 1 and found["next_offset"] == 1
    note = found["notes"][0]
    assert note["id"] == german["hund"]
    assert note["fields"] == {"Front": "der Hund", "Back": "the dog"}
    assert note["decks"] == ["German::Nouns"] and set(note["tags"]) == {"nouns", "animals"}

    rest = call(SearchNotesTool, service, query="deck:German tag:nouns", limit=1, offset=1)
    assert rest["notes"][0]["id"] == german["katze"] and "next_offset" not in rest


def test_search_cuts_long_fields(service, german):
    short = call(SearchNotesTool, service, query="Front:laufen")["notes"][0]["fields"]["Back"]
    full = call(SearchNotesTool, service, query="Front:laufen", full_fields=True)["notes"][0]["fields"]["Back"]
    assert "more chars]" in short and len(full) == len("to run ") + 500


def test_bad_search_is_a_readable_error(service, german):
    with pytest.raises(Exception, match="opening bracket") as error:
        SearchNotesTool(service)(query="deck:(")
    assert "⁨" not in str(error.value)


def test_card_info(service, col, german):
    cards = call(GetCardInfoTool, service, note_ids=[german["hund"]])
    assert len(cards) == 1
    assert cards[0]["state"] == "new" and cards[0]["due"].startswith("new card #")
    assert cards[0]["deck"] == "German::Nouns"

    col.sched.suspend_cards([cards[0]["card_id"]])
    assert call(GetCardInfoTool, service, note_ids=[german["hund"]])[0]["state"] == "suspended"


def test_card_info_unknown_note(service):
    with pytest.raises(Exception, match="No note with id 123"):
        GetCardInfoTool(service)(note_ids=[123])


def test_list_tags(service, german):
    assert call(ListTagsTool, service)["tags"] == ["animals", "nouns", "verbs"]
    assert call(ListTagsTool, service, contains="ERB")["tags"] == ["verbs"]


def test_collection_stats(service, german):
    stats = call(CollectionStatsTool, service)
    assert stats["notes"] == 4 and stats["cards"] == 4
    assert stats["new_cards"] == 4 and stats["suspended_cards"] == 0
    assert stats["due_today"]["new"] == 4


def test_an_empty_search_names_the_unknown_deck(service, col, german):
    from conftest import add_note

    add_note(col, "Test German", Front="Haus", Back="house")
    add_note(col, 'Polish::01. Nouns ("Rzeczowniki")', Front="dom", Back="house")

    # Without quotes Anki reads deck 'Test' and the word 'German'
    result = call(SearchNotesTool, service, query="deck:Test German")
    assert result["total"] == 0
    assert "No deck is named 'Test'" in result["hint"] and '"deck:Test German"' in result["hint"]

    result = call(SearchNotesTool, service, query='"deck:01. rzeczowniki"')
    assert '"deck:Polish::01. Nouns ("Rzeczowniki")"' in result["hint"]

    # The deck exists and the search simply finds nothing: no hint
    assert "hint" not in call(SearchNotesTool, service, query='"deck:Test German" Katze')
    assert "hint" not in call(SearchNotesTool, service, query="deck:Germ* Katzenklo")
