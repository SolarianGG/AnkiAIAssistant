import json

import pytest
from anki import hooks
from smolagents.models import get_tool_json_schema

from anki_ai_assistant.anki_api.tools_write import (AddNotesTool, AddTagsTool, CreateDeckTool, CreateNoteTypeTool,
                                                    ExtendNoteTypeTool, StoreMediaTool, UpdateNotesTool, write_tools)

from test_web import image_bytes

TEMPLATE = {"name": "Card 1", "front": "{{Word}}", "back": "{{FrontSide}}<hr id=answer>{{Meaning}}"}


def call(tool, **kwargs):
    return json.loads(tool(**kwargs))


def mark_synced(col):
    """Pretend the collection was just synced, so a schema change needs the user's consent."""
    col.db.execute("update col set ls = scm + 1")


def test_all_write_tools_are_valid_smolagents_tools(service, tmp_path):
    tools = write_tools(service, tmp_path)
    for tool in tools:
        assert tool.risk == "write"
        assert get_tool_json_schema(tool)["function"]["name"] == tool.name


def test_create_deck(service, col):
    assert call(CreateDeckTool(service), name="German::Verbs")["created"] is True
    assert col.decks.id_for_name("German") is not None
    assert col.undo_status().undo == "AI: create deck German::Verbs"
    assert call(CreateDeckTool(service), name="German::Verbs")["created"] is False


def test_create_deck_rejects_bad_names(service):
    with pytest.raises(Exception, match="Invalid deck name"):
        CreateDeckTool(service)(name="German::")


def test_add_notes(service, col, german):
    result = call(AddNotesTool(service), deck="German::Nouns", tags=["new"], notes=[
        {"Front": "das Haus", "Back": "the house"},
        {"Front": "der Hund", "Back": "the dog"},     # already in the collection
        {"Front": "das Haus", "Back": "home"},        # repeats the first one
        {"Front": "", "Back": "nothing"},
    ])
    assert [a["index"] for a in result["added"]] == [0]
    assert [s["index"] for s in result["skipped"]] == [1, 2, 3]
    assert "duplicate" in result["skipped"][0]["reason"] and "empty" in result["skipped"][2]["reason"]

    note = col.get_note(result["added"][0]["id"])
    assert note["Back"] == "the house" and note.tags == ["new"]
    assert col.decks.name(note.cards()[0].did) == "German::Nouns"
    assert col.undo_status().undo == "AI: add 1 note to German::Nouns"
    col.undo()
    assert col.note_count() == 4


def test_add_notes_needs_an_existing_deck(service, col, german):
    """A misspelt deck name must not create a new deck: the model gets the close names instead."""
    with pytest.raises(Exception, match="Did you mean: German::Nouns"):
        AddNotesTool(service)(deck='German::"Nouns"', notes=[{"Front": "x"}])
    assert col.decks.id_for_name('German::"Nouns"') is None


def test_add_notes_allow_duplicates(service, col, german):
    result = call(AddNotesTool(service), deck="German", notes=[{"Front": "der Hund"}], allow_duplicates=True)
    assert len(result["added"]) == 1


def test_add_notes_unknown_field_changes_nothing(service, col, german):
    undo_before = col.undo_status().undo
    with pytest.raises(Exception, match="Its fields are: Front, Back"):
        AddNotesTool(service)(deck="German", notes=[{"Vorderseite": "x"}])
    assert col.undo_status().undo == undo_before


def test_add_cloze_notes(service, col):
    result = call(AddNotesTool(service), deck="Default", note_type="Cloze", notes=[
        {"Text": "{{c1::Berlin}} is the capital of {{c2::Germany}}"},
        {"Text": "no cloze here"},
    ])
    assert result["added"][0]["cards"] == 2
    assert "cloze" in result["skipped"][0]["reason"]


def test_add_notes_only_skipped_makes_no_undo_step(service, col, german):
    undo_before = col.undo_status().undo
    result = call(AddNotesTool(service), deck="German", notes=[{"Front": "der Hund", "Back": "x"}])
    assert result["added"] == [] and col.undo_status().undo == undo_before


def test_update_notes(service, col, german):
    result = call(UpdateNotesTool(service), notes=[
        {"id": german["hund"], "fields": {"Back": "the dog (m)"}},
        {"id": german["katze"], "fields": {"Back": "the cat"}},  # same content
    ])
    assert result == {"updated": [{"id": german["hund"], "fields": ["Back"]}], "unchanged": [german["katze"]]}
    assert col.get_note(german["hund"])["Back"] == "the dog (m)"
    col.undo()
    assert col.get_note(german["hund"])["Back"] == "the dog"


def test_update_notes_errors(service, german):
    with pytest.raises(Exception, match="No note with id 1"):
        UpdateNotesTool(service)(notes=[{"id": 1, "fields": {"Back": "x"}}])
    with pytest.raises(Exception, match="has no field"):
        UpdateNotesTool(service)(notes=[{"id": german["hund"], "fields": {"Rückseite": "x"}}])


def test_add_tags(service, col, german):
    result = call(AddTagsTool(service), note_ids=[german["hund"], german["laufen"]], tags=["review::week1"])
    assert result["notes_changed"] == 2
    assert "review::week1" in col.get_note(german["laufen"]).tags
    with pytest.raises(Exception, match="spaces"):
        AddTagsTool(service)(note_ids=[german["hund"]], tags=["two words"])
    with pytest.raises(Exception, match="No notes with ids"):
        AddTagsTool(service)(note_ids=[german["hund"], 42], tags=["x"])


def test_add_and_update_notes_need_media_in_the_collection(service, col, german):
    col.media.write_data("hund.jpg", b"jpeg data")
    tool = AddNotesTool(service)
    for back in ['<img src="</body>">', '<img src="&lt;/body">', '<img src="https://example.com/x.jpg">',
                 "<img src='katze.jpg'>", "[sound:katze.mp3]"]:
        with pytest.raises(Exception, match="not in the collection"):
            tool(deck="German::Nouns", notes=[{"Front": "das Pferd", "Back": back}])
    assert col.note_count() == 4

    added = call(tool, deck="German::Nouns", notes=[{"Front": "das Pferd", "Back": 'horse <IMG class=x SRC="hund.jpg">'}])
    note_id = added["added"][0]["id"]
    with pytest.raises(Exception, match="not in the collection"):
        UpdateNotesTool(service)(notes=[{"id": note_id, "fields": {"Back": "<img src=katze.jpg>"}}])
    with pytest.raises(Exception, match="cut at a quote"):
        UpdateNotesTool(service)(notes=[{"id": note_id, "fields": {"Back": "<img src="}}])
    call(UpdateNotesTool(service), notes=[{"id": note_id, "fields": {"Back": "<img src=hund.jpg>"}}])


def test_store_media_from_download_dir(service, col, tmp_path):
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    jpeg = image_bytes((300, 200), fmt="JPEG")
    (downloads / "hund.jpg").write_bytes(jpeg)
    tool = StoreMediaTool(service, downloads)

    first = call(tool, source=str(downloads / "hund.jpg"))
    # No quotes: models leave them unescaped in JSON arguments, and the field came out cut as '<img src='
    assert first == {"filename": "hund.jpg", "html": "<img src=hund.jpg>", "bytes": len(jpeg)}
    # Anki (Check Media, sync) still sees the file as used
    assert col.media.files_in_str(col.models.by_name("Basic")["id"], first["html"]) == ["hund.jpg"]
    assert col.media.have("hund.jpg")

    (downloads / "other.jpg").write_bytes(image_bytes((300, 100), fmt="JPEG"))
    second = call(tool, source=str(downloads / "other.jpg"), filename="hund.jpg")
    assert second["filename"] != "hund.jpg"  # Anki keeps the first file and renames the new one

    (downloads / "hund.mp3").write_bytes(b"ID3 mp3 data")
    audio = call(tool, source=str(downloads / "hund.mp3"))
    assert audio["html"] == "[sound:hund.mp3]"


def test_store_media_shrinks_images_and_rejects_web_pages(service, col, tmp_path):
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    (downloads / "big.png").write_bytes(image_bytes((1600, 1200)))
    (downloads / "page.mp3").write_bytes(b"<!doctype html><html>403 Forbidden</html>")
    tool = StoreMediaTool(service, downloads)

    stored = call(tool, source=str(downloads / "big.png"))
    assert stored["filename"] == "big.jpg" and col.media.have("big.jpg")
    with pytest.raises(Exception, match="web page"):
        tool(source=str(downloads / "page.mp3"))


def test_store_media_refuses_other_files(service, tmp_path):
    downloads = tmp_path / "downloads"
    downloads.mkdir()
    secret = tmp_path / "secret.jpg"
    secret.write_bytes(b"x")
    tool = StoreMediaTool(service, downloads)
    with pytest.raises(Exception, match="Only files from the download folder"):
        tool(source=str(downloads / ".." / "secret.jpg"))
    (downloads / "notes.txt").write_bytes(b"x")
    with pytest.raises(Exception, match="Unsupported file type"):
        tool(source=str(downloads / "notes.txt"))


def test_create_note_type_from_scratch(service, col):
    result = call(CreateNoteTypeTool(service), name="Vocab", fields=["Word", "Meaning"], templates=[TEMPLATE],
                  css=".card { font-size: 30px; }")
    assert result["fields"] == ["Word", "Meaning"] and result["templates"] == ["Card 1"]
    model = col.models.by_name("Vocab")
    assert model["css"] == ".card { font-size: 30px; }"

    added = call(AddNotesTool(service), deck="Default", note_type="Vocab", notes=[{"Word": "Hund", "Meaning": "dog"}])
    assert added["added"][0]["cards"] == 1


def test_create_cloze_note_type(service, col):
    call(CreateNoteTypeTool(service), name="My Cloze", kind="cloze", fields=["Text", "Extra"],
         templates=[{"name": "Cloze", "front": "{{cloze:Text}}", "back": "{{cloze:Text}}<br>{{Extra}}"}])
    assert col.models.by_name("My Cloze")["type"] == 1


def test_create_note_type_clone(service, col):
    mark_synced(col)
    result = call(CreateNoteTypeTool(service), name="Basic (mine)", clone_from="Basic")
    assert result["fields"] == ["Front", "Back"]
    assert col.models.by_name("Basic") is not None
    assert not col.schema_changed()  # a new note type syncs normally


def test_create_note_type_errors(service):
    tool = CreateNoteTypeTool(service)
    with pytest.raises(Exception, match="already exists"):
        tool(name="Basic", clone_from="Cloze")
    with pytest.raises(Exception, match="either clone_from or fields"):
        tool(name="X", clone_from="Basic", fields=["A"])
    with pytest.raises(Exception, match="field names"):
        tool(name="X", fields=["A", "A"], templates=[TEMPLATE])
    # Anki itself rejects a template whose front uses no field
    with pytest.raises(Exception):
        tool(name="X", fields=["A"], templates=[{"name": "C", "front": "static", "back": "text"}])


def test_extend_note_type(service, col, german):
    mark_synced(col)
    result = call(ExtendNoteTypeTool(service), note_type="Basic", add_fields=["Example"],
                  add_templates=[{"name": "Reverse", "front": "{{Back}}", "back": "{{FrontSide}}<hr id=answer>{{Front}}"}])
    assert result["fields"] == ["Front", "Back", "Example"] and result["templates"] == ["Card 1", "Reverse"]
    assert len(col.get_note(german["hund"]).cards()) == 2
    assert col.schema_changed()

    col.undo()
    assert [f["name"] for f in col.models.by_name("Basic")["flds"]] == ["Front", "Back"]
    assert len(col.get_note(german["hund"]).cards()) == 1


def test_extend_note_type_declined(service, col):
    mark_synced(col)
    decline = lambda proceed: False
    hooks.schema_will_change.append(decline)
    try:
        result = call(ExtendNoteTypeTool(service), note_type="Basic", add_fields=["Example"])
    finally:
        hooks.schema_will_change.remove(decline)
    assert result["changed"] is False
    assert [f["name"] for f in col.models.by_name("Basic")["flds"]] == ["Front", "Back"]
    assert not col.schema_changed()


def test_extend_note_type_errors(service):
    tool = ExtendNoteTypeTool(service)
    with pytest.raises(Exception, match="already has field"):
        tool(note_type="Basic", add_fields=["Front"])
    with pytest.raises(Exception, match="cloze note type"):
        tool(note_type="Cloze", add_templates=[TEMPLATE])
