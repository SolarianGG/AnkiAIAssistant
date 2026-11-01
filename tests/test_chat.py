"""The chat's logic without Qt: transcript, request context, model presets, card previews, config check."""

import json

import pytest
from smolagents.agents import ToolOutput
from smolagents.memory import ToolCall

from anki_ai_assistant.anki_api.previews import note_ids_from_output, previews_html
from anki_ai_assistant.core.config import check_config, load_config
from anki_ai_assistant.core.context import (NotesContext, manager_choices, parse_presets, task_with_context,
                                            with_manager)
from anki_ai_assistant.core.transcript import BROWSE_LINK, Step, Transcript
from conftest import add_note


def tool_events(name, arguments, observation, call_id="c1"):
    call = ToolCall(name=name, arguments=arguments, id=call_id)
    return call, ToolOutput(id=call_id, output=observation, is_final_answer=False, observation=observation,
                            tool_call=call)


# Transcript

def test_a_tool_call_and_its_output_are_one_collapsed_step():
    transcript = Transcript()
    call, output = tool_events("search_notes", {"query": "deck:German"}, json.dumps({"total": 12, "notes": []}))
    assert transcript.add_event(call)
    assert "search_notes" in transcript.render() and "…" in transcript.render()
    transcript.add_event(output)
    assert len(transcript.entries) == 1
    html_text = transcript.render()
    assert '&quot;total&quot;: 12' in html_text and "▸" in html_text and "Result:" not in html_text


def test_parallel_calls_get_their_own_outputs():
    transcript = Transcript()
    call_a, output_a = tool_events("list_decks", {}, "decks", "a")
    call_b, output_b = tool_events("list_tags", {}, "tags", "b")
    for event in (call_a, call_b, output_b, output_a):
        transcript.add_event(event)
    assert [(s.name, s.observation) for s in transcript.entries] == [("list_decks", "decks"), ("list_tags", "tags")]


def test_expanding_a_step_shows_the_full_output():
    transcript = Transcript()
    long_output = json.dumps({"notes": ["x" * 50] * 20})
    for event in tool_events("search_notes", {"query": "x"}, long_output):
        transcript.add_event(event)
    assert "Result:" not in transcript.render()
    transcript.toggle(0)
    assert "Result:" in transcript.render() and transcript.render().count("x" * 50) == 20
    transcript.toggle(0)
    assert not transcript.entries[0].expanded


def test_final_answer_calls_are_not_steps():
    transcript = Transcript()
    call, output = tool_events("final_answer", {"answer": "Done"}, "Done")
    output.is_final_answer = True
    assert not transcript.add_event(call) and not transcript.add_event(output)
    assert transcript.entries == []


def test_messages_are_escaped_and_answers_rendered_as_markdown():
    transcript = Transcript()
    transcript.add_message("user", "<b>hi</b>", context="3 selected notes")
    transcript.add_message("assistant", "Added:\n- **der Hund**\n- <script>x</script>")
    html_text = transcript.render()
    assert "&lt;b&gt;hi&lt;/b&gt;" in html_text and "[3 selected notes]" in html_text
    assert "<strong>der Hund</strong>" in html_text and "<li>" in html_text
    assert "<script>" not in html_text


def test_toggle_ignores_entries_that_are_not_steps():
    transcript = Transcript()
    transcript.add_notice("Stopped.")
    transcript.toggle(0)
    transcript.toggle(5)
    assert not isinstance(transcript.entries[0], Step)


# Context

def test_the_request_names_the_selected_notes():
    task = task_with_context("Translate these", NotesContext((11, 12), "browser"))
    assert "selected these notes in Anki's browser: note ids [11, 12]" in task
    assert task.endswith("Request: Translate these")
    assert task_with_context("Hi", None) == "Hi"
    assert NotesContext((1,), "editor").label == "the note in the editor"
    assert NotesContext((1, 2), "browser").label == "2 selected notes"


# Model presets

def config_with_presets():
    config = load_config({})
    config["roles"]["manager"] = {"provider": "ollama", "model": "qwen3.5:9b", "reasoning_effort": "none",
                                  "context_length": 32768, "max_steps": 30}
    config["manager_presets"] = [{"provider": "openrouter", "model": "z-ai/glm-5.3-flash"},
                                 {"provider": "ollama", "model": "qwen3.5:9b"}]
    return config


def test_choices_start_with_the_current_model_without_repeats():
    choices = manager_choices(config_with_presets())
    assert [(c["provider"], c["model"]) for c in choices] == [("ollama", "qwen3.5:9b"),
                                                              ("openrouter", "z-ai/glm-5.3-flash")]
    assert "max_steps" not in choices[0]


def test_switching_models_keeps_the_settings_of_both():
    config = config_with_presets()
    switched = with_manager(config, manager_choices(config)[1])
    assert switched["roles"]["manager"] == {"provider": "openrouter", "model": "z-ai/glm-5.3-flash", "max_steps": 30}
    back = with_manager(switched, manager_choices(switched)[1])
    assert back["roles"]["manager"] == config["roles"]["manager"]
    # The original config is not changed
    assert config["roles"]["manager"]["model"] == "qwen3.5:9b"


def test_presets_are_parsed_from_lines():
    old = [{"provider": "ollama", "model": "qwen3.5:9b", "context_length": 16384}]
    presets = parse_presets("openrouter / z-ai/glm-5.3-flash\n\nollama / qwen3.5:9b\n", old)
    assert presets == [{"provider": "openrouter", "model": "z-ai/glm-5.3-flash"}, old[0]]
    with pytest.raises(ValueError):
        parse_presets("just-a-model", [])


# Config check

def test_the_default_config_is_usable():
    assert check_config(load_config({})) == []


def test_config_problems_are_reported():
    config = load_config({})
    config["roles"]["manager"]["provider"] = "missing"
    config["roles"]["searcher"]["model"] = ""
    config["roles"]["validator"]["extra_body"] = "x"
    config["providers"]["ollama"]["api_base"] = "localhost:11434"
    problems = "\n".join(check_config(config))
    for text in ("'manager': choose one", "'searcher': the model is empty", "extra_body", "Provider 'ollama'"):
        assert text in problems


# Previews

def test_note_ids_are_read_from_add_and_update_results():
    added = json.dumps({"added": [{"index": 0, "id": 5, "cards": 1}], "skipped": []})
    assert note_ids_from_output("add_notes", added) == [5]
    assert note_ids_from_output("update_notes", json.dumps({"updated": [{"id": 7, "fields": ["Back"]}]})) == [7]
    assert note_ids_from_output("search_notes", added) == []
    assert note_ids_from_output("add_notes", "Error: no such deck") == []


def test_previews_show_both_sides_and_link_to_the_browser(col):
    ids = [add_note(col, "German", Front=f'der Hund {i} <img src="hund.jpg">[sound:hund.mp3]', Back="the dog")
           for i in range(4)]
    html_text = previews_html(col, ids)
    assert "der Hund 0" in html_text and "the dog" in html_text and "der Hund 3" not in html_text
    assert '<img width="160" src="hund.jpg">' in html_text
    assert "[sound:" not in html_text and "<style" not in html_text
    assert f'href="{BROWSE_LINK}{",".join(map(str, ids))}"' in html_text
    assert "1 more not shown" in html_text


def test_previews_skip_notes_that_are_gone(col):
    note_id = add_note(col, "German", Front="a", Back="b")
    col.remove_notes([note_id])
    assert "<table" not in previews_html(col, [note_id])


# Runner

def test_a_new_model_is_used_from_the_next_request_of_the_same_dialog():
    import threading

    from anki_ai_assistant.core.runner import AgentRunner

    class FakeAgent:
        managed_agents = {}

        def __init__(self):
            self.model, self.resets = "old", []

        def run(self, task, stream, reset):
            self.resets.append(reset)
            yield from ()

    agents, done = [], threading.Event()

    def factory():
        agents.append(FakeAgent())
        return agents[-1]

    runner = AgentRunner(factory, on_event=print, on_done=lambda _: done.set(), on_error=print, on_stopped=print)
    runner.change_model(lambda: "ignored")  # no dialog yet: the agent is built from the config anyway
    for model in (None, "new"):
        if model:
            runner.change_model(lambda: model)
        done.clear()
        runner.submit("hi")
        assert done.wait(5)
        runner._thread.join(5)
    assert len(agents) == 1 and agents[0].model == "new"


def test_failed_calls_are_marked():
    from smolagents.memory import ActionStep

    transcript = Transcript()
    for event in tool_events("list_decks", {"limit": 1}, "Error: the call was not run: ...", "a"):
        transcript.add_event(event)
    transcript.add_event(ToolCall(name="list_tags", arguments={}, id="b"))
    transcript.add_event(ActionStep(step_number=1, timing=None))
    assert [(s.is_error, s.failed) for s in transcript.entries] == [(True, False), (True, True)]
    assert "⚠️ Error: the call" in transcript.render() and "⚠️ not run" in transcript.render()


def test_an_answer_given_after_the_last_step_is_text():
    from anki_ai_assistant.core.runner import MAX_STEPS_NOTE, answer_text

    parts = [{"type": "text", "text": "Card added."}]
    assert answer_text(parts) == "Card added."
    assert answer_text(parts, max_steps_reached=True) == MAX_STEPS_NOTE + "Card added."
    assert answer_text("plain") == "plain"
