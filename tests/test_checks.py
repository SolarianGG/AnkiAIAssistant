import json
import uuid

import pytest

from smolagents import LogLevel
from smolagents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction, MessageRole

from anki_ai_assistant.anki_api.tools_read import read_tools
from anki_ai_assistant.anki_api.tools_write import write_tools
from anki_ai_assistant.core.agent import AssistantAgent
from anki_ai_assistant.core.checks import (EMPTY_ANSWER_MESSAGE, NO_CHANGES_MESSAGE, check_answer_has_text,
                                           check_changes_made)
from anki_ai_assistant.core.models import AssistantModel


class ScriptedModel(AssistantModel):
    """Answers with the given replies in turn: a tool call (name, arguments) or plain text."""

    def __init__(self, replies):
        super().__init__(model_id="test", api_base="http://localhost:1/v1", api_key="test")
        self.replies = list(replies)
        self.prompts = []

    def generate(self, messages, *args, **kwargs):
        self.prompts.append(json.dumps([m.dict() if hasattr(m, "dict") else m for m in messages], default=str))
        reply = self.replies.pop(0)
        if isinstance(reply, str):
            return ChatMessage(role=MessageRole.ASSISTANT, content=reply)
        name, arguments = reply
        return ChatMessage(role=MessageRole.ASSISTANT, content="", tool_calls=[ChatMessageToolCall(
            id=str(uuid.uuid4()), type="function",
            function=ChatMessageToolCallFunction(name=name, arguments=arguments))])


def make_agent(service, tmp_path, replies):
    model = ScriptedModel(replies)
    agent = AssistantAgent(model=model, tools=read_tools(service) + write_tools(service, tmp_path),
                             final_answer_checks=[check_answer_has_text, check_changes_made], max_steps=8, verbosity_level=LogLevel.OFF)
    return agent, model


def test_claimed_change_without_a_write_goes_back_to_the_model(col, service, tmp_path):
    agent, model = make_agent(service, tmp_path, [
        ("list_decks", {}),
        "I added a test card to the deck Default.",
        ("add_notes", {"deck": "Default", "notes": [{"Front": "test", "Back": "card"}]}),
        ("final_answer", {"answer": "Added a test card to Default."}),
    ])
    assert agent.run("Add a test card") == "Added a test card to Default."
    assert NO_CHANGES_MESSAGE in model.prompts[2]
    assert len(col.find_notes("")) == 1


def test_answer_without_changes_is_accepted_when_repeated(service, tmp_path):
    agent, model = make_agent(service, tmp_path, [
        ("list_decks", {}),
        ("final_answer", {"answer": "You have one deck: Default."}),
        ("final_answer", {"answer": "You have one deck: Default."}),
    ])
    assert agent.run("What decks do I have?") == "You have one deck: Default."
    assert not model.replies


def test_answer_after_a_write_is_accepted_at_once(service, tmp_path):
    agent, model = make_agent(service, tmp_path, [
        ("create_deck", {"name": "German"}),
        ("final_answer", {"answer": "Created the deck German."}),
    ])
    assert agent.run("Create the deck German") == "Created the deck German."
    assert not model.replies


def test_each_request_of_a_dialog_is_checked_on_its_own(service, tmp_path):
    agent, model = make_agent(service, tmp_path, [
        ("create_deck", {"name": "German"}),
        ("final_answer", {"answer": "Created the deck German."}),
        "Added a card to German.",
        ("final_answer", {"answer": "Nothing was added yet."}),
    ])
    agent.run("Create the deck German")
    # The write of the first request does not count for the second one
    assert agent.run("Now add a card to it", reset=False) == "Nothing was added yet."
    assert NO_CHANGES_MESSAGE in model.prompts[3]


def test_the_check_is_shown_in_the_chat_as_a_note_not_an_error(service, tmp_path):
    from smolagents.memory import ActionStep

    from anki_ai_assistant.core.transcript import describe_error

    agent, _ = make_agent(service, tmp_path, ["Done.", ("final_answer", {"answer": "Done."})])
    agent.run("Hi")
    errors = [step for step in agent.memory.steps if isinstance(step, ActionStep) and step.error]
    assert describe_error(errors[0]).startswith("↩ The collection was not changed")


def test_a_wrong_call_stays_next_to_its_error_with_the_valid_arguments(service, tmp_path):
    agent, model = make_agent(service, tmp_path, [
        ("list_decks", {"limit": "10"}),
        ("list_decks", {}),
        ("final_answer", {"answer": "One deck."}),
        ("final_answer", {"answer": "One deck."}),
    ])
    assert agent.run("What decks?") == "One deck."
    prompt = model.prompts[1]
    assert "'arguments': {'limit': '10'}" in prompt and "list_decks takes no arguments" in prompt


def test_a_failed_write_does_not_count_as_a_change(service, tmp_path):
    agent, model = make_agent(service, tmp_path, [
        ("add_notes", {"deck": "No such deck", "notes": [{"Front": "a", "Back": "b"}]}),
        ("final_answer", {"answer": "Added the card."}),
        ("final_answer", {"answer": "Could not add: no such deck."}),
    ])
    assert agent.run("Add a card") == "Could not add: no such deck."
    assert "No deck named" in model.prompts[1]
    assert NO_CHANGES_MESSAGE in model.prompts[2]


@pytest.mark.parametrize("answer", ["", "  ", "</tool_call>", "s</tool_call>", "<|tool_calls_end|>", "<function=x>", "."])
def test_answers_without_text_are_detected(answer):
    from anki_ai_assistant.core.models import is_empty_reply

    assert is_empty_reply(answer)


@pytest.mark.parametrize("answer", ["完了しました。", "OK", "Added 3 cards.", "Done </tool_call>", "Use <b>bold</b>"])
def test_real_answers_pass(answer):
    from anki_ai_assistant.core.models import is_empty_reply

    assert not is_empty_reply(answer)


def test_markup_answer_after_a_write_goes_back_to_the_model(service, tmp_path):
    agent, model = make_agent(service, tmp_path, [
        ("create_deck", {"name": "German"}),
        ("final_answer", {"answer": "s</tool_call>"}),
        ("final_answer", {"answer": "Created the deck German."}),
    ])
    assert agent.run("Create the deck German") == "Created the deck German."
    assert EMPTY_ANSWER_MESSAGE in model.prompts[2]
