import json
import uuid

from smolagents import LogLevel, ToolCallingAgent
from smolagents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction, MessageRole

from anki_ai_assistant.anki_api.tools_read import read_tools
from anki_ai_assistant.anki_api.tools_write import write_tools
from anki_ai_assistant.core.checks import NO_CHANGES_MESSAGE, check_changes_made
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
    agent = ToolCallingAgent(model=model, tools=read_tools(service) + write_tools(service, tmp_path),
                             final_answer_checks=[check_changes_made], max_steps=8, verbosity_level=LogLevel.OFF)
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

    from anki_ai_assistant.core.runner import describe_event

    agent, _ = make_agent(service, tmp_path, ["Done.", ("final_answer", {"answer": "Done."})])
    agent.run("Hi")
    errors = [step for step in agent.memory.steps if isinstance(step, ActionStep) and step.error]
    assert describe_event(errors[0]).startswith("↩ The collection was not changed")
