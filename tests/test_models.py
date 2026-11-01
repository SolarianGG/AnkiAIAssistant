import pytest
from smolagents.models import ChatMessage, MessageRole

from anki_ai_assistant.core.models import AssistantModel


@pytest.fixture
def model():
    return AssistantModel(model_id="test", api_base="http://localhost:1/v1", api_key="test")


def parse(model, text):
    message = model.parse_tool_calls(ChatMessage(role=MessageRole.ASSISTANT, content=text))
    call = message.tool_calls[0].function
    return call.name, call.arguments


def test_json_call_in_text(model):
    name, arguments = parse(model, '[[{"name": "search_notes", "parameters": {"query": "deck:German"}}]]')
    assert (name, arguments) == ("search_notes", {"query": "deck:German"})


def test_plain_text_is_the_answer(model):
    assert parse(model, "Done, the card was added.") == ("final_answer", {"answer": "Done, the card was added."})


@pytest.mark.parametrize("text", ["<tool_call>add_notes<arg_key>deck</arg_key>", "<too", "<tool_c"])
def test_call_syntax_is_never_shown_as_the_answer(model, text):
    """A call the API did not parse goes back to the model as an error instead of reaching the user."""
    with pytest.raises(ValueError, match="could not be read"):
        parse(model, text)


def test_context_cut_by_the_server_is_an_error(model):
    from anki_ai_assistant.core.models import ContextOverflowError

    model._check_context(3, 3275)
    model._check_context(5, 7389)
    # More messages but fewer tokens: the server dropped the start of the conversation
    with pytest.raises(ContextOverflowError, match="7389 to 4098"):
        model._check_context(7, 4098)


def test_new_dialog_with_fewer_tokens_is_fine(model):
    model._check_context(9, 7000)
    model._check_context(2, 3000)  # fewer messages: a fresh history, not a cut one
