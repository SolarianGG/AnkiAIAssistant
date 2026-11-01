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


def history(*texts):
    return [ChatMessage(role=MessageRole.USER, content=text) for text in texts]


def test_context_cut_by_the_server_is_an_error(model):
    from anki_ai_assistant.core.models import ContextOverflowError

    model._check_context(history("system", "task"), 3275)
    model._check_context(history("system", "task", "step 1"), 7389)
    # The same conversation and more, but fewer tokens: the server dropped its start
    with pytest.raises(ContextOverflowError, match="7389 to 4098"):
        model._check_context(history("system", "task", "step 1", "step 2"), 4098)


def test_a_request_without_tools_is_not_compared_with_steps(model):
    """After the last step smolagents asks for the final answer without the tools: no schemas, fewer tokens."""
    model._check_context(history("system", "task", "step 1"), 16634, ("add_notes", "search_notes"))
    model._check_context(history("system", "task", "step 1", "step 2"), 10939, ())
    model._check_context(history("system", "task", "step 1", "step 2", "step 3"), 17000, ("add_notes", "search_notes"))


def test_unrelated_prompts_are_not_compared(model):
    """One model serves many runs (image_searcher for each word, some in parallel threads): a shorter
    unrelated prompt is not a cut one, even with as many messages."""
    model._check_context(history("system", "find 'maison house french'", "step 1"), 2638)
    model._check_context(history("system", "find 'chat (cat)'", "step 1"), 2636)
    model._check_context(history("system"), 1000)
    model._check_context(history("system", "find 'chat (cat)'", "step 1", "step 2"), 3000)


def test_history_format_call_in_text(model):
    """Models copy smolagents' history format, a Python literal instead of JSON."""
    text = ("Calling tools:\n[{'id': 'call-1', 'type': 'function', 'function': "
            "{'name': 'web_image_search', 'arguments': {'query': 'house', 'limit': 5}}}]")
    assert parse(model, text) == ("web_image_search", {"query": "house", "limit": 5})
    with pytest.raises(ValueError, match="could not be read"):
        parse(model, "Calling tools:\n[{'id': 'call-1', 'function': {'name': 'web_im")
