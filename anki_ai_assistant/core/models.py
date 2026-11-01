"""LLMs for the agent roles. Every provider (OpenRouter, Ollama, ...) is reached through its OpenAI-compatible API."""

import ast
import json
import os
import re
import threading
import uuid
from collections import deque
from typing import Any

from smolagents import OpenAIServerModel
from smolagents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction, MessageRole

from .config import ConfigError
from .history import is_native, native_to_dict


class ProviderError(Exception):
    """The provider answered without a completion, e.g. with an error in the body of a 200 response."""


class ContextOverflowError(Exception):
    """The server silently cut the conversation to fit the model's context window."""


def _call_in(obj: Any) -> tuple[str, Any] | None:
    """The first object with a "name" in obj, nested in lists or not, as (name, arguments)."""
    if isinstance(obj, list):
        for item in obj:
            found = _call_in(item)
            if found:
                return found
        return None
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("function"), dict):
        obj = obj["function"]
    if not isinstance(obj.get("name"), str):
        return None
    arguments = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass
    return obj["name"], arguments


# How smolagents writes earlier calls into the history; models copy it, with Python quotes instead of JSON
HISTORY_CALL_MARKER = "Calling tools:"


def _find_tool_call(text: str) -> tuple[str, Any] | None:
    """Find a tool call written as text: the first object with a "name", nested in lists or not.

    Free models write calls in many shapes, e.g. [[{"name": "final_answer", "parameters": {...}}]],
    {"type": "function", "function": {"name": ..., "arguments": "..."}}, or a copy of the history format
    "Calling tools: [{'id': ..., 'function': {...}}]" as a Python literal."""
    if HISTORY_CALL_MARKER in text:
        literal = text.split(HISTORY_CALL_MARKER, 1)[1].strip()
        try:
            found = _call_in(ast.literal_eval(literal))
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            found = None
        if found:
            return found
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        found = _call_in(obj)
        if found:
            return found
        start = text.find("{", start + 1)
    return None


def _message_key(message: Any) -> str:
    """A message's role and content, to tell whether one prompt continues another."""
    if isinstance(message, dict):
        return f"{message.get('role')}:{message.get('content')}:{message.get('tool_calls')}"
    return f"{getattr(message, 'role', None)}:{getattr(message, 'content', None)}:{getattr(message, 'tool_calls', None)}"


class AssistantModel(OpenAIServerModel):
    """OpenAIServerModel that copes with the quirks of free and local models."""

    def create_client(self):
        client = super().create_client()
        create = client.chat.completions.create

        def checked_create(**kwargs):
            response = create(**kwargs)
            # OpenRouter reports some provider errors in a 200 response without choices;
            # smolagents then fails with "'NoneType' object is not subscriptable", hiding the reason.
            # A message containing 429 makes smolagents retry it as a rate limit
            if not kwargs.get("stream") and not getattr(response, "choices", None):
                error = (getattr(response, "model_extra", None) or {}).get("error") or response
                raise ProviderError(f"The model provider returned no answer: {error}")
            return response

        client.chat.completions.create = checked_create
        return client

    def _prepare_completion_kwargs(self, messages, *args, **kwargs):
        """Messages of the native tool history (core/history.py) go to the API as they are; smolagents' cleaning
        knows no "tool" role and would merge them. The other messages are cleaned by smolagents in runs."""
        if not any(is_native(message) for message in messages):
            return super()._prepare_completion_kwargs(messages, *args, **kwargs)
        with_tools = bool(kwargs.get("tools_to_call_from"))
        api_messages, plain = [], []

        def flush():
            if plain:
                cleaned = super(AssistantModel, self)._prepare_completion_kwargs(list(plain), *args, **kwargs)
                api_messages.extend(cleaned["messages"])
                plain.clear()

        for message in messages:
            if is_native(message):
                flush()
                api_messages.append(native_to_dict(message, with_tools))
            else:
                plain.append(message)
        flush()
        completion_kwargs = super()._prepare_completion_kwargs([], *args, **kwargs)
        completion_kwargs["messages"] = api_messages
        return completion_kwargs

    def generate(self, messages, *args, **kwargs):
        response = super().generate(messages, *args, **kwargs)
        tools = tuple(sorted(t.name for t in kwargs.get("tools_to_call_from") or []))
        self._check_context(messages, response.token_usage.input_tokens if response.token_usage else None, tools)
        return response

    def _check_context(self, messages: list, prompt_tokens: int | None, tools: tuple = ()) -> None:
        """Fail clearly when the server cut the conversation to fit the model's context window.

        Servers like Ollama drop the start of a prompt that is too long without any error: the model loses
        its instructions and tools and answers nonsense. A prompt that continues an earlier one (the same
        messages and then more) cannot have fewer tokens, unless the server cut it.

        Only continuations are compared: one model serves several agent runs, some of them at the same time
        (the manager calls image_searcher for several words in one step), and their prompts are unrelated.
        Only requests with the same tools are compared too: their schemas are part of the prompt. smolagents asks
        for the final answer after the last step without tools, and that prompt is ~5k tokens shorter."""
        if prompt_tokens is None:
            return
        keys = tuple(_message_key(message) for message in messages)
        state = self.__dict__.setdefault("_context_state", {"lock": threading.Lock(), "prompts": deque(maxlen=32)})
        with state["lock"]:
            cut_from = None
            for old_tools, old_keys, old_tokens in state["prompts"]:
                if (old_tools == tools and len(old_keys) < len(keys) and keys[:len(old_keys)] == old_keys
                        and prompt_tokens < old_tokens):
                    cut_from = max(cut_from or 0, old_tokens)
            state["prompts"].append((tools, keys, prompt_tokens))
        if cut_from is not None:
            raise ContextOverflowError(
                f"The conversation no longer fits the model's context window: the server cut it "
                f"from {cut_from} to {prompt_tokens} tokens. Start a new dialog, or give the model a larger context "
                f"(for Ollama set context_length of the role in the add-on config, e.g. 32768).")

    def parse_tool_calls(self, message: ChatMessage) -> ChatMessage:
        """Called when the API returned no tool_calls: the call may be in the text instead."""
        if not message.tool_calls and isinstance(message.content, str) and message.content.strip():
            text = message.content.strip()
            found = _find_tool_call(text)
            if found is None and _looks_like_tool_call(text):
                # A call we cannot read must not reach the user as the answer: the parse error goes back
                # to the model, which then writes the call again
                raise ValueError("The tool call could not be read. Call the tool again with valid arguments.")
            # Plain text without any call is the model talking to the user: treat it as the answer
            name, arguments = found or ("final_answer", {"answer": text})
            message.role = MessageRole.ASSISTANT
            message.tool_calls = [ChatMessageToolCall(
                id=str(uuid.uuid4()), type="function",
                function=ChatMessageToolCallFunction(name=name, arguments=arguments),
            )]
        return super().parse_tool_calls(message)


_CALL_MARKERS = ("<tool_call", "<function", "<|tool", HISTORY_CALL_MARKER)
# Tags of call syntax, opening or closing: "</tool_call>", "<|tool_calls_end|>", "<function=list_decks>"
_CALL_TAG = re.compile(r"</?\s*(?:tool_calls?|function)\b[^>]*>?|<\|[^>]*?\|>")
# A word of at least two letters or digits, in any script
_WORD = re.compile(r"[^\W_]{2,}")


def is_empty_reply(text: str) -> bool:
    """Whether a reply has nothing for the user: empty, or only bits of call syntax like "s</tool_call>"."""
    return _WORD.search(_CALL_TAG.sub(" ", text)) is None


def _looks_like_tool_call(text: str) -> bool:
    """Call syntax the API did not turn into tool_calls. Some providers also cut the reply right there ('<too')."""
    return any(marker in text or marker.startswith(text) for marker in _CALL_MARKERS)


def build_model(config: dict[str, Any], role: str) -> AssistantModel:
    role_config = config["roles"].get(role)
    if not role_config:
        raise ConfigError(f"No settings for the '{role}' role in the add-on config.")
    provider_name = role_config.get("provider", "")
    provider = config["providers"].get(provider_name)
    if provider is None:
        raise ConfigError(f"Role '{role}' uses the provider '{provider_name}', which is not in 'providers'.")
    model_id = role_config.get("model")
    if not model_id:
        raise ConfigError(f"No model is set for the '{role}' role.")
    api_key = provider.get("api_key") or os.environ.get(f"{provider_name.upper()}_API_KEY")
    if not api_key:
        raise ConfigError(f"No API key for the provider '{provider_name}'. Set it in the add-on config.")

    context_length = int(role_config.get("context_length") or 0)
    if context_length:
        from .ollama import model_with_context

        model_id = model_with_context(provider["api_base"], model_id, context_length)

    kwargs = {}
    if role_config.get("reasoning_effort"):
        kwargs["reasoning_effort"] = role_config["reasoning_effort"]
    # Provider-specific request fields, e.g. OpenRouter's {"provider": {"order": ["z-ai"]}} to pick who serves the model
    if role_config.get("extra_body"):
        kwargs["extra_body"] = role_config["extra_body"]
    return AssistantModel(
        model_id=model_id,
        api_base=provider["api_base"],
        api_key=api_key,
        # A hung request once blocked a run for 14 minutes: time out and resend instead
        client_kwargs={"timeout": float(config["llm_timeout"]), "max_retries": int(config["llm_max_retries"])},
        **kwargs,
    )
