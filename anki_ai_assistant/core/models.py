"""LLMs for the agent roles. Every provider (OpenRouter, Ollama, ...) is reached through its OpenAI-compatible API."""

import json
import os
import re
import uuid
from typing import Any

from smolagents import OpenAIServerModel
from smolagents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction, MessageRole

from .config import ConfigError


class ProviderError(Exception):
    """The provider answered without a completion, e.g. with an error in the body of a 200 response."""


class ContextOverflowError(Exception):
    """The server silently cut the conversation to fit the model's context window."""


def _find_tool_call(text: str) -> tuple[str, Any] | None:
    """Find a tool call written as JSON text: the first object with a "name", nested in lists or not.

    Free models write calls in many shapes, e.g. [[{"name": "final_answer", "parameters": {...}}]]
    or {"type": "function", "function": {"name": ..., "arguments": "..."}}."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(obj.get("function"), dict):
            obj = obj["function"]
        if isinstance(obj.get("name"), str):
            arguments = obj.get("arguments", obj.get("parameters", {}))
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    pass
            return obj["name"], arguments
        start = text.find("{", start + 1)
    return None


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

    def generate(self, messages, *args, **kwargs):
        response = super().generate(messages, *args, **kwargs)
        self._check_context(len(messages), response.token_usage.input_tokens if response.token_usage else None)
        return response

    def _check_context(self, message_count: int, prompt_tokens: int | None) -> None:
        """Fail clearly when the server cut the conversation to fit the model's context window.

        Servers like Ollama drop the start of a prompt that is too long without any error: the model loses
        its instructions and tools and answers nonsense. Within a dialog the history only grows,
        so fewer prompt tokens for at least as many messages means the server cut it."""
        last = getattr(self, "_last_prompt", None)
        if prompt_tokens is None:
            return
        self._last_prompt = (message_count, prompt_tokens)
        if last is not None and message_count >= last[0] and prompt_tokens < last[1]:
            raise ContextOverflowError(
                f"The conversation no longer fits the model's context window: the server cut it "
                f"from {last[1]} to {prompt_tokens} tokens. Start a new dialog, or give the model a larger context "
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


_CALL_MARKERS = ("<tool_call", "<function", "<|tool")
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
