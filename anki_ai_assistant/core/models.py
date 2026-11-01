"""LLMs for the agent roles. Every provider (OpenRouter, Ollama, ...) is reached through its OpenAI-compatible API."""

import json
import os
import uuid
from typing import Any

from smolagents import OpenAIServerModel
from smolagents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction, MessageRole

from .config import ConfigError


class ProviderError(Exception):
    """The provider answered without a completion, e.g. with an error in the body of a 200 response."""


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

    def parse_tool_calls(self, message: ChatMessage) -> ChatMessage:
        """Called when the API returned no tool_calls: the call may be in the text instead."""
        if not message.tool_calls and isinstance(message.content, str) and message.content.strip():
            found = _find_tool_call(message.content)
            # Plain text without any call is the model talking to the user: treat it as the answer
            name, arguments = found or ("final_answer", {"answer": message.content.strip()})
            message.role = MessageRole.ASSISTANT
            message.tool_calls = [ChatMessageToolCall(
                id=str(uuid.uuid4()), type="function",
                function=ChatMessageToolCallFunction(name=name, arguments=arguments),
            )]
        return super().parse_tool_calls(message)


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

    kwargs = {}
    if role_config.get("reasoning_effort"):
        kwargs["reasoning_effort"] = role_config["reasoning_effort"]
    return AssistantModel(
        model_id=model_id,
        api_base=provider["api_base"],
        api_key=api_key,
        # A hung request once blocked a run for 14 minutes: time out and resend instead
        client_kwargs={"timeout": float(config["llm_timeout"]), "max_retries": int(config["llm_max_retries"])},
        **kwargs,
    )
