"""The agent's history in the API's own tool call format.

smolagents writes earlier tool calls into the prompt as assistant text:
    Calling tools:
    [{'id': 'call-1', 'type': 'function', 'function': {'name': 'download_file', 'arguments': {...}}}]
and their results as user messages "Observation: ...". Models learn from their own history, so they copied
that text instead of calling a tool, and stuffed its 'call_id', 'type', 'function', 'parameters' into
the arguments of real calls. Here the history keeps every call as an assistant message with tool_calls and
every result as a "tool" message with its tool_call_id, the format the models were trained on."""

import json
from typing import Any

from smolagents.memory import ActionStep
from smolagents.models import ChatMessage, ChatMessageToolCall, ChatMessageToolCallFunction, MessageRole

# Kept in ChatMessage.raw: smolagents' message roles have no "tool" role and no tool_call_id
NATIVE = "native_tool_history"


class NativeToolHistory:
    """Mixin for a ToolCallingAgent: keeps each call's own result and sends the history as native tool calls."""

    def process_tool_calls(self, chat_message, memory_step):
        # smolagents joins the results of all calls of a step into one text: keep each call's own result
        results = memory_step.__dict__.setdefault("call_results", {})
        for output in super().process_tool_calls(chat_message, memory_step):
            if hasattr(output, "observation") and getattr(output, "id", None) is not None:
                results[output.id] = output.observation
            yield output

    def write_memory_to_messages(self, summary_mode: bool = False) -> list[ChatMessage]:
        messages = self.memory.system_prompt.to_messages(summary_mode=summary_mode)
        for step in self.memory.steps:
            if isinstance(step, ActionStep) and step.tool_calls:
                messages.extend(_native_step_messages(step))
            else:
                messages.extend(step.to_messages(summary_mode=summary_mode))
        return messages


def _text(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(str(part.get("text", "")) for part in content if isinstance(part, dict)
                         and part.get("type") == "text")
    return "" if content is None else str(content)


def _native_step_messages(step: ActionStep) -> list[ChatMessage]:
    calls = [ChatMessageToolCall(id=str(call.id), type="function",
                                 function=ChatMessageToolCallFunction(name=call.name, arguments=call.arguments or {}))
             for call in step.tool_calls]
    results = step.__dict__.get("call_results", {})
    messages = [ChatMessage(role=MessageRole.ASSISTANT, content=_text(step.model_output).strip() or None,
                            tool_calls=calls, raw={NATIVE: "calls"})]
    for call in step.tool_calls:
        if call.id in results:
            result = results[call.id]
        elif step.error is not None:
            result = f"Error: {step.error}"
        else:
            result = "(no result)"
        messages.append(ChatMessage(role=MessageRole.TOOL_RESPONSE, content=str(result),
                                    raw={NATIVE: "result", "tool_call_id": str(call.id), "name": call.name}))
    if step.observations_images:
        messages.append(ChatMessage(role=MessageRole.USER,
                                    content=[{"type": "image", "image": image} for image in step.observations_images]))
    if step.error is not None:
        messages.append(ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": (
            f"Error: {step.error}\nNow let's retry: take care not to repeat previous errors! "
            "If you have retried several times, try a completely different approach.")}]))
    return messages


def is_native(message: Any) -> bool:
    return isinstance(getattr(message, "raw", None), dict) and NATIVE in message.raw


def native_to_dict(message: ChatMessage, with_tools: bool) -> dict[str, Any]:
    """An API message for a native history message. Without tools in the request (smolagents' last answer after
    max_steps) some providers reject tool messages: then the calls and results are written as plain prose."""
    if message.raw[NATIVE] == "calls":
        if with_tools:
            return {"role": "assistant", "content": message.content, "tool_calls": [
                {"id": call.id, "type": "function",
                 "function": {"name": call.function.name, "arguments": json.dumps(call.function.arguments,
                                                                                  ensure_ascii=False)}}
                for call in message.tool_calls]}
        used = ", ".join(f"{call.function.name} with {json.dumps(call.function.arguments, ensure_ascii=False)}"
                         for call in message.tool_calls)
        text = (message.content + "\n" if message.content else "") + f"(I used the tool {used}.)"
        return {"role": "assistant", "content": text}
    if with_tools:
        return {"role": "tool", "tool_call_id": message.raw["tool_call_id"], "content": message.content}
    return {"role": "user", "content": f"(Result of {message.raw['name']}: {message.content})"}
