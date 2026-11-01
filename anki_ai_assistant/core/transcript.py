"""The chat transcript: messages and agent steps, rendered as HTML for the chat dock.

Kept free of Qt, so it is tested without a GUI, and imports smolagents only when events come:
the chat must open even when the add-on's dependencies are missing, to say so. A tool call and its output form one step: a line
"🔧 search_notes(...) → 12 notes" that expands into the full arguments and output on a click."""

import html
import json
from dataclasses import dataclass, field
from typing import Any

SUMMARY_ARGS_CHARS = 120
SUMMARY_OUTPUT_CHARS = 160
DETAILS_CHARS = 4000

# Links inside the transcript; the dock handles them
STEP_LINK = "step:"
BROWSE_LINK = "browse:"

# What AssistantAgent puts before the result of a failed call (core/agent.py)
TOOL_ERROR_PREFIX = "Error: "

GRAY = "#808080"
RED = "#c0392b"


@dataclass
class Message:
    role: str  # "user" or "assistant"
    text: str
    context: str = ""  # what the user's message was about, e.g. "3 selected notes"


@dataclass
class Step:
    call_id: str
    name: str
    arguments: Any
    observation: str | None = None
    expanded: bool = False
    # The agent's step failed before this call returned, e.g. on a reply it could not parse
    failed: bool = False

    @property
    def is_error(self) -> bool:
        return self.failed or (self.observation or "").startswith(TOOL_ERROR_PREFIX)


@dataclass
class Notice:
    text: str
    error: bool = False


@dataclass
class Html:
    """Ready HTML, e.g. card previews."""
    html: str


@dataclass
class Transcript:
    entries: list = field(default_factory=list)

    def clear(self) -> None:
        self.entries.clear()

    def add_message(self, role: str, text: str, context: str = "") -> None:
        self.entries.append(Message(role, text, context))

    def add_notice(self, text: str, error: bool = False) -> None:
        self.entries.append(Notice(text, error))

    def add_html(self, html_text: str) -> None:
        self.entries.append(Html(html_text))

    def add_event(self, event: Any) -> bool:
        """Add an agent stream event. Returns whether the transcript changed."""
        from smolagents.agents import ToolOutput
        from smolagents.memory import ToolCall

        if isinstance(event, ToolCall):
            if event.name == "final_answer":
                return False
            self.entries.append(Step(event.id, event.name, event.arguments))
            return True
        if isinstance(event, ToolOutput):
            if event.is_final_answer:
                return False
            step = self._step(event.tool_call.id if event.tool_call else None)
            if step is None:
                name = event.tool_call.name if event.tool_call else "tool"
                step = Step(event.id, name, event.tool_call.arguments if event.tool_call else {})
                self.entries.append(step)
            step.observation = str(event.observation)
            return True
        from smolagents.memory import ActionStep

        if isinstance(event, ActionStep):
            # Calls of this step that never returned are not running any more
            for entry in self.entries:
                if isinstance(entry, Step) and entry.observation is None and not entry.failed:
                    entry.failed = True
        text = describe_error(event)
        if text:
            self.entries.append(Notice(text))
            return True
        return False

    def toggle(self, index: int) -> None:
        if 0 <= index < len(self.entries) and isinstance(self.entries[index], Step):
            self.entries[index].expanded = not self.entries[index].expanded

    def render(self) -> str:
        return "".join(_render(index, entry) for index, entry in enumerate(self.entries))

    def _step(self, call_id: str | None) -> Step | None:
        for entry in reversed(self.entries):
            if isinstance(entry, Step) and entry.call_id == call_id:
                return entry
        return None


def describe_error(event: Any) -> str | None:
    """A note about a failed step, or None for events that are not one."""
    from smolagents.memory import ActionStep

    from .checks import EMPTY_ANSWER_MESSAGE, NO_CHANGES_MESSAGE

    if isinstance(event, ActionStep) and event.error is not None:
        if EMPTY_ANSWER_MESSAGE in str(event.error):
            return "↩ The answer had no text for you: it went back to the model."
        if NO_CHANGES_MESSAGE in str(event.error):
            return "↩ The collection was not changed: the answer went back to the model to check it."
        return f"⚠️ {preview(str(event.error), 300)}"
    return None


def preview(value: Any, limit: int) -> str:
    """value on one line, cut to limit characters."""
    if not isinstance(value, str):
        value = _json(value, indent=None)
    value = " ".join(value.split())
    return value if len(value) <= limit else value[:limit] + "…"


def _json(value: Any, indent: int | None) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, indent=indent)
    except (TypeError, ValueError):
        return str(value)


def _cut(text: str, limit: int = DETAILS_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + f"\n… ({len(text) - limit} more characters)"


def _pretty(text: str) -> str:
    """Tool outputs are mostly JSON: indent it to be readable."""
    try:
        return _json(json.loads(text), indent=2)
    except (TypeError, ValueError):
        return text


def to_html(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def markdown_to_html(text: str) -> str:
    """Models answer in Markdown (lists, bold). Raw HTML in the answer is shown as text, not rendered."""
    try:
        from markdown_it import MarkdownIt
    except ImportError:
        return to_html(text)
    return MarkdownIt("commonmark", {"html": False, "breaks": True}).enable("table").render(text)


def _render(index: int, entry: Any) -> str:
    if isinstance(entry, Message):
        if entry.role == "user":
            context = (f' <span style="color: {GRAY};">[{html.escape(entry.context)}]</span>'
                       if entry.context else "")
            return f"<p><b>You:</b>{context} {to_html(entry.text)}</p>"
        return f"<div><b>Assistant:</b>{markdown_to_html(entry.text)}</div>"
    if isinstance(entry, Notice):
        if entry.error:
            return f'<p style="color: {RED};"><b>Error:</b> {to_html(entry.text)}</p>'
        return f'<p style="color: {GRAY}; margin-left: 12px;">{to_html(entry.text)}</p>'
    if isinstance(entry, Html):
        return entry.html
    if isinstance(entry, Step):
        return _render_step(index, entry)
    return ""


def _render_step(index: int, step: Step) -> str:
    arrow = "▾" if step.expanded else "▸"
    line = f"🔧 {step.name}({preview(step.arguments, SUMMARY_ARGS_CHARS)})"
    if step.observation is None:
        line += " ⚠️ not run" if step.failed else " …"
    elif not step.expanded:
        line += f" {'⚠️' if step.is_error else '→'} {preview(step.observation, SUMMARY_OUTPUT_CHARS)}"
    out = (f'<p style="color: {GRAY}; margin: 2px 0 2px 12px;">'
           f'<a href="{STEP_LINK}{index}" style="text-decoration: none;">{arrow}</a> {html.escape(line)}</p>')
    if step.expanded:
        details = "Arguments:\n" + _cut(_json(step.arguments, indent=2))
        if step.observation is not None:
            details += "\n\nResult:\n" + _cut(_pretty(step.observation))
        out += (f'<pre style="color: {GRAY}; margin: 0 0 4px 28px; white-space: pre-wrap;">'
                f"{html.escape(details)}</pre>")
    return out
