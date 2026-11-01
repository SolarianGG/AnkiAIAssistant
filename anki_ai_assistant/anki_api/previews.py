"""Previews of the notes the agent added or changed, for the chat.

The chat is a QTextBrowser, which knows only simple HTML: no scripts, little CSS. So a preview is a rough
picture of the card (text and images), with a link to open the notes in Anki's browser for the real thing."""

import html
import json
import re

from anki.collection import Collection
from anki.notes import NoteId

from ..core.transcript import BROWSE_LINK, GRAY

# Tool name → key of its JSON result that lists the notes
PREVIEW_TOOLS = {"add_notes": "added", "update_notes": "updated"}
MAX_PREVIEWS = 3
IMAGE_WIDTH = 160

_DROP = re.compile(r"<(style|script)\b.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_AV_TAG = re.compile(r"\[(sound:[^\]]*|anki:[^\]]*)\]")
_IMG = re.compile(r"<img\b", re.IGNORECASE)
# The answer side usually repeats the question, followed by <hr id=answer>
_ANSWER_SPLIT = re.compile(r"<hr[^>]*id\s*=\s*[\"']?answer[^>]*>", re.IGNORECASE)


def note_ids_from_output(tool_name: str, observation: str) -> list[int]:
    """The ids of the notes a tool added or changed, read from its JSON result."""
    key = PREVIEW_TOOLS.get(tool_name)
    if key is None:
        return []
    try:
        result = json.loads(observation)
    except (TypeError, ValueError):
        return []
    if not isinstance(result, dict) or not isinstance(result.get(key), list):
        return []
    return [item["id"] for item in result[key] if isinstance(item, dict) and isinstance(item.get("id"), int)]


def _side(text: str) -> str:
    text = _AV_TAG.sub("🔊", _DROP.sub("", text))
    # Only the width: Qt keeps the aspect ratio
    return _IMG.sub(f'<img width="{IMAGE_WIDTH}"', text).strip()


def card_sides(col: Collection, note_id: int) -> tuple[str, str] | None:
    """Question and answer HTML of the note's first card, or None if the note is gone (e.g. undone)."""
    try:
        note = col.get_note(NoteId(note_id))
    except Exception:
        return None
    cards = note.cards()
    if not cards:
        return None
    card = cards[0]
    question = card.question()
    answer = card.answer()
    parts = _ANSWER_SPLIT.split(answer, maxsplit=1)
    if len(parts) == 2:
        answer = parts[1]
    return _side(question), _side(answer)


def previews_html(col: Collection, note_ids: list[int]) -> str:
    """Previews of the first notes and a link to open all of them in the browser."""
    if not note_ids:
        return ""
    rows = []
    for note_id in note_ids[:MAX_PREVIEWS]:
        sides = card_sides(col, note_id)
        if sides is None:
            continue
        question, answer = sides
        rows.append(
            '<table border="1" cellpadding="6" cellspacing="0" width="100%" '
            'style="border-color: #999; border-style: solid; margin: 4px 0 4px 12px;">'
            f'<tr><td>{question}</td></tr><tr><td>{answer}</td></tr></table>')
    more = len(note_ids) - MAX_PREVIEWS
    link_text = f"Open {'the note' if len(note_ids) == 1 else f'all {len(note_ids)} notes'} in the browser"
    if more > 0:
        link_text = f"{more} more not shown. " + link_text
    ids = ",".join(str(i) for i in note_ids)
    rows.append(f'<p style="color: {GRAY}; margin: 2px 0 6px 12px;">'
                f'<a href="{BROWSE_LINK}{ids}">{html.escape(link_text)}</a></p>')
    return "".join(rows)
