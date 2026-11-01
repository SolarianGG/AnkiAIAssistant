"""What the user's request is about, when it comes from Anki's browser or editor rather than the chat alone."""

from dataclasses import dataclass

MAX_CONTEXT_IDS = 200


@dataclass(frozen=True)
class NotesContext:
    note_ids: tuple[int, ...]
    source: str  # "browser" or "editor"

    @property
    def label(self) -> str:
        """Shown in the chat next to the request."""
        count = len(self.note_ids)
        if self.source == "editor":
            return "the note in the editor"
        return f"{count} selected note" + ("" if count == 1 else "s")


ENRICH_REQUEST = ("Enrich this note: fill its empty fields (e.g. a translation, grammar, an example sentence) "
                  "with dictionary_lookup, and add a picture and a pronunciation recording if the note type has fields "
                  "for them. Do not change fields that are filled.")


def task_with_context(task: str, context: NotesContext | None) -> str:
    """The request as the model gets it: the note ids first, so "these notes" can be resolved."""
    if context is None or not context.note_ids:
        return task
    ids = list(context.note_ids[:MAX_CONTEXT_IDS])
    if context.source == "editor":
        where = "The user is editing this note in Anki's editor"
    else:
        where = "The user selected these notes in Anki's browser"
    text = f"{where}: note ids {ids}."
    if len(context.note_ids) > MAX_CONTEXT_IDS:
        text += f" Only the first {MAX_CONTEXT_IDS} of {len(context.note_ids)} are listed."
    text += " Look them up with search_notes (e.g. query \"nid:1,2\") before acting on them."
    return f"{text}\n\nRequest: {task}"
