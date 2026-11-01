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


def model_label(provider: str, model: str) -> str:
    return f"{provider} / {model}"


# Settings of the role that stay with the role when its model changes; the rest belongs to the model
ROLE_ONLY_FIELDS = ("max_steps",)


def _key(entry: dict) -> tuple[str, str]:
    return entry.get("provider", ""), entry.get("model", "")


def manager_choices(config: dict) -> list[dict]:
    """The models the chat offers for the manager: the current one first, then the presets, no repeats.

    A preset is a role config without max_steps: {"provider", "model"} and optionally reasoning_effort,
    context_length, extra_body."""
    current = {k: v for k, v in config["roles"]["manager"].items() if k not in ROLE_ONLY_FIELDS}
    choices = [current]
    for preset in config.get("manager_presets") or []:
        if isinstance(preset, dict) and preset.get("provider") and preset.get("model") \
                and all(_key(c) != _key(preset) for c in choices):
            choices.append(dict(preset))
    return choices


def with_manager(config: dict, choice: dict) -> dict:
    """The config with the chosen manager model.

    The previous model goes into the presets with its settings (reasoning_effort, context_length, ...),
    so switching back to it restores them."""
    manager = config["roles"]["manager"]
    if _key(manager) == _key(choice):
        return config
    previous = {k: v for k, v in manager.items() if k not in ROLE_ONLY_FIELDS}
    presets = [dict(p) for p in config.get("manager_presets") or [] if isinstance(p, dict)]
    for index, preset in enumerate(presets):
        if _key(preset) == _key(previous):
            presets[index] = previous
            break
    else:
        presets.insert(0, previous)
    new_manager = {k: v for k, v in manager.items() if k in ROLE_ONLY_FIELDS}
    new_manager.update({k: v for k, v in choice.items() if k not in ROLE_ONLY_FIELDS})
    return {**config, "roles": {**config["roles"], "manager": new_manager}, "manager_presets": presets}


def parse_presets(text: str, old_presets: list[dict]) -> list[dict]:
    """Lines "provider / model" → presets. A preset already in the config keeps its other settings."""
    old = {(p.get("provider"), p.get("model")): p for p in old_presets if isinstance(p, dict)}
    presets = []
    for line in text.splitlines():
        if not line.strip():
            continue
        provider, sep, model = line.partition("/")
        if not sep or not provider.strip() or not model.strip():
            raise ValueError(f"Chat models: write each line as 'provider / model', not '{line.strip()}'.")
        key = (provider.strip(), model.strip())
        presets.append(dict(old.get(key, {"provider": key[0], "model": key[1]})))
    return presets
