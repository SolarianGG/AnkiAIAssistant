# AI Assistant for Anki Desktop

An Anki add-on with a chat that works with your collection. Ask in plain words and the assistant looks up your
decks, note types and notes, creates and edits cards, and finds pictures, pronunciation recordings and
dictionary data for them. It runs on [smolagents](https://github.com/huggingface/smolagents) with any
OpenAI-compatible model API: OpenRouter, a local [Ollama](https://ollama.com), or another provider.

Cards are made to work in AnkiDroid too: plain HTML and CSS, with media stored in the collection, so they
reach your phone through a normal AnkiWeb sync.

## What it can do

- **Read:** decks with card counts, deck options, note types with their fields, templates and CSS, note search
  (Anki's search syntax), card scheduling info, tags, collection stats.
- **Add:** decks, notes (duplicates are skipped), tags, media, new note types (from scratch or copied from
  another), fields and templates added to a note type.
- **Change (asks you first):** deleting notes and decks, renaming and moving decks, find and replace, editing more
  than 20 notes at once, changing note types, editing fields, templates and CSS, deck options, rescheduling,
  renaming and removing tags.
- **Web:** image search on Wikipedia and Wikimedia Commons, with a vision model that checks each image fits
  the word; dictionary entries from [Kaikki.org](https://kaikki.org) (Wiktionary data: gender, plural, forms,
  IPA, meanings, recordings); example sentences from [Tatoeba](https://tatoeba.org); reading a web page as text.

Example requests:

- "Add 10 cards with common German kitchen words to the deck German::A1, with a picture and the article."
- "Find notes in Spanish verbs without an example sentence and add one to each."
- "Rename the tag `todo` to `review` and suspend the cards of the deck Old stuff."
- "Make a note type like Basic with an extra Audio field on the back."

## Safety

- Reading and adding happen right away. Deleting, bulk edits and changes to note types, deck options or scheduling
  first show a dialog with the exact changes (field and template diffs, lists of what goes away). Decline with a
  comment, and the assistant takes it into account. "Don't ask again for this kind of action" lasts until you
  start a new dialog.
- Each action of the assistant is one step in **Edit > Undo** ("AI: …").
- Before its first change in a dialog the assistant makes a collection backup.
- Changes that force a full sync (adding or removing fields and templates) carry a warning.

## Requirements

- Anki Desktop **26.8 or newer** on Windows (Anki's Python 3.13). The add-on bundles its Python dependencies in
  `vendor/`, built for that Python and platform.
- An API key for a model provider, or a local Ollama.
  - The **assistant** (manager) needs a model that handles tool calling well; a cloud model is recommended.
    Small local models (around 9B) work for simple requests but make mistakes with 25 tools.
  - The **image checker** (validator) needs a model with image input, e.g. `qwen3.5:9b` in Ollama.

## Installing

1. Download or build `anki_ai_assistant-<version>.ankiaddon` (see below).
2. Double-click it, or use **Tools > Add-ons > Install from file…**, and restart Anki.
3. Open **Tools > Add-ons > AI Assistant > Config** (or the ⚙ button of the chat) and set:
   - **Providers:** the API base (ending with `/v1`) and the key of each provider, e.g. OpenRouter
     `https://openrouter.ai/api/v1`, Ollama `http://localhost:11434/v1`;
   - **Models:** the provider and the model of each role.
4. Open the chat with **Tools > AI Assistant**.

All settings are described in [`anki_ai_assistant/config.md`](anki_ai_assistant/config.md), which Anki also
shows next to the config. API keys are stored in plain text in Anki's add-on folder.

Other ways in:

- **Browser:** select notes, right-click, then **Ask AI Assistant about the selected notes**.
- **Editor:** the **AI** button fills empty fields of the current note with a picture, a recording and
  dictionary data.

## Troubleshooting

Errors in the chat say what to check: a missing or rejected key, Ollama not running, an unknown model, a rate or
credit limit, a dialog that no longer fits the model's context. **Details in the log** opens the add-on's log:
`user_files/logs/assistant.log` in the add-on folder (`%APPDATA%\Anki2\addons21\anki_ai_assistant`). The log has
every request, each step with its tool calls and results, and the full error.

On OpenRouter a free or cheap host sometimes breaks tool calls. Pick the hosts with the role's `extra_body`,
e.g. `{"provider": {"order": ["fireworks"]}}`.

## Building from source

The project uses [uv](https://docs.astral.sh/uv/).

```
uv sync
uv run scripts/build_addon.py               # vendor/ for Python 3.13 + dist/anki_ai_assistant-<version>.ankiaddon
uv run scripts/build_addon.py --python 3.14 # for an Anki with another Python (Help > About)
```

`scripts/build_addon.py` installs the locked runtime dependencies (`uv.lock`) into `anki_ai_assistant/vendor` as
wheels for Anki's Python and Windows, then zips the add-on folder without caches, your settings (`meta.json`) and
`user_files`. After a large Anki update that changes its Python version, rebuild: `pillow`, `pydantic-core` and
`jiter` are native.

## Development

- `scripts/dev_link.ps1` links the source folder into Anki's add-ons folder (a directory junction), so Anki runs the
  add-on from the repository. Run `uv run scripts/vendor.py` once to fill `vendor/`. Restart Anki after edits.
- `uv run main.py "<request>"` runs the same agent outside Anki on a separate collection
  (`dev_collection/collection.anki2`, or `--collection path/to/copy.anki2`). Anki locks the collection of the open
  profile, so point it at a copy. It uses the add-on's config and the keys from `.env`
  (`OPENROUTER_API_KEY=...`). Confirmations are asked in the console.
- `uv run pytest` runs the tests on temporary collections, without Anki's GUI and without network access.

Layout:

```
anki_ai_assistant/
  __init__.py        menu entry; everything else is imported on first use
  core/              agent, models, prompts, background runner, config, chat transcript, errors, log
  anki_api/          collection service (main thread, undo, backup), tools, confirmations, card previews
  web/               HTTP with a disk cache, image search, dictionary, media, page reading
  ui/                chat dock, confirmation dialog, settings dialog, browser and editor entry points
scripts/             vendor.py, build_addon.py, dev_link.ps1
tests/
main.py              dev CLI
```
