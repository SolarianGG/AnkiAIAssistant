"""Dev CLI: runs the add-on's agent outside Anki, on a collection file opened directly.

Anki locks the collection of the open profile, so by default the CLI works on its own dev collection
(dev_collection/collection.anki2, created on first run). Point --collection at a copy of a real one to test on it.
Uses the add-on's config (with the settings saved in Anki, if any) and API keys from .env.

    uv run main.py "Create the deck 'Test German Deck' with 2 cards for common German nouns"
    uv run main.py --collection path/to/copy.anki2 "What decks do I have?"
"""

import argparse
import json
from pathlib import Path

from anki.collection import Collection
from dotenv import load_dotenv
from smolagents import LogLevel

from anki_ai_assistant.anki_api.service import AnkiService
from anki_ai_assistant.anki_api.tools_read import read_tools
from anki_ai_assistant.anki_api.tools_write import write_tools
from anki_ai_assistant.core.agent import build_agent
from anki_ai_assistant.core.config import load_config

ROOT = Path(__file__).resolve().parent
ADDON_DIR = ROOT / "anki_ai_assistant"
DOWNLOAD_DIR = ROOT / "images"
DEFAULT_COLLECTION = ROOT / "dev_collection" / "collection.anki2"

DEFAULT_TASK = """Create the Anki deck 'Test German Deck' with 2 cards for common German nouns.
Each card: Front = an image of the thing + the German word with its article (e.g. 'der Hund'), Back = the English translation."""


def user_config() -> dict:
    """The settings saved in Anki's add-on config dialog live in meta.json."""
    meta = ADDON_DIR / "meta.json"
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8")).get("config", {})
    return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--collection", type=Path, default=DEFAULT_COLLECTION, help="collection file (.anki2)")
    parser.add_argument("task", nargs="*", help="the request to the assistant")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    args.collection.parent.mkdir(parents=True, exist_ok=True)
    col = Collection(str(args.collection))
    try:
        # Everything runs in this one thread, so the service calls the collection directly
        service = AnkiService(lambda: col)
        tools = read_tools(service) + write_tools(service, DOWNLOAD_DIR)
        agent = build_agent(load_config(user_config()), tools=tools, download_dir=DOWNLOAD_DIR,
                            verbosity_level=LogLevel.INFO)
        print(agent.run(" ".join(args.task) or DEFAULT_TASK))
    finally:
        col.close()


if __name__ == "__main__":
    main()
