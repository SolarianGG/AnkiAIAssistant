"""Dev CLI: runs the add-on's agent outside Anki.

Uses the add-on's config (with the settings saved in Anki, if any) and API keys from .env.
Until the collection tools of stage 2 exist, it reaches Anki through Anki-Connect (tools/anki_tools.py).

    uv run main.py "Create the deck 'Test German Deck' with 2 cards for common German nouns"
"""

import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from smolagents import LogLevel

from anki_ai_assistant.core.agent import build_agent
from anki_ai_assistant.core.config import load_config
from tools.anki_tools import anki_tools

ADDON_DIR = Path(__file__).parent / "anki_ai_assistant"
DOWNLOAD_DIR = Path("images")

DEFAULT_TASK = """Create the Anki deck 'Test German Deck' with 2 cards for common German nouns.
Each card: Front = an image of the thing + the German word with its article (e.g. 'der Hund'), Back = the English translation."""


def user_config() -> dict:
    """The settings saved in Anki's add-on config dialog live in meta.json."""
    meta = ADDON_DIR / "meta.json"
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8")).get("config", {})
    return {}


def main():
    load_dotenv()
    task = " ".join(sys.argv[1:]) or DEFAULT_TASK
    agent = build_agent(load_config(user_config()), tools=anki_tools(), download_dir=DOWNLOAD_DIR,
                        verbosity_level=LogLevel.INFO)
    print(agent.run(task))


if __name__ == "__main__":
    main()
