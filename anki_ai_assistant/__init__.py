"""AI Assistant add-on for Anki Desktop.

Anki imports this module at startup, so it only registers the menu entry.
Everything heavy (smolagents, openai, the chat UI) is imported on first use.
Outside Anki (the dev CLI, tests) aqt.mw is None and nothing is registered.
"""

import os
import sys

try:
    from aqt import mw
except ImportError:
    mw = None

ADDON_DIR = os.path.dirname(__file__)
VENDOR_DIR = os.path.join(ADDON_DIR, "vendor")

if mw is not None:
    # Appended, not prepended: for packages Anki itself ships (requests, jinja2, ...) Anki's versions win,
    # so the add-on cannot break Anki by shadowing them
    if os.path.isdir(VENDOR_DIR) and VENDOR_DIR not in sys.path:
        sys.path.append(VENDOR_DIR)

    from .ui.entry import setup

    setup(mw, __name__)
