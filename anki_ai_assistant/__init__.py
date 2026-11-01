"""AI Assistant add-on for Anki Desktop.

Anki imports this module at startup, so it only registers hooks and the menu entry.
Everything heavy (the UI, later smolagents and openai) is imported on first use.
"""

import os
import sys

from aqt import gui_hooks, mw
from aqt.qt import QAction

ADDON_DIR = os.path.dirname(__file__)
VENDOR_DIR = os.path.join(ADDON_DIR, "vendor")

# Bundled dependencies take priority over whatever other add-ons put on sys.path
if os.path.isdir(VENDOR_DIR) and VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

_dock = None


def toggle_chat_dock() -> None:
    global _dock
    if _dock is None:
        from .ui.chat_dock import create_chat_dock

        _dock = create_chat_dock(mw)
    if _dock.isVisible():
        _dock.hide()
    else:
        _dock.show()
        _dock.raise_()


def _setup_menu() -> None:
    # No shortcut: Ctrl+Shift+A already opens Anki's add-on manager
    action = QAction("AI Assistant", mw)
    action.triggered.connect(toggle_chat_dock)
    mw.form.menuTools.addAction(action)


gui_hooks.main_window_did_init.append(_setup_menu)
