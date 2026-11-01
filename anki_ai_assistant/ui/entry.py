"""Menu entry and the lazily created chat dock."""

from aqt import gui_hooks
from aqt.main import AnkiQt
from aqt.qt import QAction

_dock = None


def setup(mw: AnkiQt, addon_module: str) -> None:
    def toggle_chat_dock() -> None:
        global _dock
        if _dock is None:
            from .chat_dock import ChatDock

            _dock = ChatDock(mw, addon_module)
        if _dock.isVisible():
            _dock.hide()
        else:
            _dock.show()
            _dock.raise_()

    def on_config_updated(_config: dict) -> None:
        if _dock is not None:
            _dock.config_changed()

    def add_menu_entry() -> None:
        # No shortcut: Ctrl+Shift+A already opens Anki's add-on manager
        action = QAction("AI Assistant", mw)
        action.triggered.connect(toggle_chat_dock)
        mw.form.menuTools.addAction(action)

    mw.addonManager.setConfigUpdatedAction(addon_module, on_config_updated)
    gui_hooks.main_window_did_init.append(add_menu_entry)
