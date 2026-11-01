"""Entry points: Tools > AI Assistant, the browser's context menu, the editor button, the settings dialog.

The chat dock is created on first use."""

from aqt import gui_hooks
from aqt.main import AnkiQt
from aqt.qt import QAction, QMenu
from aqt.utils import tooltip

from ..core.context import ENRICH_REQUEST, NotesContext

ENRICH_COMMAND = "ai_assistant_enrich"

_dock = None


def setup(mw: AnkiQt, addon_module: str) -> None:
    def get_dock():
        global _dock
        if _dock is None:
            from .chat_dock import ChatDock

            _dock = ChatDock(mw, addon_module)
        return _dock

    def show_dock():
        dock = get_dock()
        dock.show()
        dock.raise_()
        # The request may come from the browser or the editor window: bring the chat to the front
        mw.activateWindow()
        return dock

    def toggle_chat_dock() -> None:
        dock = get_dock()
        if dock.isVisible():
            dock.hide()
        else:
            show_dock()

    def ask_about(context: NotesContext, request: str = "") -> None:
        show_dock().set_context(context, request)

    def open_settings() -> None:
        get_dock().open_settings()

    def on_config_updated(_config: dict) -> None:
        if _dock is not None:
            _dock.config_changed()

    def add_menu_entry() -> None:
        # No shortcut: Ctrl+Shift+A already opens Anki's add-on manager
        action = QAction("AI Assistant", mw)
        action.triggered.connect(toggle_chat_dock)
        mw.form.menuTools.addAction(action)

    def on_browser_menu(browser, menu: QMenu) -> None:
        note_ids = tuple(browser.selected_notes())
        menu.addSeparator()
        action = menu.addAction("Ask AI Assistant about the selected notes")
        action.setEnabled(bool(note_ids))
        action.triggered.connect(lambda: ask_about(NotesContext(note_ids, "browser")))

    def on_editor_buttons(buttons: list[str], editor) -> None:
        def enrich(editor) -> None:
            note = getattr(editor, "note", None)
            if note is None:
                return
            if not note.id:
                tooltip("Add the note first, then ask the assistant to enrich it from the browser.",
                        parent=getattr(editor, "parentWindow", None))
                return
            # The editor saves typed text with a delay: let it reach the collection before the agent reads it
            editor.call_after_note_saved(lambda: ask_about(NotesContext((note.id,), "editor"), ENRICH_REQUEST))

        buttons.append(editor.addButton(None, ENRICH_COMMAND, enrich, tip="Enrich this note with the AI Assistant",
                                        label="AI"))

    mw.addonManager.setConfigAction(addon_module, open_settings)
    mw.addonManager.setConfigUpdatedAction(addon_module, on_config_updated)
    gui_hooks.main_window_did_init.append(add_menu_entry)
    gui_hooks.browser_will_show_context_menu.append(on_browser_menu)
    gui_hooks.editor_did_init_buttons.append(on_editor_buttons)
