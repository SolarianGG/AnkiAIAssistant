"""The chat dock: the dialog with the agent, its steps, previews of the cards it made, the model picker."""

from pathlib import Path
from typing import Any

from aqt import dialogs, gui_hooks
from aqt.main import AnkiQt
from aqt.qt import (QComboBox, QDesktopServices, QDockWidget, QHBoxLayout, QKeySequence, QLabel, QPlainTextEdit,
                    QPushButton, QShortcut, QSizePolicy, QTextBrowser, QToolButton, QUrl, QVBoxLayout, QWidget,
                    Qt)

from ..core.config import load_config
from ..core.context import NotesContext, manager_choices, model_label, task_with_context, with_manager
from ..core.transcript import BROWSE_LINK, STEP_LINK, Transcript

ADDON_DIR = Path(__file__).resolve().parent.parent
# user_files survives add-on updates
DOWNLOAD_DIR = ADDON_DIR / "user_files" / "downloads"


class ChatDock(QDockWidget):
    def __init__(self, mw: AnkiQt, addon_module: str):
        super().__init__("AI Assistant", mw)
        self.mw = mw
        self.addon_module = addon_module
        self.runner = None
        self.transcript = Transcript()
        self.context: NotesContext | None = None
        self.setObjectName("AIAssistantDock")
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        body = QWidget(self)
        layout = QVBoxLayout(body)

        top = QHBoxLayout()
        self.model_picker = QComboBox(body)
        self.model_picker.setToolTip("The assistant's model. Add more in the settings.")
        self.model_picker.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.model_picker.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.model_picker.activated.connect(self._on_model_picked)
        top.addWidget(self.model_picker, stretch=1)
        settings_button = QToolButton(body)
        settings_button.setText("⚙")
        settings_button.setToolTip("Settings")
        settings_button.clicked.connect(self.open_settings)
        top.addWidget(settings_button)
        layout.addLayout(top)

        self.history = QTextBrowser(body)
        # Links are handled here: step toggles, opening notes in the browser, web links
        self.history.setOpenLinks(False)
        self.history.anchorClicked.connect(self._on_link)
        layout.addWidget(self.history, stretch=1)

        context_row = QHBoxLayout()
        self.context_label = QLabel("", body)
        self.context_label.setStyleSheet("color: gray;")
        context_row.addWidget(self.context_label, stretch=1)
        self.context_clear = QToolButton(body)
        self.context_clear.setText("✕")
        self.context_clear.setToolTip("Ask without these notes")
        self.context_clear.clicked.connect(lambda: self.set_context(None))
        context_row.addWidget(self.context_clear)
        layout.addLayout(context_row)

        self.input = QPlainTextEdit(body)
        self.input.setPlaceholderText("Ask the assistant… (Ctrl+Enter to send)")
        self.input.setMaximumHeight(100)
        layout.addWidget(self.input)
        send_shortcut = QShortcut(QKeySequence("Ctrl+Return"), self.input)
        send_shortcut.setContext(Qt.ShortcutContext.WidgetShortcut)
        send_shortcut.activated.connect(self.send)

        buttons = QHBoxLayout()
        self.send_button = QPushButton("Send", body)
        self.send_button.clicked.connect(self.send)
        self.stop_button = QPushButton("Stop", body)
        self.stop_button.clicked.connect(self.stop)
        self.new_button = QPushButton("New dialog", body)
        self.new_button.clicked.connect(self.new_dialog)
        for button in (self.send_button, self.stop_button, self.new_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.status = QLabel("", body)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.setWidget(body)

        mw.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self)
        self.hide()
        gui_hooks.profile_will_close.append(self.stop)
        self.set_context(None)
        self._fill_model_picker()
        self._set_busy(False)

    # Actions

    def send(self) -> None:
        task = self.input.toPlainText().strip()
        if not task or self._busy():
            return
        try:
            runner = self._get_runner()
        except ImportError as e:
            self._add_error(f"The add-on dependencies are missing ({e}). Build them with scripts/vendor.py.")
            return
        self.input.clear()
        context = self.context
        self.transcript.add_message("user", task, context.label if context else "")
        # The context is for this request; the ids stay in the dialog's history for the next ones
        self.set_context(None)
        self._render(to_bottom=True)
        self._set_busy(True)
        runner.submit(task_with_context(task, context))

    def stop(self) -> None:
        if self._busy():
            self.runner.stop()
            self.status.setText("Stopping after the current step…")

    def new_dialog(self) -> None:
        if self.runner is not None:
            if self.runner.busy:
                return
            self.runner.new_dialog()
        self.transcript.clear()
        self._render()
        self.status.setText("")

    def set_context(self, context: NotesContext | None, request: str = "") -> None:
        """Make the next request about these notes. request fills an empty input, e.g. "Enrich this note"."""
        self.context = context if context and context.note_ids else None
        visible = self.context is not None
        self.context_label.setText(f"About {self.context.label}" if visible else "")
        self.context_label.setVisible(visible)
        self.context_clear.setVisible(visible)
        if visible:
            if request and not self.input.toPlainText().strip():
                self.input.setPlainText(request)
            self.input.setFocus()

    def config_changed(self) -> None:
        self._fill_model_picker()
        self.status.setText("Settings changed: they apply in a new dialog.")

    def open_settings(self) -> None:
        from .settings_dialog import SettingsDialog

        def save(config: dict) -> None:
            self.mw.addonManager.writeConfig(self.addon_module, config)
            self.config_changed()

        SettingsDialog(self.mw.app.activeWindow() or self.mw, self._config(), save).exec()

    # Model picker

    def _config(self) -> dict:
        return load_config(self.mw.addonManager.getConfig(self.addon_module))

    def _fill_model_picker(self) -> None:
        self.model_picker.clear()
        try:
            choices = manager_choices(self._config())
        except (KeyError, TypeError, ValueError):
            return
        for choice in choices:
            self.model_picker.addItem(model_label(choice.get("provider", ""), choice.get("model", "")), choice)
        self.model_picker.setCurrentIndex(0)

    def _on_model_picked(self, index: int) -> None:
        choice = self.model_picker.itemData(index)
        if choice is None or index == 0:
            return
        if self._busy():
            self.model_picker.setCurrentIndex(0)
            return
        config = with_manager(self._config(), choice)
        self.mw.addonManager.writeConfig(self.addon_module, config)
        self._fill_model_picker()
        if self.runner is not None:
            from ..core.models import build_model

            # Built from the saved config when the next request starts
            self.runner.change_model(lambda: build_model(self._config(), "manager"))
        label = model_label(choice["provider"], choice["model"])
        self.transcript.add_notice(f"Model: {label}. It answers from the next request on.")
        self._render(to_bottom=True)

    # Runner

    def _busy(self) -> bool:
        return self.runner is not None and self.runner.busy

    def _get_runner(self):
        if self.runner is None:
            # Imported here: smolagents and openai take a few seconds to import
            from ..core.runner import AgentRunner

            self.runner = AgentRunner(
                self._build_agent,
                on_event=lambda event: self.mw.taskman.run_on_main(lambda: self._show_event(event)),
                on_done=lambda answer: self.mw.taskman.run_on_main(lambda: self._show_answer(answer)),
                on_error=lambda error: self.mw.taskman.run_on_main(lambda: self._show_error(error)),
                on_stopped=lambda: self.mw.taskman.run_on_main(self._show_stopped),
            )
        return self.runner

    def _build_agent(self):
        """Called in the worker thread on the first request of a dialog."""
        from ..anki_api.tools_destructive import destructive_tools
        from ..anki_api.tools_read import read_tools
        from ..anki_api.tools_write import write_tools
        from ..core.agent import build_agent

        service = self._build_service()
        tools = read_tools(service) + write_tools(service, DOWNLOAD_DIR) + destructive_tools(service)
        return build_agent(self._config(), tools=tools, download_dir=DOWNLOAD_DIR)

    def _build_service(self):
        """One service per dialog, so a dialog backs the collection up before its first write."""
        from aqt.operations import on_op_finished

        from ..anki_api.service import AnkiService
        from .confirm_dialog import confirm_change

        mw = self.mw
        return AnkiService(
            lambda: mw.col,
            run_on_main=mw.taskman.run_on_main,
            # What CollectionOp does after an operation: refresh the undo menu, the browser and the main screen
            on_changes=lambda changes: on_op_finished(mw, changes, None),
            backup=lambda col: col.create_backup(backup_folder=mw.pm.backupFolder(), force=True,
                                                 wait_for_completion=False),
            # Over the window the user works in, e.g. the browser
            confirm=lambda plan: confirm_change(mw.app.activeWindow() or mw, plan),
        )

    def _show_event(self, event: Any) -> None:
        from smolagents.agents import ToolOutput

        if not self.transcript.add_event(event):
            return
        if isinstance(event, ToolOutput) and event.tool_call is not None and self.mw.col is not None:
            from ..anki_api.previews import note_ids_from_output, previews_html

            note_ids = note_ids_from_output(event.tool_call.name, str(event.observation))
            if note_ids:
                self.transcript.add_html(previews_html(self.mw.col, note_ids))
        self._render()

    def _show_answer(self, answer: Any) -> None:
        self.transcript.add_message("assistant", "" if answer is None else str(answer))
        self._render()
        self._set_busy(False)

    def _show_error(self, error: Exception) -> None:
        from ..core.config import ConfigError

        self._add_error(str(error) if isinstance(error, ConfigError) else f"{type(error).__name__}: {error}")
        self._set_busy(False)

    def _show_stopped(self) -> None:
        self.transcript.add_notice("Stopped.")
        self._render()
        self._set_busy(False)

    # Helpers

    def _on_link(self, url: QUrl) -> None:
        link = url.toString()
        if link.startswith(STEP_LINK):
            self.transcript.toggle(int(link[len(STEP_LINK):]))
            self._render(keep_position=True)
        elif link.startswith(BROWSE_LINK):
            note_ids = link[len(BROWSE_LINK):]
            # search_for also works when the browser is already open
            dialogs.open("Browser", self.mw).search_for(f"nid:{note_ids}")
        elif url.scheme() in ("http", "https"):
            QDesktopServices.openUrl(url)

    def _render(self, to_bottom: bool = False, keep_position: bool = False) -> None:
        """Show the transcript. It follows new messages unless the user scrolled up to read."""
        scrollbar = self.history.verticalScrollBar()
        position = scrollbar.value()
        at_bottom = position >= scrollbar.maximum() - 8
        if self.mw.col is not None:
            # Card previews refer to images by their filename in the collection's media folder
            self.history.setSearchPaths([self.mw.col.media.dir()])
        self.history.setHtml(self.transcript.render())
        if keep_position or not (to_bottom or at_bottom):
            scrollbar.setValue(position)
        else:
            scrollbar.setValue(scrollbar.maximum())

    def _add_error(self, message: str) -> None:
        self.transcript.add_notice(message, error=True)
        self._render(to_bottom=True)

    def _set_busy(self, busy: bool) -> None:
        self.send_button.setEnabled(not busy)
        self.new_button.setEnabled(not busy)
        self.model_picker.setEnabled(not busy)
        self.stop_button.setEnabled(busy)
        self.status.setText("Working…" if busy else "")
