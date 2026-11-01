"""The chat dock: a minimal chat with the agent (history, input, Stop, New dialog)."""

import html
from pathlib import Path
from typing import Any

from aqt import gui_hooks
from aqt.main import AnkiQt
from aqt.qt import (QDockWidget, QHBoxLayout, QKeySequence, QLabel, QPlainTextEdit, QPushButton, QShortcut,
                    QTextBrowser, QVBoxLayout, QWidget, Qt)

ADDON_DIR = Path(__file__).resolve().parent.parent
# user_files survives add-on updates
DOWNLOAD_DIR = ADDON_DIR / "user_files" / "downloads"


class ChatDock(QDockWidget):
    def __init__(self, mw: AnkiQt, addon_module: str):
        super().__init__("AI Assistant", mw)
        self.mw = mw
        self.addon_module = addon_module
        self.runner = None
        self.setObjectName("AIAssistantDock")
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

        body = QWidget(self)
        layout = QVBoxLayout(body)
        self.history = QTextBrowser(body)
        self.history.setOpenExternalLinks(True)
        layout.addWidget(self.history, stretch=1)

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
        layout.addWidget(self.status)
        self.setWidget(body)

        mw.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self)
        self.hide()
        gui_hooks.profile_will_close.append(self.stop)
        self._set_busy(False)

    # Actions

    def send(self) -> None:
        task = self.input.toPlainText().strip()
        if not task or (self.runner is not None and self.runner.busy):
            return
        try:
            runner = self._get_runner()
        except ImportError as e:
            self._append_error(f"The add-on dependencies are missing ({e}). Build them with scripts/vendor.py.")
            return
        self.input.clear()
        self._append(f"<p><b>You:</b> {_to_html(task)}</p>")
        self._set_busy(True)
        runner.submit(task)

    def stop(self) -> None:
        if self.runner is not None and self.runner.busy:
            self.runner.stop()
            self.status.setText("Stopping after the current step…")

    def new_dialog(self) -> None:
        if self.runner is not None:
            if self.runner.busy:
                return
            self.runner.new_dialog()
        self.history.clear()
        self.status.setText("")

    def config_changed(self) -> None:
        self.status.setText("Settings changed: they apply in a new dialog.")

    # Runner

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
        from ..anki_api.tools_read import read_tools
        from ..anki_api.tools_write import write_tools
        from ..core.agent import build_agent
        from ..core.config import load_config

        config = load_config(self.mw.addonManager.getConfig(self.addon_module))
        service = self._build_service()
        tools = read_tools(service) + write_tools(service, DOWNLOAD_DIR)
        return build_agent(config, tools=tools, download_dir=DOWNLOAD_DIR)

    def _build_service(self):
        """One service per dialog, so a dialog backs the collection up before its first write."""
        from aqt.operations import on_op_finished

        from ..anki_api.service import AnkiService

        mw = self.mw
        return AnkiService(
            lambda: mw.col,
            run_on_main=mw.taskman.run_on_main,
            # What CollectionOp does after an operation: refresh the undo menu, the browser and the main screen
            on_changes=lambda changes: on_op_finished(mw, changes, None),
            backup=lambda col: col.create_backup(backup_folder=mw.pm.backupFolder(), force=True,
                                                 wait_for_completion=False),
        )

    def _show_event(self, event: Any) -> None:
        from ..core.runner import describe_event

        text = describe_event(event)
        if text:
            self._append(f'<p style="color: gray; margin-left: 12px;">{_to_html(text)}</p>')

    def _show_answer(self, answer: Any) -> None:
        self._append(f"<p><b>Assistant:</b> {_to_html(str(answer))}</p>")
        self._set_busy(False)

    def _show_error(self, error: Exception) -> None:
        from ..core.config import ConfigError

        message = str(error) if isinstance(error, ConfigError) else f"{type(error).__name__}: {error}"
        self._append_error(message)
        self._set_busy(False)

    def _show_stopped(self) -> None:
        self._append('<p style="color: gray;"><i>Stopped.</i></p>')
        self._set_busy(False)

    # Helpers

    def _append(self, html_text: str) -> None:
        self.history.append(html_text)
        scrollbar = self.history.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _append_error(self, message: str) -> None:
        self._append(f'<p style="color: #c0392b;"><b>Error:</b> {_to_html(message)}</p>')

    def _set_busy(self, busy: bool) -> None:
        self.send_button.setEnabled(not busy)
        self.new_button.setEnabled(not busy)
        self.stop_button.setEnabled(busy)
        self.status.setText("Working…" if busy else "")


def _to_html(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")
