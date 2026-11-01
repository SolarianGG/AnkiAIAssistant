"""Chat dock of the assistant. For now an empty placeholder (stage 0)."""

from aqt.main import AnkiQt
from aqt.qt import QDockWidget, QLabel, QTextEdit, QVBoxLayout, QWidget, Qt


def create_chat_dock(mw: AnkiQt) -> QDockWidget:
    dock = QDockWidget("AI Assistant", mw)
    dock.setObjectName("AIAssistantDock")
    dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

    body = QWidget(dock)
    layout = QVBoxLayout(body)
    history = QTextEdit(body)
    history.setReadOnly(True)
    history.setPlaceholderText("Chat with the assistant will appear here.")
    layout.addWidget(history)
    layout.addWidget(QLabel("The agent is not connected yet.", body))
    dock.setWidget(body)

    mw.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
    dock.hide()
    return dock
