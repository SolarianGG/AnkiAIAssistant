"""The dialog that shows a destructive change before the agent makes it."""

import html

from aqt.qt import (QCheckBox, QDialog, QDialogButtonBox, QFontDatabase, QLabel, QLineEdit, QTextEdit,
                    QVBoxLayout, QWidget)

from ..anki_api.confirm import FULL_SYNC_WARNING, ChangePlan, Decision


def _details_html(details: str) -> str:
    """Diff-like lines in colour: "+" added, "-" removed or lost."""
    lines = []
    for line in details.splitlines():
        text = html.escape(line) or "&nbsp;"
        if line.startswith("+") and not line.startswith("+++"):
            text = f'<span style="color: #2e7d32;">{text}</span>'
        elif line.startswith("-") and not line.startswith("---"):
            text = f'<span style="color: #c62828;">{text}</span>'
        lines.append(text)
    return '<pre style="white-space: pre-wrap; margin: 0;">' + "<br>".join(lines) + "</pre>"


class ConfirmDialog(QDialog):
    def __init__(self, parent: QWidget | None, plan: ChangePlan):
        super().__init__(parent)
        self.setWindowTitle("AI Assistant: confirm the change")
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)

        title = QLabel(plan.title, self)
        font = title.font()
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() * 1.15)
        title.setFont(font)
        title.setWordWrap(True)
        layout.addWidget(title)

        summary = QLabel(plan.summary, self)
        summary.setWordWrap(True)
        layout.addWidget(summary)

        if plan.full_sync:
            warning = QLabel("⚠️ " + FULL_SYNC_WARNING, self)
            warning.setWordWrap(True)
            warning.setStyleSheet("color: #c62828;")
            layout.addWidget(warning)

        if plan.details:
            details = QTextEdit(self)
            details.setReadOnly(True)
            details.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
            details.setHtml(_details_html(plan.details))
            details.setMinimumHeight(220)
            layout.addWidget(details, stretch=1)

        layout.addWidget(QLabel("Undo with Edit > Undo after it is done.", self))

        self.comment = QLineEdit(self)
        self.comment.setPlaceholderText("Comment for the assistant (optional), e.g. what to do instead")
        layout.addWidget(self.comment)

        self.trust = QCheckBox("Don't ask again for this kind of change in this dialog", self)
        layout.addWidget(self.trust)

        buttons = QDialogButtonBox(self)
        apply_button = buttons.addButton("Apply", QDialogButtonBox.ButtonRole.AcceptRole)
        decline_button = buttons.addButton("Decline", QDialogButtonBox.ButtonRole.RejectRole)
        apply_button.setAutoDefault(False)
        # Enter in the comment field declines: the safe choice
        decline_button.setDefault(True)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        decline_button.setFocus()

    def decision(self) -> Decision:
        approved = self.exec() == QDialog.DialogCode.Accepted
        return Decision(approved, self.comment.text().strip(), trust_kind=approved and self.trust.isChecked())


def confirm_change(parent: QWidget | None, plan: ChangePlan) -> Decision:
    """Show the plan and wait for the user's answer. Must be called on the main thread."""
    return ConfirmDialog(parent, plan).decision()
