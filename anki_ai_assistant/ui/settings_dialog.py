"""The settings dialog: providers, the models of each role and the other settings, instead of editing JSON.

It edits a copy of the full config, so keys it has no fields for are kept as they are."""

import copy
import json
from collections.abc import Callable

from aqt.qt import (QComboBox, QDialog, QDialogButtonBox, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
                    QLabel, QLineEdit, QPlainTextEdit, QPushButton, QSpinBox, QTableWidget, QTabWidget,
                    QVBoxLayout, QWidget)
from aqt.utils import showWarning

from ..core.config import ROLES, check_config
from ..core.context import model_label, parse_presets

ROLE_TITLES = {
    "manager": "Assistant (works with your collection; a cloud model is recommended)",
    "searcher": "Image search",
    "validator": "Image check (the model must accept images)",
}
REASONING_EFFORTS = ["", "none", "low", "medium", "high"]


class RoleForm(QGroupBox):
    def __init__(self, parent: QWidget, role: str, config: dict):
        super().__init__(ROLE_TITLES[role], parent)
        self.role = role
        self.config = dict(config)
        form = QFormLayout(self)

        self.provider = QComboBox(self)
        form.addRow("Provider", self.provider)
        self.model = QLineEdit(config.get("model", ""), self)
        self.model.setPlaceholderText("e.g. z-ai/glm-5.3-flash or qwen3.5:9b")
        form.addRow("Model", self.model)

        self.reasoning = QComboBox(self)
        self.reasoning.setEditable(True)
        self.reasoning.addItems(REASONING_EFFORTS)
        self.reasoning.setCurrentText(config.get("reasoning_effort", ""))
        self.reasoning.setToolTip("Empty: the model's default. 'none' turns thinking off (much faster with Ollama).")
        form.addRow("Reasoning effort", self.reasoning)

        self.max_steps = None
        if role != "validator":
            self.max_steps = QSpinBox(self)
            self.max_steps.setRange(1, 200)
            self.max_steps.setValue(int(config.get("max_steps", 20)))
            form.addRow("Max steps per request", self.max_steps)

        self.context_length = QSpinBox(self)
        self.context_length.setRange(0, 1048576)
        self.context_length.setSingleStep(4096)
        self.context_length.setSpecialValueText("server default")
        self.context_length.setValue(int(config.get("context_length") or 0))
        self.context_length.setToolTip("Ollama only, other providers ignore it. The assistant needs at least 16384 tokens, 32768 is better.")
        form.addRow("Context length (Ollama)", self.context_length)

        extra = config.get("extra_body")
        self.extra_body = QLineEdit(json.dumps(extra, ensure_ascii=False) if extra else "", self)
        self.extra_body.setPlaceholderText('optional JSON, e.g. {"provider": {"order": ["fireworks"]}}')
        form.addRow("Extra request fields", self.extra_body)

    def set_providers(self, names: list[str], renamed: dict[str, str]) -> None:
        current = self.provider.currentText() or self.config.get("provider", "")
        current = renamed.get(current, current)
        self.provider.clear()
        self.provider.addItems(names)
        if current and current not in names:
            # A provider that was renamed or removed stays visible, and saving reports it
            self.provider.addItem(current)
        self.provider.setCurrentText(current)

    def value(self) -> dict:
        """The role config; raises ValueError for invalid extra fields JSON."""
        role = dict(self.config)
        role.update(provider=self.provider.currentText(), model=self.model.text().strip(),
                    reasoning_effort=self.reasoning.currentText().strip())
        if self.max_steps is not None:
            role["max_steps"] = self.max_steps.value()
        if self.context_length.value():
            role["context_length"] = self.context_length.value()
        else:
            role.pop("context_length", None)
        text = self.extra_body.text().strip()
        if text:
            try:
                role["extra_body"] = json.loads(text)
            except json.JSONDecodeError as e:
                raise ValueError(f"{ROLE_TITLES[self.role]}: extra request fields are not valid JSON ({e}).")
        else:
            role.pop("extra_body", None)
        return role


class SettingsDialog(QDialog):
    def __init__(self, parent: QWidget | None, config: dict, save: Callable[[dict], None]):
        super().__init__(parent)
        self.setWindowTitle("AI Assistant settings")
        self.setMinimumWidth(640)
        self.config = copy.deepcopy(config)
        self._save = save
        layout = QVBoxLayout(self)
        tabs = QTabWidget(self)
        layout.addWidget(tabs)

        # Providers
        providers_tab = QWidget(self)
        providers_layout = QVBoxLayout(providers_tab)
        providers_layout.addWidget(QLabel(
            "OpenAI-compatible endpoints. The API base ends with /v1 (Ollama: http://localhost:11434/v1). "
            "Keys are stored in the add-on config in plain text; an empty key falls back to the environment "
            "variable <NAME>_API_KEY.", providers_tab, wordWrap=True))
        self.providers = QTableWidget(0, 3, providers_tab)
        self.providers.setHorizontalHeaderLabels(["Name", "API base", "API key"])
        self.providers.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.providers.verticalHeader().hide()
        providers_layout.addWidget(self.providers)
        for name, provider in self.config["providers"].items():
            self._add_provider_row(name, provider.get("api_base", ""), provider.get("api_key", ""))
        row_buttons = QHBoxLayout()
        add_button = QPushButton("Add provider", providers_tab)
        add_button.clicked.connect(lambda: self._add_provider_row("", "https://", ""))
        remove_button = QPushButton("Remove selected", providers_tab)
        remove_button.clicked.connect(self._remove_provider_row)
        row_buttons.addWidget(add_button)
        row_buttons.addWidget(remove_button)
        row_buttons.addStretch()
        providers_layout.addLayout(row_buttons)
        tabs.addTab(providers_tab, "Providers")

        # Models
        models_tab = QWidget(self)
        models_layout = QVBoxLayout(models_tab)
        self.roles = {role: RoleForm(models_tab, role, self.config["roles"].get(role, {})) for role in ROLES}
        for form in self.roles.values():
            models_layout.addWidget(form)
        presets_box = QGroupBox("Other models to pick in the chat", models_tab)
        presets_layout = QVBoxLayout(presets_box)
        presets_layout.addWidget(QLabel("One per line: provider / model", presets_box))
        self.presets = QPlainTextEdit(presets_box)
        self.presets.setPlainText("\n".join(
            model_label(p.get("provider", ""), p.get("model", ""))
            for p in self.config.get("manager_presets") or [] if isinstance(p, dict)))
        self.presets.setMaximumHeight(90)
        presets_layout.addWidget(self.presets)
        models_layout.addWidget(presets_box)
        tabs.addTab(models_tab, "Models")

        # General
        general_tab = QWidget(self)
        general = QFormLayout(general_tab)
        self.openserp = QLineEdit(self.config.get("openserp_base_url", ""), general_tab)
        self.openserp.setPlaceholderText("optional: images come from Wikimedia without it")
        general.addRow("OpenSERP URL", self.openserp)
        self.timeout = QSpinBox(general_tab)
        self.timeout.setRange(10, 3600)
        self.timeout.setSuffix(" s")
        self.timeout.setValue(int(self.config.get("llm_timeout", 180)))
        general.addRow("Model request timeout", self.timeout)
        self.retries = QSpinBox(general_tab)
        self.retries.setRange(0, 10)
        self.retries.setValue(int(self.config.get("llm_max_retries", 2)))
        general.addRow("Retries of a failed request", self.retries)
        tabs.addTab(general_tab, "General")

        tabs.currentChanged.connect(lambda _index: self._refresh_providers())
        self._refresh_providers()

        layout.addWidget(QLabel("Changes apply in the next dialog of the chat.", self))
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel,
                                   self)
        buttons.accepted.connect(self._on_save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add_provider_row(self, name: str, api_base: str, api_key: str) -> None:
        row = self.providers.rowCount()
        self.providers.insertRow(row)
        for column, value in enumerate((name, api_base, api_key)):
            edit = QLineEdit(value, self.providers)
            edit.setFrame(False)
            if column == 0:
                # To follow a rename into the roles that use the provider
                edit.setProperty("original_name", name)
                edit.editingFinished.connect(self._refresh_providers)
            if column == 2:
                edit.setEchoMode(QLineEdit.EchoMode.PasswordEchoOnEdit)
            self.providers.setCellWidget(row, column, edit)
        self.providers.setColumnWidth(0, 110)
        self.providers.setColumnWidth(2, 180)

    def _remove_provider_row(self) -> None:
        row = self.providers.currentRow()
        if row >= 0:
            self.providers.removeRow(row)

    def _provider_rows(self) -> list[tuple[str, str, str]]:
        return [tuple(self.providers.cellWidget(row, column).text().strip() for column in range(3))
                for row in range(self.providers.rowCount())]

    def _renamed_providers(self) -> dict[str, str]:
        renamed = {}
        for row in range(self.providers.rowCount()):
            edit = self.providers.cellWidget(row, 0)
            original, name = edit.property("original_name"), edit.text().strip()
            if original and name and original != name:
                renamed[original] = name
        return renamed

    def _refresh_providers(self) -> None:
        names = [name for name, _, _ in self._provider_rows() if name]
        renamed = self._renamed_providers()
        for form in self.roles.values():
            form.set_providers(names, renamed)

    def collect(self) -> dict:
        """The edited config; raises ValueError with a message for the user."""
        self._refresh_providers()
        renamed = self._renamed_providers()
        original_names = {new: old for old, new in renamed.items()}
        config = copy.deepcopy(self.config)
        providers = {}
        for name, api_base, api_key in self._provider_rows():
            if not name and api_base in ("", "http://", "https://"):
                continue  # an empty row
            if name in providers:
                raise ValueError(f"The provider name '{name}' is used twice.")
            old = self.config["providers"].get(original_names.get(name, name), {})
            providers[name] = {**old, "api_base": api_base, "api_key": api_key}
        config["providers"] = providers
        config["roles"] = {**config["roles"], **{role: form.value() for role, form in self.roles.items()}}
        presets = parse_presets(self.presets.toPlainText(), self.config.get("manager_presets") or [])
        for preset in presets:
            preset["provider"] = renamed.get(preset["provider"], preset["provider"])
        config["manager_presets"] = presets
        config["openserp_base_url"] = self.openserp.text().strip()
        config["llm_timeout"] = self.timeout.value()
        config["llm_max_retries"] = self.retries.value()
        problems = check_config(config)
        if problems:
            raise ValueError("\n".join(problems))
        return config

    def _on_save(self) -> None:
        try:
            config = self.collect()
        except ValueError as e:
            showWarning(str(e), parent=self, title="AI Assistant settings")
            return
        self._save(config)
        self.accept()
