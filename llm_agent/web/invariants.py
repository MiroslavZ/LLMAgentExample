"""Просмотр и явное редактирование общих правил вне диалога с моделью."""

from collections.abc import Callable
from dataclasses import dataclass

from nicegui import ui

from ..invariants import Invariant, InvariantSet, InvariantStorageError
from ..service import ConversationService


@dataclass
class RuleEditor:
    root: ui.card
    id_input: ui.input
    description_input: ui.textarea


class InvariantPanel:
    def __init__(self, service: ConversationService, *, on_save: Callable[[], None]) -> None:
        self.service = service
        self.on_save = on_save
        self.available = False
        self._rules_key: object = None
        self.dialog: ui.dialog | None = None
        self.editors: list[RuleEditor] = []
        self._editing: InvariantSet | None = None

    def build(self) -> None:
        with ui.column().classes("w-full gap-3 mt-5 pb-5 border-b border-gray-200") as self.root:
            with ui.row().classes("items-center gap-2"):
                ui.icon("verified_user", size="20px")
                ui.label("Инварианты агента").classes("font-medium")
            ui.label(
                "Обязательные правила для всех диалогов. Сообщения, профиль и память "
                "не могут их отменить. При конфликте агент объяснит отказ."
            ).classes("setting-note")
            self.status = ui.label().classes("setting-note")
            self.details = ui.column().classes("w-full gap-2")
            self.edit_button = ui.button(
                "Изменить правила", icon="edit", on_click=self.open,
            ).props("outline no-caps").classes("w-full")
            with ui.expansion("Файл и CLI", icon="description").classes("w-full"):
                ui.label(f"Файл: {self.service.invariants_path}").classes("memory-value")
                ui.label(
                    "Также можно изменить JSON-файл или импортировать правила через CLI: "
                    "--invariants-file PATH --invariants-import SOURCE. "
                    "Изменения действуют для всех диалогов со следующего запроса."
                ).classes("setting-note")

    def refresh(self) -> None:
        try:
            invariants = self.service.get_invariants()
        except InvariantStorageError:
            self.available = False
            self.edit_button.set_enabled(False)
            self._rules_key = None
            self.details.clear()
            self.status.set_text("Не удалось прочитать инварианты. Исправьте файл правил; отправка временно недоступна.")
            return
        self.available = True
        self.edit_button.set_enabled(True)
        self.status.set_text(f"Обязательных правил: {len(invariants.rules)}" if invariants.rules else "Инварианты не заданы.")
        if invariants == self._rules_key:
            return
        self._rules_key = invariants
        self.details.clear()
        with self.details:
            for rule in invariants.rules:
                with ui.column().classes("w-full gap-1"):
                    ui.label(rule.id).classes("font-medium break-words")
                    ui.label(rule.description).classes("memory-value")

    def open(self) -> None:
        if self.dialog and self.dialog.value:
            return
        try:
            self._editing = self.service.get_invariants()
        except InvariantStorageError:
            self.refresh()
            ui.notify("Не удалось прочитать правила. Проверьте файл инвариантов.", type="negative")
            return
        self.editors = []
        with self.root, ui.dialog().props("persistent") as dialog, ui.card().classes("invariant-dialog"):
            self.dialog = dialog
            with ui.row().classes("w-full items-center no-wrap"):
                ui.label("Изменение инвариантов").classes("text-lg font-semibold")
                ui.space()
                ui.button(icon="close", on_click=dialog.close).props(
                    'flat round dense aria-label="Закрыть редактор инвариантов"'
                )
            ui.label(
                "Правила общие для всех диалогов. Изменения начнут действовать после сохранения, "
                "со следующего запроса. Уже выполняющийся запрос использует прежние правила."
            ).classes("memory-description")
            ui.label(
                "ID — уникальное короткое имя, например stack или architecture: "
                "латинские буквы, цифры, дефис и подчёркивание."
            ).classes("memory-description")
            self.editor_error = ui.label().classes("memory-status")
            self.rows = ui.column().classes("w-full gap-3")
            for rule in self._editing.rules:
                self.add_rule(rule)
            self.add_button = ui.button(
                "Добавить правило", icon="add", on_click=lambda: self.add_rule(),
            ).props("outline no-caps")
            ui.label(
                "Удаление всех правил отключит проверку инвариантов. "
                "Добавление, изменение и удаление применяются только кнопкой «Сохранить»."
            ).classes("memory-description")
            with ui.row().classes("w-full justify-end"):
                ui.button("Отмена", on_click=dialog.close).props("flat no-caps")
                self.save_button = ui.button(
                    "Сохранить", icon="save", on_click=self.save,
                ).props("unelevated no-caps")
        dialog.on("hide", dialog.delete)
        dialog.open()

    def add_rule(self, rule: Invariant | None = None) -> None:
        if rule is None:
            used = {editor.id_input.value for editor in self.editors}
            number = 1
            while f"rule-{number}" in used:
                number += 1
            rule_id, description = f"rule-{number}", ""
        else:
            rule_id, description = rule.id, rule.description
        with self.rows, ui.card().classes("w-full gap-3 shadow-none border border-gray-200") as root:
            with ui.row().classes("w-full items-start no-wrap"):
                id_input = ui.input("ID правила", value=rule_id).props("outlined").classes("flex-1 min-w-0")
                delete = ui.button(icon="delete_outline").props(
                    'flat round aria-label="Удалить правило"'
                ).tooltip("Удалить из списка; применяется после сохранения")
            description_input = ui.textarea(
                "Обязательное правило", value=description,
                placeholder="Например: сервер реализуется только на Python",
            ).props("outlined autogrow rows=2").classes("w-full")
        editor = RuleEditor(root, id_input, description_input)
        self.editors.append(editor)
        delete.on_click(lambda: self.remove_rule(editor))

    def remove_rule(self, editor: RuleEditor) -> None:
        if editor in self.editors:
            self.editors.remove(editor)
            editor.root.delete()

    def save(self) -> None:
        if self.dialog is None or not self.dialog.value or self._editing is None:
            return
        try:
            invariants = InvariantSet(tuple(
                Invariant((editor.id_input.value or "").strip(), (editor.description_input.value or "").strip())
                for editor in self.editors
            ))
            self.service.save_invariants(invariants, expected=self._editing)
        except ValueError as error:
            self.editor_error.set_text(str(error))
            return
        except InvariantStorageError:
            self.editor_error.set_text("Не удалось сохранить правила. Проверьте файл и доступ к каталогу. Правки остаются в форме.")
            return
        self.dialog.close()
        self.refresh()
        self.on_save()
        ui.notify("Инварианты сохранены. Новые правила действуют со следующего запроса.", type="positive")
