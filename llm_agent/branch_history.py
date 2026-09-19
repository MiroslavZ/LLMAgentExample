from copy import deepcopy
from pathlib import Path

from .history import DEFAULT_HISTORY_PATH, HistoryManager


class BranchHistoryManager(HistoryManager):
    """Хранит независимые диалоги и неизменяемые снимки в одном JSON-файле."""

    def __init__(
        self, path: str | Path = DEFAULT_HISTORY_PATH, *, branch: str | None = None,
    ) -> None:
        if branch is not None:
            self.validate_name(branch)
        super().__init__(path)
        if branch is not None:
            self.switch_branch(branch)

    @staticmethod
    def validate_name(name: str) -> str:
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise ValueError("Имя ветки или checkpoint должно быть непустой строкой без пробелов по краям")
        return name

    @property
    def active_branch(self) -> str:
        return self._active_branch

    def list_branches(self) -> list[str]:
        return list(self._branches)

    def list_checkpoints(self) -> list[str]:
        return list(self._checkpoints)

    def create_checkpoint(self, name: str) -> None:
        """Сохранить текущий диалог; существующий checkpoint нельзя перезаписать."""
        self.validate_name(name)
        if name in self._checkpoints:
            raise ValueError(f"Checkpoint уже существует: {name}")
        checkpoints = {**self._checkpoints, name: deepcopy(self._branches[self._active_branch])}
        self._commit(self._branches, checkpoints, self._active_branch)

    def create_branch(self, name: str, *, from_checkpoint: str) -> None:
        """Создать независимую копию checkpoint и сделать новую ветку активной."""
        self.validate_name(name)
        self.validate_name(from_checkpoint)
        if name in self._branches:
            raise ValueError(f"Ветка уже существует: {name}")
        if from_checkpoint not in self._checkpoints:
            raise ValueError(f"Checkpoint не найден: {from_checkpoint}")
        branches = {**self._branches, name: deepcopy(self._checkpoints[from_checkpoint])}
        self._commit(branches, self._checkpoints, name)

    def switch_branch(self, name: str) -> None:
        """Переключить контекст и сохранить выбор для следующих запусков."""
        self.validate_name(name)
        if name not in self._branches:
            raise ValueError(f"Ветка не найдена: {name}")
        if name != self._active_branch:
            self._commit(self._branches, self._checkpoints, name)

    def clear(self) -> None:
        """Очистить только активную ветку; остальные ветки и checkpoints сохраняются."""
        super().clear()

    def _read_data(self) -> object:
        data = super()._read_data()
        if isinstance(data, dict) and "branches" in data:
            if (
                set(data) != {"version", "active_branch", "branches", "checkpoints"}
                or type(data["version"]) is not int or data["version"] != 1
                or not isinstance(data["branches"], dict) or "main" not in data["branches"]
                or not isinstance(data["checkpoints"], dict)
            ):
                raise ValueError("Некорректный формат истории веток")
            self.validate_name(data["active_branch"])
            if data["active_branch"] not in data["branches"]:
                raise ValueError("Активная ветка отсутствует в истории")
            for snapshots in (data["branches"], data["checkpoints"]):
                for name, snapshot in snapshots.items():
                    self.validate_name(name)
                    self._decode_data(snapshot)
            self._branches = data["branches"]
            self._checkpoints = data["checkpoints"]
            self._active_branch = data["active_branch"]
        else:
            # Старые сообщения, память и статистика становятся началом ветки main.
            self._decode_data(data)
            self._branches = {"main": data}
            self._checkpoints = {}
            self._active_branch = "main"
        self._loaded_data = data
        return self._branches[self._active_branch]

    def _write_data(self, data: object) -> None:
        branches = {**self._branches, self._active_branch: deepcopy(data)}
        self._commit(branches, self._checkpoints, self._active_branch)

    def _commit(
        self, branches: dict[str, object], checkpoints: dict[str, object], active_branch: str,
    ) -> None:
        state = self._decode_data(deepcopy(branches[active_branch]))
        if super()._read_data() != self._loaded_data:
            raise ValueError("История изменена другим экземпляром; загрузите её заново перед продолжением")
        data = {
            "version": 1,
            "active_branch": active_branch,
            "branches": branches,
            "checkpoints": checkpoints,
        }
        super()._write_data(data)
        self._loaded_data = data
        self._branches = branches
        self._checkpoints = checkpoints
        self._active_branch = active_branch
        self._messages, self._summary, self._facts, self._archived_usage = state
        self._task_state = self.task_from_data(branches[active_branch])
