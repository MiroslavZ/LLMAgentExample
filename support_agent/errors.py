"""Ошибки, текст которых безопасно показывать в интерфейсе."""


class SupportError(RuntimeError):
    def __init__(self, message: str, status_code: int = 503) -> None:
        super().__init__(message)
        self.status_code = status_code
