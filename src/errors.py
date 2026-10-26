"""Errors exposed by the command-line application."""


class ConfigError(ValueError):
    """Invalid configuration; safe to display without a traceback."""


class DataError(ValueError):
    """Input does not satisfy the battery CSV contract."""


class StageNotImplementedError(RuntimeError):
    """A registered command belongs to a later implementation stage."""

    def __init__(self, command: str, stage: int) -> None:
        self.command = command
        self.stage = stage
        super().__init__(f"'{command}' belum diimplementasikan; tersedia pada Tahap {stage}. "
                         "Gunakan main.py --help untuk melihat status implementasi perintah.")
