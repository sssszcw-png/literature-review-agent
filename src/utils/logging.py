"""Structured logging setup with Rich handler."""

import logging
from pathlib import Path


def setup_logging(level: str = "INFO", log_file: Path | None = None) -> None:
    """Configure structured logging for the application."""
    from rich.logging import RichHandler

    fmt = "%(name)s: %(message)s"
    rich_handler = RichHandler(rich_tracebacks=True, markup=True)
    rich_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    rich_handler.setFormatter(logging.Formatter(fmt))

    root = logging.getLogger("academic_research")
    root.setLevel(logging.DEBUG)
    root.handlers.clear()
    root.addHandler(rich_handler)
    root.propagate = False

    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger for the given module name."""
    return logging.getLogger(f"academic_research.{name}")
