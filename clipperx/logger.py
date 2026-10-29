"""Central logging configuration for ClipperX.

Provides a file logger by default and an optional console logger for CLI.
When the Textual UI is active, we can enable a UI log handler that streams
messages directly into the UI's log widget to avoid interfering with the
terminal rendering.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from typing import Callable, Optional, Type

from rich.logging import RichHandler

from .local_paths import paths


_ui_writer: Optional[Callable[[str], None]] = None


class UILogHandler(logging.Handler):
    """Logging handler that forwards messages to a UI writer callback.

    The UI writer should accept a single string argument and render it in the UI.
    """

    def emit(self, record: logging.LogRecord) -> None:  # pragma: no cover - integration
        global _ui_writer
        if _ui_writer is None:
            return
        try:
            msg = self.format(record)
            _ui_writer(msg)
        except Exception:
            # Never raise from logging
            pass


def _has_handler(root: logging.Logger, handler_type: Type[logging.Handler]) -> bool:
    return any(isinstance(h, handler_type) for h in root.handlers)


def _remove_handlers(root: logging.Logger, handler_type: Type[logging.Handler]) -> None:
    to_remove = [h for h in root.handlers if isinstance(h, handler_type)]
    for h in to_remove:
        root.removeHandler(h)


def configure_logging(use_console: bool = True) -> None:
    """Configure root logging with file handler and optional console handler.

    Idempotent: safe to call multiple times.
    """
    paths.logs_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Ensure file handler exists
    if not _has_handler(root, RotatingFileHandler):
        file_handler = RotatingFileHandler(
            paths.logs_dir / "clipperx.log",
            maxBytes=5_000_000,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"))
        root.addHandler(file_handler)

    # Optionally add console rich handler for CLI
    if use_console and not _has_handler(root, RichHandler):
        console_handler = RichHandler(markup=True)
        console_handler.setLevel(logging.INFO)
        root.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def enable_ui_logging(writer: Callable[[str], None]) -> None:
    """Enable UI logging by routing records to the provided writer.

    Also removes the Rich console handler to avoid interfering with the Textual
    UI rendering.
    """
    global _ui_writer
    _ui_writer = writer

    root = logging.getLogger()
    # Remove console rich handler if present
    _remove_handlers(root, RichHandler)

    # Add UI handler if not present
    if not _has_handler(root, UILogHandler):
        ui_handler = UILogHandler()
        ui_handler.setLevel(logging.INFO)
        ui_handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
        root.addHandler(ui_handler)


def disable_ui_logging() -> None:
    """Disable UI logging and restore console rich handler if desired later."""
    global _ui_writer
    _ui_writer = None

    root = logging.getLogger()
    _remove_handlers(root, UILogHandler)
