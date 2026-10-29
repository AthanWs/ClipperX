"""Focused regression checks for Textual logging in the simple interface."""

from __future__ import annotations

import asyncio
import inspect
import logging
from typing import Any

import pytest
from textual import constants
from textual._log import LogGroup, LogVerbosity
from textual.app import App
from textual.widgets import Log

from clipperx import interface, logger
from clipperx.progress import reporter


def test_public_log_uses_inherited_framework_callback_when_connected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[tuple[object, object, inspect.Traceback, tuple[object, ...], dict[str, object]]] = []

    def framework_log(
        self: App[Any],
        group: object,
        verbosity: object,
        caller: inspect.Traceback,
        *objects: object,
        **kwargs: object,
    ) -> None:
        received.append((group, verbosity, caller, objects, kwargs))

    monkeypatch.setattr(constants, "LOG_FILE", None)
    monkeypatch.setattr(App, "_log", framework_log)
    monkeypatch.setattr(App, "_is_devtools_connected", property(lambda self: True))
    app = interface.SimpleUI()
    received.clear()

    app.log("Task 80 public message", detail="keyword payload")

    assert len(received) == 1
    group, verbosity, caller, objects, kwargs = received[0]
    assert group is LogGroup.INFO
    assert verbosity is LogVerbosity.NORMAL
    assert caller.function == "test_public_log_uses_inherited_framework_callback_when_connected"
    assert objects == ("Task 80 public message",)
    assert kwargs == {"detail": "keyword payload"}


def test_headless_ui_log_writes_real_widget_and_cleans_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    original_writer = logger._ui_writer
    with reporter._lock:
        original_subscribers = reporter._subscribers[:]
    configured: list[bool] = []
    monkeypatch.setattr(interface, "configure_logging", lambda use_console: configured.append(use_console))
    monkeypatch.setattr(interface, "start", lambda *args, **kwargs: pytest.fail("pipeline started"))
    monkeypatch.setattr(interface, "stop", lambda: pytest.fail("pipeline stopped"))

    app = interface.SimpleUI()

    async def exercise_ui() -> None:
        async with app.run_test(headless=True) as pilot:
            widget = app.query_one("#log", Log)
            with reporter._lock:
                assert app._on_progress in reporter._subscribers
            ui_handlers = [handler for handler in root.handlers if isinstance(handler, logger.UILogHandler)]
            assert len(ui_handlers) == 1

            app._write_ui_log("Task 80 visible message\n")
            ui_handlers[0].handle(
                logging.LogRecord("clipperx.task80", logging.INFO, __file__, 0, "handler message", (), None)
            )
            await pilot.pause()
            assert widget.lines == [
                "Task 80 visible message",
                "INFO | clipperx.task80 | handler message",
            ]

    try:
        asyncio.run(exercise_ui())
        assert configured == [False]
        with reporter._lock:
            assert reporter._subscribers == original_subscribers
        assert logger._ui_writer is None
        assert not any(isinstance(handler, logger.UILogHandler) for handler in root.handlers)
    finally:
        for handler in root.handlers[:]:
            if handler not in original_handlers:
                root.removeHandler(handler)
                handler.close()
        for handler in original_handlers:
            if handler not in root.handlers:
                root.addHandler(handler)
        root.setLevel(original_level)
        logger._ui_writer = original_writer
        with reporter._lock:
            reporter._subscribers[:] = original_subscribers
