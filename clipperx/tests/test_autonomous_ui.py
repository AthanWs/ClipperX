"""Automatic operation is accessible from the existing interface and runner."""
import asyncio
import ctypes
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
from textual.widgets import Input, Log, ProgressBar, Select

from clipperx import cli, interface, runner


def test_ui_defaults_to_automatic_and_starts_selected_video(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"UI input only")
    output = tmp_path / "out"
    output.mkdir()
    summary = output / "autonomous-summary.json"
    summary.write_text(json.dumps({"status": "partial", "rendered_clips": 0}), encoding="utf-8")
    calls = []
    monkeypatch.setattr(interface, "configure_logging", lambda **kwargs: None)

    def start(video, output_dir, *, mode):
        calls.append((video, output_dir, mode))
        async def finished():
            return [summary]
        return asyncio.create_task(finished())

    monkeypatch.setattr(interface, "start", start)
    app = interface.SimpleUI()
    async def exercise():
        async with app.run_test(headless=True, size=(100, 50)) as pilot:
            assert app.query_one("#mode", Select).value == "auto"
            app.query_one("#video", Input).value = str(source)
            app.query_one("#output", Input).value = str(output)
            await pilot.click("#start")
            await pilot.pause()
            assert calls == [(source, output, "auto")]
            text = "\n".join(app.query_one("#log", Log).lines)
            assert "parcial" in text.lower()
            assert "autonomous-summary.json" in text
    asyncio.run(exercise())


def test_worker_uses_selected_luna_and_reports_partial_results(tmp_path, monkeypatch):
    from clipperx import autonomous_pipeline
    from clipperx.progress import reporter

    source = tmp_path / "source.mp4"
    source.write_bytes(b"Worker input only")
    summary = tmp_path / "autonomous-summary.json"
    summary.write_text('{"status":"partial"}', encoding="utf-8")
    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"already rendered")
    messages = []
    client = object()
    monkeypatch.setattr(cli, "_automatic_client", lambda model, context, admitted: client)
    def run(video, output, **kwargs):
        assert video == source and output == tmp_path
        assert kwargs["client"] is client and kwargs["model_name"] == "gpt-6-luna(low)"
        return SimpleNamespace(summary_path=summary, clips=(clip,))
    monkeypatch.setattr(autonomous_pipeline, "run_autonomous_pipeline", run)
    with reporter._lock:
        previous = reporter._subscribers[:]
    try:
        runner._pipeline_worker(str(source), str(tmp_path), "auto", SimpleNamespace(put=messages.append))
    finally:
        with reporter._lock:
            reporter._subscribers[:] = previous
    final, = [message for message in messages if message["kind"] == "result"]
    assert final["ok"] is True and final["status"] == "partial"
    assert final["paths"] == [str(summary), str(clip)]


def test_progress_updates_from_ui_loop_and_background_thread(monkeypatch):
    from clipperx.progress import reporter
    monkeypatch.setattr(interface, "configure_logging", lambda **kwargs: None)
    app = interface.SimpleUI()
    async def exercise():
        async with app.run_test(headless=True, size=(100, 50)) as pilot:
            reporter.set_percent("automatic", 37, "Etapa editorial")
            await pilot.pause()
            assert app.query_one("#progress", ProgressBar).progress == 37
            await asyncio.to_thread(reporter.set_percent, "automatic", 68, "Etapa visual")
            await pilot.pause()
            assert app.query_one("#progress", ProgressBar).progress == 68
            assert "Etapa visual" in "\n".join(app.query_one("#log", Log).lines)
    asyncio.run(exercise())


@pytest.mark.skipif(os.name != "nt", reason="Windows process-tree retirement")
def test_stop_retires_owned_worker_descendant(tmp_path, monkeypatch):
    marker = tmp_path / "owned-child.pid"
    code = (
        "import pathlib,subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(90)'], "
        "creationflags=subprocess.CREATE_NO_WINDOW); "
        "pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(90)"
    )
    process = subprocess.Popen([sys.executable, "-c", code, str(marker)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW)

    class Adapter:
        pid = process.pid
        def is_alive(self):
            return process.poll() is None
        def join(self, timeout=None):
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass
        def terminate(self):
            process.terminate()
        def kill(self):
            process.kill()

    deadline = time.monotonic() + 15
    handle = None
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.WaitForSingleObject.restype = ctypes.c_ulong
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    try:
        while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(.05)
        assert marker.exists(), "owned child did not start"
        child_pid = int(marker.read_text())
        # Holding the process handle prevents PID reuse while checking/cleaning it.
        handle = kernel.OpenProcess(0x00100000, False, child_pid)
        assert handle and kernel.WaitForSingleObject(handle, 0) == 0x102
        monkeypatch.setattr(runner, "_current_process", Adapter())
        monkeypatch.setattr(runner, "_current_task", None)
        runner.stop()
        assert process.poll() is not None
        assert kernel.WaitForSingleObject(handle, 3000) == 0
    finally:
        if process.poll() is None:
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                capture_output=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
            process.wait(timeout=5)
        if handle:
            if kernel.WaitForSingleObject(handle, 0) == 0x102:
                subprocess.run(["taskkill", "/PID", str(child_pid), "/T", "/F"],
                    capture_output=True, timeout=10, creationflags=subprocess.CREATE_NO_WINDOW)
            kernel.CloseHandle(handle)
