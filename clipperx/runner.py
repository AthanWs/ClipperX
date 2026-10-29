"""Cancelable process runner for long ClipperX jobs."""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import queue
import subprocess
import threading
import traceback
from asyncio import Task
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any

from .logger import get_logger


logger = get_logger(__name__)
_current_task: Task[list[Path]] | None = None
_current_process: BaseProcess | None = None
_process_lock = threading.Lock()


def _pipeline_worker(video: str, output_dir: str | None, mode: str, result_queue: Any) -> None:
    try:
        from .progress import reporter

        def forward_progress(event: Any) -> None:
            result_queue.put(
                {
                    "kind": "progress",
                    "stage": event.stage,
                    "percent": event.percent,
                    "message": event.message,
                    "minutes_done": event.minutes_done,
                    "total_minutes": event.total_minutes,
                }
            )

        reporter.subscribe(forward_progress)

        if mode == "auto":
            import json
            from uuid import uuid4
            from .autonomous_pipeline import run_autonomous_pipeline
            from .cli import _automatic_client
            from .local_paths import paths
            from .style_library import StyleLibrary
            from .youtube import download_video, is_youtube_url

            client = _automatic_client("gpt-6-luna(low)", 65536, True)
            source = Path(video)
            if is_youtube_url(video):
                source = download_video(video, paths.downloads_dir / f"auto-{uuid4().hex}")
            result = run_autonomous_pipeline(source,
                Path(output_dir) if output_dir else paths.clips_dir,
                client=client, model_name="gpt-6-luna(low)",
                style_library=StyleLibrary(paths.data_dir / "processed" / "style_library"))
            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            result_queue.put({"kind": "result", "ok": True, "status": summary["status"],
                              "paths": [str(result.summary_path), *(str(path) for path in result.clips)]})
            return
        if mode == "index":
            from .timeline import build_timeline

            results = [build_timeline(Path(video), include_audio=True)]
        elif mode == "transcribe":
            from .stages import ensure_transcript

            _, transcript = ensure_transcript(Path(video))
            results = [transcript]
        elif mode == "pipeline":
            from .pipeline import full_pipeline

            results = full_pipeline(
                video_path=video,
                output_dir=Path(output_dir) if output_dir else None,
            )
        else:
            raise ValueError(f"Modo desconocido: {mode}")
        result_queue.put({"kind": "result", "ok": True, "paths": [str(path) for path in results]})
    except BaseException as exc:
        result_queue.put(
            {
                "kind": "result",
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


async def run_pipeline(
    video_path: Path | str,
    output_dir: Path | None = None,
    *,
    mode: str = "pipeline",
) -> list[Path]:
    global _current_process
    context = mp.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_pipeline_worker,
        args=(str(video_path), str(output_dir) if output_dir else None, mode, result_queue),
        name="ClipperXPipeline",
    )
    with _process_lock:
        _current_process = process
    process.start()
    result: dict[str, Any] | None = None
    try:
        while process.is_alive() or result is None:
            try:
                message = result_queue.get_nowait()
            except queue.Empty:
                if not process.is_alive():
                    await asyncio.sleep(0.1)
                    try:
                        message = result_queue.get_nowait()
                    except queue.Empty:
                        break
                else:
                    await asyncio.sleep(0.1)
                    continue
            if message.get("kind") == "progress":
                from .progress import ProgressEvent, reporter

                reporter.emit(
                    ProgressEvent(
                        stage=message["stage"],
                        percent=float(message["percent"]),
                        message=message.get("message"),
                        minutes_done=message.get("minutes_done"),
                        total_minutes=message.get("total_minutes"),
                    )
                )
            elif message.get("kind") == "result":
                result = message
        await asyncio.to_thread(process.join)
        if result is None:
            raise RuntimeError(f"Pipeline terminó sin resultado (exit={process.exitcode})")
        if not result.get("ok"):
            logger.error("[runner] worker traceback\n%s", result.get("traceback", ""))
            raise RuntimeError(result.get("error") or "Pipeline falló")
        return [Path(path) for path in result.get("paths", [])]
    finally:
        with _process_lock:
            if _current_process is process:
                _current_process = None
        result_queue.close()


def start(
    video: Path | str,
    output_dir: Path | None = None,
    *,
    mode: str = "pipeline",
) -> Task[list[Path]] | None:
    global _current_task
    if _current_task and not _current_task.done():
        logger.warning("[runner] intento de iniciar con pipeline en curso")
        return None

    logger.info(
        "[runner] lanzando pipeline",
        extra={"video": str(video), "output": str(output_dir) if output_dir else None, "mode": mode},
    )
    loop = asyncio.get_running_loop()
    _current_task = loop.create_task(run_pipeline(video, output_dir, mode=mode))

    def _done_callback(task: Task[list[Path]]) -> None:
        try:
            result = task.result()
            logger.info("[runner] proceso completado", extra={"clips": len(result)})
        except asyncio.CancelledError:
            logger.info("[runner] tarea cancelada")
        except Exception as exc:
            logger.exception("[runner] error en la pipeline: %s", exc)

    _current_task.add_done_callback(_done_callback)
    return _current_task


def stop() -> None:
    global _current_task, _current_process
    with _process_lock:
        process = _current_process
    if process and process.is_alive():
        logger.info("[runner] terminando proceso de pipeline", extra={"pid": process.pid})
        if os.name == "nt":
            try:
                # The owned spawned worker can itself launch venv Python and
                # FFmpeg children. Address this exact worker's tree only.
                subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    capture_output=True, timeout=10, check=False,
                    creationflags=subprocess.CREATE_NO_WINDOW)
            except (OSError, subprocess.TimeoutExpired):
                logger.warning("[runner] no se pudo cerrar el árbol; intentando cerrar el worker")
        if process.is_alive():
            process.terminate()
        process.join(timeout=5.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=2.0)
    with _process_lock:
        _current_process = None
    if _current_task and not _current_task.done():
        _current_task.cancel()
    _current_task = None
