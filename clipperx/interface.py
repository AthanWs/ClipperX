"""Minimal Textual-based interface for selecting video and running pipeline."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import Button, Footer, Header, Input, Label, Log, ProgressBar, Select

from .runner import start, stop
from .logger import configure_logging, enable_ui_logging, disable_ui_logging
from .progress import reporter, ProgressEvent, prompts


class SimpleUI(App):
    CSS_PATH = None

    video_path = reactive("", layout=True)
    output_path = reactive("", layout=True)
    log_message = reactive("Pulsa Examinar para elegir rutas y luego Iniciar\n", layout=True)

    def compose(self) -> ComposeResult:
        yield Header()
        with Container():
            yield Label("Modo de procesamiento")
            yield Select(
                [
                    ("Editar clips automáticamente", "auto"),
                    ("Indexar timeline sin LLM", "index"),
                    ("Transcribir y alinear", "transcribe"),
                    ("Pipeline heredado (requiere modelo)", "pipeline"),
                ],
                value="auto",
                id="mode",
            )
            yield Label("Selecciona el video fuente")
            yield Input(placeholder="C:/ruta/video.mp4", id="video")
            yield Button("Examinar...", id="pick_video", variant="primary")
            yield Label("Directorio de salida")
            yield Input(placeholder="C:/salida", id="output")
            yield Button("Examinar...", id="pick_output", variant="primary")
            yield Button("Iniciar", id="start", variant="success")
            yield Button("Detener", id="stop", variant="warning")
            yield ProgressBar(id="progress", total=100)
            yield Log(id="log", highlight=True)
        yield Footer()

    def show_prompt(self, prompt_id: int, message: str, choices: list[str]) -> None:
        # Simple inline prompt in log with buttons; minimal modal
        self._write_ui_log(f"[prompt] {message} ({'/'.join(choices)})")
        # Create ephemeral buttons to capture response
        for choice in choices:
            btn = Button(choice, id=f"prompt_{prompt_id}_{choice}")
            self.mount(btn)
            # Auto-remove after click handled

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id and event.button.id.startswith("prompt_"):
            try:
                _, pid, choice = event.button.id.split("_", 2)
                prompts.respond(int(pid), choice)
                # Remove prompt buttons
                for w in list(self.children):
                    if isinstance(w, Button) and w.id and w.id.startswith(f"prompt_{pid}_"):
                        w.remove()
                self._write_ui_log(f"[prompt] Seleccionado: {choice}")
            except Exception:
                pass
            return
        if event.button.id == "start":
            video_value = self.video_path or self.query_one("#video", Input).value
            output_value = self.output_path or self.query_one("#output", Input).value
            output_dir = Path(output_value) if output_value else None
            mode_value = self.query_one("#mode", Select).value
            mode = str(mode_value) if mode_value is not Select.BLANK else "index"

            if mode == "pipeline":
                from .config import settings

                if settings.llm_provider.lower() == "disabled":
                    self.bell()
                    self._write_ui_log("Selecciona y configura un modelo antes de usar el pipeline heredado.")
                    return

            # Accept both local paths and YouTube URLs
            from .youtube import is_youtube_url
            if is_youtube_url(video_value):
                self._write_ui_log(f"Descargando video de YouTube: {video_value}")
                video_input: Path | str = video_value
            else:
                video = Path(video_value)
                if not video.exists():
                    self.bell()
                    self._write_ui_log("Ruta de video inválida.")
                    return
                video_input = video

            self._write_ui_log(f"Iniciando modo {mode} para {video_value}")
            self._set_controls_enabled(False)
            self._set_progress(5)
            task = start(video_input, output_dir, mode=mode)
            if task is None:
                self._write_ui_log("Ya hay un procesamiento en curso.")
                self._set_controls_enabled(True)
            else:
                task.add_done_callback(self._render_result)
        elif event.button.id == "stop":
            stop()
            self._write_ui_log("Pipeline detenida.")
            self._set_controls_enabled(True)
        elif event.button.id == "pick_video":
            selection = self._ask_file()
            if selection:
                self.video_path = selection
                self.query_one("#video", Input).value = selection
                self._write_ui_log(f"Video seleccionado: {selection}")
        elif event.button.id == "pick_output":
            selection = self._ask_directory()
            if selection:
                self.output_path = selection
                self.query_one("#output", Input).value = selection
                self._write_ui_log(f"Directorio seleccionado: {selection}")

    def _ask_file(self) -> str | None:
        try:
            from tkinter import Tk, filedialog
        except Exception:  # pragma: no cover
            self.bell()
            return None
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        filename = filedialog.askopenfilename(
            title="Selecciona el video",
            filetypes=[
                ("Videos", "*.mp4 *.mov *.mkv *.m4v *.avi"),
                ("Todos los archivos", "*.*"),
            ],
        )
        root.destroy()
        return filename or None

    def _ask_directory(self) -> str | None:
        try:
            from tkinter import Tk, filedialog
        except Exception:  # pragma: no cover
            self.bell()
            return None
        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        directory = filedialog.askdirectory(title="Selecciona el directorio de salida")
        root.destroy()
        return directory or None

    def _write_ui_log(self, message: str) -> None:
        log_widget = self.query_one(Log)
        log_widget.write(message)

    def _set_progress(self, value: int) -> None:
        try:
            bar = self.query_one("#progress", ProgressBar)
            bar.progress = max(0, min(100, value))
        except Exception:
            pass

    def _set_controls_enabled(self, enabled: bool) -> None:
        try:
            self.query_one("#start", Button).disabled = not enabled
            self.query_one("#pick_video", Button).disabled = not enabled
            self.query_one("#pick_output", Button).disabled = not enabled
            # Stop is the inverse: enabled only when running
            self.query_one("#stop", Button).disabled = enabled
        except Exception:
            pass

    def _render_result(self, task: asyncio.Task[list[Path]]) -> None:
        try:
            result = task.result()
            summary_path = next((path for path in result if path.name == "autonomous-summary.json"), None)
            status = None
            if summary_path is not None:
                try:
                    status = json.loads(summary_path.read_text(encoding="utf-8"))["status"]
                except (OSError, ValueError, KeyError, TypeError):
                    status = "unknown"
                labels = {
                    "rendered_unreviewed": "Clips exportados; revisión audiovisual pendiente",
                    "partial": "Proceso parcial: quedan partes pendientes",
                    "review_required": "Hay momentos que requieren revisión",
                    "budget_exhausted": "Se agotó el presupuesto de IA; el progreso está guardado",
                    "empty": "No se encontraron clips para exportar",
                    "error": "El proceso terminó con un error",
                    "unknown": "No se pudo confirmar el estado del proceso",
                }
                self._write_ui_log(labels.get(status, "Estado de la operación no reconocido"))
            else:
                self._write_ui_log(f"Proceso completado. Artefactos generados: {len(result)}")
            for path in result:
                self._write_ui_log(f" - {path}")
            if status in (None, "rendered_unreviewed", "empty"):
                self._set_progress(100)
        except Exception as exc:  # pragma: no cover
            self._write_ui_log(f"Error en la pipeline: {exc}")
        finally:
            self._set_controls_enabled(True)

    def on_mount(self) -> None:
        # Configure logging for UI mode: disable console output, route to UI log
        configure_logging(use_console=False)
        enable_ui_logging(lambda msg: self._write_ui_log(msg))
        self._set_progress(0)
        self._set_controls_enabled(True)
        reporter.subscribe(self._on_progress)

    def on_unmount(self) -> None:
        try:
            reporter.unsubscribe(self._on_progress)
        finally:
            disable_ui_logging()

    def _on_progress(self, event: ProgressEvent) -> None:
        # call_later uses Textual's thread-safe message queue from either thread.
        try:
            percent = int(round(event.percent))
            self.call_later(self._set_progress, percent)
            if event.message:
                self.call_later(self._write_ui_log, f"[{event.stage}] {event.message}")
            if event.stage == "prompt" and event.prompt_id and event.choices:
                self.call_later(self.show_prompt, event.prompt_id, event.message or "", list(event.choices))
        except Exception:
            pass


def run_ui() -> None:
    SimpleUI().run()


