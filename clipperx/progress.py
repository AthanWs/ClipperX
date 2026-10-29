"""Thread-safe progress reporting with UI subscriptions.

This module provides a singleton `reporter` that components can use to emit
progress updates. The UI can subscribe to receive updates and reflect them in
widgets without interfering with the main pipeline execution.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from math import ceil, floor
from typing import Callable, Dict, List, Optional


@dataclass(slots=True)
class ProgressEvent:
    stage: str
    percent: float
    message: str | None = None
    minutes_done: int | None = None
    total_minutes: int | None = None
    # Optional interactive prompt support
    prompt_id: int | None = None
    choices: List[str] | None = None


class ProgressReporter:
    def __init__(self) -> None:
        self._subscribers: list[Callable[[ProgressEvent], None]] = []
        self._lock = threading.Lock()

    def subscribe(self, fn: Callable[[ProgressEvent], None]) -> None:
        with self._lock:
            if fn not in self._subscribers:
                self._subscribers.append(fn)

    def unsubscribe(self, fn: Callable[[ProgressEvent], None]) -> None:
        with self._lock:
            if fn in self._subscribers:
                self._subscribers.remove(fn)

    def emit(self, event: ProgressEvent) -> None:
        with self._lock:
            subscribers: List[Callable[[ProgressEvent], None]] = list(self._subscribers)
        for fn in subscribers:
            try:
                fn(event)
            except Exception:
                # Never raise from UI callbacks
                pass

    def set_percent(self, stage: str, percent: float, message: str | None = None) -> None:
        event = ProgressEvent(stage=stage, percent=max(0.0, min(100.0, percent)), message=message)
        self.emit(event)

    def request_choice(self, message: str, choices: List[str]) -> str:
        """Request a user choice via UI and block until response.

        This is used for graceful, interactive retries (e.g., when Ollama is down).
        """
        prompt = prompts.create(message, choices)
        event = ProgressEvent(
            stage="prompt",
            percent=0.0,
            message=message,
            prompt_id=prompt.id,
            choices=choices,
        )
        self.emit(event)
        return prompts.wait(prompt.id)


class TranscriptionEstimator:
    """Background estimator that publishes approximate per-minute progress.

    It estimates completion time based on audio duration and a real-time factor
    (RTF). It updates the percent between a start and end bound so callers can
    map the long-running transcription to a segment of the global progress bar.
    """

    def __init__(
        self,
        *,
        duration_seconds: float,
        rtf: float = 0.6,
        start_percent: float = 20.0,
        end_percent: float = 55.0,
        tick_seconds: float = 1.0,
    ) -> None:
        self.duration_seconds = max(1.0, duration_seconds)
        self.rtf = max(0.01, rtf)
        self.start_percent = start_percent
        self.end_percent = end_percent
        self.tick_seconds = tick_seconds
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="TranscriptionEstimator", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    def _run(self) -> None:
        total_estimated_time = self.duration_seconds / self.rtf
        start_time = time.perf_counter()
        total_minutes = ceil(self.duration_seconds / 60.0)
        last_reported_minute = -1
        while not self._stop.is_set():
            elapsed = time.perf_counter() - start_time
            est_processed_seconds = min(self.duration_seconds, elapsed * self.rtf)
            minute_done = floor(est_processed_seconds / 60.0)
            # Linear interpolation of percent within bounds
            ratio = min(1.0, elapsed / total_estimated_time)
            percent = self.start_percent + (self.end_percent - self.start_percent) * ratio
            if minute_done != last_reported_minute:
                last_reported_minute = minute_done
                reporter.emit(
                    ProgressEvent(
                        stage="transcription",
                        percent=percent,
                        message=f"~{minute_done}/{total_minutes} min",
                        minutes_done=minute_done,
                        total_minutes=total_minutes,
                    )
                )
            else:
                reporter.set_percent("transcription", percent)
            if ratio >= 1.0:
                break
            self._stop.wait(self.tick_seconds)


reporter = ProgressReporter()


class _Prompt:
    def __init__(self, prompt_id: int, message: str, choices: List[str]) -> None:
        self.id = prompt_id
        self.message = message
        self.choices = choices
        self._event = threading.Event()
        self._response: Optional[str] = None

    def respond(self, choice: str) -> None:
        if choice not in self.choices:
            return
        self._response = choice
        self._event.set()

    def wait(self, timeout: Optional[float] = None) -> str:
        self._event.wait(timeout)
        return self._response or (self.choices[0] if self.choices else "")


class PromptManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_id = 1
        self._prompts: Dict[int, _Prompt] = {}

    def create(self, message: str, choices: List[str]) -> _Prompt:
        with self._lock:
            pid = self._next_id
            self._next_id += 1
            p = _Prompt(pid, message, choices)
            self._prompts[pid] = p
            return p

    def respond(self, prompt_id: int, choice: str) -> None:
        with self._lock:
            p = self._prompts.get(prompt_id)
        if p is not None:
            p.respond(choice)

    def wait(self, prompt_id: int, timeout: Optional[float] = None) -> str:
        with self._lock:
            p = self._prompts.get(prompt_id)
        if p is None:
            return ""
        result = p.wait(timeout)
        with self._lock:
            self._prompts.pop(prompt_id, None)
        return result


prompts = PromptManager()


