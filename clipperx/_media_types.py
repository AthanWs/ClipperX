"""Consumed contracts at the installed ffmpeg-python and soundfile boundaries."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypedDict, cast

if TYPE_CHECKING:
    import numpy as np


class FFmpegNode(Protocol):
    def __getitem__(self, index: int) -> FFmpegStream: ...


class FFmpegStream(Protocol):
    @property
    def video(self) -> FFmpegStream: ...

    @property
    def audio(self) -> FFmpegStream: ...

    @property
    def node(self) -> FFmpegNode: ...

    def filter(self, name: str, *args: object, **kwargs: object) -> FFmpegStream: ...
    def split(self) -> FFmpegNode: ...
    def drawtext(self, **kwargs: object) -> FFmpegStream: ...
    def setpts(self, expression: str) -> FFmpegStream: ...


class FFmpegOutput(Protocol):
    def overwrite_output(self) -> FFmpegOutput: ...
    def run(self, *, quiet: bool, overwrite_output: bool) -> object: ...


class FFmpegProbeStream(TypedDict, total=False):
    codec_type: str
    width: str | int
    height: str | int


class FFmpegProbeResult(TypedDict):
    streams: list[FFmpegProbeStream]


class FFmpegModule(Protocol):
    def input(self, filename: str, **kwargs: object) -> FFmpegStream: ...
    def output(self, *streams_and_filename: FFmpegStream | str, **kwargs: object) -> FFmpegOutput: ...
    def concat(self, *streams: FFmpegStream, **kwargs: object) -> FFmpegStream: ...
    def overlay(self, main: FFmpegStream, overlay: FFmpegStream, **kwargs: object) -> FFmpegStream: ...
    def probe(self, filename: str) -> FFmpegProbeResult: ...
    def run(self, stream: FFmpegOutput, *, quiet: bool, overwrite_output: bool) -> object: ...


class SoundFileHandle(Protocol):
    @property
    def samplerate(self) -> int: ...

    @property
    def frames(self) -> int: ...

    def __enter__(self) -> SoundFileHandle: ...
    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> object: ...
    def seek(self, frames: int) -> int: ...
    def read(self, frames: int, *, dtype: str, always_2d: bool) -> np.ndarray: ...


class SoundFileModule(Protocol):
    __version__: str

    def SoundFile(self, file: str) -> SoundFileHandle: ...
    def read(self, file: str, *, dtype: str, always_2d: bool) -> tuple[np.ndarray, int]: ...
    def write(self, file: str | Path, data: np.ndarray, samplerate: int,
              *, subtype: str) -> None: ...


def load_ffmpeg() -> FFmpegModule:
    """Import eagerly at the caller's existing import point; retain module identity."""
    return cast(FFmpegModule, import_module("ffmpeg"))


def load_soundfile() -> SoundFileModule:
    """Import at the caller's existing eager or lazy import point."""
    return cast(SoundFileModule, import_module("soundfile"))
