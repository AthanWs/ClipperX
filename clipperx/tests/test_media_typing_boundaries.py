"""Bounded behavior guards for media typing seams (no decoders or models)."""

from __future__ import annotations

import importlib
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType
from typing import cast

import numpy as np
import pytest

from clipperx import audio, config, diarization_worker, editing
from clipperx._media_types import FFmpegStream


_MODULE_ROOT = Path(__file__).resolve().parents[1]
_WHISPERX_EXPORTS = ("align", "load_align_model", "load_audio", "load_model")


def _isolated_module(
    monkeypatch: pytest.MonkeyPatch, name: str, source: Path,
) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def _transcription_probe(
    monkeypatch: pytest.MonkeyPatch, whisperx_module: ModuleType | None,
) -> ModuleType:
    native = ModuleType("faster_whisper")
    setattr(native, "WhisperModel", object())
    monkeypatch.setitem(sys.modules, "faster_whisper", native)
    monkeypatch.setitem(sys.modules, "whisperx", whisperx_module)
    return _isolated_module(
        monkeypatch, "clipperx._transcription_probe", _MODULE_ROOT / "transcription.py",
    )


def test_cuda_path_candidates_keep_first_seen_order_and_existing_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "getsitepackages", lambda: ["site"])
    monkeypatch.setattr(config.sys, "prefix", "prefix")
    monkeypatch.setattr(config, "glob", lambda pattern: {
        "site/nvidia/cudnn/bin": ["first", "shared"],
        "site/nvidia/cublas/bin": ["shared", "second"],
        "prefix/nvidia/cudnn/bin": ["first", "third"],
    }.get(pattern.replace("\\", "/"), []))
    monkeypatch.setattr(config.os.path, "isdir", lambda path: path != "second")
    calls: list[str] = []
    monkeypatch.setattr(config.os, "add_dll_directory", calls.append)
    monkeypatch.setenv("PATH", os.pathsep.join(["existing", "shared"]))

    config._add_windows_cuda_dll_paths()

    assert os.environ["PATH"].split(os.pathsep) == [
        "first", "second", "third", "existing", "shared",
    ]
    assert calls == ["first", "shared", "third"]


def test_batch_audio_options_forward_unchanged_and_unknown_options_fail(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    seen: list[tuple[Path, dict[str, object]]] = []

    def preprocess(path: Path, **options: object) -> Path:
        seen.append((path, options))
        return tmp_path / f"{path.stem}.wav"

    monkeypatch.setattr(audio, "preprocess_audio", preprocess)
    inputs = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    options = dict(output_dir=tmp_path, loudnorm=True, channels=2, sample_rate=48_000,
                   start=1.25, end=3.75, output_path=tmp_path / "forced.wav")
    assert audio.batch_preprocess(
        inputs, output_dir=tmp_path, loudnorm=True, channels=2, sample_rate=48_000,
        start=1.25, end=3.75, output_path=tmp_path / "forced.wav",
    ) == [tmp_path / "a.wav", tmp_path / "b.wav"]
    assert seen == [(path, options) for path in inputs]

    monkeypatch.undo()
    with pytest.raises(TypeError, match="unexpected keyword"):
        getattr(audio, "batch_preprocess")([tmp_path / "a.mp4"], unsupported=True)


class _StreamSpy:
    def __init__(self, calls: list[tuple[object, ...]]) -> None:
        self.calls = calls

    def filter(self, name: str, *args: object, **kwargs: object) -> _StreamSpy:
        self.calls.append(("filter", name, args, kwargs))
        return self

    def split(self) -> tuple[_StreamSpy, _StreamSpy]:
        self.calls.append(("split",))
        return self, self


def test_graph_helpers_preserve_filters_and_overlay_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    stream = _StreamSpy(calls)
    typed_stream = cast(FFmpegStream, stream)

    def overlay(a: object, b: object, **kwargs: object) -> _StreamSpy:
        calls.append(("overlay", a, b, kwargs))
        return stream

    monkeypatch.setattr(editing.ffmpeg, "overlay", overlay)

    assert editing._content_without_camera(typed_stream, (0, 120, 520, 420), 1920, 1080) is stream
    assert calls[0] == ("filter", "crop", (1381, 1080, 539, 0), {})
    calls.clear()
    assert editing._compose_cover(typed_stream, 1080, 648) is stream
    assert calls == [
        ("filter", "scale", (1080, 648), {"force_original_aspect_ratio": "increase"}),
        ("filter", "crop", (1080, 648), {}),
    ]
    calls.clear()
    assert editing._compose_contained(typed_stream, 1080, 1920, "#000000") is stream
    assert calls == [
        ("split",),
        ("filter", "scale", (1080, 1920), {"force_original_aspect_ratio": "increase"}),
        ("filter", "crop", (1080, 1920), {}),
        ("filter", "boxblur", (12, 2), {}),
        ("filter", "scale", (1080, 1920), {"force_original_aspect_ratio": "decrease"}),
        ("overlay", stream, stream,
         {"x": "(main_w-overlay_w)/2", "y": "(main_h-overlay_h)/2"}),
    ]
    calls.clear()
    assert editing._compose_camera_panel(typed_stream, 1080, 480, 334, 271) is stream
    assert calls == [
        ("filter", "scale", (1080, 480), {"force_original_aspect_ratio": "increase"}),
        ("filter", "crop", (1080, 480, "(in_w-out_w)/2", "0"), {}),
    ]


def test_whisperx_loader_preserves_model_choice_and_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []
    model = object()
    fake_whisperx = ModuleType("whisperx")
    for name in _WHISPERX_EXPORTS:
        setattr(fake_whisperx, name, object())
    transcription = _transcription_probe(monkeypatch, fake_whisperx)

    def load_model(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return model

    monkeypatch.setattr(transcription, "load_whisperx_model", load_model)
    assert transcription.load_whisperx("chosen") is model
    assert calls == [(('chosen',), {'device': 'cuda', 'compute_type': 'float16'})]


def test_normalized_embedding_keeps_float32_math() -> None:
    source = np.array([[3.0, 4.0], [6.0, 8.0]], dtype=np.float64)
    result = diarization_worker._normalized_embedding(source)
    assert result.dtype == np.float32
    np.testing.assert_array_equal(result, np.array([0.6, 0.8], dtype=np.float32))


def test_optional_import_fallbacks_keep_expected_names(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "mediapipe", None)
    face_module = _isolated_module(monkeypatch, "clipperx._face_probe", _MODULE_ROOT / "face.py")
    assert face_module.__dict__["_MEDIAPIPE_AVAILABLE"] is False
    assert "mp" not in face_module.__dict__

    transcription_module = _transcription_probe(monkeypatch, None)
    assert transcription_module.__dict__["WHISPERX_AVAILABLE"] is False
    assert all(transcription_module.__dict__[name] is None for name in (
        "whisperx_align", "load_align_model", "whisperx_load_audio", "load_whisperx_model",
    ))


@pytest.mark.parametrize("missing", _WHISPERX_EXPORTS)
def test_importable_whisperx_missing_export_selects_native_fallback(
    monkeypatch: pytest.MonkeyPatch, missing: str,
) -> None:
    fake_whisperx = ModuleType("whisperx")
    for name in _WHISPERX_EXPORTS:
        if name != missing:
            setattr(fake_whisperx, name, object())

    module = _transcription_probe(monkeypatch, fake_whisperx)

    assert module.__dict__["WHISPERX_AVAILABLE"] is False
    assert all(module.__dict__[name] is None for name in (
        "whisperx_align", "load_align_model", "whisperx_load_audio", "load_whisperx_model",
    ))


def test_whisperx_present_none_exports_remain_importable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_whisperx = ModuleType("whisperx")
    for name in _WHISPERX_EXPORTS:
        setattr(fake_whisperx, name, None)

    module = _transcription_probe(monkeypatch, fake_whisperx)

    assert module.__dict__["WHISPERX_AVAILABLE"] is True
    assert all(module.__dict__[name] is None for name in (
        "whisperx_align", "load_align_model", "whisperx_load_audio", "load_whisperx_model",
    ))


def test_whisperx_import_time_attribute_error_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import = importlib.import_module

    def import_with_internal_failure(name: str, package: str | None = None) -> ModuleType:
        if name == "whisperx":
            raise AttributeError("internal import failure")
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", import_with_internal_failure)
    with pytest.raises(AttributeError, match="internal import failure"):
        _transcription_probe(monkeypatch, ModuleType("whisperx"))
