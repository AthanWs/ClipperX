"""CPU boundary doubles verify policy and chunk orchestration, not model accuracy."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import soundfile as sf


def test_streaming_reads_bounded_windows_carries_cache_and_flushes_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    worker = importlib.import_module("clipperx.nemotron_worker")
    audio = tmp_path / "audio.flac"
    sf.write(audio, np.arange(3600, dtype=np.float32) / 10000, 16000, subtype="PCM_24")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    calls: list[tuple[int, bool, bool, float]] = []
    cache_inputs: list[object] = []

    class Inputs(dict):
        def to(self, device: object, *, dtype: object) -> Inputs:
            return self

    class Processor:
        feature_extractor = SimpleNamespace(sampling_rate=16000)
        num_samples_first_audio_chunk = 1600
        num_samples_per_audio_chunk = 1600
        num_mel_frames_per_step = 1

        def set_streaming_mode(self, mode: str) -> None:
            assert mode == "low_latency"

        def audio_chunk_start(self, mel_frame_idx: int) -> int:
            return mel_frame_idx * 1280

        def __call__(self, samples: np.ndarray, *, sampling_rate: int,
                     is_streaming: bool, is_first_audio_chunk: bool,
                     is_last_audio_chunk: bool = False) -> Inputs:
            assert sampling_rate == 16000 and is_streaming
            calls.append((len(samples), is_first_audio_chunk, is_last_audio_chunk, float(samples[0])))
            return Inputs(input_features=torch.zeros((1, 2, 8)),
                          attention_mask=torch.ones((1, 2)), num_lookahead_frames=0 if is_last_audio_chunk else 4)

        def extract_speaker_dict(self, logits: object) -> list[list[dict]]:
            assert logits.device.type == "cpu"
            assert tuple(logits.shape) == (1, 23, 8)
            return [[{"Start": 0.0, "End": 0.23, "Speaker": 0},
                     {"Start": 0.05, "End": 0.15, "Speaker": 1}]]

    class Model:
        device = torch.device("cpu")
        dtype = torch.float32

        def eval(self) -> Model:
            return self

        def __call__(self, *, speaker_cache: object, **kwargs: object) -> SimpleNamespace:
            cache_inputs.append(speaker_cache)
            count = len(cache_inputs)
            return SimpleNamespace(logits=torch.zeros((1, 7 if count == 3 else 8, 8)),
                                   speaker_cache=f"cache-{count}")

    def from_processor(path: str, **kwargs: object) -> Processor:
        assert path == str(model_dir.resolve())
        assert kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is False
        return Processor()

    def from_model(path: str, **kwargs: object) -> Model:
        assert path == str(model_dir.resolve())
        assert kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is False
        return Model()

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoProcessor=SimpleNamespace(from_pretrained=from_processor),
        AutoModelForAudioFrameClassification=SimpleNamespace(from_pretrained=from_model),
    ))
    output = tmp_path / "diarization.json"
    worker.run_diarization(audio, output, model_path=model_dir, device="cpu")

    assert [(n, first, last) for n, first, last, _ in calls] == [
        (1600, True, False), (1600, False, False), (1040, False, True),
    ]
    assert [sample for _, _, _, sample in calls] == pytest.approx([0.0, 0.128, 0.256], abs=1e-6)
    assert cache_inputs == [None, "cache-1", "cache-2"]
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["backend"] == "nemotron"
    assert payload["turns"] == [
        {"start": 0.0, "end": 0.225, "speaker": "SPEAKER_00"},
        {"start": 0.05, "end": 0.15, "speaker": "SPEAKER_01"},
    ]
    assert payload["exclusive_turns"] == []
    assert payload["runtime"]["device"] == "cpu"


@pytest.mark.parametrize("minimum,maximum", [(0, None), (None, 9), (3, 2), (True, 2)])
def test_invalid_speaker_bounds_fail_before_loading_models(
    tmp_path: Path, minimum: int | None, maximum: int | None,
) -> None:
    worker = importlib.import_module("clipperx.nemotron_worker")
    with pytest.raises(ValueError, match="speaker|hablante"):
        worker.run_diarization(tmp_path / "missing.flac", tmp_path / "out.json",
                               model_path=tmp_path / "missing-model", min_speakers=minimum,
                               max_speakers=maximum, device="cpu")


def test_forced_speaker_count_is_rejected_instead_of_silently_ignored(tmp_path: Path) -> None:
    worker = importlib.import_module("clipperx.nemotron_worker")
    with pytest.raises(ValueError, match="forzar|unsupported"):
        worker.run_diarization(tmp_path / "missing.flac", tmp_path / "out.json",
                               model_path=tmp_path / "missing-model", min_speakers=2,
                               max_speakers=2, device="cpu")


@pytest.mark.parametrize("turn", [
    {"Start": float("nan"), "End": 0.4, "Speaker": 0},
    {"Start": 0.1, "End": float("inf"), "Speaker": 0},
    {"Start": 0.4, "End": 0.1, "Speaker": 0},
    {"Start": 0.1, "End": 0.4, "Speaker": 8},
])
def test_invalid_model_turns_are_rejected(turn: dict) -> None:
    worker = importlib.import_module("clipperx.nemotron_worker")
    with pytest.raises(ValueError, match="turn|speaker|hablante"):
        worker.normalize_turns([turn], duration=1.0)


@pytest.mark.parametrize("segments,duration", [
    ([{"Start": 0.0, "End": 0.4, "Speaker": 0}], float("nan")),
    ([{"Start": 0.0, "End": 0.4, "Speaker": 0}], float("inf")),
    ([{"Start": "bad", "End": 0.4, "Speaker": 0}], 1.0),
    ([{"Start": 0.0, "End": 0.4}], 1.0),
])
def test_malformed_turn_payloads_raise_validation_error(segments: list[dict], duration: float) -> None:
    worker = importlib.import_module("clipperx.nemotron_worker")
    with pytest.raises(ValueError, match="turn|duración|speaker|hablante"):
        worker.normalize_turns(segments, duration=duration)
