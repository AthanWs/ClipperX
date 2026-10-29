from __future__ import annotations

from pathlib import Path
from fractions import Fraction as F

import numpy as np
import pytest

from clipperx import audio_features
from clipperx.audio_features import AudioFeatureExtractor, AudioWindowFeatures

sf = audio_features.sf


def test_audio_feature_extractor_reuses_identical_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio_path = tmp_path / "audio.flac"
    audio_path.write_bytes(b"not-decoded-by-the-test")
    calls: list[tuple[Path, float, float]] = []

    def fake_extract(path: Path, start: float, end: float) -> AudioWindowFeatures:
        calls.append((path, start, end))
        return AudioWindowFeatures(
            start=start,
            end=end,
            rms=0.1,
            peak=0.2,
            rms_db=-20.0,
            peak_db=-14.0,
            silence_ratio=0.1,
            speech_activity_ratio=0.8,
            energy_variation_db=4.0,
            high_energy_ratio=0.2,
            zero_crossing_rate=0.05,
            clipping_ratio=0.0,
        )

    monkeypatch.setattr(audio_features, "_extract_window", fake_extract)
    extractor = AudioFeatureExtractor(audio_path)

    first = extractor.window(1.0, 2.0)
    second = extractor.window(1.0002, 2.0002)

    assert first is second
    assert calls == [(audio_path, 1.0, 2.0)]


def test_exact_indices_match_complete_legacy_array_features(tmp_path: Path) -> None:
    audio = tmp_path / "exact.wav"
    values = np.sin(np.arange(16000, dtype=np.float32) * .17) * .8
    sf.write(audio, values, 16000, subtype="PCM_16")
    extractor = AudioFeatureExtractor(audio)
    assert hasattr(extractor, "window_samples"), "exact sample feature API absent"
    expected = extractor.window(.125, .625).to_dict()
    actual = extractor.window_samples(2000, 8000, observation_id="a" * 64,
        source_start=F(1, 8), source_end=F(5, 8)).to_dict()
    assert actual == expected


def test_exact_cache_separates_submilliseconds_and_observation_clock(tmp_path: Path) -> None:
    audio = tmp_path / "zeros.wav"
    sf.write(audio, np.zeros(16000, dtype=np.float32), 16000, subtype="PCM_16")
    extractor = AudioFeatureExtractor(audio)
    assert hasattr(extractor, "window_samples"), "exact sample feature API absent"
    first = extractor.window_samples(0, 2, observation_id="a" * 64, source_start=F(0), source_end=F(2, 16000))
    assert first is extractor.window_samples(0, 2, observation_id="a" * 64, source_start=F(0), source_end=F(2, 16000))
    nearby = extractor.window_samples(1, 2, observation_id="a" * 64, source_start=F(1, 16000), source_end=F(3, 16000))
    shifted = extractor.window_samples(0, 2, observation_id="b" * 64, source_start=F(1, 100000), source_end=F(1, 100000) + F(2, 16000))
    assert first is not nearby and first is not shifted
    assert shifted.start != first.start and nearby.start != first.start


@pytest.mark.parametrize("first,count,start,end", [(-1, 1, F(0), F(1, 16000)),
    (0, 0, F(0), F(0)), (True, 1, F(0), F(1, 16000)), (0, 1, F(0), F(2, 16000)),
    (15999, 2, F(0), F(2, 16000))])
def test_exact_features_reject_invalid_or_short_reads(tmp_path: Path, first: int, count: int, start: F, end: F) -> None:
    audio = tmp_path / "short.wav"
    sf.write(audio, np.zeros(16000, dtype=np.float32), 16000, subtype="PCM_16")
    extractor = AudioFeatureExtractor(audio)
    assert hasattr(extractor, "window_samples"), "exact sample feature API absent"
    with pytest.raises(ValueError):
        extractor.window_samples(first, count, observation_id="a" * 64, source_start=start, source_end=end)
