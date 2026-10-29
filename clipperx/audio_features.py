"""Audio feature extraction for short windows."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
import re

import numpy as np

from ._media_types import load_soundfile
sf = load_soundfile()

from .render_clock import exact  # noqa: E402 - soundfile remains eager before render_clock


@dataclass(slots=True)
class AudioWindowFeatures:
    start: float
    end: float
    rms: float
    peak: float
    rms_db: float
    peak_db: float
    silence_ratio: float
    speech_activity_ratio: float
    energy_variation_db: float
    high_energy_ratio: float
    zero_crossing_rate: float
    clipping_ratio: float

    def to_dict(self) -> dict[str, float]:
        return {
            "start": self.start,
            "end": self.end,
            "rms": self.rms,
            "peak": self.peak,
            "rms_db": self.rms_db,
            "peak_db": self.peak_db,
            "silence_ratio": self.silence_ratio,
            "speech_activity_ratio": self.speech_activity_ratio,
            "energy_variation_db": self.energy_variation_db,
            "high_energy_ratio": self.high_energy_ratio,
            "zero_crossing_rate": self.zero_crossing_rate,
            "clipping_ratio": self.clipping_ratio,
        }


class AudioFeatureExtractor:
    """Extract lightweight features from short windows with caching."""

    def __init__(self, audio_path: Path) -> None:
        self.audio_path = audio_path
        self._cache: dict[tuple[int, int], AudioWindowFeatures] = {}
        self._sample_cache: dict[tuple[str, int, int, Fraction, Fraction], AudioWindowFeatures] = {}

    def window(self, start: float, end: float) -> AudioWindowFeatures:
        start = max(0.0, float(start))
        end = max(start, float(end))
        key = (int(round(start * 1000)), int(round(end * 1000)))
        cached = self._cache.get(key)
        if cached:
            return cached
        features = _extract_window(self.audio_path, start, end)
        self._cache[key] = features
        return features

    def window_samples(
        self, first: int, count: int, *, observation_id: str,
        source_start: Fraction, source_end: Fraction,
    ) -> AudioWindowFeatures:
        """Read a complete immutable observation slice; never pad or round indices."""
        if type(first) is not int or first < 0 or type(count) is not int or count <= 0:
            raise ValueError("Exact sample indices/count must be nonnegative/positive integers")
        if not isinstance(observation_id, str) or not re.fullmatch(r"[0-9a-f]{64}", observation_id):
            raise ValueError("Exact samples require a receipt observation identity")
        start, end = exact(source_start), exact(source_end)
        key = observation_id, first, count, start, end
        if key in self._sample_cache:
            return self._sample_cache[key]
        with sf.SoundFile(str(self.audio_path)) as handle:
            sr = handle.samplerate
            if start < 0 or end - start != Fraction(count, sr):
                raise ValueError("Included source bounds disagree with count/rate")
            if first + count > handle.frames:
                raise ValueError("Short exact audio read")
            handle.seek(first)
            data = handle.read(count, dtype="float32", always_2d=False)
            if len(data) != count:
                raise ValueError("Short exact audio read")
        features = _sample_array_features(data, sr, float(start), float(end))
        self._sample_cache[key] = features
        return features


def _extract_window(audio_path: Path, start: float, end: float) -> AudioWindowFeatures:
    if end <= start:
        return _empty_features(start, end)

    with sf.SoundFile(str(audio_path)) as handle:
        sr = handle.samplerate
        start_frame = int(start * sr)
        frames = int((end - start) * sr)
        handle.seek(start_frame)
        data = handle.read(frames, dtype="float32", always_2d=False)

    if data.size == 0:
        return _empty_features(start, end)

    return _sample_array_features(data, sr, start, end)


def _sample_array_features(
    data: np.ndarray, sr: int, start: float, end: float,
) -> AudioWindowFeatures:
    """Shared unchanged float32 arithmetic for both legacy and exact reads."""

    if data.ndim > 1:
        data = data.mean(axis=1)

    data = np.asarray(data, dtype=np.float32)
    abs_data = np.abs(data)
    peak = float(abs_data.max()) if abs_data.size else 0.0
    rms = float(np.sqrt(np.mean(data * data))) if data.size else 0.0
    eps = 1e-9
    peak_db = float(20.0 * np.log10(peak + eps))
    rms_db = float(20.0 * np.log10(rms + eps))
    silence_ratio = float(np.mean(abs_data < 0.01)) if abs_data.size else 1.0
    frame_length = max(1, int(sr * 0.02))
    usable = data[: (data.size // frame_length) * frame_length]
    if usable.size:
        frame_rms = np.sqrt(np.mean(usable.reshape(-1, frame_length) ** 2, axis=1))
        frame_db = 20.0 * np.log10(frame_rms + eps)
        noise_floor = float(np.percentile(frame_db, 20))
        activity_threshold = min(-30.0, max(-48.0, noise_floor + 10.0))
        speech_activity_ratio = float(np.mean(frame_db > activity_threshold))
        energy_variation_db = float(np.percentile(frame_db, 90) - np.percentile(frame_db, 10))
        high_threshold = max(-24.0, float(np.median(frame_db)) + 10.0)
        high_energy_ratio = float(np.mean(frame_db >= high_threshold))
    else:
        speech_activity_ratio = 0.0
        energy_variation_db = 0.0
        high_energy_ratio = 0.0
    if data.size > 1:
        zero_crossing_rate = float(np.mean(np.signbit(data[1:]) != np.signbit(data[:-1])))
    else:
        zero_crossing_rate = 0.0
    clipping_ratio = float(np.mean(abs_data >= 0.99)) if abs_data.size else 0.0

    return AudioWindowFeatures(
        start=start,
        end=end,
        rms=rms,
        peak=peak,
        rms_db=rms_db,
        peak_db=peak_db,
        silence_ratio=silence_ratio,
        speech_activity_ratio=speech_activity_ratio,
        energy_variation_db=energy_variation_db,
        high_energy_ratio=high_energy_ratio,
        zero_crossing_rate=zero_crossing_rate,
        clipping_ratio=clipping_ratio,
    )


def _empty_features(start: float, end: float) -> AudioWindowFeatures:
    return AudioWindowFeatures(
        start=start,
        end=end,
        rms=0.0,
        peak=0.0,
        rms_db=-120.0,
        peak_db=-120.0,
        silence_ratio=1.0,
        speech_activity_ratio=0.0,
        energy_variation_db=0.0,
        high_energy_ratio=0.0,
        zero_crossing_rate=0.0,
        clipping_ratio=0.0,
    )
