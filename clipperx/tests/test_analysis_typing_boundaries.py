"""Bounded behavior checks for analysis typing boundaries."""

from pathlib import Path
from typing import Sequence, cast

import pytest

from clipperx.analysis import RefineConfig, _audit_boundaries, _safe_float
from clipperx.audio_features import AudioFeatureExtractor
from clipperx.schema import ClipCandidate


@pytest.mark.parametrize("container", [list, tuple])
def test_audit_boundaries_returns_list_with_original_clips(
    container: type[list[ClipCandidate]] | type[tuple[ClipCandidate, ...]],
) -> None:
    clips = [ClipCandidate(start=0.0, end=2.0, text="x", coherence=1.0)]
    input_clips: Sequence[ClipCandidate] = container(clips)
    result = _audit_boundaries(
        Path("unused.mp4"), [], input_clips, cast(AudioFeatureExtractor, object()),
        RefineConfig(max_iterations=0), None,
    )
    assert type(result) is list
    assert len(result) == 1 and result[0] is clips[0]


def test_audit_boundaries_empty_sequence_returns_list() -> None:
    result = _audit_boundaries(
        Path("unused.mp4"), [], (), cast(AudioFeatureExtractor, object()),
        RefineConfig(max_iterations=0), None,
    )
    assert type(result) is list and result == []


class _Convertible:
    def __float__(self) -> float:
        return 2.75


class _BrokenConversion:
    def __float__(self) -> float:
        raise ValueError("conversion failed")


def test_safe_float_preserves_supported_conversion_and_fallback() -> None:
    assert _safe_float("1.25", 9.0) == 1.25
    assert _safe_float(_Convertible(), 9.0) == 2.75
    assert _safe_float(_BrokenConversion(), 9.0) == 9.0
    assert _safe_float(None, 9.0) == 9.0
