import json
from pathlib import Path

import pytest

from clipperx.speaker_identity import analyze_speakers, compute_speaker_stats


def test_speaker_ranking_preserves_identity_uncertainty() -> None:
    stats = compute_speaker_stats(
        [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},
            {"start": 20.0, "end": 22.0, "speaker": "SPEAKER_00"},
            {"start": 40.0, "end": 42.0, "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 18.0, "speaker": "SPEAKER_01"},
        ],
        sections=3,
    )

    ranked = sorted(stats, key=lambda item: item.ranking_hint, reverse=True)
    assert ranked[0].speaker == "SPEAKER_00"
    assert ranked[0].section_coverage == 1.0
    assert {item.speaker for item in stats} == {"SPEAKER_00", "SPEAKER_01"}


@pytest.mark.parametrize("payload", [{"turns": [], "exclusive_turns": []}, []])
def test_silence_produces_empty_unresolved_analysis(tmp_path: Path, payload: object) -> None:
    diarization = tmp_path / "diarization.json"
    diarization.write_text(json.dumps(payload), encoding="utf-8")
    output = tmp_path / "analysis.json"

    analyze_speakers(diarization, output)

    analysis = json.loads(output.read_text(encoding="utf-8"))
    assert analysis["identity_status"] == "unresolved"
    assert analysis["candidate"] is None
    assert analysis["speakers"] == []
