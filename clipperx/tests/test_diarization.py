from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from clipperx import diarization
from clipperx.diarization import DiarizationTurn, assign_speakers_to_transcript
from clipperx.jobs import JobStore


def test_assigns_speakers_by_maximum_overlap(tmp_path: Path) -> None:
    transcript_path = tmp_path / "transcript.json"
    output_path = tmp_path / "speakers.json"
    transcript_path.write_text(
        json.dumps(
            [
                {
                    "start": 0.0,
                    "end": 4.0,
                    "text": "hola mundo",
                    "words": [
                        {"start": 0.0, "end": 1.0, "word": "hola"},
                        {"start": 2.0, "end": 4.0, "word": "mundo"},
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    turns = [
        DiarizationTurn(0.0, 1.5, "STREAMER"),
        DiarizationTurn(1.5, 4.0, "GUEST"),
    ]

    assign_speakers_to_transcript(transcript_path, turns, output_path)

    result = json.loads(output_path.read_text(encoding="utf-8"))
    assert result[0]["speaker"] == "GUEST"
    assert result[0]["words"][0]["speaker"] == "STREAMER"
    assert result[0]["words"][1]["speaker"] == "GUEST"


def test_simultaneous_voices_leave_word_identity_ambiguous(tmp_path: Path) -> None:
    transcript = tmp_path / "transcript.json"
    transcript.write_text(json.dumps([{
        "start": 0.0, "end": 2.0, "words": [{"start": 0.5, "end": 1.0, "word": "hola"}],
    }]), encoding="utf-8")
    output = tmp_path / "speakers.json"

    assign_speakers_to_transcript(transcript, [
        DiarizationTurn(0.0, 2.0, "SPEAKER_00"),
        DiarizationTurn(0.7, 1.2, "SPEAKER_01"),
    ], output)

    word = json.loads(output.read_text(encoding="utf-8"))[0]["words"][0]
    assert word["speaker"] is None
    assert word["speaker_candidates"] == ["SPEAKER_00", "SPEAKER_01"]
    assert word["speaker_status"] == "overlap_ambiguous"


def test_overlapping_voice_transcript_round_trips_through_render_reader(tmp_path: Path) -> None:
    from clipperx.plan_render import load_transcript
    transcript = tmp_path / "transcript.json"
    transcript.write_text(json.dumps([{
        "start": 0., "end": 2., "text": "hola", "words": [
            {"start": .5, "end": 1., "word": "hola"}],
    }]), encoding="utf-8")
    output = tmp_path / "speakers.json"
    assign_speakers_to_transcript(transcript, [
        DiarizationTurn(0., 2., "SPEAKER_00"),
        DiarizationTurn(.7, 1.2, "SPEAKER_01"),
    ], output)
    rows = load_transcript(output)
    assert rows[0].words[0].speaker is None
    assert rows[0].words[0].speaker_candidates == ["SPEAKER_00", "SPEAKER_01"]
    assert rows[0].words[0].speaker_status == "overlap_ambiguous"


def _local_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> JobStore:
    import clipperx.stages

    video = tmp_path / "video.mp4"
    video.write_bytes(b"source-preserved")
    job = JobStore.for_video(video, jobs_root=tmp_path / "jobs")
    job.audio_path.write_bytes(b"prepared-audio")
    job.transcript_path.write_text(json.dumps([{
        "start": 0.0, "end": 1.0, "text": "hola", "words": [],
    }]), encoding="utf-8")
    monkeypatch.setattr(clipperx.stages, "ensure_transcript", lambda path: (job, job.transcript_path))
    monkeypatch.setattr(diarization, "settings", SimpleNamespace(
        diarization_backend="nemotron", nemotron_model_path=str(tmp_path / "model"),
        nvidia_speech_python="", hf_token="", diarization_model="pyannote/community-1",
        diarization_python="",
    ))
    return job


def test_nemotron_route_needs_no_hf_token_and_preserves_analysis_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _local_job(tmp_path, monkeypatch)

    def inference(audio_path: Path, *, output_path: Path, **kwargs: object) -> dict:
        assert audio_path == job.audio_path
        payload = {"version": 1, "backend": "nemotron", "turns": [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"start": 0.2, "end": 0.7, "speaker": "SPEAKER_01"},
        ], "exclusive_turns": []}
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(diarization, "_run_nemotron_worker", inference, raising=False)

    paths = diarization.diarize_video(job.source)

    assert paths == (job.diarization_path, job.speaker_transcript_path)
    payload = json.loads(job.diarization_path.read_text(encoding="utf-8"))
    assert len(payload["turns"]) == 2
    assert payload["provenance"]["backend"] == "nemotron"
    analysis = json.loads(job.speaker_analysis_path.read_text(encoding="utf-8"))
    assert analysis["identity_status"] == "unresolved"
    assert {row["speaker"] for row in analysis["speakers"]} == {"SPEAKER_00", "SPEAKER_01"}


def test_old_backend_cache_requires_force_without_touching_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _local_job(tmp_path, monkeypatch)
    job.diarization_path.write_text('{"model":"pyannote/community-1","turns":[]}', encoding="utf-8")
    job.speaker_transcript_path.write_text("[]", encoding="utf-8")
    job.speaker_analysis_path.write_text('{"identity_status":"unresolved"}', encoding="utf-8")
    job.mark_complete("diarization", artifacts=[job.diarization_path, job.speaker_transcript_path])
    original = [path.read_bytes() for path in (
        job.diarization_path, job.speaker_transcript_path, job.speaker_analysis_path,
    )]

    with pytest.raises(RuntimeError, match="force"):
        diarization.diarize_video(job.source)

    assert original == [path.read_bytes() for path in (
        job.diarization_path, job.speaker_transcript_path, job.speaker_analysis_path,
    )]


def test_force_retains_old_artifacts_and_new_cache_tracks_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _local_job(tmp_path, monkeypatch)
    artifacts = (job.diarization_path, job.speaker_transcript_path, job.speaker_analysis_path)
    originals = (b'{"model":"old","turns":[]}', b"[]", b'{"old":true}')
    for path, content in zip(artifacts, originals):
        path.write_bytes(content)

    def inference(audio_path: Path, *, output_path: Path, **kwargs: object) -> dict:
        return {"version": 1, "backend": "nemotron", "turns": [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
        ], "exclusive_turns": []}

    monkeypatch.setattr(diarization, "_run_nemotron_worker", inference, raising=False)
    diarization.diarize_video(job.source, force=True)
    archived = list((job.root / "history" / "diarization").glob("*"))
    assert len(archived) == 1
    assert [archived[0].joinpath(path.name).read_bytes() for path in artifacts] == list(originals)
    diarization.diarize_video(job.source)
    job.transcript_path.write_text("[]", encoding="utf-8")
    with pytest.raises(RuntimeError, match="force"):
        diarization.diarize_video(job.source)


def test_nemotron_subprocess_has_offline_environment_and_finite_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np
    import soundfile as sf

    audio = tmp_path / "audio.flac"
    sf.write(audio, np.zeros(16000, dtype=np.float32), 16000)
    python = tmp_path / "python.exe"
    python.touch()
    model = tmp_path / "model"
    model.mkdir()
    output = tmp_path / "result.json"
    monkeypatch.setattr(diarization, "settings", SimpleNamespace(
        nvidia_speech_python=str(python), nemotron_model_path=str(model),
    ))
    monkeypatch.setenv("HF_TOKEN", "private-parent-token")

    def subprocess_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        assert command[:4] == [str(python), "-m", "clipperx.nemotron_worker", "run"]
        assert command[command.index("--model") + 1] == str(model.resolve())
        environment = kwargs["env"]
        assert environment["HF_HUB_OFFLINE"] == "1"
        assert environment["TRANSFORMERS_OFFLINE"] == "1"
        assert "HF_TOKEN" not in environment
        assert 0 < kwargs["timeout"] <= 86400
        output.write_text('{"backend":"nemotron","turns":[]}', encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(diarization, "run_worker", subprocess_run)
    assert diarization._run_nemotron_worker(
        audio, output_path=output, min_speakers=None, max_speakers=None,
    )["backend"] == "nemotron"


def test_changing_local_model_invalidates_cached_diarization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _local_job(tmp_path, monkeypatch)
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text('{"version":1}', encoding="utf-8")

    def inference(audio_path: Path, **kwargs: object) -> dict:
        return {"turns": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]}

    monkeypatch.setattr(diarization, "_run_nemotron_worker", inference, raising=False)
    diarization.diarize_video(job.source)
    (model / "config.json").write_text('{"version":2,"changed":true}', encoding="utf-8")

    with pytest.raises(RuntimeError, match="force"):
        diarization.diarize_video(job.source)


def test_backend_rejects_invalid_count_before_preparing_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _local_job(tmp_path, monkeypatch)
    import clipperx.stages

    def forbidden_preparation(path: Path) -> None:
        pytest.fail("invalid speaker bounds must fail before ASR preparation")

    monkeypatch.setattr(clipperx.stages, "ensure_transcript", forbidden_preparation)
    with pytest.raises(ValueError, match="speaker|hablante"):
        diarization.diarize_video(tmp_path / "video.mp4", min_speakers=0)


@pytest.mark.parametrize("existing", ["corrupt", "orphan"])
def test_incomplete_cache_cannot_overwrite_existing_evidence_without_force(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: str,
) -> None:
    job = _local_job(tmp_path, monkeypatch)
    if existing == "corrupt":
        protected = job.diarization_path
        protected.write_text("truncated{", encoding="utf-8")
    else:
        protected = job.speaker_analysis_path
        protected.write_text('{"historic":true}', encoding="utf-8")
    original = protected.read_bytes()

    def forbidden_inference(*args: object, **kwargs: object) -> None:
        pytest.fail("existing evidence requires force before inference")

    monkeypatch.setattr(diarization, "_run_nemotron_worker", forbidden_inference)
    with pytest.raises(RuntimeError, match="force"):
        diarization.diarize_video(job.source)
    assert protected.read_bytes() == original


def test_explicit_pyannote_backend_preserves_legacy_worker_and_exclusive_turns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _local_job(tmp_path, monkeypatch)
    diarization.settings.diarization_backend = "pyannote"
    diarization.settings.hf_token = "test-only-token"

    def inference(audio_path: Path, *, token: str, model_name: str,
                  min_speakers: int | None, max_speakers: int | None, **kwargs: object) -> dict:
        assert token == "test-only-token"
        assert model_name == "pyannote/community-1"
        assert (min_speakers, max_speakers) == (2, 3)
        return {"turns": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}],
                "exclusive_turns": [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_01"}]}

    monkeypatch.setattr(diarization, "_run_community_worker", inference)
    diarization.diarize_video(job.source, min_speakers=2, max_speakers=3)

    transcript = json.loads(job.speaker_transcript_path.read_text(encoding="utf-8"))
    assert transcript[0]["speaker"] == "SPEAKER_01"


def test_failed_force_run_keeps_original_artifacts_and_records_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _local_job(tmp_path, monkeypatch)
    artifacts = [job.diarization_path, job.speaker_transcript_path, job.speaker_analysis_path]
    original = [b'{"old":1}', b"[]", b'{"old":2}']
    for path, content in zip(artifacts, original):
        path.write_bytes(content)

    def failure(*args: object, **kwargs: object) -> None:
        raise RuntimeError("model boundary failed")

    monkeypatch.setattr(diarization, "_run_nemotron_worker", failure)
    with pytest.raises(RuntimeError, match="model boundary failed"):
        diarization.diarize_video(job.source, force=True)
    assert [path.read_bytes() for path in artifacts] == original
    state = json.loads(job.state_path.read_text(encoding="utf-8"))
    assert state["stages"]["diarization"]["status"] == "failed"
