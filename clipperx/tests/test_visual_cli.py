from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
from typer.testing import CliRunner

from clipperx.cli import app


runner = CliRunner()


def _write_timeline(path: Path) -> Path:
    payload = {
        "contract_version": 1,
        "source_fingerprint": "synthetic-source",
        "duration_seconds": 600.0,
        "analyzer_version": "visual-v1",
        "model_hashes": {"yunet": "local-only", "sface": "local-only"},
        "sampling_config": {
            "sparse_interval": 10.0,
            "dense_interval": 0.5,
        },
        "samples": [
            {
                "sample_id": "sample_10",
                "at": 10.0,
                "histogram_change_score": 0.1,
                "structural_change_score": 0.1,
                "text_density": 0.0,
                "brightness": 0.5,
                "sharpness": 1.0,
                "motion": 0.1,
                "evidence_ids": ["sample_10"],
            },
            {
                "sample_id": "sample_20",
                "at": 20.0,
                "histogram_change_score": 0.2,
                "structural_change_score": 0.2,
                "text_density": 0.1,
                "face_track_ids": ["face_streamer"],
                "webcam_track_id": "webcam_main",
                "webcam_rect": {
                    "x": 0.0,
                    "y": 0.0,
                    "width": 0.25,
                    "height": 0.3,
                },
                "brightness": 0.5,
                "sharpness": 1.0,
                "motion": 0.2,
                "audio_window": {"start": 19.0, "end": 21.0},
                "transcript_window": {"start": 19.0, "end": 21.0},
                "evidence_ids": ["sample_20"],
            },
            {
                "sample_id": "sample_30",
                "at": 30.0,
                "histogram_change_score": 0.1,
                "structural_change_score": 0.1,
                "text_density": 0.0,
                "brightness": 0.5,
                "sharpness": 1.0,
                "motion": 0.1,
                "evidence_ids": ["sample_30"],
            },
        ],
        "spans": [
            {
                "span_id": "span_before",
                "source": {"start": 0.0, "end": 15.0},
                "state": "content_only",
                "confidence": 0.8,
                "streamer_track_id": None,
                "identity_status": "unavailable",
                "webcam_track_id": None,
                "important_regions": [],
                "text_regions": [],
                "evidence_ids": ["span_before"],
                "change_reasons": [],
            },
            {
                "span_id": "span_selected",
                "source": {"start": 15.0, "end": 25.0},
                "state": "content_with_camera",
                "confidence": 0.9,
                "streamer_track_id": "face_streamer",
                "identity_status": "confident",
                "webcam_track_id": "webcam_main",
                "important_regions": [],
                "text_regions": [],
                "evidence_ids": ["span_selected"],
                "change_reasons": ["webcam_appeared"],
            },
            {
                "span_id": "span_after",
                "source": {"start": 25.0, "end": 40.0},
                "state": "content_only",
                "confidence": 0.8,
                "streamer_track_id": None,
                "identity_status": "unavailable",
                "webcam_track_id": None,
                "important_regions": [],
                "text_regions": [],
                "evidence_ids": ["span_after"],
                "change_reasons": [],
            },
        ],
        "face_tracks": [
            {
                "track_id": "face_streamer",
                "first_seen": 10.0,
                "last_seen": 30.0,
                "observation_count": 3,
                "embedding_centroid": [1.0, 0.0],
                "positions": [
                    {
                        "x": 0.0,
                        "y": 0.0,
                        "width": 0.25,
                        "height": 0.3,
                    }
                ],
                "recurrent_states": ["content_with_camera"],
                "full_camera_seconds": 0.0,
                "mean_confidence": 0.9,
                "evidence_ids": ["face_streamer"],
            }
        ],
        "webcam_tracks": [
            {
                "track_id": "webcam_main",
                "source_ranges": [{"start": 10.0, "end": 30.0}],
                "observations": [
                    {
                        "at": 10.0,
                        "rect": {
                            "x": 0.0,
                            "y": 0.0,
                            "width": 0.25,
                            "height": 0.3,
                        },
                        "confidence": 0.9,
                        "evidence_id": "webcam_10",
                    },
                    {
                        "at": 20.0,
                        "rect": {
                            "x": 0.0,
                            "y": 0.0,
                            "width": 0.25,
                            "height": 0.3,
                        },
                        "confidence": 0.9,
                        "evidence_id": "webcam_20",
                    },
                    {
                        "at": 30.0,
                        "rect": {
                            "x": 0.0,
                            "y": 0.0,
                            "width": 0.25,
                            "height": 0.3,
                        },
                        "confidence": 0.9,
                        "evidence_id": "webcam_30",
                    },
                ],
                "face_track_ids": ["face_streamer"],
                "confidence": 0.9,
                "evidence_ids": ["webcam_main"],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_enrollment_timeline(
    path: Path,
    *,
    tracks: tuple[tuple[str, list[float], int], ...],
) -> Path:
    payload = {
        "contract_version": 1,
        "source_fingerprint": "synthetic-source",
        "duration_seconds": 60.0,
        "analyzer_version": "visual-v1",
        "model_hashes": {
            "yunet": "a" * 64,
            "sface": "b" * 64,
        },
        "sampling_config": {
            "inspection_start_seconds": 10.0,
            "inspection_end_seconds": 20.0,
            "inspection_interval_seconds": 0.5,
        },
        "samples": [],
        "spans": [],
        "face_tracks": [
            {
                "track_id": track_id,
                "first_seen": 10.0,
                "last_seen": 20.0,
                "observation_count": observations,
                "embedding_centroid": embedding,
                "positions": [
                    {
                        "x": 0.42,
                        "y": 0.30,
                        "width": 0.16,
                        "height": 0.25,
                    }
                ],
                "recurrent_states": ["content_only"],
                "full_camera_seconds": 0.0,
                "mean_confidence": 0.95,
                "evidence_ids": [f"evidence:{track_id}"],
            }
            for track_id, embedding, observations in tracks
        ],
        "webcam_tracks": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_index_visual_does_not_require_provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"source")
    timeline = _write_timeline(tmp_path / "source_visual_timeline.json")
    captured: dict[str, object] = {}

    def fake_builder(video_path: Path, **kwargs: object) -> Path:
        captured["video_path"] = video_path
        captured.update(kwargs)
        return timeline

    monkeypatch.setattr(
        "clipperx.visual_timeline.build_source_visual_timeline",
        fake_builder,
    )

    result = runner.invoke(app, ["index-visual", str(video)])

    assert result.exit_code == 0, result.output
    assert "Source visual timeline:" in result.stdout
    assert captured["allow_model_download"] is False
    assert captured["sparse_interval"] == 10.0
    assert captured["dense_interval"] == 0.5
    assert captured["force"] is False


def test_index_visual_forwards_only_explicit_local_options(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"source")
    timeline = _write_timeline(tmp_path / "source_visual_timeline.json")
    captured: dict[str, object] = {}

    def fake_builder(video_path: Path, **kwargs: object) -> Path:
        captured.update(kwargs)
        return timeline

    monkeypatch.setattr(
        "clipperx.visual_timeline.build_source_visual_timeline",
        fake_builder,
    )

    result = runner.invoke(
        app,
        [
            "index-visual",
            str(video),
            "--sparse-interval",
            "8",
            "--dense-interval",
            "0.25",
            "--allow-model-download",
            "--force",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "sparse_interval": 8.0,
        "dense_interval": 0.25,
        "allow_model_download": True,
        "force": True,
    }


def test_inspect_visual_rejects_invalid_or_oversized_ranges(tmp_path: Path) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"source")

    reversed_range = runner.invoke(
        app,
        ["inspect-visual", str(video), "--start", "20", "--end", "20"],
    )
    oversized_range = runner.invoke(
        app,
        ["inspect-visual", str(video), "--start", "0", "--end", "300.01"],
    )

    assert reversed_range.exit_code != 0
    assert "greater than --start" in reversed_range.output
    assert oversized_range.exit_code != 0
    assert "300 seconds" in oversized_range.output


def test_inspect_visual_emits_valid_standalone_source_time_subset(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"source")
    timeline_path = _write_timeline(tmp_path / "source_visual_timeline.json")
    captured: dict[str, object] = {}

    def fake_window_builder(video_path: Path, **kwargs: object) -> Path:
        captured["video_path"] = video_path
        captured.update(kwargs)
        return timeline_path

    monkeypatch.setattr(
        "clipperx.visual_timeline.build_source_visual_timeline_window",
        fake_window_builder,
        raising=False,
    )
    monkeypatch.setattr(
        "clipperx.visual_timeline.build_source_visual_timeline",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("inspect-visual must not call the canonical builder")
        ),
    )

    result = runner.invoke(
        app,
        [
            "inspect-visual",
            str(video),
            "--start",
            "18",
            "--end",
            "22",
            "--sample-interval",
            "0.5",
        ],
    )

    assert result.exit_code == 0, result.output
    output_path = Path(result.output.strip().split(": ", 1)[1])
    assert output_path == timeline_path
    assert captured == {
        "video_path": video.resolve(),
        "start": 18.0,
        "end": 22.0,
        "sample_interval": 0.5,
        "allow_model_download": False,
    }


def test_inspect_visual_reports_out_of_duration_instead_of_empty_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"source")

    def reject_window(*args: object, **kwargs: object) -> Path:
        raise ValueError("--end 61.0 exceeds source duration 60.0")

    monkeypatch.setattr(
        "clipperx.visual_timeline.build_source_visual_timeline_window",
        reject_window,
        raising=False,
    )

    result = runner.invoke(
        app,
        [
            "inspect-visual",
            str(video),
            "--start",
            "59",
            "--end",
            "61",
        ],
    )

    assert result.exit_code != 0
    assert "source duration 60.0" in result.output


def test_export_visual_schemas_writes_provider_free_contracts(tmp_path: Path) -> None:
    output_dir = tmp_path / "schemas"

    result = runner.invoke(
        app,
        ["export-visual-schemas", "--output-dir", str(output_dir)],
    )

    assert result.exit_code == 0, result.output
    source_schema = output_dir / "SourceVisualTimeline.schema.json"
    plan_schema = output_dir / "VisualCompositionPlan.schema.json"
    assert source_schema.is_file()
    assert plan_schema.is_file()
    assert json.loads(source_schema.read_text(encoding="utf-8"))["title"] == (
        "SourceVisualTimeline"
    )
    assert "Schema:" in result.output


def test_export_visual_schemas_preserves_existing_target_when_replace_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output_dir = tmp_path / "schemas"
    output_dir.mkdir()
    source_schema = output_dir / "SourceVisualTimeline.schema.json"
    source_schema.write_text("previous-valid-schema", encoding="utf-8")
    original_replace = Path.replace

    def fail_source_replace(path: Path, target: Path) -> Path:
        if Path(target) == source_schema:
            raise OSError("simulated atomic replace failure")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_source_replace)

    result = runner.invoke(
        app,
        ["export-visual-schemas", "--output-dir", str(output_dir)],
    )

    assert result.exit_code != 0
    assert source_schema.read_text(encoding="utf-8") == "previous-valid-schema"
    assert not list(output_dir.glob(".*.tmp"))


def test_visual_cli_paths_work_when_env_settings_instantiation_fails(
    tmp_path: Path,
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"source")
    artifact = _write_timeline(tmp_path / "isolated.json")
    schema_dir = tmp_path / "schemas"
    script = textwrap.dedent(
        f"""
        from pathlib import Path
        from pydantic_settings import BaseSettings
        from typer.testing import CliRunner

        def fail_settings(self, *args, **kwargs):
            raise AssertionError("EnvSettings must not be instantiated")

        BaseSettings.__init__ = fail_settings

        import clipperx.cli as cli
        import clipperx.local_paths as local_paths
        import clipperx.visual_timeline as visual_timeline

        artifact = Path({str(artifact)!r})
        local_paths.paths.profiles_dir = artifact.parent / "profiles"
        local_paths.paths.jobs_dir = artifact.parent / "jobs"
        visual_timeline.build_source_visual_timeline = lambda *args, **kwargs: artifact
        visual_timeline.build_source_visual_timeline_window = lambda *args, **kwargs: artifact
        runner = CliRunner()
        commands = (
            ["index-visual", {str(video)!r}],
            ["inspect-visual", {str(video)!r}, "--start", "1", "--end", "2"],
            ["export-visual-schemas", "--output-dir", {str(schema_dir)!r}],
            ["enroll-streamer-face", {str(video)!r}, "--start", "1", "--end", "2"],
        )
        for command in commands:
            result = runner.invoke(cli.app, command)
            if result.exit_code != 0:
                raise AssertionError(result.output) from result.exception
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr or result.stdout


def test_atomic_text_writer_uses_unique_temporaries_under_concurrency(
    tmp_path: Path,
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from clipperx.local_paths import atomic_write_text

    target = tmp_path / "artifact.json"
    payloads = [
        f'{{"writer": {index}, "data": "' + ("x" * 2000) + '"}}'
        for index in range(12)
    ]

    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(lambda payload: atomic_write_text(target, payload), payloads))

    assert target.read_text(encoding="utf-8") in payloads
    assert not list(tmp_path.glob(".*.tmp"))


def _prepare_enrollment_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[Path, Path]:
    from clipperx.local_paths import paths

    profiles_dir = tmp_path / "profiles"
    jobs_dir = tmp_path / "jobs"
    monkeypatch.setattr(paths, "profiles_dir", profiles_dir)
    monkeypatch.setattr(paths, "jobs_dir", jobs_dir)
    return profiles_dir, jobs_dir


def test_enroll_streamer_face_auto_selects_one_serializable_track(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from clipperx.local_paths import source_fingerprint, source_job_root

    video = tmp_path / "source.mp4"
    video.write_bytes(b"source")
    timeline_path = _write_enrollment_timeline(
        tmp_path / "enrollment.json",
        tracks=(("face_001", [3.0, 4.0], 7),),
    )
    profiles_dir, _ = _prepare_enrollment_paths(monkeypatch, tmp_path)
    captured: dict[str, object] = {}

    def fake_window_builder(video_path: Path, **kwargs: object) -> Path:
        captured["video_path"] = video_path
        captured.update(kwargs)
        return timeline_path

    monkeypatch.setattr(
        "clipperx.visual_timeline.build_source_visual_timeline_window",
        fake_window_builder,
    )

    result = runner.invoke(
        app,
        [
            "enroll-streamer-face",
            str(video),
            "--start",
            "10",
            "--end",
            "20",
            "--sample-interval",
            "1",
            "--name",
            "Athan W",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "video_path": video.resolve(),
        "start": 10.0,
        "end": 20.0,
        "sample_interval": 1.0,
        "allow_model_download": False,
    }
    global_profile = profiles_dir / "streamer_identity.json"
    job_profile = source_job_root(video) / "streamer_identity.json"
    assert global_profile.is_file()
    assert job_profile.is_file()
    payload = json.loads(global_profile.read_text(encoding="utf-8"))
    assert payload == json.loads(job_profile.read_text(encoding="utf-8"))
    assert payload["contract_version"] == 1
    assert payload["profile_type"] == "streamer_face"
    assert payload["name"] == "Athan W"
    assert payload["track_id"] == "face_001"
    assert payload["embedding"] == [0.6, 0.8]
    assert payload["source_fingerprint"] == source_fingerprint(video)
    assert payload["source_reference"] == str(video.resolve())
    assert payload["source_range"] == {"start": 10.0, "end": 20.0}
    assert payload["observations"] == 7
    assert payload["model_hashes"] == {"yunet": "a" * 64, "sface": "b" * 64}
    assert payload["created_at"].endswith("Z")


def test_enroll_streamer_face_rejects_zero_tracks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"source")
    timeline_path = _write_enrollment_timeline(
        tmp_path / "empty.json",
        tracks=(),
    )
    _prepare_enrollment_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "clipperx.visual_timeline.build_source_visual_timeline_window",
        lambda *args, **kwargs: timeline_path,
    )

    result = runner.invoke(
        app,
        ["enroll-streamer-face", str(video), "--start", "10", "--end", "20"],
    )

    assert result.exit_code != 0
    assert "no serializable face tracks" in result.output.lower()


def test_enroll_streamer_face_requires_valid_track_id_for_multiple_tracks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"source")
    timeline_path = _write_enrollment_timeline(
        tmp_path / "multiple.json",
        tracks=(
            ("face_001", [1.0, 0.0], 4),
            ("face_002", [0.0, 1.0], 9),
        ),
    )
    profiles_dir, _ = _prepare_enrollment_paths(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "clipperx.visual_timeline.build_source_visual_timeline_window",
        lambda *args, **kwargs: timeline_path,
    )

    missing = runner.invoke(
        app,
        ["enroll-streamer-face", str(video), "--start", "10", "--end", "20"],
    )
    invalid = runner.invoke(
        app,
        [
            "enroll-streamer-face",
            str(video),
            "--start",
            "10",
            "--end",
            "20",
            "--track-id",
            "face_999",
        ],
    )
    selected = runner.invoke(
        app,
        [
            "enroll-streamer-face",
            str(video),
            "--start",
            "10",
            "--end",
            "20",
            "--track-id",
            "face_002",
        ],
    )

    assert missing.exit_code != 0
    assert "--track-id" in missing.output
    assert invalid.exit_code != 0
    assert "face_999" in invalid.output
    assert selected.exit_code == 0, selected.output
    payload = json.loads(
        (profiles_dir / "streamer_identity.json").read_text(encoding="utf-8")
    )
    assert payload["track_id"] == "face_002"
    assert payload["embedding"] == [0.0, 1.0]


def test_enroll_streamer_face_preserves_previous_profile_when_atomic_replace_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    video = tmp_path / "source.mp4"
    video.write_bytes(b"source")
    timeline_path = _write_enrollment_timeline(
        tmp_path / "enrollment.json",
        tracks=(("face_001", [1.0, 0.0], 5),),
    )
    profiles_dir, _ = _prepare_enrollment_paths(monkeypatch, tmp_path)
    profiles_dir.mkdir(parents=True)
    target = profiles_dir / "streamer_identity.json"
    target.write_text('{"previous": true}', encoding="utf-8")
    original_replace = Path.replace

    def fail_profile_replace(path: Path, destination: Path) -> Path:
        if Path(destination) == target:
            raise OSError("simulated profile replace failure")
        return original_replace(path, destination)

    monkeypatch.setattr(Path, "replace", fail_profile_replace)
    monkeypatch.setattr(
        "clipperx.visual_timeline.build_source_visual_timeline_window",
        lambda *args, **kwargs: timeline_path,
    )

    result = runner.invoke(
        app,
        ["enroll-streamer-face", str(video), "--start", "10", "--end", "20"],
    )

    assert isinstance(result.exception, OSError)
    assert "simulated profile replace failure" in str(result.exception)
    assert result.exit_code != 0
    assert target.read_text(encoding="utf-8") == '{"previous": true}'
    assert not list(profiles_dir.glob(".*.tmp"))


def test_enroll_streamer_face_rolls_back_global_when_job_profile_write_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from clipperx.local_paths import source_job_root

    video = tmp_path / "source.mp4"
    video.write_bytes(b"source")
    timeline_path = _write_enrollment_timeline(
        tmp_path / "enrollment.json",
        tracks=(("face_001", [1.0, 0.0], 5),),
    )
    profiles_dir, _ = _prepare_enrollment_paths(monkeypatch, tmp_path)
    global_profile = profiles_dir / "streamer_identity.json"
    job_profile = source_job_root(video) / "streamer_identity.json"
    global_profile.parent.mkdir(parents=True)
    job_profile.parent.mkdir(parents=True)
    global_profile.write_text('{"global": "previous"}', encoding="utf-8")
    job_profile.write_text('{"job": "previous"}', encoding="utf-8")
    original_replace = Path.replace
    failed_once = False

    def fail_first_job_replace(path: Path, destination: Path) -> Path:
        nonlocal failed_once
        if Path(destination) == job_profile and not failed_once:
            failed_once = True
            raise OSError("simulated second profile replace failure")
        return original_replace(path, destination)

    monkeypatch.setattr(Path, "replace", fail_first_job_replace)
    monkeypatch.setattr(
        "clipperx.visual_timeline.build_source_visual_timeline_window",
        lambda *args, **kwargs: timeline_path,
    )

    result = runner.invoke(
        app,
        ["enroll-streamer-face", str(video), "--start", "10", "--end", "20"],
    )

    assert isinstance(result.exception, OSError)
    assert "simulated second profile replace failure" in str(result.exception)
    assert global_profile.read_text(encoding="utf-8") == '{"global": "previous"}'
