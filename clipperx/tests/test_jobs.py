from __future__ import annotations

from pathlib import Path
import threading
import time

import pytest

from clipperx.jobs import JobStore


def test_visual_runs_are_unique_and_review_does_not_reuse_success(tmp_path: Path) -> None:
    source = tmp_path / "source.mp4"
    source.write_bytes(b"synthetic")
    job = JobStore.for_video(source, jobs_root=tmp_path / "jobs")
    first = job.new_visual_run("../clip")
    second = job.new_visual_run("../clip")
    assert first != second and first.is_dir() and second.is_dir()
    assert first.resolve().is_relative_to(job.root.resolve())
    job.mark_complete("visual", artifacts=[first / "plan.json"])
    job.mark_review_required("visual", reason="unresolved", receipt=second / "receipt.json")
    assert not job.stage_complete("visual")
    state = job.load()["stages"]["visual"]
    assert state["status"] == "review_required"
    assert "completed_at" not in state
    assert state["artifacts"] == [str(second / "receipt.json")]


def test_job_store_persists_stage_state(tmp_path: Path) -> None:
    source = tmp_path / "stream.mp4"
    source.write_bytes(b"source")
    jobs_root = tmp_path / "jobs"

    job = JobStore.for_video(source, jobs_root=jobs_root)
    artifact = job.root / "artifact.json"
    artifact.write_text("{}", encoding="utf-8")
    job.mark_started("timeline")
    job.mark_complete("timeline", artifacts=[artifact])

    restored = JobStore.for_video(source, jobs_root=jobs_root)
    assert restored.root == job.root
    assert restored.stage_complete("timeline", artifact=artifact)


def test_mark_complete_clears_previous_failure_metadata(tmp_path: Path) -> None:
    source = tmp_path / "stream.mp4"
    source.write_bytes(b"source")
    job = JobStore.for_video(source, jobs_root=tmp_path / "jobs")

    job.mark_failed("timeline", "temporary failure")
    job.mark_complete("timeline")

    stage = job.load()["stages"]["timeline"]
    assert stage["status"] == "complete"
    assert "failed_at" not in stage
    assert "error" not in stage


def test_mark_started_clears_previous_completion_metadata(tmp_path: Path) -> None:
    source = tmp_path / "stream.mp4"
    source.write_bytes(b"source")
    job = JobStore.for_video(source, jobs_root=tmp_path / "jobs")
    artifact = job.root / "old-artifact.json"

    job.mark_complete("timeline", artifacts=[artifact])
    job.mark_started("timeline")

    stage = job.load()["stages"]["timeline"]
    assert stage["status"] == "running"
    assert "completed_at" not in stage
    assert "artifacts" not in stage


def test_job_fingerprint_changes_with_source(tmp_path: Path) -> None:
    source = tmp_path / "stream.mp4"
    source.write_bytes(b"first")
    first = JobStore.for_video(source, jobs_root=tmp_path / "jobs")

    source.write_bytes(b"second-version")
    second = JobStore.for_video(source, jobs_root=tmp_path / "jobs")

    assert first.fingerprint != second.fingerprint
    assert first.root != second.root


def test_job_store_exposes_identity_artifact_paths(tmp_path: Path) -> None:
    source = tmp_path / "stream.mp4"
    source.write_bytes(b"source")
    job = JobStore.for_video(source, jobs_root=tmp_path / "jobs")

    assert job.face_tracks_path == job.root / "face_tracks.json"
    assert job.streamer_identity_path == job.root / "streamer_identity.json"
    assert job.webcam_tracks_path == job.root / "webcam_tracks.json"
    assert job.source_visual_timeline_path == job.root / "source_visual_timeline.json"


def test_job_store_stage_lock_serializes_threads(tmp_path: Path) -> None:
    source = tmp_path / "stream.mp4"
    source.write_bytes(b"source")
    job = JobStore.for_video(source, jobs_root=tmp_path / "jobs")
    entered: list[str] = []
    first_ready = threading.Event()
    release_first = threading.Event()

    def first() -> None:
        with job.stage_lock("source_visual_timeline"):
            entered.append("first")
            first_ready.set()
            release_first.wait(timeout=2.0)

    def second() -> None:
        assert first_ready.wait(timeout=2.0)
        with job.stage_lock("source_visual_timeline"):
            entered.append("second")

    first_thread = threading.Thread(target=first)
    second_thread = threading.Thread(target=second)
    first_thread.start()
    second_thread.start()
    assert first_ready.wait(timeout=2.0)
    time.sleep(0.1)
    assert entered == ["first"]
    release_first.set()
    first_thread.join(timeout=2.0)
    second_thread.join(timeout=2.0)

    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert entered == ["first", "second"]


def test_job_store_concurrent_state_updates_are_not_lost(tmp_path: Path) -> None:
    source = tmp_path / "stream.mp4"
    source.write_bytes(b"source")
    job = JobStore.for_video(source, jobs_root=tmp_path / "jobs")
    barrier = threading.Barrier(3)

    def update(stage: str) -> None:
        barrier.wait(timeout=2.0)
        job.mark_started(stage)

    threads = [threading.Thread(target=update, args=(stage,)) for stage in ("one", "two")]
    for thread in threads:
        thread.start()
    barrier.wait(timeout=2.0)

    for thread in threads:
        thread.join(timeout=2.0)

    assert set(job.load()["stages"]) == {"one", "two"}
    assert list(job.root.glob(".state.*.tmp")) == []

def test_late_initializer_cannot_overwrite_existing_stage_state(tmp_path: Path) -> None:
    source = tmp_path / "stream.mp4"
    source.write_bytes(b"source")
    job = JobStore.for_video(source, jobs_root=tmp_path / "jobs")
    job.mark_started("existing")

    job._ensure_initialized(
        {
            "version": 1,
            "source": str(source),
            "fingerprint": job.fingerprint,
            "created_at": "late",
            "updated_at": "late",
            "stages": {},
        }
    )

    assert "existing" in job.load()["stages"]
def test_observed_paths_are_separate_and_allocation_is_exclusive(tmp_path: Path) -> None:
    from clipperx.jobs import JobStore
    video = tmp_path / "source.mp4"
    video.write_bytes(b"source")
    job = JobStore.for_video(video, jobs_root=tmp_path / "jobs")
    assert hasattr(job, "new_observed_generation"), "observed namespace absent"
    first, second = job.new_observed_generation(), job.new_observed_generation()
    assert first != second and first.parent == second.parent == job.root / "visual_observed" / "generations"
    assert job.source_visual_observed_latest_path == job.root / "visual_observed" / "latest.json"
    assert job.source_visual_timeline_path == job.root / "source_visual_timeline.json"

@pytest.mark.parametrize("binding", [None, "old", "current"])
def test_preferred_transcript_rejects_stale_or_unbound_speaker_text(tmp_path, binding):
    import hashlib
    import json
    source=tmp_path/"video.mp4"
    source.write_bytes(b"source")
    job=JobStore.for_video(source,jobs_root=tmp_path/"jobs")
    job.transcript_path.write_text("new native text")
    job.speaker_transcript_path.write_text("preserved older speaker text")
    if binding is not None:
        value=hashlib.sha256(job.transcript_path.read_bytes()).hexdigest() if binding=="current" else "old"
        job.diarization_path.write_text(json.dumps({"provenance":{"transcript_sha256":value}}))
    assert job.preferred_transcript_path == (job.speaker_transcript_path if binding=="current" else job.transcript_path)
    assert job.speaker_transcript_path.read_text()=="preserved older speaker text"

