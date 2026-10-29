"""Persistent per-source job state and artifact paths."""

from __future__ import annotations

import json
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import tempfile
import time
from typing import Any, BinaryIO
from types import TracebackType
from uuid import uuid4

from .local_paths import safe_slug as _safe_slug
from .local_paths import source_fingerprint as _source_fingerprint
from .local_paths import source_job_root
from .local_paths import paths as _local_paths


class _InterProcessFileLock:
    """Small blocking OS lock used for state and per-stage serialization."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stream: BinaryIO | None = None

    def __enter__(self) -> "_InterProcessFileLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("a+b")
        self._stream.seek(0, os.SEEK_END)
        if self._stream.tell() == 0:
            self._stream.write(b"\0")
            self._stream.flush()
        self._stream.seek(0)
        if os.name == "nt":
            import msvcrt

            deadline = time.monotonic() + 300.0
            while True:
                try:
                    msvcrt.locking(self._stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError as exc:
                    contention = (
                        getattr(exc, "winerror", None) in {33, 36}
                        or exc.errno == 13
                    )
                    if not contention:
                        self._stream.close()
                        self._stream = None
                        raise
                    if time.monotonic() >= deadline:
                        self._stream.close()
                        self._stream = None
                        raise TimeoutError(
                            f"Tiempo agotado esperando el lock: {self.path}"
                        ) from exc
                    time.sleep(0.05)
        else:
            import fcntl

            fcntl.flock(self._stream.fileno(), fcntl.LOCK_EX)  # type: ignore[attr-defined]
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._stream is None:
            return
        try:
            self._stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self._stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
        finally:
            self._stream.close()
            self._stream = None

@dataclass(slots=True)
class JobStore:
    source: Path
    root: Path
    fingerprint: str

    @classmethod
    def for_video(cls, source: Path, *, jobs_root: Path | None = None) -> "JobStore":
        source = source.resolve()
        fingerprint = _source_fingerprint(source)
        root = source_job_root(source, jobs_root=jobs_root)
        root.mkdir(parents=True, exist_ok=True)
        store = cls(source=source, root=root, fingerprint=fingerprint)
        store._ensure_initialized(
            {
                "version": 1,
                "source": str(source),
                "fingerprint": fingerprint,
                "created_at": _utc_now(),
                "updated_at": _utc_now(),
                "stages": {},
            }
        )
        return store

    @property
    def state_path(self) -> Path:
        return self.root / "state.json"

    @property
    def audio_path(self) -> Path:
        return self.root / "audio_16k_mono.flac"

    @property
    def transcript_path(self) -> Path:
        return self.root / "transcript.json"

    @property
    def timeline_path(self) -> Path:
        return self.root / "timeline.json"

    @property
    def diarization_path(self) -> Path:
        return self.root / "diarization.json"

    def stage_lock(self, stage: str) -> _InterProcessFileLock:
        return _InterProcessFileLock(self.root / f".{_safe_slug(stage)}.lock")

    def _state_lock(self) -> _InterProcessFileLock:
        return _InterProcessFileLock(self.root / ".state.lock")

    @property
    def speaker_transcript_path(self) -> Path:
        return self.root / "transcript_speakers.json"

    @property
    def preferred_transcript_path(self) -> Path:
        """Use voice attribution only when it is bound to the current ASR text.

        Retain old/orphaned artifacts for recovery, but never promote their old
        words or timestamps after retranscription. Missing provenance is stale.
        """
        if self.speaker_transcript_path.is_file() and self.transcript_path.is_file():
            try:
                evidence = json.loads(self.diarization_path.read_text(encoding="utf-8"))
                provenance = evidence.get("provenance", {})
                actual = hashlib.sha256(self.transcript_path.read_bytes()).hexdigest()
                if provenance.get("transcript_sha256") == actual:
                    return self.speaker_transcript_path
            except (OSError, ValueError, AttributeError, TypeError):
                pass
        return self.transcript_path

    @property
    def face_tracks_path(self) -> Path:
        return self.root / "face_tracks.json"

    @property
    def streamer_identity_path(self) -> Path:
        return self.root / "streamer_identity.json"

    @property
    def webcam_tracks_path(self) -> Path:
        return self.root / "webcam_tracks.json"

    @property
    def source_visual_timeline_path(self) -> Path:
        return self.root / "source_visual_timeline.json"

    @property
    def source_visual_observed_latest_path(self) -> Path:
        return self.root / "visual_observed" / "latest.json"

    def new_observed_generation(self) -> Path:
        target = self.root / "visual_observed" / "generations" / uuid4().hex
        target.mkdir(parents=True, exist_ok=False)
        return target

    @property
    def style_evidence_path(self) -> Path:
        """Shared local style evidence location; no environment/provider loading."""
        return (_local_paths.data_dir / "processed" / "style_library" / "evidence" /
                self.fingerprint / "style_evidence.json")

    @property
    def speaker_analysis_path(self) -> Path:
        return self.root / "speaker_analysis.json"

    @property
    def speaker_profile_path(self) -> Path:
        return self.root / "speaker_profile.json"

    @property
    def frames_dir(self) -> Path:
        target = self.root / "frames"
        target.mkdir(parents=True, exist_ok=True)
        return target

    @property
    def probes_dir(self) -> Path:
        target = self.root / "probes"
        target.mkdir(parents=True, exist_ok=True)
        return target

    @property
    def planning_dir(self) -> Path:
        target = self.root / "planning"
        target.mkdir(parents=True, exist_ok=True)
        return target

    @property
    def evidence_dir(self) -> Path:
        target = self.planning_dir / "evidence"
        target.mkdir(parents=True, exist_ok=True)
        return target

    def new_visual_run(self, clip_id: str) -> Path:
        """Allocate an exclusive per-invocation directory; never reuse a ready plan."""
        target = self.planning_dir / "visual" / _safe_slug(clip_id) / uuid4().hex
        target.mkdir(parents=True, exist_ok=False)
        return target

    def mark_review_required(self, stage: str, *, reason: str, receipt: Path) -> None:
        self._update_stage(stage, {
            "status": "review_required", "reason": reason, "artifacts": [str(receipt)],
        })

    def load(self) -> dict[str, Any]:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def stage_complete(self, stage: str, *, artifact: Path | None = None) -> bool:
        record = self.load().get("stages", {}).get(stage, {})
        if record.get("status") != "complete":
            return False
        return artifact is None or artifact.exists()

    def mark_started(self, stage: str) -> None:
        self._update_stage(stage, {"status": "running", "started_at": _utc_now()})

    def mark_complete(self, stage: str, *, artifacts: list[Path] | None = None) -> None:
        self._update_stage(
            stage,
            {
                "status": "complete",
                "completed_at": _utc_now(),
                "artifacts": [str(path) for path in artifacts or []],
            },
        )

    def mark_failed(self, stage: str, error: str) -> None:
        self._update_stage(stage, {"status": "failed", "failed_at": _utc_now(), "error": error})

    def _update_stage(self, stage: str, values: dict[str, Any]) -> None:
        with self._state_lock():
            state = self.load()
            stages = state.setdefault("stages", {})
            record = stages.setdefault(stage, {})
            record.update(values)
            status = values.get("status")
            if status == "running":
                for key in ("completed_at", "failed_at", "error", "artifacts", "reason"):
                    record.pop(key, None)
            elif status == "complete":
                for key in ("failed_at", "error", "reason"):
                    record.pop(key, None)
            elif status == "failed":
                for key in ("completed_at", "artifacts"):
                    record.pop(key, None)
            elif status == "review_required":
                for key in ("completed_at", "failed_at", "error"):
                    record.pop(key, None)
            state["updated_at"] = _utc_now()
            self._write_state_unlocked(state)

    def _ensure_initialized(self, state: dict[str, Any]) -> None:
        with self._state_lock():
            if self.state_path.exists():
                return
            self._write_state_unlocked(state)

    def _write_state_unlocked(self, state: dict[str, Any]) -> None:
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.state_path.parent,
                prefix=".state.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(json.dumps(state, ensure_ascii=False, indent=2))
            temporary.replace(self.state_path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
