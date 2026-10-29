from __future__ import annotations

import json
from multiprocessing import get_context
from pathlib import Path
from typing import Protocol

import pytest

from clipperx import local_paths


class _SetEvent(Protocol):
    def set(self) -> None: ...


class _WaitEvent(Protocol):
    def wait(self, timeout: float | None = None) -> bool: ...


def _paused_batch_writer(
    first: str,
    second: str,
    value: str,
    ready: _SetEvent,
    release: _WaitEvent,
) -> None:
    first_path = Path(first)
    original = local_paths._replace_with_windows_retry

    def pause_after_first(temporary: Path, target: Path) -> None:
        original(temporary, target)
        if target.resolve() == first_path.resolve():
            ready.set()
            if not release.wait(10.0):
                raise TimeoutError("test release was not signaled")

    local_paths._replace_with_windows_retry = pause_after_first
    local_paths.atomic_write_text_batch(
        (Path(first), Path(second)),
        value,
    )


def _ordinary_batch_writer(
    first: str,
    second: str,
    value: str,
    completed: _SetEvent,
) -> None:
    local_paths.atomic_write_text_batch(
        (Path(first), Path(second)),
        value,
    )
    completed.set()


def test_batch_publication_serializes_independent_processes(tmp_path: Path) -> None:
    first = tmp_path / "global.json"
    second = tmp_path / "job.json"
    first.write_text("old", encoding="utf-8")
    second.write_text("old", encoding="utf-8")
    context = get_context("spawn")
    ready = context.Event()
    release = context.Event()
    completed = context.Event()
    paused = context.Process(
        target=_paused_batch_writer,
        args=(str(first), str(second), "writer-a", ready, release),
    )
    ordinary = context.Process(
        target=_ordinary_batch_writer,
        args=(str(first), str(second), "writer-b", completed),
    )

    paused.start()
    assert ready.wait(10.0)
    ordinary.start()
    completed_before_release = completed.wait(1.5)
    release.set()
    paused.join(10.0)
    ordinary.join(10.0)

    assert paused.exitcode == 0
    assert ordinary.exitcode == 0
    assert completed_before_release is False
    assert local_paths.atomic_read_text_batch((first, second)) == (
        "writer-b",
        "writer-b",
    )


def test_batch_reader_recovers_interrupted_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = tmp_path / "global.json"
    second = tmp_path / "job.json"
    first.write_text("old-global", encoding="utf-8")
    second.write_text("old-job", encoding="utf-8")
    original = local_paths._replace_with_windows_retry

    def interrupt_second(temporary: Path, target: Path) -> None:
        if target.resolve() == second.resolve():
            raise KeyboardInterrupt("simulated process termination")
        original(temporary, target)

    monkeypatch.setattr(local_paths, "_replace_with_windows_retry", interrupt_second)
    with pytest.raises(KeyboardInterrupt, match="simulated process termination"):
        local_paths.atomic_write_text_batch((first, second), "new")
    monkeypatch.setattr(local_paths, "_replace_with_windows_retry", original)

    assert local_paths.atomic_read_text_batch((first, second)) == (
        "old-global",
        "old-job",
    )


@pytest.mark.parametrize("without_common_parent", [False, True])
@pytest.mark.parametrize("job_existed", [False, True])
def test_batch_reader_recovers_interrupted_publication_in_reverse_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    without_common_parent: bool,
    job_existed: bool,
) -> None:
    global_profile = tmp_path / "global" / "identity.json"
    job_profile = tmp_path / "job" / "identity.json"
    global_profile.parent.mkdir()
    job_profile.parent.mkdir()
    global_profile.write_text("old-global", encoding="utf-8")
    if job_existed:
        job_profile.write_text("old-job", encoding="utf-8")
    if without_common_parent:
        def different_drives(parents: object) -> str:
            raise ValueError("Paths don't have the same drive")

        monkeypatch.setattr(local_paths.os.path, "commonpath", different_drives)

    original = local_paths._replace_with_windows_retry

    def interrupt_after_publication(temporary: Path, target: Path) -> None:
        original(temporary, target)
        if target.resolve() == job_profile.resolve():
            raise KeyboardInterrupt("simulated process termination")

    monkeypatch.setattr(local_paths, "_replace_with_windows_retry", interrupt_after_publication)
    with pytest.raises(KeyboardInterrupt, match="simulated process termination"):
        local_paths.atomic_write_text_batch((global_profile, job_profile), "new")
    monkeypatch.setattr(local_paths, "_replace_with_windows_retry", original)
    _, journal = local_paths._batch_state_paths((global_profile, job_profile))
    assert journal.is_file()
    assert (global_profile.read_text(), job_profile.read_text()) == ("new", "new")

    # The CLI publishes global then job; the visual loader reads job then global.
    assert local_paths.atomic_read_text_batch((job_profile, global_profile)) == (
        "old-job" if job_existed else None,
        "old-global",
    )
    assert job_profile.exists() is job_existed
    assert not journal.exists()


def test_batch_reader_restores_existing_string_from_valid_journal(tmp_path: Path) -> None:
    first = tmp_path / "global.json"
    second = tmp_path / "job.json"
    first.write_text("current-global", encoding="utf-8")
    second.write_text("new-job", encoding="utf-8")
    _, journal = local_paths._batch_state_paths((first, second))
    journal.write_text(json.dumps({"version": 1, "entries": [
        {"path": str(first.resolve()), "existed": True, "previous_text": "old-global"},
        {"path": str(second.resolve()), "existed": False, "previous_text": None},
    ]}), encoding="utf-8")

    assert local_paths.atomic_read_text_batch((first, second)) == ("old-global", None)
    assert first.read_text(encoding="utf-8") == "old-global"
    assert not second.exists()
    assert not journal.exists()


@pytest.mark.parametrize("invalid_text", ["missing", "null", "number"])
def test_batch_reader_rejects_invalid_previous_text_before_any_target_mutation(
    tmp_path: Path,
    invalid_text: str,
) -> None:
    first = tmp_path / "global.json"
    second = tmp_path / "job.json"
    for target in (first, second):
        target.write_text("current", encoding="utf-8")
    invalid_entry: dict[str, object] = {"path": str(second.resolve()), "existed": True}
    if invalid_text == "null":
        invalid_entry["previous_text"] = None
    elif invalid_text == "number":
        invalid_entry["previous_text"] = 42
    _, journal = local_paths._batch_state_paths((first, second))
    journal_text = json.dumps({"version": 1, "entries": [
        {"path": str(first.resolve()), "existed": True, "previous_text": "old-global"},
        invalid_entry,
    ]})
    journal.write_text(journal_text, encoding="utf-8")
    before = {target: target.read_bytes() for target in (first, second, journal)}

    with pytest.raises(RuntimeError, match="invalid batch recovery journal"):
        local_paths.atomic_read_text_batch((first, second))

    assert {target: target.read_bytes() for target in before} == before


@pytest.mark.parametrize("malformed", ["duplicate", "foreign", "missing", "invalid_state"])
def test_batch_reader_rejects_malformed_journal_before_restoring_any_target(
    tmp_path: Path,
    malformed: str,
) -> None:
    first = tmp_path / "global.json"
    second = tmp_path / "job.json"
    foreign = tmp_path / "foreign.json"
    for target in (first, second, foreign):
        target.write_text("current", encoding="utf-8")
    entries = [
        {"path": str(target.resolve()), "existed": True, "previous_text": old}
        for target, old in ((first, "old-global"), (second, "old-job"))
    ]
    if malformed == "duplicate":
        entries[1]["path"] = entries[0]["path"]
    elif malformed == "foreign":
        entries[1]["path"] = str(foreign.resolve())
    elif malformed == "missing":
        entries.pop()
    else:
        entries[1]["existed"] = "true"
    _, journal = local_paths._batch_state_paths((first, second))
    journal_text = json.dumps({"version": 1, "entries": entries})
    journal.write_text(journal_text, encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid batch recovery journal"):
        local_paths.atomic_read_text_batch((first, second))

    assert tuple(target.read_text() for target in (first, second, foreign)) == (
        "current", "current", "current",
    )
    assert journal.read_text() == journal_text


@pytest.mark.parametrize("interrupted", [False, True])
def test_batch_without_common_parent_preserves_publication_and_recovery(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    interrupted: bool,
) -> None:
    first = tmp_path / "global" / "identity.json"
    second = tmp_path / "job" / "identity.json"
    for target, value in ((first, "old-global"), (second, "old-job")):
        target.parent.mkdir()
        target.write_text(value, encoding="utf-8")

    def different_drives(parents: object) -> str:
        raise ValueError("Paths don't have the same drive")

    # Exercise Windows' mixed-volume error while keeping test files disposable.
    monkeypatch.setattr(local_paths.os.path, "commonpath", different_drives)
    original = local_paths._replace_with_windows_retry
    if interrupted:
        def interrupt_second(temporary: Path, target: Path) -> None:
            if target.resolve() == second.resolve():
                raise KeyboardInterrupt("simulated process termination")
            original(temporary, target)

        monkeypatch.setattr(local_paths, "_replace_with_windows_retry", interrupt_second)
        with pytest.raises(KeyboardInterrupt, match="simulated process termination"):
            local_paths.atomic_write_text_batch((first, second), "new")
        monkeypatch.setattr(local_paths, "_replace_with_windows_retry", original)
        expected = ("old-global", "old-job")
    else:
        local_paths.atomic_write_text_batch((first, second), "new")
        expected = ("new", "new")

    assert local_paths._batch_state_paths((first, second)) == (
        local_paths._batch_state_paths((second, first))
    )
    assert local_paths.atomic_read_text_batch((first, second)) == expected
    _, journal = local_paths._batch_state_paths((first, second))
    assert not journal.exists()
