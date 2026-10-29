"""Cleanup-only proof: real owner/files, scripted workers and filesystem only."""

import importlib
from pathlib import Path
from typing import Any

import pytest

from clipperx.tests.test_native_render_qualification import (
    fixture_adapter, no_native as no_native,
)
from clipperx.tests.test_source_host_ownership import FakeEvent, finish_job


def module() -> Any:
    return importlib.import_module("scripts.qualify_cleanup_io")


def test_default_cleanup_entry_is_inert() -> None:
    assert module().main([]) == 0
    with pytest.raises(ValueError, match="confirm-host"):
        module().main(["--native"])


def test_cleanup_manifest_requires_complete_explicit_pins() -> None:
    q = module()
    root = Path("O:/HOST/repo")
    data: dict[str, Any] = dict(version=1, run_id="cleanup-new", repository=str(root),
        interpreter=str(root/".venv-whisperx/Scripts/python.exe"), interpreter_sha256="a"*64,
        pins={path: "b"*64 for path in q.PIN_PATHS},
        lineage={path: "c"*64 for path in q.PROVENANCE_PATHS})
    q.validate_cleanup_manifest(data, root, "cleanup-new")
    for key in ("pins", "lineage"):
        broken = {**data, key: dict(data[key])}
        broken[key].pop(next(iter(broken[key])))
        with pytest.raises(ValueError):
            q.validate_cleanup_manifest(broken, root, "cleanup-new")
    with pytest.raises(ValueError):
        q.validate_cleanup_manifest(data, root, "cleanup-replayed")


@pytest.mark.parametrize("case", ["success", "skipped", "wrong_target", "wrong_token", "expiry", "995", "truncated"])
def test_cleanup_proof_requires_measured_real_gate_and_certain_retirement(
        monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    from scripts import render_visual_smoke as smoke
    q = module()
    c = fixture_adapter(monkeypatch)
    s = c.session
    s.root = s.root/"proof"
    s.job._grace = 5_000_000_000
    # Only external event waiting is scripted. Owner, file service and cancel gate are real.
    proof = q.CleanupProof(s, event_factory=FakeEvent, identity=lambda: c.h.current, controller=None)
    original_action = c.files._path_action
    c.files._path_action = s.record_path_action(proof.path_action(original_action))
    original_cancel = s.job._cancel
    def cancel(resource: Any) -> None:
        proof.cancel_gate(resource, original_cancel)
    s.job._cancel = cancel
    native = s.job._handle_resource.io.request_cancel
    def request(handle: int) -> Any:
        return proof.native_request(handle, native)
    s.job._handle_resource.io.request_cancel = request
    def wait(timeout: float) -> bool:
        assert 0 <= timeout <= .01
        c.h.current = None
        try:
            if case == "skipped":
                s.job._cleanup_lock.acquire()
                try:
                    s.job.service_cleanup()
                finally:
                    s.job._cleanup_lock.release()
            elif case == "wrong_target":
                s.job._cancel(None)
            elif case == "wrong_token":
                proof.protected = object()
                s.job._cancel(s.job._handle_resource)
            else:
                if case == "expiry":
                    c.h.now = s.job.snapshot().failure.cleanup_deadline_ns
                s.job.service_cleanup()
            if case == "truncated":
                for _ in range(33):
                    proof.note("gate_return")
        finally:
            c.h.current = c.h.driver
        if not proof.release.is_set():
            c.h.now = s.job.snapshot().failure.cleanup_deadline_ns
        return proof.release.is_set()
    proof.release.wait = wait
    if case == "995":
        def fail(operation: str, path: Path) -> None:
            if operation == "unlink" and path == proof.leaf:
                error = OSError(22, "operation aborted")
                setattr(error, "winerror", 995)
                raise error
        c.fs.hook = fail
    s.run_driver(proof.work)
    owner = finish_job(c.h)
    assert owner.own0
    if case == "success":
        result = proof.finish()
        assert result["status"] == "MEASURED_CLEANUP_ONLY"
        assert result["children"] == 0 and result["files"].files_clear
        assert result["failure"].code == "validation"
        assert len(c.fs.data.get(proof.leaf, b"")) == 0
        assert s.state["driver_error"]["message"] == q.ORIGINAL_TAG
        assert len([e for e in c.fs.events if e == ("unlink", proof.leaf)]) == 1
        assert s.campaign.records[-1]["driver_error"]["message"] == q.ORIGINAL_TAG
    else:
        with pytest.raises(smoke.QualificationFailure):
            proof.finish()
    assert not c.h.children


def test_cleanup_process_admission_always_rejected() -> None:
    q = module()
    with pytest.raises(ValueError, match="zero child"):
        q.reject_process(("anything",))


@pytest.mark.parametrize("case", ["pin_mismatch", "guard_replay"])
def test_activation_checks_live_pins_and_absence_before_driver(
        monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    import json
    from scripts import render_visual_smoke as smoke
    q = module()
    root = Path(q.__file__).resolve().parents[1]
    data: dict[str, Any] = dict(version=1, run_id="cleanup-once", repository=str(root),
        interpreter=str(root/".venv-whisperx/Scripts/python.exe"), interpreter_sha256="a"*64,
        pins={path: "b"*64 for path in q.PIN_PATHS},
        lineage={path: "c"*64 for path in q.PROVENANCE_PATHS})
    monkeypatch.setattr(smoke, "bounded_read", lambda path, cap: json.dumps(data).encode())
    checked = []
    def digest(path: Path, cap: int, deadline: int) -> str:
        checked.append(path)
        if case == "pin_mismatch":
            return "0"*64
        return ("a"*64 if path == Path(data["interpreter"]) else
                data["pins"].get(str(path.relative_to(root)).replace("\\", "/"), "c"*64))
    monkeypatch.setattr(smoke, "file_hash", digest)
    monkeypatch.setattr(Path, "is_dir", lambda path: True)
    monkeypatch.setattr(Path, "exists", lambda path: True)
    with pytest.raises(ValueError, match="pin mismatch" if case == "pin_mismatch" else "replayed"):
        q.main(["--native", "--confirm-host", "--manifest", "cleanup-manifest.json", "--run-id", "cleanup-once"])
    assert checked
    if case == "guard_replay":
        assert len(checked) == len(data["pins"])+len(data["lineage"])+1
