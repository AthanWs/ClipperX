"""One opt-in cleanup-only proof; default/import performs no setup or native IO."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import threading
import time
from typing import Any, Callable, Sequence

from scripts import render_visual_smoke as smoke

ORIGINAL_TAG = "cleanup-proof-original-validation"
PIN_PATHS = tuple(dict.fromkeys((*smoke.UNIT_PATHS, *smoke.AUTHORITY_PATHS,
    "scripts/qualify_cleanup_io.py", "clipperx/tests/test_cleanup_io_qualification.py")))
PROVENANCE_PATHS = (*smoke.LINEAGE,
    ".superpowers/sdd/windows-cleanup-995-design.md",
    ".superpowers/sdd/phase3-task-59-report.md",
    ".superpowers/sdd/phase3-task-59-review.md")
_CURRENT = object()


def reject_process(argv: tuple[str, ...]) -> Any:
    raise ValueError("cleanup proof requires zero child admissions")


def validate_cleanup_manifest(data: Any, root: Path, run_id: str) -> None:
    if (type(data) is not dict or set(data) != {"version", "run_id", "repository", "interpreter",
            "interpreter_sha256", "pins", "lineage"} or data["version"] != 1
            or data["run_id"] != run_id or data["repository"] != str(root)
            or data["interpreter"] != str(root/".venv-whisperx/Scripts/python.exe")):
        raise ValueError("exact cleanup manifest repository/interpreter/run identity required")
    for key, required in (("pins", PIN_PATHS), ("lineage", PROVENANCE_PATHS)):
        if type(data[key]) is not dict or not set(required) <= set(data[key]) or len(data[key]) > 128:
            raise ValueError("complete explicit source/provenance pins required")
        for name, digest in data[key].items():
            smoke.safe_relative(name)
            if type(digest) is not str or not re.fullmatch("[0-9a-f]{64}", digest):
                raise ValueError("explicit SHA256 pin required")
    if type(data["interpreter_sha256"]) is not str or not re.fullmatch("[0-9a-f]{64}", data["interpreter_sha256"]):
        raise ValueError("explicit interpreter SHA256 required")


class CleanupProof:
    """Fixture-local delegates observe the actual owner gate and filesystem return."""

    def __init__(self, session: smoke.FixtureSession, *, event_factory: Callable[[], Any] = threading.Event,
                 identity: Callable[[], object] = threading.current_thread, controller: object = _CURRENT) -> None:
        self.session, self.identity = session, identity
        self.controller = identity() if controller is _CURRENT else controller
        self.leaf = session.root/"probe.bin"
        self.release = event_factory()
        self.records: list[Any] = []
        self._records_lock = threading.Lock()
        self.truncated = False
        self.protected: object | None = None
        self.gate = self.unlinked = self.removed = False
        self.native_count = 0
        self.protected_native_count = 0
        self.failure: Any = None

    def note(self, tag: str, **values: Any) -> None:
        if tag not in ("protected_enter", "gate_enter", "gate_return", "native_request", "unlink_return",
                       "rmdir_return", "original_failure", "final"):
            raise ValueError("fixed proof evidence tag required")
        who = self.identity()
        role = ("controller" if who is self.controller else "driver" if who is self.session.job._driver
                else "watchdog" if self.session.job._watch_resource is not None
                and who is self.session.job._watch_resource.worker else "unknown")
        record = dict(tag=tag, role=role, at_ns=self.session.clock(), **values)
        with self._records_lock:
            if len(self.records) >= 32:
                self.truncated = True
                return
            self.records.append(record)

    def native_request(self, handle: int, delegate: Callable[[int], Any]) -> Any:
        token = self.session.job._cleanup_token
        failure = self.session.job.snapshot().failure
        protected = (token is not None and token is self.protected and token.active
                     and failure is not None and self.session.clock() < failure.cleanup_deadline_ns)
        with self._records_lock:
            self.native_count += 1
            if protected:
                self.protected_native_count += 1
        self.note("native_request", handle=handle)
        return delegate(handle)

    def cancel_gate(self, resource: Any, delegate: Callable[[Any], None]) -> None:
        s, who = self.session, self.identity()
        now = s.clock()
        with s.job._lock:
            token, failure = s.job._cleanup_token, s.job._failure
            eligible = (who is self.controller and token is not None and token is self.protected
                and token.active and resource is s.job._handle_resource and resource is not None
                and resource.handle is not None and resource.row.phase == "live"
                and not resource.row.in_flight and not resource.row.uncertain and failure is not None
                and now < failure.cleanup_deadline_ns)
            handle = None if resource is None else resource.handle
        if eligible:
            assert token is not None and failure is not None
            self.note("gate_enter", token=id(token), handle=handle, native_count=self.native_count,
                      deadline_ns=failure.cleanup_deadline_ns)
        before = self.native_count
        delegate(resource)  # Exactly once, preserving return/exception from the real gate.
        if eligible:
            assert token is not None and failure is not None
            after = s.clock()
            self.note("gate_return", token=id(token), handle=handle, native_count=self.native_count,
                      deadline_ns=failure.cleanup_deadline_ns)
            with s.job._lock:
                still_protected = (s.job._cleanup_token is token and token.active
                    and s.job._failure is not None and after < s.job._failure.cleanup_deadline_ns)
            if still_protected and before == self.native_count and not self.protected_native_count and not self.truncated:
                self.gate = True
                self.release.set()

    def path_action(self, delegate: Callable[[str, Path, Path | None], object]
                    ) -> Callable[[str, Path, Path | None], object]:
        def action(operation: str, path: Path, destination: Path | None) -> object:
            if operation == "unlink" and path == self.leaf:
                s = self.session
                token = s.job._cleanup_token
                failure = s.job.snapshot().failure
                row = next((row for row in s.files.snapshot().rows
                            if row.path == str(path) and row.operation == "delete" and row.in_flight), None)
                if (not s._retiring or token is None or not token.active or failure is None
                        or self.protected is not None or row is None):
                    raise ValueError("actual active cleanup action/token required")
                self.protected, self.failure = token, failure
                self.note("protected_enter", token=id(token), action=row.generation,
                          deadline_ns=failure.cleanup_deadline_ns)
                while not self.release.is_set():
                    remaining = failure.cleanup_deadline_ns-s.clock()
                    if remaining <= 0:
                        raise ValueError("cleanup proof overlap expired")
                    self.release.wait(min(.01, remaining/1e9))
                if not self.gate or s.clock() >= failure.cleanup_deadline_ns:
                    raise ValueError("cleanup proof requires measured gate before deadline")
                result = delegate(operation, path, destination)
                self.unlinked = True
                self.note("unlink_return", token=id(token), action=row.generation)
                return result
            result = delegate(operation, path, destination)
            if operation == "rmdir" and path == self.session.root:
                self.removed = True
                self.note("rmdir_return")
            return result
        return action

    def work(self) -> None:
        s = self.session
        s.register(s.root, "directory", 0)
        with s.io_scope():
            s.files.path_action("mkdir", s.root)
        s.put("probe.bin", bytes(range(256))*32, role="temporary", cap=8192)
        self.note("original_failure")
        raise ValueError(ORIGINAL_TAG)

    def finish(self) -> dict[str, Any]:
        s = self.session
        try:
            s.finish_result(cancel=False)
        except smoke.QualificationFailure as error:
            original = s.state.get("error")
            status, files = s.job.snapshot(), s.files.snapshot()
            failure = status.failure
            if (type(original) is not ValueError or str(original) != ORIGINAL_TAG or error.__cause__ is not original
                    or self.failure is None or failure is None or failure != self.failure or failure.code != "validation"
                    or failure.stage != "driver" or failure.cleanup_deadline_ns-failure.first_ns != 5_000_000_000
                    or s.state.get("cleanup_exception") is not None or s.state.get("cleanup_error") is not None
                    or not self.gate or not self.unlinked or not self.removed or self.truncated or self.protected_native_count
                    or s.campaign.children != 0 or status.current is not None
                    or not status.own0 or not files.files_clear):
                raise smoke.QualificationFailure(ValueError("cleanup proof incomplete/failed"),
                    dict(records=self.records, original_evidence=error.evidence, truncated=self.truncated)) from error
            self.note("final", own0=status.own0, files_clear=files.files_clear)
            if self.truncated:
                raise smoke.QualificationFailure(ValueError("cleanup proof evidence truncated"), self.records) from error
            return dict(status="MEASURED_CLEANUP_ONLY", scope="one finite cleanup scenario; no media/A2 attribution",
                failure=failure, owner=status, files=files, children=0, records=self.records,
                original_evidence=error.evidence)
        raise smoke.QualificationFailure(ValueError("missing exact original validation"), self.records)


def run_cleanup(base: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    from clipperx._source_host_ownership import SourceHostJob
    from clipperx._retained_files import RetainedFiles, _path_action
    from clipperx._windows_pipe_io import WindowsSyncIo
    session = smoke.FixtureSession(base/"scratch", manifest, smoke.Campaign("prerequisites"))
    proof = CleanupProof(session)
    class ProofIo(WindowsSyncIo):
        def request_cancel(self, handle: int) -> Any:
            return proof.native_request(handle, super().request_cancel)
    def driver_main() -> None:
        session.job.register_current_driver()
        session.run_driver(proof.work)
    driver = threading.Thread(target=driver_main, name="cleanup-proof-driver", daemon=False)
    session.job = SourceHostJob(driver, deadline_ns=session.clock()+120*10**9,
        clock=session.clock, process_factory=reject_process, io_factory=ProofIo)
    original_cancel = SourceHostJob._cancel.__get__(session.job)
    setattr(session.job, "_cancel", lambda resource: proof.cancel_gate(resource, original_cancel))
    session.files = RetainedFiles(session.job, root=base, clock=session.clock,
        _path_action=session.record_path_action(proof.path_action(_path_action)))
    session.job.start_driver()
    while True:
        status = session.job.service_cleanup()
        if status.own0:
            return proof.finish()
        if status.failure is not None and session.clock() >= status.failure.cleanup_deadline_ns:
            raise smoke.QualificationFailure(ValueError("cleanup grace exhausted; custody retained"),
                dict(owner=status, files=session.files.snapshot(), records=proof.records,
                     driver_error=session.state.get("driver_error")))
        driver.join(.01)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--confirm-host", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    if not args.native:
        print("actual NOT RUN: cleanup proof requires Root manifest and new exclusive guard")
        return 0
    smoke.validate_activation(args.native, args.confirm_host)
    if args.manifest is None or type(args.run_id) is not str or not re.fullmatch("[A-Za-z0-9_-]{1,64}", args.run_id):
        raise ValueError("--manifest and fresh ASCII --run-id required")
    repository = Path(__file__).resolve().parents[1]
    interpreter = repository/".venv-whisperx/Scripts/python.exe"
    if Path(sys.executable).resolve() != interpreter.resolve():
        raise ValueError("exact approved interpreter required")
    raw = smoke.bounded_read(args.manifest, 1_000_000)
    manifest = json.loads(raw)
    validate_cleanup_manifest(manifest, repository, args.run_id)
    deadline = time.monotonic_ns()+120*10**9
    smoke.check_source_pins(manifest, repository, deadline)
    if smoke.file_hash(interpreter, 32*smoke.MiB, deadline) != manifest["interpreter_sha256"]:
        raise ValueError("approved interpreter bytes changed")
    parent = repository/"data/processed/phase3-native-qualification"
    if not parent.is_dir():
        raise ValueError("Root must provision campaign parent")
    base = parent/args.run_id
    smoke.require_absence(base.exists(), (base/".guard").exists())
    base.mkdir(exist_ok=False)
    digest = hashlib.sha256(raw).hexdigest()
    with (base/".guard").open("xb") as stream:
        stream.write(smoke.json_bytes(dict(consumed=True, manifest_sha256=digest, run_id=args.run_id)))
    try:
        result = run_cleanup(base, manifest)
    except Exception as error:
        with (base/"controller-error.json").open("xb") as stream:
            stream.write(smoke.json_bytes(dict(status="STOPPED; custody retained", manifest_sha256=digest,
                error=smoke.error_details(error), evidence=getattr(error, "evidence", None))))
        raise
    with (base/"controller-summary.json").open("xb") as stream:
        stream.write(smoke.json_bytes(dict(result=result, manifest_sha256=digest, run_id=args.run_id)))
    print(f"MEASURED cleanup only; Root acceptance pending: {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
