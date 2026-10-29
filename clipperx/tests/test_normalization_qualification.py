"""Source-only tests: no native process, thread start, or runtime-binary hashing."""

from array import array
import copy
import ctypes
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
from typing import Any

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/qualify_normalization_transport.py"


def denied(*args: Any, **kwargs: Any) -> Any:
    raise AssertionError("actual process/thread/native execution forbidden in source-only tests")


@pytest.fixture(autouse=True)
def no_native(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "Popen", denied)
    monkeypatch.setattr(subprocess, "run", denied)
    monkeypatch.setattr(threading.Thread, "start", denied)
    monkeypatch.setattr(ctypes, "WinDLL", denied, raising=False)
    from clipperx import render_runtime
    from clipperx._source_host_ownership import SourceHostJob
    from clipperx._windows_pipe_io import WindowsSyncIo
    monkeypatch.setattr(SourceHostJob, "start_driver", denied)
    monkeypatch.setattr(WindowsSyncIo, "__init__", denied)
    monkeypatch.setattr(render_runtime, "resolve_render_runtime", denied)
    monkeypatch.setattr(render_runtime, "_hash", denied)


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    if not SCRIPT.is_file():
        pytest.skip("Harness absent: the explicit existence contract supplies the initial RED")
    spec = importlib.util.spec_from_file_location("qualification_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module.platform, "system", lambda: "Windows")
    monkeypatch.setattr(module, "_default_runner", denied)
    return module


def test_harness_source_exists() -> None:
    assert SCRIPT.is_file(), "Task54 harness does not exist yet"


@pytest.fixture
def campaign(harness: Any, tmp_path: Path) -> Any:
    pins = []
    for name in harness.REQUIRED_PINS:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"tiny source pin fixture\n")
        pins.append({"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest().upper()})
    manifest = {
        "version": 1, "kind": "normalization-qualification-v1",
        "interpreter": sys.executable, "source_pins": pins,
        "cases": list(harness.CASES), "limits": dict(harness.LIMITS),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    run_dir = tmp_path / "data/processed/qualification/normalization/unit-run"
    guard = tmp_path / ".superpowers/sdd/qualification/unit-run.guard"
    argv = ["--manifest", str(manifest_path), "--run-dir", str(run_dir),
            "--guard", str(guard), "--confirm-host"]
    return harness, manifest, manifest_path, run_dir, guard, argv


def passed(case: str) -> dict[str, Any]:
    positive = case in ("linear_mono", "dynamic_stereo")
    size = 1536000 if case == "dynamic_stereo" else 192000
    return {
        "case_id": case, "error": None, "cleanup_errors": [],
        "own0": True, "window_resources_zero": True, "files_clear": True,
        "process_count": 2 if positive else 0, "pass_count": 2 if positive else 0,
        "inference_calls": 0, "expected_rejection": case == "rejected_binding",
        "result_present": positive, "file_generations": 12 if positive else 0,
        "mode": ("dynamic" if case == "dynamic_stereo" else "linear") if positive else None,
        "observed_mode": ("dynamic" if case == "dynamic_stereo" else "linear") if positive else None,
        "samples": (192000 if case == "dynamic_stereo" else 48000) if positive else 0,
        "output_bytes": size if positive else 0, "finite": positive,
        "association_verified": positive,
    }


def test_cli_requires_confirmation_before_any_runtime(harness: Any) -> None:
    with pytest.raises(SystemExit) as caught:
        harness.main(["--manifest", "absent", "--run-dir", "absent", "--guard", "absent"])
    assert caught.value.code == 2


@pytest.mark.parametrize("option", ["--manifest", "--run-dir", "--guard"])
def test_all_cli_options_required(harness: Any, option: str) -> None:
    argv = ["--confirm-host"]
    for name in ("--manifest", "--run-dir", "--guard"):
        if name != option:
            argv.extend((name, "absent"))
    with pytest.raises(SystemExit):
        harness.main(argv)


def test_import_and_default_factory_are_inert(harness: Any) -> None:
    from clipperx import _source_host_ownership as owner
    from clipperx import render_runtime
    assert callable(owner.SourceHostJob)
    assert callable(render_runtime.resolve_render_runtime)
    assert not hasattr(harness, "runtime")


@pytest.mark.parametrize("change", ["unknown", "version_bool", "kind", "limits", "nan",
    "cases", "duplicate_case", "pin_hash", "pin_path", "duplicate_pin", "missing_pin",
    "missing", "interpreter", "relative_interpreter"])
def test_bad_manifest_never_starts_case(campaign: Any, change: str) -> None:
    h, manifest, path, run, guard, argv = campaign
    if change == "unknown":
        manifest["argv"] = ["arbitrary"]
    elif change == "version_bool":
        manifest["version"] = True
    elif change == "kind":
        manifest["kind"] = "other"
    elif change in ("limits", "nan"):
        manifest["limits"]["positive_job_seconds"] = 101 if change == "limits" else float("nan")
    elif change == "cases":
        manifest["cases"].reverse()
    elif change == "duplicate_case":
        manifest["cases"][1] = manifest["cases"][0]
    elif change == "pin_hash":
        manifest["source_pins"][0]["sha256"] = "A" * 64
    elif change == "pin_path":
        manifest["source_pins"][0]["path"] = "../outside.py"
    elif change == "duplicate_pin":
        manifest["source_pins"].append(manifest["source_pins"][0])
    elif change == "missing_pin":
        manifest["source_pins"].pop()
    elif change == "missing":
        del manifest["kind"]
    elif change == "interpreter":
        manifest["interpreter"] = str(path.parent / "other/python.exe")
    else:
        manifest["interpreter"] = "python.exe"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    assert h.main(argv, _case_runner=denied) == 2
    assert not run.exists() and not guard.exists()


@pytest.mark.parametrize("existing", ["guard", "run"])
def test_exclusive_guard_and_absent_run(campaign: Any, existing: str) -> None:
    h, _, _, run, guard, argv = campaign
    target = guard if existing == "guard" else run
    target.parent.mkdir(parents=True, exist_ok=True)
    if existing == "guard":
        target.write_text("do not replace", encoding="utf-8")
    else:
        target.mkdir()
    assert h.main(argv, _case_runner=denied) == 2
    if existing == "guard":
        assert guard.read_text(encoding="utf-8") == "do not replace"


def test_fixed_order_guard_persists_and_small_artifacts(campaign: Any) -> None:
    h, _, manifest_path, run, guard, argv = campaign
    calls = []
    def fake(case: str, scratch: Path, deadline: int) -> dict[str, Any]:
        assert guard.exists() and scratch.is_dir() and deadline > 0
        calls.append(case)
        return passed(case)
    assert h.main(argv, _case_runner=fake) == 0
    assert calls == list(h.CASES)
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "PASS" and summary["process_count"] == 4
    assert summary["inference_calls"] == 0
    assert summary["completed_cases"] == list(h.CASES)
    claim = json.loads(guard.read_text(encoding="utf-8"))
    assert (run / "manifest.json").read_bytes() == manifest_path.read_bytes()
    assert claim["manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest().upper()
    assert claim["run_id"] == "unit-run"
    assert h.main(argv, _case_runner=denied) == 2


@pytest.mark.parametrize("field,value", [
    ("own0", False), ("window_resources_zero", False), ("files_clear", False),
    ("mode", "dynamic"), ("observed_mode", "dynamic"), ("samples", 47999),
    ("output_bytes", 191999), ("process_count", 1), ("pass_count", 1),
    ("finite", False), ("association_verified", False), ("inference_calls", 1),
    ("inference_calls", False), ("file_generations", 0),
    ("error", {"type": "ValueError", "message": "unexpected"}),
])
def test_failed_gate_stops_before_next_case(campaign: Any, field: str, value: Any) -> None:
    h, _, _, run, _, argv = campaign
    calls = []
    def fake(case: str, scratch: Path, deadline: int) -> dict[str, Any]:
        calls.append(case)
        record = passed(case)
        record[field] = value
        return record
    assert h.main(argv, _case_runner=fake) == 1
    assert calls == ["linear_mono"]
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == ("cleanup_required" if field in
        ("own0", "window_resources_zero", "files_clear") else "FAIL")
    assert summary["failed_case"] == "linear_mono"


@pytest.mark.parametrize("field,value", [("expected_rejection", False),
    ("error", {"type": "ValueError", "message": "unrelated"}),
    ("process_count", 1), ("file_generations", 1), ("result_present", True)])
def test_rejection_is_strictly_scoped(campaign: Any, field: str, value: Any) -> None:
    h, _, _, run, _, argv = campaign
    def fake(case: str, scratch: Path, deadline: int) -> dict[str, Any]:
        record = passed(case)
        if case == "rejected_binding":
            record[field] = value
        return record
    assert h.main(argv, _case_runner=fake) == 1
    assert json.loads((run / "summary.json").read_text())["failed_case"] == "rejected_binding"


def test_default_dispatch_selects_real_runner_without_executing_it(
        campaign: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    h, _, _, _, _, argv = campaign
    calls = []
    def factory() -> Any:
        calls.append("real-default-factory")
        return lambda case, scratch, deadline: passed(case)
    monkeypatch.setattr(h, "_default_runner", factory)
    assert h.main(argv) == 0
    assert calls == ["real-default-factory"]


@pytest.mark.parametrize("case,frames,channels,size", [
    ("linear_mono", 48000, 1, 192000),
    ("dynamic_stereo", 192000, 2, 1536000),
    ("disabled", 48000, 1, 0), ("rejected_binding", 48000, 1, 192000)])
def test_exact_pure_pcm_fixture(harness: Any, case: str, frames: int,
        channels: int, size: int) -> None:
    schedule, pcm, binding, fmt = harness._fixture(case)
    assert schedule.total_samples == frames and fmt.channels == channels
    assert len(pcm) == size
    if case == "disabled":
        assert binding is None and schedule.enabled_samples == 0
    else:
        assert binding.decoder_identity == "generated-pcm-v1"
        assert len(binding.source_fingerprint) == 16
        assert (binding.selected_pcm_sha256 == hashlib.sha256(pcm).hexdigest()) == (case != "rejected_binding")
    if case == "dynamic_stereo":
        values = array("f")
        values.frombytes(pcm)
        if sys.byteorder != "little":
            values.byteswap()
        assert values[20000] == pytest.approx(.95)


def test_bounded_json_rejects_pcm_nan_and_unknown_values(harness: Any) -> None:
    for value in ({"pcm": b"raw"}, {"pcm": [0.1]}, {"v": float("nan")},
                  {"v": object()}, {"v": "x" * 1048576}):
        with pytest.raises(ValueError):
            harness._json_bytes(value, 1048576)
    assert json.loads(harness._json_bytes({"stderr_tail": b"a\x00b"}, 1024)) == {
        "stderr_tail": {"base64": "YQBi"}}


def test_duplicate_json_keys_are_rejected(campaign: Any) -> None:
    h, _, path, _, _, argv = campaign
    path.write_text('{"version":1,"version":1}', encoding="utf-8")
    assert h.main(argv, _case_runner=denied) == 2


def test_external_write_paths_rejected(campaign: Any) -> None:
    h, _, _, _, guard, argv = campaign
    escaped = copy.copy(argv)
    escaped[escaped.index("--run-dir") + 1] = str(guard.parent / "outside")
    assert h.main(escaped, _case_runner=denied) == 2


def test_non_windows_rejected(campaign: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    h, _, _, _, _, argv = campaign
    monkeypatch.setattr(h.platform, "system", lambda: "Linux")
    assert h.main(argv, _case_runner=denied) == 2


def test_real_factory_selects_pinned_cpu_pair_and_real_case_dispatch(
        harness: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from clipperx import render_runtime
    calls = []
    runtime = SimpleNamespace(runtime_id=render_runtime.NATIVE_RUNTIME,
        ffmpeg=tmp_path / "ffmpeg.exe", ffprobe=tmp_path / "ffprobe.exe",
        ffmpeg_sha256="A" * 64, ffprobe_sha256="B" * 64, device_index=None)
    def resolver(requirements: Any) -> Any:
        assert requirements.runtime_id == render_runtime.NATIVE_RUNTIME
        assert requirements.needs_vulkan is False
        calls.append("resolver")
        return runtime
    def dispatch(case: str, scratch: Path, deadline: int, selected: Any) -> dict[str, Any]:
        assert selected is runtime
        calls.append(case)
        return passed(case)
    monkeypatch.setattr(render_runtime, "resolve_render_runtime", resolver)
    monkeypatch.setattr(harness, "_run_real_case", dispatch)
    # Fixture's CLI default-factory denial stays in place; explicitly call original source function.
    spec = importlib.util.spec_from_file_location("factory_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    monkeypatch.setattr(original, "_run_real_case", dispatch)
    runner = original._default_runner()
    record = runner("linear_mono", tmp_path, 1)
    assert record["runtime_setup"]["scope"] == "resolver_reads_outside_bridge"
    assert calls == ["resolver", "linear_mono"]


@pytest.fixture
def fake_owned_driver(harness: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Exercise orchestration with a synchronous fake, never an actual job start."""
    from clipperx import _normalization_files as files_module
    from clipperx import _normalization_transport as transport
    from clipperx import _source_host_ownership as owner
    events: list[str] = []
    class FakeThread:
        def __init__(self, *, target: Any, name: str, daemon: bool) -> None:
            assert daemon is False
            self.target = target
            events.append("unstarted-driver")
    class FakeJob:
        def __init__(self, driver: Any, **kwargs: Any) -> None:
            self.driver = driver
            self.done = False
            self.failure: Any = None
        def start_driver(self) -> Any:
            events.append("owner-start-only")
            self.driver.target()
            return self.snapshot()
        def register_current_driver(self) -> None:
            events.append("register")
        def driver_done(self) -> None:
            events.append("done")
            self.done = True
        def fail(self, *args: Any) -> None:
            self.failure = SimpleNamespace(code="validation", cleanup_deadline_ns=2**62)
        def snapshot(self) -> Any:
            return SimpleNamespace(own0=self.done, failure=self.failure)
        def service_cleanup(self) -> Any:
            events.append("owner-cleanup")
            return self.snapshot()
    class FakeFiles:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self._input_consumers: tuple[Any, ...] = ()
            events.append("files")
        def retire(self) -> Any:
            events.append("files-retire")
            return self.snapshot()
        def snapshot(self) -> Any:
            return SimpleNamespace(files_clear=True, rows=())
    monkeypatch.setattr(threading, "Thread", FakeThread)
    monkeypatch.setattr(owner, "SourceHostJob", FakeJob)
    monkeypatch.setattr(files_module, "NormalizationFiles", FakeFiles)
    return events, transport


@pytest.mark.parametrize("outcome", ["expected", "wrong_error", "wrong_stage", "returned"])
def test_actual_runner_rejection_only_from_prepare_call(harness: Any, fake_owned_driver: Any,
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path, outcome: str) -> None:
    events, transport = fake_owned_driver
    def prepare(*args: Any, **kwargs: Any) -> Any:
        events.append("prepare")
        if outcome == "returned":
            return None
        raise ValueError("unrelated" if outcome == "wrong_error" else
                         "enabled bus source/hash/count mismatch")
    monkeypatch.setattr(transport, "prepare_normalized_audio", prepare)
    if outcome == "wrong_stage":
        def fixture(case: str) -> Any:
            raise ValueError("enabled bus source/hash/count mismatch")
        monkeypatch.setattr(harness, "_fixture", fixture)
    record = harness._run_real_case("rejected_binding", tmp_path, 2**62, object())
    assert record["expected_rejection"] is (outcome == "expected")
    assert (record["error"] is None) is (outcome == "expected")
    assert events[:3] == ["unstarted-driver", "owner-start-only", "register"]
    assert events.index("files-retire") < events.index("done") < events.index("owner-cleanup")
    assert record["own0"] and record["files_clear"]


def test_actual_runner_disabled_has_no_prepare_result(harness: Any, fake_owned_driver: Any,
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    events, transport = fake_owned_driver
    def prepare(schedule: Any, pcm: bytes, **kwargs: Any) -> Any:
        assert schedule.enabled_samples == 0 and pcm == b"" and kwargs["binding"] is None
        return None
    monkeypatch.setattr(transport, "prepare_normalized_audio", prepare)
    record = harness._run_real_case("disabled", tmp_path, 2**62, object())
    assert harness._status("disabled", record) == "PASS"
    assert record["fixture"]["decoder_identity"] == "generated-pcm-v1"
    assert record["fixture"]["enabled_bytes"] == 0
    assert "owner-start-only" in events


@pytest.fixture
def positive_metadata(harness: Any, tmp_path: Path) -> Any:
    """Pure, expressly invented metadata for association checks, not native evidence."""
    from clipperx import audio_mix as audio
    from clipperx._normalization_files import FileEvidence, NormalizationFileStatus
    from clipperx._normalization_transport import NormalizationPassEvidence, PreparedNormalization
    from clipperx._source_host_ownership import HostDecodeDiagnostic
    from clipperx.render_runtime import NATIVE_RUNTIME, RenderRuntime
    schedule, pcm, binding, fmt = harness._fixture("linear_mono")
    runtime = RenderRuntime(tmp_path / "ffmpeg.exe", tmp_path / "ffprobe.exe", NATIVE_RUNTIME,
        "A" * 64, "B" * 64, "A" * 64, "B" * 64, "-/filter_complex", None)
    stats = audio.LoudnessStats(-20., 0., -10., -30., 0.)
    receipt = audio.NormalizationReceipt(schedule.identity, binding, audio._runtime_identity(runtime),
        fmt, schedule.enabled_samples, stats, "linear", "pure invented test stats")
    enabled_file = FileEvidence("enabled", (1, 2, len(pcm)), len(pcm), binding.selected_pcm_sha256)
    output_file = replace(enabled_file, slot="normalized", stamp=(1, 3, len(pcm)))
    passes = []
    controls = []
    for index in range(2):
        filter_text = (f"loudnorm={audio.TARGETS}:print_format=json" if index == 0
                       else audio.normalization_filter(receipt))
        canonical = (str(runtime.ffmpeg), "-hide_banner", "-loglevel", "info", "-f", "f32le",
            "-ar", "48000", "-ac", "1", "-channel_layout", "mono", "-i", "pipe:0", "-af",
            filter_text, *(("-f", "null", "-") if index == 0 else
                          ("-c:a", "pcm_f32le", "-f", "f32le", "pipe:1")))
        actual = (*canonical[:13], str(tmp_path / "enabled.f32"), *canonical[14:])
        stdout = b"" if index == 0 else pcm
        digest = hashlib.sha256(stdout).hexdigest()
        evidence = NormalizationPassEvidence("measure" if index == 0 else "process", index + 1,
            index + 1, canonical, actual, enabled_file, len(stdout), digest, 3, True, 0)
        diagnostic = HostDecodeDiagnostic(index + 1, "normalizer", len(stdout), digest, 3,
                                          b"log", False, False, True, 0)
        status = SimpleNamespace(observation_id=index + 1, diagnostics=(diagnostic,))
        controls.append(SimpleNamespace(ownership_snapshot=lambda status=status: status))
        passes.append(evidence)
    normalized = audio.NormalizedPCM(pcm, "linear", schedule.enabled_samples, stats, (-14., 0., -1.5))
    result = PreparedNormalization(schedule.identity, binding, audio._runtime_identity(runtime), fmt,
        receipt, (passes[0], passes[1]), normalized, output_file)
    releases = []
    def output_path(exact: Any) -> Path:
        assert exact is result
        return tmp_path / "normalized.f32"
    def release_output(exact: Any) -> None:
        assert exact is result
        releases.append("no_consumer_remains")
    files = SimpleNamespace(_input_consumers=tuple(controls), output_path=output_path,
        snapshot=lambda: NormalizationFileStatus((), False, True, True, True, False),
        release_output=release_output)
    return result, files, tmp_path, schedule, pcm, binding, fmt, runtime, releases


def test_positive_metadata_has_no_pcm_and_releases_identity_bound_output(
        harness: Any, positive_metadata: Any) -> None:
    record = harness._positive(*positive_metadata[:-1])
    encoded = harness._json_bytes(record, 1048576)
    assert record["association_verified"] and record["pass_count"] == 2
    assert positive_metadata[-1] == ["no_consumer_remains"]
    assert b'"pcm"' not in encoded and b"pipe:0" in encoded
    assert record["before_release"].normalized_present


@pytest.mark.parametrize("change", ["control", "generation", "canonical", "actual", "filter",
    "stdout_count", "stdout_hash", "input_hash", "input_stamp", "output_hash", "sample_count",
    "receipt_format", "nonfinite"])
def test_positive_metadata_rejects_broken_association(harness: Any, positive_metadata: Any,
        change: str) -> None:
    result, files, scratch, schedule, pcm, binding, fmt, runtime, releases = positive_metadata
    evidence = result.passes[1]
    if change == "control":
        evidence = replace(evidence, control_id=result.passes[0].control_id)
    elif change == "generation":
        evidence = replace(evidence, generation=result.passes[0].generation)
    elif change == "canonical":
        evidence = replace(evidence, canonical_pipe_argv=("unknown",))
    elif change == "actual":
        evidence = replace(evidence, actual_argv=evidence.canonical_pipe_argv)
    elif change == "filter":
        argv = list(evidence.canonical_pipe_argv)
        argv[15] = "anull"
        evidence = replace(evidence, canonical_pipe_argv=tuple(argv))
    elif change == "stdout_count":
        evidence = replace(evidence, stdout_bytes=len(pcm) - 4)
    elif change == "stdout_hash":
        evidence = replace(evidence, stdout_sha256="0" * 64)
    elif change == "input_hash":
        evidence = replace(evidence, input_file=replace(evidence.input_file, sha256="0" * 64))
    elif change == "input_stamp":
        evidence = replace(evidence, input_file=replace(evidence.input_file, stamp=(1, 99, len(pcm))))
    elif change == "output_hash":
        result = replace(result, normalized_file=replace(result.normalized_file, sha256="0" * 64))
    elif change == "sample_count":
        result = replace(result, normalized=replace(result.normalized, samples=47999))
    elif change == "receipt_format":
        from clipperx.audio_mix import AudioFormat
        result = replace(result, receipt=replace(result.receipt, audio_format=AudioFormat(48000, "stereo")))
    else:
        values = array("f", [float("nan")])
        result = replace(result, normalized=replace(result.normalized, pcm=values.tobytes() + pcm[4:]))
    result = replace(result, passes=(result.passes[0], evidence))
    with pytest.raises(ValueError):
        harness._positive(result, files, scratch, schedule, pcm, binding, fmt, runtime)
    assert releases == []
def test_retained_action_dependency_is_pinned(harness: Any) -> None:
    assert "clipperx/_retained_files.py" in harness.REQUIRED_PINS
    assert len(harness.REQUIRED_PINS) == 17

