"""Explicit, single-use HOST qualification; importing this script performs no work.

Only main opts into real owned files/FFmpeg. Tests use a private trusted runner;
that seam is neither a CLI option nor a manifest/environment-selected backend.
"""

from array import array
import argparse
import base64
from collections.abc import Callable, Sequence
from dataclasses import fields, is_dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import platform
import re
import sys
import threading
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_INTERPRETER = str(ROOT / ".venv-whisperx/Scripts/python.exe")
CASES = ("linear_mono", "dynamic_stereo", "disabled", "rejected_binding")
LIMITS = {"positive_job_seconds": 100, "negative_job_seconds": 5, "cleanup_seconds": 5,
          "campaign_seconds": 240, "case_json_bytes": 1048576,
          "total_evidence_bytes": 5242880, "scratch_bytes": 16000000}
REQUIRED_PINS = (
    "clipperx/__init__.py", "clipperx/_windows_pipe_io.py",
    "clipperx/_source_host_ownership.py", "clipperx/_normalization_files.py", "clipperx/_retained_files.py",
    "clipperx/_normalization_transport.py", "clipperx/audio_mix.py", "clipperx/render_runtime.py",
    "clipperx/filtergraph.py", "clipperx/motion_compiler.py", "clipperx/output_timeline.py",
    "clipperx/render_assets.py", "clipperx/visual_contracts.py", "clipperx/planner_contracts.py",
    "clipperx/schema.py", "clipperx/local_paths.py", "scripts/qualify_normalization_transport.py",
)
_REJECTION = "enabled bus source/hash/count mismatch"
CaseRunner = Callable[[str, Path, int], dict[str, Any]]
_SNAPSHOT_TYPES = frozenset(("SourceJobStatus", "ObservationStatus", "HostResourceStatus",
    "HostDecodeDiagnostic", "HostFailure", "NormalizationFileStatus", "FileActionStatus",
    "FileEvidence", "NormalizationPassEvidence", "NormalizationReceipt", "LoudnessStats",
    "SourceClockBinding", "AudioFormat"))


def _plain(value: Any, *, key: str = "", depth: int = 0, budget: list[int]) -> Any:
    budget[0] -= 1
    if depth > 20 or budget[0] < 0 or key in ("pcm", "stdout", "enabled_pcm", "normalized"):
        raise ValueError("bounded metadata only; PCM forbidden")
    if value is None or type(value) in (str, int, bool):
        if type(value) is str and len(value) > 262144:
            raise ValueError("oversized metadata text")
        if type(value) is int and value.bit_length() > 128:
            raise ValueError("oversized metadata integer")
        budget[1] -= len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
        if budget[1] < 0:
            raise ValueError("metadata byte budget exceeded before collection")
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if type(value) is bytes and key == "stderr_tail" and len(value) <= 65536:
        value = {"base64": base64.b64encode(value).decode("ascii")}
    if is_dataclass(value) and type(value).__name__ in _SNAPSHOT_TYPES:
        value = {field.name: getattr(value, field.name) for field in fields(value)}
    if type(value) is dict and len(value) <= 8192:
        if any(type(k) is not str or len(k) > 128 for k in value):
            raise ValueError("plain metadata keys required")
        return {k: _plain(v, key=k, depth=depth + 1, budget=budget) for k, v in value.items()}
    if type(value) in (tuple, list) and len(value) <= 8192:
        return [_plain(v, depth=depth + 1, budget=budget) for v in value]
    raise ValueError("unknown/nonfinite metadata value")


def _json_bytes(value: Any, cap: int) -> bytes:
    detached = _plain(value, budget=[8192, cap])
    encoded = (json.dumps(detached, allow_nan=False, sort_keys=True,
                          ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
    if len(encoded) > cap:
        raise ValueError("evidence byte cap exceeded")
    return encoded


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _source_path(name: str) -> Path:
    if (type(name) is not str or "\\" in name or ":" in name or "\0" in name
            or PurePosixPath(name).is_absolute() or ".." in name.split("/")
            or str(PurePosixPath(name)) != name):
        raise ValueError("workspace-relative canonical source path required")
    path = (ROOT / name).resolve()
    if not path.is_relative_to(ROOT.resolve()):
        raise ValueError("source pin escapes project")
    return path


def _manifest(path: Path) -> tuple[dict[str, Any], bytes]:
    with path.open("rb") as stream:
        raw = stream.read(131073)
    if len(raw) > 131072:
        raise ValueError("manifest cap exceeded")
    value = json.loads(raw, object_pairs_hook=_unique_object)
    if type(value) is not dict or set(value) != {
            "version", "kind", "source_pins", "interpreter", "cases", "limits"}:
        raise ValueError("exact manifest schema required")
    if (type(value["version"]) is not int or value["version"] != 1
            or value["kind"] != "normalization-qualification-v1"
            or type(value["cases"]) is not list or value["cases"] != list(CASES)):
        raise ValueError("unsupported manifest version/kind/cases")
    limits = value["limits"]
    if (type(limits) is not dict or set(limits) != set(LIMITS)
            or any(type(limits[k]) is not int or limits[k] != v for k, v in LIMITS.items())):
        raise ValueError("exact supported limits required")
    executable = value["interpreter"]
    if (type(executable) is not str or not Path(executable).is_absolute()
            or executable != sys.executable or executable != EXPECTED_INTERPRETER
            or platform.system() != "Windows"):
        raise ValueError("exact Windows qualification interpreter required")
    pins = value["source_pins"]
    if type(pins) is not list or len(pins) != len(REQUIRED_PINS):
        raise ValueError("exact source pin set required")
    names: set[str] = set()
    checked: list[tuple[Path, str]] = []
    for pin in pins:
        if type(pin) is not dict or set(pin) != {"path", "sha256"}:
            raise ValueError("exact source pin required")
        name, digest = pin["path"], pin["sha256"]
        source = _source_path(name)
        if (name not in REQUIRED_PINS or name in names or type(digest) is not str
                or re.fullmatch(r"[A-F0-9]{64}", digest) is None):
            raise ValueError("unknown/duplicate/malformed source pin")
        names.add(name)
        checked.append((source, digest))
    if names != set(REQUIRED_PINS):
        raise ValueError("missing source pin")
    for source, digest in checked:
        with source.open("rb") as stream:
            content = stream.read(2000001)
        if len(content) > 2000000 or hashlib.sha256(content).hexdigest().upper() != digest:
            raise ValueError("source pin mismatch")
    return value, raw


def _destinations(run: Path, guard: Path) -> tuple[Path, Path]:
    for path in (run, guard):
        if not path.is_absolute() or ".." in path.parts or "\0" in str(path):
            raise ValueError("absolute exact destination required")
    run, guard = run.resolve(), guard.resolve()
    namespace = (ROOT / "data/processed/qualification/normalization").resolve()
    guard_namespace = (ROOT / ".superpowers/sdd/qualification").resolve()
    if (run.parent != namespace or guard.parent != guard_namespace
            or not namespace.is_relative_to(ROOT.resolve())
            or not guard_namespace.is_relative_to(ROOT.resolve())
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run.name) is None
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", guard.name) is None
            or run.exists() or guard.exists()):
        raise ValueError("qualification destination occupied/outside namespace")
    return run, guard


def _fixture(case: str) -> tuple[Any, bytes, Any, Any]:
    from clipperx import audio_mix as audio
    if case not in CASES:
        raise ValueError("unknown fixed case")
    duration, channels = (4, 2) if case == "dynamic_stereo" else (1, 1)
    frames = duration * 48000
    span = audio.AudioSpan(Fraction(0), Fraction(duration), Fraction(0), Fraction(duration),
                           0, frames, case != "disabled", 0.0, False)
    schedule = audio.AudioSchedule("54594e5448455449", (span,), ())
    fmt = audio.AudioFormat(48000, "stereo" if channels == 2 else "mono")
    if case == "disabled":
        return schedule, b"", None, fmt
    amplitude = .002 if channels == 2 else .1
    values = array("f", (amplitude * math.sin(2 * math.pi * 997 * frame / 48000)
                         for frame in range(frames) for _ in range(channels)))
    if channels == 2:
        values[20000] = .95
    if values.itemsize != 4:
        raise ValueError("four-byte float array required")
    if sys.byteorder != "little":
        values.byteswap()
    pcm = values.tobytes()
    digest = hashlib.sha256(pcm).hexdigest()
    if case == "rejected_binding":
        digest = "0" * 64 if digest != "0" * 64 else "1" * 64
    binding = audio.SourceClockBinding(schedule.source_fingerprint, "generated-pcm-v1", digest)
    return schedule, pcm, binding, fmt


def _error(exc: Exception) -> dict[str, str]:
    return {"type": type(exc).__name__, "message": str(exc)[:1024]}


def _empty(case: str) -> dict[str, Any]:
    return {"case_id": case, "error": None, "cleanup_errors": [], "own0": False,
            "window_resources_zero": False, "files_clear": False, "process_count": 0,
            "pass_count": 0, "inference_calls": 0, "expected_rejection": False,
            "result_present": False, "file_generations": 0, "mode": None,
            "observed_mode": None, "samples": 0, "output_bytes": 0, "finite": False,
            "association_verified": False}


def _positive(result: Any, files: Any, scratch: Path, schedule: Any,
              pcm: bytes, binding: Any, fmt: Any, runtime: Any) -> dict[str, Any]:
    """Explicit metadata only; never serialize PreparedNormalization or its PCM."""
    from clipperx import audio_mix as audio
    output = result.normalized.pcm
    output_hash = hashlib.sha256(output).hexdigest()
    finite = all(math.isfinite(v) for v in memoryview(output).cast("f"))
    passes = result.passes
    controls = files._input_consumers
    if len(passes) != 2 or len(controls) != 2:
        raise ValueError("exactly two actual normalization passes required")
    checks = (result.schedule_identity == schedule.identity, result.binding == binding,
        result.audio_format == fmt, result.receipt.binding == binding,
        result.receipt.schedule_identity == schedule.identity,
        result.receipt.audio_format == fmt,
        result.receipt.runtime_identity == audio._runtime_identity(runtime),
        result.runtime_identity == audio._runtime_identity(runtime),
        result.receipt.samples == schedule.enabled_samples,
        result.normalized.samples == schedule.enabled_samples,
        result.normalized_file.bytes == len(pcm), result.normalized_file.sha256 == output_hash,
        result.normalized_file.slot == "normalized", passes[0].input_file == passes[1].input_file,
        len(output) == len(pcm), finite,
        len({p.control_id for p in passes}) == 2, len({p.generation for p in passes}) == 2)
    if not all(checks):
        raise ValueError("normalization result identity/count/hash/finiteness mismatch")
    for index, (evidence, control) in enumerate(zip(passes, controls, strict=True)):
        canonical, actual = evidence.canonical_pipe_argv, evidence.actual_argv
        status = control.ownership_snapshot()
        if len(status.diagnostics) != 1:
            raise ValueError("one actual child diagnostic per pass required")
        diagnostic = status.diagnostics[0]
        expected_filter = (f"loudnorm={audio.TARGETS}:print_format=json" if index == 0
                           else audio.normalization_filter(result.receipt))
        if (len(canonical) < 17 or canonical[12:16] != ("-i", "pipe:0", "-af", expected_filter)
                or canonical[0] != str(runtime.ffmpeg)
                or actual != (*canonical[:13], str(scratch / "enabled.f32"), *canonical[14:])
                or evidence.pass_kind != ("measure" if index == 0 else "process")
                or evidence.control_id != status.observation_id
                or evidence.generation != diagnostic.generation
                or evidence.input_file.sha256 != binding.selected_pcm_sha256
                or evidence.input_file.slot != "enabled"
                or evidence.input_file.bytes != len(pcm)
                or evidence.stdout_bytes != (0 if index == 0 else len(pcm))
                or evidence.stdout_sha256 != (hashlib.sha256(b"").hexdigest() if index == 0 else output_hash)
                or diagnostic.stdout_bytes != evidence.stdout_bytes
                or diagnostic.stdout_sha256 != evidence.stdout_sha256
                or diagnostic.log_bytes != evidence.log_bytes
                or len(diagnostic.stderr_tail) != evidence.log_bytes
                or diagnostic.truncated or diagnostic.transfer_uncertain
                or not diagnostic.natural_eof or diagnostic.exit_code != 0
                or not evidence.natural_eof or evidence.exit_code != 0):
            raise ValueError("actual pass/argv/child/result association mismatch")
    # output_path is an identity-bound handoff. This harness creates no consumer.
    output_path = files.output_path(result)
    before_release = files.snapshot()
    if output_path != scratch / "normalized.f32" or not before_release.normalized_present:
        raise ValueError("identity-bound normalized output unavailable")
    record = {"result_present": True, "receipt": result.receipt, "passes": passes,
        "normalized_file": result.normalized_file, "before_release": before_release,
        "output_path": str(output_path), "output_consumer": "none_created_by_harness",
        "mode": result.receipt.mode, "observed_mode": result.normalized.observed_mode,
        "samples": result.normalized.samples, "output_bytes": len(output), "finite": finite,
        "output_sha256": output_hash, "output_loudness": result.normalized.output_loudness,
        "upstream_observation": result.normalized.upstream_observation,
        "association_verified": True, "pass_count": len(passes)}
    files.release_output(result)
    return record


def _run_real_case(case: str, scratch: Path, campaign_deadline: int, runtime: Any) -> dict[str, Any]:
    from clipperx._normalization_files import NormalizationFiles
    from clipperx._normalization_transport import prepare_normalized_audio
    from clipperx._source_host_ownership import SourceHostJob
    started = time.monotonic_ns()
    seconds = LIMITS["positive_job_seconds"] if case in CASES[:2] else LIMITS["negative_job_seconds"]
    deadline = min(campaign_deadline, started + seconds * 1000000000)
    finish_deadline = min(campaign_deadline, deadline + LIMITS["cleanup_seconds"] * 1000000000)
    record = _empty(case)
    holder: list[Any] = []

    def driver_target() -> None:
        try:
            job.register_current_driver()
            files = NormalizationFiles(job, runtime, scratch, clock=time.monotonic_ns)
            holder.append(files)
            schedule, pcm, binding, fmt = _fixture(case)
            if len(pcm) > 1536000 or 2 * len(pcm) > LIMITS["scratch_bytes"]:
                raise ValueError("fixed fixture/scratch bound exceeded")
            record["fixture"] = {"source_fingerprint": schedule.source_fingerprint,
                "decoder_identity": "generated-pcm-v1", "source_decode_receipt": None,
                "frames_per_channel": schedule.total_samples, "channels": fmt.channels,
                "sample_rate": fmt.sample_rate, "enabled_bytes": len(pcm),
                "enabled_sha256": hashlib.sha256(pcm).hexdigest(),
                "selected_pcm_sha256": None if binding is None else binding.selected_pcm_sha256,
                "maximum_owned_scratch_bytes": 2 * len(pcm)}
            try:
                result = prepare_normalized_audio(schedule, pcm, binding=binding,
                    audio_format=fmt, runtime=runtime, job=job, files=files)
            except ValueError as exc:
                if case == "rejected_binding" and type(exc) is ValueError and str(exc) == _REJECTION:
                    record["expected_rejection"] = True
                    return
                raise
            if case in CASES[:2]:
                if result is None:
                    raise ValueError("positive returned no result")
                record.update(_positive(result, files, scratch, schedule, pcm, binding, fmt, runtime))
            elif result is not None or case != "disabled":
                raise ValueError("negative case unexpectedly returned")
        except Exception as exc:
            record["error"] = _error(exc)
            if job.snapshot().failure is None:
                job.fail("validation", "driver")
        finally:
            if holder:
                try:
                    holder[0].retire()
                except Exception as exc:
                    record["cleanup_errors"].append(_error(exc))
            try:
                job.driver_done()
            except Exception as exc:
                record["cleanup_errors"].append(_error(exc))

    driver = threading.Thread(target=driver_target, name=f"normalization-{case}", daemon=False)
    job = SourceHostJob(driver, deadline_ns=deadline,
                        cleanup_grace_ns=5000000000, clock=time.monotonic_ns)
    try:
        job.start_driver()
    except Exception as exc:
        if record["error"] is None:
            record["error"] = _error(exc)
    while True:
        status = job.service_cleanup()
        if status.own0 or time.monotonic_ns() >= finish_deadline:
            break
        # Original owner deadline is never extended by this polling loop.
        if status.failure is not None and time.monotonic_ns() >= status.failure.cleanup_deadline_ns:
            break
        time.sleep(.01)
    status = job.snapshot()
    final = dict(record)
    final.update(source_job=status, own0=status.own0, elapsed_ns=time.monotonic_ns() - started,
                 job_deadline_ns=deadline, poll_finish_deadline_ns=finish_deadline)
    if holder:
        files = holder[0]
        file_status = files.snapshot()
        controls = files._input_consumers
        if len(controls) > 2:
            raise ValueError("bounded input consumer diagnostic set exceeded")
        snapshots = tuple(control.ownership_snapshot() for control in controls)
        if any(len(snapshot.diagnostics) > 2 for snapshot in snapshots):
            raise ValueError("bounded pass diagnostic set exceeded")
        final.update(files=file_status, files_clear=file_status.files_clear,
            file_generations=len(file_status.rows), controls=snapshots,
            window_resources_zero=all(s.window_resources_zero and not s.cleanup_required
                and not s.scope_open and not s.consumer_active and not s.payload_retained
                and not s.lease_outstanding for s in snapshots),
            process_count=sum(row.role == "process" and row.pid is not None
                for s in snapshots for row in s.resources),
            process_launch_attempts=sum(len(s.diagnostics) for s in snapshots),
            pass_count=sum(d.natural_eof and d.exit_code == 0
                for s in snapshots for d in s.diagnostics))
    if status.failure is not None and final["error"] is None:
        final["error"] = {"type": "HostFailure", "message": status.failure.code}
    return final


def _default_runner() -> CaseRunner:
    from clipperx.render_runtime import NATIVE_RUNTIME, RenderRequirements, resolve_render_runtime
    started = time.monotonic_ns()
    runtime = resolve_render_runtime(RenderRequirements(NATIVE_RUNTIME, False))
    setup_ns = time.monotonic_ns() - started
    def run(case: str, scratch: Path, deadline: int) -> dict[str, Any]:
        record = _run_real_case(case, scratch, deadline, runtime)
        record["runtime_setup"] = {"scope": "resolver_reads_outside_bridge",
            "elapsed_ns": setup_ns, "runtime_id": runtime.runtime_id,
            "ffmpeg": str(runtime.ffmpeg), "ffprobe": str(runtime.ffprobe),
            "ffmpeg_sha256": runtime.ffmpeg_sha256, "ffprobe_sha256": runtime.ffprobe_sha256,
            "device_index": runtime.device_index}
        return record
    return run


def _status(case: str, record: dict[str, Any]) -> str:
    if any(record.get(k) is not True for k in ("own0", "window_resources_zero", "files_clear")):
        return "cleanup_required"
    if (record.get("case_id") != case or record.get("error") is not None
            or record.get("cleanup_errors") != [] or type(record.get("inference_calls")) is not int
            or record.get("inference_calls") != 0):
        return "FAIL"
    if case in CASES[:2]:
        generations = record.get("file_generations")
        if type(generations) is not int or not 1 <= generations <= 16:
            return "FAIL"
        mode, samples, size = (("linear", 48000, 192000) if case == "linear_mono"
                               else ("dynamic", 192000, 1536000))
        expected = {"mode": mode, "observed_mode": mode, "samples": samples,
            "output_bytes": size, "process_count": 2, "pass_count": 2,
            "finite": True, "association_verified": True, "result_present": True,
            "expected_rejection": False}
    else:
        expected = {"result_present": False, "process_count": 0, "pass_count": 0,
            "file_generations": 0, "expected_rejection": case == "rejected_binding"}
    if any(type(record.get(k)) is not type(v) or record.get(k) != v for k, v in expected.items()):
        return "FAIL"
    return "PASS"


def main(argv: Sequence[str] | None = None, *, _case_runner: CaseRunner | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--guard", type=Path, required=True)
    parser.add_argument("--confirm-host", action="store_true", required=True)
    args = parser.parse_args(argv)
    try:
        _, raw = _manifest(args.manifest)
        run, guard = _destinations(args.run_dir, args.guard)
        digest = hashlib.sha256(raw).hexdigest().upper()
        guard.parent.mkdir(parents=True, exist_ok=True)
        with guard.open("xb") as stream:
            stream.write(_json_bytes({"manifest_sha256": digest, "run_id": run.name}, 4096))
        run.mkdir(parents=True, exist_ok=False)
    except (ValueError, OSError, TypeError, UnicodeError):
        print("FAIL: qualification preflight")
        return 2
    started = time.monotonic_ns()
    deadline = started + LIMITS["campaign_seconds"] * 1000000000
    used = 0

    def write_encoded(path: Path, data: bytes, cap: int) -> None:
        nonlocal used
        if len(data) > cap or used + len(data) > LIMITS["total_evidence_bytes"]:
            raise ValueError("total evidence cap exceeded")
        with path.open("xb") as stream:
            stream.write(data)
        used += len(data)

    def write(path: Path, value: Any, cap: int) -> None:
        write_encoded(path, _json_bytes(value, cap), cap)

    summary: dict[str, Any] = {"status": "FAIL", "run_id": run.name,
        "manifest_sha256": digest, "completed_cases": [], "failed_case": None,
        "process_count": 0, "pass_count": 0, "inference_calls": 0,
        "python": platform.python_version(), "pydantic": None, "cases": [],
        "limits": LIMITS, "evidence_kind": "real_owned_normalization" if _case_runner is None
        else "trusted_fake_orchestration_only"}
    resources_may_remain = False
    try:
        import pydantic
        summary["pydantic"] = pydantic.__version__
        write_encoded(run / "manifest.json", raw, 131072)
        runner = _default_runner() if _case_runner is None else _case_runner
        for case in CASES:
            summary["failed_case"] = case
            if time.monotonic_ns() >= deadline:
                raise ValueError("campaign deadline before next case")
            directory = run / case
            directory.mkdir()
            scratch = directory / "scratch"
            scratch.mkdir()
            resources_may_remain = _case_runner is None
            record = runner(case, scratch, deadline)
            resources_may_remain = any(record.get(key) is not True
                for key in ("own0", "window_resources_zero", "files_clear"))
            status = _status(case, record)
            record["status"] = status
            write(directory / "case.json", record, LIMITS["case_json_bytes"])
            summary["cases"].append({"case_id": case, "status": status,
                                     "evidence": f"{case}/case.json"})
            summary["process_count"] += record["process_count"]
            summary["pass_count"] += record["pass_count"]
            summary["inference_calls"] += record["inference_calls"]
            if status != "PASS":
                summary["status"] = status
                break
            summary["completed_cases"].append(case)
        else:
            summary["status"], summary["failed_case"] = "PASS", None
    except Exception as exc:
        summary["error"] = _error(exc)
        summary["status"] = "cleanup_required" if resources_may_remain else "FAIL"
    summary["elapsed_ns"] = time.monotonic_ns() - started
    summary["evidence_bytes_before_summary"] = used
    try:
        write(run / "summary.json", summary, 131072)
    except (ValueError, OSError):
        print(f"FAIL: {run / 'summary.json'}")
        return 1
    print(f"{summary['status']}: {run / 'summary.json'}")
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    raise SystemExit(main())
