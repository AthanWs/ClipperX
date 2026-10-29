"""Pure normalization transport checks with finite synthetic PCM and no HOST work."""

from array import array
import ctypes
from dataclasses import replace
from fractions import Fraction
import hashlib
from pathlib import Path
import subprocess
import sys
import threading
from typing import Any

import pytest


def forbidden(*args: object, **kwargs: object) -> Any:
    raise AssertionError("normalization test reached real HOST work")


@pytest.fixture
def audio(monkeypatch: pytest.MonkeyPatch) -> Any:
    # Fail-only process/native seams precede the lazy production import.
    for module, names in (
        (subprocess, ("Popen", "run", "call", "check_call", "check_output")),
        (ctypes, ("WinDLL", "CDLL", "PyDLL", "OleDLL")),
        (threading.Thread, ("start", "join")),
        (threading.Timer, ("start",)), (threading.Event, ("wait",)),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden, raising=False)
    from clipperx import audio_mix, render_runtime
    for module in (audio_mix, render_runtime):
        monkeypatch.setattr(module, "_hash", forbidden)
        monkeypatch.setattr(module, "_verify_pins", forbidden)
    for name in ("resolve_render_runtime", "qualify_render_runtime", "_run"):
        monkeypatch.setattr(render_runtime, name, forbidden)
    return audio_mix


@pytest.fixture
def bus(audio: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    from clipperx.render_runtime import RenderRuntime
    values = array("f", [.125]) * 19200
    if sys.byteorder != "little":
        values.byteswap()
    pcm = values.tobytes()
    schedule = audio.AudioSchedule("pure-source", (
        audio.AudioSpan(Fraction(10), Fraction(52, 5), Fraction(0), Fraction(2, 5),
                        0, 19200, True, 0., False),), ())
    binding = audio.SourceClockBinding("pure-source", "pure-decoder",
                                       hashlib.sha256(pcm).hexdigest())
    fmt = audio.AudioFormat(48000, "mono")
    runtime = RenderRuntime(Path("C:/PURE/ffmpeg.exe"), Path("C:/PURE/ffprobe.exe"),
        "system", "a" * 64, "b" * 64, None, None, "-filter_complex_script", None)

    def fake_hash(path: Path) -> str:
        return {runtime.ffmpeg: runtime.ffmpeg_sha256,
                runtime.ffprobe: runtime.ffprobe_sha256}[path]

    def fake_pins(value: object) -> None:
        assert value is runtime

    # Only the filesystem/pin boundary is faked; PCM/receipt/parser checks stay real.
    monkeypatch.setattr(audio, "_hash", fake_hash)
    monkeypatch.setattr(audio, "_verify_pins", fake_pins)
    return schedule, pcm, binding, fmt, runtime


OBSERVATION = (
    b'{"input_i":"-20.0","input_lra":"5.0","input_tp":"-10.0",'
    b'"input_thresh":"-30.0","target_offset":"0.0","normalization_type":"linear",'
    b'"output_i":"-14.0","output_lra":"5.0","output_tp":"-4.0"}'
)
MEASURE_FILTER = "loudnorm=I=-14:LRA=11:TP=-1.5:print_format=json"
PROCESS_FILTER = (
    "loudnorm=I=-14:LRA=11:TP=-1.5:measured_I=-20:measured_LRA=5:measured_TP=-10:"
    "measured_thresh=-30:offset=0:linear=true:print_format=json,aresample=48000"
)


def expected_argv(runtime: Any, filters: str, *, measure: bool) -> tuple[str, ...]:
    return (str(runtime.ffmpeg), "-hide_banner", "-loglevel", "info", "-f", "f32le",
        "-ar", "48000", "-ac", "1", "-channel_layout", "mono", "-i", "pipe:0",
        "-af", filters, *(("-f", "null", "-") if measure else
                          ("-c:a", "pcm_f32le", "-f", "f32le", "pipe:1")))


def receipt_for(audio: Any, bus: Any) -> Any:
    schedule, _, binding, fmt, runtime = bus
    stats = audio.LoudnessStats(-20., 5., -10., -30., 0.)
    mode, reason = audio.select_normalization_mode(stats)
    return audio.NormalizationReceipt(schedule.identity, binding, audio._runtime_identity(runtime),
        fmt, 19200, stats, mode, reason)


def legacy_counter(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    calls = [0]

    def unexpected(*args: object, **kwargs: object) -> Any:
        calls[0] += 1
        return forbidden(*args, **kwargs)

    monkeypatch.setattr(subprocess, "run", unexpected)
    return calls


@pytest.mark.parametrize("helper", ["measure", "process"])
def test_supplied_runner_uses_real_normalization_helpers(
        audio: Any, bus: Any, monkeypatch: pytest.MonkeyPatch, helper: str) -> None:
    schedule, pcm, binding, fmt, runtime = bus
    calls: list[tuple[tuple[str, ...], bytes, bool]] = []
    default_calls = legacy_counter(monkeypatch)
    output = array("f", [.25]).tobytes() * 19200
    result = subprocess.CompletedProcess[bytes]((), 0, output, OBSERVATION)

    def runner(argv: tuple[str, ...], data: bytes, *, measure: bool
               ) -> subprocess.CompletedProcess[bytes]:
        calls.append((argv, data, measure))
        return result

    if helper == "measure":
        receipt = audio.measure_normalization(schedule, pcm, binding=binding,
            audio_format=fmt, runtime=runtime, _runner=runner)
        assert receipt == receipt_for(audio, bus)
        assert receipt.scope == "mixed_track_normalization"
    else:
        normalized = audio.process_normalized_pcm(schedule, pcm, receipt=receipt_for(audio, bus),
            binding=binding, audio_format=fmt, runtime=runtime, _runner=runner)
        assert normalized.pcm is output
        assert normalized.observed_mode == "linear"
        assert normalized.samples == 19200
        assert normalized.upstream_observation == audio.LoudnessStats(-20., 5., -10., -30., 0.)
        assert normalized.output_loudness == (-14., 5., -4.)
    assert len(calls) == 1
    argv, data, measure = calls[0]
    assert type(argv) is tuple
    assert argv == expected_argv(runtime,
        MEASURE_FILTER if helper == "measure" else PROCESS_FILTER, measure=helper == "measure")
    assert data is pcm
    assert measure is (helper == "measure")
    assert default_calls == [0]


@pytest.mark.parametrize("measure", [False, True])
def test_legacy_default_keeps_exact_call_and_result_identity(
        audio: Any, bus: Any, monkeypatch: pytest.MonkeyPatch, measure: bool) -> None:
    _, pcm, _, fmt, runtime = bus
    calls: list[tuple[object, dict[str, object]]] = []
    result = subprocess.CompletedProcess[bytes]((), 0, b"finite-result", b"finite-log")

    def fake_run(args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert audio._run_pcm(runtime, pcm, fmt, "literal-filter", measure=measure) is result
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert type(args) is list
    assert args == list(expected_argv(runtime, "literal-filter", measure=measure))
    assert kwargs == dict(input=pcm, capture_output=True, timeout=45, check=False)
    assert kwargs["input"] is pcm


@pytest.mark.parametrize("failure", ["os", "timeout", "value", "nonzero"])
@pytest.mark.parametrize("helper", ["measure", "process", "legacy"])
def test_runner_failure_preserves_errors_without_legacy_retry(
        audio: Any, bus: Any, monkeypatch: pytest.MonkeyPatch,
        failure: str, helper: str) -> None:
    schedule, pcm, binding, fmt, runtime = bus
    default_calls = legacy_counter(monkeypatch)
    calls = [0]
    error = {"os": OSError("synthetic OS failure"),
             "timeout": subprocess.TimeoutExpired("synthetic-command", 45),
             "value": ValueError("synthetic value failure")}.get(failure)

    def fail_result() -> subprocess.CompletedProcess[bytes]:
        calls[0] += 1
        if error is not None:
            raise error
        return subprocess.CompletedProcess[bytes]((), 7, b"", b"synthetic nonzero")

    def runner(argv: tuple[str, ...], data: bytes, *, measure: bool
               ) -> subprocess.CompletedProcess[bytes]:
        return fail_result()

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return fail_result()

    def invoke() -> Any:
        if helper == "legacy":
            return audio._run_pcm(runtime, pcm, fmt, MEASURE_FILTER)
        if helper == "measure":
            return audio.measure_normalization(schedule, pcm, binding=binding,
                audio_format=fmt, runtime=runtime, _runner=runner)
        return audio.process_normalized_pcm(schedule, pcm, receipt=receipt_for(audio, bus),
            binding=binding, audio_format=fmt, runtime=runtime, _runner=runner)

    if helper == "legacy":
        monkeypatch.setattr(subprocess, "run", fake_run)
    if failure == "value":
        with pytest.raises(ValueError) as exc:
            invoke()
        assert exc.value is error
    else:
        with pytest.raises(audio.AudioPreparationError) as typed:
            invoke()
        assert typed.value.code == "tool_failure"
        if error is not None:
            assert typed.value.__cause__ is error
        else:
            assert "synthetic nonzero" in str(typed.value)
    assert calls == [1]
    assert default_calls == [0]


@pytest.mark.parametrize("control", ["stats", "mode", "count", "receipt"])
def test_injected_helpers_keep_real_rejection_checks(
        audio: Any, bus: Any, monkeypatch: pytest.MonkeyPatch, control: str) -> None:
    schedule, pcm, binding, fmt, runtime = bus
    default_calls = legacy_counter(monkeypatch)
    calls = [0]
    receipt = receipt_for(audio, bus)
    observation = b"{}" if control == "stats" else OBSERVATION
    if control == "mode":
        observation = OBSERVATION.replace(b'"linear"', b'"dynamic"')
    if control == "receipt":
        receipt = replace(receipt, mode="dynamic")
    output = pcm[:-4] if control == "count" else pcm

    def runner(argv: tuple[str, ...], data: bytes, *, measure: bool
               ) -> subprocess.CompletedProcess[bytes]:
        calls[0] += 1
        return subprocess.CompletedProcess[bytes](argv, 0, output, observation)

    with pytest.raises(audio.AudioPreparationError) as exc:
        if control == "stats":
            audio.measure_normalization(schedule, pcm, binding=binding,
                audio_format=fmt, runtime=runtime, _runner=runner)
        else:
            audio.process_normalized_pcm(schedule, pcm, receipt=receipt, binding=binding,
                audio_format=fmt, runtime=runtime, _runner=runner)
    assert exc.value.code == {"stats": "malformed_measurement", "mode": "mode_mismatch",
        "count": "sample_count_mismatch", "receipt": "mode_mismatch"}[control]
    assert calls == [0 if control == "receipt" else 1]
    assert default_calls == [0]
