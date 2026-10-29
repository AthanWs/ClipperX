"""Pure declaration/graph tests. Every fixture path is metadata, never media."""

import builtins
import ctypes
from dataclasses import replace
from fractions import Fraction
import io
import ntpath
import os
from pathlib import Path, PosixPath, WindowsPath
import re
import subprocess
import sys
import threading
from types import MappingProxyType
from typing import Any, cast

import pytest


def forbidden(*args: object, **kwargs: object) -> Any:
    raise AssertionError("pure audio compilation reached HOST work")


_IO_WORKSPACE = str(Path(__file__).parents[2])
_IO_CODE_ROOTS = tuple(ntpath.normcase(ntpath.normpath(root)) for root in (
    str(Path(__file__).parents[1]), sys.prefix, sys.base_prefix))
_IO_CODE_ANCESTORS = frozenset(ntpath.normcase(str(parent)) for root in
    (Path(__file__).parents[1], Path(sys.prefix), Path(sys.base_prefix))
    for parent in (root, *root.parents))


def _code_io_path(value: object, *, metadata: bool) -> bool:
    # Lexical only: never stat/resolve/open an input to decide whether it is safe.
    if type(value) not in (str, WindowsPath, PosixPath):
        return False
    text = str(value)
    if not text or len(text) > 4096 or text.startswith(("\\\\", "//")) or "\0" in text:
        return False
    drive, _ = ntpath.splitdrive(text)
    if drive and not ntpath.isabs(text):
        return False
    normalized = ntpath.normcase(ntpath.normpath(text if ntpath.isabs(text)
        else ntpath.join(_IO_WORKSPACE, text)))
    if normalized == ntpath.normcase(ntpath.join(_IO_WORKSPACE, "clipperx.egg-info", "entry_points.txt")):
        return True
    if metadata and normalized == ntpath.normcase(ntpath.join(sys.base_prefix,
            f"python{sys.version_info.major}{sys.version_info.minor}.zip")):
        return True
    if metadata and normalized in _IO_CODE_ANCESTORS:
        return True
    for root in _IO_CODE_ROOTS:
        if not normalized.startswith(root + "\\"):
            continue
        relative = normalized[len(root) + 1:]
        # Pydantic's code import discovers installed Python plugins through this
        # one distribution metadata filename. No other data files are permitted.
        if (root != _IO_CODE_ROOTS[0] and re.fullmatch(
                r"lib\\site-packages\\[a-z0-9_+.-]{1,200}\.dist-info\\entry_points\.txt", relative)):
            return True
        parts = relative.split("\\")
        directories = parts[:-1]
        if not all(re.fullmatch(r"[a-z0-9_-]{1,128}", part) for part in directories):
            return False
        if re.fullmatch(r"[a-z0-9_.-]{1,200}\.(?:py|pyc|pyd)", parts[-1]):
            return True
        return metadata and bool(re.fullmatch(r"[a-z0-9_-]{1,128}", parts[-1]))
    return False


def _guard_test_io(original: Any, operation: str) -> Any:
    def checked(path: object, *args: object, **kwargs: object) -> Any:
        if operation == "open":
            parameters = ("mode", "buffering", "encoding", "errors", "newline", "closefd", "opener")
            if (len(args) > len(parameters) or any(name not in parameters for name in kwargs)
                    or any(name in kwargs for name in parameters[:len(args)])):
                return forbidden(path)
            mode = args[0] if args else kwargs.get("mode", "r")
            opener = args[6] if len(args) == 7 else kwargs.get("opener")
            if (type(mode) is not str or mode not in ("r", "rb", "rt", "br", "tr")
                    or opener is not None):
                return forbidden(path)
        elif operation == "os_open":
            flags = args[0] if args else kwargs.get("flags")
            read_flags = (os.O_RDONLY | os.O_BINARY | os.O_TEXT | os.O_NOINHERIT
                          | getattr(os, "O_CLOEXEC", 0))
            if (type(flags) is not int or flags < 0 or flags & ~read_flags
                    or args and "flags" in kwargs):
                return forbidden(path)
        elif operation not in ("stat", "lstat", "resolve"):
            return forbidden(path)
        if not _code_io_path(path, metadata=operation in ("stat", "lstat", "resolve")):
            if type(path) in (str, WindowsPath, PosixPath):
                raise AssertionError("pure audio compilation reached HOST work: denied " + str(path)[:4096])
            return forbidden(path)
        return original(path, *args, **kwargs)
    return checked


@pytest.fixture(autouse=True)
def host_guard(monkeypatch: pytest.MonkeyPatch) -> Any:
    # Install process/native/worker boundaries BEFORE the lazy production import.
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    module: Any
    names: tuple[str, ...]
    for module, names in (
        (subprocess, ("Popen", "run", "call", "check_call", "check_output")),
        (os, ("system", "popen", "startfile", "spawnl", "spawnle", "spawnlp", "spawnlpe",
              "spawnv", "spawnve", "spawnvp", "spawnvpe")),
        (ctypes, ("WinDLL", "CDLL", "PyDLL", "OleDLL")),
        (threading.Thread, ("start", "join")),
        (threading.Timer, ("start",)), (threading.Event, ("wait",)),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden, raising=False)

    for module, names in ((builtins, ("open",)), (io, ("open",)),
                          (os, ("open", "stat", "lstat")),
                          (Path, ("open", "stat", "lstat", "resolve"))):
        for name in names:
            operation = "os_open" if module is os and name == "open" else name
            monkeypatch.setattr(module, name, _guard_test_io(getattr(module, name), operation))
    from clipperx import audio_mix as audio
    from clipperx import local_paths, render_assets, render_runtime, source_decode
    from clipperx import _source_host_ownership, _windows_pipe_io
    for module, names in (
        (audio, ("_run_pcm", "measure_normalization", "process_normalized_pcm", "_hash", "_verify_pins")),
        (render_runtime, ("resolve_render_runtime", "qualify_render_runtime", "_run", "_hash", "_verify_pins")),
        (render_assets, ("validate_asset_loadability", "prepare_brand_asset", "default_render_assets")),
        (local_paths, ("source_fingerprint",)),
        (source_decode, ("probe_source", "observe_float_audio_window", "observe_audio_window",
                         "observe_windows", "stream_observed_window", "stream_payloads", "payload_stream")),
        (_source_host_ownership, ("_thread_worker", "_popen_source", "WindowsSyncIo")),
        (_windows_pipe_io, ("WindowsSyncIo",)),
    ):
        for name in names:
            monkeypatch.setattr(module, name, forbidden)
    yield audio


def test_tail_guard_blocks_host_work(host_guard: Any) -> None:
    from clipperx import render_assets, render_runtime, source_decode, _source_host_ownership, _windows_pipe_io
    audio = host_guard
    def unexpected_original(*args: object, **kwargs: object) -> Any:
        raise AssertionError("unguarded IO reached original")
    # Originals are fail-only BEFORE probing even a broken wrapper. No disk IO.
    for operation in ("open", "os_open", "stat", "lstat", "resolve"):
        guarded = _guard_test_io(unexpected_original, operation)
        for ordinary in ("ordinary-audio.wav", "C:/ordinary/audio.wav", "clipperx/private.json",
                         object(), b"clipperx/audio_mix.py", 0):
            with pytest.raises(AssertionError, match="HOST work"):
                guarded(ordinary, os.O_RDONLY) if operation == "os_open" else guarded(ordinary)
    code = "clipperx/audio_mix.py"
    guarded_open = _guard_test_io(unexpected_original, "open")
    def unexpected_opener(*args: object, **kwargs: object) -> Any:
        raise AssertionError("custom opener unexpectedly invoked")
    # Both original and opener are fail-only BEFORE testing the broken branch.
    for positional, keywords in (
        (("r", -1, None, None, None, True, unexpected_opener), {}),
        (("r",), {"opener": unexpected_opener}),
        (("r", -1, None, None, None, True, None), {"opener": None}),
        (("r", -1), {"buffering": -1}),
        (("r", -1, None), {"encoding": None}),
        (("r", -1, None, None, None, True, None, None), {}),
        (("r",), {"file": code}),
        (("r",), {"unknown_option": None}),
    ):
        with pytest.raises(AssertionError, match="HOST work"):
            guarded_open(code, *positional, **keywords)
    for mode in ("w", "wb", "a", "ab", "x", "xb", "r+", "rb+", "w+", "a+b", "x+b", object(), True):
        with pytest.raises(AssertionError, match="HOST work"):
            guarded_open(code, mode)
    for flag in (os.O_WRONLY, os.O_RDWR, os.O_CREAT, os.O_TRUNC, os.O_APPEND, os.O_EXCL,
                 os.O_RDONLY | os.O_CREAT, True, object(), -1):
        with pytest.raises(AssertionError, match="HOST work"):
            _guard_test_io(unexpected_original, "os_open")(code, flag)
    # Fake originals prove permitted direction without ever reading the source.
    called: list[object] = []
    def source_original(path: object, *args: object, **kwargs: object) -> str:
        called.append(path)
        return "CODE_READ_SENTINEL"
    for operation in ("open", "os_open", "stat", "lstat", "resolve"):
        guarded = _guard_test_io(source_original, operation)
        assert (guarded(code, os.O_RDONLY) if operation == "os_open" else guarded(code)) == "CODE_READ_SENTINEL"
    assert len(called) == 5
    for mode in ("r", "rb", "rt", "br", "tr"):
        assert _guard_test_io(source_original, "open")(code, mode=mode) == "CODE_READ_SENTINEL"
    assert _guard_test_io(source_original, "open")(
        code, "r", -1, None, None, None, True, None) == "CODE_READ_SENTINEL"
    with pytest.raises(AssertionError, match="HOST work"):
        guarded_open(code, "r", opener=source_original)
    for call in (lambda: subprocess.run([]), lambda: subprocess.Popen([]),
                 lambda: ctypes.WinDLL("PURE.dll"), lambda: threading.Thread().start(),
                 lambda: threading.Thread().join(), lambda: threading.Timer(1, forbidden).start(),
                 lambda: threading.Event().wait(), lambda: Path("PURE.wav").stat(),
                 lambda: Path("PURE.wav").lstat(), lambda: Path("PURE.wav").resolve(),
                 lambda: Path(r"\\PURE\invalid\audio.wav").resolve(),
                 lambda: open("PURE.wav"), lambda: io.open("PURE.wav"), lambda: os.stat("PURE.wav"),
                 lambda: audio._run_pcm(), lambda: audio.measure_normalization(),
                 lambda: cast(Any, render_runtime.resolve_render_runtime)(),
                 lambda: cast(Any, render_runtime.qualify_render_runtime)(),
                 lambda: cast(Any, source_decode.probe_source)(), lambda: cast(Any, source_decode.observe_float_audio_window)(),
                 lambda: cast(Any, render_assets.validate_asset_loadability)(), lambda: cast(Any, _source_host_ownership._thread_worker)(),
                 lambda: cast(Any, _source_host_ownership._popen_source)(), lambda: _windows_pipe_io.WindowsSyncIo()):
        with pytest.raises(AssertionError, match="HOST work"):
            call()
    with pytest.raises(ValueError):
        audio.compile_prepared_audio(cast(Any, None), schedule_identity="0" * 64,
                                     audio_format=audio.AudioFormat(48000, "stereo"))


def plan(*, modes: tuple[bool, ...] | None = None, duration: float = 1.) -> Any:
    from clipperx.visual_contracts import VisualCompositionPlan
    shots = []
    for i in range(len(modes) if modes is not None else 1):
        a, b = float(Fraction(str(duration)) * i), float(Fraction(str(duration)) * (i + 1))
        source = 10. * (i + 1)
        shot = dict(shot_id=f"s{i}", source=dict(start=source, end=source + duration),
            output=dict(start=a, end=b), primary_source="content", reason="pure declaration",
            evidence_ids=["declared"], confidence=1., layers=[dict(layer_id="root", kind="solid", source_ref="source",
                destination=dict(x=0, y=0, width=1, height=1), z_index=0,
                fill="solid", solid_color="#112233")])
        if modes is not None:
            shot["audio"] = dict(normalize_dialogue=modes[i], limiter=False)
        shots.append(shot)
    return VisualCompositionPlan.model_validate(dict(clip_id="pure", shots=shots,
        source_fingerprint="0123456789abcdef", composition_fingerprint="declared",
        source_timeline_fingerprint="declared", identity_fingerprint="declared",
        style_library_fingerprint="declared"))


def context(audio: Any, p: Any, layout: str = "stereo") -> dict[str, Any]:
    from clipperx.render_runtime import RenderRuntime
    fmt = audio.AudioFormat(48000, layout)
    runtime = RenderRuntime(Path("C:/PURE/ffmpeg.exe"), Path("C:/PURE/ffprobe.exe"),
        "system", "a" * 64, "b" * 64, None, None, "-filter_complex_script", None)
    schedule = audio.schedule_audio(p)
    result: dict[str, Any] = dict(audio_format=fmt)
    if schedule.enabled_samples:
        binding = audio.SourceClockBinding(p.source_fingerprint, "declared-decoder", "c" * 64)
        stats = audio.LoudnessStats(-20., 5., -10., -30., 0.)
        mode, reason = audio.select_normalization_mode(stats)
        receipt = audio.NormalizationReceipt(schedule.identity, binding, audio._runtime_identity(runtime),
            fmt, schedule.enabled_samples, stats, mode, reason)
        result.update(binding=binding, runtime=runtime, receipt=receipt)
    return result


def prepared(audio: Any, p: Any, **changes: Any) -> Any:
    schedule = audio.schedule_audio(p)
    args = context(audio, p)
    args.update(schedule_identity=schedule.identity,
        selected_input_label="0:a" if schedule.enabled_samples < schedule.total_samples else None,
        normalized_input_label=("1:a" if schedule.enabled_samples < schedule.total_samples else "0:a")
            if schedule.enabled_samples else None)
    args.update(changes)
    return audio.compile_prepared_audio(p, **args)


DEFAULT_GRAPH = (
    "[0:a]anull[src0];[src0]atrim=start=10:end=11,asetpts=PTS-10/TB,aresample=48000,"
    "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,atrim=end_sample=48000[cut0];"
    "[cut0]anull[selected];[selected]anull[part0];"
    "[part0]atrim=start_sample=0:end_sample=48000,asetpts=N/48000/TB[raw0];"
    "[raw0]anull[normbus];[normbus]loudnorm=I=-14:LRA=11:TP=-1.5:measured_I=-20:"
    "measured_LRA=5:measured_TP=-10:measured_thresh=-30:offset=0:linear=true:print_format=json,"
    "aresample=48000[normalized];[normalized]anull[normpart0];"
    "[normpart0]atrim=start_sample=0:end_sample=48000,asetpts=N/48000/TB[restore0];"
    "[restore0]volume=0dB[gain0];[gain0]anull[gained];[gained]anull[mixed];[mixed]anull[limpart0];"
    "[limpart0]atrim=start_sample=0:end_sample=48000,asetpts=N/48000/TB,"
    "alimiter=limit=0.944:level=0:latency=1:level_in=1:level_out=1:attack=5:release=50:asc=0,anull[limited0];"
    "[limited0]anull[finalbus];[finalbus]atrim=end_sample=48000,asetpts=N/48000/TB[audio_out]"
)

DUCK_ONE = (
    "aeval='val(ch)*pow(10,(min(0,(-6)*(min(1,min(max(0,(n/48000-0.1)/0.005),"
    "max(0,(0.3-n/48000)/0.01))))*(min(1,min(max(0,(n/48000-0.1)/0.005),"
    "max(0,(0.3-n/48000)/0.01))))*(3-2*(min(1,min(max(0,(n/48000-0.1)/0.005),"
    "max(0,(0.3-n/48000)/0.01)))))))/20)':c=same"
)

DUCK_TWO = (
    "aeval='val(ch)*pow(10,(min(min(0,(-6)*(min(1,min(max(0,(n/48000-0.1)/0.005),"
    "max(0,(0.3-n/48000)/0.01))))*(min(1,min(max(0,(n/48000-0.1)/0.005),"
    "max(0,(0.3-n/48000)/0.01))))*(3-2*(min(1,min(max(0,(n/48000-0.1)/0.005),"
    "max(0,(0.3-n/48000)/0.01)))))),(-12)*(min(1,min(max(0,(n/48000-0.2)/0.005),"
    "max(0,(0.4-n/48000)/0.01))))*(min(1,min(max(0,(n/48000-0.2)/0.005),"
    "max(0,(0.4-n/48000)/0.01))))*(3-2*(min(1,min(max(0,(n/48000-0.2)/0.005),"
    "max(0,(0.4-n/48000)/0.01)))))))/20)':c=same"
)

NONDEFAULT_GRAPH = (
    "[0:a]asplit=3[src0][src1][src2];"
    "[src0]atrim=start=10:end=11,asetpts=PTS-10/TB,aresample=48000,"
    "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,atrim=end_sample=48000[cut0];"
    "[src1]atrim=start=20:end=21,asetpts=PTS-20/TB,aresample=48000,"
    "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,atrim=end_sample=48000[cut1];"
    "[src2]atrim=start=30:end=31,asetpts=PTS-30/TB,aresample=48000,"
    "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo,atrim=end_sample=48000[cut2];"
    "[cut0][cut1][cut2]concat=n=3:v=0:a=1[selected];[selected]asplit=3[part0][part1][part2];"
    "[part0]atrim=start_sample=0:end_sample=48000,asetpts=N/48000/TB[raw0];"
    "[part1]atrim=start_sample=48000:end_sample=96000,asetpts=N/48000/TB[raw1];"
    "[part2]atrim=start_sample=96000:end_sample=144000,asetpts=N/48000/TB[raw2];"
    "[raw0][raw2]concat=n=2:v=0:a=1[normbus];[normbus]loudnorm=I=-14:LRA=11:TP=-1.5:"
    "measured_I=-20:measured_LRA=5:measured_TP=-10:measured_thresh=-30:offset=0:linear=true:"
    "print_format=json,aresample=48000[normalized];[normalized]asplit=2[normpart0][normpart1];"
    "[normpart0]atrim=start_sample=0:end_sample=48000,asetpts=N/48000/TB[restore0];"
    "[normpart1]atrim=start_sample=48000:end_sample=96000,asetpts=N/48000/TB[restore2];"
    "[restore0]volume=-1.125dB[gain0];[raw1]volume=-6dB[gain1];[restore2]volume=2.25dB[gain2];"
    "[gain0][gain1][gain2]concat=n=3:v=0:a=1[gained];[gained]" + DUCK_ONE + "[ducked];"
    "[1:a]anull[cueasset1_0];[cueasset1_0]atrim=end_sample=9600,asetpts=N/48000/TB,"
    "apad=whole_len=9600,atrim=end_sample=9600,volume=-3dB,adelay=4800S:all=1[cue0];"
    "[ducked][cue0]amix=inputs=2:duration=first:weights='1':normalize=0:dropout_transition=0[mixed];"
    "[mixed]asplit=3[limpart0][limpart1][limpart2];"
    "[limpart0]atrim=start_sample=0:end_sample=48000,asetpts=N/48000/TB,"
    "alimiter=limit=0.944:level=0:latency=1:level_in=1:level_out=1:attack=5:release=50:asc=0,anull[limited0];"
    "[limpart1]atrim=start_sample=48000:end_sample=96000,asetpts=N/48000/TB,anull[limited1];"
    "[limpart2]atrim=start_sample=96000:end_sample=144000,asetpts=N/48000/TB,"
    "alimiter=limit=0.944:level=0:latency=1:level_in=1:level_out=1:attack=5:release=50:asc=0,anull[limited2];"
    "[limited0][limited1][limited2]concat=n=3:v=0:a=1[finalbus];"
    "[finalbus]atrim=end_sample=144000,asetpts=N/48000/TB[audio_out]"
)


def cue_context(audio: Any, p: Any) -> dict[str, Any]:
    from clipperx.render_assets import RenderAssetRegistry
    return dict(assets=RenderAssetRegistry({}, {}, {"impact": Path("C:/PURE/impact.wav")}),
                cue_formats={"impact": audio.AudioFormat(48000, "stereo")})


def nondefault_plan() -> Any:
    from clipperx.visual_contracts import AudioCue
    p = plan(modes=(True, False, True))
    for shot, gain, limited in zip(p.shots, (-1.125, -6., 2.25), (True, False, True), strict=True):
        shot.audio.source_gain_db, shot.audio.limiter = gain, limited
    p.shots[0].audio.cues.append(AudioCue(asset_id="impact", start=.1, end=.3, gain_db=-3., duck_source_db=-6.))
    return p


def test_legacy_default_graph_parity(host_guard: Any) -> None:
    p = plan()
    assert p.shots[0].audio.normalize_dialogue and p.shots[0].audio.limiter
    args = context(host_guard, p)
    result = host_guard.compile_audio(p, **args)
    assert result.filter_complex == DEFAULT_GRAPH
    assert result.schedule == host_guard.schedule_audio(p) and result.receipt is args["receipt"]
    assert result.cue_inputs == () and result.output_label == "audio_out"
    assert result.normalization_scope == "mixed_track_normalization"
    assert result.requirements.required_filters == (
        "aformat", "alimiter", "anull", "aresample", "asetpts", "asplit", "atrim", "concat", "loudnorm", "volume")


def test_legacy_nondefault_graph_parity(host_guard: Any) -> None:
    p = nondefault_plan()
    args = context(host_guard, p)
    result = host_guard.compile_audio(p, **args, **cue_context(host_guard, p))
    assert result.filter_complex == NONDEFAULT_GRAPH
    assert result.receipt is args["receipt"] and result.schedule.total_samples == 144000
    assert result.cue_inputs == (("impact", Path("C:/PURE/impact.wav"), "1:a"),)
    assert result.requirements.required_filters == (
        "adelay", "aeval", "aformat", "alimiter", "amix", "anull", "apad", "aresample", "asetpts",
        "asplit", "atrim", "concat", "loudnorm", "volume")


def test_prepared_offset_has_no_second_trim_or_loudnorm(host_guard: Any) -> None:
    p = plan(modes=(True, True))
    result = prepared(host_guard, p)
    assert result.schedule.total_samples == 96000
    graph = result.filter_complex
    assert "[0:a]asplit=2[normpart0][normpart1]" in graph
    assert "start_sample=0:end_sample=48000" in graph
    assert "start_sample=48000:end_sample=96000" in graph
    assert all(text not in graph for text in ("atrim=start=", "PTS-", "aresample", "loudnorm", "normbus", "[selected]"))


@pytest.mark.parametrize("modes,expected", [((True,), ("0:a",)),
    ((True, False, True), ("0:a", "1:a")), ((False,), ("0:a",))])
def test_prepared_bus_usage(host_guard: Any, modes: tuple[bool, ...], expected: tuple[str, ...]) -> None:
    import re
    from clipperx.visual_contracts import AudioCue
    p = plan(modes=modes)
    result = prepared(host_guard, p)
    graph = result.filter_complex
    assert tuple(dict.fromkeys(re.findall(r"\[([0-9]+:a)\]", graph))) == expected
    outputs, inputs = [], []
    for fragment in graph.split(";"):
        match = re.match(r"((?:\[[^]]+\])+)(.*?)(\[[^]]+\](?:\[[^]]+\])*)$", fragment)
        assert match is not None
        inputs.extend(re.findall(r"\[([^]]+)\]", match[1]))
        outputs.extend(re.findall(r"\[([^]]+)\]", match[3]))
    assert len(set(outputs)) == len(outputs)
    assert set(inputs) - set(expected) == set(outputs) - {"audio_out"}
    assert all(inputs.count(label) == 1 for label in outputs if label != "audio_out")
    p.shots[0].audio.cues.append(AudioCue(asset_id="impact", start=.1, end=.3))
    cue_index = 2 if len(expected) == 2 else 1
    with_cue = prepared(host_guard, p, **cue_context(host_guard, p), cue_input_start=cue_index)
    assert tuple(dict.fromkeys(re.findall(r"\[([0-9]+:a)\]", with_cue.filter_complex))) == (*expected, f"{cue_index}:a")


def test_prepared_restore_offsets_and_gain_once(host_guard: Any) -> None:
    p = nondefault_plan()
    p.shots[0].audio.cues.clear()
    graph = prepared(host_guard, p).filter_complex
    assert graph.startswith("[0:a]anull[bypass0];[bypass0]atrim=start_sample=48000:end_sample=96000,asetpts=N/48000/TB[raw1];")
    assert "[1:a]asplit=2[normpart0][normpart1]" in graph
    assert "[normpart1]atrim=start_sample=48000:end_sample=96000,asetpts=N/48000/TB[restore2]" in graph
    for fragment in ("[restore0]volume=-1.125dB[gain0]", "[raw1]volume=-6dB[gain1]",
                     "[restore2]volume=2.25dB[gain2]"):
        assert graph.count(fragment) == 1


@pytest.mark.parametrize("case", ("schedule", "source", "hash", "format", "runtime", "count", "mode"))
def test_prepared_binding_mismatch_is_pure(host_guard: Any, monkeypatch: pytest.MonkeyPatch, case: str) -> None:
    audio, p = host_guard, plan()
    args = context(audio, p)
    receipt = args["receipt"]
    changes: dict[str, Any] = {}
    if case == "schedule":
        changes["schedule_identity"] = "0" * 64
    elif case == "source":
        changes["binding"] = replace(args["binding"], source_fingerprint="different")
    elif case == "hash":
        changes["binding"] = replace(args["binding"], selected_pcm_sha256="d" * 64)
    elif case == "format":
        changes["audio_format"] = audio.AudioFormat(48000, "mono")
    elif case == "runtime":
        changes["runtime"] = replace(args["runtime"], runtime_id="other")
    elif case == "count":
        changes["receipt"] = replace(receipt, samples=47999)
    else:
        changes["receipt"] = replace(receipt, mode="dynamic")
    monkeypatch.setattr(audio, "_append_audio_tail", forbidden)
    with pytest.raises(audio.AudioPreparationError, match="mode_mismatch|identity_mismatch"):
        prepared(audio, p, **changes)


@pytest.mark.parametrize("changes", [dict(selected_input_label="1:a"), dict(normalized_input_label=None),
    dict(normalized_input_label="00:a"), dict(normalized_input_label="-1:a"),
    dict(normalized_input_label="32768:a"), dict(normalized_input_label="0:a;evil"),
    dict(normalized_input_label="x" * 12), dict(normalized_input_label=True),
    dict(cue_input_start=True), dict(cue_input_start=-1), dict(cue_input_start=32768)])
def test_prepared_input_presence_and_collision(host_guard: Any, changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        prepared(host_guard, plan(), **changes)
    p = nondefault_plan()
    for extra in (dict(selected_input_label=None), dict(normalized_input_label="0:a"),
                  dict(cue_input_start=1), dict(cue_input_start=32767, normalized_input_label="32767:a")):
        with pytest.raises(ValueError):
            prepared(host_guard, p, **cue_context(host_guard, p), **extra)
    disabled = plan(modes=(False,))
    for extra in (dict(normalized_input_label="1:a"), dict(selected_input_label=None),
                  dict(binding=context(host_guard, plan())["binding"]),
                  dict(receipt=context(host_guard, plan())["receipt"])):
        with pytest.raises(ValueError):
            prepared(host_guard, disabled, **extra)
    for key in ("binding", "runtime", "receipt"):
        with pytest.raises(host_guard.AudioPreparationError, match="measurement_required"):
            prepared(host_guard, plan(), **{key: None})
    from clipperx.visual_contracts import AudioCue
    from clipperx.render_assets import RenderAssetRegistry
    p.shots[0].audio.cues.append(AudioCue(asset_id="click", start=.4, end=.5))
    with pytest.raises(ValueError, match="index limit"):
        prepared(host_guard, p, cue_input_start=32767,
            assets=RenderAssetRegistry({}, {}, {"impact": Path("C:/PURE/impact.wav"), "click": Path("C:/PURE/click.wav")}),
            cue_formats={"impact": host_guard.AudioFormat(48000, "stereo"), "click": host_guard.AudioFormat(48000, "stereo")})


def test_prepared_cue_duck_and_limiter_tail_parity(host_guard: Any) -> None:
    from clipperx.visual_contracts import AudioCue
    audio, p = host_guard, nondefault_plan()
    p.shots[0].audio.cues.append(AudioCue(asset_id="impact", start=.2, end=.4, gain_db=-4., duck_source_db=-12.))
    args = context(audio, p)
    old = audio.compile_audio(p, **args, **cue_context(audio, p), cue_input_start=2)
    new = prepared(audio, p, **cue_context(audio, p), cue_input_start=2)
    assert old.filter_complex.split("[restore0]volume=", 1)[1] == new.filter_complex.split("[restore0]volume=", 1)[1]
    assert old.cue_inputs == new.cue_inputs == (("impact", Path("C:/PURE/impact.wav"), "2:a"),)
    assert "[2:a]asplit=2[cueasset1_0][cueasset1_1]" in new.filter_complex
    assert new.filter_complex.count(audio.LIMITER) == 2
    schedule = audio.schedule_audio(p)
    assert audio._duck_expression(schedule) == DUCK_TWO
    assert audio._duck_expression(schedule, max_bytes=len(DUCK_TWO)) == DUCK_TWO
    assert [audio.duck_db(schedule, t) for t in (0., .15, .25, .4)] == [0., -6., -12., 0.]
    p.shots[0].audio.cues.pop()
    assert audio._duck_expression(audio.schedule_audio(p)) == DUCK_ONE
    # Both sides of a visual/audio policy split still form one maximal limiter run.
    p.shots[1].audio.limiter = True
    assert prepared(audio, p, **cue_context(audio, p), cue_input_start=2).filter_complex.count(audio.LIMITER) == 1


@pytest.mark.parametrize("layout", ("mono", "stereo"))
def test_prepared_requirements_counts_and_layout(host_guard: Any, layout: str) -> None:
    from clipperx.render_runtime import NATIVE_RUNTIME, RenderRequirements, combine_render_requirements
    p = plan()
    args = context(host_guard, p, layout)
    result = prepared(host_guard, p, **args)
    assert result.receipt is args["receipt"] and result.receipt.samples == result.schedule.total_samples == 48000
    assert result.requirements.required_filters == ("alimiter", "anull", "asetpts", "asplit", "atrim", "concat", "volume")
    combined = combine_render_requirements(result.requirements, RenderRequirements(NATIVE_RUNTIME, True, ("libplacebo",)))
    assert combined.needs_vulkan and combined.runtime_id == NATIVE_RUNTIME
    assert combined.required_filters == tuple(sorted((*result.requirements.required_filters, "libplacebo")))
    from clipperx.visual_contracts import AudioCue
    p.shots[0].audio.cues.append(AudioCue(asset_id="impact", start=.1, end=.3))
    cues = cue_context(host_guard, p)
    cues["cue_formats"] = {"impact": host_guard.AudioFormat(48000, layout)}
    with_cue = prepared(host_guard, p, **context(host_guard, p, layout), **cues)
    assert with_cue.cue_inputs == (("impact", Path("C:/PURE/impact.wav"), "1:a"),)
    assert with_cue.schedule.total_samples == 48000 and with_cue.schedule.cues[0].last == 14400
    assert {"adelay", "aeval", "apad", "amix"} <= set(with_cue.requirements.required_filters)
    assert not {"loudnorm", "aresample", "aformat"} & set(with_cue.requirements.required_filters)


def test_prepared_absolute_clock_and_subdivision(host_guard: Any) -> None:
    audio = host_guard
    p = plan(modes=(False, False, False), duration=.10001)
    result = prepared(audio, p)
    assert [(s.first, s.last) for s in result.schedule.spans] == [(0, 4800), (4800, 9601), (9601, 14401)]
    assert "start_sample=4800:end_sample=9601" in result.filter_complex
    # Same contiguous source and policy coalesces despite visual subdivision.
    whole = plan(modes=(False,), duration=.3)
    divided = plan(modes=(False, False, False), duration=.1)
    for i, shot in enumerate(divided.shots):
        shot.source.start, shot.source.end = float(Fraction(100 + i, 10)), float(Fraction(101 + i, 10))
    assert audio.schedule_audio(divided) == audio.schedule_audio(whole)
    assert prepared(audio, divided).filter_complex == prepared(audio, whole).filter_complex
    repeated = plan(modes=(False, False))
    repeated.shots[1].source = repeated.shots[0].source.model_copy()
    assert prepared(audio, repeated).schedule.total_samples == 96000


def test_prepared_guards_precede_traversal_and_generation(host_guard: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    audio = host_guard
    short_paths = {"a": Path("a/b"), "b": Path("c/d"), "c": Path("e/f")}
    # Three exact item entries (table edge + key/value) and two Path components
    # each: 15 total. The third item must fail BEFORE text charging/insertion.
    original_text_size = audio._utf8_size
    def before_third_item(value: str, limit: int) -> int:
        if value == "c":
            forbidden()
        return original_text_size(value, limit)
    with monkeypatch.context() as patch:
        patch.setattr(audio, "_PREPARED_ENTRY_LIMIT", 14, raising=False)
        patch.setattr(audio, "_utf8_size", before_third_item)
        with pytest.raises(ValueError, match="metadata entry limit"):
            audio._prepared_table(short_paths, "path", [0, 0])
    with monkeypatch.context() as patch:
        patch.setattr(audio, "_PREPARED_ENTRY_LIMIT", 15, raising=False)
        budget = [0, 0]
        assert audio._prepared_table(short_paths, "path", budget) == short_paths
        assert budget[1] == 15
    # Consumption carries across records and BOTH tables, never per-child reset.
    from clipperx.render_runtime import RenderRuntime
    runtime = RenderRuntime(Path("a/b"), Path("c/d"), "system", "a", "b", None, None, "graph", None)
    initial = (audio.AudioFormat(48000, "mono"), runtime)
    assert audio._prepared_entry_count(initial, 6) == 6  # two slots + four path components
    with pytest.raises(ValueError, match="metadata entry limit"):
        audio._prepared_entry_count(initial, 5)
    with monkeypatch.context() as patch:
        patch.setattr(audio, "_PREPARED_ENTRY_LIMIT", 19)
        budget = [0, 6]
        assert audio._prepared_table({"format": audio.AudioFormat(48000, "mono")}, "format", budget)
        assert budget[1] == 9
        assert audio._prepared_table({"a": Path("a/b"), "b": Path("c/d")}, "path", budget)
        assert budget[1] == 19
        patch.setattr(audio, "_utf8_size", forbidden)
        with pytest.raises(ValueError, match="metadata entry limit"):
            audio._prepared_table({"c": Path("e/f")}, "path", budget)
    # Invalid structures cannot reach schedule recursion or the tail.
    with monkeypatch.context() as patch:
        patch.setattr(audio, "schedule_audio", forbidden)
        bad: Any
        for bad in (None, {}, plan().model_copy(update={"shots": iter(())}),
                    plan().model_copy(update={"shots": [object()]})):
            with pytest.raises(ValueError):
                audio.compile_prepared_audio(bad, schedule_identity="0" * 64,
                    audio_format=audio.AudioFormat(48000, "mono"))
    p = plan()
    args = context(audio, p)
    for key, record_type in (("audio_format", audio.AudioFormat), ("binding", audio.SourceClockBinding),
                             ("receipt", audio.NormalizationReceipt), ("runtime", type(args["runtime"]))):
        with pytest.raises(ValueError):
            prepared(audio, p, **{key: object.__new__(record_type)})
    for change in (dict(schedule_identity=True), dict(schedule_identity="A" * 64),
        dict(schedule_identity="0" * 65), dict(audio_format=object()),
        dict(receipt=replace(args["receipt"], stats=audio.LoudnessStats(True, 5., -10., -30., 0.))),
        dict(receipt=replace(args["receipt"], stats=audio.LoudnessStats(float("nan"), 5., -10., -30., 0.))),
        dict(receipt=replace(args["receipt"], stats=audio.LoudnessStats(-(1 << 2048), 5., -10., -30., 0.))),
        dict(runtime=replace(args["runtime"], ffmpeg="PURE-relative")),
        dict(runtime=replace(args["runtime"], device_index=True)),
        dict(receipt=replace(args["receipt"], scope="other")),
        dict(receipt=replace(args["receipt"], runtime_identity=("x",)))):
        with pytest.raises(ValueError):
            prepared(audio, p, **change)
    from clipperx.render_assets import RenderAssetRegistry
    for extra in (dict(assets=object()), dict(cue_formats={"../bad": args["audio_format"]}),
                  dict(assets=object.__new__(RenderAssetRegistry)),
                  dict(cue_formats={"impact": object()}), dict(cue_formats=iter(()))):
        with pytest.raises(ValueError):
            prepared(audio, p, **extra)
    with pytest.raises(ValueError):
        prepared(audio, p, receipt=replace(args["receipt"], stats=object.__new__(audio.LoudnessStats)))
    registry = RenderAssetRegistry({}, {}, {})
    object.__setattr__(registry, "sfx", {"impact": "PURE-not-a-Path"})
    with pytest.raises(ValueError):
        prepared(audio, p, assets=registry)
    cue_plan = nondefault_plan()
    for extra in ({}, dict(assets=cue_context(audio, p)["assets"]),
                  {**cue_context(audio, p), "selected_input_label": "4:a", "normalized_input_label": "5:a",
                       "cue_input_start": 32767, "cue_formats": {"impact": audio.AudioFormat(48000, "mono")}}):
        with pytest.raises(ValueError):
            prepared(audio, cue_plan, **extra)
    cycle = plan()
    cycle.shots[0].audio.cues.append(cycle)
    with pytest.raises(ValueError):
        audio.compile_prepared_audio(cycle, schedule_identity="0" * 64, audio_format=args["audio_format"])

    class LyingDict(dict[str, Any]):
        def __len__(self) -> int:
            return 0

    class GeneratorItems(dict[str, Any]):
        def items(self) -> Any:
            def never() -> Any:
                forbidden()
                yield ("impact", args["audio_format"])
            return never()

    with monkeypatch.context() as patch:
        patch.setattr(audio, "_PREPARED_COLLECTION_LIMIT", 2)
        for mapping in (MappingProxyType(LyingDict(a=args["audio_format"], b=args["audio_format"], c=args["audio_format"])),
                        MappingProxyType(GeneratorItems())):
            with pytest.raises(ValueError):
                prepared(audio, p, cue_formats=mapping)
            object.__setattr__(registry, "sfx", mapping)
            with pytest.raises(ValueError):
                prepared(audio, p, assets=registry)
        crowded = plan()
        crowded.shots[0].audio.cues = cue_plan.shots[0].audio.cues * 3
        with pytest.raises(ValueError):
            audio.compile_prepared_audio(crowded, schedule_identity="0" * 64, audio_format=args["audio_format"])
        with pytest.raises(ValueError):
            audio.compile_prepared_audio(plan(modes=(False, False, False)),
                schedule_identity="0" * 64, audio_format=args["audio_format"])
    with monkeypatch.context() as patch:
        patch.setattr(audio, "_PREPARED_TEXT_LIMIT", 64)
        patch.setattr(audio, "schedule_audio", forbidden)
        with pytest.raises(ValueError):
            audio.compile_prepared_audio(p, schedule_identity="0" * 64, audio_format=args["audio_format"])
    with monkeypatch.context() as patch:
        patch.setattr(audio, "_PREPARED_GRAPH_LIMIT", 100)
        with pytest.raises(ValueError):
            prepared(audio, p)
    with monkeypatch.context() as patch:
        patch.setattr(audio, "_seconds", forbidden)
        with pytest.raises(ValueError):
            audio._audio_seconds(Fraction(1 << 4096), True)
    # A view that changes after validation must fail before a snapshot escapes.
    table = {"impact": args["audio_format"]}
    original_size = audio._utf8_size
    def mutate(value: str, limit: int) -> int:
        if value == "impact":
            table["second"] = args["audio_format"]
        return original_size(value, limit)
    with monkeypatch.context() as patch:
        patch.setattr(audio, "_utf8_size", mutate)
        with pytest.raises(ValueError):
            prepared(audio, p, cue_formats=MappingProxyType(table))
    # Exact-fit vs one-over rejects before any fragment join/retention.
    for operation, literal in (
        (lambda g: audio._audio_split(g, "0:a", 2, "x"), "[0:a]asplit=2[x0][x1]"),
        (lambda g: audio._audio_concat(g, ["x0", "x1"], "out"), "[x0][x1]concat=n=2:v=0:a=1[out]"),
        (lambda g: g.append_parts(("one", "two")), "onetwo"),
    ):
        graph = audio._AudioGraph(limit=len(literal))
        operation(graph)
        assert graph.finish() == literal
        graph = audio._AudioGraph(limit=len(literal) - 1)
        with monkeypatch.context() as patch:
            patch.setattr(audio, "_join_audio_parts", forbidden)
            with pytest.raises(ValueError):
                operation(graph)
        assert graph.finish() == ""
    graph = audio._AudioGraph(limit=4)
    graph.append_parts(("a",))
    graph.append_parts(("bb",))
    assert graph.finish() == "a;bb"
    with pytest.raises(ValueError):
        graph.append_parts(("",))
    with pytest.raises(ValueError):
        audio._AudioGraph(limit=10).append_parts((object(),))
    with monkeypatch.context() as patch:
        patch.setattr(audio, "_PREPARED_FRAGMENT_LIMIT", 1)
        graph = audio._AudioGraph(limit=100)
        graph.append_parts(("a",))
        with pytest.raises(ValueError):
            graph.append_parts(("b",))
    schedule = audio.schedule_audio(nondefault_plan())
    assert audio._duck_expression(schedule, max_bytes=len(DUCK_ONE)) == DUCK_ONE
    with monkeypatch.context() as patch:
        patch.setattr(audio, "_join_audio_parts", forbidden)
        with pytest.raises(ValueError):
            audio._duck_expression(schedule, max_bytes=len(DUCK_ONE) - 1)
    # UTF8 bytes and separators are charged, and two-cue expansion uses exact bytes.
    graph = audio._AudioGraph(limit=3)
    graph.append_parts(("é",))
    with pytest.raises(ValueError):
        graph.append_parts(("x",))
    assert graph.finish() == "é"
    from clipperx.visual_contracts import AudioCue
    two = nondefault_plan()
    two.shots[0].audio.cues.append(AudioCue(asset_id="impact", start=.2, end=.4,
                                          gain_db=-4., duck_source_db=-12.))
    with pytest.raises(ValueError):
        audio._duck_expression(audio.schedule_audio(two), max_bytes=len(DUCK_TWO)-1)
    # Legacy remains unlimited and accepts its wider ordinary label grammar.
    legacy = audio.compile_audio(plan(modes=(False,)), audio_format=args["audio_format"], input_label="host_original_audio_40000:a", cue_input_start=40000)
    assert legacy.filter_complex.startswith("[host_original_audio_40000:a]")
