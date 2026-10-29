"""Bounded paired-decode protocol and independently known HOST input controls."""

from fractions import Fraction as F
import importlib.util
from pathlib import Path
import subprocess
from typing import Any
import zlib

import pytest

from clipperx.tests.test_render_clock import make_origin, audio_payload, stream_recipe


def test_float_audio_pcm_byte_finiteness_and_detachment(monkeypatch: pytest.MonkeyPatch) -> None:
    import copy
    from clipperx.tests.test_render_clock import _float_recipe
    from clipperx.render_clock import SourceFloatAudioDecodeReceipt, digest
    d = decoder()
    origin, runtime, limits, pcm, payload = _float_recipe()
    receipt = SourceFloatAudioDecodeReceipt(origin, payload, runtime=runtime, limits=limits.record())
    for buffer in (pcm, bytearray(pcm), memoryview(pcm)):
        decoded = d.DecodedFloatAudioWindow(receipt, buffer)
        assert decoded.pcm == pcm and type(decoded.pcm) is bytes
    mutable = bytearray(pcm)
    detached = d.DecodedFloatAudioWindow(receipt, mutable)
    mutable[0] = 1
    assert detached.pcm == pcm
    for raw in (bytes.fromhex("00000080 01000000 0000c03f 000000b5"),
                bytes.fromhex("0000807f 00000000 00000000 00000000"),
                bytes.fromhex("0000c07f 00000000 00000000 00000000")):
        p = copy.deepcopy(payload)
        p["audio"][0]["pcm_sha256"] = digest(raw)
        for stage in ("input", "output"):
            p["audio"][0][stage][0]["checksum"] = f"{zlib.adler32(raw, 0):08X}"
        p["audio"][0]["output"][0]["sha256"] = digest(raw)
        r = SourceFloatAudioDecodeReceipt(origin, p, runtime=runtime, limits=limits.record())
        if raw[2:4] in (b"\x80\x7f", b"\xc0\x7f"):
            with pytest.raises(ValueError, match="finite"):
                d.DecodedFloatAudioWindow(r, raw)
        else:
            assert d.DecodedFloatAudioWindow(r, raw).pcm == raw
    with pytest.raises(ValueError):
        d.DecodedFloatAudioWindow(receipt, pcm[:-1])
    with pytest.raises(ValueError):
        d.DecodedFloatAudioWindow(receipt, bytes.fromhex("01000000") + pcm[4:])
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("hash reached before byte capacity")
    monkeypatch.setattr(d, "_validate_pcm", forbidden)
    class NoCopy(bytearray):
        def __bytes__(self) -> bytes:
            raise AssertionError("copy reached before byte capacity")
    with pytest.raises(ValueError):
        d.DecodedFloatAudioWindow(receipt, NoCopy(33))


def _float_fake_host(monkeypatch: pytest.MonkeyPatch, fault: str = "") -> Any:
    """In-memory HOST/process/clock boundaries; metadata worker runs inline, never OS threads."""
    import io
    from types import SimpleNamespace
    from clipperx.tests.test_render_clock import _float_recipe
    from clipperx.render_runtime import RenderRuntime
    d = decoder()
    origin, runtime_record, limits, pcm, payload = _float_recipe()
    if fault in ("observed_phase", "observed_phase_two"):
        start = F(1, 2**512 + 1)
        origin = d.SourceTimeOrigin(origin.path, origin.size, origin.mtime_ns, origin.fingerprint,
            -start, origin.duration, origin.streams, origin.decoder_id, {})
    runtime = RenderRuntime(Path(runtime_record["ffmpeg"]), Path(runtime_record["ffprobe"]), "HOST",
        "A" * 64, "B" * 64, None, None, "-filter_complex_script", None)
    children: list[Any] = []
    scheduling: list[Any] = []
    reads: list[int | None] = []
    forbid_reads = [False]
    fingerprints: list[int] = []
    def fingerprint(path: Path) -> str:
        fingerprints.append(1)
        return "0" * 16 if fault == "post_source" and len(fingerprints) > 1 else origin.fingerprint
    class Worker:
        def __init__(self, target: Any = None, **kwargs: Any) -> None:
            self.target = target
        def start(self) -> None:
            scheduling.append("inline metadata")
            self.target()
        def join(self, timeout: Any = None) -> None:
            scheduling.append("metadata joined")
        def is_alive(self) -> bool:
            return False
    class Timer:
        def __init__(self, seconds: Any, callback: Any) -> None:
            self.callback = callback
        def start(self) -> None:
            scheduling.append("watchdog armed")
        def cancel(self) -> None:
            scheduling.append("watchdog cancelled")
        def join(self, timeout: Any = None) -> None:
            scheduling.append("watchdog joined")
    class Raw(io.BytesIO):
        def read(self, count: int | None = -1) -> bytes:
            reads.append(count)
            if forbid_reads[0]:
                raise AssertionError("PCM read before complete observed-phase budget")
            return super().read(min(count, 3) if fault == "partial" and count is not None else count)
    class Child:
        def __init__(self, argv: Any, **kwargs: Any) -> None:
            self.argv, self.killed, self.done = argv, False, False
            children.append(self)
            is_video = "-vf" in argv
            raw = b"\x01\x02\x03" if is_video else pcm if "f32le" in argv else b"\x00\x00" * 4
            if fault == "nonfinite":
                raw = bytes.fromhex("0000807f") + raw[4:]
            checksum = zlib.adler32(raw, 0) ^ (1 if fault == "checksum" else 0)
            out_format = "flt" if "f32le" in argv else "s16"
            input_format = "s16" if fault == "input_format" else "flt"
            if fault == "output_format":
                out_format = "s16"
            rate, channels, samples, pts = 48000, 1, 4, 0
            n: int | str = 0
            if fault == "rate":
                rate = 44100
            if fault == "channels":
                channels = 2
            if fault == "sequence":
                n = 1
            if fault == "block_samples":
                samples = 9
            if fault == "huge_token":
                n = "1" * 1236
            def line(stage: str, fmt: str) -> str:
                observed_pts = 1 if fault in ("observed_phase", "observed_phase_two") and stage == "cxaout" else pts
                observed_samples = 8 if fault == "observed_phase_two" and stage == "cxain" else samples
                return (f"[ashowinfo@{stage} @ 000] n: {n} pts: {observed_pts} "
                    f"fmt:{fmt} channels:{channels} rate:{rate} nb_samples:{observed_samples} checksum:{checksum:08X}\n")
            text = ("[showinfo@cxv @ 000] config in time_base: 1/60\n" +
                    video_line(0, 0, raw, size="1x1") + "\n" if is_video else
                    line("cxain", input_format) + line("cxaout", out_format))
            if fault == "missing_input":
                text = line("cxaout", out_format)
            if fault == "log":
                text = "x" * 4097 + "\n"
            if fault == "short":
                raw = raw[:-1]
            if fault == "trailing":
                raw += b"x"
            if fault == "gap":
                text += line("cxaout", out_format).replace("n: 0", "n: 1").replace("pts: 0", "pts: 5")
                raw += raw
            if fault == "observed_phase_two":
                text += line("cxaout", out_format).replace("n: 0", "n: 1").replace("pts: 1", "pts: 5")
                raw += raw
            self.stdout = Raw(raw)
            self.stderr = io.BytesIO(text.encode())
        def poll(self) -> Any:
            return 0 if self.done else None
        def kill(self) -> None:
            self.killed = True
            self.done = True
        def wait(self, timeout: Any = None) -> int:
            self.done = True
            if fault == "post_origin":
                object.__setattr__(origin, "duration", F(0))
            return 7 if fault == "child" else 0
    monkeypatch.setattr(d.threading, "Thread", Worker)
    monkeypatch.setattr(d.threading, "Timer", Timer)
    monkeypatch.setattr(d.subprocess, "Popen", Child)
    monkeypatch.setattr(d, "_runtime_identity", lambda value: origin.decoder_id)
    monkeypatch.setattr(d, "_verify_pins", lambda value: None)
    monkeypatch.setattr(d, "source_fingerprint", fingerprint)
    return SimpleNamespace(d=d, origin=origin, runtime=runtime, limits=limits, pcm=pcm,
        payload=payload, children=children, reads=reads, scheduling=scheduling, fingerprints=fingerprints,
        forbid_reads=forbid_reads)


def test_float_audio_fake_pipeline_one_audio_and_legacy_parity(monkeypatch: pytest.MonkeyPatch) -> None:
    h = _float_fake_host(monkeypatch)
    d = h.d
    result = d.observe_float_audio_window(h.runtime, h.origin, d.Window(F(0), F(1, 12000)),
        audio_policy="render48k-f32-mono-v1", limits=h.limits)
    assert result.pcm == h.pcm and len(h.children) == 1
    assert "-vn" in h.children[0].argv and "-vf" not in h.children[0].argv
    assert len(h.fingerprints) == 2 and all(n <= 16 for n in h.reads)
    assert h.children[0].stdout.closed and h.children[0].stderr.closed
    assert "watchdog cancelled" in h.scheduling and "metadata joined" in h.scheduling
    old = d.observe_audio_window(h.runtime, h.origin, d.Window(F(0), F(1, 12000)),
        audio_policy="render48k-mono", limits=d.AudioDecodeLimits(preroll=F(0), postroll=F(0),
            max_window=F(1), max_blocks=2, max_samples=8, max_decoded_bytes=32, max_retained_bytes=32))
    assert old.pcm == b"\0\0" * 4 and "s16le" in h.children[1].argv
    argv = d.video_decode_argv(str(h.runtime.ffmpeg), h.origin, F(0), F(1, 12000))
    with d.payload_stream(argv, "video", F(1, 60), d.DecodeLimits()) as stream:
        pairs = list(stream)
    assert len(pairs) == 1 and pairs[0][1] == b"\x01\x02\x03"
    with pytest.raises(ValueError, match="stream|owner"):
        with d._audio_window(h.runtime, h.origin, d.Window(F(0), F(1, 12000)),
            (d.Window(F(0), F(1, 12000)),), audio_policy="render48k-f32-mono-v1",
            limits=h.limits, _owner=object(), _audio_contract=d._FLOAT_AUDIO):
            pass
    assert len(h.children) == 3


@pytest.mark.parametrize("fault", ["short", "trailing", "input_format", "output_format", "sequence",
    "gap", "rate", "channels", "checksum", "nonfinite", "child", "missing_input", "post_source", "post_origin"])
def test_float_audio_protocol_failures_keep_diagnostics_and_cleanup(monkeypatch: pytest.MonkeyPatch,
                                                                    fault: str) -> None:
    h = _float_fake_host(monkeypatch, fault)
    with pytest.raises(h.d.DecodeFailure) as caught:
        h.d.observe_float_audio_window(h.runtime, h.origin, h.d.Window(F(0), F(1, 12000)),
            audio_policy="render48k-f32-mono-v1", limits=h.limits)
    diagnostic = caught.value.record()
    assert diagnostic["argv"] == h.children[0].argv
    assert len(diagnostic["stdout_sha256"]) == 64 and "stderr_tail" in diagnostic
    assert h.children[0].done and h.children[0].stdout.closed and h.children[0].stderr.closed
    assert "watchdog cancelled" in h.scheduling and "metadata joined" in h.scheduling


def test_float_audio_invalid_inputs_and_caps_precede_child_or_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    from dataclasses import replace
    from clipperx.render_clock import StreamClock
    h = _float_fake_host(monkeypatch)
    d = h.d
    window = d.Window(F(0), F(1, 12000))
    policies: tuple[Any, ...] = ("render48k-mono", [], "render48k-f32-stereo-v1")
    for policy in policies:
        with pytest.raises(ValueError):
            d.observe_float_audio_window(h.runtime, h.origin, window, audio_policy=policy, limits=h.limits)
    for changes in (dict(sample_format="s16"), dict(sample_format="dbl"), dict(sample_rate=32000)):
        audio = h.origin.streams[1]
        bad = StreamClock(audio.index, "audio", audio.time_base, audio.start, audio.end,
                          sample_rate=changes.get("sample_rate", audio.sample_rate), channels=audio.channels,
                          sample_format=changes.get("sample_format", audio.sample_format))
        origin = d.SourceTimeOrigin(h.origin.path, h.origin.size, h.origin.mtime_ns, h.origin.fingerprint,
            h.origin.origin, h.origin.duration, (h.origin.streams[0], bad), h.origin.decoder_id, {})
        with pytest.raises(ValueError):
            d.observe_float_audio_window(h.runtime, origin, window, audio_policy="render48k-f32-mono-v1", limits=h.limits)
    malformed = d.Window(F(0), F(1, 12000))
    object.__setattr__(malformed, "start", 0)
    for bad_window, bad_limits in ((malformed, h.limits), (window, d.AudioDecodeLimits()),
            (d.Window(F(0), F(1, 10000000)), h.limits),
            (window, replace(h.limits, max_metadata_bytes=32))):
        with pytest.raises(ValueError):
            d.observe_float_audio_window(h.runtime, h.origin, bad_window, audio_policy="render48k-f32-mono-v1", limits=bad_limits)
    assert not h.children
    # Payload and record ceilings are independent and fail before a stdout read.
    for field, value, fault in (("max_decoded_bytes", 15, ""), ("max_retained_bytes", 15, ""),
            ("max_samples", 3, ""), ("max_block_samples", 3, ""),
            ("max_blocks", 1, "gap"), ("max_metadata_bytes", 1800, ""),
            ("max_log_bytes", 20, ""), ("max_block_samples", 8, "huge_token")):
        with monkeypatch.context() as patch:
            bounded = _float_fake_host(patch, fault)
            with pytest.raises(d.DecodeFailure):
                d.observe_float_audio_window(bounded.runtime, bounded.origin, window,
                    audio_policy="render48k-f32-mono-v1", limits=replace(bounded.limits, **{field: value}))
            assert bounded.reads == ([16] if field == "max_blocks" else []), field
    partial = _float_fake_host(monkeypatch, "partial")
    assert d.observe_float_audio_window(partial.runtime, partial.origin, window,
        audio_policy="render48k-f32-mono-v1", limits=partial.limits).pcm == partial.pcm
    # Valid nonzero rational origin and one-sample observed shift expand the
    # complete header. Real receipt and real producer must agree before PCM work.
    from clipperx.render_clock import SourceFloatAudioDecodeReceipt, rational, float_audio_decode_argv, digest
    for cap, fits, blocks in ((3700, False, 1), (4096, True, 1), (4096, True, 2)):
        with monkeypatch.context() as patch:
            phase = _float_fake_host(patch, "observed_phase" if blocks == 1 else "observed_phase_two")
            phase.limits = replace(phase.limits, max_metadata_bytes=cap)
            start = -phase.origin.origin
            requested = d.Window(start, start + F(blocks, 12000))
            payload = phase.payload
            payload["windows"] = [[rational(requested.start), rational(requested.end)]]
            payload["limits"] = dict(phase.limits.record(), decoder_start=rational(requested.start),
                                     decoder_end=rational(requested.end))
            payload["argv"] = [float_audio_decode_argv(str(phase.runtime.ffmpeg), phase.origin,
                requested.start, requested.end, "render48k-f32-mono-v1")]
            audio = payload["audio"][0]
            audio["output"][0]["pts"] = 1
            audio["sample_grid_origin"], audio["source_phase"] = "1/48000", rational(F(1, 48000) + start)
            audio["coverage"] = [dict(window=payload["windows"][0], status="unavailable")]
            if blocks == 2:
                audio["input"][0]["samples"] = 8
                audio["output"].append(dict(audio["output"][0], n=1, pts=5))
                audio["samples"], audio["bytes"], audio["pcm_sha256"] = 8, 32, digest(phase.pcm * 2)
            runtime = d._runtime_record(phase.runtime)
            if fits:
                real = SourceFloatAudioDecodeReceipt(phase.origin, payload, runtime=runtime,
                                                     limits=phase.limits.record())
                result = d.observe_float_audio_window(phase.runtime, phase.origin, requested,
                    audio_policy="render48k-f32-mono-v1", limits=phase.limits)
                assert result.pcm == phase.pcm * blocks and result.receipt.to_json() == real.to_json()
                assert result.receipt.record()["payload"]["audio"][0]["coverage"][0]["status"] == "unavailable"
            else:
                with pytest.raises(ValueError):
                    SourceFloatAudioDecodeReceipt(phase.origin, payload, runtime=runtime,
                                                  limits=phase.limits.record())
                phase.forbid_reads[0] = True
                with pytest.raises(d.DecodeFailure):
                    d.observe_float_audio_window(phase.runtime, phase.origin, requested,
                        audio_policy="render48k-f32-mono-v1", limits=phase.limits)
                assert phase.reads == [], "observed header exceeded cap before PCM read/retention/hash"
                assert phase.children[0].stdout.closed and phase.children[0].stderr.closed


def decoder() -> Any:
    assert importlib.util.find_spec("clipperx.source_decode") is not None, "decode API absent"
    from clipperx import source_decode
    return source_decode


def video_line(n: int, pts: int, payload: bytes, *, size: str = "2x2") -> str:
    return (f"[showinfo@cxv @ 000] n: {n} pts: {pts} pts_time: 999 fmt:bgr24 "
            f"s:{size} iskey:1 type:I checksum:{zlib.adler32(payload, 0):08X} "
            "duration: 0 duration_time:0")


def fake_stream_host(monkeypatch: pytest.MonkeyPatch, fault: str = "none") -> Any:
    """Fake only HOST/child/clock/I/O boundaries; real tags, pixels and validators."""
    import io
    from types import SimpleNamespace

    d = decoder()
    origin, limits, _, _ = stream_recipe()
    clock = [10**9]
    timers: list[Any] = []
    children: list[Any] = []
    events: list[Any] = []
    read_hooks: list[Any] = []

    class Timer:
        def __init__(self, seconds: float, callback: Any) -> None:
            self.callback, self.seconds, self.cancelled = callback, seconds, False
            self.daemon = False
            timers.append(self)

        def start(self) -> None:
            pass

        def cancel(self) -> None:
            self.cancelled = True

        def join(self, timeout: float = 0) -> None:
            pass

        def is_alive(self) -> bool:
            return False

    class Raw(io.BytesIO):
        def close(self) -> None:
            if fault == "handle_close":
                raise OSError("HOST fake handle close failed")
            super().close()
        def readinto(self, buffer: Any) -> Any:
            events.append(("readinto", len(buffer)))
            for hook in read_hooks:
                hook()
            if fault == "none_read":
                return None
            return super().readinto(buffer[: min(len(buffer), 3)])

    class Child:
        def __init__(self, argv: Any, **kwargs: Any) -> None:
            self.argv, self.pid, self.killed, self.done = argv, 900 + len(children), False, False
            is_video = "-vf" in argv
            bodies = [bytes([n]) * 12 for n in range(3)]
            pts = [5015, 5050, 5090]
            if fault == "outside_predecessor":
                pts[0] = 4990
            elif fault == "outside_successor":
                pts[1] = 5200
            if fault == "predecessor":
                bodies[0] = bytes([8]) * 12
            if fault == "successor":
                bodies[1] = bytes([8]) * 12
            lines = "[showinfo@cxv @ 000] config in time_base: 1/1000, frame_rate: 0/0\n"
            lines += "\n".join(video_line(n, t, bytes([n]) * 12) for n, t in enumerate(pts)) + "\n"
            if fault == "tag_count":
                lines = lines.replace("n: 1 pts: 5050", "n: 2 pts: 5050")
            elif fault == "tag_order":
                lines = lines.replace("pts: 5050", "pts: 5014")
            elif fault == "tag_dimensions":
                lines = lines.replace("n: 1 pts: 5050 pts_time: 999 fmt:bgr24 s:2x2",
                                      "n: 1 pts: 5050 pts_time: 999 fmt:bgr24 s:1x4")
            elif fault == "tag_tb":
                lines = lines.replace("[showinfo@cxv @ 000] n: 1", "[showinfo@cxv @ 000] config in time_base: 1/90000, frame_rate: 0/0\n[showinfo@cxv @ 000] n: 1")
            raw = b"".join(bodies)
            if fault == "short" and is_video:
                raw = raw[:-1]
            if fault == "extra" and is_video:
                raw += b"x"
            if fault == "untagged" and is_video:
                lines = "HOST diagnostic\n"
            if fault == "full_queue" and is_video:
                lines += "\n".join(video_line(n, 5090 + n, bytes(12)) for n in range(3, 40)) + "\n"
            if not is_video:
                ins, outs = (3969, 4320) if fault == "unavailable_audio" else (4410, 4800)
                ipts, opts = (220941, 240480) if fault == "unavailable_audio" else (220500, 240000)
                lines = (
                    f"[ashowinfo@cxain @ 000] n:0 pts:{ipts} fmt:s16 channels:1 rate:44100 "
                    f"nb_samples:{ins} checksum:00000000\n[ashowinfo@cxaout @ 000] n:0 pts:{opts} "
                    f"fmt:s16 channels:1 rate:48000 nb_samples:{outs} checksum:00000000\n"
                )
                raw = bytes(outs * 2)
                if fault == "pcm_blocks":
                    lines = ("[ashowinfo@cxain @ 000] n:0 pts:220500 fmt:s16 channels:1 rate:44100 "
                        "nb_samples:4410 checksum:00000000\n"
                        "[ashowinfo@cxaout @ 000] n:0 pts:240000 fmt:s16 channels:1 rate:48000 "
                        "nb_samples:2400 checksum:00000000\n"
                        "[ashowinfo@cxaout @ 000] n:1 pts:242400 fmt:s16 channels:1 rate:48000 "
                        "nb_samples:2400 checksum:00000000\n")
                if fault == "pcm_corrupt":
                    raw = b"x" + raw[1:]
                if fault == "audio_conversion":
                    lines = lines.replace("nb_samples:4410", "nb_samples:4400")
            self.stdout, self.stderr = Raw(raw), io.BytesIO(lines.encode())
            children.append(self)
            events.append(("child", "video" if is_video else "audio"))

        def poll(self) -> int | None:
            return 0 if self.done else None

        def kill(self) -> None:
            if fault == "kill":
                raise OSError("HOST fake kill failed")
            self.killed = True
            events.append(("kill", self.pid))

        def wait(self, timeout: float = 0) -> Any:
            self.waits = getattr(self, "waits", 0) + 1
            if fault == "reap":
                raise subprocess.TimeoutExpired(self.argv, timeout)
            if fault == "wait_timeout" and self.waits == 1:
                raise subprocess.TimeoutExpired(self.argv, timeout)
            self.done = True
            if fault == "exit_bool":
                return False
            if fault == "exit_float":
                return 0.0
            return 1 if fault == "late_exit" else 0

    monkeypatch.setattr(d, "_runtime_identity", lambda runtime: origin.decoder_id)
    monkeypatch.setattr(d, "_verify_pins", lambda runtime: None)
    fingerprints = [origin.fingerprint]
    monkeypatch.setattr(d, "source_fingerprint", lambda path: fingerprints[0])
    monkeypatch.setattr(d.subprocess, "Popen", Child)
    monkeypatch.setattr(d.threading, "Timer", Timer)
    monkeypatch.setattr(d.time, "monotonic_ns", lambda: clock[0])
    return SimpleNamespace(
        d=d,
        origin=origin,
        limits=limits,
        runtime=SimpleNamespace(ffmpeg=Path("HOST-ffmpeg")),
        window=d.Window(F(0), F(1, 10)),
        requests=(F(1, 50), F(1, 25), F(3, 50)),
        clock=clock,
        timers=timers,
        children=children,
        events=events,
        fingerprints=fingerprints,
        read_hooks=read_hooks,
    )


@pytest.mark.parametrize(
    "fault",
    [
        "none",
        "predecessor",
        "successor",
        "unsorted",
        "before",
        "terminal",
        "missing",
        "tag_count",
        "tag_order",
        "tag_tb",
        "tag_dimensions",
        "frame_limit",
        "decoded_limit",
        "outside_predecessor", "outside_successor",
        "provisional_bool_n", "provisional_bool_pts0", "provisional_bool_pts1",
        "provisional_bool_width", "provisional_valid_rows", "provisional_selected_shape",
    ],
)
def test_stream_shared_selector_materialized_parity(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    d = decoder()
    assert hasattr(d, "stream_observed_window"), "stream observer API absent"
    if fault.startswith("provisional_"):
        pts = 1 if fault == "provisional_bool_pts1" else 0
        frame_pixels = bytes(3)
        tag = d.VideoTag(0, pts, F(1), 1, 1, zlib.adler32(frame_pixels, 0))
        next_pixels = bytes([1]) * 3
        successor_tag = d.VideoTag(1, pts + 1, F(1), 1, 1, zlib.adler32(next_pixels, 0))
        frame = d.ObservedFrame(tag, F(pts + 1), frame_pixels)
        selected = {
            **tag.record(),
            "bytes": len(frame_pixels),
            "sha256": d.digest(frame_pixels),
            "coverage_end": d.rational(frame.raw_end),
        }
        successor = {
            **successor_tag.record(),
            "bytes": len(next_pixels),
            "sha256": d.digest(next_pixels),
            "coverage_end": None,
        }
        if fault == "provisional_bool_n":
            selected["n"] = False
        elif fault == "provisional_bool_pts0":
            selected["pts"] = False
        elif fault == "provisional_bool_pts1":
            selected["pts"] = True
        elif fault == "provisional_bool_width":
            selected["width"] = True
        if fault != "provisional_valid_rows":
            selected_argument = None if fault == "provisional_selected_shape" else selected
            with pytest.raises(ValueError):
                d.ProvisionalSelection(
                    "00000000-0000-4000-8000-000000000001",
                    0,
                    (0,),
                    frame,
                    selected_argument,
                    successor,
                    previous_chain_sha256="0" * 64,
                )
        else:
            selection = d.ProvisionalSelection(
                "00000000-0000-4000-8000-000000000001",
                0,
                (0,),
                frame,
                selected,
                successor,
                previous_chain_sha256="0" * 64,
            )
            assert (
                selection.selected_record() == selected
                and selection.successor_record() == successor
            )
            selected.clear()
            successor.clear()
            detached = selection.selected_record()
            detached["pts"] = 9
            assert selection.selected_record()["pts"] == 0
            assert selection.successor_record()["pts"] == 1 and selection.frame.bgr is frame_pixels
        return
    h = fake_stream_host(monkeypatch, fault)
    if fault in ("outside_predecessor", "outside_successor"):
        with pytest.raises(ValueError):
            with d.stream_observed_window(
                h.runtime,
                h.origin,
                h.window,
                h.requests,
                audio_policy="render48k-mono",
                limits=h.limits,
            ) as stream:
                stream.next_selection()
        assert len(h.children) == 1 and all(c.done for c in h.children)
        return
    if fault in ("none", "unsorted", "before", "terminal", "missing"):
        requests = h.requests
        if fault == "unsorted":
            requests = tuple(reversed(requests))
        elif fault == "before":
            requests = (F(0),)
        elif fault == "terminal":
            requests = (F(95, 1000),)
        elif fault == "missing":
            requests = (F(1, 1000),)
        bodies = [bytes([n]) * 12 for n in range(3)]
        tags = [
            d.VideoTag(n, t, F(1, 1000), 2, 2, zlib.adler32(bodies[n], 0))
            for n, t in enumerate((5015, 5050, 5090))
        ]
        # Unknown selected-stream start distinguishes missing predecessor from before-start.
        if fault == "missing":
            from dataclasses import replace

            h.origin = d.SourceTimeOrigin(
                h.origin.path,
                h.origin.size,
                h.origin.mtime_ns,
                h.origin.fingerprint,
                h.origin.origin,
                h.origin.duration,
                (replace(h.origin.streams[0], start=None), h.origin.streams[1]),
                h.origin.decoder_id,
                dict(h.origin.color),
            )
        old = d.select_frames(
            zip(tags, bodies), h.origin, requests, d.DecodeLimits(preroll=F(0), postroll=F(0))
        )
        if fault == "unsorted":
            assert [o.frame_n for o in old.outcomes] == [1, 0, 0]
        elif fault in ("before", "missing", "terminal"):
            assert (
                old.outcomes[0].status
                == {
                    "before": "before_video_start",
                    "missing": "missing_predecessor",
                    "terminal": "unknown_terminal",
                }[fault]
            )
        if fault != "none":
            with pytest.raises(ValueError):
                with d.stream_observed_window(
                    h.runtime,
                    h.origin,
                    h.window,
                    requests,
                    audio_policy="render48k-mono",
                    limits=h.limits,
                ) as stream:
                    stream.next_selection()
            return
        with d.stream_observed_window(
            h.runtime, h.origin, h.window, requests, audio_policy="render48k-mono", limits=h.limits
        ) as stream:
            records, pixels = [], []
            while (selection := stream.next_selection()) is not None:
                records.append(selection.selected_record())
                pixels.append(selection.frame.bgr)
                index = selection.delivery_index
                del selection
                stream.acknowledge(index)
            final = stream.finish()
        assert pixels == [f.bgr for f in old.frames]
        assert final.record()["payload"]["video"] == [
            __import__("json").loads(r) for r in old.observations
        ]
        assert [r["n"] for r in records] == [0, 1]
    elif fault in ("predecessor", "successor"):
        with pytest.raises(ValueError):
            with d.stream_observed_window(
                h.runtime,
                h.origin,
                h.window,
                h.requests,
                audio_policy="render48k-mono",
                limits=h.limits,
            ) as stream:
                stream.next_selection()
        assert len(h.children) == 1
    else:
        from dataclasses import replace

        # Exercise the ONE selector directly with actual corrupted tags and tiny bytes.
        bodies = [bytes(12), bytes(12)]
        tags = [d.VideoTag(0, 5015, F(1, 1000), 2, 2, 0), d.VideoTag(1, 5050, F(1, 1000), 2, 2, 0)]
        kwargs: dict[str, Any] = {}
        if fault == "tag_count":
            tags[1] = replace(tags[1], n=2)
        elif fault == "tag_order":
            tags[1] = replace(tags[1], pts=5014)
        elif fault == "tag_tb":
            tags[1] = replace(tags[1], time_base=F(1, 90000))
        elif fault == "tag_dimensions":
            tags[1] = replace(tags[1], width=1, height=4)
        elif fault == "frame_limit":
            kwargs["max_frames"] = 1
        elif fault == "decoded_limit":
            kwargs["max_decoded_bytes"] = 12
        with pytest.raises(ValueError):
            d.select_frames(zip(tags, bodies), h.origin, h.requests, d.DecodeLimits(**kwargs))
        with pytest.raises(ValueError):
            with d.stream_observed_window(
                h.runtime,
                h.origin,
                h.window,
                h.requests,
                audio_policy="render48k-mono",
                limits=replace(h.limits, **kwargs),
            ) as stream:
                stream.next_selection()


@pytest.mark.parametrize("fault", ["none", "live_cap", "selected_cap", "emission_cap", "none_read"])
def test_stream_release_precedes_next_buffer_allocation(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    from dataclasses import replace

    d = decoder()
    assert hasattr(d, "ObservedWindowStream"), "stream pull API absent"
    h = fake_stream_host(monkeypatch, fault)
    if fault != "none":
        kwargs = {
            "live_cap": {"max_live_payload_bytes": 38399},
            "selected_cap": {"max_selected_unique_bytes": 12},
            "emission_cap": {"max_request_emission_bytes": 24},
            "none_read": {},
        }[fault]
        with pytest.raises(ValueError):
            with d.stream_observed_window(
                h.runtime,
                h.origin,
                h.window,
                h.requests,
                audio_policy="render48k-mono",
                limits=replace(h.limits, **kwargs),
            ) as stream:
                while (selection := stream.next_selection()) is not None:
                    index = selection.delivery_index
                    del selection
                    stream.acknowledge(index)
        if fault == "live_cap":
            assert h.children == []
        return
    with d.stream_observed_window(
        h.runtime, h.origin, h.window, h.requests, audio_policy="render48k-mono", limits=h.limits
    ) as stream:
        selection = stream.next_selection()
        assert selection is not None and selection.frame.bgr == bytes(12)
        selected_pixels = selection.frame.bgr
        assert stream._owner.live_payload_bytes == 24
        with pytest.raises(ValueError):
            stream.next_selection()
        index = selection.delivery_index
        del selection
        stream.acknowledge(index)
        assert stream._owner.lease is None and stream._owner.live_payload_bytes == 12
        import sys

        assert sys.getrefcount(selected_pixels) == 2, (
            "observer still owns an acknowledged selected alias"
        )

        def before_read() -> None:
            assert sys.getrefcount(selected_pixels) == 2

        h.read_hooks.append(before_read)
        selection = stream.next_selection()
        h.read_hooks.clear()
        assert selection is not None and selection.frame.bgr == bytes([1]) * 12
        index = selection.delivery_index
        del selection
        stream.acknowledge(index)
        assert stream.next_selection() is None
        assert stream._owner.live_payload_bytes == 0
        final = stream.finish()
        account = final.record()["payload"]["accounting"]
        assert account["reserved_payload_capacity_bytes"] == 38400
        assert (
            account["selected_unique_bytes"] == 24 and account["acknowledged_request_bytes"] == 36
        )
        assert stream._owner.live_payload_bytes == 0
        assert all(e[1] <= 64 for e in h.events if e[0] == "readinto")


@pytest.mark.parametrize(
    "fault", ["child", "window", "boundary", "late_finish", "between_children", "cross_thread"]
)
def test_stream_deadline_owner_with_fake_clock_and_children(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    d = decoder()
    assert hasattr(d, "stream_observed_window"), "stream observer API absent"
    h = fake_stream_host(monkeypatch)
    if fault == "cross_thread":
        import threading
        complete = threading.Event()
        workers: list[Any] = []
        with pytest.raises(ValueError):
            with d.stream_observed_window(h.runtime, h.origin, h.window, h.requests,
                    audio_policy="render48k-mono", limits=h.limits) as stream:
                def suspended_read() -> None:
                    h.read_hooks.clear()
                    h.clock[0] += h.limits.timeout * 10**9
                    def watchdog() -> None:
                        try:
                            stream._owner._expire()
                        finally:
                            complete.set()
                    worker = threading.Thread(target=watchdog, name="fake-stream-watchdog")
                    workers.append(worker)
                    worker.start()
                    assert complete.wait(2), "watchdog blocked on a lock held by readinto"
                    worker.join(2)
                    assert not worker.is_alive() and stream.state == "FAILED"
                    assert h.children[0].killed
                h.read_hooks.append(suspended_read)
                stream.next_selection()
        for worker in workers:
            worker.join(2)
            assert not worker.is_alive()
        assert len(h.children) == 1 and all(c.done for c in h.children)
        return
    if fault == "between_children":
        with pytest.raises(ValueError):
            with d.stream_observed_window(
                h.runtime,
                h.origin,
                h.window,
                h.requests,
                audio_policy="render48k-mono",
                limits=h.limits,
            ) as stream:
                while (selection := stream.next_selection()) is not None:
                    index = selection.delivery_index
                    del selection
                    stream.acknowledge(index)
                h.clock[0] = stream.deadline_ns
                for timer in tuple(h.timers):
                    if not timer.cancelled:
                        timer.callback()
                assert stream.state == "FAILED", "window watchdog must survive video cleanup"
                stream.finish()
        assert len(h.children) == 1
        return
    with pytest.raises(ValueError):
        with d.stream_observed_window(
            h.runtime,
            h.origin,
            h.window,
            h.requests,
            audio_policy="render48k-mono",
            limits=h.limits,
        ) as stream:
            selection = stream.next_selection()
            assert selection is not None
            deadline = stream.deadline_ns
            # Fake watchdog fires independently while caller holds the borrowed selection.
            if fault == "window":
                h.clock[0] = deadline
            else:
                h.clock[0] += h.limits.timeout * 10**9
            for timer in tuple(h.timers):
                if not timer.cancelled:
                    timer.callback()
            assert stream.state == "FAILED" and h.children[0].killed
            assert stream.deadline_ns == deadline
            if fault == "late_finish":
                stream.finish()
            else:
                stream.acknowledge(selection.delivery_index)
    assert all(c.done for c in h.children) and len(h.children) == 1


@pytest.mark.parametrize(
    "fault",
    [
        "wrong_ack",
        "duplicate_ack",
        "early_finish",
        "close",
        "break",
        "short",
        "extra",
        "untagged",
        "late_exit",
        "full_queue",
        "reap",
        "terminal",
        "join",
        "source_change",
        "pin_change",
        "concurrent_close",
        "join_error", "handle_close", "kill",
        "exit_bool", "exit_float", "wait_timeout",
    ],
)
def test_stream_ack_failure_and_cleanup_ownership(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    d = decoder()
    assert hasattr(d, "stream_observed_window"), "stream observer API absent"
    h = fake_stream_host(monkeypatch, fault)
    stream = None

    def run() -> None:
        nonlocal stream
        with d.stream_observed_window(
            h.runtime,
            h.origin,
            h.window,
            h.requests,
            audio_policy="render48k-mono",
            limits=h.limits,
        ) as stream:
            if fault == "concurrent_close":

                def close_during_read() -> None:
                    h.read_hooks.clear()
                    with pytest.raises(ValueError, match="consumer"):
                        stream.close()

                h.read_hooks.append(close_during_read)
            selection = stream.next_selection()
            assert selection is not None
            if fault == "kill":
                stream.close()
            if fault == "wrong_ack":
                stream.acknowledge(1)
            elif fault == "early_finish":
                stream.finish()
            elif fault in ("close", "break"):
                stream.close()
                assert stream.state == "CLOSED"
                return
            index = selection.delivery_index
            del selection
            stream.acknowledge(index)
            if fault == "duplicate_ack":
                stream.acknowledge(index)
            while (selection := stream.next_selection()) is not None:
                index = selection.delivery_index
                del selection
                stream.acknowledge(index)
            if fault == "source_change":
                h.fingerprints[0] = "f" * 16
            elif fault == "pin_change":

                def changed(runtime: Any) -> None:
                    raise ValueError("changed pins")

                monkeypatch.setattr(d, "_verify_pins", changed)
            final = stream.finish()
            assert final is not None
            if fault == "terminal":
                stream.next_selection()

    if fault in ("join", "join_error"):
        from threading import Thread

        class Unjoined(Thread):
            def join(self, timeout: float | None = None) -> None:
                if fault == "join_error":
                    raise RuntimeError("HOST fake join failed")
                super().join(timeout)

            def is_alive(self) -> bool:
                return True

        monkeypatch.setattr(d.threading, "Thread", Unjoined)
    if fault in ("close", "break"):
        run()
    else:
        with pytest.raises(ValueError):
            run()
    assert stream is not None
    if fault in ("reap", "join", "join_error", "handle_close", "kill"):
        assert stream.cleanup_required
        assert stream._owner.survivors
        first_deadline = stream._owner.cleanup_deadline_ns
        with pytest.raises(ValueError):
            stream.close()
        assert stream._owner.cleanup_deadline_ns == first_deadline
    else:
        stream.close()
        assert not stream.cleanup_required and all(c.done for c in h.children)


@pytest.mark.parametrize("fault", ["none", "unavailable_audio", "pcm_corrupt", "audio_conversion", "pcm_blocks"])
def test_stream_pcm_bytes_and_resource_reconciliation(
    monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    d = decoder()
    assert hasattr(d, "stream_observed_window"), "stream observer API absent"
    h = fake_stream_host(monkeypatch, fault)

    def run() -> Any:
        with d.stream_observed_window(
            h.runtime,
            h.origin,
            h.window,
            h.requests,
            audio_policy="render48k-mono",
            limits=h.limits,
        ) as stream:
            while (selection := stream.next_selection()) is not None:
                index = selection.delivery_index
                del selection
                stream.acknowledge(index)
            final = stream.finish()
            assert stream.state == "FINALIZED" and not stream.cleanup_required
            assert stream._owner.live_payload_bytes == 0
            return final

    if fault in ("pcm_corrupt", "audio_conversion"):
        with pytest.raises(ValueError):
            run()
    else:
        final = run()
        payload = final.record()["payload"]
        pcm = bytes(8640 if fault == "unavailable_audio" else 9600)
        assert payload["audio"]["pcm_sha256"] == d.digest(pcm)
        offset = 0
        for block in payload["audio"]["output"]:
            assert block["sha256"] == d.digest(pcm[offset:offset + block["bytes"]])
            assert block["checksum"] == "00000000"
            offset += block["bytes"]
        assert offset == len(pcm)
        assert payload["audio"]["output"][0]["checksum"] == "00000000"
        assert payload["audio"]["coverage"][0]["status"] == (
            "unavailable" if fault == "unavailable_audio" else "covered"
        )
        assert [c["kind"] for c in payload["completion"]["children"]] == ["video", "audio"]
    assert len(h.children) == (1 if fault == "late_exit" else 2)
    assert all(c.done for c in h.children)


def test_tagged_integer_pts_and_pixel_identity_cover_half_open_intervals() -> None:
    d = decoder()
    origin = make_origin()
    pixels = [bytes([i]) * 12 for i in (20, 40, 80)]
    tags = [d.parse_video(video_line(n, pts, pixels[n]), F(1, 1000))
            for n, pts in enumerate((5015, 5050, 5130))]
    result = d.select_frames(zip(tags, pixels), origin,
                             (F(0), F(15, 1000), F(49, 1000), F(50, 1000), F(130, 1000)),
                             d.DecodeLimits())
    assert [x.status for x in result.outcomes] == ["before_video_start", "covered", "covered",
                                                 "covered", "unknown_terminal"]
    assert result.frames[0].bgr == pixels[0] and result.frames[1].bgr == pixels[1]
    assert result.frames[0].raw_start == F(5015, 1000)
    assert result.frames[0].raw_end == F(5050, 1000)
    assert result.frames[0].sha256 != result.frames[1].sha256
    mutable = result.frames[0].array_copy()
    mutable[:] = 0
    assert result.frames[0].bgr == pixels[0]


@pytest.mark.parametrize("fault", ["checksum", "short", "duplicate", "order", "dimensions",
                                   "frames", "retained", "timebase", "missing"])
def test_protocol_disagreement_fails_without_qualified_receipt(fault: str) -> None:
    d = decoder()
    payload = b"\x03" * 12
    first = d.parse_video(video_line(0, 5015, payload), F(1, 1000))
    second = d.parse_video(video_line(1, 5050, payload), F(1, 1000))
    limits = d.DecodeLimits()
    pairs = [(first, payload), (second, payload)]
    if fault == "checksum":
        pairs[0] = (first, b"\x04" * 12)
    elif fault == "short":
        pairs[0] = (first, payload[:-1])
    elif fault == "duplicate":
        pairs[1] = (first, payload)
    elif fault == "order":
        pairs[1] = (d.parse_video(video_line(1, 5014, payload), F(1, 1000)), payload)
    elif fault == "dimensions":
        pairs[1] = (d.parse_video(video_line(1, 5050, payload, size="1x4"), F(1, 1000)), payload)
    elif fault == "frames":
        limits = d.DecodeLimits(max_frames=1)
    elif fault == "retained":
        limits = d.DecodeLimits(max_retained_bytes=1)
    elif fault == "timebase":
        pairs[1] = (d.parse_video(video_line(1, 5050, payload), F(1, 90000)), payload)
    elif fault == "missing":
        pairs[1] = (d.parse_video(video_line(2, 5050, payload), F(1, 1000)), payload)
    with pytest.raises(ValueError):
        d.select_frames(iter(pairs), make_origin(), (F(20, 1000),), limits)


def test_coalescing_preserves_windows_and_enforces_finite_limits() -> None:
    d = decoder()
    windows = [d.Window(F(1), F(2)), d.Window(F(21, 10), F(3))]
    assert d.coalesce(windows, d.DecodeLimits()) == (d.Window(F(1), F(3)),)
    with pytest.raises(ValueError):
        d.coalesce([d.Window(F(1), F(100))], d.DecodeLimits())
    for kwargs in ({"timeout": 46}, {"max_frames": 0}, {"preroll": F(-1)},
                   {"postroll": float("nan")}):
        with pytest.raises(ValueError):
            d.DecodeLimits(**kwargs)


def test_audio_tags_and_named_policies_are_explicit() -> None:
    d = decoder()
    tag = d.parse_audio("[ashowinfo@cxaout @ 000] n:0 pts:80001 pts_time:5.00006 "
                        "fmt:s16 channels:1 chlayout:mono rate:16000 nb_samples:160 checksum:00000000")
    assert tag.pts == 80001 and tag.time_base == F(1, 16000)
    assert d.audio_format("analysis16k-mono") == (16000, 1)
    assert d.audio_format("render48k-stereo") == (48000, 2)
    with pytest.raises(ValueError):
        d.audio_format("healed-any-rate")


def test_audio_only_policy_does_not_expand_combined_window_limit() -> None:
    d = decoder()
    with pytest.raises(ValueError):
        d.DecodeLimits(max_window=F(9))
    assert d.AudioDecodeLimits().max_window == F(300)
    with pytest.raises(ValueError):
        d.AudioDecodeLimits(max_window=F(301))
    for key, ceiling in (("timeout", 45), ("max_blocks", 65536), ("max_samples", 14400000),
                         ("max_decoded_bytes", 60000000), ("max_retained_bytes", 60000000),
                         ("max_log_bytes", 16000000)):
        for bad in (True, 0, ceiling + 1, 1.5):
            with pytest.raises(ValueError):
                d.AudioDecodeLimits(**{key: bad})
    for kwargs in ({"preroll": F(-1)}, {"postroll": F(2)}, {"max_window": float("nan")}):
        with pytest.raises(ValueError):
            d.AudioDecodeLimits(**kwargs)


def test_decoded_audio_window_detaches_and_checks_pcm_identity() -> None:
    import json
    d = decoder()
    origin = make_origin()
    payload = audio_payload()
    pcm = bytearray(9600)
    payload["audio"][0]["pcm_sha256"] = d.digest(bytes(pcm))
    payload["audio"][0]["output"][0]["sha256"] = d.digest(bytes(pcm))
    receipt = d.SourceAudioDecodeReceipt(origin.identity, origin.decoder_id, d.AUDIO_DECODE_POLICY, payload)
    result = d.DecodedAudioWindow(receipt, memoryview(pcm))
    pcm[0] = 1
    assert type(result.pcm) is bytes and result.pcm == bytes(9600)
    for bad in (pcm, bytes(9599), "0", 9600):
        with pytest.raises(ValueError):
            d.DecodedAudioWindow(receipt, bad)
    bad = json.loads(receipt.payload_json)
    bad["audio"][0]["output"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        d.DecodedAudioWindow(d.SourceAudioDecodeReceipt(origin.identity, origin.decoder_id,
                            d.AUDIO_DECODE_POLICY, bad), bytes(9600))


@pytest.mark.parametrize("fault", ["none", "seek", "long", "unavailable", "checksum", "sequence",
    "rate", "channels", "format", "gap", "conversion", "input_format", "retained", "decoded",
    "samples", "input_samples", "blocks", "input_blocks", "log", "timeout", "source_change"])
def test_audio_only_fake_child_protocol_and_cleanup(monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    import json
    import sys
    import threading
    from clipperx.render_clock import SourceTimeOrigin
    d = decoder()
    assert hasattr(d, "observe_audio_window"), "audio-only API absent"
    base = make_origin()
    origin = SourceTimeOrigin(base.path, base.size, base.mtime_ns, base.fingerprint,
        base.origin, F(12), base.streams, base.decoder_id, dict(base.color))
    # Unknown stream terminal bounds permit a long fake interval without asserting video coverage.
    audio = d.StreamClock(1, "audio", F(1, 44100), F(5), None,
                          sample_rate=44100, channels=1, sample_format="s16")
    origin = SourceTimeOrigin(origin.path, origin.size, origin.mtime_ns, origin.fingerprint,
        origin.origin, origin.duration, [base.streams[0], audio], origin.decoder_id, {})
    monkeypatch.setattr(d, "_runtime_identity", lambda runtime: origin.decoder_id)
    monkeypatch.setattr(d, "_verify_pins", lambda runtime: None)
    fingerprints = iter([origin.fingerprint, "f" * 16 if fault == "source_change" else origin.fingerprint])
    monkeypatch.setattr(d, "source_fingerprint", lambda path: next(fingerprints))
    from types import SimpleNamespace
    runtime = SimpleNamespace(ffmpeg=Path("HOST-ffmpeg"))
    seconds = 9 if fault == "long" else F(1, 10)
    in_samples, out_samples = int(seconds * 44100), int(seconds * 48000)
    input_pts, output_pts = 220500, 240000
    window = d.Window(F(1, 2), F(3, 5)) if fault == "seek" else d.Window(F(0), F(seconds))
    if fault == "seek":
        input_pts += 22050
        output_pts += 24000
    if fault == "unavailable":
        input_pts += 441
        output_pts += 480
        in_samples -= 441
        out_samples -= 480
    if fault == "conversion":
        out_samples -= 10
    payload = bytes(out_samples * 2)
    out_rate = 16000 if fault == "rate" else 48000
    channels = 3 if fault == "channels" else 1
    fmt = "fltp" if fault == "format" else "s16"
    checksum = "12345678" if fault == "checksum" else "00000000"
    n = 1 if fault == "sequence" else 0
    input_fmt = "fltp" if fault == "input_format" else "s16"
    lines = (f"[ashowinfo@cxain @ 000] n:0 pts:{input_pts} fmt:{input_fmt} channels:1 "
        f"rate:44100 nb_samples:{in_samples} checksum:00000000\n"
        f"[ashowinfo@cxaout @ 000] n:{n} pts:{output_pts} fmt:{fmt} channels:{channels} "
        f"rate:{out_rate} nb_samples:{out_samples} checksum:{checksum}\n")
    if fault in ("blocks", "input_blocks", "gap"):
        if fault != "blocks":
            lines += (f"[ashowinfo@cxain @ 000] n:1 pts:{input_pts + in_samples} fmt:s16 channels:1 "
                      f"rate:44100 nb_samples:1 checksum:00000000\n")
        if fault != "input_blocks":
            lines += (f"[ashowinfo@cxaout @ 000] n:1 pts:{output_pts + out_samples + (1 if fault == 'gap' else 0)} "
                      "fmt:s16 channels:1 rate:48000 nb_samples:1 checksum:00000000\n")
            payload += bytes(2)
    script = ("import sys,time;sys.stderr.write('HOST diagnostic\\n'+" + repr(lines)
              + ");sys.stderr.flush();sys.stdout.buffer.write(bytes(" + str(len(payload)) + "));sys.stdout.flush()")
    if fault == "timeout":
        script = "import sys,time;sys.stderr.write('HOST timeout\\n');sys.stderr.flush();time.sleep(3)"
    original = subprocess.Popen
    children, commands = [], []
    def tracked(argv: Any, **kwargs: Any) -> Any:
        commands.append(argv)
        child = original([sys.executable, "-c", script], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(d.subprocess, "Popen", tracked)
    kwargs = {"preroll": F(0), "postroll": F(0), "timeout": 1 if fault == "timeout" else 3}
    if fault in ("retained", "decoded", "samples", "input_samples", "blocks", "input_blocks", "log"):
        key = {"retained": "max_retained_bytes", "decoded": "max_decoded_bytes",
            "samples": "max_samples", "input_samples": "max_samples", "blocks": "max_blocks",
            "input_blocks": "max_blocks", "log": "max_log_bytes"}[fault]
        kwargs[key] = 4500 if fault == "samples" else 1
    limits = d.AudioDecodeLimits(**kwargs)
    if fault in ("none", "seek", "long", "unavailable"):
        result = d.observe_audio_window(runtime, origin, window,
                                       audio_policy="render48k-mono", limits=limits)
        assert result.pcm == payload
        data = json.loads(result.receipt.to_json())
        assert data["kind"] == "audio_only" and "video" not in data["payload"]
        assert data["payload"]["argv"] == commands
        assert data["payload"]["audio"][0]["coverage"][0]["status"] == (
            "unavailable" if fault == "unavailable" else "covered")
        # Independently stated prior safe literal; do not derive it from the builder.
        seek = ["-ss", "0.500000"] if fault == "seek" else []
        duration = "9.000000" if fault == "long" else "0.100000"
        trim = "14.000000" if fault == "long" else "5.600000" if fault == "seek" else "5.100000"
        expected = ["HOST-ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-loglevel", "info",
            "-copyts", *seek, "-t", duration, "-threads", "2", "-i", "HOST", "-map", "0:1",
            "-vn", "-sn", "-dn", "-af",
            f"atrim=end={trim},asettb=1/44100,ashowinfo@cxain,aresample=48000:async=0,"
            "aformat=sample_fmts=s16:channel_layouts=mono,asettb=1/48000,ashowinfo@cxaout",
            "-c:a", "pcm_s16le", "-threads", "1", "-f", "s16le", "pipe:1"]
        assert commands == [expected]
    else:
        with pytest.raises(ValueError) as captured:
            d.observe_audio_window(runtime, origin, window,
                                   audio_policy="render48k-mono", limits=limits)
        if fault != "source_change":
            assert isinstance(captured.value, d.DecodeFailure)
            details = captured.value.record()
            assert details["argv"] == commands[0] and "HOST" in details["stderr_tail"]
            assert details["stdout_bytes"] <= len(payload)
    assert len(commands) == 1 and "-vn" in commands[0] and "-vf" not in commands[0]
    assert commands[0].count("-map") == 1 and commands[0][commands[0].index("-map") + 1] == "0:1"
    assert all(child.poll() is not None for child in children)
    assert not any(t.name == "source-decode-metadata" for t in threading.enumerate())


@pytest.mark.parametrize("fault", ["window", "effective", "policy", "rate", "channels", "runtime",
                                  "source", "limits_type", "window_type"])
def test_audio_only_invalid_inputs_launch_no_children(monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    from types import SimpleNamespace
    d = decoder()
    assert hasattr(d, "observe_audio_window"), "audio-only API absent"
    base = make_origin()
    audio = d.StreamClock(1, "audio", F(1, 44100), F(5), F(7),
        sample_rate=22050 if fault == "rate" else 44100, channels=3 if fault == "channels" else 1,
        sample_format="s16")
    origin = d.SourceTimeOrigin(base.path, base.size, base.mtime_ns, base.fingerprint,
        base.origin, base.duration, [base.streams[0], audio], base.decoder_id, {})
    monkeypatch.setattr(d, "_runtime_identity", lambda runtime: "wrong" if fault == "runtime" else origin.decoder_id)
    monkeypatch.setattr(d, "_verify_pins", lambda runtime: None)
    monkeypatch.setattr(d, "source_fingerprint", lambda path: "f" * 16 if fault == "source" else origin.fingerprint)
    children = []
    monkeypatch.setattr(d.subprocess, "Popen", lambda *args, **kwargs: children.append(args))
    window = d.Window(F(0), F(3) if fault == "window" else F(1))
    limits = d.AudioDecodeLimits(max_window=F(1) if fault == "effective" else F(300))
    with pytest.raises(ValueError):
        d.observe_audio_window(SimpleNamespace(ffmpeg=Path("HOST")), origin,
            [window] if fault == "window_type" else window,
            audio_policy="bad" if fault == "policy" else "analysis16k-mono",
            limits=d.DecodeLimits() if fault == "limits_type" else limits)
    assert children == []


def test_pipe_reader_rejects_byte_mismatch_and_timeout() -> None:
    import sys
    d = decoder()
    for script in ("import sys;sys.stderr.write('HOST\\n');sys.stdout.buffer.write(b'x')",
                   "import time;time.sleep(3)"):
        with pytest.raises(ValueError):
            list(d.stream_payloads([sys.executable, "-c", script], "video", F(1),
                                   d.DecodeLimits(timeout=1)))


def test_pipe_progress_carriage_return_cannot_hide_a_tag() -> None:
    import sys
    d = decoder()
    first, second, third = b"\x03" * 12, b"\x07" * 12, b"\x09" * 12
    lines = ("[showinfo@cxv @ 000] config in time_base: 1/1000, frame_rate: 10/1\n"
             + video_line(0, 5015, first) + "\n"
             + "frame=1 fps=0\r" + video_line(1, 5050, second) + "\n"
             + video_line(2, 5130, third) + "\n")
    script = ("import sys;sys.stderr.write(" + repr(lines) + ");sys.stderr.flush();"
              "sys.stdout.buffer.write(" + repr(first + second + third) + ")")
    result = list(d.stream_payloads([sys.executable, "-c", script], "video", F(1, 1000),
                                    d.DecodeLimits(timeout=3)))
    assert [tag.n for tag, payload in result] == [0, 1, 2]
    assert [payload for tag, payload in result] == [first, second, third]


def test_pipe_failure_preserves_bounded_diagnostics_without_rgb_capture() -> None:
    import sys
    d = decoder()
    assert hasattr(d, "DecodeFailure"), "Structured failure diagnostics absent"
    script = "import sys;sys.stderr.write('diagnostic\\n'*10000);sys.stdout.buffer.write(b'bad')"
    command = [sys.executable, "-c", script]
    with pytest.raises(d.DecodeFailure) as captured:
        list(d.stream_payloads(command, "video", F(1), d.DecodeLimits(timeout=3)))
    details = captured.value.record()
    assert details["argv"] == command
    assert len(details["stderr_tail"].encode()) <= 65536
    assert details["stderr_truncated"] is True
    assert details["stdout_bytes"] == 1
    assert details["stdout_sha256"] == d.digest(b"b")
    assert "without associated metadata" in details["error"]


@pytest.mark.parametrize("fault", ["sequence", "dimensions", "post_iteration_audio"])
def test_consumer_failures_retain_diagnostics_and_cleanup(
        monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    import sys
    import threading
    d = decoder()
    assert hasattr(d, "payload_stream"), "Consumer diagnostic context absent"
    children = []
    original = subprocess.Popen
    def tracked(*args: Any, **kwargs: Any) -> Any:
        process = original(*args, **kwargs)
        children.append(process)
        return process
    monkeypatch.setattr(d.subprocess, "Popen", tracked)
    pixels = b"\x03" * 12
    n = 2 if fault == "sequence" else 1
    size = "1x4" if fault == "dimensions" else "2x2"
    lines = ("[showinfo@cxv @ 000] config in time_base: 1/1000, frame_rate: 10/1\n"
             + video_line(0, 5015, pixels) + "\n" + video_line(n, 5050, pixels, size=size) + "\n")
    script = ("import sys;sys.stderr.write('diagnostic\\n'*10000+" + repr(lines) + ");sys.stderr.flush();"
              "sys.stdout.buffer.write(" + repr(pixels * 2) + ")")
    argv = [sys.executable, "-c", script]
    with pytest.raises(d.DecodeFailure) as captured:
        with d.payload_stream(argv, "video", F(1, 1000), d.DecodeLimits(timeout=3)) as pairs:
            d.select_frames(pairs, make_origin(), (F(1, 50),), d.DecodeLimits())
            if fault == "post_iteration_audio":
                d.validate_audio_conversion(441, 44100, F(5), 4800, 48000, F(5))
    details = captured.value.record()
    assert details["argv"] == argv and details["stdout_bytes"] == 24
    assert len(details["stderr_tail"].encode()) <= 65536 and details["stderr_truncated"]
    assert all(child.poll() is not None for child in children)
    assert not any(t.name == "source-decode-metadata" for t in threading.enumerate())


@pytest.mark.parametrize("offset,source_rate,policy,source_channels", [
    (15, 44100, "analysis16k-mono", 1), (59, 48000, "render48k-stereo", 1),
    (59, 44100, "render48k-mono", 1), (15, 48000, "analysis16k-mono", 1),
    (15, 44100, "render48k-stereo", 1), (59, 48000, "render48k-mono", 1),
    (15, 44100, "analysis16k-mono", 2), (59, 48000, "render48k-stereo", 2)])
def test_host_offset_vfr_reorder_pixels_and_audio_phase(
        tmp_path: Path, offset: int, source_rate: int, policy: str, source_channels: int) -> None:
    """Tiny CPU source construction, NOT production encode/delivered AAC QA."""
    import array
    import json
    import wave
    from clipperx.render_runtime import resolve_render_runtime

    d = decoder()
    runtime = resolve_render_runtime(d.decode_requirements(), device_index=None)
    ffmpeg = str(runtime.ffmpeg)
    rgb = tmp_path / "HOST.rgb"
    colors = [(20 + n * 17, 40 + n * 11, 60 + n * 7) for n in range(12)]
    rgb.write_bytes(b"".join(bytes(color) * (16 * 16) for color in colors))
    wav = tmp_path / "HOST.wav"
    audio = array.array("h", [0] * (source_rate * 2 * source_channels))
    audio[(source_rate // 2) * source_channels] = 20000
    audio[source_rate * source_channels + source_channels - 1] = -20000
    with wave.open(str(wav), "wb") as stream:
        stream.setnchannels(source_channels)
        stream.setsampwidth(2)
        stream.setframerate(source_rate)
        stream.writeframes(audio.tobytes())
    source = tmp_path / "HOST-offset-vfr-bframes.nut"
    expression = f"settb=1/1000,setpts=5000+{offset}+N*100+if(gte(N\\,4)\\,70\\,0)"
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
               "-pixel_format", "rgb24", "-video_size", "16x16", "-framerate", "10",
               "-i", str(rgb), "-i", str(wav), "-vf", expression,
               "-af", "asetpts=PTS+5/TB", "-fps_mode", "passthrough", "-enc_time_base:v", "1/1000",
               "-c:v", "libx264rgb", "-crf", "18", "-bf", "2", "-g", "12",
               "-x264-params", "b-adapt=0:scenecut=0",
               "-c:a", "pcm_s16le", "-threads", "1", "-y", str(source)]
    built = subprocess.run(command, capture_output=True, timeout=15, check=False)
    assert built.returncode == 0, built.stderr.decode(errors="replace")
    # Independent frame-only reference: no showinfo parser, no requested-time labels.
    reference = subprocess.run([ffmpeg, "-v", "error", "-i", str(source), "-an",
        "-fps_mode", "passthrough", "-pix_fmt", "bgr24", "-threads", "1", "-f", "rawvideo", "pipe:1"],
        capture_output=True, timeout=15, check=False)
    assert reference.returncode == 0 and len(reference.stdout) == 12 * 16 * 16 * 3
    frame_size = 16 * 16 * 3
    for n, color in enumerate(colors):
        actual = reference.stdout[n * frame_size:n * frame_size + 3]
        assert all(abs(a - b) <= 5 for a, b in zip(actual, reversed(color)))
    reorder = subprocess.run([str(runtime.ffprobe), "-v", "error", "-select_streams", "v:0",
        "-show_entries", "frame=pict_type,pts", "-of", "json", str(source)],
        capture_output=True, timeout=15, check=False)
    frame_info = json.loads(reorder.stdout)["frames"]
    assert "B" in {frame["pict_type"] for frame in frame_info}
    expected_pts = [5000 + offset + n * 100 + (70 if n >= 4 else 0) for n in range(12)]
    # NUT retains a finer rational video timebase; compare raw times below.
    origin = d.probe_source(source, runtime)
    assert origin.origin == F(5)
    video_stream = next(s for s in origin.streams if s.kind == "video")
    assert video_stream.start == F(5000 + offset, 1000)
    assert [frame["pts"] * video_stream.time_base for frame in frame_info] == [F(p, 1000) for p in expected_pts]
    seek_case = policy == "render48k-mono"
    start = F(1, 4) if seek_case else F(0)
    preroll = F(1, 10) if seek_case else F(0)
    requests = [F(offset + 350, 1000), F(offset + 470, 1000)]
    if not seek_case:
        requests = [F(0), F(offset, 1000), *requests]
    observed = d.observe_windows(runtime, origin, [d.Window(start, F(11, 10))], requests,
                                audio_policy=policy, limits=d.DecodeLimits(preroll=preroll, postroll=F(1, 5)))
    result = observed[0]
    receipt = json.loads(result.receipt.to_json())["payload"]
    assert d.SourceDecodeReceipt.from_json(result.receipt.to_json(), origin=origin,
                                          policy=d.DECODE_POLICY) == result.receipt
    if not seek_case:
        assert receipt["outcomes"][0]["status"] == "before_video_start"
    selected = (3, 4) if seek_case else (0, 3, 4)
    assert [frame.raw_start for frame in result.frames] == [F(expected_pts[n], 1000) for n in selected]
    for frame, n in zip(result.frames, selected):
        assert frame.bgr == reference.stdout[n * frame_size:(n + 1) * frame_size]
    rate, channels = d.audio_format(policy)
    samples = array.array("h")
    samples.frombytes(result.pcm)
    mono = samples[::channels]
    phase = F(3, 20) if seek_case else F(0)
    assert abs(max(range(len(mono)), key=lambda i: mono[i]) - (F(1, 2) - phase) * rate) <= 1
    negative_channel = samples[1::2] if source_channels == channels == 2 else mono
    assert abs(min(range(len(negative_channel)), key=lambda i: negative_channel[i]) - (1 - phase) * rate) <= 1
    if source_channels == channels == 2:
        assert mono[rate // 2] == 20000 and negative_channel[rate] == -20000
        assert mono[rate] == 0 and negative_channel[rate // 2] == 0
    elif source_channels == 2:
        assert max(mono) > 0 and min(mono) < 0
    assert len(mono) == (F(13, 10) - phase) * rate
    assert receipt["audio"][0]["sample_grid_origin"] == d.rational(F(5) + phase)
    assert receipt["audio"][0]["source_phase"] == d.rational(phase)


@pytest.mark.parametrize("source_rate,policy", [(44100, "analysis16k-mono"),
                                                (48000, "render48k-stereo")])
def test_host_audio_only_continuous_pcm_reference(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_rate: int, policy: str) -> None:
    """Two tiny synthetic controls; independent uninterrupted WAV-to-PCM reference."""
    import array
    import json
    import threading
    import wave
    from clipperx.render_runtime import resolve_render_runtime

    d = decoder()
    runtime = resolve_render_runtime(d.decode_requirements(), device_index=None)
    long = source_rate == 44100
    duration = 12 if long else 2
    # Bracket both source8s and8s after decoder seek0.25s. No restart at either.
    impulses = ([(F(799, 100), 0, 20000), (F(801, 100), 1, -18000),
                 (F(824, 100), 0, 16000), (F(826, 100), 1, -14000)] if long else
                [(F(1, 2), 0, 20000), (F(1), 1, -20000)])
    audio = array.array("h", [0]) * (source_rate * duration * 2)
    for time, channel, value in impulses:
        sample = time * source_rate
        assert sample.denominator == 1
        audio[int(sample) * 2 + channel] = value
    wav = tmp_path / "HOST-audio-only.wav"
    with wave.open(str(wav), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(source_rate)
        stream.writeframes(audio.tobytes())
    # Exactly one tiny video stream satisfies the established metadata probe.
    # Audio-only observation below must never launch a video decoder.
    rgb = tmp_path / "HOST-audio-only.rgb"
    rgb.write_bytes(bytes([20, 40, 60]) * 16 * 16 * duration)
    source = tmp_path / "HOST-audio-only-offset.nut"
    command = [str(runtime.ffmpeg), "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
        "-pixel_format", "rgb24", "-video_size", "16x16", "-framerate", "1",
        "-i", str(rgb), "-i", str(wav), "-vf", "settb=1/1000,setpts=5000+N*1000",
        "-af", "asetpts=PTS+5/TB", "-fps_mode", "passthrough", "-enc_time_base:v", "1/1000",
        "-c:v", "rawvideo", "-pix_fmt", "rgb24", "-c:a", "pcm_s16le", "-threads", "1",
        "-y", str(source)]
    built = subprocess.run(command, capture_output=True, timeout=15, check=False)
    assert built.returncode == 0, built.stderr.decode(errors="replace")
    origin = d.probe_source(source, runtime)
    assert origin.origin == F(5)
    if long:
        window = d.Window(F(3, 4), F(11))
        limits = d.AudioDecodeLimits()
        expected_start, expected_end = F(1, 4), F(45, 4)
        reference_start, reference_duration = "0.250000", "11.000000"
    else:
        window = d.Window(F(1, 4), F(3, 2))
        limits = d.AudioDecodeLimits(preroll=F(1, 8), postroll=F(1, 8))
        expected_start, expected_end = F(1, 8), F(13, 8)
        reference_start, reference_duration = "0.125000", "1.500000"
    if long:
        assert expected_end - expected_start > 8
        with pytest.raises(ValueError):
            d.coalesce([window], d.DecodeLimits())
    children, executed = [], []
    original = subprocess.Popen
    def tracked(argv: Any, **kwargs: Any) -> Any:
        executed.append(argv)
        child = original(argv, **kwargs)
        children.append(child)
        return child
    with monkeypatch.context() as patch:
        patch.setattr(d.subprocess, "Popen", tracked)
        result = d.observe_audio_window(runtime, origin, window, audio_policy=policy, limits=limits)
    assert len(executed) == 1 and "-vn" in executed[0] and "-vf" not in executed[0]
    assert all(child.poll() is not None for child in children)
    assert not any(t.name == "source-decode-metadata" for t in threading.enumerate())
    rate, channels = d.audio_format(policy)
    layout = "mono" if channels == 1 else "stereo"
    # Independent source WAV, explicit known interval, one continuous resampler,
    # no observation parser, origin metadata, showinfo, or block reconstruction.
    reference_command = [str(runtime.ffmpeg), "-hide_banner", "-nostdin", "-v", "error",
        "-ss", reference_start, "-t", reference_duration, "-i", str(wav), "-map", "0:a:0",
        "-vn", "-sn", "-dn", "-af",
        f"aresample={rate}:async=0,aformat=sample_fmts=s16:channel_layouts={layout}",
        "-c:a", "pcm_s16le", "-threads", "1", "-f", "s16le", "pipe:1"]
    reference = subprocess.run(reference_command, capture_output=True, timeout=15, check=False)
    assert reference.returncode == 0, reference.stderr.decode(errors="replace")
    assert len(reference.stdout) <= limits.max_retained_bytes
    assert result.pcm == reference.stdout
    expected_samples = (expected_end - expected_start) * rate
    assert expected_samples.denominator == 1 and len(result.pcm) == expected_samples * channels * 2
    receipt = json.loads(result.receipt.to_json())
    assert receipt["kind"] == "audio_only"
    payload = receipt["payload"]
    assert set(payload) == {"windows", "limits", "audio", "argv"} and payload["argv"] == executed
    assert d.SourceAudioDecodeReceipt.from_json(result.receipt.to_json(), origin=origin,
                                              policy=d.AUDIO_DECODE_POLICY) == result.receipt
    record = payload["audio"][0]
    assert record["samples"] == expected_samples
    assert sum(block["samples"] for block in record["input"]) == (expected_end - expected_start) * source_rate
    assert F(record["sample_grid_origin"]) == origin.origin + expected_start
    assert F(record["source_phase"]) == expected_start
    assert record["coverage"] == [{"window": [d.rational(window.start), d.rational(window.end)],
                                   "status": "covered"}]
    assert record["pcm_sha256"] == d.digest(reference.stdout)
    # Verify every retained byte/block hash and exact sample-grid continuity.
    offset = 0
    previous = None
    for block in record["output"]:
        chunk = result.pcm[offset:offset + block["bytes"]]
        assert d.digest(chunk) == block["sha256"]
        assert f"{zlib.adler32(chunk, 0):08X}" == block["checksum"]
        if previous is not None:
            assert block["pts"] == previous["pts"] + previous["samples"]
        previous = block
        offset += block["bytes"]
    assert offset == len(result.pcm)
    decoded = array.array("h")
    decoded.frombytes(result.pcm)
    for time, channel, value in impulses:
        expected_index = int((time - expected_start) * rate)
        observed_channel = decoded[channel::2] if channels == 2 else decoded
        neighborhood = observed_channel[expected_index - 1:expected_index + 2]
        assert (max(neighborhood) > 0 if value > 0 else min(neighborhood) < 0)
        if channels == 2:
            assert observed_channel[expected_index] == value
            assert decoded[(1 - channel)::2][expected_index] == 0
    print(d.canonical({"HOST": policy, "source_seconds": duration, "origin": d.rational(origin.origin),
        "decoder_start": payload["limits"]["decoder_start"], "decoder_end": payload["limits"]["decoder_end"],
        "input_samples": sum(b["samples"] for b in record["input"]), "output_samples": record["samples"],
        "input_blocks": len(record["input"]), "output_blocks": len(record["output"]),
        "sample_grid_origin": record["sample_grid_origin"], "source_phase": record["source_phase"],
        "bytes": len(result.pcm), "pcm_sha256": d.digest(result.pcm),
        "reference_sha256": d.digest(reference.stdout), "executed_audio_children": len(executed)}))


# Task45: scoped additions only. All preceding helper/test bodies are preserved.
@pytest.fixture
def hosted_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    import ctypes
    import threading
    import queue
    from clipperx.tests.test_source_host_ownership import forbidden, guard_source_stat
    guard_source_stat(monkeypatch)
    d = decoder()
    monkeypatch.setattr(ctypes, "WinDLL", forbidden, raising=False)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(threading.Thread, "start", forbidden)
    monkeypatch.setattr(threading.Thread, "join", forbidden)
    monkeypatch.setattr(threading.Timer, "start", forbidden)
    monkeypatch.setattr(threading.Event, "wait", forbidden)
    monkeypatch.setattr(d, "source_fingerprint", forbidden)
    monkeypatch.setattr(d, "_verify_pins", forbidden)
    monkeypatch.setattr(d, "probe_source", forbidden)
    class ScriptQueue(queue.Queue[Any]):
        def get(self, block: bool = True, timeout: float | None = None) -> Any:
            return super().get(False)
        def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:
            super().put(item, False)
    monkeypatch.setattr(queue, "Queue", ScriptQueue)


def _hosted_float_setup(monkeypatch: pytest.MonkeyPatch, fault: str = "") -> Any:
    import io
    from clipperx.tests.test_source_host_ownership import admitted
    from clipperx.tests.test_windows_pipe_io import FakeHandle
    env = _float_fake_host(monkeypatch, fault)
    owner = admitted()
    factory = env.d.subprocess.Popen
    def process(argv: tuple[str, ...]) -> Any:
        child = factory(argv)
        child.pid, child.returncode = 90 + len(env.children), None
        child._handle = FakeHandle(400 + len(env.children))
        child.terminate = child.kill
        wait = child.wait
        def reaped(timeout: float) -> int:
            child.returncode = wait(timeout)
            return child.returncode
        child.wait = reaped
        if fault == "two":
            lines = []
            for stage in ("cxain", "cxaout"):
                for n in range(2):
                    lines.append(f"[ashowinfo@{stage} @ 000] n:{n} pts:{2*n} fmt:flt channels:1 rate:48000 "
                        f"nb_samples:2 checksum:{zlib.adler32(env.pcm[8*n:8*n+8], 0):08X}\n")
            child.stderr = io.BytesIO("".join(lines).encode())
        return child
    owner.job._processes = process
    monkeypatch.setattr(env.d.subprocess, "Popen", lambda argv, **kwargs: process(tuple(argv)))
    env.owner = owner
    monkeypatch.setattr(env.d.time, "monotonic_ns", lambda: owner.now)
    return env


@pytest.mark.usefixtures("hosted_boundaries")
@pytest.mark.parametrize("fault", ["", "two", "exact_cap", "post_source", "checksum", "observed_phase", "trailing"])
def test_hosted_float_real_path(monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    from dataclasses import replace
    from clipperx.tests.test_source_host_ownership import finish_job
    env = _hosted_float_setup(monkeypatch, fault)
    d, h = env.d, env.owner
    control = h.job.new_observation("materialized_f32", deadline_ns=10000)
    window = d.Window(F(0), F(1, 12000))
    if fault == "observed_phase":
        env.limits = replace(env.limits, max_metadata_bytes=3700)
        start = -env.origin.origin
        window = d.Window(start, start + F(1, 12000))
        env.forbid_reads[0] = True
    if fault in ("exact_cap", "trailing"):
        env.limits = replace(env.limits, max_decoded_bytes=16, max_retained_bytes=16,
            max_samples=4, max_block_samples=4)
    kwargs = dict(audio_policy="render48k-f32-mono-v1", limits=env.limits)
    if fault in ("post_source", "checksum", "observed_phase", "trailing"):
        with pytest.raises(ValueError):
            d.observe_float_audio_window(env.runtime, env.origin, window,
                _host=control, **kwargs)
        assert h.job.snapshot().failure is not None
        if fault == "observed_phase":
            assert env.reads == []
        if fault == "trailing":
            assert control.ownership_snapshot().diagnostics[0].stdout_bytes == 17
    else:
        result = d.observe_float_audio_window(env.runtime, env.origin, d.Window(F(0), F(1, 12000)),
            _host=control, **kwargs)
        assert result.pcm == env.pcm
        assert control.ownership_snapshot().phase == "retired_ok"
        assert control.ownership_snapshot().window_resources_zero
        legacy = d.observe_float_audio_window(env.runtime, env.origin, d.Window(F(0), F(1, 12000)), **kwargs)
        assert legacy.pcm == result.pcm and legacy.receipt.to_json() == result.receipt.to_json()
    assert finish_job(h).own0


@pytest.mark.usefixtures("hosted_boundaries")
@pytest.mark.parametrize("fault", ["", "borrowed", "extra"])
def test_hosted_stream_real_path(monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    from clipperx.render_clock import StreamClock, StreamDecodeLimits, SourceStreamFinalization
    from clipperx.tests.test_source_host_ownership import ScriptProcess, finish_job
    env = _hosted_float_setup(monkeypatch)
    d, h = env.d, env.owner
    # Real float followed by real streamed observation on the same driver handle.
    first = h.job.new_observation("materialized_f32", deadline_ns=10000)
    d.observe_float_audio_window(env.runtime, env.origin, d.Window(F(0), F(1, 12000)),
        audio_policy="render48k-f32-mono-v1", limits=env.limits, _host=first)
    first_generation = first.ownership_snapshot().diagnostics[0].generation
    # Reuse after a certain close is legitimate; the live driver handle differs.
    h.kernel.output -= 1
    video = StreamClock(0, "video", F(1, 48000), F(0), F(1), width=1, height=1)
    origin = d.SourceTimeOrigin(env.origin.path, env.origin.size, env.origin.mtime_ns, env.origin.fingerprint,
        F(0), F(1), (video, env.origin.streams[1]), env.origin.decoder_id, {})
    limits = StreamDecodeLimits(preroll=F(0), postroll=F(0), max_window=F(1), timeout=1,
        max_frames=3, max_samples=8, max_pixels=1, max_decoded_bytes=48, max_log_bytes=4096,
        window_timeout=2, cleanup_grace=1, max_requests=2, max_deliveries=2,
        io_chunk_bytes=16, max_live_payload_bytes=128, max_selected_unique_bytes=48,
        max_request_emission_bytes=48, max_finalization_bytes=100000)
    children: list[Any] = []
    def process(argv: tuple[str, ...]) -> Any:
        is_video = "-vf" in argv
        if is_video:
            raw = b"".join(bytes([n]) * 3 for n in range(3))
            text = "[showinfo@cxv @ 000] config in time_base: 1/48000\n"
            text += "\n".join(video_line(n, n*2, bytes([n])*3, size="1x1") for n in range(3)) + "\n"
            if fault == "extra":
                raw += b"x"
        else:
            assert children[0].stdout.closed and children[0].stderr.closed
            raw = bytes(10)
            text = ("[ashowinfo@cxain @ 000] n:0 pts:0 fmt:flt channels:1 rate:48000 nb_samples:5 checksum:00000000\n"
                "[ashowinfo@cxaout @ 000] n:0 pts:0 fmt:s16 channels:1 rate:48000 nb_samples:5 checksum:00000000\n")
        child = ScriptProcess(raw, text.encode(), pid=100 + len(children))
        children.append(child)
        return child
    h.job._processes = process
    control = h.job.new_observation("streamed_s16", deadline_ns=10000)
    def consume(host: Any) -> Any:
        with d.stream_observed_window(env.runtime, origin, d.Window(F(0), F(5, 48000)),
                (F(1, 48000), F(3, 48000)), audio_policy="render48k-mono", limits=limits, _host=host) as stream:
            for i in range(2):
                selection = stream.next_selection()
                assert selection is not None and selection.delivery_index == i
                if fault == "borrowed":
                    raise ValueError("borrowed failure")
                stream.acknowledge(i)
                del selection
            assert stream.next_selection() is None
            final = stream.finish()
            if host:
                assert host.ownership_snapshot().scope_open
                assert host.ownership_snapshot().window_resources_zero
            loaded = SourceStreamFinalization.from_json(final.to_json(), origin=origin, limits=limits,
                policy=d.STREAM_DECODE_POLICY, expected_session_id=stream.session_id)
            assert loaded.to_json() == final.to_json()
            return final
    if fault:
        with pytest.raises(ValueError):
            consume(control)
        assert h.job.snapshot().failure is not None
    else:
        final = consume(control)
        assert control.ownership_snapshot().phase == "retired_ok"
        assert all(row.generation > first_generation for row in control.ownership_snapshot().diagnostics)
        assert len(children) == 2 and h.kernel.events.count("duplicate") == 4
        # No-host path uses the same real selector/PCM/receipt loops with scripted workers.
        class Inline:
            daemon = False
            ident = None
            def __init__(self, target: Any, args: Any = (), **kwargs: Any) -> None:
                self.target, self.args = target, args
            def start(self) -> None:
                self.target(*self.args)
            def join(self, timeout: float = 0) -> None:
                pass
            def is_alive(self) -> bool:
                return False
        monkeypatch.setattr(d.threading, "Thread", Inline)
        class StreamTimer:
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                pass
            def start(self) -> None:
                pass
            def cancel(self) -> None:
                pass
            def join(self, timeout: float = 0) -> None:
                pass
            def is_alive(self) -> bool:
                return False
        monkeypatch.setattr(d.threading, "Timer", StreamTimer)
        monkeypatch.setattr(d.subprocess, "Popen", lambda argv, **kwargs: process(tuple(argv)))
        children.clear()
        legacy = consume(None)
        assert legacy.record()["payload"]["audio"] == final.record()["payload"]["audio"]
    assert finish_job(h).own0
