"""Explicit bounded CPU observation through the supplied pinned FFmpeg pair.

One ordered tagged metadata stream is matched to the SAME process's raw bytes.
Only selected video payloads survive; two neighboring frames suffice for coverage.
No FPS-derived timestamps, source padding, independent stream rebasing or fallback.
"""

from collections.abc import Generator, Iterable, Sequence, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import queue
import re
import subprocess
import struct
import threading
import time
from typing import Any, Protocol, TypedDict
import uuid
import zlib

from .local_paths import source_fingerprint, SourceIdentityFiles
from ._source_host_ownership import (HostedObservationControl, HostedObservationFailure,
                                     ChildLease, ObservationKind)
from .render_clock import (SourceDecodeReceipt, SourceAudioDecodeReceipt, SourceTimeOrigin, StreamClock,
                           canonical, digest, exact, rational, validate_audio_conversion,
                           validate_decode_limits, validate_audio_decode_limits, audio_format,
                           audio_decode_argv, decoder_input_argv, video_decode_argv,
                           SourceStreamFinalization, StreamDecodeLimits, STREAM_DECODE_POLICY,
                           _stream_session, _stream_int, _stream_fraction, _stream_rational,
                           _stream_guard, _stream_link, _stream_keys, _stream_header, _stream_origin,
                           _validate_video_origin_bounds)
from .render_clock import (DECODE_POLICY, validate_float_audio_limits, source_decoder_identity,
                           _FLOAT_AUDIO, _S16_AUDIO, _AudioContract, float_audio_decode_argv,
                           SourceFloatAudioDecodeReceipt, _bounded_json, _float_origin)
from .render_runtime import (FFMPEG_SHA256, FFPROBE_SHA256, NATIVE_RUNTIME,
                             RenderRequirements, RenderRuntime, RuntimeHashFiles, RuntimeToolRunner, _verify_pins)


class SourceFiles(RuntimeHashFiles, SourceIdentityFiles, Protocol):
    pass


class _SourceFileOptions(TypedDict, total=False):
    _files: SourceFiles

AUDIO_DECODE_POLICY = DECODE_POLICY + ";audio-only;v1"


class DecodeFailure(ValueError):
    """Bounded diagnostics for a failed child, never a qualified receipt."""

    def __init__(self, error: str, argv: Sequence[str], stderr_tail: bytes,
                 stderr_truncated: bool, stdout_bytes: int, stdout_sha256: str) -> None:
        super().__init__(error)
        self.diagnostic_json = canonical({"error": error, "argv": list(argv),
            "stderr_tail": stderr_tail.decode("utf-8", errors="replace"),
            "stderr_truncated": stderr_truncated, "stdout_bytes": stdout_bytes,
            "stdout_sha256": stdout_sha256, "stdout_scope": "consumed prefix; no RGB retained"})

    def record(self) -> dict[str, object]:
        return dict(json.loads(self.diagnostic_json))


@dataclass(frozen=True, slots=True)
class Window:
    start: Fraction
    end: Fraction

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", exact(self.start))
        object.__setattr__(self, "end", exact(self.end))
        if not 0 <= self.start < self.end:
            raise ValueError("Invalid bounded window")


@dataclass(frozen=True, slots=True)
class DecodeLimits:
    preroll: Fraction = Fraction(1, 2)
    postroll: Fraction = Fraction(1, 4)
    max_window: Fraction = Fraction(8)
    timeout: int = 45
    max_frames: int = 1200
    max_samples: int = 480000
    max_pixels: int = 4096 * 2160
    max_decoded_bytes: int = 2_000_000_000
    max_retained_bytes: int = 64_000_000
    max_log_bytes: int = 2_000_000
    max_windows: int = 16

    def __post_init__(self) -> None:
        for name in ("preroll", "postroll", "max_window"):
            object.__setattr__(self, name, exact(getattr(self, name)))
        validate_decode_limits(self.record())

    def record(self) -> dict[str, object]:
        return {name: rational(value) if isinstance(value, Fraction) else value
                for name in self.__dataclass_fields__ for value in (getattr(self, name),)}


@dataclass(frozen=True, slots=True)
class AudioDecodeLimits:
    preroll: Fraction = Fraction(1, 2)
    postroll: Fraction = Fraction(1, 4)
    max_window: Fraction = Fraction(300)
    timeout: int = 45
    max_blocks: int = 65536
    max_samples: int = 14400000
    max_decoded_bytes: int = 60000000
    max_retained_bytes: int = 60000000
    max_log_bytes: int = 16000000

    def __post_init__(self) -> None:
        for name in ("preroll", "postroll", "max_window"):
            object.__setattr__(self, name, exact(getattr(self, name)))
        validate_audio_decode_limits(self.record())

    def record(self) -> dict[str, object]:
        return {name: rational(value) if isinstance(value, Fraction) else value
                for name in self.__dataclass_fields__ for value in (getattr(self, name),)}


@dataclass(frozen=True, slots=True, kw_only=True)
class FloatAudioDecodeLimits:
    preroll: Fraction = Fraction(1, 2)
    postroll: Fraction = Fraction(1, 4)
    max_window: Fraction = Fraction(180)
    timeout: int = 45
    max_blocks: int = 16384
    max_samples: int = 9_000_000
    max_block_samples: int = 65536
    max_decoded_bytes: int = 72_000_000
    max_retained_bytes: int = 72_000_000
    max_log_bytes: int = 16_000_000
    max_metadata_bytes: int = 8_000_000
    io_chunk_bytes: int = 65536

    def __post_init__(self) -> None:
        for name in ("preroll", "postroll", "max_window"):
            _stream_fraction(getattr(self, name))
        validate_float_audio_limits(self.record())

    def record(self) -> dict[str, object]:
        return {name: rational(value) if isinstance(value, Fraction) else value
                for name in self.__dataclass_fields__ for value in (getattr(self, name),)}


def coalesce(windows: Sequence[Window], limits: DecodeLimits) -> tuple[Window, ...]:
    if not windows or len(windows) > limits.max_windows:
        raise ValueError("Window count limit")
    result: list[Window] = []
    for window in sorted(windows, key=lambda w: w.start):
        if window.end - window.start + limits.preroll + limits.postroll > limits.max_window:
            raise ValueError("Decode window limit")
        if (result and window.start <= result[-1].end + limits.preroll + limits.postroll
                and max(result[-1].end, window.end) - result[-1].start
                + limits.preroll + limits.postroll <= limits.max_window):
            result[-1] = Window(result[-1].start, max(result[-1].end, window.end))
        else:
            result.append(window)
    return tuple(result)


@dataclass(frozen=True, slots=True)
class VideoTag:
    n: int
    pts: int
    time_base: Fraction
    width: int
    height: int
    checksum: int

    @property
    def raw_start(self) -> Fraction:
        return self.pts * self.time_base

    @property
    def byte_count(self) -> int:
        return self.width * self.height * 3

    def record(self) -> dict[str, object]:
        return {"n": self.n, "pts": self.pts, "time_base": rational(self.time_base),
                "width": self.width, "height": self.height, "format": "bgr24",
                "checksum": f"{self.checksum:08X}"}


@dataclass(frozen=True, slots=True)
class AudioTag:
    n: int
    pts: int
    rate: int
    channels: int
    samples: int
    sample_format: str
    checksum: int
    stage: str

    @property
    def time_base(self) -> Fraction:
        # Both named observation pads explicitly use asettb=1/sample_rate.
        return Fraction(1, self.rate)

    @property
    def byte_count(self) -> int:
        return self.samples * self.channels * 2

    def record(self) -> dict[str, object]:
        return {"n": self.n, "pts": self.pts, "time_base": rational(self.time_base),
                "rate": self.rate, "channels": self.channels, "samples": self.samples,
                "format": self.sample_format, "checksum": f"{self.checksum:08X}",
                "stage": self.stage}


def _audio_payload_size(tag: AudioTag, contract: _AudioContract,
                        limits: DecodeLimits | AudioDecodeLimits | FloatAudioDecodeLimits) -> int:
    if contract is _FLOAT_AUDIO:
        if not isinstance(limits, FloatAudioDecodeLimits):
            raise ValueError("Float payload limits missing")
        for name in ("samples", "channels", "rate"):
            _stream_int(getattr(tag, name), 1)
        if (tag.channels not in (1, 2) or tag.samples > limits.max_block_samples
                or tag.samples > limits.max_samples or tag.sample_format != "flt" or tag.rate != 48000):
            raise ValueError("Float payload block/format limit")
    return tag.samples * tag.channels * contract.bytes_per_scalar


def _field(line: str, name: str, pattern: str) -> str:
    match = re.search(r"(?:^|\s)" + name + r":\s*(" + pattern + r")(?=\s|$)", line)
    if match is None:
        raise ValueError(f"Missing/malformed tagged metadata {name}: {line[:240]}")
    return match.group(1)


def parse_video(line: str, time_base: Fraction) -> VideoTag:
    if not line.startswith("[showinfo@cxv @") or _field(line, "fmt", r"\w+") != "bgr24":
        raise ValueError("Wrong video metadata tag/format")
    size = _field(line, "s", r"\d+x\d+").split("x")
    time_base = exact(time_base)
    if time_base <= 0:
        raise ValueError("Invalid observed video timebase")
    return VideoTag(int(_field(line, "n", r"\d+")), int(_field(line, "pts", r"-?\d+")),
                    time_base, int(size[0]), int(size[1]),
                    int(_field(line, "checksum", r"[0-9A-Fa-f]{8}"), 16))


def parse_audio(line: str) -> AudioTag:
    match = re.match(r"\[ashowinfo@(cxain|cxaout) @", line)
    if match is None:
        raise ValueError("Wrong audio metadata tag")
    return AudioTag(int(_field(line, "n", r"\d+")), int(_field(line, "pts", r"-?\d+")),
                    int(_field(line, "rate", r"\d+")), int(_field(line, "channels", r"\d+")),
                    int(_field(line, "nb_samples", r"\d+")), _field(line, "fmt", r"\w+"),
                    int(_field(line, "checksum", r"[0-9A-Fa-f]{8}"), 16), match.group(1))


@dataclass(frozen=True, slots=True)
class ObservedFrame:
    tag: VideoTag
    raw_end: Fraction
    bgr: bytes

    def __post_init__(self) -> None:
        object.__setattr__(self, "bgr", bytes(self.bgr))
        if self.raw_end <= self.raw_start or len(self.bgr) != self.tag.byte_count:
            raise ValueError("Invalid observed frame payload/coverage")

    @property
    def raw_start(self) -> Fraction:
        return self.tag.raw_start

    @property
    def sha256(self) -> str:
        return digest(self.bgr)

    def array_copy(self) -> Any:
        import numpy as np
        return np.frombuffer(self.bgr, dtype=np.uint8).reshape(
            self.tag.height, self.tag.width, 3).copy()


def _checked_video(tag: VideoTag, payload: bytes) -> dict[str, object]:
    if type(tag) is not VideoTag or type(payload) is not bytes:
        raise ValueError("Video requires actual immutable tag/pixels")
    for name in ("n", "width", "height", "checksum", "pts"):
        _stream_int(getattr(tag, name), None if name == "pts" else 1 if name in ("width", "height") else 0)
    if (_stream_fraction(tag.time_base) <= 0 or tag.checksum > 0xFFFFFFFF
            or len(payload) != tag.byte_count or zlib.adler32(payload, 0) != tag.checksum):
        raise ValueError("Video byte/checksum/timebase identity mismatch")
    return {**tag.record(), "bytes": len(payload), "sha256": digest(payload), "coverage_end": None}


@dataclass(frozen=True, slots=True, init=False)
class ProvisionalSelection:
    session_id: str
    delivery_index: int
    request_ordinals: tuple[int, ...]
    frame: ObservedFrame
    selected_row_json: str
    successor_row_json: str
    delivery_chain_sha256: str

    def __init__(self, session_id: str, delivery_index: int, request_ordinals: Sequence[int],
                 frame: ObservedFrame, selected_row: Mapping[str, object],
                 successor_row: Mapping[str, object], *, previous_chain_sha256: str) -> None:
        _stream_session(session_id)
        _stream_int(delivery_index)
        if (type(request_ordinals) not in (list, tuple) or not 0 < len(request_ordinals) <= 1200
                or type(frame) is not ObservedFrame or type(frame.raw_end) is not Fraction):
            raise ValueError("Provisional selection requires actual frame/ordinal group")
        ordinals = tuple(_stream_int(n) for n in request_ordinals)
        if any(b <= a for a, b in zip(ordinals, ordinals[1:])):
            raise ValueError("Provisional ordinal order")
        _stream_fraction(frame.raw_end)
        if not isinstance(selected_row, Mapping) or not isinstance(successor_row, Mapping):
            raise ValueError("Provisional rows require known mappings")
        _stream_guard(selected_row, 16384)
        _stream_guard(successor_row, 16384)
        if (type(previous_chain_sha256) is not str
                or re.fullmatch(r"[0-9a-f]{64}", previous_chain_sha256) is None):
            raise ValueError("Invalid preceding chain")
        selected: dict[str, Any] = dict(selected_row)
        successor: dict[str, Any] = dict(successor_row)
        for row in (selected, successor):
            _stream_keys(row, "n pts time_base width height format checksum bytes sha256 coverage_end")
            for key in ("n", "pts", "width", "height", "bytes"):
                _stream_int(row[key], None if key == "pts" else 1 if key in ("width", "height", "bytes") else 0)
            if any(type(row[key]) is not str for key in ("time_base", "format", "checksum", "sha256")):
                raise ValueError("Provisional row requires plain string identity fields")
        _stream_rational(selected["time_base"])
        _stream_rational(selected["coverage_end"])
        tb = _stream_rational(successor["time_base"])
        actual = _checked_video(frame.tag, frame.bgr)
        actual["coverage_end"] = rational(frame.raw_end)
        if (selected != actual or successor["n"] != frame.tag.n + 1
                or tb != frame.tag.time_base or successor["pts"] * tb != frame.raw_end
                or frame.raw_end <= frame.raw_start or successor["coverage_end"] is not None
                or any(successor[k] != actual[k] for k in ("width", "height", "format", "bytes"))
                or type(successor["checksum"]) is not str
                or re.fullmatch(r"[0-9A-F]{8}", successor["checksum"]) is None
                or type(successor["sha256"]) is not str
                or re.fullmatch(r"[0-9a-f]{64}", successor["sha256"]) is None):
            raise ValueError("Provisional frame/closed row/successor association mismatch")
        delivery = dict(index=delivery_index, request_ordinals=list(ordinals), frame_n=frame.tag.n,
                        successor_frame_n=frame.tag.n + 1)
        for name, value in (("session_id", session_id), ("delivery_index", delivery_index),
                ("request_ordinals", ordinals), ("frame", frame), ("selected_row_json", canonical(selected)),
                ("successor_row_json", canonical(successor)),
                ("delivery_chain_sha256", _stream_link(previous_chain_sha256, delivery, selected, successor))):
            object.__setattr__(self, name, value)

    def selected_record(self) -> dict[str, object]:
        return dict(json.loads(self.selected_row_json))

    def successor_record(self) -> dict[str, object]:
        return dict(json.loads(self.successor_row_json))

    def delivery_record(self) -> dict[str, object]:
        return dict(index=self.delivery_index, request_ordinals=list(self.request_ordinals),
            frame_n=self.frame.tag.n, successor_frame_n=self.frame.tag.n + 1,
            chain_sha256=self.delivery_chain_sha256)


@dataclass(frozen=True, slots=True)
class SampleOutcome:
    requested: Fraction
    status: str
    frame_n: int | None

    def record(self) -> dict[str, object]:
        return {"requested": rational(self.requested), "status": self.status,
                "frame_n": self.frame_n}


@dataclass(frozen=True, slots=True)
class VideoSelection:
    frames: tuple[ObservedFrame, ...]
    outcomes: tuple[SampleOutcome, ...]
    observations: tuple[str, ...]


class _FrameSelector:
    """The sole incremental half-open covering-frame selector, with no output list."""
    def __init__(self, origin: SourceTimeOrigin, requests: Sequence[Fraction],
                 limits: DecodeLimits | StreamDecodeLimits) -> None:
        self.video = next(s for s in origin.streams if s.kind == "video")
        self.requested = tuple(exact(t) for t in requests)
        self.raw = tuple(origin.to_raw(t) for t in self.requested)
        self.outcomes: list[SampleOutcome | None] = [SampleOutcome(t, "before_video_start", None)
            if self.video.start is not None and r < self.video.start else None
            for t, r in zip(self.requested, self.raw)]
        self.limits = limits
        self.observations: list[str] = []
        self.metadata_bytes = 0
        self.prior: tuple[VideoTag, bytes] | None = None
        self.total_bytes = 0

    def accept(self, tag: VideoTag, payload: bytes) -> tuple[ObservedFrame, tuple[int, ...]] | None:
        row = _checked_video(tag, payload)
        if (tag.n != len(self.observations) or tag.n >= self.limits.max_frames
                or tag.width * tag.height > self.limits.max_pixels
                or (tag.width, tag.height) != (self.video.width, self.video.height)):
            raise ValueError("Video count/dimension/byte/checksum disagreement or limit")
        self.total_bytes += len(payload)
        if self.total_bytes > self.limits.max_decoded_bytes:
            raise ValueError("Decoded byte limit")
        event = None
        if self.prior is not None:
            previous, pixels = self.prior
            if tag.time_base != previous.time_base or tag.raw_start <= previous.raw_start:
                raise ValueError("Nonmonotonic video PTS/timebase")
            record = json.loads(self.observations[-1])
            record["coverage_end"] = rational(tag.raw_start)
            closed = canonical(record)
            self.metadata_bytes += len(closed) - len(self.observations[-1])
            self.observations[-1] = closed
            selected = tuple(i for i, r in enumerate(self.raw) if self.outcomes[i] is None
                             and previous.raw_start <= r < tag.raw_start)
            if selected:
                event = ObservedFrame(previous, tag.raw_start, pixels), selected
                for i in selected:
                    self.outcomes[i] = SampleOutcome(self.requested[i], "covered", previous.n)
        encoded = canonical(row)
        self.observations.append(encoded)
        self.metadata_bytes += len(encoded)
        self.prior = tag, payload
        return event

    def complete(self) -> tuple[SampleOutcome, ...]:
        final = tuple(outcome if outcome is not None else SampleOutcome(self.requested[i],
            "unknown_terminal" if self.prior is not None and self.raw[i] >= self.prior[0].raw_start
            else "missing_predecessor", None) for i, outcome in enumerate(self.outcomes))
        self.prior = None
        return final


def select_frames(pairs: Iterable[tuple[VideoTag, bytes]], origin: SourceTimeOrigin,
                  requests: Sequence[Fraction], limits: DecodeLimits) -> VideoSelection:
    selector = _FrameSelector(origin, requests, limits)
    retained: list[ObservedFrame] = []
    retained_bytes = 0
    for tag, payload in pairs:
        event = selector.accept(tag, payload)
        if event is not None:
            frame, _ = event
            retained_bytes += len(frame.bgr)
            if retained_bytes > limits.max_retained_bytes:
                raise ValueError("Retained byte limit")
            retained.append(frame)
    return VideoSelection(tuple(retained), selector.complete(), tuple(selector.observations))


def stream_payloads(argv: Sequence[str], kind: str, time_base: Fraction,
                    limits: DecodeLimits | AudioDecodeLimits | FloatAudioDecodeLimits,
                    input_audio: list[AudioTag] | None = None, *,
                    _diagnostics: list[tuple[bytes, bool, int, str]] | None = None,
                    _audio_contract: _AudioContract = _S16_AUDIO,
                    _audio_clock: StreamClock | None = None,
                    _metadata_budget: list[int] | None = None,
                    _audio_phase: tuple[Fraction, int] | None = None,
                    _host: HostedObservationControl | None = None
                    ) -> Generator[tuple[Any, bytes], None, None]:
    """Bounded stderr reader + backpressure queue, raw reader, deadline watchdog.

    A metadata tag precedes each raw payload. The queue has 16 slots and the
    consumer reads exactly one payload at a time. Child/thread cleanup also runs
    when validation fails or the iterator is explicitly closed by the caller.
    """
    if kind not in ("video", "audio") or (kind == "video" and not isinstance(limits, DecodeLimits)):
        raise ValueError("Unknown pipe kind")
    if _audio_contract is _FLOAT_AUDIO and (kind != "audio" or type(limits) is not FloatAudioDecodeLimits
            or type(_audio_clock) is not StreamClock or _metadata_budget is None or _audio_phase is None):
        raise ValueError("Float payloads require bounded audio operation context")
    if _audio_contract is _FLOAT_AUDIO:
        assert _audio_phase is not None
        _stream_fraction(_audio_phase[0])
        _stream_int(_audio_phase[1], 1)
    max_blocks = limits.max_frames if isinstance(limits, DecodeLimits) else limits.max_blocks
    pending: queue.Queue[VideoTag | AudioTag | Exception | None] = queue.Queue(maxsize=16)
    stop = threading.Event()
    expired = threading.Event()
    stderr_tail = bytearray()
    log_bytes = 0
    stdout_bytes = 0
    stdout_hash = hashlib.sha256()
    lease = (_host._launch("video" if kind == "video" else
        "audio_f32" if _audio_contract is _FLOAT_AUDIO else "audio_s16", tuple(argv), limits.timeout * 10**9)
        if _host is not None else None)
    process = (subprocess.Popen(list(argv), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, bufsize=0) if _host is None else None)

    def read_raw(count: int) -> bytes:
        nonlocal stdout_bytes
        if _host is not None:
            assert lease is not None
            chunk = _host._read_stdout(lease, count)
        else:
            assert process is not None and process.stdout is not None
            chunk = process.stdout.read(count)
        stdout_bytes += len(chunk)
        stdout_hash.update(chunk)
        return chunk

    def put(item: VideoTag | AudioTag | Exception | None) -> None:
        while not stop.is_set():
            if _host is not None:
                _host._check()
            try:
                pending.put(item, timeout=0.05)
                return
            except queue.Full:
                continue

    def read_metadata() -> None:
        nonlocal log_bytes
        observed_tb: Fraction | None = None
        input_count = 0
        input_samples = 0
        previous_input: AudioTag | None = None
        output_count, output_samples = 0, 0
        previous_output: AudioTag | None = None
        try:
            def line_read() -> bytes:
                if _host is not None:
                    assert lease is not None
                    return _host._readline_stderr(lease, 16385)
                assert process is not None and process.stderr is not None
                return process.stderr.readline(16385)
            while line_bytes := line_read():
                log_bytes += len(line_bytes)
                stderr_tail.extend(line_bytes)
                if len(stderr_tail) > 65536:
                    del stderr_tail[:-65536]
                if len(line_bytes) > 16384 or log_bytes > limits.max_log_bytes:
                    raise ValueError("Metadata log byte limit")
                line = line_bytes.decode("utf-8", errors="replace").strip()
                # FFmpeg stats end in CR, not LF, and may prefix the next tag.
                line = line.rsplit("\r", 1)[-1].strip()
                if line.startswith("[showinfo@cxv @"):
                    if "config in time_base:" in line:
                        match = re.search(r"time_base:\s*(\d+/\d+)", line)
                        if match is None or observed_tb is not None:
                            raise ValueError("Missing/duplicate video timebase")
                        observed_tb = exact(match.group(1))
                    elif re.search(r"\bn:\s*", line):
                        if observed_tb is None:
                            raise ValueError("Frame precedes observed timebase")
                        put(parse_video(line, observed_tb))
                elif line.startswith("[ashowinfo@") and re.search(r"\bn:\s*", line):
                    if _audio_contract is _FLOAT_AUDIO:
                        for name in ("n", "pts", "rate", "channels", "nb_samples"):
                            token = _field(line, name, r"-?\d+")
                            if len(token) > 1235:
                                raise ValueError("Float metadata integer token limit")
                            _stream_int(int(token), None if name == "pts" else 0)
                    tag = parse_audio(line)
                    if _audio_contract is _FLOAT_AUDIO:
                        assert isinstance(limits, FloatAudioDecodeLimits)
                        assert _audio_clock is not None and _metadata_budget is not None
                        phase_charge = 0
                        expected = ((_audio_clock.sample_rate, _audio_clock.channels, _audio_clock.sample_format)
                                    if tag.stage == "cxain" else (48000, _audio_clock.channels, "flt"))
                        if ((tag.rate, tag.channels, tag.sample_format) != expected
                                or not 0 < tag.samples <= limits.max_block_samples):
                            raise ValueError("Float metadata source/format/block limit")
                        if tag.stage == "cxaout":
                            output_count += 1
                            output_samples += tag.samples
                            if (tag.n != output_count - 1 or output_count > max_blocks
                                    or output_samples > limits.max_samples or (previous_output is not None
                                        and tag.pts != previous_output.pts + previous_output.samples)):
                                raise ValueError("Float output sequence/sample/gap limit")
                            if output_count == 1:
                                assert _audio_phase is not None
                                grid = _stream_fraction(Fraction(tag.pts, tag.rate))
                                phase = _stream_fraction(grid - _audio_phase[0])
                                # The provisional header already owns its reserved widths.
                                # Charge any actual expansion once, before queue/raw work;
                                # a shorter observed value never releases earlier charges.
                                actual_width = sum(_stream_guard(rational(value), limits.max_metadata_bytes)
                                                   for value in (grid, phase))
                                phase_charge = max(0, actual_width - _audio_phase[1])
                            previous_output = tag
                        record = tag.record()
                        if tag.stage == "cxaout":
                            record.update(bytes=_audio_payload_size(tag, _audio_contract, limits), sha256="0" * 64)
                        charge = phase_charge + _stream_guard(record,
                            limits.max_metadata_bytes - _metadata_budget[0] - phase_charge) + 1
                        if _metadata_budget[0] + charge > limits.max_metadata_bytes:
                            raise ValueError("Float producer metadata byte limit")
                        _metadata_budget[0] += charge
                    if tag.stage == "cxain":
                        if (tag.n != input_count or input_count >= max_blocks
                                or tag.samples <= 0 or tag.rate <= 0 or tag.channels not in (1, 2)):
                            raise ValueError("Invalid input audio sequence/limit")
                        if previous_input is not None and (tag.rate != previous_input.rate
                                or tag.channels != previous_input.channels
                                or tag.sample_format != previous_input.sample_format
                                or tag.pts != previous_input.pts + previous_input.samples):
                            raise ValueError("Input audio gap/overlap/format change")
                        previous_input = tag
                        input_count += 1
                        input_samples += tag.samples
                        if input_samples > limits.max_samples:
                            raise ValueError("Input sample limit")
                        if input_audio is not None:
                            input_audio.append(tag)
                    else:
                        put(tag)
            put(None)
        except Exception as exc:
            put(exc)

    def deadline() -> None:
        expired.set()
        assert process is not None
        if process.poll() is None:
            process.kill()

    reader = None
    timer = None
    if _host is None:
        reader = threading.Thread(target=read_metadata, name="source-decode-metadata", daemon=True)
        timer = threading.Timer(limits.timeout, deadline)
        reader.start()
        timer.start()
    else:
        assert lease is not None
        _host._start_metadata(lease, read_metadata)
    total = 0
    try:
        while True:
            if _host is not None:
                _host._check()
            try:
                wait_seconds = float(limits.timeout)
                if _host is not None:
                    status = _host.ownership_snapshot()
                    wait_seconds = min(wait_seconds,
                        max(0, (status.child_deadline_ns or status.deadline_ns) - time.monotonic_ns()) / 1e9)
                tag = pending.get(timeout=wait_seconds)
            except queue.Empty as exc:
                raise ValueError("Decode metadata timeout") from exc
            if _host is not None:
                _host._check()
            if isinstance(tag, Exception):
                raise ValueError(f"Malformed decode metadata: {tag}") from tag
            if tag is None:
                break
            count = (_audio_payload_size(tag, _audio_contract, limits) if isinstance(tag, AudioTag)
                     else tag.byte_count)
            if ((kind == "video") != isinstance(tag, VideoTag) or count <= 0
                    or (isinstance(tag, VideoTag) and isinstance(limits, DecodeLimits)
                        and tag.width * tag.height > limits.max_pixels)
                    or (isinstance(tag, AudioTag) and (tag.samples > limits.max_samples
                        or tag.channels not in (1, 2) or tag.sample_format != _audio_contract.format))):
                raise ValueError("Unexpected/unbounded raw payload metadata")
            total += count
            if total > limits.max_decoded_bytes:
                raise ValueError("Raw decoded byte limit")
            if isinstance(limits, FloatAudioDecodeLimits) and isinstance(tag, AudioTag) and total > min(limits.max_retained_bytes,
                    4 * tag.channels * limits.max_samples):
                raise ValueError("Float retained byte limit before read")
            payload = bytearray()
            while len(payload) < count:
                remaining = count - len(payload)
                chunk = read_raw(min(remaining, limits.io_chunk_bytes)
                                 if isinstance(limits, FloatAudioDecodeLimits) else remaining)
                if not chunk:
                    raise ValueError("Raw payload truncated or decode timeout")
                payload.extend(chunk)
            if zlib.adler32(payload, 0) != tag.checksum:
                raise ValueError(f"Same-decode checksum mismatch at {kind} n={tag.n}: "
                                 f"expected={tag.checksum:08X} actual={zlib.adler32(payload, 0):08X}")
            if _audio_contract is _FLOAT_AUDIO:
                _finite_float_pcm(payload)
            yield tag, bytes(payload)
        if read_raw(1):
            raise ValueError("Raw bytes without associated metadata")
        if _host is not None:
            assert lease is not None
            result = _host._finish_child(lease, True).exit_code
        else:
            assert process is not None
            result = process.wait(timeout=limits.timeout)
        if result or expired.is_set():
            raise ValueError(f"Decode child failed/timeout: {result}")
    except Exception as exc:
        raise DecodeFailure(str(exc), argv, bytes(stderr_tail), log_bytes > 65536,
                            stdout_bytes, stdout_hash.hexdigest()) from exc
    finally:
        stop.set()
        if _host is None:
            assert process is not None and reader is not None and timer is not None
            assert process.stdout is not None and process.stderr is not None
            timer.cancel()
            if process.poll() is None:
                process.kill()
            process.wait(timeout=5)
            reader.join(timeout=5)
            timer.join(timeout=5)
            process.stdout.close()
            process.stderr.close()
        if _diagnostics is not None:
            _diagnostics.append((bytes(stderr_tail), log_bytes > 65536,
                                 stdout_bytes, stdout_hash.hexdigest()))
        if reader is not None and reader.is_alive():
            raise ValueError("Metadata reader did not terminate")


@contextmanager
def payload_stream(argv: Sequence[str], kind: str, time_base: Fraction,
                   limits: DecodeLimits | AudioDecodeLimits | FloatAudioDecodeLimits,
                   input_audio: list[AudioTag] | None = None, *,
                   _audio_contract: _AudioContract = _S16_AUDIO,
                   _audio_clock: StreamClock | None = None,
                   _metadata_budget: list[int] | None = None,
                   _audio_phase: tuple[Fraction, int] | None = None,
                   _host: HostedObservationControl | None = None
                   ) -> Generator[Generator[tuple[Any, bytes], None, None], None, None]:
    """Keep bounded diagnostics through consumer validation, including after EOF."""
    snapshots: list[tuple[bytes, bool, int, str]] = []
    stream = stream_payloads(argv, kind, time_base, limits, input_audio, _diagnostics=snapshots,
        _audio_contract=_audio_contract, _audio_clock=_audio_clock, _metadata_budget=_metadata_budget,
        _audio_phase=_audio_phase, _host=_host)
    try:
        yield stream
    except DecodeFailure:
        raise
    except Exception as exc:
        stream.close()
        tail, truncated, count, sha = snapshots[-1] if snapshots else (b"", False, 0, digest(b""))
        raise DecodeFailure(str(exc), argv, tail, truncated, count, sha) from exc
    finally:
        stream.close()


def decode_requirements() -> RenderRequirements:
    return RenderRequirements(NATIVE_RUNTIME, False,
        ("showinfo", "ashowinfo", "format", "aformat", "aresample", "asettb", "trim", "atrim"))


def _runtime_identity(runtime: RenderRuntime) -> str:
    if (runtime.runtime_id != NATIVE_RUNTIME or runtime.device_index is not None
            or runtime.ffmpeg_sha256 != FFMPEG_SHA256 or runtime.ffprobe_sha256 != FFPROBE_SHA256
            or runtime.ffmpeg.parent != runtime.ffprobe.parent):
        raise ValueError("Observation requires the explicitly pinned paired CPU runtime")
    return source_decoder_identity(_runtime_record(runtime))


def _runtime_record(runtime: RenderRuntime) -> dict[str, object]:
    return dict(runtime_id=runtime.runtime_id, ffmpeg=str(runtime.ffmpeg), ffprobe=str(runtime.ffprobe),
                ffmpeg_sha256=runtime.ffmpeg_sha256, ffprobe_sha256=runtime.ffprobe_sha256)


def probe_source(source: Path, runtime: RenderRuntime, *, timeout: int = 15,
                 _files: SourceFiles | None = None, _runner: RuntimeToolRunner | None = None) -> SourceTimeOrigin:
    """Explicit small metadata/stat probe; never reads source media to hash it."""
    if (_files is None) != (_runner is None):
        raise ValueError("Owned source probe requires paired file and tool services")
    decoder_id = _runtime_identity(runtime)
    file_options = {} if _files is None else {"_files": _files}
    _verify_pins(runtime, **file_options)
    if type(timeout) is not int or not 0 < timeout <= 45:
        raise ValueError("Invalid probe timeout")
    source = source.resolve() if _files is None else _files.resolve_path(source)
    before = source.stat() if _files is None else _files.stat_path(source)
    argv = [str(runtime.ffprobe), "-v", "error", "-show_entries",
        "format=start_time,duration:stream=index,codec_type,time_base,start_pts,duration_ts,"
        "width,height,sample_rate,channels,sample_fmt,color_range,color_space,color_transfer,color_primaries",
        "-of", "json", str(source)]
    result = (subprocess.run(argv, capture_output=True, timeout=timeout, check=False) if _runner is None else
              _runner(tuple(argv), timeout=timeout, max_stdout_bytes=1_000_000, max_stderr_bytes=1_000_000))
    if result.returncode or len(result.stdout) > 1_000_000:
        raise ValueError("Source metadata probe failed/limit")
    data = json.loads(result.stdout)
    metadata = data["format"]
    streams: list[StreamClock] = []
    colors: dict[str, str] = {}
    for stream in data["streams"]:
        if stream.get("codec_type") not in ("video", "audio"):
            continue
        tb = exact(stream["time_base"])
        start = int(stream["start_pts"]) * tb if "start_pts" in stream else None
        end = start + int(stream["duration_ts"]) * tb if start is not None and "duration_ts" in stream else None
        streams.append(StreamClock(int(stream["index"]), stream["codec_type"], tb, start, end,
            int(stream.get("width", 0)), int(stream.get("height", 0)),
            int(stream.get("sample_rate", 0)), int(stream.get("channels", 0)), stream.get("sample_fmt", "")))
        for key in ("color_range", "color_space", "color_transfer", "color_primaries"):
            if key in stream:
                colors[f"{stream['index']}:{key}"] = stream[key]
    if sum(s.kind == "video" for s in streams) != 1 or sum(s.kind == "audio" for s in streams) != 1:
        raise ValueError("Observation requires exactly one video and one mixed audio stream")
    fingerprint = source_fingerprint(source, **file_options)
    after = source.stat() if _files is None else _files.stat_path(source)
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Source changed during metadata probe")
    return SourceTimeOrigin(str(source), after.st_size, after.st_mtime_ns, fingerprint,
                            exact(metadata["start_time"]), exact(metadata["duration"]),
                            streams, decoder_id, colors)


@dataclass(frozen=True, slots=True)
class DecodedWindow:
    receipt: SourceDecodeReceipt
    frames: tuple[ObservedFrame, ...]
    pcm: bytes



@dataclass(frozen=True, slots=True, init=False)
class DecodedAudioWindow:
    """Immutable same-decode PCM whose aggregate and block identity is checked."""

    receipt: SourceAudioDecodeReceipt
    pcm: bytes

    def __init__(self, receipt: SourceAudioDecodeReceipt,
                 pcm: bytes | bytearray | memoryview) -> None:
        if type(receipt) is not SourceAudioDecodeReceipt or not isinstance(pcm, (bytes, bytearray, memoryview)):
            raise ValueError("Decoded audio requires an audio receipt and a byte-like PCM buffer")
        immutable = bytes(pcm)
        audio = json.loads(receipt.payload_json)["audio"][0]
        _validate_pcm(immutable, audio)
        object.__setattr__(self, "receipt", receipt)
        object.__setattr__(self, "pcm", immutable)


@dataclass(frozen=True, slots=True, init=False)
class DecodedFloatAudioWindow:
    """Detached observed packed-float bytes, with exact identity and finite-word checks."""
    receipt: SourceFloatAudioDecodeReceipt
    pcm: bytes

    def __init__(self, receipt: SourceFloatAudioDecodeReceipt,
                 pcm: bytes | bytearray | memoryview) -> None:
        if type(receipt) is not SourceFloatAudioDecodeReceipt or not isinstance(pcm, (bytes, bytearray, memoryview)):
            raise ValueError("Decoded float audio requires exact receipt and byte-like PCM")
        data, _ = _bounded_json(receipt.payload_json, 8_000_000)
        limits = data["limits"]
        validate_float_audio_limits(limits, window=True)
        audio = data["audio"][0]
        size = pcm.nbytes if isinstance(pcm, memoryview) else len(pcm)
        channels = _stream_int(audio["output"][0]["channels"], 1)
        expected = _stream_int(audio["bytes"], 1)
        if size != expected or size > min(limits["max_decoded_bytes"], limits["max_retained_bytes"],
                                        4 * channels * limits["max_samples"]):
            raise ValueError("Decoded float PCM byte capacity/count mismatch")
        immutable = bytes(pcm)
        _validate_pcm(immutable, audio)
        _finite_float_pcm(immutable)
        object.__setattr__(self, "receipt", receipt)
        object.__setattr__(self, "pcm", immutable)


def _finite_float_pcm(pcm: bytes | bytearray) -> None:
    if len(pcm) % 4 or any(word & 0x7f800000 == 0x7f800000
                           for (word,) in struct.iter_unpack("<I", pcm)):
        raise ValueError("Observed float PCM requires finite complete little-endian words")


def _validate_pcm(pcm: bytes, audio: Mapping[str, Any]) -> None:
    if len(pcm) != audio["bytes"] or digest(pcm) != audio["pcm_sha256"]:
        raise ValueError("Decoded audio aggregate PCM identity mismatch")
    view = memoryview(pcm)
    offset = 0
    try:
        for block in audio["output"]:
            segment = view[offset:offset + block["bytes"]]
            try:
                if (len(segment) != block["bytes"] or hashlib.sha256(segment).hexdigest() != block["sha256"]
                        or f"{zlib.adler32(segment, 0):08X}" != block["checksum"]):
                    raise ValueError("Decoded audio block PCM identity mismatch")
            finally:
                segment.release()
            offset += block["bytes"]
        if offset != len(pcm):
            raise ValueError("Decoded audio final offset mismatch")
    finally:
        view.release()


def _check_observation_source(runtime: RenderRuntime, origin: SourceTimeOrigin,
                              audio_policy: str, _audio_contract: _AudioContract = _S16_AUDIO, *,
                              _files: SourceFiles | None = None) -> None:
    if _runtime_identity(runtime) != origin.decoder_id:
        raise ValueError("Decoder identity mismatch")
    options = {} if _files is None else {"_files": _files}
    _verify_pins(runtime, **options)
    if source_fingerprint(Path(origin.path), **options) != origin.fingerprint:
        raise ValueError("Stale source identity")
    _, channels = _audio_contract.audio_format(audio_policy)
    streams = [s for s in origin.streams if s.kind == "audio"]
    if (len(streams) != 1 or streams[0].sample_rate not in (44100, 48000)
            or streams[0].channels not in (1, 2)):
        raise ValueError("Unsupported original audio rate/channels/selection")
    if _audio_contract is _FLOAT_AUDIO and (streams[0].sample_format not in ("flt", "fltp")
                                           or channels != streams[0].channels):
        raise ValueError("Unsupported float source format/channel preservation")


def _decoder_interval(origin: SourceTimeOrigin, window: Window,
                      limits: DecodeLimits | AudioDecodeLimits | FloatAudioDecodeLimits | StreamDecodeLimits) -> tuple[Fraction, Fraction]:
    origin.to_raw(window.start)
    origin.to_raw(window.end)
    start = max(Fraction(0), window.start - limits.preroll)
    end = min(origin.duration, window.end + limits.postroll)
    if end - start > limits.max_window:
        raise ValueError("Effective decode window limit")
    return start, end


def _decoder_common(runtime: RenderRuntime, origin: SourceTimeOrigin,
                    start: Fraction, end: Fraction) -> list[str]:
    return decoder_input_argv(str(runtime.ffmpeg), origin, start, end)


@contextmanager
def _audio_window(runtime: RenderRuntime, origin: SourceTimeOrigin, window: Window,
                  declared: Sequence[Window], *, audio_policy: str,
                  limits: DecodeLimits | AudioDecodeLimits | FloatAudioDecodeLimits | StreamDecodeLimits, retained_bytes: int = 0,
                  _owner: "_StreamOwner | None" = None,
                  _audio_contract: _AudioContract = _S16_AUDIO,
                  _metadata_budget: list[int] | None = None,
                  _audio_phase: tuple[Fraction, int] | None = None,
                  _host: HostedObservationControl | None = None
                  ) -> Generator[tuple[bytes, dict[str, object], list[str]], None, None]:
    """One continuous audio child, shared by both observers.

    Yield inside payload_stream so receipt/association errors after EOF retain
    the same bounded child diagnostics and cleanup as streaming failures.
    """
    if _owner is not None and (_audio_contract is _FLOAT_AUDIO or _host is not None):
        raise ValueError("Streamed owner does not support float observation")
    audio = next(s for s in origin.streams if s.kind == "audio")
    rate, channels = _audio_contract.audio_format(audio_policy)
    start, end = _decoder_interval(origin, window, limits)
    builder = float_audio_decode_argv if _audio_contract is _FLOAT_AUDIO else audio_decode_argv
    audio_argv = builder(str(runtime.ffmpeg), origin, start, end, audio_policy)
    max_blocks = limits.max_blocks if isinstance(limits, (AudioDecodeLimits, FloatAudioDecodeLimits)) else limits.max_frames
    input_audio: list[AudioTag] = []
    audio_records: list[dict[str, object]] = []
    capacity = limits.max_samples * channels * 2 if _owner is not None else 0
    if _owner is not None:
        _owner.payload(capacity)
    pcm = bytearray(capacity)
    offset = 0
    samples = 0
    previous: AudioTag | None = None
    if _owner is not None:
        stream_context = _owner.pairs(audio_argv, "audio", audio.time_base, input_audio)
    else:
        assert isinstance(limits, (DecodeLimits, AudioDecodeLimits, FloatAudioDecodeLimits))
        stream_context = payload_stream(audio_argv, "audio", audio.time_base, limits, input_audio,
            _audio_contract=_audio_contract, _audio_clock=audio, _metadata_budget=_metadata_budget,
            _audio_phase=_audio_phase, _host=_host)
    with stream_context as audio_pairs:
        for n, (tag, payload) in enumerate(audio_pairs):
            if (not isinstance(tag, AudioTag) or tag.n != n or n >= max_blocks
                    or tag.rate != rate or tag.channels != channels or tag.samples <= 0
                    or (previous is not None and tag.pts != previous.pts + previous.samples)):
                raise ValueError("Audio sequence/format/gap/overlap disagreement")
            samples += tag.samples
            if (samples > limits.max_samples or (_owner is None
                    and isinstance(limits, (DecodeLimits, AudioDecodeLimits, FloatAudioDecodeLimits))
                    and len(pcm) + len(payload) + retained_bytes > limits.max_retained_bytes)
                    or (_owner is not None and offset + len(payload) > capacity)):
                raise ValueError("Audio sample/retained byte limit")
            if _owner is None:
                if _host is not None:
                    _host._note_payload(True, False)
                pcm.extend(payload)
            else:
                pcm[offset:offset + len(payload)] = payload
            offset += len(payload)
            audio_records.append({**tag.record(), "bytes": len(payload), "sha256": digest(payload)})
            previous = tag
            del payload
        if not audio_records or not input_audio:
            raise ValueError("Missing observed audio")
        first_pts = int(str(audio_records[0]["pts"]))
        audio_start = Fraction(first_pts, rate)
        assert previous is not None
        audio_end = Fraction(previous.pts + previous.samples, rate)
        validate_audio_conversion(sum(t.samples for t in input_audio), audio.sample_rate,
            input_audio[0].pts * input_audio[0].time_base, samples, rate, audio_start)
        coverage = [{"window": [rational(w.start), rational(w.end)],
                     "status": "covered" if audio_start <= origin.to_raw(w.start)
                     and audio_end >= origin.to_raw(w.end) else "unavailable"} for w in declared]
        if _owner is None:
            immutable = bytes(pcm)
        else:
            _owner.check()
            _owner.payload(capacity + offset)
            view = memoryview(pcm)
            segment = view[:offset]
            immutable = bytes(segment)
            segment.release()
            view.release()
            del pcm
            _owner.payload(offset)
        record: dict[str, object] = {
            "policy": audio_policy, "input": [t.record() for t in input_audio],
            "output": audio_records, "samples": samples, "bytes": len(immutable),
            "pcm_sha256": digest(immutable), "sample_grid_origin": rational(audio_start),
            "source_phase": rational(audio_start - origin.origin), "coverage": coverage}
        if _owner is not None:
            _validate_pcm(immutable, record)
        yield immutable, record, audio_argv


def _observe_float_audio_window(runtime: RenderRuntime, origin: SourceTimeOrigin, window: Window, *,
                               audio_policy: str, limits: FloatAudioDecodeLimits,
                               _host: HostedObservationControl | None = None,
                               _runtime_files: SourceFiles | None = None) -> DecodedFloatAudioWindow:
    """Observe one bounded float window; no source finiteness or prepared-grid claim."""
    if type(runtime) is not RenderRuntime or type(window) is not Window or type(limits) is not FloatAudioDecodeLimits:
        raise ValueError("Float observation requires actual runtime, Window and float limits")
    for name in ("preroll", "postroll", "max_window"):
        _stream_fraction(getattr(limits, name))
    host = limits.record()
    validate_float_audio_limits(host)
    _float_origin(origin, limits.max_metadata_bytes)
    _stream_fraction(window.start)
    _stream_fraction(window.end)
    if not 0 <= window.start < window.end:
        raise ValueError("Invalid float window")
    start, end = _decoder_interval(origin, window, limits)
    runtime_record = _runtime_record(runtime)
    argv = float_audio_decode_argv(str(runtime.ffmpeg), origin, start, end, audio_policy)
    # Reserve known envelope/argv/limits/coverage and maximum-width totals/hashes.
    skeleton: dict[str, Any] = dict(windows=[[rational(window.start), rational(window.end)]],
        limits=dict(host, decoder_start=rational(start), decoder_end=rational(end)), argv=[argv],
        audio=[dict(policy=audio_policy, input=[], output=[], samples=limits.max_samples,
            bytes=limits.max_decoded_bytes, pcm_sha256="0" * 64,
            sample_grid_origin=rational(origin.to_raw(start)), source_phase=rational(start),
            coverage=[dict(window=[rational(window.start), rational(window.end)], status="unavailable")])])
    reserved = _stream_guard(dict(version=1, kind="audio_only_float", scope="window",
        origin_id="0" * 64, origin_decoder_id=origin.decoder_id, operation_id="0" * 64,
        policy="audio-only;pcm-f32le;48k;swr-fltp;copyts;no-healing;v1", runtime=runtime_record,
        payload=skeleton, sha256="0" * 64), limits.max_metadata_bytes)
    phase_width = sum(_stream_guard(skeleton["audio"][0][name], limits.max_metadata_bytes)
                      for name in ("sample_grid_origin", "source_phase"))
    # Bind complete runtime schema before HOST checks, without upgrading a synthetic identity.
    if source_decoder_identity(runtime_record) != origin.decoder_id:
        raise ValueError("Float current runtime/origin identity mismatch")
    options = {} if _runtime_files is None else {"_files": _runtime_files}
    _check_observation_source(runtime, origin, audio_policy, _FLOAT_AUDIO, **options)
    with _audio_window(runtime, origin, window, (window,), audio_policy=audio_policy, limits=limits,
                       _audio_contract=_FLOAT_AUDIO, _metadata_budget=[reserved],
                       _audio_phase=(origin.origin, phase_width), _host=_host) as (pcm, record, actual):
        skeleton["audio"], skeleton["argv"] = [record], [actual]
        receipt = SourceFloatAudioDecodeReceipt(origin, skeleton, runtime=runtime_record, limits=host)
        result = DecodedFloatAudioWindow(receipt, pcm)
        if _runtime_files is not None:
            _verify_pins(runtime, _files=_runtime_files)
        if source_fingerprint(Path(origin.path), **options) != origin.fingerprint:
            raise ValueError("Source changed during float observation")
        return result


@contextmanager
def _hosted_scope(host: HostedObservationControl | None, kind: ObservationKind,
                  limits: FloatAudioDecodeLimits | StreamDecodeLimits) -> Generator[None, None, None]:
    if host is None:
        yield
        return
    success = False
    try:
        host._enter(kind, limits.max_log_bytes, limits.max_decoded_bytes,
            limits.window_timeout * 10**9 if isinstance(limits, StreamDecodeLimits) else None,
            limits.cleanup_grace * 10**9 if isinstance(limits, StreamDecodeLimits) else 5_000_000_000)
        yield
        host._check()
        success = True
    except HostedObservationFailure:
        raise
    except Exception:
        if host.ownership_snapshot().failure is None:
            host._fail("validation", "validate")
        raise
    finally:
        if not success and kind == "materialized_f32":
            host._note_payload(False, False)
        host._leave(success)
        status = host.ownership_snapshot()
        if success and status.failure is not None:
            raise HostedObservationFailure(status.failure, host._job.snapshot(), status.diagnostics)


def observe_float_audio_window(runtime: RenderRuntime, origin: SourceTimeOrigin, window: Window, *,
                               audio_policy: str, limits: FloatAudioDecodeLimits,
                               _host: HostedObservationControl | None = None,
                               _runtime_files: SourceFiles | None = None) -> DecodedFloatAudioWindow:
    """Observe finite float PCM, transferring it only after hosted retirement."""
    if _runtime_files is not None and _host is None:
        raise ValueError("Owned source files require paired hosted control")
    if type(limits) is not FloatAudioDecodeLimits:
        raise ValueError("Float observation requires actual float limits")
    with _hosted_scope(_host, "materialized_f32", limits):
        result = _observe_float_audio_window(runtime, origin, window, audio_policy=audio_policy,
            limits=limits, _host=_host, **({} if _runtime_files is None else {"_runtime_files": _runtime_files}))
        if _host is not None:
            _host._note_payload(False, False, len(result.pcm))
    return result


def observe_audio_window(runtime: RenderRuntime, origin: SourceTimeOrigin, window: Window, *,
                         audio_policy: str, limits: AudioDecodeLimits = AudioDecodeLimits()
                         ) -> DecodedAudioWindow:
    """Prepare one bounded audio interval without launching a video decoder."""
    if type(window) is not Window or type(limits) is not AudioDecodeLimits:
        raise ValueError("Audio observation requires one Window and AudioDecodeLimits")
    start, end = _decoder_interval(origin, window, limits)
    _check_observation_source(runtime, origin, audio_policy)
    with _audio_window(runtime, origin, window, (window,), audio_policy=audio_policy, limits=limits) as (
            pcm, audio_record, argv):
        payload: dict[str, object] = {
            "windows": [[rational(window.start), rational(window.end)]],
            "limits": {**limits.record(), "decoder_start": rational(start), "decoder_end": rational(end)},
            "audio": [audio_record], "argv": [argv]}
        receipt = SourceAudioDecodeReceipt(origin.identity, origin.decoder_id, AUDIO_DECODE_POLICY, payload)
        receipt.validate_origin(origin)
        result = DecodedAudioWindow(receipt, pcm)
        if source_fingerprint(Path(origin.path)) != origin.fingerprint:
            raise ValueError("Source changed during observation")
        return result


def observe_windows(runtime: RenderRuntime, origin: SourceTimeOrigin,
                    windows: Sequence[Window], requests: Sequence[Fraction], *,
                    audio_policy: str, limits: DecodeLimits = DecodeLimits()) -> tuple[DecodedWindow, ...]:
    """One video + one audio child per coalesced bounded window; no retries.

    Audio payload includes explicitly reported decoder margins. Its sample grid
    has an absolute origin/phase; it is not a prepared Task5 bus or healed track.
    """
    _check_observation_source(runtime, origin, audio_policy)
    video = next(s for s in origin.streams if s.kind == "video")
    requested = tuple(exact(t) for t in requests)
    if len(requested) > limits.max_frames or any(not any(w.start <= t < w.end for w in windows) for t in requested):
        raise ValueError("Sample requests must belong to declared windows and fit limits")
    results: list[DecodedWindow] = []
    for window in coalesce(windows, limits):
        start, end = _decoder_interval(origin, window, limits)
        video_argv = video_decode_argv(str(runtime.ffmpeg), origin, start, end)
        with payload_stream(video_argv, "video", video.time_base, limits) as pairs:
            selection = select_frames(pairs, origin,
                tuple(t for t in requested if window.start <= t < window.end), limits)
        declared = [w for w in windows if window.start <= w.start and w.end <= window.end]
        with _audio_window(runtime, origin, window, declared, audio_policy=audio_policy, limits=limits,
                           retained_bytes=sum(len(f.bgr) for f in selection.frames)) as (
                pcm, audio_record, audio_argv):
            receipt_payload: dict[str, object] = {"windows": [[rational(w.start), rational(w.end)] for w in windows
                                if window.start <= w.start and w.end <= window.end],
                   "limits": {**limits.record(), "decoder_start": rational(start),
                              "decoder_end": rational(end)},
                   "video": [json.loads(record) for record in selection.observations],
                   "audio": [audio_record],
                   "outcomes": [outcome.record() for outcome in selection.outcomes],
                   "argv": [video_argv, audio_argv]}
            receipt = SourceDecodeReceipt(origin.identity, origin.decoder_id, DECODE_POLICY, receipt_payload)
            receipt.validate_origin(origin)
            results.append(DecodedWindow(receipt, selection.frames, bytes(pcm)))
            if sum(len(r.pcm) + sum(len(f.bgr) for f in r.frames) for r in results) > limits.max_retained_bytes:
                raise ValueError("Batch retained byte limit")
    if source_fingerprint(Path(origin.path)) != origin.fingerprint:
        raise ValueError("Source changed during observation")
    return tuple(results)


class _StreamCleanupFailure(ValueError):
    def __init__(self, survivors: Sequence[str]) -> None:
        self.cleanup_required = True
        self.survivors = tuple(survivors)
        super().__init__("Stream cleanup incomplete; still owned: " + ", ".join(survivors))


class _StreamChild:
    """One child's handles and bounded metadata, always retained until cleanup."""
    def __init__(self, process: Any, kind: str, started_ns: int, deadline_ns: int) -> None:
        self.hosted_lease: ChildLease | None = None
        self.process, self.kind = process, kind
        self.started_ns, self.deadline_ns = started_ns, deadline_ns
        self.pending: queue.Queue[Any] = queue.Queue(maxsize=16)
        self.stop = threading.Event()
        self.reader: threading.Thread | None = None
        self.log_bytes = 0
        self.stderr_tail = bytearray()
        self.stdout_bytes = 0
        self.stdout_hash = hashlib.sha256()
        self.exit_code: int | None = None
        self.finished_ns: int | None = None


class _StreamOwner:
    """Lifetime and absolute deadlines live here, outside every suspended iterator.

    The watchdog never closes a generator. Cleanup retains every surviving
    process/thread/handle so the HOST can account for a failed attempt.
    """
    def __init__(self, origin: SourceTimeOrigin, window: Window, requests: tuple[Fraction, ...],
                 limits: StreamDecodeLimits, audio_policy: str, runtime: RenderRuntime, entry_ns: int,
                 _host: HostedObservationControl | None = None, *,
                 _runtime_files: SourceFiles | None = None) -> None:
        self._host = _host
        self._runtime_files = _runtime_files
        self.origin, self.window, self.requests, self.limits = origin, window, requests, limits
        self.audio_policy = audio_policy
        self.session_id = str(uuid.uuid4())
        self.runtime = runtime
        self.entry_ns = entry_ns
        self.deadline_ns = self.entry_ns + limits.window_timeout * 10**9
        self.state = "OPEN"
        self.lock = threading.RLock()
        self.cleanup_lock = threading.Lock()
        self.cancelled = threading.Event()
        self.active: _StreamChild | None = None
        self.timers: list[Any] = []
        self.cleanup_deadline_ns: int | None = None
        self.survivors: tuple[str, ...] = ()
        self.failure: str | None = None
        self.lease: ProvisionalSelection | None = None
        video = next(s for s in origin.streams if s.kind == "video")
        self.frame_bytes = video.width * video.height * 3
        self.audio_capacity = limits.max_samples * audio_format(audio_policy)[1] * 2
        self.reservation = max(3 * self.frame_bytes, 4 * self.audio_capacity)
        if video.width * video.height > limits.max_pixels or self.reservation > limits.max_live_payload_bytes:
            raise ValueError("Stream payload reservation/pixel limit before child allocation")
        self.live_payload_bytes = 0
        self.peak_payload_bytes = 0
        self.children: list[dict[str, object]] = []
        self.logs: dict[str, int] = {}
        self._arm(self.deadline_ns)

    def _arm(self, deadline_ns: int) -> None:
        if self._host is not None:
            return
        for timer in self.timers[1:]:
            timer.cancel()
        timer = threading.Timer(max(0, deadline_ns - time.monotonic_ns()) / 10**9, self._expire)
        timer.daemon = False
        self.timers.append(timer)
        timer.start()

    def _expire(self) -> None:
        with self.lock:
            if self.state in ("FINALIZED", "CLOSED"):
                return
            deadline = min(self.deadline_ns, self.active.deadline_ns) if self.active else self.deadline_ns
            if time.monotonic_ns() < deadline:
                return
            self.failure, self.state = "Stream absolute deadline expired", "FAILED"
            self.cancelled.set()
        try:
            self.cleanup()
        except _StreamCleanupFailure:
            # Explicit survivors remain attached to this owner; no success escapes.
            pass

    def check(self) -> None:
        if self._host is not None:
            self._host._check()
        with self.lock:
            if self.state not in ("OPEN", "VIDEO_COMPLETE"):
                raise ValueError(self.failure or "Terminal stream cannot resume")
            now = time.monotonic_ns()
            deadline = min(self.deadline_ns, self.active.deadline_ns) if self.active else self.deadline_ns
            if self.failure or self.cancelled.is_set() or now >= deadline:
                self.state = "FAILED"
                self.failure = self.failure or "Stream absolute deadline expired"
                self.cancelled.set()
                raise ValueError(self.failure)

    def payload(self, amount: int) -> None:
        if amount < 0 or amount > self.reservation:
            raise ValueError("Stream live-payload ledger limit")
        self.live_payload_bytes = amount
        self.peak_payload_bytes = max(self.peak_payload_bytes, amount)
        if self._host is not None:
            self._host._note_payload(bool(amount), self.lease is not None)

    def _remaining(self) -> float:
        self.check()
        deadline = min(self.deadline_ns, self.active.deadline_ns) if self.active else self.deadline_ns
        return max(0, deadline - time.monotonic_ns()) / 10**9

    def _put(self, child: _StreamChild, value: Any) -> None:
        while not child.stop.is_set() and not self.cancelled.is_set():
            try:
                child.pending.put(value, timeout=min(0.05, self._remaining()))
                return
            except queue.Full:
                continue

    def _metadata(self, child: _StreamChild, time_base: Fraction, input_audio: list[AudioTag]) -> None:
        observed_tb: Fraction | None = None
        previous_input: AudioTag | None = None
        input_samples = 0
        try:
            while not child.stop.is_set() and not self.cancelled.is_set():
                self.check()
                raw = (self._host._readline_stderr(child.hosted_lease, 16385)
                    if self._host is not None and child.hosted_lease is not None
                    else child.process.stderr.readline(16385))
                self.check()
                if not raw:
                    break
                child.log_bytes += len(raw)
                child.stderr_tail.extend(raw)
                del child.stderr_tail[:-65536]
                if len(raw) > 16384 or child.log_bytes > self.limits.max_log_bytes:
                    raise ValueError("Stream metadata line/log limit")
                line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                if child.kind == "video" and line.startswith("[showinfo@cxv @"):
                    match = re.search(r"config in time_base:\s*(\d+/\d+)", line)
                    if match:
                        observed_tb = _stream_rational(match.group(1))
                        if observed_tb <= 0:
                            raise ValueError("Invalid observed video timebase")
                    elif re.search(r"\bn:\s*", line):
                        if observed_tb is None:
                            raise ValueError("Frame precedes observed timebase")
                        self._put(child, parse_video(line, observed_tb))
                elif child.kind == "audio" and line.startswith("[ashowinfo@") and re.search(r"\bn:\s*", line):
                    tag = parse_audio(line)
                    for name in ("n", "pts", "rate", "channels", "samples", "checksum"):
                        _stream_int(getattr(tag, name), None if name == "pts" else 0)
                    if tag.stage == "cxain":
                        if (tag.n != len(input_audio) or tag.n >= self.limits.max_frames or tag.samples <= 0
                                or tag.rate <= 0 or tag.channels not in (1, 2)
                                or (previous_input is not None and (tag.pts != previous_input.pts + previous_input.samples
                                    or any(getattr(tag, k) != getattr(previous_input, k)
                                           for k in ("rate", "channels", "sample_format"))))):
                            raise ValueError("Invalid input audio sequence/limit")
                        input_samples += tag.samples
                        if input_samples > self.limits.max_samples:
                            raise ValueError("Input sample limit")
                        input_audio.append(tag)
                        previous_input = tag
                    else:
                        self._put(child, tag)
            self._put(child, None)
        except Exception as exc:
            if not child.stop.is_set() and not self.cancelled.is_set():
                self._put(child, exc)

    def _read(self, child: _StreamChild, size: int) -> bytes:
        self.check()
        base = self.live_payload_bytes if child.kind == "audio" else (
            self.frame_bytes if child.stdout_bytes else 0)
        if size > (self.frame_bytes if child.kind == "video" else self.audio_capacity):
            raise ValueError("Stream raw payload capacity")
        self.payload(base + size)
        storage = bytearray(size)
        view = memoryview(storage)
        offset = 0
        try:
            while offset < size:
                self.check()
                part = view[offset:min(size, offset + self.limits.io_chunk_bytes)]
                try:
                    count = (self._host._readinto_stdout(child.hosted_lease, part)
                        if self._host is not None and child.hosted_lease is not None
                        else child.process.stdout.readinto(part))
                finally:
                    part.release()
                self.check()
                if type(count) is not int or not 0 < count <= min(size - offset, self.limits.io_chunk_bytes):
                    raise ValueError("Stream raw truncated/unsupported short-read protocol")
                offset += count
            self.payload(base + size * 2)
            immutable = bytes(view)
            child.stdout_hash.update(view)
            child.stdout_bytes += size
        finally:
            view.release()
            del storage
        self.payload(base + size)
        return immutable

    @contextmanager
    def pairs(self, argv: Sequence[str], kind: str, time_base: Fraction,
              input_audio: list[AudioTag] | None = None
              ) -> Generator[Generator[tuple[Any, bytes], None, None], None, None]:
        self.check()
        file_options: _SourceFileOptions = {} if self._runtime_files is None else {"_files": self._runtime_files}
        _check_observation_source(self.runtime, self.origin, self.audio_policy, **file_options)
        self.check()
        if self.active is not None:
            raise ValueError("Stream already owns an active child")
        began = time.monotonic_ns()
        # Do not hold the cancellation lock across a HOST launch. If this call
        # returns after expiry, attach the handle before rejecting it for cleanup.
        hosted_lease = (self._host._launch("video" if kind == "video" else "audio_s16",
            tuple(argv), self.limits.timeout * 10**9) if self._host is not None else None)
        process = (subprocess.Popen(list(argv), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE, bufsize=0) if self._host is None else None)
        child = _StreamChild(process, kind, began, min(began + self.limits.timeout * 10**9, self.deadline_ns))
        child.hosted_lease = hosted_lease
        with self.lock:
            self.active = child
            self.check()
        if self._host is not None:
            assert hosted_lease is not None
            self._host._start_metadata(hosted_lease,
                lambda: self._metadata(child, time_base, input_audio if input_audio is not None else []))
        else:
            with self.lock:
                child.reader = threading.Thread(target=self._metadata,
                    args=(child, time_base, input_audio if input_audio is not None else []),
                    name="source-stream-metadata", daemon=False)
                child.reader.start()
                self._arm(child.deadline_ns)
        iterator = self._raw_pairs(child)
        try:
            yield iterator
            if child.finished_ns is not None:
                self.check()
        except Exception as exc:
            self.failure = self.failure or str(exc)
            self.state = "FAILED"
            self.cancelled.set()
            if isinstance(exc, _StreamCleanupFailure):
                raise
            raise DecodeFailure(str(exc), argv, bytes(child.stderr_tail), child.log_bytes > 65536,
                                child.stdout_bytes, child.stdout_hash.hexdigest()) from exc
        finally:
            # Consumer and watchdog coordinate handles, never concurrent generator.close.
            iterator.close()
            self.cleanup()

    def _raw_pairs(self, child: _StreamChild) -> Generator[tuple[Any, bytes], None, None]:
        n = total = 0
        while True:
            self.check()
            try:
                tag = child.pending.get(timeout=self._remaining())
            except queue.Empty as exc:
                raise ValueError("Stream metadata deadline") from exc
            self.check()
            if isinstance(tag, Exception):
                raise ValueError(f"Malformed stream metadata: {tag}") from tag
            if tag is None:
                break
            if n >= self.limits.max_frames:
                raise ValueError("Stream raw block/frame count limit")
            if child.kind == "video":
                if (type(tag) is not VideoTag or tag.n != n or tag.width * tag.height > self.limits.max_pixels
                        or tag.byte_count != self.frame_bytes):
                    raise ValueError("Stream video metadata dimensions/count")
            elif (type(tag) is not AudioTag or tag.n != n or tag.samples <= 0
                    or tag.samples > self.limits.max_samples or tag.channels not in (1, 2)
                    or tag.sample_format != "s16"):
                raise ValueError("Stream output audio metadata/count")
            total += tag.byte_count
            if total > self.limits.max_decoded_bytes:
                raise ValueError("Stream decoded child byte limit")
            immutable = self._read(child, tag.byte_count)
            if zlib.adler32(immutable, 0) != tag.checksum:
                raise ValueError("Stream actual payload checksum mismatch")
            yield tag, immutable
            # These aliases MUST disappear before another buffer can be allocated.
            del immutable, tag
            self.payload(self.audio_capacity if child.kind == "audio" else self.frame_bytes)
            n += 1
        self.check()
        base = self.audio_capacity if child.kind == "audio" else self.frame_bytes if child.stdout_bytes else 0
        self.payload(base + 1)
        extra = bytearray(1)
        count = (self._host._readinto_stdout(child.hosted_lease, memoryview(extra))
            if self._host is not None and child.hosted_lease is not None else child.process.stdout.readinto(extra))
        self.check()
        if type(count) is not int or count != 0:
            raise ValueError("Stream raw bytes without associated metadata")
        del extra
        self.payload(base)
        if self._host is not None:
            assert child.hosted_lease is not None
            outcome = self._host._finish_child(child.hosted_lease, True)
            child.exit_code = outcome.exit_code
        else:
            child.exit_code = child.process.wait(timeout=self._remaining())
        self.check()
        child.finished_ns = time.monotonic_ns()
        if _stream_int(child.exit_code, None) != 0:
            raise ValueError("Stream late decoder failure")

    def cleanup(self, *, final: bool = False) -> None:
        if self._host is not None:
            if self.cleanup_deadline_ns is None:
                self.cleanup_deadline_ns = time.monotonic_ns() + self.limits.cleanup_grace * 10**9
            self._host._retain_cleanup_deadline(self.cleanup_deadline_ns)
            if self.failure and self._host.ownership_snapshot().failure is None:
                self._host._fail("validation", "validate")
            self._host._cleanup_child()
            status = self._host.ownership_snapshot()
            if not status.window_resources_zero:
                self.survivors = tuple(str(r.resource_id) for r in status.resources
                    if r.phase not in ("closed", "joined", "never_created"))
                raise _StreamCleanupFailure(self.survivors)
            child = self.active
            if child is not None:
                self.logs[child.kind] = child.log_bytes
                if child.finished_ns is not None and child.exit_code == 0 and not self.failure:
                    self.children.append(dict(kind=child.kind, exit_code=0,
                        started_ns=child.started_ns - self.entry_ns, finished_ns=child.finished_ns - self.entry_ns,
                        cleanup_complete=True))
                self.active = None
            return
        # One absolute budget persists across waits, close calls and watchdog races.
        with self.lock:
            if self.cleanup_deadline_ns is None:
                self.cleanup_deadline_ns = time.monotonic_ns() + self.limits.cleanup_grace * 10**9
            cleanup_deadline = self.cleanup_deadline_ns
        def remaining() -> float:
            return max(0, cleanup_deadline - time.monotonic_ns()) / 10**9
        if not self.cleanup_lock.acquire(timeout=remaining()):
            raise _StreamCleanupFailure(self.survivors or ("cleanup-worker",))
        try:
            survivors: list[str] = []
            child = self.active
            if child is not None:
                child.stop.set()
                try:
                    if child.process.poll() is None:
                        child.process.kill()
                    child.process.wait(timeout=remaining())
                except Exception:
                    survivors.append(f"pid:{child.process.pid}")
                # Closing owned pipes wakes the reader after kill; arbitrary OS close isn't interruptible.
                for name in ("stdout", "stderr"):
                    handle = getattr(child.process, name)
                    try:
                        handle.close()
                    except Exception:
                        survivors.append(f"pid:{child.process.pid}:{name}")
                    else:
                        if not handle.closed:
                            survivors.append(f"pid:{child.process.pid}:{name}")
                if child.reader is not None:
                    joined = True
                    try:
                        if child.reader is not threading.current_thread():
                            child.reader.join(timeout=remaining())
                    except Exception:
                        joined = False
                    if not joined or child.reader.is_alive():
                        survivors.append(f"thread:{child.reader.name}:{child.reader.ident}")
            timers = self.timers if final or self.cancelled.is_set() else self.timers[1:]
            for timer in timers:
                joined = True
                try:
                    timer.cancel()
                    if timer is not threading.current_thread():
                        timer.join(timeout=remaining())
                except Exception:
                    joined = False
                if not joined or timer.is_alive():
                    survivors.append(f"timer:{id(timer)}:{getattr(timer, 'ident', None)}")
            self.survivors = tuple(survivors)
            if survivors:
                self.state = "FAILED"
                self.failure = self.failure or "Stream cleanup incomplete"
                self.cancelled.set()
                raise _StreamCleanupFailure(survivors)
            if child is not None:
                self.logs[child.kind] = child.log_bytes
                if child.finished_ns is not None and child.exit_code == 0 and not self.failure:
                    self.children.append(dict(kind=child.kind, exit_code=0,
                        started_ns=child.started_ns - self.entry_ns, finished_ns=child.finished_ns - self.entry_ns,
                        cleanup_complete=True))
                self.active = None
        finally:
            self.cleanup_lock.release()


class ObservedWindowStream:
    """Single-consumer borrowed pull interface; only the owner holds live resources."""
    def __init__(self, owner: _StreamOwner, runtime: RenderRuntime) -> None:
        self._owner, self._runtime = owner, runtime
        self._consumer_lock = threading.Lock()
        self._selector = _FrameSelector(owner.origin, owner.requests, owner.limits)
        start, end = _decoder_interval(owner.origin, owner.window, owner.limits)
        self._payload: dict[str, Any] = dict(window=[rational(owner.window.start), rational(owner.window.end)],
            requests=[rational(t) for t in owner.requests], operation_limits=owner.limits.operation_record(),
            delivery_limits=owner.limits.delivery_record(), decoder_start=rational(start), decoder_end=rational(end),
            audio={"policy": owner.audio_policy})
        self._chain = digest(canonical(_stream_header(owner.origin, owner.session_id, self._payload)))
        self._video_argv = video_decode_argv(str(runtime.ffmpeg), owner.origin, start, end)
        self._video_context: Any = None
        self._pairs_iterator: Any = None
        self._deliveries: list[dict[str, object]] = []
        self._unique_bytes = self._emission_bytes = self._acked_requests = 0
        self._ordinals = 0
        self._outcomes: tuple[SampleOutcome, ...] = ()

    @property
    def session_id(self) -> str: return self._owner.session_id
    @property
    def origin_id(self) -> str: return self._owner.origin.identity
    @property
    def decoder_id(self) -> str: return self._owner.origin.decoder_id
    @property
    def policy(self) -> str: return STREAM_DECODE_POLICY
    @property
    def limits(self) -> StreamDecodeLimits: return self._owner.limits
    @property
    def window(self) -> Window: return self._owner.window
    @property
    def requests(self) -> tuple[Fraction, ...]: return self._owner.requests
    @property
    def deadline_ns(self) -> int: return self._owner.deadline_ns
    @property
    def state(self) -> str: return self._owner.state
    @property
    def cleanup_required(self) -> bool:
        return bool(self._owner.active is not None or self._owner.survivors or self._owner.lease is not None
                    or self._selector.prior is not None or any(t.is_alive() for t in self._owner.timers))

    def _fail(self, exc: Exception) -> None:
        self._owner.failure = self._owner.failure or str(exc)
        self._owner.state = "FAILED"
        self._owner.cancelled.set()
        self._release(error=exc)

    def next_selection(self) -> ProvisionalSelection | None:
        if not self._consumer_lock.acquire(blocking=False):
            raise ValueError("Stream requires one consumer operation")
        try:
            return self._next_selection()
        finally:
            self._consumer_lock.release()

    def _next_selection(self) -> ProvisionalSelection | None:
        owner = self._owner
        owner.check()
        if owner.lease is not None:
            raise ValueError("A selection is still borrowed")
        if self.state == "VIDEO_COMPLETE":
            raise ValueError("Video completion already returned; finish or close")
        try:
            if self._video_context is None:
                self._video_context = owner.pairs(self._video_argv, "video", self._selector.video.time_base)
                self._pairs_iterator = self._video_context.__enter__()
            while True:
                try:
                    tag, payload = next(self._pairs_iterator)
                except StopIteration:
                    self._outcomes = self._selector.complete()
                    self._release_video()
                    owner.payload(0)
                    owner.check()
                    if any(o.status != "covered" for o in self._outcomes):
                        raise ValueError("Stream requests are unavailable")
                    with owner.lock:
                        owner.check()
                        owner.state = "VIDEO_COMPLETE"
                    return None
                _stream_int(tag.pts, None)
                _validate_video_origin_bounds(tag.record(), self._selector.video, owner.origin,
                    exact(self._payload["decoder_start"]), exact(self._payload["decoder_end"]))
                event = self._selector.accept(tag, payload)
                del tag, payload
                # Keep metadata bounded during production, before final construction.
                if self._selector.metadata_bytes > self.limits.max_finalization_bytes:
                    raise ValueError("Stream video metadata byte limit")
                if event is None:
                    owner.payload(self._selector.video.width * self._selector.video.height * 3)
                    continue
                frame, ordinals = event
                del event
                if (ordinals != tuple(range(self._ordinals, self._ordinals + len(ordinals)))
                        or len(self._deliveries) >= self.limits.max_deliveries):
                    raise ValueError("Stream unavailable request/delivery order")
                self._unique_bytes += len(frame.bgr)
                self._emission_bytes += len(frame.bgr) * len(ordinals)
                if (self._unique_bytes > self.limits.max_selected_unique_bytes
                        or self._emission_bytes > self.limits.max_request_emission_bytes):
                    raise ValueError("Stream selected/emission accounting limit")
                selection = ProvisionalSelection(self.session_id, len(self._deliveries), ordinals, frame,
                    json.loads(self._selector.observations[frame.tag.n]),
                    json.loads(self._selector.observations[frame.tag.n + 1]), previous_chain_sha256=self._chain)
                self._chain = selection.delivery_chain_sha256
                self._deliveries.append(selection.delivery_record())
                self._ordinals += len(ordinals)
                with owner.lock:
                    owner.check()
                    owner.lease = selection
                    owner.payload(2 * owner.frame_bytes)
                return selection
        except Exception as exc:
            self._fail(exc)
            raise

    def acknowledge(self, delivery_index: int) -> None:
        with self._owner.lock:
            self._owner.check()
            lease = self._owner.lease
            if type(delivery_index) is not int or lease is None or delivery_index != lease.delivery_index:
                raise ValueError("Acknowledgment does not match the outstanding selection")
            self._acked_requests += len(lease.request_ordinals)
            self._owner.lease = None
            del lease
            # Only successor remains; no suspended selected-frame alias in this API.
            self._owner.payload(self._owner.frame_bytes)

    def _release_video(self, error: Exception | None = None) -> None:
        self._selector.prior = None
        if self._video_context is not None:
            context, self._video_context = self._video_context, None
            self._pairs_iterator = None
            if error is not None:
                context.__exit__(type(error), error, error.__traceback__)
            else:
                context.__exit__(None, None, None)

    def _release(self, error: Exception | None = None) -> None:
        self._owner.lease = None
        self._selector.prior = None
        try:
            self._release_video(error=error)
        finally:
            self._owner.payload(0)
            self._owner.cleanup()

    def finish(self) -> SourceStreamFinalization:
        if not self._consumer_lock.acquire(blocking=False):
            raise ValueError("Stream requires one consumer operation")
        try:
            return self._finish()
        finally:
            self._consumer_lock.release()

    def _finish(self) -> SourceStreamFinalization:
        owner = self._owner
        owner.check()
        if self.state != "VIDEO_COMPLETE" or owner.lease is not None or self._acked_requests != len(self.requests):
            raise ValueError("Finish requires video EOF and all acknowledgments")
        try:
            owner.payload(0)
            with _audio_window(self._runtime, owner.origin, self.window, (self.window,),
                    audio_policy=owner.audio_policy, limits=self.limits, _owner=owner) as (pcm, audio, argv):
                _validate_pcm(pcm, audio)
            del pcm
            owner.payload(0)
            file_options: _SourceFileOptions = {} if owner._runtime_files is None else {"_files": owner._runtime_files}
            _check_observation_source(self._runtime, owner.origin, owner.audio_policy, **file_options)
            owner.check()
            owner.cleanup(final=True)
            self._payload.update(video=[json.loads(r) for r in self._selector.observations], audio=audio,
                outcomes=[o.record() for o in self._outcomes], argv=[self._video_argv, argv], deliveries=self._deliveries)
            self._payload["accounting"] = dict(decoded_video_frames=len(self._selector.observations),
                decoded_video_bytes=self._selector.total_bytes, decoded_audio_output_bytes=audio["bytes"],
                selected_unique_frames=len(self._deliveries), selected_unique_bytes=self._unique_bytes,
                offered_deliveries=len(self._deliveries), offered_payload_bytes=self._unique_bytes,
                acknowledged_deliveries=len(self._deliveries), acknowledged_requests=self._acked_requests,
                acknowledged_request_bytes=self._emission_bytes, reserved_payload_capacity_bytes=owner.reservation,
                peak_live_payload_bytes_accounted=owner.peak_payload_bytes,
                metadata_bytes=len(canonical(self._payload).encode("utf-8")),
                video_log_bytes=owner.logs["video"], audio_log_bytes=owner.logs["audio"])
            self._payload["completion"] = dict(children=owner.children,
                source_fingerprint_after=owner.origin.fingerprint, pins_verified_before_children=True,
                all_requests_acknowledged=True, cleanup_complete=True, elapsed_ns=time.monotonic_ns() - owner.entry_ns)
            final = SourceStreamFinalization(owner.origin, self.session_id, self._payload, limits=self.limits)
            with owner.lock:
                owner.check()
                owner.state = "FINALIZED"
            return final
        except Exception as exc:
            self._fail(exc)
            raise

    def close(self) -> None:
        if not self._consumer_lock.acquire(blocking=False):
            self._owner.failure = "Close during active consumer operation; cleanup initiated"
            self._owner.state = "FAILED"
            self._owner.cancelled.set()
            self._owner.cleanup()
            raise ValueError(self._owner.failure)
        try:
            was_finalized = self.state == "FINALIZED"
            self._owner.cancelled.set()
            self._release()
            if not was_finalized and self.state != "FAILED":
                self._owner.state = "CLOSED"
        finally:
            self._consumer_lock.release()


@contextmanager
def _stream_observed_window(runtime: RenderRuntime, origin: SourceTimeOrigin,
    window: Window, requests: Sequence[Fraction], *, audio_policy: str,
    limits: StreamDecodeLimits = StreamDecodeLimits(),
    _host: HostedObservationControl | None = None,
    _runtime_files: SourceFiles | None = None) -> Generator[ObservedWindowStream, None, None]:
    entry_ns = time.monotonic_ns()
    if type(limits) is not StreamDecodeLimits or type(window) is not Window:
        raise ValueError("Stream requires one actual Window and StreamDecodeLimits")
    _stream_origin(origin, limits.max_finalization_bytes)
    _stream_fraction(window.start)
    _stream_fraction(window.end)
    _decoder_interval(origin, window, limits)
    if type(requests) not in (list, tuple) or not 0 < len(requests) <= min(limits.max_requests, limits.max_frames):
        raise ValueError("Stream request count/type limit")
    requested = tuple(_stream_fraction(t) for t in requests)
    if any(not window.start <= t < window.end for t in requested) or any(
            b < a for a, b in zip(requested, requested[1:])):
        raise ValueError("Stream requests must be ordered and within the window")
    if type(audio_policy) is not str:
        raise ValueError("Stream audio policy must be a plain string")
    audio_format(audio_policy)
    _stream_guard(str(runtime.ffmpeg), limits.max_finalization_bytes)
    owner = _StreamOwner(origin, window, requested, limits, audio_policy, runtime, entry_ns, _host,
        **({} if _runtime_files is None else {"_runtime_files": _runtime_files}))
    stream: ObservedWindowStream | None = None
    try:
        file_options: _SourceFileOptions = {} if _runtime_files is None else {"_files": _runtime_files}
        _check_observation_source(runtime, origin, audio_policy, **file_options)
        owner.check()
        stream = ObservedWindowStream(owner, runtime)
        yield stream
    finally:
        if stream is not None:
            stream.close()
        else:
            owner.cancelled.set()
            owner.cleanup()


@contextmanager
def stream_observed_window(runtime: RenderRuntime, origin: SourceTimeOrigin,
    window: Window, requests: Sequence[Fraction], *, audio_policy: str,
    limits: StreamDecodeLimits = StreamDecodeLimits(),
    _host: HostedObservationControl | None = None,
    _runtime_files: SourceFiles | None = None) -> Generator[ObservedWindowStream, None, None]:
    """Shared streamed selector with optional exclusive hosted resource ownership."""
    if _runtime_files is not None and _host is None:
        raise ValueError("Owned source files require paired hosted control")
    if type(limits) is not StreamDecodeLimits:
        raise ValueError("Stream requires actual StreamDecodeLimits")
    with _hosted_scope(_host, "streamed_s16", limits):
        with _stream_observed_window(runtime, origin, window, requests, audio_policy=audio_policy,
                limits=limits, _host=_host,
                **({} if _runtime_files is None else {"_runtime_files": _runtime_files})) as stream:
            yield stream
