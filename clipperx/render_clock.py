"""Pure exact common-source clocks and detached JSON window receipts.

No host operations at construction/import. The source fingerprint explicitly
identifies path/size/mtime, not a hash of media bytes. Seconds are rational.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import re
from typing import Any
import uuid

DECODE_POLICY = "bgr24-showinfo;pcm-s16-ashowinfo;copyts;no-healing;v1"
FLOAT_AUDIO_DECODE_POLICY = "audio-only;pcm-f32le;48k;swr-fltp;copyts;no-healing;v1"


@dataclass(frozen=True, slots=True)
class _AudioContract:
    format: str
    codec: str
    muxer: str
    bytes_per_scalar: int
    policies: tuple[tuple[str, int, int], ...]

    def audio_format(self, policy: str) -> tuple[int, int]:
        if type(policy) is str:
            for name, rate, channels in self.policies:
                if name == policy:
                    return rate, channels
        raise ValueError("Unsupported audio preparation policy")


_S16_AUDIO = _AudioContract("s16", "pcm_s16le", "s16le", 2, (
    ("analysis16k-mono", 16000, 1), ("render48k-mono", 48000, 1),
    ("render48k-stereo", 48000, 2)))
_FLOAT_AUDIO = _AudioContract("flt", "pcm_f32le", "f32le", 4, (
    ("render48k-f32-mono-v1", 48000, 1), ("render48k-f32-stereo-v1", 48000, 2)))


def exact(value: object) -> Fraction:
    """Only exact integer/rational/decimal text; no binary floats or booleans."""
    if isinstance(value, Fraction):
        return Fraction(value.numerator, value.denominator)
    if type(value) not in (int, str):
        raise ValueError("Time requires finite exact rational metadata")
    assert isinstance(value, (int, str))
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("Time requires finite exact rational metadata") from exc


def rational(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def digest(value: str | bytes) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


@dataclass(frozen=True, slots=True)
class StreamClock:
    index: int
    kind: str
    time_base: Fraction
    start: Fraction | None
    end: Fraction | None
    width: int = 0
    height: int = 0
    sample_rate: int = 0
    channels: int = 0
    sample_format: str = ""

    def __post_init__(self) -> None:
        if type(self.kind) is not str or type(self.sample_format) is not str:
            raise ValueError("Stream identity scalars must be immutable strings")
        for name in ("index", "width", "height", "sample_rate", "channels"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"Invalid stream {name}")
        for name in ("time_base", "start", "end"):
            value = getattr(self, name)
            if value is not None or name == "time_base":
                object.__setattr__(self, name, exact(value))
        if self.time_base <= 0 or self.kind not in ("video", "audio"):
            raise ValueError("Invalid stream kind/timebase")
        if self.start is not None and self.end is not None and self.end <= self.start:
            raise ValueError("Invalid stream bounds")
        if self.kind == "video" and (self.width <= 0 or self.height <= 0):
            raise ValueError("Unknown video dimensions")
        if self.kind == "audio" and (self.sample_rate <= 0 or self.channels <= 0
                                     or not self.sample_format):
            raise ValueError("Unknown audio format")

    def record(self) -> dict[str, object]:
        return {name: rational(value) if isinstance(value, Fraction) else value
                for name in self.__dataclass_fields__ if (value := getattr(self, name)) is not None}


@dataclass(frozen=True, slots=True, init=False)
class SourceTimeOrigin:
    path: str
    size: int
    mtime_ns: int
    fingerprint: str
    origin: Fraction
    duration: Fraction
    streams: tuple[StreamClock, ...]
    decoder_id: str
    color: tuple[tuple[str, str], ...]
    fingerprint_method: str = "sha256(path|size|mtime_ns)[:16]"
    policy: str = "raw=format.start_time+source;v1"

    def __init__(self, path: str, size: int, mtime_ns: int, fingerprint: str,
                 origin: Fraction, duration: Fraction, streams: Sequence[StreamClock],
                 decoder_id: str, color: Mapping[str, str]) -> None:
        if (any(type(v) is not str for v in (path, fingerprint, decoder_id))
                or not path or type(size) is not int or size < 0 or type(mtime_ns) is not int
                or not re.fullmatch(r"[0-9a-f]{16}", fingerprint) or not decoder_id):
            raise ValueError("Invalid source identity")
        origin, duration = exact(origin), exact(duration)
        if (duration <= 0 or not streams or any(type(s) is not StreamClock for s in streams)
                or len({s.index for s in streams}) != len(streams)):
            raise ValueError("Invalid duration/stream identities")
        if not isinstance(color, Mapping) or any(
                type(k) is not str or type(v) is not str for k, v in color.items()):
            raise ValueError("Invalid color metadata")
        for name, value in (("path", path), ("size", size), ("mtime_ns", mtime_ns),
                            ("fingerprint", fingerprint), ("origin", origin),
                            ("duration", duration), ("streams", tuple(streams)),
                            ("decoder_id", decoder_id), ("color", tuple(sorted(color.items()))),
                            ("fingerprint_method", "sha256(path|size|mtime_ns)[:16]"),
                            ("policy", "raw=format.start_time+source;v1")):
            object.__setattr__(self, name, value)

    def to_raw(self, source: Fraction) -> Fraction:
        source = exact(source)
        if not 0 <= source <= self.duration:
            raise ValueError("Source time outside known container bounds")
        return self.origin + source

    def to_source(self, raw: Fraction) -> Fraction:
        source = exact(raw) - self.origin
        self.to_raw(source)
        return source

    def record(self) -> dict[str, object]:
        return {"path": self.path, "size": self.size, "mtime_ns": self.mtime_ns,
                "fingerprint": self.fingerprint, "fingerprint_method": self.fingerprint_method,
                "origin": rational(self.origin), "duration": rational(self.duration),
                "streams": [s.record() for s in self.streams], "decoder_id": self.decoder_id,
                "color": dict(self.color), "policy": self.policy}

    @property
    def identity(self) -> str:
        return digest(canonical(self.record()))


def decoder_seconds(value: Fraction) -> str:
    """Existing microsecond CLI serialization, only for decoder margins."""
    micros = value.numerator * 1_000_000 // value.denominator
    return f"{micros // 1_000_000}.{micros % 1_000_000:06d}"


def audio_format(policy: str) -> tuple[int, int]:
    return _S16_AUDIO.audio_format(policy)


def decoder_input_argv(executable: str, origin: SourceTimeOrigin,
                       start: Fraction, end: Fraction) -> list[str]:
    """Pure canonical prefix shared with the unchanged combined video operation."""
    if type(executable) is not str or not executable:
        raise ValueError("Decoder command requires an executable string")
    start, end = exact(start), exact(end)
    origin.to_raw(start)
    origin.to_raw(end)
    if start >= end:
        raise ValueError("Invalid decoder command interval")
    seek = ["-ss", decoder_seconds(start)] if start > 0 else []
    return [executable, "-hide_banner", "-nostdin", "-nostats", "-loglevel", "info",
            "-copyts", *seek, "-t", decoder_seconds(end - start),
            "-threads", "2", "-i", origin.path]


def audio_decode_argv(executable: str, origin: SourceTimeOrigin,
                      start: Fraction, end: Fraction, policy: str) -> list[str]:
    """The sole safe audio command, pure and exact rather than a general interpreter.

    Receipt association compares the entire argv against this same producer
    command. Executable qualification remains the pinned runtime's responsibility.
    """
    return _audio_decode_argv(executable, origin, start, end, policy, _S16_AUDIO)


def _audio_decode_argv(executable: str, origin: SourceTimeOrigin,
                       start: Fraction, end: Fraction, policy: str,
                       contract: _AudioContract) -> list[str]:
    rate, channels = contract.audio_format(policy)
    streams = [s for s in origin.streams if s.kind == "audio"]
    if (len(streams) != 1 or streams[0].sample_rate not in (44100, 48000)
            or streams[0].channels not in (1, 2)):
        raise ValueError("Unsupported original audio rate/channels/selection")
    audio = streams[0]
    if contract is _FLOAT_AUDIO and (audio.sample_format not in ("flt", "fltp")
                                    or audio.channels != channels):
        raise ValueError("Float audio requires original float format and unchanged channels")
    common = decoder_input_argv(executable, origin, start, end)
    layout = "mono" if channels == 1 else "stereo"
    options = ":osf=flt:tsf=fltp" if contract is _FLOAT_AUDIO else ""
    graph = (f"atrim=end={decoder_seconds(origin.to_raw(end))},asettb=1/{audio.sample_rate},ashowinfo@cxain,"
             f"aresample={rate}:async=0{options},aformat=sample_fmts={contract.format}:channel_layouts={layout},"
             f"asettb=1/{rate},ashowinfo@cxaout")
    return [*common, "-map", f"0:{audio.index}", "-vn", "-sn", "-dn", "-af", graph,
            "-c:a", contract.codec, "-threads", "1", "-f", contract.muxer, "pipe:1"]


def float_audio_decode_argv(executable: str, origin: SourceTimeOrigin,
                            start: Fraction, end: Fraction, audio_policy: str) -> list[str]:
    _float_origin(origin, 8_000_000)
    start, end = _stream_fraction(start), _stream_fraction(end)
    if start >= end or end - start > 180 or decoder_seconds(end - start) == "0.000000":
        raise ValueError("Invalid float effective decoder interval/duration")
    _stream_guard(executable, 8_000_000)
    return _audio_decode_argv(executable, origin, start, end, audio_policy, _FLOAT_AUDIO)


def video_decode_argv(executable: str, origin: SourceTimeOrigin,
                      start: Fraction, end: Fraction) -> list[str]:
    streams = [s for s in origin.streams if s.kind == "video"]
    if len(streams) != 1:
        raise ValueError("Video decoder requires one selected video stream")
    video = streams[0]
    return [*decoder_input_argv(executable, origin, start, end),
            "-map", f"0:{video.index}", "-an", "-sn", "-dn",
            "-vf", f"trim=end={decoder_seconds(origin.to_raw(end))},format=bgr24,showinfo@cxv",
            "-fps_mode", "passthrough", "-enc_time_base", "filter", "-threads", "1",
            "-c:v", "rawvideo", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]


@dataclass(frozen=True, slots=True, init=False)
class SourceDecodeReceipt:
    """Immutable canonical evidence envelope; decoding supplies bounded observations.

    ``payload`` is serialized immediately, so no caller container is retained.
    Deserialization requires the currently probed origin and explicit policy.
    Hashes detect corruption, not adversarial authenticity or legacy qualification.
    """

    origin_id: str
    decoder_id: str
    policy: str
    payload_json: str

    def __init__(self, origin_id: str, decoder_id: str, policy: str,
                 payload: Mapping[str, object]) -> None:
        if (any(type(v) is not str for v in (origin_id, decoder_id, policy))
                or not re.fullmatch(r"[0-9a-f]{64}", origin_id) or not decoder_id or not policy):
            raise ValueError("Invalid receipt identity")
        if not isinstance(payload, Mapping) or set(payload) != {"windows", "limits", "video", "audio", "outcomes", "argv"}:
            raise ValueError("Invalid receipt fields")
        try:
            encoded = canonical(payload)
        except TypeError as exc:
            raise ValueError("Receipt contains unsupported JSON values") from exc
        data = json.loads(encoded)
        if not isinstance(data["limits"], dict) or any(
                not isinstance(data[k], list) for k in ("windows", "video", "audio", "outcomes", "argv")):
            raise ValueError("Invalid receipt containers")
        for window in data["windows"]:
            if type(window) is not list or len(window) != 2 or not 0 <= exact(window[0]) < exact(window[1]):
                raise ValueError("Invalid receipt window")
        if not data["windows"] or len(data["audio"]) != 1 or len(data["argv"]) != 2:
            raise ValueError("Missing required window/audio/argv evidence")
        if any(type(argv) is not list or not argv or any(type(arg) is not str for arg in argv)
               for argv in data["argv"]):
            raise ValueError("Invalid receipt argv evidence")
        _validate_observations(data)
        for name, value in (("origin_id", origin_id), ("decoder_id", decoder_id),
                            ("policy", policy), ("payload_json", encoded)):
            object.__setattr__(self, name, value)

    def to_json(self) -> str:
        body = {"version": 1, "scope": "window", "origin_id": self.origin_id,
                "decoder_id": self.decoder_id, "policy": self.policy,
                "payload": json.loads(self.payload_json)}
        return canonical({**body, "sha256": digest(canonical(body))})

    @classmethod
    def from_json(cls, text: str, *, origin: SourceTimeOrigin,
                  policy: str) -> "SourceDecodeReceipt":
        def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in items:
                if key in result:
                    raise ValueError("Duplicate receipt field")
                result[key] = value
            return result

        data = json.loads(text, object_pairs_hook=pairs)
        if type(data) is not dict or set(data) != {"version", "scope", "origin_id", "decoder_id", "policy", "payload", "sha256"}:
            raise ValueError("Invalid receipt schema")
        if (type(data["version"]) is not int or data["version"] != 1 or data["scope"] != "window"
                or data["origin_id"] != origin.identity or data["decoder_id"] != origin.decoder_id
                or data["policy"] != policy):
            raise ValueError("Stale or mismatched receipt identity/policy")
        claimed = data.pop("sha256")
        if digest(canonical(data)) != claimed:
            raise ValueError("Receipt hash mismatch")
        receipt = cls(data["origin_id"], data["decoder_id"], policy, data["payload"])
        receipt.validate_origin(origin)
        return receipt

    def validate_origin(self, origin: SourceTimeOrigin) -> None:
        """Shared producer/loader association with the currently selected source."""
        if self.origin_id != origin.identity or self.decoder_id != origin.decoder_id:
            raise ValueError("Receipt origin/decoder identity mismatch")
        data = json.loads(self.payload_json)
        _validate_observation_origin(data, origin)


def _validate_observation_origin(data: dict[str, Any], origin: SourceTimeOrigin) -> None:
    """Numeric association shared by explicit aggregate and streamed wrappers."""
    windows = [(exact(w[0]), exact(w[1])) for w in data["windows"]]
    for start, end in windows:
        origin.to_raw(start)
        origin.to_raw(end)
    limits = data["limits"]
    start, end = exact(limits["decoder_start"]), exact(limits["decoder_end"])
    if start != max(
        Fraction(0), min(w[0] for w in windows) - exact(limits["preroll"])
    ) or end != min(origin.duration, max(w[1] for w in windows) + exact(limits["postroll"])):
        raise ValueError("Decoder margins disagree with declared windows/origin")
    video_streams = [s for s in origin.streams if s.kind == "video"]
    audio_streams = [s for s in origin.streams if s.kind == "audio"]
    if len(video_streams) != 1 or len(audio_streams) != 1:
        raise ValueError("Receipt needs one selected video/audio stream")
    video_stream, audio_stream = video_streams[0], audio_streams[0]
    for frame in data["video"]:
        _validate_video_origin_bounds(frame, video_stream, origin, start, end)
    for outcome in data["outcomes"]:
        if outcome["status"] == "covered":
            frame = data["video"][outcome["frame_n"]]
            raw = origin.to_raw(exact(outcome["requested"]))
            if not frame["pts"] * exact(frame["time_base"]) <= raw < exact(frame["coverage_end"]):
                raise ValueError("Receipt outcome outside observed coverage")
    _validate_audio_origin(data, origin, audio_stream, start, end, windows)


def _validate_video_origin_bounds(
    frame: Mapping[str, Any],
    video: StreamClock,
    origin: SourceTimeOrigin,
    start: Fraction,
    end: Fraction,
) -> None:
    if (frame["width"], frame["height"]) != (video.width, video.height):
        raise ValueError("Observed dimensions disagree with selected source")
    raw = frame["pts"] * exact(frame["time_base"])
    if not origin.to_raw(start) - Fraction(1, 1000000) <= raw < origin.to_raw(end):
        raise ValueError("Observed video outside decoder window")


def _nearest_input_sample(seconds: Fraction, rate: int) -> int:
    """Exact nearest input-grid coordinate, with half ties away from zero."""
    numerator = seconds.numerator * rate
    magnitude = (2 * abs(numerator) + seconds.denominator) // (2 * seconds.denominator)
    return magnitude if numerator >= 0 else -magnitude


def _validate_audio_origin(data: dict[str, Any], origin: SourceTimeOrigin,
                           audio_stream: StreamClock, start: Fraction, end: Fraction,
                           windows: list[tuple[Fraction, Fraction]]) -> None:
    audio = data["audio"][0]
    first_input, first_output = audio["input"][0], audio["output"][0]
    if (first_input["rate"], first_input["channels"], first_input["format"]) != (
            audio_stream.sample_rate, audio_stream.channels, audio_stream.sample_format):
        raise ValueError("Input audio metadata disagrees with selected source")
    audio_start = exact(audio["sample_grid_origin"])
    audio_end = audio_start + Fraction(audio["samples"], first_output["rate"])
    input_start = Fraction(first_input["pts"], first_input["rate"])
    input_end = input_start + Fraction(sum(b["samples"] for b in audio["input"]), first_input["rate"])
    # Accurate seek quantizes the serialized relative CLI time plus the common
    # origin, not an independently serialized absolute time. At start zero the
    # same grid bound retains the source-start constraint without a CLI seek.
    lower_pts = _nearest_input_sample(origin.origin + exact(decoder_seconds(start)), first_input["rate"])
    if audio_stream.start is not None:
        lower_pts = max(lower_pts, _nearest_input_sample(audio_stream.start, first_input["rate"]))
    upper = min(origin.to_raw(end), audio_stream.end) if audio_stream.end is not None else origin.to_raw(end)
    # The independently checked conversion permits at most one count and
    # one initial-phase output sample. These are evidence bounds, not retiming.
    if (first_input["pts"] < lower_pts
            or input_end > upper + Fraction(1, first_input["rate"])
            or audio_end > input_end + Fraction(2, first_output["rate"])):
        raise ValueError("Observed audio outside selected source/decoder bounds")
    if exact(audio["source_phase"]) != audio_start - origin.origin:
        raise ValueError("Audio source phase disagrees with common origin")
    for coverage, window in zip(audio["coverage"], windows):
        expected = "covered" if audio_start <= origin.to_raw(window[0]) and audio_end >= origin.to_raw(window[1]) else "unavailable"
        if coverage["status"] != expected:
            raise ValueError("Audio coverage contradicts observed sample bounds")



def validate_decode_limits(limits: Mapping[str, Any], *, window: bool = False) -> None:
    """One legal host-limit schema used by construction and receipt loading."""
    rational_names = {"preroll", "postroll", "max_window"}
    integer_names = {"timeout", "max_frames", "max_samples", "max_pixels", "max_decoded_bytes",
                     "max_retained_bytes", "max_log_bytes", "max_windows"}
    expected = rational_names | integer_names | ({"decoder_start", "decoder_end"} if window else set())
    if not isinstance(limits, Mapping) or set(limits) != expected:
        raise ValueError("Invalid decode limit schema")
    if any(type(limits[k]) is not int or limits[k] <= 0 for k in integer_names):
        raise ValueError("Decode limits require positive integers, not booleans")
    if (not 0 <= exact(limits["preroll"]) <= 1 or not 0 <= exact(limits["postroll"]) <= 1
            or not 0 < exact(limits["max_window"]) <= 8 or limits["timeout"] > 45):
        raise ValueError("Invalid decoder margin/window/timeout limit")
    if window and not 0 <= exact(limits["decoder_start"]) < exact(limits["decoder_end"]):
        raise ValueError("Invalid decoder window bounds")


def validate_audio_conversion(input_samples: int, input_rate: int, input_start: Fraction,
                              output_samples: int, output_rate: int, output_start: Fraction) -> None:
    """Exact count and initial phase invariant shared by producer and loader."""
    if (abs(Fraction(output_samples) - Fraction(input_samples * output_rate, input_rate)) > 1
            or abs(output_start - input_start) > Fraction(1, output_rate)):
        raise ValueError("Audio conversion count/phase disagreement")


def _validate_observations(data: dict[str, Any]) -> None:
    """Reject malformed records even when the envelope was rehashed by a caller."""
    limits = data["limits"]
    validate_decode_limits(limits, window=True)
    if (len(data["windows"]) > limits["max_windows"] or len(data["video"]) > limits["max_frames"]
            or len(data["outcomes"]) > limits["max_frames"]
            or exact(limits["decoder_end"]) - exact(limits["decoder_start"]) > exact(limits["max_window"])):
        raise ValueError("Observed window/frame budget exceeded")

    _validate_video_rows(data)
    _validate_audio_records(data, "max_frames")
    retained_indices = {o["frame_n"] for o in data["outcomes"] if o["status"] == "covered"}
    retained_bytes = sum(data["video"][n]["bytes"] for n in retained_indices) + data["audio"][0]["bytes"]
    if (any(f["width"] * f["height"] > limits["max_pixels"] for f in data["video"])
            or sum(f["bytes"] for f in data["video"]) > limits["max_decoded_bytes"]
            or data["audio"][0]["bytes"] > limits["max_decoded_bytes"]
            or retained_bytes > limits["max_retained_bytes"]):
        raise ValueError("Observed pixel/byte/retained budget exceeded")


def _validate_video_rows(data: dict[str, Any]) -> None:
    def integer(value: object, minimum: int = 0) -> bool:
        return type(value) is int and value >= minimum

    def hashed(record: dict[str, Any]) -> bool:
        return (isinstance(record.get("sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None
                and isinstance(record.get("checksum"), str)
                and re.fullmatch(r"[0-9A-F]{8}", record["checksum"]) is not None)

    previous: dict[str, Any] | None = None
    for n, frame in enumerate(data["video"]):
        required = {"n", "pts", "time_base", "width", "height", "format", "checksum",
                    "bytes", "sha256", "coverage_end"}
        if (not isinstance(frame, dict) or set(frame) != required or frame["n"] != n
                or not integer(frame["n"]) or type(frame["pts"]) is not int
                or not integer(frame["width"], 1) or not integer(frame["height"], 1)
                or frame["format"] != "bgr24" or not hashed(frame)
                or not integer(frame["bytes"], 1) or frame["bytes"] != frame["width"] * frame["height"] * 3
                or exact(frame["time_base"]) <= 0):
            raise ValueError("Invalid video observation")
        start = frame["pts"] * exact(frame["time_base"])
        if frame["coverage_end"] is not None and exact(frame["coverage_end"]) <= start:
            raise ValueError("Invalid video coverage")
        if previous is not None and (previous["time_base"] != frame["time_base"]
                or previous["coverage_end"] != rational(start)
                or previous["pts"] >= frame["pts"]
                or (previous["width"], previous["height"]) != (frame["width"], frame["height"])):
            raise ValueError("Invalid video observation order/coverage")
        previous = frame
    if previous is not None and previous["coverage_end"] is not None:
        raise ValueError("Terminal coverage requires independent EOF evidence")
    statuses = {"covered", "before_video_start", "missing_predecessor", "unknown_terminal"}
    for outcome in data["outcomes"]:
        if (not isinstance(outcome, dict) or set(outcome) != {"requested", "status", "frame_n"}
                or type(outcome["status"]) is not str or outcome["status"] not in statuses or not any(
                    exact(w[0]) <= exact(outcome["requested"]) < exact(w[1]) for w in data["windows"])):
            raise ValueError("Invalid sample outcome")
        frame_n = outcome["frame_n"]
        if outcome["status"] == "covered":
            if not integer(frame_n) or frame_n >= len(data["video"]) or data["video"][frame_n]["coverage_end"] is None:
                raise ValueError("Invalid sample outcome coverage")
        elif frame_n is not None:
            raise ValueError("Unavailable outcome cannot identify a frame")



def _validate_audio_records(data: dict[str, Any], block_limit: str,
                            _audio_contract: _AudioContract = _S16_AUDIO) -> None:
    """Shared block/conversion/phase/count rules for both receipt schemas."""
    def integer(value: object, minimum: int = 0) -> bool:
        return type(value) is int and value >= minimum

    def hashed(record: dict[str, Any]) -> bool:
        return (isinstance(record.get("sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None
                and isinstance(record.get("checksum"), str)
                and re.fullmatch(r"[0-9A-F]{8}", record["checksum"]) is not None)

    limits = data["limits"]
    for audio in data["audio"]:
        required_audio = {"policy", "input", "output", "samples", "bytes", "pcm_sha256",
                          "sample_grid_origin", "source_phase", "coverage"}
        if (not isinstance(audio, dict) or set(audio) != required_audio
                or type(audio["policy"]) is not str
                or audio["policy"] not in tuple(p[0] for p in _audio_contract.policies)
                or not isinstance(audio["pcm_sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", audio["pcm_sha256"]) is None):
            raise ValueError("Invalid audio observation")
        if (type(audio["coverage"]) is not list or len(audio["coverage"]) != len(data["windows"])
                or not integer(audio["samples"], 1) or not integer(audio["bytes"], 1)):
            raise ValueError("Invalid audio coverage/totals evidence")
        for coverage, window in zip(audio["coverage"], data["windows"]):
            if (type(coverage) is not dict or set(coverage) != {"window", "status"}
                    or type(coverage["window"]) is not list or coverage["window"] != window
                    or type(coverage["status"]) is not str or coverage["status"] not in ("covered", "unavailable")):
                raise ValueError("Invalid audio coverage window/status")
        for stage in ("input", "output"):
            previous = None
            if not isinstance(audio[stage], list) or not audio[stage]:
                raise ValueError("Missing audio observation")
            for n, block in enumerate(audio[stage]):
                fields = {"n", "pts", "time_base", "rate", "channels", "samples", "format", "checksum", "stage"}
                if stage == "output":
                    fields |= {"bytes", "sha256"}
                if (not isinstance(block, dict) or set(block) != fields or not integer(block["n"]) or block["n"] != n
                        or type(block["pts"]) is not int or not integer(block["samples"], 1)
                        or not integer(block["rate"], 1) or not integer(block["channels"], 1) or block["channels"] not in (1, 2)
                        or type(block["format"]) is not str or not block["format"]
                        or type(block["checksum"]) is not str or re.fullmatch(r"[0-9A-F]{8}", block["checksum"]) is None
                        or exact(block["time_base"]) != Fraction(1, block["rate"])
                        or block["stage"] != ("cxain" if stage == "input" else "cxaout")):
                    raise ValueError("Invalid audio block observation")
                if previous is not None and (block["pts"] != previous["pts"] + previous["samples"]
                        or any(block[k] != previous[k] for k in ("rate", "channels", "format"))):
                    raise ValueError("Invalid audio block continuity")
                if stage == "output" and (block["format"] != _audio_contract.format or not hashed(block) or not integer(block["bytes"], 1)
                        or block["bytes"] != block["samples"] * block["channels"] * _audio_contract.bytes_per_scalar):
                    raise ValueError("Invalid output audio association")
                previous = block
            if (len(audio[stage]) > limits[block_limit]
                    or sum(b["samples"] for b in audio[stage]) > limits["max_samples"]):
                raise ValueError("Observed audio frame/sample budget exceeded")
        if (audio["samples"] != sum(b["samples"] for b in audio["output"])
                or audio["bytes"] != sum(b["bytes"] for b in audio["output"])):
            raise ValueError("Invalid audio observation totals")
        first = audio["output"][0]
        if exact(audio["sample_grid_origin"]) != Fraction(first["pts"], first["rate"]):
            raise ValueError("Invalid audio sample phase")
        exact(audio["source_phase"])
        expected_rate, expected_channels = _audio_contract.audio_format(audio["policy"])
        if first["rate"] != expected_rate or first["channels"] != expected_channels:
            raise ValueError("Audio policy mismatch")
        first_input = audio["input"][0]
        if first_input["rate"] not in (44100, 48000):
            raise ValueError("Unsupported input audio rate")
        validate_audio_conversion(sum(b["samples"] for b in audio["input"]), first_input["rate"],
            Fraction(first_input["pts"], first_input["rate"]), audio["samples"], first["rate"],
            Fraction(first["pts"], first["rate"]))


AUDIO_LIMIT_CEILINGS = {
    "timeout": 45, "max_blocks": 65536, "max_samples": 14400000,
    "max_decoded_bytes": 60000000, "max_retained_bytes": 60000000, "max_log_bytes": 16000000,
}


def validate_audio_decode_limits(limits: Mapping[str, Any], *, window: bool = False) -> None:
    expected = {"preroll", "postroll", "max_window"} | set(AUDIO_LIMIT_CEILINGS)
    if window:
        expected |= {"decoder_start", "decoder_end"}
    if not isinstance(limits, Mapping) or set(limits) != expected:
        raise ValueError("Invalid audio-only limit schema")
    for name, ceiling in AUDIO_LIMIT_CEILINGS.items():
        if type(limits[name]) is not int or not 0 < limits[name] <= ceiling:
            raise ValueError("Audio-only limits must tighten finite integer ceilings")
    if (not 0 <= exact(limits["preroll"]) <= 1 or not 0 <= exact(limits["postroll"]) <= 1
            or not 0 < exact(limits["max_window"]) <= 300):
        raise ValueError("Invalid audio-only decoder margin/window")
    if window and not (0 <= exact(limits["decoder_start"]) < exact(limits["decoder_end"])
            and exact(limits["decoder_end"]) - exact(limits["decoder_start"]) <= exact(limits["max_window"])):
        raise ValueError("Invalid audio-only effective decoder window")


def _strict_json_scalars(value: object) -> None:
    """Reject scalar subclasses before JSON can erase their mutable identity."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("Receipt keys require plain strings")
            _strict_json_scalars(item)
    elif type(value) in (list, tuple):
        assert isinstance(value, (list, tuple))
        for item in value:
            _strict_json_scalars(item)
    elif value is not None and type(value) not in (str, int, bool):
        raise ValueError("Receipt values require exact plain JSON scalars")


def _unique_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise ValueError("Duplicate receipt field")
        result[key] = value
    return result


def _validate_audio_argv(argv: Any, _audio_contract: _AudioContract = _S16_AUDIO) -> None:
    """Constructor shape checks; semantic binding needs validate_origin's source."""
    if (type(argv) is not list or len(argv) != 1 or type(argv[0]) is not list
            or not argv[0] or any(type(arg) is not str or not arg for arg in argv[0])):
        raise ValueError("Audio-only receipt requires one actual argv")
    command = argv[0]
    for flag in ("-i", "-map", "-af", "-c:a", "-f"):
        if command.count(flag) != 1 or command.index(flag) + 1 >= len(command):
            raise ValueError("Missing/duplicate actual audio argv option")
    if (any(command.count(flag) != 1 for flag in ("-vn", "-sn", "-dn", "-copyts"))
            or any(flag in command for flag in ("-vf", "-filter:v", "-filter_complex", "-c:v", "-an"))
            or command[command.index("-c:a") + 1] != _audio_contract.codec
            or command[command.index("-f") + 1] != _audio_contract.muxer or command[-1] != "pipe:1"):
        raise ValueError("Audio-only argv must execute only the observed PCM path")
    graph = command[command.index("-af") + 1]
    if not all(part in graph for part in ("atrim=end=", "ashowinfo@cxain", ":async=0",
            f"aformat=sample_fmts={_audio_contract.format}:channel_layouts=", "ashowinfo@cxaout")):
        raise ValueError("Missing observed no-healing audio graph")


@dataclass(frozen=True, slots=True, init=False)
class SourceAudioDecodeReceipt:
    """Detached audio-only evidence; never implies a video observation."""

    origin_id: str
    decoder_id: str
    policy: str
    payload_json: str

    def __init__(self, origin_id: str, decoder_id: str, policy: str,
                 payload: Mapping[str, object]) -> None:
        if (any(type(v) is not str for v in (origin_id, decoder_id, policy))
                or not re.fullmatch(r"[0-9a-f]{64}", origin_id) or not decoder_id
                or not policy.endswith(";audio-only;v1")):
            raise ValueError("Invalid audio-only receipt identity/policy")
        if not isinstance(payload, Mapping) or set(payload) != {"windows", "limits", "audio", "argv"}:
            raise ValueError("Invalid audio-only receipt fields")
        _strict_json_scalars(payload)
        try:
            encoded = canonical(payload)
        except (TypeError, ValueError) as exc:
            raise ValueError("Receipt contains unsupported JSON values") from exc
        data = json.loads(encoded)
        if (type(data["windows"]) is not list or len(data["windows"]) != 1
                or type(data["audio"]) is not list or len(data["audio"]) != 1):
            raise ValueError("Audio-only receipt requires one window/audio record")
        window = data["windows"][0]
        if type(window) is not list or len(window) != 2 or not 0 <= exact(window[0]) < exact(window[1]):
            raise ValueError("Invalid audio-only window")
        _validate_audio_argv(data["argv"])
        validate_audio_decode_limits(data["limits"], window=True)
        _validate_audio_records(data, "max_blocks")
        audio = data["audio"][0]
        if audio["bytes"] > min(data["limits"]["max_decoded_bytes"], data["limits"]["max_retained_bytes"]):
            raise ValueError("Observed audio byte budget exceeded")
        for name, value in (("origin_id", origin_id), ("decoder_id", decoder_id),
                            ("policy", policy), ("payload_json", encoded)):
            object.__setattr__(self, name, value)

    def to_json(self) -> str:
        body = {"version": 1, "kind": "audio_only", "scope": "window", "origin_id": self.origin_id,
                "decoder_id": self.decoder_id, "policy": self.policy, "payload": json.loads(self.payload_json)}
        return canonical({**body, "sha256": digest(canonical(body))})

    @classmethod
    def from_json(cls, text: str, *, origin: SourceTimeOrigin,
                  policy: str) -> "SourceAudioDecodeReceipt":
        data = json.loads(text, object_pairs_hook=_unique_pairs)
        if type(data) is not dict or set(data) != {
                "version", "kind", "scope", "origin_id", "decoder_id", "policy", "payload", "sha256"}:
            raise ValueError("Invalid audio-only receipt schema")
        if (type(data["version"]) is not int or data["version"] != 1 or data["kind"] != "audio_only"
                or data["scope"] != "window" or data["origin_id"] != origin.identity
                or data["decoder_id"] != origin.decoder_id or data["policy"] != policy):
            raise ValueError("Stale or mismatched audio-only receipt identity/policy")
        claimed = data.pop("sha256")
        if digest(canonical(data)) != claimed:
            raise ValueError("Receipt hash mismatch")
        receipt = cls(data["origin_id"], data["decoder_id"], policy, data["payload"])
        receipt.validate_origin(origin)
        return receipt

    def validate_origin(self, origin: SourceTimeOrigin) -> None:
        if self.origin_id != origin.identity or self.decoder_id != origin.decoder_id:
            raise ValueError("Receipt origin/decoder identity mismatch")
        data = json.loads(self.payload_json)
        windows = [(exact(w[0]), exact(w[1])) for w in data["windows"]]
        for start, end in windows:
            origin.to_raw(start)
            origin.to_raw(end)
        limits = data["limits"]
        start, end = exact(limits["decoder_start"]), exact(limits["decoder_end"])
        if (start != max(Fraction(0), windows[0][0] - exact(limits["preroll"]))
                or end != min(origin.duration, windows[0][1] + exact(limits["postroll"]))):
            raise ValueError("Decoder margins disagree with declared window/origin")
        streams = [s for s in origin.streams if s.kind == "audio"]
        if len(streams) != 1:
            raise ValueError("Receipt requires one selected audio stream")
        command = data["argv"][0]
        expected = audio_decode_argv(command[0], origin, start, end, data["audio"][0]["policy"])
        if command != expected:
            raise ValueError("Audio argv contradicts canonical source/interval/conversion operation")
        _validate_audio_origin(data, origin, streams[0], start, end, windows)


STREAM_DECODE_POLICY = "bgr24-showinfo;pcm-s16-ashowinfo;copyts;no-healing;v1;selected-stream;v1"

FLOAT_AUDIO_LIMIT_CEILINGS = {
    "timeout": 45, "max_blocks": 16384, "max_samples": 9_000_000,
    "max_block_samples": 65536, "max_decoded_bytes": 72_000_000,
    "max_retained_bytes": 72_000_000, "max_log_bytes": 16_000_000,
    "max_metadata_bytes": 8_000_000, "io_chunk_bytes": 65536,
}


def validate_float_audio_limits(limits: Mapping[str, object], *, window: bool = False) -> None:
    expected = "preroll postroll max_window " + " ".join(FLOAT_AUDIO_LIMIT_CEILINGS)
    _stream_keys(limits, expected + (" decoder_start decoder_end" if window else ""))
    for name, ceiling in FLOAT_AUDIO_LIMIT_CEILINGS.items():
        if _stream_int(limits[name], 1) > ceiling:
            raise ValueError("Float limits must tighten finite ceilings")
    for name, fraction_ceiling in (("preroll", Fraction(1, 2)), ("postroll", Fraction(1, 4)),
                          ("max_window", Fraction(180))):
        value = _stream_rational(limits[name])
        if value < 0 or value > fraction_ceiling or (name == "max_window" and value == 0):
            raise ValueError("Invalid float decoder margin/window")
    if window:
        start, end = (_stream_rational(limits[n]) for n in ("decoder_start", "decoder_end"))
        if not 0 <= start < end or end - start > _stream_rational(limits["max_window"]):
            raise ValueError("Invalid float decoder bounds")


def _float_runtime(runtime: Mapping[str, object]) -> None:
    _stream_keys(runtime, "runtime_id ffmpeg ffprobe ffmpeg_sha256 ffprobe_sha256")
    _stream_guard(runtime, 8_000_000)
    for name, value in runtime.items():
        if type(value) is not str or not value:
            raise ValueError("Float runtime requires nonempty plain strings")
        if name.endswith("sha256") and re.fullmatch(r"[0-9A-F]{64}", value) is None:
            raise ValueError("Float runtime requires uppercase SHA256 pins")


def _float_origin(origin: SourceTimeOrigin, cap: int) -> None:
    try:
        _stream_origin(origin, cap)
        if type(origin.streams) is not tuple or any(type(getattr(origin, name)) is not str
                or getattr(origin, name) != SourceTimeOrigin.__dataclass_fields__[name].default
                for name in ("policy", "fingerprint_method")):
            raise ValueError("Float origin immutable policy/schema mismatch")
        # Revalidate bypassed known clocks, including the mandatory timebase.
        # The existing StreamClock owns semantics; bounded39 owns numeric guards.
        for clock in origin.streams:
            _stream_fraction(clock.time_base)
            StreamClock(**{name: getattr(clock, name) for name in clock.__dataclass_fields__})
    except (TypeError, AttributeError, KeyError, OverflowError) as exc:
        raise ValueError("Malformed float source origin") from exc


def source_decoder_identity(runtime: Mapping[str, object]) -> str:
    _float_runtime(runtime)
    return digest(canonical({"runtime": runtime["runtime_id"], "ffmpeg": runtime["ffmpeg_sha256"],
                             "ffprobe": runtime["ffprobe_sha256"], "policy": DECODE_POLICY}))


def _stream_int(value: object, minimum: int | None = 0) -> int:
    if type(value) is not int or value.bit_length() > 4096 or (minimum is not None and value < minimum):
        raise ValueError("Stream scalar requires a bounded plain integer")
    return value


def _stream_fraction(value: object) -> Fraction:
    if type(value) is not Fraction:
        raise ValueError("Stream time requires a plain Fraction")
    assert isinstance(value, Fraction)
    _stream_int(value.numerator, None)
    _stream_int(value.denominator, 1)
    return Fraction(value.numerator, value.denominator)


def _stream_rational(value: object) -> Fraction:
    if type(value) is not str or len(value) > 2470 or re.fullmatch(r"-?\d+/[1-9]\d*", value) is None:
        raise ValueError("Stream time requires bounded canonical rational text")
    numerator, denominator = value.split("/")
    _stream_int(int(numerator), None)
    _stream_int(int(denominator), 1)
    result = exact(value)
    _stream_fraction(result)
    if rational(result) != value:
        raise ValueError("Noncanonical stream rational")
    return result


def _stream_session(value: object) -> str:
    if type(value) is not str or len(value) != 36:
        raise ValueError("Stream session requires canonical UUIDv4")
    parsed = uuid.UUID(value)
    if parsed.version != 4 or parsed.variant != uuid.RFC_4122 or str(parsed) != value:
        raise ValueError("Stream session requires canonical UUIDv4")
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class StreamDecodeLimits:
    preroll: Fraction = Fraction(1, 2)
    postroll: Fraction = Fraction(1, 4)
    max_window: Fraction = Fraction(8)
    timeout: int = 45
    max_frames: int = 1200
    max_samples: int = 480000
    max_pixels: int = 8847360
    max_decoded_bytes: int = 2000000000
    max_log_bytes: int = 2000000
    window_timeout: int = 90
    cleanup_grace: int = 5
    max_requests: int = 1200
    max_deliveries: int = 1200
    max_live_selections: int = 1
    io_chunk_bytes: int = 65536
    max_live_payload_bytes: int = 96000000
    max_selected_unique_bytes: int = 2000000000
    max_request_emission_bytes: int = 8000000000
    max_finalization_bytes: int = 8000000

    def __post_init__(self) -> None:
        for name, field in self.__dataclass_fields__.items():
            value = getattr(self, name)
            ceiling: Any = field.default
            if name in ("preroll", "postroll", "max_window"):
                value = _stream_fraction(value)
                object.__setattr__(self, name, value)
                if value > ceiling or (value <= 0 if name == "max_window" else value < 0):
                    raise ValueError("Stream time limit may only tighten HOST ceiling")
            elif _stream_int(value, 1) > ceiling:
                raise ValueError("Stream limit may only tighten HOST ceiling")

    def operation_record(self) -> dict[str, object]:
        return {name: rational(value) if isinstance(value, Fraction) else value
                for name in tuple(self.__dataclass_fields__)[:9] for value in (getattr(self, name),)}

    def delivery_record(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in tuple(self.__dataclass_fields__)[9:]}

    @classmethod
    def from_records(cls, operation: Mapping[str, object], delivery: Mapping[str, object]) -> "StreamDecodeLimits":
        keys = tuple(cls.__dataclass_fields__)
        if (not isinstance(operation, Mapping) or not isinstance(delivery, Mapping)
                or len(operation) != 9 or len(delivery) != 10
                or set(operation) != set(keys[:9]) or set(delivery) != set(keys[9:])):
            raise ValueError("Stream limit record schema mismatch")
        _stream_guard(operation, 8000000)
        _stream_guard(delivery, 8000000)
        values: dict[str, Any] = {**operation, **delivery}
        for name in keys[:3]:
            values[name] = _stream_rational(values[name])
        return cls(**values)


def _stream_guard(value: object, cap: int, depth: int = 0) -> int:
    """Bound known JSON metadata before cloning; count conservative ASCII expansion."""
    if depth > 10:
        raise ValueError("Stream metadata nesting limit")
    if isinstance(value, Mapping):
        used = 2
        for key, item in value.items():
            if type(key) is not str:
                raise ValueError("Stream JSON keys require plain strings")
            used += _stream_guard(key, cap - used, depth + 1) + 2
            used += _stream_guard(item, cap - used, depth + 1)
    elif type(value) is list:
        assert isinstance(value, list)
        if len(value) > cap:
            raise ValueError("Stream metadata count limit")
        used = 2
        for item in value:
            used += 1 + _stream_guard(item, cap - used, depth + 1)
    elif type(value) is str:
        assert isinstance(value, str)
        if len(value) > cap:
            raise ValueError("Stream string limit")
        used = 2
        for offset in range(0, len(value), 1024):
            chunk = value[offset:offset + 1024]
            try:
                chunk.encode("utf-8")
            except UnicodeError as exc:
                raise ValueError("Invalid stream UTF8") from exc
            used += sum(12 if ord(c) > 65535 else 6 if ord(c) < 32 or ord(c) > 126
                        else 2 if c in ('"', '\\') else 1 for c in chunk)
            if used > cap:
                raise ValueError("Stream JSON expansion limit")
    elif type(value) is int:
        used = len(str(_stream_int(value, None)))
    elif value is None or type(value) is bool:
        used = 5
    else:
        raise ValueError("Stream requires exact plain JSON metadata")
    if used > cap:
        raise ValueError("Stream metadata byte limit")
    return used


def _stream_keys(value: Any, names: str) -> None:
    if type(value) is not dict or set(value) != set(names.split()):
        raise ValueError("Stream record schema mismatch")


def _float_limits(data: Any, host: Mapping[str, object]) -> None:
    validate_float_audio_limits(host)
    validate_float_audio_limits(data, window=True)
    for name in host:
        value, ceiling = data[name], host[name]
        if name in ("preroll", "postroll", "max_window"):
            value, ceiling = _stream_rational(value), _stream_rational(ceiling)
        if value > ceiling:
            raise ValueError("Presented float limits exceed HOST")


def _float_payload(origin: SourceTimeOrigin, data: dict[str, Any], runtime: Mapping[str, object],
                   host: Mapping[str, object]) -> None:
    _stream_keys(data, "windows limits audio argv")
    _float_limits(data["limits"], host)
    limits = data["limits"]
    if (type(data["windows"]) is not list or len(data["windows"]) != 1
            or type(data["windows"][0]) is not list or len(data["windows"][0]) != 2
            or type(data["audio"]) is not list or len(data["audio"]) != 1):
        raise ValueError("Float receipt requires one window/audio")
    window = (_stream_rational(data["windows"][0][0]), _stream_rational(data["windows"][0][1]))
    if not 0 <= window[0] < window[1] <= origin.duration:
        raise ValueError("Invalid float source window")
    start, end = (_stream_rational(limits[n]) for n in ("decoder_start", "decoder_end"))
    if (start != max(Fraction(0), window[0] - _stream_rational(limits["preroll"]))
            or end != min(origin.duration, window[1] + _stream_rational(limits["postroll"]))):
        raise ValueError("Float decoder margins disagree with source window")
    audio = data["audio"][0]
    _stream_keys(audio, "policy input output samples bytes pcm_sha256 sample_grid_origin source_phase coverage")
    _stream_int(audio["samples"], 1)
    _stream_int(audio["bytes"], 1)
    _stream_rational(audio["sample_grid_origin"])
    _stream_rational(audio["source_phase"])
    selected = next(s for s in origin.streams if s.kind == "audio")
    for stage in ("input", "output"):
        blocks = audio[stage]
        if type(blocks) is not list or not blocks or len(blocks) > limits["max_blocks"]:
            raise ValueError("Float per-stage block limit")
        for block in blocks:
            _stream_keys(block, "n pts time_base rate channels samples format checksum stage" +
                         (" bytes sha256" if stage == "output" else ""))
            for name in ("n", "pts", "rate", "channels", "samples"):
                _stream_int(block[name], None if name == "pts" else 0 if name == "n" else 1)
            _stream_rational(block["time_base"])
            if block["samples"] > limits["max_block_samples"]:
                raise ValueError("Float per-block sample limit")
            if stage == "input" and (block["rate"], block["channels"], block["format"]) != (
                    selected.sample_rate, selected.channels, selected.sample_format):
                raise ValueError("Float input format/source mismatch")
            if stage == "output":
                _stream_int(block["bytes"], 1)
    _validate_audio_records(data, "max_blocks", _FLOAT_AUDIO)
    if audio["bytes"] > min(limits["max_decoded_bytes"], limits["max_retained_bytes"],
                             4 * selected.channels * limits["max_samples"]):
        raise ValueError("Float payload byte limit")
    _validate_audio_argv(data["argv"], _FLOAT_AUDIO)
    if data["argv"] != [float_audio_decode_argv(str(runtime["ffmpeg"]), origin, start, end, audio["policy"])]:
        raise ValueError("Float argv contradicts current runtime/source/window")
    _validate_audio_origin(data, origin, selected, start, end, [window])


@dataclass(frozen=True, slots=True, init=False)
class SourceFloatAudioDecodeReceipt:
    origin_id: str
    origin_decoder_id: str
    operation_id: str
    policy: str
    runtime_json: str
    payload_json: str

    def __init__(self, origin: SourceTimeOrigin, payload: Mapping[str, object], *,
                 runtime: Mapping[str, object], limits: Mapping[str, object]) -> None:
        validate_float_audio_limits(limits)
        cap = _stream_int(limits["max_metadata_bytes"], 1)
        _float_origin(origin, cap)
        _float_runtime(runtime)
        _stream_guard(payload, cap)
        _stream_keys(payload, "windows limits audio argv")
        _float_limits(payload["limits"], limits)
        presented = payload["limits"]
        assert isinstance(presented, dict)
        effective_cap = _stream_int(presented["max_metadata_bytes"], 1)
        # Reserve the complete envelope before canonical, origin hashing or cloning.
        _stream_guard(dict(version=1, kind="audio_only_float", scope="window", origin_id="0" * 64,
            origin_decoder_id=origin.decoder_id, operation_id="0" * 64, policy=FLOAT_AUDIO_DECODE_POLICY,
            runtime=runtime, payload=payload, sha256="0" * 64), effective_cap)
        if origin.decoder_id != source_decoder_identity(runtime):
            raise ValueError("Float origin decoder/current runtime mismatch")
        encoded = canonical(payload)
        data = json.loads(encoded)
        try:
            _float_payload(origin, data, runtime, limits)
        except (TypeError, KeyError, IndexError, OverflowError, ZeroDivisionError, StopIteration) as exc:
            raise ValueError("Malformed float audio metadata") from exc
        operation = digest(canonical(dict(origin_decoder_id=origin.decoder_id, runtime=runtime,
            policy=FLOAT_AUDIO_DECODE_POLICY, audio_policy=data["audio"][0]["policy"])))
        for name, value in (("origin_id", origin.identity), ("origin_decoder_id", origin.decoder_id),
                            ("operation_id", operation), ("policy", FLOAT_AUDIO_DECODE_POLICY),
                            ("runtime_json", canonical(runtime)), ("payload_json", encoded)):
            object.__setattr__(self, name, value)
        _stream_guard(self.record(), effective_cap)

    def record(self) -> dict[str, Any]:
        body = dict(version=1, kind="audio_only_float", scope="window", origin_id=self.origin_id,
            origin_decoder_id=self.origin_decoder_id, operation_id=self.operation_id, policy=self.policy,
            runtime=json.loads(self.runtime_json), payload=json.loads(self.payload_json))
        return {**body, "sha256": digest(canonical(body))}

    def to_json(self) -> str:
        return canonical(self.record())

    @property
    def sha256(self) -> str:
        return str(self.record()["sha256"])

    @classmethod
    def from_json(cls, text: str, *, origin: SourceTimeOrigin,
                  runtime: Mapping[str, object], limits: Mapping[str, object]) -> "SourceFloatAudioDecodeReceipt":
        validate_float_audio_limits(limits)
        data, size = _bounded_json(text, _stream_int(limits["max_metadata_bytes"], 1))
        _stream_keys(data, "version kind scope origin_id origin_decoder_id operation_id policy runtime payload sha256")
        if type(data["version"]) is not int or data["version"] != 1:
            raise ValueError("Invalid float envelope version")
        _stream_keys(data["payload"], "windows limits audio argv")
        _float_limits(data["payload"]["limits"], limits)
        cap = _stream_int(data["payload"]["limits"]["max_metadata_bytes"], 1)
        if size > cap:
            raise ValueError("Float presented envelope byte limit")
        _stream_guard(data, cap)
        receipt = cls(origin, data["payload"], runtime=runtime, limits=limits)
        if data != receipt.record():
            raise ValueError("Float envelope identity/schema/hash mismatch")
        return receipt

    def validate_origin(self, origin: SourceTimeOrigin, *, runtime: Mapping[str, object]) -> None:
        _float_origin(origin, 8_000_000)
        _float_runtime(runtime)
        data = self.record()["payload"]
        host = {k: v for k, v in data["limits"].items() if k not in ("decoder_start", "decoder_end")}
        checked = type(self)(origin, data, runtime=runtime, limits=host)
        if self.to_json() != checked.to_json():
            raise ValueError("Float receipt association mismatch")


def _stream_header(origin: SourceTimeOrigin, session: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {"chain": "selected-stream-v1", "session_id": session, "origin_id": origin.identity,
        "decoder_id": origin.decoder_id, "policy": STREAM_DECODE_POLICY,
        **{k: payload[k] for k in ("window", "requests", "operation_limits", "delivery_limits",
                                  "decoder_start", "decoder_end")}, "audio_policy": payload["audio"]["policy"]}


def _stream_link(previous: str, delivery: Mapping[str, Any], selected: Mapping[str, Any],
                 successor: Mapping[str, Any]) -> str:
    return digest(canonical({"previous": previous,
        "delivery": {k: delivery[k] for k in ("index", "request_ordinals", "frame_n", "successor_frame_n")},
        "selected": selected, "successor": {k: v for k, v in successor.items() if k != "coverage_end"}}))


def _stream_origin(origin: SourceTimeOrigin, cap: int) -> None:
    if type(origin) is not SourceTimeOrigin:
        raise ValueError("Stream requires actual SourceTimeOrigin")
    # Revalidate known scalar values before origin hashing or command expansion.
    if (len(origin.streams) != 2 or sum(s.kind == "video" for s in origin.streams) != 1
            or sum(s.kind == "audio" for s in origin.streams) != 1):
        raise ValueError("Stream requires one video and one audio clock")
    for name in ("path", "fingerprint", "decoder_id"):
        _stream_guard(getattr(origin, name), cap)
    _stream_int(origin.size)
    _stream_int(origin.mtime_ns, None)
    if type(origin.color) is not tuple or len(origin.color) > cap:
        raise ValueError("Stream color metadata count/type limit")
    used = 2
    for pair in origin.color:
        if type(pair) is not tuple or len(pair) != 2:
            raise ValueError("Stream color metadata shape")
        used += _stream_guard(pair[0], cap - used) + _stream_guard(pair[1], cap - used)
    _stream_fraction(origin.origin)
    _stream_fraction(origin.duration)
    for stream in origin.streams:
        if type(stream) is not StreamClock:
            raise ValueError("Stream clock identity mismatch")
        for name in ("index", "width", "height", "sample_rate", "channels"):
            _stream_int(getattr(stream, name))
        for name in ("time_base", "start", "end"):
            if (value := getattr(stream, name)) is not None:
                _stream_fraction(value)
    _stream_guard(origin.record(), cap)
    SourceTimeOrigin(origin.path, origin.size, origin.mtime_ns, origin.fingerprint, origin.origin,
                     origin.duration, origin.streams, origin.decoder_id, dict(origin.color))


def _stream_payload_limits(data: Any, host: StreamDecodeLimits) -> StreamDecodeLimits:
    """Validate the presented limits against HOST before envelope-size/hash work."""
    _stream_keys(data, "window requests operation_limits delivery_limits decoder_start decoder_end "
                       "video audio outcomes argv deliveries accounting completion")
    limits = StreamDecodeLimits.from_records(data["operation_limits"], data["delivery_limits"])
    for name in limits.__dataclass_fields__:
        if getattr(limits, name) > getattr(host, name):
            raise ValueError("Presented stream limits exceed HOST")
    return limits


def _validate_stream_payload(origin: SourceTimeOrigin, session: str, data: dict[str, Any],
                             host: StreamDecodeLimits) -> StreamDecodeLimits:
    limits = _stream_payload_limits(data, host)
    window = data["window"]
    if type(window) is not list or len(window) != 2:
        raise ValueError("Stream requires one window")
    start, end = map(_stream_rational, window)
    if not 0 <= start < end <= origin.duration:
        raise ValueError("Stream window outside origin")
    requests = data["requests"]
    if type(requests) is not list or not 0 < len(requests) <= min(limits.max_requests, limits.max_frames):
        raise ValueError("Stream request count limit")
    values = [_stream_rational(t) for t in requests]
    if values != sorted(values) or any(not start <= t < end for t in values):
        raise ValueError("Stream requests require ordered in-window times")
    decoder_start, decoder_end = _stream_rational(data["decoder_start"]), _stream_rational(data["decoder_end"])
    if (decoder_start != max(Fraction(0), start - limits.preroll)
            or decoder_end != min(origin.duration, end + limits.postroll)
            or decoder_end - decoder_start > limits.max_window):
        raise ValueError("Stream decoder margins/limit mismatch")
    for key in ("video", "outcomes", "deliveries", "argv"):
        if type(data[key]) is not list:
            raise ValueError("Stream requires bounded arrays")
    video, audio = data["video"], data["audio"]
    if not 2 <= len(video) <= limits.max_frames or len(data["outcomes"]) != len(requests):
        raise ValueError("Stream video/outcome count mismatch")
    _stream_keys(audio, "policy input output samples bytes pcm_sha256 sample_grid_origin source_phase coverage")
    for row in video:
        if type(row) is not dict:
            raise ValueError("Stream video row shape")
        _stream_rational(row.get("time_base"))
        if row.get("coverage_end") is not None:
            _stream_rational(row["coverage_end"])
    for stage in ("input", "output"):
        if type(audio[stage]) is not list or not 0 < len(audio[stage]) <= limits.max_frames:
            raise ValueError("Stream audio block limit")
        for row in audio[stage]:
            if type(row) is not dict:
                raise ValueError("Stream audio row shape")
            _stream_rational(row.get("time_base"))
    _stream_rational(audio["sample_grid_origin"])
    _stream_rational(audio["source_phase"])
    view = {"windows": [window], "limits": {**limits.operation_record(),
        "decoder_start": data["decoder_start"], "decoder_end": data["decoder_end"]},
        "video": video, "audio": [audio], "outcomes": data["outcomes"]}
    _validate_video_rows(view)
    _validate_audio_records(view, "max_frames")
    _validate_observation_origin(view, origin)
    if [r["requested"] for r in data["outcomes"]] != requests or any(
            r["status"] != "covered" for r in data["outcomes"]):
        raise ValueError("Stream success requires every request covered")
    if len(data["argv"]) != 2:
        raise ValueError("Stream requires video then audio argv")
    for command in data["argv"]:
        if (type(command) is not list or not 0 < len(command) <= 128
                or any(type(arg) is not str or not arg for arg in command)):
            raise ValueError("Stream argv shape")
    exe = data["argv"][0][0]
    if data["argv"] != [video_decode_argv(exe, origin, decoder_start, decoder_end),
                        audio_decode_argv(exe, origin, decoder_start, decoder_end, audio["policy"])]:
        raise ValueError("Stream argv contradict canonical operation")
    deliveries = data["deliveries"]
    if not 0 < len(deliveries) <= limits.max_deliveries:
        raise ValueError("Stream delivery count limit")
    ordinal, last_frame = 0, -1
    previous = digest(canonical(_stream_header(origin, session, data)))
    for index, delivery in enumerate(deliveries):
        _stream_keys(delivery, "index request_ordinals frame_n successor_frame_n chain_sha256")
        n = _stream_int(delivery["frame_n"])
        if (_stream_int(delivery["index"]) != index or n <= last_frame or n + 1 >= len(video)
                or _stream_int(delivery["successor_frame_n"]) != n + 1):
            raise ValueError("Stream delivery frame/index association")
        ordinals = delivery["request_ordinals"]
        if type(ordinals) is not list or not ordinals:
            raise ValueError("Stream requires ordinal group")
        for supplied in ordinals:
            if (_stream_int(supplied) != ordinal or ordinal >= len(requests)
                    or data["outcomes"][ordinal]["frame_n"] != n):
                raise ValueError("Stream ordinal/outcome association")
            ordinal += 1
        previous = _stream_link(previous, delivery, video[n], video[n + 1])
        if delivery["chain_sha256"] != previous:
            raise ValueError("Stream delivery chain mismatch")
        last_frame = n
    if ordinal != len(requests):
        raise ValueError("Stream missing delivery ordinals")
    a = data["accounting"]
    totals = dict(decoded_video_frames=len(video), decoded_video_bytes=sum(f["bytes"] for f in video),
        decoded_audio_output_bytes=audio["bytes"], selected_unique_frames=len(deliveries),
        selected_unique_bytes=sum(video[d["frame_n"]]["bytes"] for d in deliveries),
        offered_deliveries=len(deliveries), offered_payload_bytes=sum(video[d["frame_n"]]["bytes"] for d in deliveries),
        acknowledged_deliveries=len(deliveries), acknowledged_requests=len(requests),
        acknowledged_request_bytes=sum(video[o["frame_n"]]["bytes"] for o in data["outcomes"]))
    _stream_keys(a, " ".join(totals) + " reserved_payload_capacity_bytes peak_live_payload_bytes_accounted "
                    "metadata_bytes video_log_bytes audio_log_bytes")
    for value in a.values():
        _stream_int(value)
    if any(a[k] != v for k, v in totals.items()):
        raise ValueError("Stream accounting totals mismatch")
    streams = [s for s in origin.streams if s.kind == "video"]
    frame_bytes = streams[0].width * streams[0].height * 3
    reservation = max(3 * frame_bytes, 4 * limits.max_samples * audio_format(audio["policy"])[1] * 2)
    if (a["reserved_payload_capacity_bytes"] != reservation or reservation > limits.max_live_payload_bytes
            or not max(frame_bytes, audio["bytes"]) <= a["peak_live_payload_bytes_accounted"] <= reservation
            or a["selected_unique_bytes"] > limits.max_selected_unique_bytes
            or a["acknowledged_request_bytes"] > limits.max_request_emission_bytes
            or any(row["width"] * row["height"] > limits.max_pixels for row in video)
            or max(a["decoded_video_bytes"], audio["bytes"]) > limits.max_decoded_bytes
            or any(not 0 < a[k] <= limits.max_log_bytes for k in ("video_log_bytes", "audio_log_bytes"))):
        raise ValueError("Stream resource accounting limit")
    metadata = {k: v for k, v in data.items() if k not in ("accounting", "completion")}
    if a["metadata_bytes"] != len(canonical(metadata).encode("utf-8")):
        raise ValueError("Stream metadata byte accounting mismatch")
    completion = data["completion"]
    _stream_keys(completion, "children source_fingerprint_after pins_verified_before_children "
                            "all_requests_acknowledged cleanup_complete elapsed_ns")
    elapsed = _stream_int(completion["elapsed_ns"])
    if (completion["source_fingerprint_after"] != origin.fingerprint
            or any(completion[k] is not True for k in ("pins_verified_before_children",
                  "all_requests_acknowledged", "cleanup_complete"))
            or elapsed >= limits.window_timeout * 10**9):
        raise ValueError("Stream completion/deadline mismatch")
    children = completion["children"]
    if type(children) is not list or len(children) != 2:
        raise ValueError("Stream requires two completed children")
    last_finish = 0
    for kind, child in zip(("video", "audio"), children):
        _stream_keys(child, "kind exit_code started_ns finished_ns cleanup_complete")
        began, finished = _stream_int(child["started_ns"]), _stream_int(child["finished_ns"])
        if (child["kind"] != kind or _stream_int(child["exit_code"], None) != 0
                or child["cleanup_complete"] is not True or not last_finish <= began <= finished <= elapsed
                or finished - began >= limits.timeout * 10**9):
            raise ValueError("Stream child order/exit/deadline/cleanup mismatch")
        last_finish = finished
    return limits


def _bounded_json(text: str, cap: int) -> tuple[Any, int]:
    """Shared bounded39 JSON preflight, before parser allocation or exact arithmetic."""
    if type(text) is not str or len(text) > cap:
        raise ValueError("Stream envelope byte/type limit")
    size, depth, quoted, escaped = 0, 0, False, False
    for offset in range(0, len(text), 1024):
        try:
            size += len(text[offset:offset + 1024].encode("utf-8"))
        except UnicodeError as exc:
            raise ValueError("Invalid stream UTF8") from exc
        if size > cap:
            raise ValueError("Stream envelope byte limit")
    for char in text:
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char in "[{":
            depth += 1
            if depth > 10:
                raise ValueError("Stream JSON nesting limit")
        elif char in "]}":
            depth -= 1
    def integer(token: str) -> int:
        if len(token) > 1235:
            raise ValueError("Stream integer token limit")
        return _stream_int(int(token), None)
    def invalid(token: str) -> Any:
        raise ValueError("Stream rejects floating JSON scalars")
    try:
        data = json.loads(text, object_pairs_hook=_unique_pairs, parse_int=integer,
                          parse_float=invalid, parse_constant=invalid)
        _stream_guard(data, cap)
        return data, size
    except (RecursionError, OverflowError, TypeError, KeyError) as exc:
        raise ValueError("Malformed stream JSON") from exc


@dataclass(frozen=True, slots=True, init=False)
class SourceStreamFinalization:
    """Checked metadata consistency, not authentication or proof of HOST execution."""
    origin_id: str
    decoder_id: str
    session_id: str
    policy: str
    payload_json: str

    def __init__(self, origin: SourceTimeOrigin, session_id: str, payload: Mapping[str, object], *,
                 limits: StreamDecodeLimits = StreamDecodeLimits()) -> None:
        if type(limits) is not StreamDecodeLimits:
            raise ValueError("Stream requires actual HOST limits")
        _stream_session(session_id)
        _stream_origin(origin, limits.max_finalization_bytes)
        if not isinstance(payload, Mapping):
            raise ValueError("Stream payload must be a mapping")
        _stream_guard(payload, limits.max_finalization_bytes)
        detached = dict(payload)
        effective = _stream_payload_limits(detached, limits)
        # Reserve the known complete ASCII-JSON expansion before serialization/hashing.
        _stream_guard(dict(version=1, kind="streamed_video_window", scope="window",
            session_id=session_id, origin_id="0" * 64, decoder_id=origin.decoder_id,
            policy=STREAM_DECODE_POLICY, payload=detached, sha256="0" * 64), effective.max_finalization_bytes)
        try:
            encoded = canonical(detached)
        except (TypeError, OverflowError) as exc:
            raise ValueError("Stream payload contains unsupported containers") from exc
        data = json.loads(encoded)
        try:
            effective = _validate_stream_payload(origin, session_id, data, limits)
        except (KeyError, IndexError, TypeError, OverflowError, ZeroDivisionError) as exc:
            raise ValueError("Malformed stream metadata") from exc
        for name, value in (("origin_id", origin.identity), ("decoder_id", origin.decoder_id),
                            ("session_id", session_id), ("policy", STREAM_DECODE_POLICY), ("payload_json", encoded)):
            object.__setattr__(self, name, value)
        _stream_guard(self.record(), effective.max_finalization_bytes)
        if len(self.to_json().encode("utf-8")) > effective.max_finalization_bytes:
            raise ValueError("Stream finalization byte limit")

    def record(self) -> dict[str, Any]:
        body = dict(version=1, kind="streamed_video_window", scope="window", session_id=self.session_id,
            origin_id=self.origin_id, decoder_id=self.decoder_id, policy=self.policy, payload=json.loads(self.payload_json))
        return {**body, "sha256": digest(canonical(body))}

    def to_json(self) -> str:
        return canonical(self.record())

    @property
    def sha256(self) -> str:
        return str(self.record()["sha256"])

    @property
    def window(self) -> tuple[Fraction, Fraction]:
        values = json.loads(self.payload_json)["window"]
        return exact(values[0]), exact(values[1])

    @property
    def requests(self) -> tuple[Fraction, ...]:
        return tuple(exact(t) for t in json.loads(self.payload_json)["requests"])

    @property
    def limits(self) -> StreamDecodeLimits:
        data = json.loads(self.payload_json)
        return StreamDecodeLimits.from_records(data["operation_limits"], data["delivery_limits"])

    @classmethod
    def from_json(cls, text: str, *, origin: SourceTimeOrigin, policy: str,
                  limits: StreamDecodeLimits, expected_session_id: str) -> "SourceStreamFinalization":
        if type(limits) is not StreamDecodeLimits:
            raise ValueError("Stream envelope byte/type limit")
        try:
            data, size = _bounded_json(text, limits.max_finalization_bytes)
            _stream_keys(data, "version kind scope session_id origin_id decoder_id policy payload sha256")
            effective = _stream_payload_limits(data["payload"], limits)
            if size > effective.max_finalization_bytes:
                raise ValueError("Stream presented envelope byte limit")
            _stream_guard(data, effective.max_finalization_bytes)
            _stream_session(expected_session_id)
            _stream_origin(origin, limits.max_finalization_bytes)
            if (type(data["version"]) is not int or data["version"] != 1
                    or data["kind"] != "streamed_video_window" or data["scope"] != "window"
                    or data["session_id"] != expected_session_id or data["origin_id"] != origin.identity
                    or data["decoder_id"] != origin.decoder_id or type(policy) is not str
                    or policy != STREAM_DECODE_POLICY or data["policy"] != STREAM_DECODE_POLICY):
                raise ValueError("Stream envelope identity mismatch")
            claimed = data.pop("sha256")
            if claimed != digest(canonical(data)):
                raise ValueError("Stream envelope hash mismatch")
            return cls(origin, expected_session_id, data["payload"], limits=limits)
        except (RecursionError, OverflowError, TypeError, KeyError) as exc:
            raise ValueError("Malformed stream JSON") from exc
