"""Pure mixed-track scheduling/graphs and explicit, bounded PCM measurement helpers.

Task6 must supply qualified original-PTS inputs or materialize the selected bus
with the SAME source/decoder binding. This module never chooses seek/rebase
policy, tools, assets, normalization fallbacks, or production scratch storage.
"""

from array import array
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from decimal import Decimal, localcontext
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path, PosixPath, WindowsPath
import re
import subprocess
import sys
from types import MappingProxyType
from typing import Any, Literal, Protocol, cast

from pydantic import BaseModel

from .filtergraph import _guard_known_input, _utf8_size
from .motion_compiler import _strict_model_input
from .output_timeline import OutputTimeline
from .render_assets import RenderAssetRegistry
from .render_runtime import RenderRequirements, RenderRuntime, RuntimeHashFiles, _hash, _verify_pins
from .visual_contracts import AudioMixPlan, VisualCompositionPlan, VisualShot

RATE = 48000
LIMITER = "alimiter=limit=0.944:level=0:latency=1:level_in=1:level_out=1:attack=5:release=50:asc=0"
TARGETS = "I=-14:LRA=11:TP=-1.5"
Mode = Literal["linear", "dynamic"]
_PREPARED_COLLECTION_LIMIT = 4096
_PREPARED_FRAGMENT_LIMIT = 32768
_PREPARED_ENTRY_LIMIT = 32768
_PREPARED_TEXT_LIMIT = 2_000_000
_PREPARED_GRAPH_LIMIT = 8_000_000
_PREPARED_SAMPLE_LIMIT = 8_592_000


def _join_audio_parts(parts: Sequence[str]) -> str:
    return "".join(parts)


def _audio_seconds(value: Fraction, bounded: bool) -> str:
    if bounded and (type(value) is not Fraction or
            max(value.numerator.bit_length(), value.denominator.bit_length()) > 4096):
        raise ValueError("Prepared numeric metadata bit limit")
    result = _seconds(value)
    if bounded and len(result) > 2048:
        raise ValueError("Prepared numeric token limit")
    return result


def _audio_number(value: float, bounded: bool) -> str:
    return _audio_seconds(Fraction(str(value)), bounded)


class AudioPreparationError(ValueError):
    """Typed strict preparation failure; never authorizes fallback processing."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


def quantize_time(seconds: float | Fraction) -> int:
    value = Fraction(str(seconds)) if not isinstance(seconds, Fraction) else seconds
    if value < 0:
        raise ValueError("Audio time must be nonnegative")
    value = value * RATE + Fraction(1, 2)
    return value.numerator // value.denominator


def _seconds(value: Fraction) -> str:
    # Plan decimals (and /4 ramps) terminate; never round a late source anchor.
    with localcontext() as context:
        context.prec = 400
        text = format(Decimal(value.numerator) / Decimal(value.denominator), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _number_text(value: float) -> str:
    """Preserve the accepted float's shortest decimal, without .17g artifacts."""
    return _seconds(Fraction(str(value)))


@dataclass(frozen=True, slots=True)
class AudioFormat:
    """Qualified prepared graph/measurement bus format, not raw-media support.

    Task6 qualifies original stream rate/layout and performs any source-to-48k
    conversion under its shared decoder/clock binding before asserting this.
    A different cue layout requires a separately qualified host mapping.
    """

    sample_rate: int
    layout: str

    def __post_init__(self) -> None:
        if self.sample_rate != RATE or self.layout not in ("mono", "stereo"):
            raise ValueError("Only qualified 48000 Hz mono/stereo audio layouts are supported")

    @property
    def channels(self) -> int:
        return 1 if self.layout == "mono" else 2


@dataclass(frozen=True, slots=True)
class AudioSpan:
    source_start: Fraction
    source_end: Fraction
    start: Fraction
    end: Fraction
    first: int
    last: int
    normalize: bool
    gain_db: float
    limiter: bool


@dataclass(frozen=True, slots=True)
class ScheduledCue:
    asset_id: str
    start: Fraction
    end: Fraction
    first: int
    last: int
    gain_db: float
    duck_db: float


@dataclass(frozen=True, slots=True, init=False)
class AudioSchedule:
    source_fingerprint: str
    spans: tuple[AudioSpan, ...]
    cues: tuple[ScheduledCue, ...]

    def __init__(self, source_fingerprint: str, spans: Sequence[AudioSpan],
                 cues: Sequence[ScheduledCue]) -> None:
        object.__setattr__(self, "source_fingerprint", source_fingerprint)
        object.__setattr__(self, "spans", tuple(spans))
        object.__setattr__(self, "cues", tuple(cues))

    @property
    def total_samples(self) -> int:
        return self.spans[-1].last

    @property
    def enabled_samples(self) -> int:
        return sum(s.last - s.first for s in self.spans if s.normalize)

    @property
    def identity(self) -> str:
        return hashlib.sha256(repr(self).encode()).hexdigest()


def schedule_audio(plan: VisualCompositionPlan) -> AudioSchedule:
    """Detach/revalidate and canonicalize visual-only subdivisions, without I/O."""
    plan = VisualCompositionPlan.model_validate(_strict_model_input(plan))
    OutputTimeline.from_shots(plan.shots)
    spans: list[AudioSpan] = []
    cues: list[ScheduledCue] = []
    for shot in plan.shots:
        a, b = Fraction(str(shot.output.start)), Fraction(str(shot.output.end))
        first, last = quantize_time(a), quantize_time(b)
        if last <= first:
            raise ValueError("Planned audio interval maps to zero samples")
        span = AudioSpan(Fraction(str(shot.source.start)), Fraction(str(shot.source.end)),
            a, b, first, last, shot.audio.normalize_dialogue, shot.audio.source_gain_db,
            shot.audio.limiter)
        if spans and (spans[-1].source_end == span.source_start and spans[-1].end == a
                and (spans[-1].normalize, spans[-1].gain_db, spans[-1].limiter)
                == (span.normalize, span.gain_db, span.limiter)):
            spans[-1] = replace(spans[-1], source_end=span.source_end, end=b, last=last)
        else:
            spans.append(span)
        for cue in shot.audio.cues:
            start, end = a + Fraction(str(cue.start)), a + Fraction(str(cue.end))
            if not a <= start < end <= b:
                raise ValueError("Cue must lie inside its shot")
            cfirst, clast = quantize_time(start), quantize_time(end)
            if clast <= cfirst:
                raise ValueError("Planned cue maps to zero samples")
            cues.append(ScheduledCue(cue.asset_id, start, end, cfirst, clast,
                                     cue.gain_db, cue.duck_source_db))
    return AudioSchedule(plan.source_fingerprint, tuple(spans), tuple(cues))


def duck_db(schedule: AudioSchedule, time: float) -> float:
    """Exact numeric counterpart of the per-sample FFmpeg envelope."""
    result = 0.
    for cue in schedule.cues:
        a, b = float(cue.start), float(cue.end)
        if a < time < b:
            attack, release = min(.005, (b-a)/4), min(.010, (b-a)/4)
            x = min(1., (time-a)/attack, (b-time)/release)
            result = min(result, cue.duck_db*x*x*(3-2*x))
    return result


def _duck_expression(schedule: AudioSchedule, *, max_bytes: int | None = None) -> str:
    bounded = max_bytes is not None
    if bounded and len(schedule.cues) > _PREPARED_COLLECTION_LIMIT:
        raise ValueError("Prepared cue count limit")
    opening, closing = "aeval='val(ch)*pow(10,(", ")/20)':c=same"
    size = len(opening) + 1 + len(closing)
    if max_bytes is not None and size > max_bytes:
        raise ValueError("Prepared duck graph byte limit")
    values: list[str] = []
    for cue in schedule.cues:
        a, b, d = cue.start, cue.end, cue.end-cue.start
        attack, release = min(Fraction('.005'), d/4), min(Fraction('.010'), d/4)
        # Validate derived operands too, before decimal formatting or expression joins.
        if bounded:
            _audio_seconds(d/4, True)
        xparts = ("min(1,min(max(0,(n/48000-", _audio_seconds(a, bounded), ")/",
            _audio_seconds(attack, bounded), "),max(0,(", _audio_seconds(b, bounded),
            "-n/48000)/", _audio_seconds(release, bounded), ")))")
        db = _audio_number(cue.duck_db, bounded)
        xsize = sum(map(len, xparts))
        value_size = 3*xsize + len(db) + len("()*()*()*(3-2*())")
        size += 6 + value_size
        if max_bytes is not None and size > max_bytes:
            raise ValueError("Prepared duck graph byte limit")
        x = _join_audio_parts(xparts)
        values.append(_join_audio_parts(("(", db, ")*(", x, ")*(", x, ")*(3-2*(", x, "))")))
    # Flat construction preserves the exact old left-nested min reduction bytes.
    parts = [opening, *["min("] * len(values), "0"]
    for value in values:
        parts.extend((",", value, ")"))
    parts.append(closing)
    return _join_audio_parts(parts)


@dataclass(frozen=True, slots=True)
class SourceClockBinding:
    source_fingerprint: str
    decoder_identity: str
    selected_pcm_sha256: str

    def __post_init__(self) -> None:
        if (not self.source_fingerprint or not self.decoder_identity
                or not re.fullmatch(r"[a-f0-9]{64}", self.selected_pcm_sha256)):
            raise ValueError("Explicit source/decoder binding and selected PCM SHA256 required")


@dataclass(frozen=True, slots=True)
class LoudnessStats:
    input_i: float
    input_lra: float
    input_tp: float
    input_thresh: float
    target_offset: float


def parse_loudness(log: str) -> tuple[LoudnessStats, str]:
    """Parse exactly one loudnorm observation; never coerce -inf into statistics."""
    blocks = re.findall(r"\{[^{}]*\}", log)
    if len(blocks) != 1:
        raise AudioPreparationError("malformed_measurement", "Expected one loudnorm JSON observation")
    try:
        payload = json.loads(blocks[0])
        fields = ("input_i", "input_lra", "input_tp", "input_thresh", "target_offset")
        if any(isinstance(payload[k], bool) for k in fields):
            raise ValueError("Boolean is not a measured statistic")
        values = [float(payload[k]) for k in fields]
        if not all(math.isfinite(v) for v in values):
            raise AudioPreparationError("undefined_measurement", "Below-gate/nonfinite loudness; not proof of silence")
        stats = LoudnessStats(*values)
        select_normalization_mode(stats)
        return stats, str(payload["normalization_type"]).lower()
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, AudioPreparationError):
            raise
        raise AudioPreparationError("malformed_measurement", "Invalid loudnorm statistics") from exc


def select_normalization_mode(stats: LoudnessStats) -> tuple[Mode, str]:
    fields = (stats.input_i, stats.input_lra, stats.input_tp, stats.input_thresh, stats.target_offset)
    limits = ((-99, 0), (0, 99), (-99, 99), (-99, 0), (-99, 99))
    if any(not math.isfinite(v) or not low <= v <= high
           for v, (low, high) in zip(fields, limits, strict=True)):
        raise AudioPreparationError("malformed_measurement", "Statistics outside measured filter ranges")
    if stats.input_lra <= 11 and stats.input_tp + (-14-stats.input_i) <= -1.5:
        return "linear", "measured LRA and projected peak permit linear scaling"
    return "dynamic", "measured LRA or projected peak requires dynamic processing"


def _runtime_identity(runtime: RenderRuntime) -> tuple[str, ...]:
    return (runtime.runtime_id, str(runtime.ffmpeg), str(runtime.ffprobe), runtime.ffmpeg_sha256,
            runtime.ffprobe_sha256, str(runtime.expected_ffmpeg_sha256),
            str(runtime.expected_ffprobe_sha256), runtime.graph_file_option, str(runtime.device_index))


@dataclass(frozen=True, slots=True)
class NormalizationReceipt:
    schedule_identity: str
    binding: SourceClockBinding
    runtime_identity: tuple[str, ...]
    audio_format: AudioFormat
    samples: int
    stats: LoudnessStats
    mode: Mode
    mode_reason: str
    scope: str = "mixed_track_normalization"

    def __post_init__(self) -> None:
        object.__setattr__(self, "runtime_identity", tuple(self.runtime_identity))


def _validate_receipt(schedule: AudioSchedule, receipt: NormalizationReceipt,
                      binding: SourceClockBinding, fmt: AudioFormat, runtime: RenderRuntime) -> None:
    if (receipt.schedule_identity != schedule.identity or receipt.binding != binding
            or binding.source_fingerprint != schedule.source_fingerprint
            or receipt.runtime_identity != _runtime_identity(runtime)
            or receipt.audio_format != fmt or receipt.samples != schedule.enabled_samples):
        raise AudioPreparationError("identity_mismatch", "Measurement source/schedule/binding/runtime identity mismatch")
    if (receipt.mode, receipt.mode_reason) != select_normalization_mode(receipt.stats):
        raise AudioPreparationError("mode_mismatch", "Receipt mode selection differs from measured statistics")


def normalization_filter(receipt: NormalizationReceipt) -> str:
    s = receipt.stats
    return (f"loudnorm={TARGETS}:measured_I={_number_text(s.input_i)}:measured_LRA={_number_text(s.input_lra)}:"
            f"measured_TP={_number_text(s.input_tp)}:measured_thresh={_number_text(s.input_thresh)}:"
            f"offset={_number_text(s.target_offset)}:"
            f"linear={'true' if receipt.mode == 'linear' else 'false'}:print_format=json,aresample={RATE}")


def verify_normalization_observation(receipt: NormalizationReceipt, observed_mode: str,
                                     samples: int) -> None:
    if observed_mode != receipt.mode:
        raise AudioPreparationError("mode_mismatch", f"Expected {receipt.mode}; observed {observed_mode}")
    if samples != receipt.samples:
        raise AudioPreparationError("sample_count_mismatch", f"Expected {receipt.samples}; observed {samples}")


def _validate_pcm(schedule: AudioSchedule, data: bytes, binding: SourceClockBinding,
                  fmt: AudioFormat, runtime: RenderRuntime, *,
                  _runtime_files: RuntimeHashFiles | None = None) -> None:
    if binding.source_fingerprint != schedule.source_fingerprint or hashlib.sha256(data).hexdigest() != binding.selected_pcm_sha256:
        raise AudioPreparationError("identity_mismatch", "Selected PCM/source identity mismatch")
    if len(data) != schedule.enabled_samples * fmt.channels * 4:
        raise AudioPreparationError("sample_count_mismatch", "Enabled bus PCM length differs")
    if _runtime_files is None:
        _verify_pins(runtime)
        pair = (_hash(runtime.ffmpeg), _hash(runtime.ffprobe))
    else:
        _verify_pins(runtime, _files=_runtime_files)
        pair = (_hash(runtime.ffmpeg, _files=_runtime_files),
                _hash(runtime.ffprobe, _files=_runtime_files))
    if pair != (runtime.ffmpeg_sha256, runtime.ffprobe_sha256):
        raise AudioPreparationError("identity_mismatch", "Paired runtime bytes changed")


class PCMRunner(Protocol):
    def __call__(self, argv: tuple[str, ...], data: bytes, *,
                 measure: bool) -> subprocess.CompletedProcess[bytes]: ...


def _run_pcm(runtime: RenderRuntime, data: bytes, fmt: AudioFormat, filters: str,
             *, measure: bool = False,
             _runner: PCMRunner | None = None) -> subprocess.CompletedProcess[bytes]:
    args = [str(runtime.ffmpeg), "-hide_banner", "-loglevel", "info", "-f", "f32le",
            "-ar", str(RATE), "-ac", str(fmt.channels), "-channel_layout", fmt.layout,
            "-i", "pipe:0", "-af", filters]
    args += ["-f", "null", "-"] if measure else ["-c:a", "pcm_f32le", "-f", "f32le", "pipe:1"]
    try:
        result = (subprocess.run(args, input=data, capture_output=True, timeout=45, check=False)
                  if _runner is None else _runner(tuple(args), data, measure=measure))
    except (OSError, subprocess.SubprocessError) as exc:
        raise AudioPreparationError("tool_failure", str(exc)) from exc
    if result.returncode:
        raise AudioPreparationError("tool_failure", result.stderr.decode(errors="replace"))
    return result


def measure_normalization(schedule: AudioSchedule, pcm: bytes, *, binding: SourceClockBinding,
                          audio_format: AudioFormat, runtime: RenderRuntime,
                          _runner: PCMRunner | None = None,
                          _runtime_files: RuntimeHashFiles | None = None) -> NormalizationReceipt:
    """Explicit HOST operation on the complete enabled f32le bus; no source seeking."""
    if _runtime_files is None:
        _validate_pcm(schedule, pcm, binding, audio_format, runtime)
    else:
        _validate_pcm(schedule, pcm, binding, audio_format, runtime, _runtime_files=_runtime_files)
    if schedule.enabled_samples < 19200:
        raise AudioPreparationError("insufficient_duration", "Enabled bus is shorter than 400ms")
    values = array("f")
    values.frombytes(pcm)
    if sys.byteorder != "little":
        values.byteswap()
    if not all(math.isfinite(v) for v in values):
        raise AudioPreparationError("malformed_pcm", "Nonfinite PCM")
    if not any(v != 0 for v in values):
        raise AudioPreparationError("digital_silence", "Independent complete-bus all-zero check")
    result = _run_pcm(runtime, pcm, audio_format, f"loudnorm={TARGETS}:print_format=json",
                      measure=True, _runner=_runner)
    stats, _ = parse_loudness(result.stderr.decode(errors="replace"))
    mode, reason = select_normalization_mode(stats)
    return NormalizationReceipt(schedule.identity, binding, _runtime_identity(runtime), audio_format,
                                 schedule.enabled_samples, stats, mode, reason)


@dataclass(frozen=True, slots=True)
class NormalizedPCM:
    pcm: bytes
    observed_mode: str
    samples: int
    upstream_observation: LoudnessStats
    output_loudness: tuple[float, float, float]


def process_normalized_pcm(schedule: AudioSchedule, pcm: bytes, *, receipt: NormalizationReceipt,
                           binding: SourceClockBinding, audio_format: AudioFormat,
                           runtime: RenderRuntime, _runner: PCMRunner | None = None,
                           _runtime_files: RuntimeHashFiles | None = None) -> NormalizedPCM:
    _validate_receipt(schedule, receipt, binding, audio_format, runtime)
    if _runtime_files is None:
        _validate_pcm(schedule, pcm, binding, audio_format, runtime)
    else:
        _validate_pcm(schedule, pcm, binding, audio_format, runtime, _runtime_files=_runtime_files)
    result = _run_pcm(runtime, pcm, audio_format, normalization_filter(receipt), _runner=_runner)
    stats, mode = parse_loudness(result.stderr.decode(errors="replace"))
    samples, remainder = divmod(len(result.stdout), 4*audio_format.channels)
    if remainder:
        raise AudioPreparationError("sample_count_mismatch", "Incomplete PCM sample")
    verify_normalization_observation(receipt, mode, samples)
    try:
        payload = json.loads(re.findall(r"\{[^{}]*\}", result.stderr.decode(errors="replace"))[0])
        fields = ("output_i", "output_lra", "output_tp")
        if any(isinstance(payload[k], bool) for k in fields):
            raise ValueError("Boolean is not a measured statistic")
        output = tuple(float(payload[k]) for k in fields)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise AudioPreparationError("malformed_measurement", "Invalid loudnorm output statistics") from exc
    if not all(math.isfinite(v) for v in output):
        raise AudioPreparationError("undefined_measurement", "Nonfinite upstream normalized output observation")
    return NormalizedPCM(result.stdout, mode, samples, stats, (output[0], output[1], output[2]))


@dataclass(frozen=True, slots=True)
class CompiledAudio:
    schedule: AudioSchedule
    filter_complex: str
    output_label: str
    cue_inputs: tuple[tuple[str, Path, str], ...]
    requirements: RenderRequirements
    receipt: NormalizationReceipt | None
    normalization_scope: str = "mixed_track_normalization"


class _AudioGraph:
    """Charge fragments and separators before any prepared graph join/retention."""

    def __init__(self, limit: int | None = None) -> None:
        self.limit = limit
        self.parts: list[str] = []
        self.size = 0

    def remaining(self) -> int | None:
        return None if self.limit is None else self.limit - self.size - bool(self.parts)

    def check(self, size: int) -> None:
        if self.limit is not None and (len(self.parts) >= _PREPARED_FRAGMENT_LIMIT
                or size > cast(int, self.remaining())):
            raise ValueError("Prepared graph byte/fragment limit")

    def append_parts(self, parts: tuple[str, ...]) -> None:
        if self.limit is not None:
            if type(parts) is not tuple or len(parts) > 32768:
                raise ValueError("Prepared fragment parts limit")
            size = 0
            for part in parts:
                size += _utf8_size(part, self.limit - size)
            self.check(size)
        else:
            size = 0
        self.parts.append(_join_audio_parts(parts))
        self.size += size + (1 if len(self.parts) > 1 else 0)

    def finish(self) -> str:
        return ";".join(self.parts)


def _audio_labels(graph: _AudioGraph, labels: Sequence[str], *, maximum: int = 4096) -> None:
    if graph.limit is not None and (not 1 <= len(labels) <= maximum
            or any(type(label) is not str or len(label) > 64
                or not re.fullmatch(r"[A-Za-z0-9_:]+", label) for label in labels)):
        raise ValueError("Prepared internal label/count limit")


def _audio_split(graph: _AudioGraph, label: str, count: int, prefix: str) -> list[str]:
    if graph.limit is not None and (type(count) is not int or not 1 <= count <= 4096
            or type(prefix) is not str or len(prefix) > 24
            or not re.fullmatch(r"[A-Za-z0-9_]+", prefix)):
        raise ValueError("Prepared split count/prefix limit")
    _audio_labels(graph, [label])
    labels = [f"{prefix}{i}" for i in range(count)]
    _audio_labels(graph, labels)
    operation = f"asplit={count}" if count > 1 else "anull"
    graph.check(len(label) + 2 + len(operation) + sum(len(v) + 2 for v in labels))
    graph.append_parts(("[", label, "]", operation, *(part for v in labels for part in ("[", v, "]"))))
    return labels


def _audio_concat(graph: _AudioGraph, labels: list[str], output: str) -> None:
    _audio_labels(graph, labels)
    _audio_labels(graph, [output])
    operation = f"concat=n={len(labels)}:v=0:a=1" if len(labels) > 1 else "anull"
    graph.check(sum(len(v) + 2 for v in labels) + len(operation) + len(output) + 2)
    graph.append_parts((*(part for v in labels for part in ("[", v, "]")), operation, "[", output, "]"))


def _append_audio_tail(
    graph: _AudioGraph, filters: set[str], schedule: AudioSchedule, *,
    bypass_labels: tuple[str | None, ...], normalized_label: str | None,
    audio_format: AudioFormat, assets: RenderAssetRegistry | None,
    cue_formats: Mapping[str, AudioFormat] | None, cue_input_start: int,
) -> tuple[tuple[str, Path, str], ...]:
    bounded = graph.limit is not None
    if normalized_label is not None:
        normparts = iter(_audio_split(graph, normalized_label,
            sum(s.normalize for s in schedule.spans), "normpart"))
        offset = 0
        for i, span in enumerate(schedule.spans):
            if span.normalize:
                end = offset + span.last-span.first
                graph.append_parts(("[", next(normparts), "]atrim=start_sample=", str(offset),
                    ":end_sample=", str(end), ",asetpts=N/48000/TB[restore", str(i), "]"))
                offset = end
    for i, span in enumerate(schedule.spans):
        label = f"restore{i}" if span.normalize else bypass_labels[i]
        assert label is not None
        graph.append_parts(("[", label, "]volume=", _audio_number(span.gain_db, bounded), "dB[gain", str(i), "]"))
    _audio_concat(graph, [f"gain{i}" for i in range(len(schedule.spans))], "gained")
    cue_inputs: list[tuple[str, Path, str]] = []
    if schedule.cues:
        remaining = graph.remaining()
        expression = _duck_expression(schedule, max_bytes=None if remaining is None
            else remaining - len("[gained][ducked]"))
        graph.append_parts(("[gained]", expression, "[ducked]"))
        filters.update(("aeval", "adelay", "apad", "amix"))
        for asset_id in dict.fromkeys(c.asset_id for c in schedule.cues):
            if assets is None or asset_id not in assets.sfx:
                raise ValueError("Cue requires a registered sfx asset ID")
            if cue_formats is None or cue_formats.get(asset_id) != audio_format:
                raise ValueError("Cue/source layout and sample rate require explicit qualified matching formats")
            label = f"{cue_input_start+len(cue_inputs)}:a"
            cue_inputs.append((asset_id, assets.resolve("sfx", asset_id), label))
            occurrences = [i for i, c in enumerate(schedule.cues) if c.asset_id == asset_id]
            labels = _audio_split(graph, label, len(occurrences), f"cueasset{len(cue_inputs)}_")
            for i, branch in zip(occurrences, labels, strict=True):
                cue = schedule.cues[i]
                count = str(cue.last-cue.first)
                graph.append_parts(("[", branch, "]atrim=end_sample=", count, ",asetpts=N/48000/TB,",
                    "apad=whole_len=", count, ",atrim=end_sample=", count, ",volume=",
                    _audio_number(cue.gain_db, bounded), "dB,adelay=", str(cue.first), "S:all=1[cue", str(i), "]"))
        labels = ["ducked", *(f"cue{i}" for i in range(len(schedule.cues)))]
        _audio_labels(graph, labels, maximum=4097)
        operation = ("amix=inputs=", str(len(labels)), ":duration=first:weights='1':",
                     "normalize=0:dropout_transition=0[mixed]")
        graph.check(sum(len(v) + 2 for v in labels) + sum(map(len, operation)))
        graph.append_parts((*(part for v in labels for part in ("[", v, "]")), *operation))
    else:
        graph.append_parts(("[gained]anull[mixed]",))
    # Maximal limiter runs ignore gain changes, cues, normalization, visual splits.
    runs: list[tuple[int, int, bool]] = []
    for span in schedule.spans:
        if runs and runs[-1][2] == span.limiter:
            runs[-1] = runs[-1][0], span.last, span.limiter
        else:
            runs.append((span.first, span.last, span.limiter))
    labels = _audio_split(graph, "mixed", len(runs), "limpart")
    for i, ((start, end, limited), label) in enumerate(zip(runs, labels, strict=True)):
        graph.append_parts(("[", label, "]atrim=start_sample=", str(start), ":end_sample=", str(end),
            ",asetpts=N/48000/TB,", LIMITER+"," if limited else "", "anull[limited", str(i), "]"))
        if limited:
            filters.add("alimiter")
    _audio_concat(graph, [f"limited{i}" for i in range(len(runs))], "finalbus")
    graph.append_parts(("[finalbus]atrim=end_sample=", str(schedule.total_samples),
                        ",asetpts=N/48000/TB[audio_out]"))
    return tuple(cue_inputs)


def _prepared_hash(value: object) -> None:
    if type(value) is not str or len(value) != 64 or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("Prepared identity requires lowercase SHA256 text")


def _prepared_fields(value: object, kind: type) -> bool:
    return all(hasattr(value, field.name) for field in fields(cast(Any, kind)))


def _prepared_format(value: object) -> None:
    if (type(value) is not AudioFormat or not _prepared_fields(value, AudioFormat)
            or type(value.sample_rate) is not int
            or value.sample_rate != RATE or type(value.layout) is not str
            or value.layout not in ("mono", "stereo")):
        raise ValueError("Prepared format requires exact 48000 Hz mono/stereo metadata")


def _prepared_binding(value: object) -> None:
    if (type(value) is not SourceClockBinding or not _prepared_fields(value, SourceClockBinding)
            or type(value.source_fingerprint) is not str or not value.source_fingerprint
            or type(value.decoder_identity) is not str or not value.decoder_identity):
        raise ValueError("Prepared source/decoder binding required")
    _prepared_hash(value.selected_pcm_sha256)


def _prepared_runtime(value: object) -> None:
    if type(value) is not RenderRuntime or not _prepared_fields(value, RenderRuntime):
        raise ValueError("Prepared runtime record required")
    if any(type(path) not in (WindowsPath, PosixPath) for path in (value.ffmpeg, value.ffprobe)):
        raise ValueError("Prepared runtime requires concrete Path metadata")
    if (any(type(text) is not str for text in (value.runtime_id, value.ffmpeg_sha256,
            value.ffprobe_sha256, value.graph_file_option))
            or any(text is not None and type(text) is not str for text in
                (value.expected_ffmpeg_sha256, value.expected_ffprobe_sha256))
            or value.device_index is not None and type(value.device_index) is not int):
        raise ValueError("Prepared runtime scalar metadata shape")


def _prepared_receipt(value: object) -> None:
    if type(value) is not NormalizationReceipt or not _prepared_fields(value, NormalizationReceipt):
        raise ValueError("Prepared normalization receipt record required")
    _prepared_hash(value.schedule_identity)
    _prepared_binding(value.binding)
    _prepared_format(value.audio_format)
    if (type(value.runtime_identity) is not tuple or len(value.runtime_identity) != 9
            or any(type(text) is not str for text in value.runtime_identity)
            or type(value.samples) is not int or not 1 <= value.samples <= _PREPARED_SAMPLE_LIMIT
            or type(value.stats) is not LoudnessStats or not _prepared_fields(value.stats, LoudnessStats)
            or type(value.mode) is not str or value.mode not in ("linear", "dynamic")
            or type(value.mode_reason) is not str or type(value.scope) is not str
            or value.scope != "mixed_track_normalization"):
        raise ValueError("Prepared receipt metadata shape/count/scope")
    values = (value.stats.input_i, value.stats.input_lra, value.stats.input_tp,
              value.stats.input_thresh, value.stats.target_offset)
    if any(type(v) not in (int, float) for v in values):
        raise ValueError("Prepared receipt requires plain finite statistics")
    # Guard integer bits before the existing selector's float conversion/arithmetic.
    _guard_known_input(values, text_limit=0)
    limits = ((-99, 0), (0, 99), (-99, 99), (-99, 0), (-99, 99))
    if any(not low <= v <= high for v, (low, high) in zip(values, limits, strict=True)):
        raise AudioPreparationError("malformed_measurement", "Statistics outside measured filter ranges")
    select_normalization_mode(value.stats)


def _prepared_label(value: object) -> None:
    if value is not None and (type(value) is not str or len(value) > 7
            or not re.fullmatch(r"(?:0|[1-9][0-9]{0,4}):a", value)
            or int(value[:-2]) > 32767):
        raise ValueError("Prepared input label must be canonical N:a with N <= 32767")


def _prepared_entry_count(value: object, remaining: int) -> int:
    """Count entries of already guarded records without cloning their children."""
    count = 0
    children: Any = ()
    if type(value) in (WindowsPath, PosixPath):
        count = len(cast(Path, value).parts)
    elif type(value) in (list, tuple):
        children = value
        count = len(children)
    elif isinstance(value, BaseModel):
        children = value.__dict__.values()
    elif type(value) in (AudioFormat, SourceClockBinding, LoudnessStats, RenderRuntime, NormalizationReceipt):
        children = (getattr(value, field.name) for field in fields(cast(Any, value)))
    if count > remaining:
        raise ValueError("Prepared metadata entry limit")
    for child in children:
        count += _prepared_entry_count(child, remaining-count)
    return count


def _prepared_table(value: object, kind: Literal["format", "path"], budget: list[int]) -> dict[str, Any]:
    if type(value) not in (dict, MappingProxyType):
        raise ValueError("Prepared table requires builtin dict or mapping proxy")
    table = cast(Mapping[str, Any], value)
    view: Any = dict.items(cast(dict[str, Any], table)) if type(value) is dict else table.items()
    if type(view) is not type({}.items()):
        raise ValueError("Prepared table requires builtin dict_items before traversal")
    initial_size = len(view)
    if initial_size > _PREPARED_COLLECTION_LIMIT:
        raise ValueError("Prepared table entry limit")
    snapshot: dict[str, Any] = {}
    consumed = 0
    item: Any
    try:
        for item in view:
            if consumed >= initial_size or consumed >= _PREPARED_COLLECTION_LIMIT:
                raise ValueError("Prepared table consumed count limit")
            consumed += 1
            if type(item) is not tuple or len(item) != 2:
                raise ValueError("Prepared table item shape")
            key, child = item
            if type(key) is not str or len(key) > 64 or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", key):
                raise ValueError("Prepared table requires safe asset IDs")
            if key in snapshot:
                raise ValueError("Prepared table duplicate asset ID")
            if kind == "format":
                _prepared_format(child)
            elif type(child) not in (WindowsPath, PosixPath):
                raise ValueError("Prepared sfx requires concrete Path metadata")
            # One table edge + the item pair's key/value entries. Count concrete
            # Path components against the SAME input budget before any retention.
            remaining = _PREPARED_ENTRY_LIMIT-budget[1]-3
            if remaining < 0:
                raise ValueError("Prepared metadata entry limit")
            budget[1] += 3 + _prepared_entry_count(child, remaining)
            budget[0] += _utf8_size(key, _PREPARED_TEXT_LIMIT-budget[0])
            budget[0] += _guard_known_input(child, record_types=(AudioFormat,),
                text_limit=_PREPARED_TEXT_LIMIT-budget[0], entry_limit=remaining)
            snapshot[key] = child
        if consumed != initial_size or len(view) != initial_size:
            raise ValueError("Prepared table changed during snapshot")
    except RuntimeError as exc:
        raise ValueError("Prepared table mutated during traversal") from exc
    return snapshot


def compile_prepared_audio(
    plan: VisualCompositionPlan, *, schedule_identity: str, audio_format: AudioFormat,
    selected_input_label: str | None = None, normalized_input_label: str | None = None,
    assets: RenderAssetRegistry | None = None, cue_formats: Mapping[str, AudioFormat] | None = None,
    binding: SourceClockBinding | None = None, runtime: RenderRuntime | None = None,
    receipt: NormalizationReceipt | None = None, cue_input_start: int = 1,
) -> CompiledAudio:
    """Pure prepared-bus graph, sharing the original compiler's effect tail.

    These checks establish declaration/receipt consistency only. HOST must bind
    actual selected/normalized/cue bytes, counts, layout and opened input indices.
    selected_pcm_sha256 continues to identify the complete ENABLED input material.
    """
    if type(plan) is not VisualCompositionPlan:
        raise ValueError("Exact visual composition plan required")
    shots: Any = plan.__dict__.get("shots")
    if type(shots) not in (list, tuple) or not 1 <= len(shots) <= _PREPARED_COLLECTION_LIMIT:
        raise ValueError("Prepared shot collection limit/shape")
    cues = 0
    for shot in shots:
        if type(shot) is not VisualShot or type(shot.__dict__.get("audio")) is not AudioMixPlan:
            raise ValueError("Prepared exact shot/audio records required")
        children: Any = shot.audio.__dict__.get("cues")
        if type(children) not in (list, tuple) or len(children) > _PREPARED_COLLECTION_LIMIT-cues:
            raise ValueError("Prepared cue collection limit/shape")
        cues += len(children)
    # Reject unknown shapes/cycles and large metadata before strict model recursion.
    _guard_known_input(plan, text_limit=_PREPARED_TEXT_LIMIT, entry_limit=_PREPARED_ENTRY_LIMIT)
    _prepared_hash(schedule_identity)
    _prepared_label(selected_input_label)
    _prepared_label(normalized_input_label)
    if type(cue_input_start) is not int or not 1 <= cue_input_start <= 32767:
        raise ValueError("Prepared cue input index requires plain int 1..32767")
    _prepared_format(audio_format)
    if binding is not None:
        _prepared_binding(binding)
    if runtime is not None:
        _prepared_runtime(runtime)
    if receipt is not None:
        _prepared_receipt(receipt)
    records = (AudioFormat, SourceClockBinding, LoudnessStats, RenderRuntime, NormalizationReceipt)
    inputs = (plan, schedule_identity, audio_format, selected_input_label, normalized_input_label,
              binding, runtime, receipt, cue_input_start)
    # Three aggregate container slots (inputs/formats/sfx), then actual entries
    # of validated input records. No aggregate table-entry tuples are cloned.
    budget = [_guard_known_input(inputs, record_types=records,
        text_limit=_PREPARED_TEXT_LIMIT, entry_limit=_PREPARED_ENTRY_LIMIT-3),
        3 + _prepared_entry_count(inputs, _PREPARED_ENTRY_LIMIT-3)]
    formats = _prepared_table(cue_formats, "format", budget) if cue_formats is not None else None
    sfx = None
    if assets is not None:
        if type(assets) is not RenderAssetRegistry or not hasattr(assets, "sfx"):
            raise ValueError("Prepared exact asset registry required")
        sfx = _prepared_table(assets.sfx, "path", budget)
    safe_assets = RenderAssetRegistry({}, {}, sfx) if sfx is not None else None
    schedule = schedule_audio(plan)
    if schedule.total_samples > _PREPARED_SAMPLE_LIMIT:
        raise ValueError("Prepared selected sample limit")
    if schedule.identity != schedule_identity:
        raise AudioPreparationError("identity_mismatch", "Prepared schedule identity mismatch")
    enabled = schedule.enabled_samples
    disabled = enabled < schedule.total_samples
    if (selected_input_label is not None) != disabled:
        raise ValueError("Selected input required iff disabled spans exist")
    if enabled:
        if receipt is None or binding is None or runtime is None or normalized_input_label is None:
            raise AudioPreparationError("measurement_required", "Matching measured receipt/binding/runtime/normalized input required")
        _validate_receipt(schedule, receipt, binding, audio_format, runtime)
    elif receipt is not None or binding is not None or normalized_input_label is not None:
        raise AudioPreparationError("identity_mismatch", "Disabled authority takes no normalization artifacts")
    bus_labels = [v for v in (selected_input_label, normalized_input_label) if v is not None]
    cue_ids = tuple(dict.fromkeys(c.asset_id for c in schedule.cues))
    cue_count = len(cue_ids)
    if cue_count and cue_input_start + cue_count - 1 > 32767:
        raise ValueError("Prepared cue input index limit")
    cue_labels = [f"{cue_input_start+i}:a" for i in range(cue_count)]
    if len(set(bus_labels + cue_labels)) != len(bus_labels) + cue_count:
        raise ValueError("Prepared audio input labels collide")
    for asset_id in cue_ids:
        if sfx is None or asset_id not in sfx:
            raise ValueError("Cue requires a registered sfx asset ID")
        if formats is None or formats.get(asset_id) != audio_format:
            raise ValueError("Cue/source layout and sample rate require explicit qualified matching formats")
    graph = _AudioGraph(limit=_PREPARED_GRAPH_LIMIT)
    filters = {"atrim", "asetpts", "asplit", "concat", "volume", "anull"}
    bypass: list[str | None] = [None] * len(schedule.spans)
    if disabled:
        assert selected_input_label is not None
        labels = iter(_audio_split(graph, selected_input_label,
            sum(not s.normalize for s in schedule.spans), "bypass"))
        for i, span in enumerate(schedule.spans):
            if not span.normalize:
                bypass[i] = f"raw{i}"
                graph.append_parts(("[", next(labels), "]atrim=start_sample=", str(span.first),
                    ":end_sample=", str(span.last), ",asetpts=N/48000/TB[", cast(str, bypass[i]), "]"))
    cue_inputs = _append_audio_tail(graph, filters, schedule, bypass_labels=tuple(bypass),
        normalized_label=normalized_input_label, audio_format=audio_format,
        assets=safe_assets, cue_formats=formats, cue_input_start=cue_input_start)
    return CompiledAudio(schedule, graph.finish(), "audio_out", cue_inputs,
        RenderRequirements(required_filters=tuple(filters)), receipt)


def compile_audio(plan: VisualCompositionPlan, *, audio_format: AudioFormat,
                  assets: RenderAssetRegistry | None = None,
                  cue_formats: Mapping[str, AudioFormat] | None = None,
                  binding: SourceClockBinding | None = None, runtime: RenderRuntime | None = None,
                  receipt: NormalizationReceipt | None = None, input_label: str = "0:a",
                  cue_input_start: int = 1) -> CompiledAudio:
    """Pure original-PTS graph. Host must verify observed mode/count before acceptance.

    Source inputs require complete qualified coverage; no source apad/async heal.
    audio_format describes the qualified 48k graph bus. Original stream metadata,
    resampling phase and original-PTS/selected-PCM binding remain Task6 authority.
    Labels are explicit host graph references, never director-provided strings.
    """
    schedule = schedule_audio(plan)
    if not re.fullmatch(r"[A-Za-z0-9_:]+", input_label) or type(cue_input_start) is not int or cue_input_start < 1:
        raise ValueError("Invalid host audio input labels")
    if schedule.enabled_samples:
        if receipt is None or binding is None or runtime is None:
            raise AudioPreparationError("measurement_required", "Matching measured normalization receipt/binding/runtime required")
        _validate_receipt(schedule, receipt, binding, audio_format, runtime)
    elif receipt is not None:
        raise AudioPreparationError("identity_mismatch", "Disabled authority takes no receipt")
    graph = _AudioGraph()
    filters = {"atrim", "asetpts", "asplit", "concat", "volume", "aformat", "anull"}
    source_labels = _audio_split(graph, input_label, len(schedule.spans), "src")
    for i, (span, label) in enumerate(zip(schedule.spans, source_labels, strict=True)):
        graph.append_parts((f"[{label}]atrim=start={_seconds(span.source_start)}:end={_seconds(span.source_end)},"
                     f"asetpts=PTS-{_seconds(span.source_start)}/TB,aresample={RATE},"
                     f"aformat=sample_fmts=fltp:sample_rates={RATE}:channel_layouts={audio_format.layout},"
                     f"atrim=end_sample={span.last-span.first}[cut{i}]",))
    filters.add("aresample")
    # Selected output is split once; enabled pieces share ONE normalization bus.
    _audio_concat(graph, [f"cut{i}" for i in range(len(schedule.spans))], "selected")
    pieces = _audio_split(graph, "selected", len(schedule.spans), "part")
    enabled: list[str] = []
    for i, (span, label) in enumerate(zip(schedule.spans, pieces, strict=True)):
        graph.append_parts((f"[{label}]atrim=start_sample={span.first}:end_sample={span.last},"
                     f"asetpts=N/{RATE}/TB[raw{i}]",))
        if span.normalize:
            enabled.append(f"raw{i}")
    if enabled:
        assert receipt is not None
        _audio_concat(graph, enabled, "normbus")
        graph.append_parts((f"[normbus]{normalization_filter(receipt)}[normalized]",))
        filters.add("loudnorm")
    cue_inputs = _append_audio_tail(graph, filters, schedule,
        bypass_labels=tuple(None if s.normalize else f"raw{i}" for i, s in enumerate(schedule.spans)),
        normalized_label="normalized" if enabled else None, audio_format=audio_format,
        assets=assets, cue_formats=cue_formats, cue_input_start=cue_input_start)
    return CompiledAudio(schedule, graph.finish(), "audio_out", cue_inputs,
                         RenderRequirements(required_filters=tuple(filters)), receipt)
