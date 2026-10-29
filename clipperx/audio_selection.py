"""Pure occurrence planning and exact interval selection of observed float PCM.

No execution or source access: the caller supplies one real42 observation per
continuous group. Hashes express consistency, never authenticated execution.
"""

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from enum import Enum
from fractions import Fraction
import math
import re
from typing import Any

from .audio_mix import (
    AudioFormat, AudioPreparationError, AudioSchedule, AudioSpan, ScheduledCue,
    quantize_time, schedule_audio,
)
from .motion_compiler import _strict_model_input
from .output_timeline import OutputTimeline
from .render_clock import (
    SourceFloatAudioDecodeReceipt, SourceTimeOrigin, canonical, digest, rational,
    StreamClock, _bounded_json, _float_origin, _float_runtime, _stream_fraction,
    _stream_guard, _stream_int, _stream_keys, _stream_rational, source_decoder_identity,
    validate_float_audio_limits,
)
from .source_decode import DecodedFloatAudioWindow, FloatAudioDecodeLimits
from .planner_contracts import TimeRange
from .visual_contracts import (
    Anchor, AudioCue, AudioMixPlan, CropKeyframe, FillMode, LayerKind, NormalizedRect,
    OverlayPlan, SubtitlePlan, VisualCompositionPlan, VisualLayer, VisualShot,
)

POLICY = "float48k;half-open-interval;group-v1"
RATE = 48000


@dataclass(frozen=True, slots=True, kw_only=True)
class AudioSelectionLimits:
    max_shots: int = 4096
    max_cues: int = 4096
    max_groups: int = 4096
    max_runs: int = 8192
    max_plan_bytes: int = 2_000_000
    max_metadata_bytes: int = 2_000_000
    max_observed_bytes: int = 72_000_000
    max_selected_samples: int = 8_592_000
    max_selected_bytes: int = 68_736_000


@dataclass(frozen=True, slots=True, kw_only=True)
class AudioSelectionSpan:
    shot_index: int
    shot_id: str
    schedule_span_index: int
    group_index: int
    source_start: Fraction
    source_end: Fraction
    start: Fraction
    end: Fraction
    first: int
    last: int


@dataclass(frozen=True, slots=True, kw_only=True)
class AudioSelectionGroup:
    index: int
    shot_first: int
    shot_last: int
    schedule_first: int
    schedule_last: int
    source_start: Fraction
    source_end: Fraction
    start: Fraction
    end: Fraction
    first: int
    last: int


@dataclass(frozen=True, slots=True, kw_only=True)
class FloatAudioSelectionPlan:
    policy: str
    origin_id: str
    origin_decoder_id: str
    runtime_json: str
    schedule: AudioSchedule
    schedule_id: str
    audio_format: AudioFormat
    spans: tuple[AudioSelectionSpan, ...]
    groups: tuple[AudioSelectionGroup, ...]
    total_samples: int
    limits: AudioSelectionLimits

    def record(self) -> dict[str, object]:
        try:
            return _plan_record(self)
        except AudioPreparationError:
            raise
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, OverflowError) as exc:
            raise AudioPreparationError("SELECTION_INVALID", "Malformed selection projection") from exc

    @property
    def sha256(self) -> str:
        return digest(canonical(self.record()))


@dataclass(frozen=True, slots=True, kw_only=True)
class AudioSelectionRun:
    first: int
    last: int
    source_first: int
    residual: Fraction


@dataclass(frozen=True, slots=True, kw_only=True)
class FloatAudioSelectionMap:
    policy: str
    selection_sha256: str
    schedule_id: str
    origin_id: str
    origin_decoder_id: str
    runtime_sha256: str
    operation_id: str
    receipt_sha256: str
    observed_pcm_sha256: str
    group: AudioSelectionGroup
    sample_grid_origin: Fraction
    observed_samples: int
    observed_bytes: int
    channels: int
    runs: tuple[AudioSelectionRun, ...]
    reused_first_interval: bool
    selected_samples: int
    selected_bytes: int

    def record(self) -> dict[str, object]:
        try:
            return _map_record(self)
        except AudioPreparationError:
            raise
        except (ValueError, TypeError, AttributeError, KeyError, IndexError, OverflowError) as exc:
            raise AudioPreparationError("SELECTION_INVALID", "Malformed map projection") from exc

    @property
    def sha256(self) -> str:
        return digest(canonical(self.record()))


@dataclass(frozen=True, slots=True, kw_only=True)
class SelectedFloatAudioGroup:
    mapping: FloatAudioSelectionMap
    map_sha256: str
    pcm: bytes
    pcm_sha256: str


def _scalars(value: Any) -> dict[str, object]:
    result: dict[str, object] = {}
    for f in fields(value):
        v = getattr(value, f.name)
        if f.type in (str, int, bool, float, Fraction):
            _require(type(v) is f.type, "SELECTION_INVALID", "Exact record scalar type required")
        if type(v) is Fraction:
            _stream_fraction(v)
            result[f.name] = rational(v)
        elif type(v) is int:
            result[f.name] = _stream_int(v, None)
        elif type(v) is float:
            _require(math.isfinite(v), "SELECTION_INVALID", "Finite record gains required")
            result[f.name] = str(v)
        elif type(v) is str:
            _stream_guard(v, 2_000_000)
            result[f.name] = v
        else:
            result[f.name] = v
    return result


def _plan_record(value: FloatAudioSelectionPlan) -> dict[str, object]:
    _require(type(value) is FloatAudioSelectionPlan, "SELECTION_INVALID", "Exact selection plan required")
    _limits(value.limits)
    _require(type(value.schedule) is AudioSchedule and type(value.audio_format) is AudioFormat
        and type(value.spans) is tuple and type(value.groups) is tuple,
        "SELECTION_INVALID", "Exact plan components required")
    _require(0 < len(value.spans) <= value.limits.max_shots
        and 0 < len(value.groups) <= value.limits.max_groups,
        "SELECTION_LIMIT", "Plan component count limit")
    _require(type(value.schedule.spans) is tuple and type(value.schedule.cues) is tuple
        and 0 < len(value.schedule.spans) <= value.limits.max_shots
        and len(value.schedule.cues) <= value.limits.max_cues,
        "SELECTION_LIMIT", "Schedule component count limit")
    _require(all(type(s) is AudioSelectionSpan for s in value.spans)
        and all(type(g) is AudioSelectionGroup for g in value.groups)
        and all(type(s) is AudioSpan for s in value.schedule.spans)
        and all(type(c) is ScheduledCue for c in value.schedule.cues),
        "SELECTION_INVALID", "Exact nested records required")
    result = _scalars(value)
    format_record = _scalars(value.audio_format)
    _require(value.policy == POLICY and type(value.schedule.source_fingerprint) is str
        and value.audio_format.sample_rate == RATE and value.audio_format.layout in ("mono", "stereo")
        and 0 < value.total_samples <= value.limits.max_selected_samples
        and value.total_samples*4*value.audio_format.channels <= value.limits.max_selected_bytes,
        "SELECTION_INVALID", "Selection format/policy/counts required")
    for name in ("origin_id", "origin_decoder_id", "schedule_id"):
        _hash_text(getattr(value, name))
    result.update(schedule=dict(source_fingerprint=value.schedule.source_fingerprint,
        spans=[], cues=[]), audio_format=format_record, spans=[], groups=[],
        limits=_scalars(value.limits))
    cap = value.limits.max_metadata_bytes
    budget = [_stream_guard(result, cap)]
    schedule = result["schedule"]
    assert isinstance(schedule, dict)
    schedule["spans"] = _project_rows(value.schedule.spans, budget, cap)
    schedule["cues"] = _project_rows(value.schedule.cues, budget, cap)
    result["spans"] = _project_rows(value.spans, budget, cap)
    result["groups"] = _project_rows(value.groups, budget, cap)
    _stream_guard(result, cap)
    return result


def _project_rows(rows: tuple[Any, ...], budget: list[int], cap: int) -> list[dict[str, object]]:
    projected: list[dict[str, object]] = []
    for row in rows:
        record = _scalars(row)
        budget[0] += _stream_guard(record, cap-budget[0]-1)+1
        projected.append(record)
    return projected


def _map_record(value: FloatAudioSelectionMap, cap: int = 2_000_000) -> dict[str, object]:
    _require(type(value) is FloatAudioSelectionMap and type(value.group) is AudioSelectionGroup
        and type(value.runs) is tuple and 1 <= len(value.runs) <= 2
        and all(type(r) is AudioSelectionRun for r in value.runs),
        "SELECTION_INVALID", "Exact bounded map records required")
    result = _scalars(value)
    result.update(group=_scalars(value.group), runs=[])
    _require(value.policy == POLICY and type(value.channels) is int and value.channels in (1, 2)
        and 0 < value.selected_samples <= 8_592_000 and 0 < value.observed_samples <= 9_000_000
        and 0 < value.observed_bytes <= 72_000_000 and 0 < value.selected_bytes <= 68_736_000,
        "SELECTION_INVALID", "Map counts/policy out of range")
    for name in ("selection_sha256", "schedule_id", "origin_id", "origin_decoder_id",
                 "runtime_sha256", "operation_id", "receipt_sha256", "observed_pcm_sha256"):
        _hash_text(getattr(value, name))
    _group_shape(value.group)
    _require(value.selected_samples == value.group.last-value.group.first
        and value.selected_bytes == value.selected_samples*4*value.channels
        and value.observed_bytes == value.observed_samples*4*value.channels,
        "SELECTION_INVALID", "Map frame/byte counts disagree")
    budget = [_stream_guard(result, cap)]
    result["runs"] = _project_rows(value.runs, budget, cap)
    prior = value.group.first
    for run in value.runs:
        _require(run.first == prior and run.first < run.last <= value.group.last
            and 0 <= run.source_first < run.source_first+run.last-run.first <= value.observed_samples
            and -Fraction(1, RATE) < run.residual <= Fraction(1, 2*RATE),
            "SELECTION_INVALID", "Map run partition/interval bounds")
        prior = run.last
    _require(prior == value.group.last and value.reused_first_interval == (len(value.runs) == 2),
             "SELECTION_INVALID", "Map run tail/reuse disagree")
    if len(value.runs) == 2:
        _require(value.runs[0].last == value.runs[0].first+1
            and value.runs[0].source_first == value.runs[1].source_first,
            "SELECTION_INVALID", "Only the first interval may repeat")
    _stream_guard(result, cap)
    return result


def _hash_text(value: str) -> None:
    _require(type(value) is str and len(value) == 64 and re.fullmatch("[0-9a-f]{64}", value) is not None,
             "SELECTION_INVALID", "Bounded lowercase digest required")


def _group_shape(group: AudioSelectionGroup) -> None:
    _scalars(group)
    _require(0 <= group.index < 4096 and 0 <= group.shot_first < group.shot_last <= 4096
        and 0 <= group.schedule_first < group.schedule_last <= 4096
        and 0 <= group.source_start < group.source_end
        and _add(group.source_end, -group.source_start) <= 179
        and 0 <= group.start < group.end <= 179 and 0 <= group.first < group.last <= 8_592_000
        and group.first == quantize_time(group.start) and group.last == quantize_time(group.end),
        "SELECTION_INVALID", "Invalid group declaration/ordinal bounds")


def _limits(value: AudioSelectionLimits) -> None:
    _require(type(value) is AudioSelectionLimits, "SELECTION_INVALID", "Exact selection limits required")
    for f in fields(AudioSelectionLimits):
        item = getattr(value, f.name)
        ceiling = f.default
        assert isinstance(ceiling, int)
        _require(type(item) is int and 0 < item <= ceiling,
                 "SELECTION_LIMIT", "Selection limits may only tighten")


def _plan_input(plan: VisualCompositionPlan, limits: AudioSelectionLimits) -> None:
    """Bound known structure before existing strict recursion or schema validation."""
    _require(type(plan) is VisualCompositionPlan and type(plan.shots) is list,
             "SELECTION_INVALID", "Exact visual plan/list required")
    _require(0 < len(plan.shots) <= limits.max_shots, "SELECTION_LIMIT", "Shot count limit")
    cues = 0
    for shot in plan.shots:
        _require(type(shot) is VisualShot and type(shot.audio) is AudioMixPlan
            and type(shot.audio.cues) is list, "SELECTION_INVALID", "Exact shot/audio shape required")
        cues += len(shot.audio.cues)
        _require(cues <= limits.max_cues, "SELECTION_LIMIT", "Cue count limit")
    models = (VisualCompositionPlan, VisualShot, TimeRange, AudioMixPlan, AudioCue,
              NormalizedRect, VisualLayer, CropKeyframe, SubtitlePlan, OverlayPlan)
    enums = (Anchor, FillMode, LayerKind)
    active: set[int] = set()
    nodes = 0
    used = 0

    def charge(amount: int) -> None:
        nonlocal used
        used += amount
        _require(used <= limits.max_plan_bytes, "SELECTION_LIMIT", "Input plan byte limit")

    def visit(value: object, depth: int) -> None:
        nonlocal nodes
        nodes += 1
        _require(nodes <= 65536 and depth <= 10, "SELECTION_LIMIT", "Input node/depth limit")
        kind = type(value)
        if kind in models or kind in (list, tuple, dict):
            _require(id(value) not in active, "SELECTION_INVALID", "Cyclic plan input")
            active.add(id(value))
            try:
                if kind in models:
                    model: Any = value
                    contents = model.__dict__
                    _require(type(contents) is dict and len(contents) <= len(kind.model_fields)
                        and all(type(key) is str for key in contents)
                        and not (contents.keys()-kind.model_fields.keys()) and not model.model_extra,
                        "SELECTION_INVALID", "Unrequested visual model fields")
                    charge(2)
                    for key, child in contents.items():
                        charge(_stream_guard(key, limits.max_plan_bytes-used)+2)
                        visit(child, depth+1)
                elif kind is dict:
                    assert isinstance(value, dict)
                    _require(len(value) <= 65536-nodes, "SELECTION_LIMIT", "Input container count")
                    charge(2)
                    for key, child in value.items():
                        _require(type(key) is str, "SELECTION_INVALID", "Plain plan keys required")
                        charge(_stream_guard(key, limits.max_plan_bytes-used)+2)
                        visit(child, depth+1)
                else:
                    assert isinstance(value, (list, tuple))
                    _require(len(value) <= 65536-nodes, "SELECTION_LIMIT", "Input container count")
                    charge(2)
                    for child in value:
                        charge(1)
                        visit(child, depth+1)
            finally:
                active.remove(id(value))
        elif kind in enums:
            assert isinstance(value, Enum)
            charge(_stream_guard(value.value, limits.max_plan_bytes-used))
        elif kind is float:
            assert isinstance(value, float)
            _require(math.isfinite(value), "SELECTION_INVALID", "Nonfinite plan float")
            charge(_stream_guard(str(value), limits.max_plan_bytes-used))
        elif kind in (str, int, bool) or value is None:
            charge(_stream_guard(value, limits.max_plan_bytes-used))
        else:
            raise AudioPreparationError("SELECTION_INVALID", "Unknown plan value")

    visit(plan, 0)


def _add(a: Fraction, b: Fraction) -> Fraction:
    _stream_fraction(a)
    _stream_fraction(b)
    width = max(a.numerator.bit_length()+b.denominator.bit_length(),
                b.numerator.bit_length()+a.denominator.bit_length())+1
    if max(width, a.denominator.bit_length()+b.denominator.bit_length()) > 16384:
        raise AudioPreparationError("SELECTION_LIMIT", "Prospective arithmetic width")
    return _stream_fraction(a+b)


def _mul(a: Fraction, b: Fraction) -> Fraction:
    _stream_fraction(a)
    _stream_fraction(b)
    if max(a.numerator.bit_length()+b.numerator.bit_length(),
           a.denominator.bit_length()+b.denominator.bit_length()) > 16384:
        raise AudioPreparationError("SELECTION_LIMIT", "Prospective arithmetic width")
    return _stream_fraction(a*b)


def _request(group: AudioSelectionGroup, n: int, origin: SourceTimeOrigin) -> Fraction:
    x = max(group.start, Fraction(n, RATE))
    return _add(_add(origin.origin, group.source_start), _add(x, -group.start))


def _run(group: AudioSelectionGroup, first: int, last: int, source_first: int,
         g: Fraction, origin: SourceTimeOrigin) -> AudioSelectionRun:
    residual = _add(_add(_add(g, Fraction(source_first-first, RATE)), -origin.origin),
                    _add(-group.source_start, group.start))
    return AudioSelectionRun(first=first, last=last, source_first=source_first, residual=residual)


def _require(condition: bool, code: str, message: str) -> None:
    if not condition:
        raise AudioPreparationError(code, message)


def _context(origin: SourceTimeOrigin, runtime: Mapping[str, object], cap: int) -> None:
    _require(type(origin) is SourceTimeOrigin and type(origin.streams) is tuple
             and len(origin.streams) == 2 and all(type(c) is StreamClock for c in origin.streams),
             "SELECTION_INVALID", "Exact origin/clock shape required")
    for name in ("path", "fingerprint", "decoder_id", "policy", "fingerprint_method"):
        _require(type(getattr(origin, name)) is str, "SELECTION_INVALID", "Plain origin strings required")
        _stream_guard(getattr(origin, name), cap)
    _stream_int(origin.size)
    _stream_int(origin.mtime_ns, None)
    _stream_fraction(origin.origin)
    _stream_fraction(origin.duration)
    _require(type(origin.color) is tuple and len(origin.color) <= cap,
             "SELECTION_LIMIT", "Origin color shape/count limit")
    for pair in origin.color:
        _require(type(pair) is tuple and len(pair) == 2
            and all(type(v) is str for v in pair), "SELECTION_INVALID", "Plain color pairs required")
    for c in origin.streams:
        for name in ("index", "width", "height", "sample_rate", "channels"):
            _stream_int(getattr(c, name))
        for name in ("time_base", "start", "end"):
            if (v := getattr(c, name)) is not None or name == "time_base":
                _stream_fraction(v)
        for name in ("kind", "sample_format"):
            _require(type(getattr(c, name)) is str, "SELECTION_INVALID", "Plain clock strings required")
            _stream_guard(getattr(c, name), cap)
    _float_origin(origin, cap)
    _require(type(runtime) is dict, "SELECTION_INVALID", "Plain current runtime required")
    _require(len(runtime) == 5 and all(type(k) is str and type(v) is str for k, v in runtime.items()),
             "SELECTION_INVALID", "Plain runtime keys/strings required")
    _stream_keys(runtime, "runtime_id ffmpeg ffprobe ffmpeg_sha256 ffprobe_sha256")
    _stream_guard(runtime, cap)
    _float_runtime(runtime)
    audio = next(c for c in origin.streams if c.kind == "audio")
    _require(audio.sample_rate in (44100, 48000) and audio.channels in (1, 2)
             and audio.sample_format in ("flt", "fltp"), "SELECTION_INVALID", "Unsupported source audio")


def _decode_limits(value: FloatAudioDecodeLimits) -> dict[str, object]:
    _require(type(value) is FloatAudioDecodeLimits, "SELECTION_INVALID", "Exact decode limits required")
    for f in fields(FloatAudioDecodeLimits):
        v = getattr(value, f.name)
        if f.name in ("preroll", "postroll", "max_window"):
            _stream_fraction(v)
            _require(0 <= v <= f.default and (f.name != "max_window" or v > 0),
                     "SELECTION_LIMIT", "Decode time limits may only tighten")
        else:
            _stream_int(v, 1)
            ceiling = f.default
            assert isinstance(ceiling, int)
            _require(v <= ceiling, "SELECTION_LIMIT", "Decode limits may only tighten")
    record = value.record()
    validate_float_audio_limits(record)
    return record


def _receipt(value: SourceFloatAudioDecodeReceipt, origin: SourceTimeOrigin,
             runtime: Mapping[str, object], limits: FloatAudioDecodeLimits) -> SourceFloatAudioDecodeReceipt:
    trusted = _decode_limits(limits)
    cap = limits.max_metadata_bytes
    _require(type(value) is SourceFloatAudioDecodeReceipt,
             "SELECTION_ASSOCIATION", "Exact float receipt required")
    for f in fields(SourceFloatAudioDecodeReceipt):
        text = getattr(value, f.name)
        _require(type(text) is str and len(text) <= cap,
                 "SELECTION_ASSOCIATION", "Bounded plain receipt strings required")
        # Stored JSON has its own raw UTF8 cap below, not the selection projection cap.
        _stream_guard(text, 12*cap+2)
    presented_runtime, _ = _bounded_json(value.runtime_json, cap)
    payload, _ = _bounded_json(value.payload_json, cap)
    _require(presented_runtime == runtime, "SELECTION_ASSOCIATION", "Receipt/current runtime disagree")
    fresh = SourceFloatAudioDecodeReceipt(origin, payload, runtime=runtime, limits=trusted)
    _require(all(getattr(value, f.name) == getattr(fresh, f.name)
                 for f in fields(SourceFloatAudioDecodeReceipt)),
             "SELECTION_ASSOCIATION", "Receipt/current origin or operation disagree")
    return fresh


def plan_float_audio_selection(plan: VisualCompositionPlan, *, origin: SourceTimeOrigin,
        runtime: Mapping[str, object], limits: AudioSelectionLimits) -> FloatAudioSelectionPlan:
    try:
        return _plan(plan, origin=origin, runtime=runtime, limits=limits)
    except AudioPreparationError:
        raise
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, OverflowError, ZeroDivisionError) as exc:
        raise AudioPreparationError("SELECTION_INVALID", "Malformed selection inputs") from exc


def _plan(plan: VisualCompositionPlan, *, origin: SourceTimeOrigin,
        runtime: Mapping[str, object], limits: AudioSelectionLimits) -> FloatAudioSelectionPlan:
    _limits(limits)
    _plan_input(plan, limits)
    _context(origin, runtime, limits.max_metadata_bytes)
    plan = VisualCompositionPlan.model_validate(_strict_model_input(plan))
    OutputTimeline.from_shots(plan.shots)
    schedule = schedule_audio(plan)
    channels = next(c.channels for c in origin.streams if c.kind == "audio")
    total = schedule.total_samples
    _require(0 < total <= limits.max_selected_samples and total*4*channels <= limits.max_selected_bytes,
             "SELECTION_LIMIT", "Selected payload capacity")
    spans: list[AudioSelectionSpan] = []
    groups: list[AudioSelectionGroup] = []
    metadata_used = _stream_guard(dict(policy=POLICY, origin_id="0"*64,
        origin_decoder_id=origin.decoder_id, runtime_json=canonical(runtime),
        schedule=dict(source_fingerprint=schedule.source_fingerprint, spans=[], cues=[]),
        schedule_id="0"*64, audio_format=dict(sample_rate=RATE, layout="mono" if channels == 1 else "stereo"),
        spans=[], groups=[], total_samples=total, limits=_scalars(limits)), limits.max_metadata_bytes)
    for rows in (schedule.spans, schedule.cues):
        for row in rows:
            metadata_used += 1+_stream_guard(_scalars(row), limits.max_metadata_bytes-metadata_used-1)
    planned_runs = 0
    schedule_index = 0
    for index, shot in enumerate(plan.shots):
        s, e, a, b = (Fraction(str(v)) for v in
                     (shot.source.start, shot.source.end, shot.output.start, shot.output.end))
        _require(0 <= s < e <= origin.duration and b <= 179,
                 "SELECTION_INVALID", "Source/output declaration bounds")
        scheduled = schedule.spans[schedule_index]
        adjacent = bool(groups and groups[-1].source_end == s and groups[-1].end == a)
        gi = len(groups)-1 if adjacent else len(groups)
        span = AudioSelectionSpan(shot_index=index, shot_id=shot.shot_id,
            schedule_span_index=schedule_index, group_index=gi, source_start=s, source_end=e,
            start=a, end=b, first=quantize_time(a), last=quantize_time(b))
        if span.first != (spans[-1].last if spans else 0) or span.last <= span.first:
            raise AudioPreparationError("SELECTION_INVALID", "Qa slices must partition output")
        metadata_used += 1+_stream_guard(_scalars(span), limits.max_metadata_bytes-metadata_used-1)
        spans.append(span)
        if adjacent:
            candidate = replace(groups[-1], shot_last=index+1, schedule_last=schedule_index+1,
                source_end=e, end=b, last=span.last)
            metadata_used -= _stream_guard(_scalars(groups[-1]), limits.max_metadata_bytes)
            metadata_used += _stream_guard(_scalars(candidate), limits.max_metadata_bytes-metadata_used)
            planned_runs += int(groups[-1].last-groups[-1].first == 1)
            groups[-1] = candidate
        else:
            candidate = AudioSelectionGroup(index=gi, shot_first=index, shot_last=index+1,
                schedule_first=schedule_index, schedule_last=schedule_index+1, source_start=s,
                source_end=e, start=a, end=b, first=span.first, last=span.last)
            _require(len(groups) < limits.max_groups, "SELECTION_LIMIT", "Group count limit")
            metadata_used += 1+_stream_guard(_scalars(candidate), limits.max_metadata_bytes-metadata_used-1)
            planned_runs += 1 if span.last-span.first == 1 else 2
            groups.append(candidate)
        _require(planned_runs <= limits.max_runs, "SELECTION_LIMIT", "Conservative planned run capacity")
        # Walk the actual schedule by its exact declared terminal anchors. Time
        # comparison alone misassigns permitted sub-nanosecond occurrence overlaps.
        if b == scheduled.end and e == scheduled.source_end:
            schedule_index += 1
    for group in groups:
        _require(_add(group.source_end, -group.source_start) <= 179,
                 "SELECTION_LIMIT", "Group source duration limit")
        first_span, last_span = schedule.spans[group.schedule_first], schedule.spans[group.schedule_last-1]
        _require(first_span.source_start == group.source_start and first_span.start == group.start
            and last_span.source_end == group.source_end and last_span.end == group.end
            and first_span.first == group.first and last_span.last == group.last,
            "SELECTION_ASSOCIATION", "Schedule cannot straddle an occurrence group")
    _require(plan.source_fingerprint == origin.fingerprint, "SELECTION_ASSOCIATION", "Source fingerprint mismatch")
    result = FloatAudioSelectionPlan(policy=POLICY, origin_id="0"*64,
        origin_decoder_id=origin.decoder_id, runtime_json=canonical(runtime), schedule=schedule,
        schedule_id="0"*64, audio_format=AudioFormat(RATE, "mono" if channels == 1 else "stereo"),
        spans=tuple(spans), groups=tuple(groups), total_samples=schedule.total_samples, limits=limits)
    _plan_record(result)
    _require(origin.decoder_id == source_decoder_identity(runtime),
             "SELECTION_ASSOCIATION", "Origin/current runtime disagree")
    return replace(result, origin_id=origin.identity, schedule_id=schedule.identity)


def map_float_audio_group(plan: VisualCompositionPlan, group_index: int,
        receipt: SourceFloatAudioDecodeReceipt, *, origin: SourceTimeOrigin,
        runtime: Mapping[str, object], decode_limits: FloatAudioDecodeLimits,
        limits: AudioSelectionLimits) -> FloatAudioSelectionMap:
    try:
        return _map(plan, group_index, receipt, origin=origin, runtime=runtime,
                    decode_limits=decode_limits, limits=limits)
    except AudioPreparationError:
        raise
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, OverflowError, ZeroDivisionError) as exc:
        raise AudioPreparationError("SELECTION_ASSOCIATION", "Malformed group association") from exc


def _map(plan: VisualCompositionPlan, group_index: int,
        receipt: SourceFloatAudioDecodeReceipt, *, origin: SourceTimeOrigin,
        runtime: Mapping[str, object], decode_limits: FloatAudioDecodeLimits,
        limits: AudioSelectionLimits) -> FloatAudioSelectionMap:
    _limits(limits)
    _decode_limits(decode_limits)
    _require(type(group_index) is int and 0 <= group_index < limits.max_groups,
             "SELECTION_ASSOCIATION", "Bounded plain group ordinal required")
    selection = plan_float_audio_selection(plan, origin=origin, runtime=runtime, limits=limits)
    _require(type(group_index) is int and 0 <= group_index < len(selection.groups),
             "SELECTION_ASSOCIATION", "Group ordinal out of range")
    receipt = _receipt(receipt, origin, runtime, decode_limits)
    group = selection.groups[group_index]
    data = receipt.record()["payload"]
    audio = data["audio"][0]
    _require(data["windows"] == [[rational(group.source_start), rational(group.source_end)]],
             "SELECTION_ASSOCIATION", "Receipt must request this exact group")
    _require(audio["coverage"][0]["status"] == "covered",
             "SELECTION_UNAVAILABLE", "Observed interval does not cover group")
    _require(audio["policy"] == "render48k-f32-"+selection.audio_format.layout+"-v1",
             "SELECTION_ASSOCIATION", "Float policy/channel mismatch")
    g = _stream_rational(audio["sample_grid_origin"])
    channels = selection.audio_format.channels
    samples = group.last-group.first
    _require(0 < audio["samples"] <= decode_limits.max_samples and
        0 < audio["bytes"] <= min(limits.max_observed_bytes, decode_limits.max_decoded_bytes,
            decode_limits.max_retained_bytes, 4*channels*decode_limits.max_samples),
        "SELECTION_LIMIT", "Observed payload capacity")
    _require(0 < samples <= limits.max_selected_samples and samples*4*channels <= limits.max_selected_bytes,
             "SELECTION_LIMIT", "Selected group capacity")
    first_source = _mul(Fraction(RATE), _add(_request(group, group.first, origin), -g)).__floor__()
    d = _mul(Fraction(RATE), _add(_add(_add(origin.origin, group.source_start), -group.start), -g))
    k = d.__floor__()
    split = first_source != group.first+k and group.last > group.first+1
    runs = ((_run(group, group.first, group.first+1, first_source, g, origin),
             _run(group, group.first+1, group.last, group.first+1+k, g, origin)) if split else
            (_run(group, group.first, group.last, first_source, g, origin),))
    for run in runs:
        _require(-Fraction(1, RATE) < run.residual <= Fraction(1, 2*RATE),
                 "SELECTION_UNAVAILABLE", "Residual outside interval bound")
        for n in (run.first, run.last-1):
            x = max(group.start, Fraction(n, RATE))
            tau = _request(group, n, origin)
            j = run.source_first+n-run.first
            left = _add(g, Fraction(j, RATE))
            right = _add(g, Fraction(j+1, RATE))
            source = _add(tau, -origin.origin)
            _require(0 <= j < audio["samples"] and group.start <= x < group.end
                and group.source_start <= source < group.source_end and left <= tau < right,
                "SELECTION_UNAVAILABLE", "Requested sample outside actual group coverage")
    result = FloatAudioSelectionMap(policy=POLICY, selection_sha256="0"*64,
        schedule_id=selection.schedule_id, origin_id=selection.origin_id,
        origin_decoder_id=selection.origin_decoder_id, runtime_sha256="0"*64,
        operation_id=receipt.operation_id, receipt_sha256="0"*64,
        observed_pcm_sha256=audio["pcm_sha256"], group=group, sample_grid_origin=g,
        observed_samples=audio["samples"], observed_bytes=audio["bytes"],
        channels=channels, runs=runs, reused_first_interval=split,
        selected_samples=samples, selected_bytes=samples*4*channels)
    _reserve_result(result, limits)
    return replace(result, selection_sha256=selection.sha256, runtime_sha256=digest(selection.runtime_json),
                   receipt_sha256=receipt.sha256)


def _reserve_result(mapping: FloatAudioSelectionMap, limits: AudioSelectionLimits) -> None:
    record = _map_record(mapping, limits.max_metadata_bytes)
    _stream_guard(dict(mapping=record, map_sha256="0"*64, pcm_sha256="0"*64,
                       pcm_bytes=mapping.selected_bytes), limits.max_metadata_bytes)


def _copy_pcm(mapping: FloatAudioSelectionMap, pcm: bytes) -> bytes:
    stride = 4*mapping.channels
    result = bytearray(mapping.selected_bytes)
    view = memoryview(pcm)
    offset = 0
    try:
        for run in mapping.runs:
            count = (run.last-run.first)*stride
            segment = view[run.source_first*stride:run.source_first*stride+count]
            try:
                result[offset:offset+count] = segment
            finally:
                segment.release()
            offset += count
    finally:
        view.release()
    _require(offset == mapping.selected_bytes and len(result) == mapping.selected_bytes,
             "SELECTION_PCM", "Selected final offset/count mismatch")
    return bytes(result)


def apply_float_audio_group(plan: VisualCompositionPlan, mapping: FloatAudioSelectionMap,
        observed: DecodedFloatAudioWindow, *, origin: SourceTimeOrigin,
        runtime: Mapping[str, object], decode_limits: FloatAudioDecodeLimits,
        limits: AudioSelectionLimits) -> SelectedFloatAudioGroup:
    try:
        return _apply(plan, mapping, observed, origin=origin, runtime=runtime,
                      decode_limits=decode_limits, limits=limits)
    except AudioPreparationError:
        raise
    except (ValueError, TypeError, AttributeError, KeyError, IndexError, OverflowError, ZeroDivisionError) as exc:
        raise AudioPreparationError("SELECTION_PCM", "Malformed observed float PCM") from exc


def _apply(plan: VisualCompositionPlan, mapping: FloatAudioSelectionMap,
        observed: DecodedFloatAudioWindow, *, origin: SourceTimeOrigin,
        runtime: Mapping[str, object], decode_limits: FloatAudioDecodeLimits,
        limits: AudioSelectionLimits) -> SelectedFloatAudioGroup:
    _limits(limits)
    _decode_limits(decode_limits)
    _require(type(mapping) is FloatAudioSelectionMap and type(observed) is DecodedFloatAudioWindow
             and type(observed.pcm) is bytes and type(mapping.runs) is tuple
             and type(mapping.group) is AudioSelectionGroup and type(mapping.channels) is int
             and type(mapping.observed_samples) is int,
             "SELECTION_ASSOCIATION", "Exact immutable map/observation required")
    _reserve_result(mapping, limits)
    _require(mapping.selected_samples <= limits.max_selected_samples
        and mapping.selected_bytes <= limits.max_selected_bytes
        and len(observed.pcm) == mapping.observed_bytes
        and len(observed.pcm) <= min(limits.max_observed_bytes, decode_limits.max_decoded_bytes,
            decode_limits.max_retained_bytes, 4*mapping.channels*decode_limits.max_samples),
        "SELECTION_LIMIT", "Observed/selected byte capacity/count mismatch")
    rebuilt = map_float_audio_group(plan, mapping.group.index, observed.receipt, origin=origin,
        runtime=runtime, decode_limits=decode_limits, limits=limits)
    if rebuilt != mapping:
        raise AudioPreparationError("SELECTION_ASSOCIATION", "Map disagrees with current inputs")
    fresh = _receipt(observed.receipt, origin, runtime, decode_limits)
    actual = DecodedFloatAudioWindow(fresh, observed.pcm)
    pcm = _copy_pcm(mapping, actual.pcm)
    return SelectedFloatAudioGroup(mapping=mapping, map_sha256=mapping.sha256,
        pcm=pcm, pcm_sha256=digest(pcm))
