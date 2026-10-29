"""Provider-free v2 preflight and static graph compilation; never executes renders."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import stat
import subprocess
from typing import Any, Literal, cast, TYPE_CHECKING
import zlib

from pydantic import ValidationError

from .audio_mix import (
    AudioFormat, CompiledAudio, NormalizationReceipt, SourceClockBinding, compile_audio,
)
from .filtergraph import (
    CompiledShotLayers, PixelRect, SourceLayer, _finite_color, _validate_size,
    _guard_known_input, _utf8_size, compile_shot_layers, compile_source_layer, rect_to_pixels,
)
from .local_paths import source_fingerprint
from .motion_compiler import CompiledCropTrack, CropPoint, FloatRect, _strict_model_input
from .output_timeline import OutputTimeline, RasterSpan, TimelineSpan, VideoSampleRequest
from .overlay_ass import (
    OverlayCompilation, OverlayEvent, _clean_text, _color, serialize_overlay_events,
)
from .render_assets import (
    BrandVisualLayer, PreparedBrandAsset, RenderAssetRegistry, compile_brand_layer,
    place_brand_asset, validate_asset_loadability,
)
from .render_clock import (
    SourceDecodeReceipt, SourceStreamFinalization, SourceTimeOrigin, StreamClock, StreamDecodeLimits,
    STREAM_DECODE_POLICY, _stream_guard, _stream_header, _stream_rational, _stream_session, _unique_pairs,
    canonical, digest, exact, rational,
)
from .render_contracts import EncoderName, RenderProfile
from .render_runtime import (
    RenderRequirements, RenderRuntime, RenderRuntimeError, RuntimeQualification, _verify_pins,
    combine_render_requirements, qualify_render_runtime, requirements_for_plan, resolve_render_runtime,
    RuntimeToolRunner,
)
if TYPE_CHECKING:
    from ._retained_files import RetainedFiles
from .schema import TranscriptSegment
from .source_decode import (
    DECODE_POLICY, DecodedWindow, DecodeLimits, ObservedFrame, ProvisionalSelection, VideoTag,
    Window, _decoder_interval, decode_requirements,
)
from .visual_contracts import (
    FillMode, LayerKind, VisualCompositionPlan, VisualLayer, _prepare_strict_visual_input,
)


class RenderPreflightError(ValueError):
    """Invalid/unsupported plan or unavailable host render dependency."""


MAX_PRIMITIVES = 4096
MAX_EVENTS = 8192
MAX_ASS_RESOURCES = 4096
MAX_BRANDS = 1024
MAX_STEPS = 32768
MAX_EVENT_CHARS = 16384
MAX_INPUT_TEXT_BYTES = 8_000_000
MAX_GENERATED_BYTES = 8_000_000
MAX_METADATA_ENTRIES = 32768
MAX_RECEIPT_BYTES = 8_000_000
MAX_BATCH_REQUESTS = 1200
MAX_RETAINED_BYTES = 64_000_000


def _guard_assembly_inputs(plan: VisualCompositionPlan, source_size: tuple[int, int],
                            overlays: OverlayCompilation) -> None:
    if type(plan) is not VisualCompositionPlan or type(overlays) is not OverlayCompilation:
        raise RenderPreflightError("Expected known plan/overlay records")
    if type(source_size) not in (tuple, list):
        raise RenderPreflightError("Source size requires a pixel pair")
    input_bytes = _guard_known_input((plan, source_size, overlays), record_types=(
        OverlayCompilation, OverlayEvent, BrandVisualLayer, PreparedBrandAsset, PixelRect),
        text_limit=MAX_INPUT_TEXT_BYTES, entry_limit=MAX_METADATA_ENTRIES)
    if (type(overlays.events) not in (tuple, list) or type(overlays.brands) not in (tuple, list)
            or type(overlays.font_paths) not in (tuple, list)
            or any(type(e) is not OverlayEvent for e in overlays.events)
            or any(type(b) is not BrandVisualLayer for b in overlays.brands)
            or any(not isinstance(p, Path) for p in overlays.font_paths)):
        raise RenderPreflightError("Invalid overlay compilation field shapes")
    primitives = sum(len(shot.layers) for shot in plan.shots)
    events, brands = len(overlays.events), len(overlays.brands)
    if primitives > MAX_PRIMITIVES or events > MAX_EVENTS or brands > MAX_BRANDS:
        raise RenderPreflightError("Primitive/event/brand resource limit")
    if any(type(event.text) is not str or len(event.text) > MAX_EVENT_CHARS
           for event in overlays.events):
        raise RenderPreflightError("Event character limit")
    # Each source/PNG separates at most one further ASS run. Reserve an upper bound
    # before sorting/cloning, then enforce actual emitted resource counts as well.
    ass_runs = min(events, primitives + brands + len(plan.shots))
    steps = len(plan.shots) * 4 + primitives * 3 + brands + ass_runs
    if ass_runs > MAX_ASS_RESOURCES or steps > MAX_STEPS:
        raise RenderPreflightError("ASS resource/step work budget")
    keyframes = sum(len(layer.crop_track) for shot in plan.shots for layer in shot.layers)
    # Includes fixed graph syntax, two native transforms, escaping, ASS headers,
    # decimal numeric bounds, detached metadata and canonical/projection documents.
    reserve = (input_bytes * 12 + primitives * 4096 + keyframes * 8192
               + events * 4096 + ass_runs * 1024 + len(plan.shots) * 2048 + brands * 4096)
    if reserve > MAX_GENERATED_BYTES:
        raise RenderPreflightError("Generated fragment/document/metadata byte budget")


class _GeneratedBudget:
    def __init__(self) -> None:
        self.used = 0

    def retain(self, text: str) -> str:
        self.used += _utf8_size(text, MAX_GENERATED_BYTES - self.used)
        return text


@dataclass(frozen=True, slots=True)
class SourceTransportSpan:
    shot_id: str
    k_start: int
    k_end: int
    slot_start: int
    slot_end: int


@dataclass(frozen=True, slots=True)
class SourceInputResource:
    key: str
    pixel_format: str
    source_size: tuple[int, int]
    frame_count: int


@dataclass(frozen=True, slots=True)
class FilterFragment:
    text: str
    input_labels: tuple[str, ...]
    output_labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AssApply:
    document_key: str
    input_label: str
    output_label: str


@dataclass(frozen=True, slots=True)
class AssDocumentResource:
    key: str
    document: str
    event_indices: tuple[int, ...]
    windows: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class BrandInputResource:
    key: str
    graph_label: str
    brand_index: int
    shot_id: str
    required_asset_id: str
    raster_window: tuple[int, int]
    placement: PixelRect
    asset: PreparedBrandAsset
    declared_interval: tuple[Fraction, Fraction]
    execution_interval: tuple[Fraction, Fraction]


@dataclass(frozen=True, slots=True)
class CompiledShotVisual:
    span: RasterSpan
    transport: SourceTransportSpan | None
    source_input_label: str | None
    source_layers: tuple[SourceLayer, ...]
    steps: tuple[FilterFragment | AssApply, ...]
    output_label: str
    output_clock: Literal["local_transport"]
    output_time_base: Fraction
    local_n_start: int
    frame_count: int


@dataclass(frozen=True, slots=True)
class CompiledVisualAssembly:
    timeline: OutputTimeline
    source_size: tuple[int, int]
    expected_source_fingerprint: str
    shots: tuple[CompiledShotVisual, ...]
    source_input: SourceInputResource | None
    ass_documents: tuple[AssDocumentResource, ...]
    brand_inputs: tuple[BrandInputResource, ...]
    canonical_ass: str
    requirements: RenderRequirements
    compiler_metadata_json: str


@dataclass(frozen=True, slots=True)
class ScheduledFrameBinding:
    transport_slot: int
    request: VideoSampleRequest
    requested_raw: Fraction
    receipt_hash: str
    frame_n: int
    frame: ObservedFrame


@dataclass(frozen=True, slots=True)
class BoundSourceBatch:
    slot_start: int
    slot_end: int
    origin_id: str
    decoder_id: str
    policy: str
    receipt_hash: str
    bindings: tuple[ScheduledFrameBinding, ...]
    verified_pcm_sha256: str
    verified_pcm_bytes: int
    verified_retained_bytes: int


@dataclass(frozen=True, slots=True)
class SourceStreamBindingContext:
    assembly: CompiledVisualAssembly
    origin: SourceTimeOrigin
    source_size: tuple[int, int]
    expected_source_fingerprint: str
    occurrence: RasterSpan
    transport: SourceTransportSpan
    slot_start: int
    slot_end: int
    session_id: str
    policy: str
    expected_audio_policy: str
    window: tuple[Fraction, Fraction]
    requests: tuple[VideoSampleRequest, ...]
    limits: StreamDecodeLimits
    chain_header_json: str


@dataclass(frozen=True, slots=True)
class SourceStreamBindingCursor:
    context: SourceStreamBindingContext
    next_delivery_index: int
    next_request_ordinal: int
    last_selected_frame_n: int | None
    chain_sha256: str
    selected_unique_bytes: int
    request_emission_bytes: int


@dataclass(frozen=True, slots=True)
class ProvisionalScheduledSlot:
    transport_slot: int
    request: VideoSampleRequest
    requested_raw: Fraction


@dataclass(frozen=True, slots=True)
class ProvisionalSourceDelivery:
    session_id: str
    delivery_index: int
    delivery_chain_sha256: str
    frame_n: int
    frame: ObservedFrame
    slots: tuple[ProvisionalScheduledSlot, ...]


@dataclass(frozen=True, slots=True)
class FinalizedScheduledSlot:
    transport_slot: int
    request: VideoSampleRequest
    requested_raw: Fraction
    frame_n: int


@dataclass(frozen=True, slots=True)
class FinalizedSourceBinding:
    slot_start: int
    slot_end: int
    origin_id: str
    decoder_id: str
    session_id: str
    policy: str
    finalization_sha256: str
    final_chain_sha256: str
    slots: tuple[FinalizedScheduledSlot, ...]
    selected_unique_bytes: int
    request_emission_bytes: int
    producer_audio_policy: str
    producer_pcm_sha256: str
    producer_pcm_bytes: int
    producer_audio_coverage: str


def _ass_boundary(k: int) -> Fraction:
    if type(k) is not int or k < 0:
        raise ValueError("Invalid raster boundary")
    return Fraction((5 * k) // 3, 100)


def _png_boundary(k: int) -> Fraction:
    if type(k) is not int or k < 0:
        raise ValueError("Invalid raster boundary")
    return Fraction(0) if k == 0 else Fraction(2 * k - 1, 120)


def _validate_text_ranges(event: OverlayEvent) -> None:
    if (type(event.text) is not str or not event.text or _clean_text(event.text) != event.text
            or type(event.lines) is not tuple or not event.lines
            or type(event.line_ranges) is not tuple or type(event.word_ranges) is not tuple
            or len(event.lines) != len(event.line_ranges)):
        raise RenderPreflightError("Invalid normalized overlay text/lines")
    for ranges in (event.line_ranges, event.word_ranges):
        if type(ranges) is not tuple:
            raise RenderPreflightError("Invalid overlay ranges")
        previous = 0
        for pair in ranges:
            if (type(pair) is not tuple or len(pair) != 2
                    or any(type(v) is not int for v in pair)):
                raise RenderPreflightError("Invalid overlay range indices")
            left, right = pair
            if (not previous <= left <= right <= len(event.text)
                    or ranges is event.word_ranges and left == right
                    or event.text[previous:left].strip()):
                raise RenderPreflightError("Overlay range omits visible text")
            previous = right
        if ranges and event.text[previous:].strip():
            raise RenderPreflightError("Overlay range omits visible text")
    if any(type(line) is not str or "\n" in line or event.text[a:b] != line
           for line, (a, b) in zip(event.lines, event.line_ranges)):
        raise RenderPreflightError("Overlay line slices disagree")
    if (type(event.active_words) is not tuple
            or any(type(i) is not int or not 0 <= i < len(event.word_ranges)
                   for i in event.active_words)
            or tuple(sorted(set(event.active_words))) != event.active_words):
        raise RenderPreflightError("Invalid active word indices")


def _finite_overlay_number(value: object) -> bool:
    """Strict finite float or safely representable integer; never coerce bool."""
    if type(value) is float:
        return math.isfinite(value)
    if type(value) is int:
        try:
            return math.isfinite(value)
        except OverflowError:
            return False
    return False


def _validate_overlays(plan: VisualCompositionPlan, timeline: OutputTimeline,
                       overlays: OverlayCompilation) -> tuple[tuple[int, int], ...]:
    if type(overlays) is not OverlayCompilation:
        raise RenderPreflightError("Expected OverlayCompilation")
    shots = {shot.shot_id: shot for shot in plan.shots}
    titles: set[str] = set()
    windows: list[tuple[int, int]] = []
    for event in overlays.events:
        if type(event) is not OverlayEvent or event.shot_id not in shots:
            raise RenderPreflightError("Foreign overlay event")
        shot = shots[event.shot_id]
        if (type(event.font_family) is not str or not event.font_family or len(event.font_family) > MAX_EVENT_CHARS
                or any(ord(c) < 32 or 127 <= ord(c) < 160 or c in ",\\{}" for c in event.font_family)
                or any(type(v) is not int for v in (event.font_size, event.outline_width,
                                                   event.layer, event.word_count))
                or type(event.bold) is not bool or event.layer < 0 or event.word_count < 0
                or type(event.destination) is not PixelRect
                or any(type(v) not in (int, float) or not 0 <= v <= bound
                       or not _finite_overlay_number(v)
                       for v, bound in ((event.measured_width, event.destination.width),
                                        (event.measured_height, event.destination.height)))
                or any(not _finite_overlay_number(v) for v in (event.start, event.end))
                or type(event.exact_interval) is not tuple or len(event.exact_interval) != 2
                or any(type(v) is not Fraction for v in event.exact_interval)):
            raise RenderPreflightError("Invalid overlay style/timing/measurement metadata")
        p = event.destination
        PixelRect(p.x, p.y, p.width, p.height)
        for color in (event.primary_color, event.outline_color):
            if type(color) is not str or re.fullmatch(r"&H00[0-9A-F]{6}", color) is None:
                raise RenderPreflightError("Invalid ASS style color")
        _validate_text_ranges(event)
        if event.kind == "title":
            title = shot.title
            if title is None or shot.shot_id in titles or title.text is None:
                raise RenderPreflightError("Missing/duplicate/foreign title")
            titles.add(shot.shot_id)
            interval = timeline.overlay_output_interval(shot.shot_id, title.start, title.end)
            if (event.text != _clean_text(title.text) or event.exact_interval != interval
                    or (event.start, event.end) != (shot.output.start + title.start,
                                                   shot.output.start + title.end)
                    or event.destination != rect_to_pixels(title.destination, width=1080, height=1920)
                    or event.layer != title.z_index or not 1 <= event.font_size <= 46
                    or event.outline_width != 2 or event.primary_color != "&H00FFFFFF"
                    or event.outline_color != "&H00000000" or event.word_count != 0
                    or event.word_ranges or event.active_words):
                raise RenderPreflightError("Title does not match planned occurrence/style/text")
        elif event.kind == "subtitle":
            subtitle = shot.subtitles
            if subtitle is None:
                raise RenderPreflightError("Unplanned subtitle")
            layer = max([v.z_index for v in shot.layers]
                        + [v.z_index for v in (shot.title, shot.brand) if v is not None]) + 1
            count = (sum(len(_clean_text(event.text[a:b]).split()) for a, b in event.word_ranges)
                     if event.word_ranges else len(event.text.split()))
            if (event.destination != rect_to_pixels(subtitle.destination, width=1080, height=1920)
                    or event.layer != layer or event.font_size != subtitle.font_size or not event.bold
                    or event.outline_width != subtitle.outline_width
                    or event.primary_color != _color(subtitle.primary_color)
                    or event.outline_color != _color(subtitle.outline_color)
                    or not 0 < count == event.word_count <= subtitle.max_words_per_event
                    or len(event.lines) > subtitle.max_lines
                    or (not subtitle.emphasize_active_word and event.active_words)
                    or (subtitle.emphasize_active_word and not event.word_ranges)
                    or (event.start, event.end) != tuple(float(v) for v in event.exact_interval)):
                raise RenderPreflightError("Subtitle does not match planned style/ranges")
        else:
            raise RenderPreflightError("Unknown overlay kind")
        windows.append(timeline.raster_window(*event.exact_interval, shot_id=event.shot_id))
    if titles != {s.shot_id for s in plan.shots if s.title is not None}:
        raise RenderPreflightError("Missing planned title")
    seen: set[str] = set()
    for brand in overlays.brands:
        if (type(brand) is not BrandVisualLayer or type(brand.asset) is not PreparedBrandAsset
                or brand.shot_id not in shots or brand.shot_id in seen):
            raise RenderPreflightError("Duplicate/foreign brand")
        seen.add(brand.shot_id)
        planned = shots[brand.shot_id].brand
        if planned is None or planned.asset_id is None or planned.font_asset_id is not None:
            raise RenderPreflightError("Unplanned/unsupported brand")
        asset = brand.asset
        if (not isinstance(asset.source_path, Path) or not isinstance(asset.path, Path)
                or type(asset.source_sha256) is not str
                or any(type(v) is not tuple for v in (asset.source_size, asset.source_bbox, asset.size))
                or type(brand.destination) is not PixelRect or type(brand.placement) is not PixelRect):
            raise RenderPreflightError("Prepared asset must have immutable typed fields")
        for p in (brand.destination, brand.placement):
            PixelRect(p.x, p.y, p.width, p.height)
        PreparedBrandAsset(asset.source_path, asset.path, asset.source_sha256,
                           asset.source_size, asset.source_bbox, asset.size)
        shot = shots[brand.shot_id]
        expected = place_brand_asset(asset,
            destination=rect_to_pixels(planned.destination, width=1080, height=1920),
            start=shot.output.start + planned.start, end=shot.output.start + planned.end,
            z_index=planned.z_index, shot_id=shot.shot_id)
        if (any(type(v) not in (int, float) for v in (brand.start, brand.end))
                or type(brand.z_index) is not int or brand != expected):
            raise RenderPreflightError("Brand differs from planned occurrence/placement")
    if seen != {s.shot_id for s in plan.shots if s.brand is not None}:
        raise RenderPreflightError("Missing planned brand")
    if overlays.document != serialize_overlay_events(overlays.events):
        raise RenderPreflightError("Canonical ASS differs from events")
    return tuple(windows)


def _span_metadata(span: RasterSpan) -> dict[str, object]:
    return {"occurrence_index": span.occurrence_index, "shot_id": span.shot_id,
        "source": [rational(span.source_start), rational(span.source_end)],
        "output": [rational(span.output_start), rational(span.output_end)],
        "raster": [span.k_start, span.k_end], "start_delta": rational(span.start_delta),
        "end_delta": rational(span.end_delta), "duration_delta": rational(span.duration_delta)}


def compile_visual_assembly(plan: VisualCompositionPlan, *, source_size: tuple[int, int],
                            overlays: OverlayCompilation) -> CompiledVisualAssembly:
    """Compile finite ordered shot templates. Only a trusted host may bind resources."""
    _guard_assembly_inputs(plan, source_size, overlays)
    _validate_size(source_size)
    source_size = (source_size[0], source_size[1])
    budget = _GeneratedBudget()
    budget.retain(overlays.document)
    resolved = resolve_visual_plan(plan)
    timeline = OutputTimeline.from_shots(resolved.shots)
    spans = timeline.raster_spans()
    event_windows = _validate_overlays(resolved, timeline, overlays)
    shots: list[CompiledShotVisual] = []
    documents: list[AssDocumentResource] = []
    brands: list[BrandInputResource] = []
    requirements: list[RenderRequirements] = []
    metadata: list[dict[str, object]] = []
    slot = 0
    for shot, span in zip(resolved.shots, spans):
        prefix = f"shot_{span.occurrence_index:04d}"
        consumers = [i for i, layer in enumerate(shot.layers) if layer.kind != LayerKind.SOLID]
        inlet = prefix + "_source" if consumers else None
        transport = (SourceTransportSpan(shot.shot_id, span.k_start, span.k_end,
                     slot, slot + span.frame_count) if consumers else None)
        if consumers:
            slot += span.frame_count
        steps: list[FilterFragment | AssApply] = []
        available = {inlet} if inlet is not None else set()
        used_labels = set(available)

        def append_step(step: FilterFragment | AssApply) -> None:
            inputs = step.input_labels if isinstance(step, FilterFragment) else (step.input_label,)
            outputs = step.output_labels if isinstance(step, FilterFragment) else (step.output_label,)
            if (not set(inputs) <= available or len(inputs) != len(set(inputs))
                    or len(outputs) != len(set(outputs)) or set(outputs) & used_labels):
                raise RenderPreflightError("Unclosed/colliding resource labels")
            if isinstance(step, FilterFragment):
                budget.retain(step.text)
            available.difference_update(inputs)
            available.update(outputs)
            used_labels.update(outputs)
            steps.append(step)
            if sum(len(s.steps) for s in shots) + len(steps) > MAX_STEPS:
                raise RenderPreflightError("Generated step limit")
        branches = {i: f"{prefix}_branch_{i}" for i in consumers}
        if inlet is not None:
            split = f"split={len(consumers)}" if len(consumers) > 1 else "null"
            append_step(FilterFragment(f"[{inlet}]settb=1/60,setpts=N,{split}"
                + "".join(f"[{v}]" for v in branches.values()), (inlet,), tuple(branches.values())))
        base = prefix + "_base"
        append_step(FilterFragment(_finite_color("black", PixelRect(0, 0, 1080, 1920),
            frame_count=span.frame_count) + f",format=rgba,setpts=N+{span.k_start}[{base}]", (), (base,)))
        primitives = tuple(compile_source_layer(shot, layer_index=i, raster_span=span,
            source_size=source_size, input_label=branches.get(i)) for i in range(len(shot.layers)))
        requirements.extend(p.requirements for p in primitives)
        # Every op carries only a typed, validated resource index; raw IDs/text never enter syntax.
        operations: list[tuple[int, int, int]] = [(p.z_index, 0, p.declaration_index) for p in primitives]
        operations.extend((e.layer, 1 if e.kind == "title" else 3, i)
                          for i, e in enumerate(overlays.events) if e.shot_id == shot.shot_id)
        operations.extend((b.z_index, 2, i) for i, b in enumerate(overlays.brands)
                          if b.shot_id == shot.shot_id)
        operations.sort()
        paint_order: list[dict[str, object]] = []
        op_index = 0
        while op_index < len(operations):
            z, tie, index = operations[op_index]
            output = f"{prefix}_paint_{op_index}"
            if tie in (1, 3):
                indices: list[int] = []
                while op_index < len(operations) and operations[op_index][1] in (1, 3):
                    indices.append(operations[op_index][2])
                    op_index += 1
                key = f"ass_{len(documents):04d}"
                projected = []
                for i in indices:
                    interval = tuple(_ass_boundary(k) for k in event_windows[i])
                    projected.append(replace(overlays.events[i], start=float(interval[0]),
                        end=float(interval[1]), exact_interval=(interval[0], interval[1])))
                if len(documents) >= MAX_ASS_RESOURCES:
                    raise RenderPreflightError("Generated ASS resource limit")
                documents.append(AssDocumentResource(key, budget.retain(serialize_overlay_events(projected)),
                    tuple(indices), tuple(event_windows[i] for i in indices)))
                append_step(AssApply(key, base, output))
                paint_order.append({"kind": "ass", "key": key, "events": indices})
            elif tie == 0:
                primitive = primitives[index]
                global_pixels = primitive.output_label + "_global"
                append_step(FilterFragment(primitive.filter_complex,
                    (primitive.input_label,) if primitive.input_label is not None else (),
                    (primitive.output_label,)))
                append_step(FilterFragment(f"[{primitive.output_label}]format=rgba,settb=1/60,"
                    f"setpts=N+{span.k_start}[{global_pixels}]", (primitive.output_label,), (global_pixels,)))
                p = primitive.destination
                append_step(FilterFragment(f"[{base}][{global_pixels}]overlay=x={p.x}:y={p.y}:"
                    f"shortest=1:format=rgb[{output}]", (base, global_pixels), (output,)))
                paint_order.append({"kind": primitive.kind.value, "layer_index": index, "z": z})
                op_index += 1
            else:
                brand = overlays.brands[index]
                planned = shot.brand
                assert planned is not None and planned.asset_id is not None
                declared = timeline.overlay_output_interval(shot.shot_id, planned.start, planned.end)
                window = timeline.raster_window(*declared, shot_id=shot.shot_id)
                execution = (_png_boundary(window[0]), _png_boundary(window[1]))
                key = f"brand_{len(brands):04d}"
                graph_label = key + "_pixels"
                brands.append(BrandInputResource(key, graph_label, index, shot.shot_id,
                    planned.asset_id, window, brand.placement, brand.asset, declared, execution))
                execution_brand = replace(brand, start=float(execution[0]), end=float(execution[1]))
                if graph_label in used_labels:
                    raise RenderPreflightError("Colliding PNG input label")
                available.add(graph_label)
                used_labels.add(graph_label)
                append_step(FilterFragment(compile_brand_layer(execution_brand, input_label=base,
                    brand_label=graph_label, output_label=output), (base, graph_label), (output,)))
                paint_order.append({"kind": "brand", "key": key, "z": z,
                    "declared": [rational(v) for v in declared], "raster": list(window),
                    "execution": [rational(v) for v in execution]})
                op_index += 1
            base = output
        output = prefix + "_output"
        append_step(FilterFragment(f"[{base}]setsar=1,settb=1/60,setpts=N[{output}]", (base,), (output,)))
        if available != {output}:
            raise RenderPreflightError("Unconsumed graph inputs/outputs")
        shots.append(CompiledShotVisual(span, transport, inlet, primitives, tuple(steps), output,
                                       "local_transport", Fraction(1, 60), 0, span.frame_count))
        metadata.append({**_span_metadata(span), "transport": None if transport is None else
            {"slots": [transport.slot_start, transport.slot_end], "source_input": inlet,
             "correspondence": "K=k_start+slot-slot_start;timeline.video_request(K)"},
            "primitives": [{"layer_id": p.layer_id, "kind": p.kind.value, "z": p.z_index,
                "declaration_index": p.declaration_index,
                "destination": [p.destination.x, p.destination.y, p.destination.width, p.destination.height],
                "crop_track": None if p.crop_track is None else [{"n": point.frame,
                    "rect": list(point.rect.values), "easing": point.easing} for point in p.crop_track.points]}
                for p in primitives], "operations": paint_order,
            "clock_stages": ["local_transport_1/60_n", "global_K/60_paint", "local_transport_1/60_n"]})
    filters = ("color", "trim", "settb", "setpts", "setsar", "overlay", "split", "null", "format")
    requirements.append(RenderRequirements(required_filters=filters))
    if documents:
        requirements.append(RenderRequirements(required_filters=("ass",)))
    if brands:
        requirements.append(RenderRequirements(required_filters=("scale", "format")))
    if slot:
        requirements.append(decode_requirements())
    compiler_metadata = budget.retain(canonical({"provenance": "compiler_resolved", "shots": metadata,
        "source_observation": "source_input_observed_at_binding_only" if slot else "not_applicable_no_source_consumer",
        "overlay_association": "structural_only;not_font_asset_or_transcript_completeness",
        "overlays": [{"event_index": i, "shot_id": event.shot_id, "kind": event.kind,
            "declared": [rational(v) for v in event.exact_interval or ()],
            "raster": list(event_windows[i]),
            "execution": [rational(_ass_boundary(k)) for k in event_windows[i]]}
            for i, event in enumerate(overlays.events)],
        "canonical_ass": "same_serializer_original_events"}))
    return CompiledVisualAssembly(timeline, source_size, resolved.source_fingerprint, tuple(shots),
        SourceInputResource("source_pixels", "bgr24", source_size, slot) if slot else None,
        tuple(documents), tuple(brands), overlays.document,
        combine_render_requirements(*requirements), compiler_metadata)


def _validate_requested_frame(frame: ObservedFrame, row: dict[str, Any], *,
                              source_size: tuple[int, int], requested_raw: Fraction) -> None:
    """Validate supplied immutable bytes against one observed row, without a receipt seal."""
    if type(frame) is not ObservedFrame or type(frame.tag) is not VideoTag or type(frame.bgr) is not bytes:
        raise RenderPreflightError("Expected immutable observed frame/tag/bytes")
    tag = frame.tag
    if (any(type(v) is not int for v in (tag.n, tag.pts, tag.width, tag.height, tag.checksum))
            or tag.n < 0 or tag.width <= 0 or tag.height <= 0 or not 0 <= tag.checksum <= 0xFFFFFFFF
            or type(tag.time_base) is not Fraction or tag.time_base <= 0
            or type(frame.raw_end) is not Fraction or frame.raw_end <= frame.raw_start
            or (tag.width, tag.height) != source_size or len(frame.bgr) != tag.byte_count
            or any(row.get(name) != value for name, value in tag.record().items())
            or row.get("bytes") != len(frame.bgr)
            or row.get("coverage_end") != rational(frame.raw_end)
            or not frame.raw_start <= requested_raw < frame.raw_end
            or hashlib.sha256(frame.bgr).hexdigest() != row.get("sha256")
            or zlib.adler32(frame.bgr, 0) != tag.checksum):
        raise RenderPreflightError("Observed BGR tag/interval/bytes/hash/checksum disagreement")


def _validate_materialized_pcm(pcm: bytes, audio: dict[str, Any], *,
                               expected_audio_policy: str) -> tuple[str, int]:
    """Verify output bytes only; numeric conversion and coverage stay in the real parser."""
    if type(pcm) is not bytes or audio["policy"] != expected_audio_policy or len(pcm) != audio["bytes"]:
        raise RenderPreflightError("Actual PCM policy/byte count disagreement")
    sha = hashlib.sha256(pcm).hexdigest()
    if sha != audio["pcm_sha256"]:
        raise RenderPreflightError("Actual PCM aggregate hash disagreement")
    view = memoryview(pcm)
    offset = 0
    for block in audio["output"]:
        end = offset + block["bytes"]
        if end > len(pcm):
            raise RenderPreflightError("Actual PCM block overflow")
        body = view[offset:end]
        if (hashlib.sha256(body).hexdigest() != block["sha256"]
                or f"{zlib.adler32(body, 0):08X}" != block["checksum"]):
            raise RenderPreflightError("Actual PCM output block hash/checksum disagreement")
        offset = end
    if offset != len(pcm):
        raise RenderPreflightError("Actual PCM final offset disagreement")
    return sha, len(pcm)


def _guard_json_work(text: str, *, cap: int, max_depth: int) -> None:
    """Lexical allocation reservation only; receipt constructors own schema semantics."""
    _utf8_size(text, cap)
    stack: list[str] = []
    quoted, escaped, previous = False, False, ""
    entries, index = 0, 0
    while index < len(text):
        char = text[index]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted, previous = False, "value"
        elif not char.isspace():
            if stack and stack[-1] == "[" and previous in ("[", ",") and char != "]":
                entries += 1
            if char == '"':
                quoted = True
            elif char in "[{":
                stack.append(char)
                if len(stack) > max_depth:
                    raise RenderPreflightError("Receipt JSON nesting limit")
            elif char in "]}":
                if not stack or stack.pop() != ("[" if char == "]" else "{"):
                    raise RenderPreflightError("Receipt JSON nesting disagreement")
            elif char == ":":
                entries += 1
            elif char == "-" or char.isdigit():
                end = index + 1
                while end < len(text) and text[end] in "0123456789":
                    end += 1
                token = text[index:end]
                if len(token.lstrip("-")) > 1234 or token != "-" and int(token).bit_length() > 4096:
                    raise RenderPreflightError("Receipt numeric bit limit")
                index = end - 1
            if entries > MAX_METADATA_ENTRIES:
                raise RenderPreflightError("Receipt metadata entry limit")
            if not quoted:
                previous = char
        index += 1
    if stack or quoted:
        raise RenderPreflightError("Receipt JSON incomplete structure")


def _receipt_rational_prework(value: str) -> Fraction:
    """Reserve legacy decimal work without expanding powers; final bit caps stay below."""
    if len(value) > 2500:
        raise RenderPreflightError("Receipt rational bit limit")
    if "/" not in value:
        try:
            lexical = Decimal(value)
        except InvalidOperation as exc:
            raise RenderPreflightError("Invalid receipt rational text") from exc
        if not lexical.is_finite():
            raise RenderPreflightError("Receipt rational must be finite")
        _, digits, exponent = lexical.as_tuple()
        # 1234 decimal digits cover a 4096-bit component. Include the bounded
        # coefficient's 2500 text digits to allow cancellation in old valid forms.
        # Decimal tuple inspection performs no power/scaling arithmetic. This
        # reserves bounded conversion work, including zero with a huge exponent.
        if (type(exponent) is not int
                or max(len(digits) + max(exponent, 0), -exponent) > 2500 + 1234):
            raise RenderPreflightError("Receipt rational expansion work limit")
    return exact(value)


def _bounded_json_metadata(text: str, *, cap: int, max_depth: int,
                           rational_parser: Callable[[str], Fraction] = _stream_rational) -> dict[str, Any]:
    """Shared pre-work byte/depth/entry/numeric guard, without receipt interpretation."""
    _guard_json_work(text, cap=cap, max_depth=max_depth)
    def integer(value: str) -> int:
        if len(value) > 1234:
            raise RenderPreflightError("Receipt numeric bit limit")
        number = int(value)
        if number.bit_length() > 4096:
            raise RenderPreflightError("Receipt numeric bit limit")
        return number

    try:
        data = json.loads(text, parse_int=integer, object_pairs_hook=_unique_pairs)
    except (RecursionError, json.JSONDecodeError) as exc:
        raise RenderPreflightError("Receipt JSON nesting/schema limit") from exc
    entries = 0
    rational_fields = {"windows", "window", "preroll", "postroll", "max_window", "decoder_start",
        "decoder_end", "requested", "time_base", "coverage_end", "sample_grid_origin", "source_phase"}

    def check(item: object, key: str = "", depth: int = 0) -> None:
        nonlocal entries
        if depth > 24:
            raise RenderPreflightError("Receipt nesting limit")
        if isinstance(item, (dict, list)):
            entries += len(item)
            if entries > MAX_METADATA_ENTRIES:
                raise RenderPreflightError("Receipt metadata entry limit")
            if isinstance(item, dict):
                for name, child in item.items():
                    _utf8_size(name, cap)
                    check(child, name, depth + 1)
            else:
                for child in item:
                    check(child, key, depth + 1)
        elif type(item) is str and key in rational_fields:
            if len(item) > 2500:
                raise RenderPreflightError("Receipt rational bit limit")
            number = rational_parser(item)
            if max(number.numerator.bit_length(), number.denominator.bit_length()) > 4096:
                raise RenderPreflightError("Receipt rational bit limit")
        elif type(item) is float and not math.isfinite(item):
            raise RenderPreflightError("Receipt nonfinite numeric value")
        elif type(item) is str:
            _utf8_size(item, cap)

    check(data)
    if type(data) is not dict:
        raise RenderPreflightError("Receipt payload schema")
    return data


def _bounded_receipt_payload(text: str) -> dict[str, Any]:
    """Materialized schema wrapper; complete receipt validation stays in the real loader."""
    data = _bounded_json_metadata(text, cap=MAX_RECEIPT_BYTES, max_depth=24,
        rational_parser=_receipt_rational_prework)
    for key in ("video", "outcomes"):
        if type(data.get(key)) is not list or len(data[key]) > MAX_BATCH_REQUESTS:
            raise RenderPreflightError("Receipt video/request row limit")
    if (type(data.get("audio")) is not list or len(data["audio"]) != 1
            or type(data["audio"][0]) is not dict):
        raise RenderPreflightError("Receipt audio schema")
    for stage in ("input", "output"):
        blocks = data["audio"][0].get(stage)
        if type(blocks) is not list or len(blocks) > MAX_BATCH_REQUESTS:
            raise RenderPreflightError("Receipt audio block limit")
    return data


def _source_slot_association(assembly: CompiledVisualAssembly, origin: SourceTimeOrigin,
                             slot_start: int, slot_end: int
                             ) -> tuple[CompiledShotVisual, tuple[VideoSampleRequest, ...]]:
    """Shared structural association; schema/resource wrappers remain separate."""
    source = assembly.source_input
    if (type(source) is not SourceInputResource or source.key != "source_pixels"
            or source.pixel_format != "bgr24" or source.source_size != assembly.source_size
            or origin.fingerprint != assembly.expected_source_fingerprint):
        raise RenderPreflightError("No source consumer or mismatched source fingerprint")
    video = [stream for stream in origin.streams if stream.kind == "video"]
    if len(video) != 1 or (video[0].width, video[0].height) != assembly.source_size:
        raise RenderPreflightError("Origin/source dimensions disagree")
    if (type(assembly.timeline) is not OutputTimeline or type(assembly.timeline.spans) is not tuple
            or any(type(s) is not TimelineSpan for s in assembly.timeline.spans)
            or type(assembly.shots) is not tuple or any(type(s) is not CompiledShotVisual for s in assembly.shots)):
        raise RenderPreflightError("Assembly timeline/shot record types disagree")
    timeline = OutputTimeline(tuple(TimelineSpan(s.shot_id, s.source_start, s.source_end,
        s.output_start, s.output_end) for s in assembly.timeline.spans))
    if tuple(s.span for s in assembly.shots) != timeline.raster_spans():
        raise RenderPreflightError("Assembly raster/timeline association disagreement")
    expected_slot = 0
    for compiled in assembly.shots:
        span = compiled.span
        if type(span) is not RasterSpan:
            raise RenderPreflightError("Expected raster span record")
        RasterSpan(span.occurrence_index, span.shot_id, span.source_start, span.source_end,
                   span.output_start, span.output_end, span.k_start, span.k_end)
        if (compiled.output_clock != "local_transport" or type(compiled.output_time_base) is not Fraction
                or compiled.output_time_base != Fraction(1, 60) or type(compiled.local_n_start) is not int
                or compiled.local_n_start != 0 or type(compiled.frame_count) is not int
                or compiled.frame_count != span.frame_count or type(compiled.source_layers) is not tuple
                or any(type(layer) is not SourceLayer or type(layer.kind) is not LayerKind
                       for layer in compiled.source_layers)):
            raise RenderPreflightError("Compiled shot local clock/layer fields disagree")
        consumes = any(layer.kind != LayerKind.SOLID for layer in compiled.source_layers)
        if consumes != (compiled.transport is not None) or consumes != (compiled.source_input_label is not None):
            raise RenderPreflightError("Authored source consumer/transport disagreement")
        t = compiled.transport
        if t is not None:
            if (type(t) is not SourceTransportSpan or any(type(v) is not int for v in
                    (t.k_start, t.k_end, t.slot_start, t.slot_end))
                    or (t.shot_id, t.k_start, t.k_end, t.slot_start, t.slot_end) !=
                    (compiled.span.shot_id, compiled.span.k_start, compiled.span.k_end,
                     expected_slot, expected_slot + compiled.span.frame_count)):
                raise RenderPreflightError("Assembly source transport disagreement")
            expected_slot = t.slot_end
    if type(source.frame_count) is not int or source.frame_count != expected_slot:
        raise RenderPreflightError("Source resource count disagreement")
    matches = [shot for shot in assembly.shots if shot.transport is not None
               and shot.transport.slot_start <= slot_start < slot_end <= shot.transport.slot_end]
    if len(matches) != 1:
        raise RenderPreflightError("Slot slice must fit one consuming occurrence")
    shot = matches[0]
    transport = shot.transport
    assert transport is not None
    requests = tuple(assembly.timeline.video_request(transport.k_start + slot - transport.slot_start)
                     for slot in range(slot_start, slot_end))
    return shot, requests


def bind_observed_batch(assembly: CompiledVisualAssembly, *, slot_start: int, slot_end: int,
                        origin: SourceTimeOrigin, decoded_window: DecodedWindow,
                        expected_audio_policy: Literal["analysis16k-mono", "render48k-mono",
                                                       "render48k-stereo"],
                        limits: DecodeLimits = DecodeLimits()) -> BoundSourceBatch:
    """Seal a COMPLETE supplied window's actual bytes to exact scheduled source requests."""
    if (type(assembly) is not CompiledVisualAssembly or type(origin) is not SourceTimeOrigin
            or type(decoded_window) is not DecodedWindow or type(limits) is not DecodeLimits
            or type(decoded_window.receipt) is not SourceDecodeReceipt
            or type(slot_start) is not int or type(slot_end) is not int
            or not 0 <= slot_start < slot_end or slot_end - slot_start > MAX_BATCH_REQUESTS
            or type(expected_audio_policy) is not str or expected_audio_policy not in
            ("analysis16k-mono", "render48k-mono", "render48k-stereo")):
        raise RenderPreflightError("Invalid COMPLETE batch records/slots/policy or request limit")
    defaults = DecodeLimits()
    _guard_known_input((assembly, origin, limits), record_types=(
        CompiledVisualAssembly, CompiledShotVisual, SourceInputResource, SourceTransportSpan,
        FilterFragment, AssApply, AssDocumentResource, BrandInputResource, SourceLayer,
        OutputTimeline, TimelineSpan, RasterSpan, CompiledCropTrack, CropPoint, FloatRect,
        PixelRect, PreparedBrandAsset, RenderRequirements, SourceTimeOrigin, StreamClock, DecodeLimits),
        text_limit=MAX_INPUT_TEXT_BYTES, entry_limit=MAX_METADATA_ENTRIES)
    origin_text = _guard_known_input(origin, record_types=(SourceTimeOrigin, StreamClock),
        text_limit=MAX_INPUT_TEXT_BYTES, entry_limit=MAX_METADATA_ENTRIES)
    if origin_text * 6 + 4096 > MAX_INPUT_TEXT_BYTES:
        raise RenderPreflightError("Origin canonical identity byte budget")
    if (type(origin.origin) is not Fraction or type(origin.duration) is not Fraction
            or type(origin.streams) is not tuple or any(type(s) is not StreamClock for s in origin.streams)
            or type(origin.color) is not tuple or any(type(pair) is not tuple or len(pair) != 2
                or any(type(v) is not str for v in pair) for pair in origin.color)
            or any(type(s.time_base) is not Fraction or any(value is not None and type(value) is not Fraction
                for value in (s.start, s.end)) for s in origin.streams)
            or any(type(getattr(limits, name)) is not Fraction for name in ("preroll", "postroll", "max_window"))):
        raise RenderPreflightError("Origin/limit record field types disagree")
    # Re-run immutable value constructors; dataclass construction alone is no validation.
    streams = [StreamClock(s.index, s.kind, s.time_base, s.start, s.end, s.width, s.height,
                           s.sample_rate, s.channels, s.sample_format) for s in origin.streams]
    checked_origin = SourceTimeOrigin(origin.path, origin.size, origin.mtime_ns, origin.fingerprint,
        origin.origin, origin.duration, streams, origin.decoder_id, dict(origin.color))
    if origin.record() != checked_origin.record() or origin.color != checked_origin.color:
        raise RenderPreflightError("Origin policy/identity metadata disagreement")
    DecodeLimits(**{name: getattr(limits, name) for name in limits.__dataclass_fields__})
    caller_limits = limits.record()
    for name, default in defaults.record().items():
        if exact(caller_limits[name]) > exact(default):
            raise RenderPreflightError("Caller decode limits exceed HOST ceilings")
    shot, requests = _source_slot_association(assembly, origin, slot_start, slot_end)
    supplied_receipt = decoded_window.receipt
    header_bytes = _guard_known_input((supplied_receipt.origin_id, supplied_receipt.decoder_id, supplied_receipt.policy),
        text_limit=MAX_RECEIPT_BYTES, entry_limit=MAX_METADATA_ENTRIES)
    payload_bytes = _utf8_size(supplied_receipt.payload_json, MAX_RECEIPT_BYTES)
    expansion = 1 if supplied_receipt.payload_json.isascii() else 6
    if payload_bytes * expansion + header_bytes * 6 + 1024 > MAX_RECEIPT_BYTES:
        raise RenderPreflightError("Receipt envelope byte budget")
    _bounded_receipt_payload(supplied_receipt.payload_json)
    if (type(decoded_window.frames) not in (tuple, list) or len(decoded_window.frames) > MAX_BATCH_REQUESTS
            or type(decoded_window.pcm) is not bytes):
        raise RenderPreflightError("Actual frame/PCM shape or count limit")
    frames = tuple(decoded_window.frames)
    actual_bytes = len(decoded_window.pcm)
    seen_payloads: set[int] = set()
    for frame in frames:
        if (type(frame) is not ObservedFrame or type(frame.tag) is not VideoTag or type(frame.bgr) is not bytes):
            raise RenderPreflightError("Expected immutable actual frame bodies")
        _guard_known_input((frame.tag, frame.raw_end), record_types=(VideoTag,))
        if id(frame) not in seen_payloads:
            actual_bytes += len(frame.bgr)
            seen_payloads.add(id(frame))
    if actual_bytes > min(MAX_RETAINED_BYTES, limits.max_retained_bytes, defaults.max_retained_bytes):
        raise RenderPreflightError("Actual retained BGR+PCM byte limit")
    supplied_json = supplied_receipt.to_json()
    _utf8_size(supplied_json, MAX_RECEIPT_BYTES)
    SourceDecodeReceipt.from_json(supplied_json, origin=origin, policy=DECODE_POLICY)
    envelope = json.loads(supplied_json)
    receipt_hash = envelope["sha256"]
    data = envelope["payload"]
    for name, maximum in caller_limits.items():
        if exact(data["limits"][name]) > exact(maximum):
            raise RenderPreflightError("Receipt limits wider than HOST limits")
    for start, end in data["windows"]:
        if not shot.span.source_start <= exact(start) < exact(end) <= shot.span.source_end:
            raise RenderPreflightError("Receipt window outside original consuming occurrence")
    outcomes = data["outcomes"]
    if (len(outcomes) != len(requests) or any(o["requested"] != rational(r.requested_source)
            or o["status"] != "covered" for o, r in zip(outcomes, requests))):
        raise RenderPreflightError("Receipt outcomes differ from exact ordered scheduled requests")
    selected = {o["frame_n"] for o in outcomes}
    if len(frames) != len(selected):
        raise RenderPreflightError("Actual frames must bijectively match selected rows")
    by_n: dict[int, ObservedFrame] = {}
    for frame in frames:
        if (type(frame) is not ObservedFrame or type(frame.tag) is not VideoTag
                or type(frame.tag.n) is not int or type(frame.bgr) is not bytes
                or frame.tag.n in by_n or frame.tag.n not in selected):
            raise RenderPreflightError("Duplicate/extra/malformed actual frame")
        by_n[frame.tag.n] = frame
    if type(decoded_window.pcm) is not bytes:
        raise RenderPreflightError("Expected immutable PCM bytes")
    retained = len(decoded_window.pcm) + sum(len(frame.bgr) for frame in frames)
    if retained > min(limits.max_retained_bytes, defaults.max_retained_bytes, MAX_RETAINED_BYTES):
        raise RenderPreflightError("Actual retained BGR+PCM byte limit")
    verified: set[int] = set()
    bindings = []
    for slot, request, outcome in zip(range(slot_start, slot_end), requests, outcomes):
        n = outcome["frame_n"]
        frame = by_n[n]
        raw = origin.to_raw(request.requested_source)
        if n not in verified:
            _validate_requested_frame(frame, data["video"][n], source_size=assembly.source_size,
                                      requested_raw=raw)
            verified.add(n)
        elif not frame.raw_start <= raw < frame.raw_end:
            raise RenderPreflightError("Requested raw time outside actual frame coverage")
        bindings.append(ScheduledFrameBinding(slot, request, raw, receipt_hash, n, frame))
    pcm_sha, pcm_bytes = _validate_materialized_pcm(decoded_window.pcm, data["audio"][0],
                                                   expected_audio_policy=expected_audio_policy)
    return BoundSourceBatch(slot_start, slot_end, origin.identity, origin.decoder_id, DECODE_POLICY,
                            receipt_hash, tuple(bindings), pcm_sha, pcm_bytes, retained)


_STREAM_BINDING_RECORDS = (
    CompiledVisualAssembly, CompiledShotVisual, SourceInputResource, SourceTransportSpan,
    FilterFragment, AssApply, AssDocumentResource, BrandInputResource, SourceLayer,
    OutputTimeline, TimelineSpan, RasterSpan, CompiledCropTrack, CropPoint, FloatRect,
    PixelRect, PreparedBrandAsset, RenderRequirements, SourceTimeOrigin, StreamClock, StreamDecodeLimits,
    VideoSampleRequest, SourceStreamBindingContext, SourceStreamBindingCursor,
    ProvisionalScheduledSlot, FinalizedScheduledSlot, FinalizedSourceBinding,
)


def _guard_stream_metadata(value: object) -> int:
    try:
        return _guard_known_input(value, record_types=_STREAM_BINDING_RECORDS,
            text_limit=MAX_INPUT_TEXT_BYTES, entry_limit=MAX_METADATA_ENTRIES)
    except AttributeError as exc:
        raise RenderPreflightError("Missing known stream metadata field") from exc


def _require_stream_fields(value: object, kind: type[Any]) -> None:
    """Only inspect declared slots of a known record; never walk frame payloads."""
    if type(value) is not kind or any(not hasattr(value, field.name) for field in fields(cast(Any, kind))):
        raise RenderPreflightError("Missing or foreign stream record fields")


def _stream_pixel_pair(value: object) -> None:
    if type(value) is not tuple or len(value) != 2 or any(type(n) is not int or n <= 0 for n in value):
        raise RenderPreflightError("Stream source requires immutable integer dimensions")


def _guard_stream_assembly(assembly: CompiledVisualAssembly, origin: SourceTimeOrigin,
                           limits: StreamDecodeLimits) -> int:
    """Known compiled shapes only; share already immutable metadata, never clone a graph."""
    _guard_stream_metadata((assembly, origin, limits))
    _stream_pixel_pair(assembly.source_size)
    if (type(assembly.timeline) is not OutputTimeline or type(assembly.timeline.spans) is not tuple
            or any(type(s) is not TimelineSpan for s in assembly.timeline.spans)
            or type(assembly.shots) is not tuple or any(type(s) is not CompiledShotVisual for s in assembly.shots)
            or type(assembly.ass_documents) is not tuple or type(assembly.brand_inputs) is not tuple
            or any(type(v) is not str for v in (assembly.expected_source_fingerprint, assembly.canonical_ass,
                                               assembly.compiler_metadata_json))):
        raise RenderPreflightError("Stream assembly requires known immutable compiled records")
    requirements = [assembly.requirements]
    for span in assembly.timeline.spans:
        for value in (span.source_start, span.source_end, span.output_start, span.output_end):
            if type(value) not in (int, float) or type(value) is int and value.bit_length() > 1023:
                raise RenderPreflightError("Timeline requires finite representable coordinates")
    for shot in assembly.shots:
        if (type(shot.source_layers) is not tuple or any(type(layer) is not SourceLayer for layer in shot.source_layers)
                or type(shot.steps) is not tuple or any(type(step) not in (FilterFragment, AssApply) for step in shot.steps)
                or type(shot.span) is not RasterSpan
                or any(type(v) is not Fraction for v in (shot.span.source_start, shot.span.source_end,
                    shot.span.output_start, shot.span.output_end))
                or any(type(v) is not int for v in (shot.span.occurrence_index, shot.span.k_start, shot.span.k_end))
                or any(type(v) is not str for v in (shot.span.shot_id, shot.output_label, shot.output_clock))
                or shot.source_input_label is not None and type(shot.source_input_label) is not str):
            raise RenderPreflightError("Stream shot requires immutable known spans/layers/steps")
        for step in shot.steps:
            if type(step) is FilterFragment:
                if (type(step.text) is not str or type(step.input_labels) is not tuple
                        or type(step.output_labels) is not tuple
                        or any(type(s) is not str for s in (*step.input_labels, *step.output_labels))):
                    raise RenderPreflightError("Stream filter fragment requires immutable labels/text")
            else:
                apply = cast(AssApply, step)
                if any(type(v) is not str for v in (apply.document_key, apply.input_label, apply.output_label)):
                    raise RenderPreflightError("Stream ASS application requires string identities")
        for layer in shot.source_layers:
            requirements.append(layer.requirements)
            if (type(layer.destination) is not PixelRect or type(layer.kind) is not LayerKind
                    or any(type(v) is not str for v in (layer.shot_id, layer.layer_id, layer.output_label, layer.filter_complex))
                    or any(type(v) is not int for v in (layer.declaration_index, layer.z_index))
                    or layer.input_label is not None and type(layer.input_label) is not str
                    or any(type(v) is not int for v in (layer.destination.x, layer.destination.y,
                                                       layer.destination.width, layer.destination.height))):
                raise RenderPreflightError("Stream source layer requires known immutable fields")
            track = layer.crop_track
            if track is not None:
                if (type(track) is not CompiledCropTrack or type(track.points) is not tuple
                        or any(type(point) is not CropPoint for point in track.points)
                        or type(track.fps) is not int or type(track.mode) is not str):
                    raise RenderPreflightError("Stream crop track requires immutable points")
                _stream_pixel_pair(track.source_size)
                for point in track.points:
                    if (type(point.rect) is not FloatRect or type(point.frame) not in (int, float)
                            or type(point.easing) is not str
                            or any(type(v) not in (int, float) for v in point.rect.values)):
                        raise RenderPreflightError("Stream crop point requires known scalar geometry")
    for document in assembly.ass_documents:
        if (type(document) is not AssDocumentResource or type(document.key) is not str
                or type(document.document) is not str or type(document.event_indices) is not tuple
                or any(type(n) is not int for n in document.event_indices) or type(document.windows) is not tuple
                or any(type(pair) is not tuple or len(pair) != 2 or any(type(n) is not int for n in pair)
                       for pair in document.windows)):
            raise RenderPreflightError("Stream ASS resource requires immutable fields")
    for resource in assembly.brand_inputs:
        if (type(resource) is not BrandInputResource or type(resource.asset) is not PreparedBrandAsset
                or type(resource.placement) is not PixelRect
                or any(type(v) is not str for v in (resource.key, resource.graph_label, resource.shot_id, resource.required_asset_id))
                or type(resource.brand_index) is not int
                or any(type(v) is not int for v in (resource.placement.x, resource.placement.y,
                                                   resource.placement.width, resource.placement.height))):
            raise RenderPreflightError("Stream brand resource requires known immutable fields")
        for pair in (resource.raster_window, resource.declared_interval, resource.execution_interval,
                     resource.asset.source_size, resource.asset.source_bbox, resource.asset.size):
            if type(pair) is not tuple or any(type(v) not in (int, Fraction) for v in pair):
                raise RenderPreflightError("Stream brand resource requires immutable coordinates")
        if (not isinstance(resource.asset.source_path, Path) or not isinstance(resource.asset.path, Path)
                or type(resource.asset.source_sha256) is not str):
            raise RenderPreflightError("Stream brand identity requires immutable path/text")
    for requirement in requirements:
        if (type(requirement) is not RenderRequirements or type(requirement.runtime_id) is not str
                or type(requirement.needs_vulkan) is not bool or type(requirement.required_filters) is not tuple
                or any(type(s) is not str for s in requirement.required_filters)):
            raise RenderPreflightError("Stream runtime requirements require immutable fields")
    source = assembly.source_input
    if type(source) is SourceInputResource:
        _stream_pixel_pair(source.source_size)
    if (type(origin.streams) is not tuple or any(type(s) is not StreamClock for s in origin.streams)
            or type(origin.color) is not tuple or any(type(pair) is not tuple or len(pair) != 2
                or any(type(v) is not str for v in pair) for pair in origin.color)
            or type(origin.origin) is not Fraction or type(origin.duration) is not Fraction
            or any(type(s.time_base) is not Fraction or any(v is not None and type(v) is not Fraction
                for v in (s.start, s.end)) for s in origin.streams)):
        raise RenderPreflightError("Stream origin requires immutable exact clocks/color")
    origin_text = _guard_known_input(origin, record_types=(SourceTimeOrigin, StreamClock),
        text_limit=MAX_INPUT_TEXT_BYTES, entry_limit=MAX_METADATA_ENTRIES)
    if origin_text * 6 + 4096 > MAX_INPUT_TEXT_BYTES:
        raise RenderPreflightError("Origin canonical identity byte budget")
    streams = tuple(StreamClock(s.index, s.kind, s.time_base, s.start, s.end, s.width, s.height,
                                s.sample_rate, s.channels, s.sample_format) for s in origin.streams)
    checked_origin = SourceTimeOrigin(origin.path, origin.size, origin.mtime_ns, origin.fingerprint,
        origin.origin, origin.duration, streams, origin.decoder_id, dict(origin.color))
    if origin != checked_origin:
        raise RenderPreflightError("Stream origin constructor/policy disagreement")
    if len(streams) != 2 or sum(s.kind == "audio" for s in streams) != 1:
        raise RenderPreflightError("Stream requires one mixed audio and one video clock")
    if any(type(getattr(limits, name)) is not Fraction for name in ("preroll", "postroll", "max_window")):
        raise RenderPreflightError("Stream limits require exact Fraction margins")
    StreamDecodeLimits(**{name: getattr(limits, name) for name in limits.__dataclass_fields__})
    return origin_text


def begin_source_stream_binding(assembly: CompiledVisualAssembly, *, slot_start: int, slot_end: int,
        origin: SourceTimeOrigin, session_id: str, window: tuple[Fraction, Fraction],
        expected_audio_policy: Literal["analysis16k-mono", "render48k-mono", "render48k-stereo"],
        limits: StreamDecodeLimits = StreamDecodeLimits()) -> SourceStreamBindingCursor:
    """Capture a pure binding value. HOST owns live identity and latest-cursor linearity."""
    if (type(assembly) is not CompiledVisualAssembly or type(origin) is not SourceTimeOrigin
            or type(limits) is not StreamDecodeLimits or type(slot_start) is not int
            or type(slot_end) is not int or not 0 <= slot_start < slot_end
            or type(window) is not tuple or len(window) != 2
            or any(type(t) is not Fraction for t in window)
            or type(expected_audio_policy) is not str
            or expected_audio_policy not in ("analysis16k-mono", "render48k-mono", "render48k-stereo")):
        raise RenderPreflightError("Invalid stream binding records/slots/window/policy")
    _stream_session(session_id)
    origin_text = _guard_stream_assembly(assembly, origin, limits)
    _guard_stream_metadata((slot_start, slot_end, window))
    if slot_end - slot_start > min(limits.max_requests, limits.max_frames, MAX_BATCH_REQUESTS):
        raise RenderPreflightError("Stream request count limit")
    if assembly.source_size[0] * assembly.source_size[1] > limits.max_pixels:
        raise RenderPreflightError("Stream source pixel limit")
    shot, requests = _source_slot_association(assembly, origin, slot_start, slot_end)
    if (not shot.span.source_start <= window[0] < window[1] <= shot.span.source_end
            or any(not window[0] <= r.requested_source < window[1] for r in requests)):
        raise RenderPreflightError("Stream window must cover exact requests inside their original occurrence")
    start, end = _decoder_interval(origin, Window(*window), limits)
    payload = dict(window=[rational(t) for t in window],
        requests=[rational(r.requested_source) for r in requests], operation_limits=limits.operation_record(),
        delivery_limits=limits.delivery_record(), decoder_start=rational(start), decoder_end=rational(end),
        audio=dict(policy=expected_audio_policy))
    # Reserve the combined payload + origin expansion before identity hashing.
    _stream_guard(payload, MAX_INPUT_TEXT_BYTES - origin_text * 6 - 4096)
    header_value = _stream_header(origin, session_id, payload)
    _stream_guard(header_value, MAX_INPUT_TEXT_BYTES)
    header = canonical(header_value)
    assert shot.transport is not None
    context = SourceStreamBindingContext(assembly, origin, assembly.source_size,
        assembly.expected_source_fingerprint, shot.span, shot.transport, slot_start, slot_end,
        session_id, STREAM_DECODE_POLICY, expected_audio_policy, window, requests, limits, header)
    _guard_stream_metadata(context)
    return SourceStreamBindingCursor(context, 0, 0, None, digest(header), 0, 0)


def _checked_stream_cursor(cursor: SourceStreamBindingCursor) -> SourceStreamBindingContext:
    if type(cursor) is not SourceStreamBindingCursor:
        raise RenderPreflightError("Expected stream binding cursor/context")
    _guard_stream_metadata(cursor)
    if type(cursor.context) is not SourceStreamBindingContext:
        raise RenderPreflightError("Expected stream binding context")
    context = cursor.context
    if (type(context.requests) is not tuple or any(type(r) is not VideoSampleRequest for r in context.requests)
            or type(context.source_size) is not tuple or type(context.transport) is not SourceTransportSpan
            or type(context.occurrence) is not RasterSpan
            or any(type(v) is not int or v < 0 for v in (cursor.next_delivery_index, cursor.next_request_ordinal,
                cursor.selected_unique_bytes, cursor.request_emission_bytes))
            or cursor.last_selected_frame_n is not None and
                (type(cursor.last_selected_frame_n) is not int or cursor.last_selected_frame_n < 0)
            or type(cursor.chain_sha256) is not str or re.fullmatch(r"[0-9a-f]{64}", cursor.chain_sha256) is None):
        raise RenderPreflightError("Stream cursor requires bounded immutable exact fields")
    _stream_pixel_pair(context.source_size)
    if (any(type(v) is not int for v in (context.transport.k_start, context.transport.k_end,
            context.transport.slot_start, context.transport.slot_end, context.occurrence.occurrence_index,
            context.occurrence.k_start, context.occurrence.k_end))
            or any(type(v) is not Fraction for v in (context.occurrence.source_start, context.occurrence.source_end,
                context.occurrence.output_start, context.occurrence.output_end))
            or type(context.transport.shot_id) is not str or type(context.occurrence.shot_id) is not str):
        raise RenderPreflightError("Captured transport/raster fields require exact types")
    for request in context.requests:
        if (any(type(v) is not int for v in (request.global_k, request.occurrence_index, request.local_n))
                or type(request.requested_source) is not Fraction or type(request.shot_id) is not str
                or type(request.request_rule) is not str):
            raise RenderPreflightError("Captured A requests require exact field types")
    initial = begin_source_stream_binding(context.assembly, slot_start=context.slot_start,
        slot_end=context.slot_end, origin=context.origin, session_id=context.session_id,
        window=context.window, expected_audio_policy=cast(Literal["analysis16k-mono", "render48k-mono",
            "render48k-stereo"], context.expected_audio_policy), limits=context.limits)
    if context != initial.context:
        raise RenderPreflightError("Captured source context changed")
    frame_bytes = context.source_size[0] * context.source_size[1] * 3
    if (cursor.next_delivery_index > min(context.limits.max_deliveries, MAX_BATCH_REQUESTS)
            or cursor.next_request_ordinal > len(context.requests)
            or cursor.next_delivery_index > cursor.next_request_ordinal
            or bool(cursor.next_delivery_index) != bool(cursor.next_request_ordinal)
            or cursor.selected_unique_bytes != cursor.next_delivery_index * frame_bytes
            or cursor.request_emission_bytes != cursor.next_request_ordinal * frame_bytes
            or cursor.selected_unique_bytes > context.limits.max_selected_unique_bytes
            or cursor.request_emission_bytes > context.limits.max_request_emission_bytes
            or (cursor.next_delivery_index == 0) != (cursor.last_selected_frame_n is None)
            or cursor.last_selected_frame_n is not None and not
                cursor.next_delivery_index - 1 <= cursor.last_selected_frame_n < context.limits.max_frames - 1
            or not cursor.next_delivery_index and cursor.chain_sha256 != initial.chain_sha256):
        raise RenderPreflightError("Stream cursor progress/counters disagree")
    return context


def bind_source_stream_selection(cursor: SourceStreamBindingCursor, *,
        selection: ProvisionalSelection) -> tuple[SourceStreamBindingCursor, ProvisionalSourceDelivery]:
    """Borrow one checked frame; no stream operations or global single-use guarantee."""
    context = _checked_stream_cursor(cursor)
    _require_stream_fields(selection, ProvisionalSelection)
    _require_stream_fields(selection.frame, ObservedFrame)
    _require_stream_fields(selection.frame.tag, VideoTag)
    if (type(selection) is not ProvisionalSelection or type(selection.frame) is not ObservedFrame
            or type(selection.frame.tag) is not VideoTag or type(selection.frame.bgr) is not bytes
            or type(selection.request_ordinals) is not tuple or type(selection.session_id) is not str
            or type(selection.delivery_index) is not int or type(selection.delivery_chain_sha256) is not str
            or len(selection.request_ordinals) > min(context.limits.max_requests, context.limits.max_frames, MAX_BATCH_REQUESTS)
            or any(type(n) is not int for n in selection.request_ordinals)):
        raise RenderPreflightError("Expected immutable known selection/frame/ordinal fields")
    frame, tag = selection.frame, selection.frame.tag
    _guard_known_input((tag, frame.raw_end, selection.request_ordinals, selection.session_id,
        selection.delivery_index, selection.delivery_chain_sha256), record_types=(VideoTag,),
        text_limit=MAX_INPUT_TEXT_BYTES, entry_limit=MAX_METADATA_ENTRIES)
    if (any(type(v) is not int for v in (tag.n, tag.pts, tag.width, tag.height, tag.checksum))
            or not 0 <= tag.n < context.limits.max_frames - 1
            or tag.width <= 0 or tag.height <= 0 or not 0 <= tag.checksum <= 0xFFFFFFFF
            or type(tag.time_base) is not Fraction or tag.time_base <= 0 or type(frame.raw_end) is not Fraction
            or (tag.width, tag.height) != context.source_size or tag.width * tag.height > context.limits.max_pixels
            or len(frame.bgr) != tag.width * tag.height * 3
            or len(frame.bgr) > context.limits.max_live_payload_bytes):
        raise RenderPreflightError("Actual source frame dimensions/body/live payload limit")
    if (selection.session_id != context.session_id or selection.delivery_index != cursor.next_delivery_index
            or not selection.request_ordinals
            or selection.request_ordinals != tuple(range(cursor.next_request_ordinal,
                cursor.next_request_ordinal + len(selection.request_ordinals)))
            or cursor.next_request_ordinal + len(selection.request_ordinals) > len(context.requests)
            or cursor.last_selected_frame_n is not None and selection.frame.tag.n <= cursor.last_selected_frame_n):
        raise RenderPreflightError("Stream delivery/session/ordinal/frame order disagreement")
    unique = cursor.selected_unique_bytes + len(frame.bgr)
    emission = cursor.request_emission_bytes + len(frame.bgr) * len(selection.request_ordinals)
    if (cursor.next_delivery_index >= min(context.limits.max_deliveries, MAX_BATCH_REQUESTS)
            or unique > context.limits.max_selected_unique_bytes
            or emission > context.limits.max_request_emission_bytes):
        raise RenderPreflightError("Stream cumulative delivery/payload limits")
    selected = _bounded_json_metadata(selection.selected_row_json, cap=min(MAX_RECEIPT_BYTES, 16384), max_depth=10)
    successor = _bounded_json_metadata(selection.successor_row_json, cap=min(MAX_RECEIPT_BYTES, 16384), max_depth=10)
    checked = ProvisionalSelection(selection.session_id, selection.delivery_index,
        selection.request_ordinals, selection.frame, selected, successor,
        previous_chain_sha256=cursor.chain_sha256)
    if checked != selection:
        raise RenderPreflightError("Provisional row/body/chain disagreement")
    requests = tuple(context.requests[n] for n in checked.request_ordinals)
    _validate_requested_frame(checked.frame, selected, source_size=context.source_size,
        requested_raw=context.origin.to_raw(requests[0].requested_source))
    slots = []
    for ordinal, request in zip(checked.request_ordinals, requests):
        raw = context.origin.to_raw(request.requested_source)
        if not checked.frame.raw_start <= raw < checked.frame.raw_end:
            raise RenderPreflightError("Requested raw time outside checked frame interval")
        slots.append(ProvisionalScheduledSlot(context.slot_start + ordinal, request, raw))
    _guard_stream_metadata(tuple(slots))
    updated = SourceStreamBindingCursor(context, cursor.next_delivery_index + 1,
        cursor.next_request_ordinal + len(slots), checked.frame.tag.n, checked.delivery_chain_sha256,
        unique, emission)
    _guard_stream_metadata((updated, tuple(slots)))
    return updated, ProvisionalSourceDelivery(context.session_id, checked.delivery_index,
        checked.delivery_chain_sha256, checked.frame.tag.n, checked.frame, tuple(slots))


def finalize_source_stream_binding(cursor: SourceStreamBindingCursor, *,
        finalization: SourceStreamFinalization) -> FinalizedSourceBinding:
    """Join actual loader-validated final metadata; absent PCM remains producer-reported."""
    context = _checked_stream_cursor(cursor)
    if (cursor.next_request_ordinal != len(context.requests) or not cursor.next_delivery_index
            or type(finalization) is not SourceStreamFinalization):
        raise RenderPreflightError("Stream finalization requires a complete binding value")
    _require_stream_fields(finalization, SourceStreamFinalization)
    if any(type(v) is not str for v in (finalization.origin_id, finalization.decoder_id,
            finalization.session_id, finalization.policy, finalization.payload_json)):
        raise RenderPreflightError("Stream finalization requires plain metadata strings")
    cap = min(context.limits.max_finalization_bytes, MAX_RECEIPT_BYTES)
    header_bytes = _guard_known_input((finalization.origin_id, finalization.decoder_id,
        finalization.session_id, finalization.policy), text_limit=cap, entry_limit=MAX_METADATA_ENTRIES)
    payload_bytes = _utf8_size(finalization.payload_json, cap)
    expansion = 1 if finalization.payload_json.isascii() else 6
    if payload_bytes * expansion + header_bytes * 6 + 1024 > cap:
        raise RenderPreflightError("Stream finalization envelope byte budget")
    _bounded_json_metadata(finalization.payload_json, cap=cap, max_depth=10)
    checked = SourceStreamFinalization.from_json(finalization.to_json(), origin=context.origin,
        policy=STREAM_DECODE_POLICY, limits=context.limits, expected_session_id=context.session_id)
    data = json.loads(checked.payload_json)
    header = canonical(_stream_header(context.origin, context.session_id, data))
    deliveries, accounting = data["deliveries"], data["accounting"]
    if (header != context.chain_header_json or checked.limits != context.limits
            or deliveries[-1]["chain_sha256"] != cursor.chain_sha256
            or len(deliveries) != cursor.next_delivery_index
            or len(data["requests"]) != cursor.next_request_ordinal
            or deliveries[-1]["frame_n"] != cursor.last_selected_frame_n
            or accounting["selected_unique_bytes"] != cursor.selected_unique_bytes
            or accounting["acknowledged_request_bytes"] != cursor.request_emission_bytes):
        raise RenderPreflightError("Final metadata disagrees with captured context or cursor")
    slots = tuple(FinalizedScheduledSlot(context.slot_start + n, r,
        context.origin.to_raw(r.requested_source), data["outcomes"][n]["frame_n"])
        for n, r in enumerate(context.requests))
    audio = data["audio"]
    final = FinalizedSourceBinding(context.slot_start, context.slot_end, checked.origin_id,
        checked.decoder_id, checked.session_id, checked.policy, checked.sha256, cursor.chain_sha256,
        slots, cursor.selected_unique_bytes, cursor.request_emission_bytes, audio["policy"],
        audio["pcm_sha256"], audio["bytes"], audio["coverage"][0]["status"])
    _guard_stream_metadata((context, final))
    return final


def resolve_visual_plan(plan: VisualCompositionPlan) -> VisualCompositionPlan:
    """Return a detached validated plan with explicit same-layer inherited fills.

    Only the immediately preceding shot participates. Fill-specific color/blur
    strength travels with an inherited fill; explicit current blur strength wins.
    Caller-owned plans remain untouched. This does not approve director output.
    """
    try:
        plan = VisualCompositionPlan.model_validate(_strict_model_input(plan))
        OutputTimeline.from_shots(plan.shots)
        previous: dict[str, VisualLayer] = {}
        for shot in plan.shots:
            if shot.transition_in != "cut" or shot.transition_out != "cut":
                raise ValueError("Unsupported transition; only cut is implemented")
            for layer in shot.layers:
                if layer.fill == FillMode.INHERIT:
                    predecessor = previous.get(layer.layer_id)
                    if predecessor is None:
                        raise ValueError(f"inherit missing predecessor: shot={shot.shot_id} "
                                         f"layer={layer.layer_id}")
                    layer.fill = predecessor.fill
                    layer.solid_color = predecessor.solid_color
                    if layer.blur_sigma is None:
                        layer.blur_sigma = predecessor.blur_sigma
                if layer.kind not in (LayerKind.SOURCE_VIDEO, LayerKind.STREAMER_CAMERA,
                                      LayerKind.SOLID):
                    raise ValueError(f"Unsupported layer kind: {layer.kind}")
                if ((layer.fill == FillMode.SOLID) != (layer.kind == LayerKind.SOLID)):
                    raise ValueError("solid fill is exclusive to solid layer kind")
                if layer.opacity != 1:
                    raise ValueError("Unsupported layer opacity; expected 1")
                if layer.kind != LayerKind.STREAMER_CAMERA and layer.source_ref != "source":
                    raise ValueError("source_video/solid requires source_ref=source")
                if layer.kind == LayerKind.STREAMER_CAMERA and not re.fullmatch(
                        r"[a-z][a-z0-9_-]{0,63}", layer.source_ref):
                    raise ValueError("Camera source_ref must be local track provenance")
                if layer.fill == FillMode.CONTAIN_BLUR and layer.blur_sigma is None:
                    raise ValueError("contain_blur requires explicit blur_sigma")
            previous = {layer.layer_id: layer for layer in shot.layers}
        return VisualCompositionPlan.model_validate(
            _prepare_strict_visual_input(VisualCompositionPlan, plan))
    except (ValidationError, ValueError) as exc:
        raise RenderPreflightError(f"Invalid v2 render plan: {exc}") from exc


def compile_visual_layers(plan: VisualCompositionPlan, *, source_size: tuple[int, int],
                          input_label: str = "0:v") -> tuple[CompiledShotLayers, ...]:
    """Pure static shot fragments, in plan order; host must also run preflight.

    Fragments intentionally remain separate for later motion/ASS/audio and
    source-branch/concat assembly. Original source start anchors shot-local PTS;
    no independent first-decoded-frame rebasing or source-window rounding.
    """
    resolved = resolve_visual_plan(plan)
    try:
        return tuple(compile_shot_layers(shot, source_size=source_size, input_label=input_label)
                     for shot in resolved.shots)
    except ValueError as exc:
        raise RenderPreflightError(str(exc)) from exc


def compile_plan_audio(plan: VisualCompositionPlan, *, audio_format: AudioFormat,
                       assets: RenderAssetRegistry | None = None,
                       cue_formats: Mapping[str, AudioFormat] | None = None,
                       binding: SourceClockBinding | None = None,
                       runtime: RenderRuntime | None = None,
                       receipt: NormalizationReceipt | None = None,
                       input_label: str = "0:a", cue_input_start: int = 1) -> CompiledAudio:
    """Pure audio bridge; carry combined requirements to the one render executor."""
    resolved = resolve_visual_plan(plan)
    visual_requirements = requirements_for_plan(resolved)
    if runtime is not None and visual_requirements.needs_vulkan and (
            runtime.runtime_id != visual_requirements.runtime_id or runtime.device_index is None):
        raise RenderPreflightError("Native audio/video assembly requires the same explicit paired runtime/device")
    result = compile_audio(resolved, audio_format=audio_format, assets=assets,
        cue_formats=cue_formats, binding=binding, runtime=runtime, receipt=receipt,
        input_label=input_label, cue_input_start=cue_input_start)
    return replace(result, requirements=combine_render_requirements(
        visual_requirements, result.requirements))


def _inspect(command: str, arguments: Sequence[str], *, runtime: RenderRuntime | None = None,
             _runtime_files: "RetainedFiles | None" = None, _tool_runner: RuntimeToolRunner | None = None) -> str:
    if (_runtime_files is None) != (_tool_runner is None):
        raise RenderPreflightError("Owned inspection requires paired file and tool services")
    if _tool_runner is not None and runtime is None:
        raise RenderPreflightError("Owned inspection requires explicit paired runtime")
    if runtime is None:
        executable = shutil.which(command)
    else:
        if command not in ("ffmpeg", "ffprobe"):
            raise RenderPreflightError("Unknown paired runtime command")
        try:
            _verify_pins(runtime, **({} if _runtime_files is None else {"_files": _runtime_files}))
        except RenderRuntimeError as exc:
            raise RenderPreflightError(str(exc)) from exc
        executable = str(getattr(runtime, command))
    if executable is None:
        raise RenderPreflightError(f"{command} not found in PATH")
    try:
        if _tool_runner is None:
            result = subprocess.run([executable, *arguments], capture_output=True, text=True,
                                    timeout=15, check=False)
        else:
            raw = _tool_runner((executable, *arguments), timeout=15,
                max_stdout_bytes=1_000_000, max_stderr_bytes=1_000_000)
            result = subprocess.CompletedProcess(raw.args, raw.returncode,
                raw.stdout.decode(errors="replace"), raw.stderr.decode(errors="replace"))
    except (OSError, subprocess.SubprocessError) as exc:
        raise RenderPreflightError(f"{command} inspection failed: {exc}") from exc
    if result.returncode:
        raise RenderPreflightError(f"{command} inspection failed ({result.returncode}): "
                                   f"{result.stderr.strip()}")
    return result.stdout or result.stderr


def binary_version(command: str, *, runtime: RenderRuntime | None = None,
                   _runtime_files: "RetainedFiles | None" = None, _tool_runner: RuntimeToolRunner | None = None) -> str:
    options: dict[str, Any] = {}
    if _runtime_files is not None:
        options["_runtime_files"] = _runtime_files
    if _tool_runner is not None:
        options["_tool_runner"] = _tool_runner
    output = _inspect(command, ("-version",), runtime=runtime, **options)
    lines = output.splitlines()
    if not lines:
        raise RenderPreflightError(f"{command} returned no version")
    return lines[0]


def available_encoders(*, runtime: RenderRuntime | None = None,
                       _runtime_files: "RetainedFiles | None" = None, _tool_runner: RuntimeToolRunner | None = None) -> set[str]:
    """Binary capability listing only; never establishes working GPU hardware."""
    options: dict[str, Any] = {}
    if _runtime_files is not None:
        options["_runtime_files"] = _runtime_files
    if _tool_runner is not None:
        options["_tool_runner"] = _tool_runner
    output = _inspect("ffmpeg", ("-hide_banner", "-encoders"), runtime=runtime, **options)
    return {match.group(1) for line in output.splitlines()
            if (match := re.match(r"^\s*[VAS][A-Z.]{5}\s+(\w+)\s", line))}


def select_encoder(encoder: EncoderName | str, *, runtime: RenderRuntime | None = None,
                   _runtime_files: "RetainedFiles | None" = None, _tool_runner: RuntimeToolRunner | None = None) -> EncoderName:
    try:
        selected = EncoderName(encoder)
    except ValueError as exc:
        raise RenderPreflightError(f"Unsupported encoder: {encoder}") from exc
    options: dict[str, Any] = {} if runtime is None else {"runtime": runtime}
    if _runtime_files is not None:
        options["_runtime_files"] = _runtime_files
    if _tool_runner is not None:
        options["_tool_runner"] = _tool_runner
    detected = available_encoders(**options)
    if selected.value not in detected:
        raise RenderPreflightError(f"Encoder {selected.value} unavailable; detected={sorted(detected)}")
    return selected


def encoder_arguments(profile: RenderProfile) -> tuple[str, ...]:
    """Pure output argument profile; availability is checked separately."""
    profile = RenderProfile.model_validate(profile.model_dump())
    video: tuple[str, ...]
    if profile.video_encoder == EncoderName.H264_NVENC:
        video = ("-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr",
                 "-cq", "19", "-b:v", "0")
    else:
        video = ("-c:v", "libx264", "-preset", "medium", "-crf", "18")
    return (*video, "-r", str(profile.fps), "-pix_fmt", profile.pixel_format,
            "-movflags", "+faststart", "-c:a", profile.audio_codec,
            "-b:a", profile.audio_bitrate, "-ar", str(profile.audio_rate))


def _probe(path: Path, *, runtime: RenderRuntime | None = None,
           _runtime_files: "RetainedFiles | None" = None, _tool_runner: RuntimeToolRunner | None = None) -> dict[str, object]:
    options: dict[str, Any] = {}
    if _runtime_files is not None:
        options["_runtime_files"] = _runtime_files
    if _tool_runner is not None:
        options["_tool_runner"] = _tool_runner
    output = _inspect("ffprobe", ("-v", "error", "-show_entries",
        "stream=codec_type,width,height,start_time,duration,color_transfer,color_primaries,"
        "color_space,color_range,pix_fmt,chroma_location,sample_rate,channels,channel_layout:"
        "format=start_time,duration",
        "-of", "json", str(path)), runtime=runtime, **options)
    try:
        payload: object = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RenderPreflightError(f"ffprobe returned invalid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise RenderPreflightError(f"ffprobe returned invalid metadata: {path}")
    return payload


def _streams(payload: dict[str, object]) -> list[dict[str, object]]:
    streams = payload.get("streams")
    if not isinstance(streams, list) or any(not isinstance(item, dict) for item in streams):
        raise RenderPreflightError("ffprobe returned invalid streams")
    return [item for item in streams if isinstance(item, dict)]


def probe_audio_format(path: Path, *, runtime: RenderRuntime | None,
                       _runtime_files: "RetainedFiles | None" = None, _tool_runner: RuntimeToolRunner | None = None) -> AudioFormat:
    """Qualify an explicitly PREPARED 48k PCM/graph input, not raw-media support.

    Do not run this on arbitrary source media to infer supported input rates.
    Task6 owns original metadata, source resampling phase and the shared clock
    binding, then probes prepared input. Unknown/different layouts need an
    explicit qualified mapping; this helper only admits mono/stereo at 48k.
    """
    options: dict[str, Any] = {}
    if _runtime_files is not None:
        options["_runtime_files"] = _runtime_files
    if _tool_runner is not None:
        options["_tool_runner"] = _tool_runner
    streams = [s for s in _streams(_probe(path, runtime=runtime, **options)) if s.get("codec_type") == "audio"]
    if len(streams) != 1:
        raise RenderPreflightError("Qualified audio format requires one mixed stream")
    stream = streams[0]
    try:
        rate = _number(stream.get("sample_rate"), "audio format sample rate")
        channels = _number(stream.get("channels"), "audio format channels")
        layout = stream.get("channel_layout")
        if not rate.is_integer() or not isinstance(layout, str):
            raise ValueError("Missing audio format layout/rate")
        fmt = AudioFormat(int(rate), layout)
        if channels != fmt.channels:
            raise ValueError("Contradictory audio format channel count/layout")
        return fmt
    except ValueError as exc:
        raise RenderPreflightError(f"Unqualified audio format: {exc}") from exc


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int, str)):
        raise RenderPreflightError(f"Invalid {label}")
    try:
        number = float(value)
    except ValueError as exc:
        raise RenderPreflightError(f"Invalid {label}") from exc
    if not math.isfinite(number) or number <= 0:
        raise RenderPreflightError(f"Invalid {label}")
    return number


def _source_span(component: dict[str, object], label: str) -> tuple[Fraction, Fraction]:
    """Use reported source timestamps, never guess a missing bound or rebase it.

    FFprobe duration is a length, not an absolute end timestamp. JSON may omit
    unavailable fields; container duration cannot prove an unknown stream span.
    """
    _number(component.get("duration"), f"{label} duration")
    value = component.get("start_time")
    if isinstance(value, bool) or not isinstance(value, (float, int, str)):
        raise RenderPreflightError(f"Missing or invalid {label} start_time")
    try:
        start = float(value)
    except (ValueError, OverflowError) as exc:
        raise RenderPreflightError(f"Missing or invalid {label} start_time") from exc
    if not math.isfinite(start):
        raise RenderPreflightError(f"Missing or invalid {label} start_time")
    exact_start = exact(str(value))
    end = exact_start + exact(str(component["duration"]))
    if end <= exact_start:
        raise RenderPreflightError(f"Invalid {label} duration/span")
    return exact_start, end


def _validate_subtitle_data(plan: VisualCompositionPlan,
                            transcript: Sequence[TranscriptSegment] | None) -> None:
    if not any(shot.subtitles is not None for shot in plan.shots):
        return
    if transcript is None:
        raise RenderPreflightError("Missing subtitle data for planned subtitles")
    previous_start = -1.0
    for segment in transcript:
        if (not math.isfinite(segment.start) or not math.isfinite(segment.end)
                or segment.start < 0 or segment.end <= segment.start
                or segment.start < previous_start or not isinstance(segment.text, str)):
            raise RenderPreflightError("Invalid subtitle segment timing/text")
        previous_start = segment.start
        word_start = segment.start
        for word in segment.words:
            if (not math.isfinite(word.start) or not math.isfinite(word.end)
                    or word.start < word_start or word.end <= word.start
                    or word.end > segment.end or not isinstance(word.word, str)):
                raise RenderPreflightError("Invalid subtitle word timing/text")
            word_start = word.start
        if any(shot.subtitles is not None and shot.subtitles.emphasize_active_word
               and segment.start < shot.source.end and segment.end > shot.source.start
               for shot in plan.shots) and segment.text.strip() and not segment.words:
            raise RenderPreflightError("Active-word subtitles require word subtitle data")


def preflight_visual_plan(plan: VisualCompositionPlan, *, source: Path,
                          assets: RenderAssetRegistry, profile: RenderProfile | None = None,
                          transcript: Sequence[TranscriptSegment] | None = None,
                          runtime: RenderRuntime | None = None,
                          device_index: int = 0, _runtime_files: "RetainedFiles | None" = None,
                          _tool_runner: RuntimeToolRunner | None = None,
                          _assembly: CompiledVisualAssembly | None = None) -> RuntimeQualification | None:
    """Validate a host-supplied v2 plan before any encoding can start.

    Revalidates even mutated/model_construct instances. Native motion also runs
    an actual bounded Vulkan capability frame AFTER source/geometry validation.
    An explicit empty transcript represents
    silence; missing subtitle data does not. Camera track IDs are provenance,
    never file inputs or instructions to infer new crops.
    """
    if (_runtime_files is None) != (_tool_runner is None):
        raise RenderPreflightError("Owned preflight requires paired file and tool services")
    if _tool_runner is not None and runtime is None:
        raise RenderPreflightError("Owned preflight requires explicit paired runtime")
    try:
        plan = resolve_visual_plan(plan)
        profile = RenderProfile.model_validate((profile or RenderProfile()).model_dump())
    except (ValidationError, ValueError) as exc:
        raise RenderPreflightError(f"Invalid v2 render plan/profile: {exc}") from exc
    file_options = {} if _runtime_files is None else {"_files": _runtime_files}
    tool_options: dict[str, Any] = {}
    if _runtime_files is not None:
        tool_options["_runtime_files"] = _runtime_files
    if _tool_runner is not None:
        tool_options["_tool_runner"] = _tool_runner
    sfx_paths: set[Path] = set()
    for shot in plan.shots:
        try:
            if shot.brand is not None:
                if shot.brand.asset_id is None:
                    raise ValueError("brand requires a registered brand asset ID")
                if shot.brand.font_asset_id is not None:
                    raise ValueError("brand PNG does not support a font asset ID")
                validate_asset_loadability("brand", assets.resolve("brand", shot.brand.asset_id), **file_options)
            if shot.title is not None:
                if shot.title.text is None:
                    raise ValueError("title requires text")
                validate_asset_loadability("font", assets.resolve(
                    "font", shot.title.font_asset_id or "headline_impact"), **file_options)
            if shot.subtitles is not None:
                validate_asset_loadability("font", assets.resolve(
                    "font", shot.subtitles.font_asset_id), **file_options)
            for cue in shot.audio.cues:
                path = assets.resolve("sfx", cue.asset_id)
                validate_asset_loadability("sfx", path, **file_options)
                sfx_paths.add(path)
        except (OSError, ValueError) as exc:
            raise RenderPreflightError(str(exc)) from exc
    _validate_subtitle_data(plan, transcript)
    source = Path(source).resolve() if _runtime_files is None else _runtime_files.resolve_path(Path(source))
    if not (source.is_file() if _runtime_files is None else stat.S_ISREG(_runtime_files.stat_path(source).st_mode)):
        raise RenderPreflightError(f"Source file missing: {source}")
    if source_fingerprint(source, **file_options) != plan.source_fingerprint:
        raise RenderPreflightError("source_fingerprint does not match source file")
    if _assembly is not None:
        _guard_stream_metadata(_assembly)
        if (type(_assembly) is not CompiledVisualAssembly or _assembly.timeline != OutputTimeline.from_shots(plan.shots)
                or _assembly.expected_source_fingerprint != plan.source_fingerprint
                or tuple(shot.span for shot in _assembly.shots) != _assembly.timeline.raster_spans()):
            raise RenderPreflightError("HOST assembly does not match resolved plan timeline/fingerprint/spans")
    requirements = requirements_for_plan(plan) if _assembly is None else _assembly.requirements
    try:
        if requirements.needs_vulkan:
            runtime = runtime or resolve_render_runtime(requirements, device_index=device_index, **file_options)
            if runtime.runtime_id != requirements.runtime_id or runtime.device_index is None:
                raise RenderRuntimeError("Native motion requires the pinned runtime and explicit Vulkan device")
    except RenderRuntimeError as exc:
        raise RenderPreflightError(str(exc)) from exc
    if runtime is None:
        binary_version("ffmpeg", **tool_options)
        binary_version("ffprobe", **tool_options)
        select_encoder(profile.video_encoder, **tool_options)
        metadata = _probe(source, **tool_options)
    else:
        binary_version("ffmpeg", runtime=runtime, **tool_options)
        binary_version("ffprobe", runtime=runtime, **tool_options)
        select_encoder(profile.video_encoder, runtime=runtime, **tool_options)
        metadata = _probe(source, runtime=runtime, **tool_options)
    streams = _streams(metadata)
    video = [item for item in streams if item.get("codec_type") == "video"]
    audio = [item for item in streams if item.get("codec_type") == "audio"]
    if len(video) != 1:
        raise RenderPreflightError("Source requires exactly one video stream")
    if len(audio) != 1:
        raise RenderPreflightError("Source requires one mixed audio track")
    for key in ("width", "height"):
        dimension = _number(video[0].get(key), "source dimensions")
        if not dimension.is_integer() or int(dimension) % 2:
            raise RenderPreflightError("Source dimensions must be positive even pixels")
    fmt = metadata.get("format")
    if not isinstance(fmt, dict):
        raise RenderPreflightError("Missing source format duration/start_time metadata")
    format_span = _source_span(fmt, "format")
    origin = format_span[0]
    for label, component in (("format", fmt), ("video", video[0]), ("audio", audio[0])):
        start, end = format_span if label == "format" else _source_span(component, label)
        if any(origin + exact(str(shot.source.start)) < start
               or origin + exact(str(shot.source.end)) > end for shot in plan.shots):
            raise RenderPreflightError(
                f"Planned source range lies outside {label} span [{start}, {end}]")
    for path in sfx_paths:
        asset_metadata = (_probe(path, **tool_options) if runtime is None else
                          _probe(path, runtime=runtime, **tool_options))
        if not any(stream.get("codec_type") == "audio" for stream in _streams(asset_metadata)):
            raise RenderPreflightError(f"sfx asset has no loadable audio stream: {path}")
    source_size = int(_number(video[0]["width"], "width")), int(_number(video[0]["height"], "height"))
    if _assembly is None:
        compile_visual_layers(plan, source_size=source_size)
    elif _assembly.source_size != source_size:
        raise RenderPreflightError("HOST assembly source dimensions differ")
    if video[0].get("color_transfer") in ("smpte2084", "arib-std-b67"):
        raise RenderPreflightError("Known HDR PQ/HLG rendering is not qualified; explicit HDR qualification required")
    if requirements.needs_vulkan:
        assert runtime is not None
        try:
            qualification_options: dict[str, Any] = dict(file_options)
            if _tool_runner is not None:
                qualification_options["_runner"] = _tool_runner
            qualified = qualify_render_runtime(runtime, requirements, **qualification_options)
        except RenderRuntimeError as exc:
            raise RenderPreflightError(str(exc)) from exc
        color = tuple((key, value if isinstance(value, str) and value not in ("unknown", "unspecified")
                       else None) for key in ("color_transfer", "color_primaries", "color_space",
                        "color_range", "pix_fmt", "chroma_location") for value in (video[0].get(key),))
        return replace(qualified, source_color=color)
    return None
