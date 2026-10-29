"""Pure deterministic pixel geometry and FFmpeg layer primitives.

No subprocesses, assets, text, or output files are handled here.
"""

from collections.abc import Sequence
from dataclasses import dataclass, fields
from enum import Enum
from fractions import Fraction
import hashlib
import math
import re
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel

from .motion_compiler import CompiledCropTrack, _strict_model_input, compile_crop_track
from .output_timeline import RasterSpan
from .render_runtime import NATIVE_POLICY, RenderRequirements, _requirements_for_layers
from .visual_contracts import (
    AudioCue, AudioMixPlan, CropKeyframe, FillMode, LayerKind, NormalizedRect, OverlayPlan,
    SubtitlePlan, TimeRange, VisualCompositionPlan, VisualLayer, VisualShot,
)

MAX_INPUT_TEXT_BYTES = 8_000_000
MAX_METADATA_ENTRIES = 32768
MAX_NUMERIC_BITS = 4096
MAX_GENERATED_BYTES = 8_000_000


def _utf8_size(value: str, limit: int) -> int:
    if type(value) is not str or len(value) > limit:
        raise ValueError("Input text byte limit")
    total = 0
    try:
        for offset in range(0, len(value), 4096):
            total += len(value[offset:offset + 4096].encode("utf-8"))
            if total > limit:
                raise ValueError("Input text byte limit")
    except UnicodeError as exc:
        raise ValueError("Invalid UTF8 input text") from exc
    return total


def _guard_known_input(value: object, *, record_types: tuple[type, ...] = (),
                       text_limit: int = MAX_INPUT_TEXT_BYTES,
                       entry_limit: int = MAX_METADATA_ENTRIES) -> int:
    """Bound known compiler input records before model_dump/formatting; no I/O."""
    model_types = (VisualCompositionPlan, VisualShot, VisualLayer, NormalizedRect, CropKeyframe,
                   OverlayPlan, SubtitlePlan, AudioMixPlan, AudioCue, TimeRange)
    sequence_fields = {"shots": VisualShot, "layers": VisualLayer, "crop_track": CropKeyframe,
        "preserve_regions": NormalizedRect, "subtitle_allowed_regions": NormalizedRect,
        "forbidden_regions": NormalizedRect, "cues": AudioCue, "evidence_ids": str}
    nested_fields = {"source": TimeRange, "output": TimeRange, "destination": NormalizedRect,
        "source_crop": NormalizedRect, "crop": NormalizedRect, "audio": AudioMixPlan,
        "title": OverlayPlan, "brand": OverlayPlan, "subtitles": SubtitlePlan}
    active: set[int] = set()
    text_bytes, entries = 0, 0

    def check(item: object, depth: int = 0) -> None:
        nonlocal text_bytes, entries
        if depth > 24:
            raise ValueError("Known input nesting limit")
        if item is None or type(item) is bool:
            return
        if type(item) is int:
            if item.bit_length() > MAX_NUMERIC_BITS:
                raise ValueError("Numeric metadata bit limit")
            return
        if type(item) is Fraction:
            if max(item.numerator.bit_length(), item.denominator.bit_length()) > MAX_NUMERIC_BITS:
                raise ValueError("Numeric metadata bit limit")
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("Nonfinite numeric metadata")
            return
        if type(item) is str:
            text_bytes += _utf8_size(item, text_limit - text_bytes)
            return
        if type(item) in (FillMode, LayerKind) or (isinstance(item, Enum)
                and type(item).__module__ == "clipperx.visual_contracts" and type(item).__name__ == "Anchor"):
            check(cast(Enum, item).value, depth + 1)
            return
        if isinstance(item, Path):
            entries += len(item.parts)
            if entries > entry_limit:
                raise ValueError("Path metadata entry limit")
            for component in item.parts:
                check(component, depth + 1)
            return
        identity = id(item)
        if identity in active:
            raise ValueError("Known input cycle")
        active.add(identity)
        try:
            children: Sequence[object]
            if isinstance(item, BaseModel):
                kind = type(item)
                if kind not in model_types:
                    raise ValueError("Unexpected input model")
                if item.__dict__.keys() - kind.model_fields.keys() or item.model_extra:
                    raise ValueError("Unexpected input model fields")
                if any(field.is_required() and name not in item.__dict__
                       for name, field in kind.model_fields.items()):
                    raise ValueError("Missing required model field")
                integer_fields = {"z_index", "font_size", "outline_width", "max_words_per_event", "max_lines", "contract_version"}
                boolean_fields = {"emphasize_active_word", "normalize_dialogue", "limiter"}
                numeric_fields = {"x", "y", "at", "start", "end", "opacity", "blur_sigma",
                                  "confidence", "source_gain_db", "gain_db", "duck_source_db"}
                for name, child in item.__dict__.items():
                    if name in sequence_fields and (type(child) not in (tuple, list)
                            or len(child) > entry_limit
                            or any(type(v) is not sequence_fields[name] for v in child)):
                        raise ValueError("Invalid known model sequence")
                    if name in nested_fields and not (child is None and name in
                            {"source_crop", "title", "brand", "subtitles"}) and type(child) is not nested_fields[name]:
                        raise ValueError("Invalid nested model record")
                    if ((name in integer_fields or kind.__name__ == "VisualCompositionPlan" and name in ("width", "height"))
                            and type(child) is not int):
                        raise ValueError("Integer model scalar required")
                    if name in boolean_fields and type(child) is not bool:
                        raise ValueError("Boolean model scalar required")
                    if type(child) is bool and name not in boolean_fields:
                        raise ValueError("Boolean is not a numeric/text model scalar")
                    if ((name in numeric_fields or kind.__name__ == "NormalizedRect" and name in ("width", "height"))
                            and child is not None and type(child) not in (int, float)):
                        raise ValueError("Numeric model scalar required")
                children = tuple(item.__dict__.values())
            elif type(item) in record_types:
                children = tuple(getattr(item, field.name) for field in fields(cast(Any, item)))
            elif type(item) in (list, tuple):
                children = cast(Sequence[object], item)
            else:
                raise ValueError("Unexpected known input shape")
            entries += len(children) if type(item) in (list, tuple) else 0
            if entries > entry_limit:
                raise ValueError("Input metadata entry limit")
            for child in children:
                check(child, depth + 1)
        finally:
            active.remove(identity)

    check(value)
    return text_bytes


@dataclass(frozen=True, slots=True)
class PixelRect:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if any(type(value) is not int for value in (self.x, self.y, self.width, self.height)):
            raise ValueError("Pixel rectangle requires integer pixels")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Pixel rectangle requires positive pixel dimensions")


@dataclass(frozen=True, slots=True)
class CompiledShotLayers:
    """Shot-local static video graph; later compilers append ASS/audio/concat."""

    shot_id: str
    filter_complex: str
    output_label: str
    layer_ids: tuple[str, ...]
    source_start: float
    source_end: float
    output_start: float
    output_end: float
    requirements: RenderRequirements = RenderRequirements()


@dataclass(frozen=True, slots=True)
class SourceLayer:
    shot_id: str
    layer_id: str
    declaration_index: int
    kind: LayerKind
    z_index: int
    destination: PixelRect
    input_label: str | None
    output_label: str
    filter_complex: str
    crop_track: CompiledCropTrack | None
    requirements: RenderRequirements


def _validate_size(size: Sequence[int]) -> None:
    if len(size) != 2 or any(type(value) is not int or value <= 0 or value % 2 for value in size):
        raise ValueError("Canvas/source size requires positive even integer pixels")


def _even(value: float) -> int:
    """Nearest even pixel edge, ties upward; shared edges quantize identically."""
    return 2 * math.floor(value / 2 + .5)


def rect_to_pixels(rect: NormalizedRect, *, width: int, height: int) -> PixelRect:
    """Revalidate before clipping; collapse below two pixels is an error."""
    _validate_size((width, height))
    rect = NormalizedRect.model_validate(rect.model_dump())
    edges = (rect.x, rect.y, rect.x + rect.width, rect.y + rect.height)
    if not all(math.isfinite(edge) for edge in edges):
        raise ValueError("Normalized rectangle requires finite pixel edges")
    x, y = _even(rect.x * width), _even(rect.y * height)
    right = min(width, _even(edges[2] * width))
    bottom = min(height, _even(edges[3] * height))
    return PixelRect(x, y, right - x, bottom - y)


def clip_source_crop(crop: PixelRect, source_size: tuple[int, int]) -> PixelRect:
    """Clip at most one source pixel on each edge; never repair larger overflow."""
    _validate_size(source_size)
    width, height = source_size
    right, bottom = crop.x + crop.width, crop.y + crop.height
    if crop.x < -1 or crop.y < -1 or right > width + 1 or bottom > height + 1:
        raise ValueError("Crop exceeds source bounds by more than one pixel")
    x, y = max(0, crop.x), max(0, crop.y)
    return PixelRect(x, y, min(right, width) - x, min(bottom, height) - y)


def source_crop_pixels(rect: NormalizedRect | None, source_size: tuple[int, int]) -> PixelRect:
    """None retains the contract's full-source semantics, without detection."""
    _validate_size(source_size)
    if rect is None:
        return PixelRect(0, 0, *source_size)
    rect = NormalizedRect.model_validate(rect.model_dump())
    # Check overflow BEFORE normalized destination clipping or even quantization.
    if ((rect.x + rect.width - 1) * source_size[0] > 1
            or (rect.y + rect.height - 1) * source_size[1] > 1):
        raise ValueError("Crop exceeds source bounds by more than one pixel")
    return rect_to_pixels(rect, width=source_size[0], height=source_size[1])


def _label(label: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]+(?::[A-Za-z0-9_]+)*", label):
        raise ValueError("Invalid filter label")
    return f"[{label}]"


def layer_label(shot_id: str, layer_id: str) -> str:
    """Hash each length-delimited identity; arbitrary plan IDs never enter syntax."""
    payload = f"{len(shot_id)}:{shot_id}{len(layer_id)}:{layer_id}".encode()
    return "vl_" + hashlib.sha256(payload).hexdigest()


def _scale(destination: PixelRect, mode: str) -> str:
    return (f"scale=w={destination.width}:h={destination.height}:"
            f"force_original_aspect_ratio={mode}:force_divisible_by=2,setsar=1")


def _cover(destination: PixelRect) -> str:
    return (f"{_scale(destination, 'increase')},crop=w={destination.width}:"
            f"h={destination.height}:x=(in_w-out_w)/2:y=(in_h-out_h)/2:exact=1")


def _frame_count(value: int | None) -> None:
    if value is not None and (type(value) is not int or value <= 0 or value.bit_length() > MAX_NUMERIC_BITS):
        raise ValueError("frame_count requires a positive integer")


def _finite_color(color: str, destination: PixelRect, *, duration: float = 1,
                  frame_count: int | None = None) -> str:
    """Generated pixels on the same exact local transport clock as supplied pixels."""
    _frame_count(frame_count)
    graph = f"color=c={color}:s={destination.width}x{destination.height}:r=60"
    return (graph + f":d={duration!r}" if frame_count is None else
            graph + f",trim=end_frame={frame_count},settb=1/60,setpts=N")


def compile_layer(*, source_size: tuple[int, int], destination: PixelRect, fill: FillMode,
                  source_crop: PixelRect | None = None, kind: LayerKind = LayerKind.SOURCE_VIDEO,
                  solid_color: str | None = None, blur_sigma: float | None = None,
                  input_label: str = "in", output_label: str = "out", fps: int = 60,
                  duration: float = 1, frame_count: int | None = None) -> str:
    """Compile a destination-sized layer; its x/y belong to the enclosing overlay.

    blur_sigma is explicit host strength: ceil(strength) luma pixel box radius,
    one box pass, ceil(radius/2) chroma radius for normalized yuv420p. It is not
    a Gaussian standard deviation. Unrepresentable strengths fail, never clamp.
    """
    _validate_size(source_size)
    _frame_count(frame_count)
    _validate_size((destination.width, destination.height))
    if type(fps) is not int or fps != 60 or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid layer timing; expected 60 FPS and finite positive duration")
    fill, kind = FillMode(fill), LayerKind(kind)
    inp, out = _label(input_label), _label(output_label)
    prefix = output_label.replace(":", "_")
    if input_label == output_label:
        raise ValueError("Input and output label must differ")
    if fill == FillMode.SOLID or kind == LayerKind.SOLID:
        if (fill != FillMode.SOLID or kind != LayerKind.SOLID or solid_color is None
                or not re.fullmatch(r"#[0-9a-fA-F]{6}", solid_color)):
            raise ValueError("solid fill requires solid kind and explicit #RRGGBB color")
        return _finite_color(f"0x{solid_color[1:]}", destination, duration=duration,
                             frame_count=frame_count) + out
    if solid_color is not None or fill == FillMode.INHERIT:
        raise ValueError("Unsupported or unresolved layer fill")
    crop = clip_source_crop(source_crop or PixelRect(0, 0, *source_size), source_size)
    crop_filter = f"crop=w={crop.width}:h={crop.height}:x={crop.x}:y={crop.y}:exact=1"
    if fill == FillMode.COVER:
        return f"{inp}{crop_filter},{_cover(destination)}{out}"
    ratio = min(destination.width / crop.width, destination.height / crop.height)
    if min(crop.width * ratio, crop.height * ratio) < 2:
        raise ValueError("Contained foreground cannot fit positive even pixel dimensions")
    fg, bg, sharp = (_label(prefix + suffix) for suffix in ("_fg", "_bg", "_sharp"))
    centered = "overlay=x=(main_w-overlay_w)/2:y=(main_h-overlay_h)/2:shortest=1"
    if fill == FillMode.BLACK:
        return (_finite_color("black", destination, duration=duration, frame_count=frame_count)
                + f"{bg};{inp}{crop_filter},"
                f"{_scale(destination, 'decrease')}{sharp};{bg}{sharp}{centered}{out}")
    if blur_sigma is None or not math.isfinite(blur_sigma) or not 0 < blur_sigma <= 50:
        raise ValueError("contain_blur requires explicit finite blur_sigma in (0,50]")
    radius = math.ceil(blur_sigma)
    chroma_radius = math.ceil(radius / 2)
    if (radius > min(destination.width, destination.height) / 2
            or chroma_radius > min(destination.width // 2, destination.height // 2) / 2):
        raise ValueError("blur_sigma cannot be represented by destination boxblur bounds")
    backdrop, blurred = (_label(prefix + suffix) for suffix in ("_backdrop", "_blurred"))
    return (f"{inp}{crop_filter},split=2{fg}{backdrop};{backdrop}{_cover(destination)},"
            f"format=yuv420p,boxblur=luma_radius={radius}:luma_power=1:"
            f"chroma_radius={chroma_radius}:chroma_power=1{blurred};"
            f"{fg}{_scale(destination, 'decrease')}{sharp};"
            f"{blurred}{sharp}{centered}{out}")


def _motion_transform(track: CompiledCropTrack, destination: PixelRect, *, cover: bool,
                      input_label: str, output_label: str, transparent: bool = False,
                      pixel_format: str = "rgba") -> str:
    """Native crop/placement. High-bit output is a host qualification oracle."""
    dw, dh = destination.width, destination.height
    if pixel_format not in ("rgba", "yuv444p16le"):
        raise ValueError("Unsupported native host output format")
    x, y, w, h = track.expressions("n")
    scale = "max" if cover else "min"
    return (f"{_label(input_label)}libplacebo=w={dw}:h={dh}:format={pixel_format}:"
            f"crop_x='{x}':crop_y='{y}':crop_w='{w}':crop_h='{h}':"
            f"pos_w='cw*{scale}(ow/cw,oh/ch)':pos_h='ch*{scale}(ow/cw,oh/ch)':"
            "pos_x='(ow-pw)/2':pos_y='(oh-ph)/2':"
            f"fillcolor={'black@0' if transparent else 'black'}:alpha_mode=straight:"
            f"reset_sar=1:{NATIVE_POLICY}{_label(output_label)}")


def compile_motion_layer(track: CompiledCropTrack, *, destination: PixelRect, fill: FillMode,
                         blur_sigma: float | None = None, input_label: str = "in",
                         output_label: str = "out", duration: float = 1,
                         preserve_regions: Sequence[NormalizedRect] = (),
                         shot_id: str = "", layer_id: str = "",
                         frame_count: int | None = None) -> str:
    """Compile approved motion geometry, with no render or quality fallback."""
    _validate_size((destination.width, destination.height))
    _frame_count(frame_count)
    _label(input_label), _label(output_label)
    if input_label == output_label or not math.isfinite(duration) or duration <= 0:
        raise ValueError("Invalid motion labels/duration")
    fill = FillMode(fill)
    if fill not in (FillMode.COVER, FillMode.BLACK, FillMode.CONTAIN_BLUR):
        raise ValueError("Unsupported or unresolved motion fill")
    track.validate_preservation(preserve_regions, destination=(destination.width, destination.height),
                                fill=fill, shot_id=shot_id, layer_id=layer_id)
    if track.mode == "static":
        r = track.sample(0)
        return compile_layer(source_size=track.source_size, destination=destination, fill=fill,
            source_crop=PixelRect(*(int(v) for v in r.values)), blur_sigma=blur_sigma,
            input_label=input_label, output_label=output_label, duration=duration,
            frame_count=frame_count)
    if fill != FillMode.COVER:
        # Ratios of affine positive widths/heights are monotone per segment;
        # endpoints bound minimum contained dimensions for either easing.
        for index, point in enumerate(track.points):
            scale = min(destination.width/point.rect.width, destination.height/point.rect.height)
            if min(point.rect.width*scale, point.rect.height*scale) < 2:
                raise ValueError(f"shot={shot_id} layer={layer_id} keyframe={index}: "
                                 "Contained foreground cannot fit two output pixels")
    prefix = output_label.replace(":", "_")
    if fill in (FillMode.COVER, FillMode.BLACK):
        return _motion_transform(track, destination, cover=fill == FillMode.COVER,
                                  input_label=input_label, output_label=output_label)
    sharp, bg = prefix+"_sharp", prefix+"_bg"
    if blur_sigma is None or not math.isfinite(blur_sigma) or not 0 < blur_sigma <= 50:
        raise ValueError("contain_blur requires explicit finite blur_sigma in (0,50]")
    radius, chroma = math.ceil(blur_sigma), math.ceil(math.ceil(blur_sigma)/2)
    if (radius > min(destination.width, destination.height)/2
            or chroma > min(destination.width//2, destination.height//2)/2):
        raise ValueError("blur_sigma cannot be represented by destination boxblur bounds")
    a, b, cover_label = prefix+"_a", prefix+"_b", prefix+"_cover"
    backdrop = (f"{_label(input_label)}split=2{_label(a)}{_label(b)};"
                + _motion_transform(track, destination, cover=True,
                                     input_label=a, output_label=cover_label)
                + f";{_label(cover_label)}format=yuv420p,boxblur=luma_radius={radius}:"
                f"luma_power=1:chroma_radius={chroma}:chroma_power=1,format=rgba{_label(bg)}")
    foreground = _motion_transform(track, destination, cover=False, transparent=True,
                                    input_label=b, output_label=sharp)
    return (f"{backdrop};{foreground};{_label(bg)}{_label(sharp)}"
            f"overlay=x=0:y=0:shortest=1:format=rgb:alpha=straight,format=rgba{_label(output_label)}")


def compile_source_layer(shot: VisualShot, *, layer_index: int,
                         raster_span: RasterSpan, source_size: tuple[int, int],
                         input_label: str | None) -> SourceLayer:
    """One independently placeable primitive, consuming scheduled local pixels only."""
    if type(shot) is not VisualShot or type(raster_span) is not RasterSpan:
        raise ValueError("Expected shot and raster span records")
    if input_label is not None and type(input_label) is not str:
        raise ValueError("Input label requires a string or None")
    checked_input_bytes = _guard_known_input((shot, raster_span, source_size, input_label),
        record_types=(RasterSpan,), text_limit=MAX_INPUT_TEXT_BYTES, entry_limit=MAX_METADATA_ENTRIES)
    if len(shot.layers) > 4096 or sum(len(layer.crop_track) for layer in shot.layers) > MAX_METADATA_ENTRIES:
        raise ValueError("Source primitive work limit")
    if type(layer_index) is not int or not 0 <= layer_index < len(shot.layers):
        raise ValueError("Invalid layer index")
    # Each current source path emits the caller inlet exactly once; branch/internal
    # labels are derived from the fixed-length output hash. Count without formatting
    # a label or graph, and reserve syntax/native expression expansion as before.
    emitted_label_bytes = (0 if input_label is None else _utf8_size(input_label, checked_input_bytes))
    if 8192 + len(shot.layers[layer_index].crop_track) * 8192 + emitted_label_bytes > MAX_GENERATED_BYTES:
        raise ValueError("Source expression byte budget")
    if type(source_size) not in (tuple, list):
        raise ValueError("Source size requires a pixel pair")
    _validate_size(source_size)
    raster_span = RasterSpan(*(getattr(raster_span, field.name) for field in fields(RasterSpan)))
    shot = VisualShot.model_validate(_strict_model_input(shot))
    if (type(layer_index) is not int or not 0 <= layer_index < len(shot.layers)):
        raise ValueError("Invalid layer index")
    if ((shot.shot_id, *(Fraction(str(v)) for v in (shot.source.start, shot.source.end,
            shot.output.start, shot.output.end))) != (raster_span.shot_id,
            raster_span.source_start, raster_span.source_end,
            raster_span.output_start, raster_span.output_end)):
        raise ValueError("Shot/raster span association mismatch")
    layer = shot.layers[layer_index]
    if (layer.opacity != 1 or layer.fill == FillMode.INHERIT
            or (layer.kind != LayerKind.STREAMER_CAMERA and layer.source_ref != "source")
            or (layer.kind == LayerKind.STREAMER_CAMERA and not re.fullmatch(
                r"[a-z][a-z0-9_-]{0,63}", layer.source_ref))):
        raise ValueError("Unsupported source layer provenance/opacity/fill")
    destination = rect_to_pixels(layer.destination, width=1080, height=1920)
    output = layer_label(shot.shot_id, layer.layer_id) + "_pixels"
    track: CompiledCropTrack | None = None
    if layer.kind == LayerKind.SOLID:
        if input_label is not None or layer.crop_track:
            raise ValueError("SOLID requires no input label/crop_track")
        graph = compile_layer(source_size=source_size, destination=destination,
            fill=layer.fill, kind=layer.kind, solid_color=layer.solid_color,
            output_label=output, frame_count=raster_span.frame_count)
    else:
        if input_label is None or layer.fill == FillMode.SOLID:
            raise ValueError("Source/camera requires input label and spatial fill")
        _label(input_label)
        try:
            representable = all(math.isfinite(value) for value in source_size)
        except OverflowError as exc:
            raise ValueError("Source dimensions exceed supported numeric geometry") from exc
        if not representable:
            raise ValueError("Source dimensions exceed supported numeric geometry")
        track = compile_crop_track(layer.crop_track or [CropKeyframe(at=0,
            crop=layer.source_crop or NormalizedRect(x=0, y=0, width=1, height=1))],
            source_size=source_size)
        graph = compile_motion_layer(track, destination=destination, fill=layer.fill,
            blur_sigma=layer.blur_sigma, input_label=input_label, output_label=output,
            duration=shot.source.duration, frame_count=raster_span.frame_count,
            preserve_regions=layer.preserve_regions, shot_id=shot.shot_id, layer_id=layer.layer_id)
    _utf8_size(graph, MAX_GENERATED_BYTES)
    return SourceLayer(shot.shot_id, layer.layer_id, layer_index, layer.kind, layer.z_index,
                       destination, input_label, output, graph, track,
                       _requirements_for_layers([layer]))


def compile_shot_layers(shot: VisualShot, *, source_size: tuple[int, int],
                        input_label: str = "0:v") -> CompiledShotLayers:
    """Compile resolved static/moving layers; no ASS, audio, concat or execution."""
    _validate_size(source_size)
    shot = VisualShot.model_validate(_strict_model_input(shot))
    layers = sorted(shot.layers, key=lambda layer: layer.z_index)
    for layer in layers:
        if layer.kind != LayerKind.STREAMER_CAMERA and layer.source_ref != "source":
            raise ValueError("source_video/solid requires source_ref=source")
        if layer.opacity != 1:
            raise ValueError("Unsupported layer opacity; expected 1")
    prefix = layer_label(shot.shot_id, "")
    duration = shot.source.duration
    base = prefix + "_base"
    parts = [f"color=c=black:s=1080x1920:r=60:d={duration!r}{_label(base)}"]
    source_layers = [layer for layer in layers if layer.kind != LayerKind.SOLID]
    if any(layer.kind == LayerKind.SOLID and layer.crop_track for layer in layers):
        raise ValueError(f"shot={shot.shot_id}: solid layer cannot have a crop_track")
    if source_layers:
        # Preserve fractional source offsets before PTS subtraction. In a 1/30
        # input TB, PTS-0.025/TB would otherwise truncate .25 ticks to zero.
        # This rational represents 30/60 and standard NTSC frame clocks exactly,
        # with one million ticks per output frame. AVTB microseconds can push
        # exact half-frame ties to alternate sides of fps's rounding boundary.
        # Static and moving layers share this grid so the timed canvas overlay
        # cannot reveal a black initial gap or advance static source content.
        trimmed = (f"{_label(input_label)}settb=1/60000000,"
                   f"trim=start={shot.source.start!r}:end={shot.source.end!r},"
                   f"setpts=PTS-{shot.source.start!r}/TB,"
                   f"fps=60:start_time=0:round=near,trim=end={duration!r}")
        branches = [_label(layer_label(shot.shot_id, layer.layer_id) + "_source")
                    for layer in source_layers]
        split = f",split={len(branches)}" if len(branches) > 1 else ",null"
        parts.append(trimmed + split + "".join(branches))
    for layer in layers:
        label = layer_label(shot.shot_id, layer.layer_id)
        destination = rect_to_pixels(layer.destination, width=1080, height=1920)
        if layer.kind == LayerKind.SOLID:
            parts.append(compile_layer(source_size=source_size, destination=destination,
                fill=layer.fill, kind=layer.kind, solid_color=layer.solid_color,
                input_label=label+"_source", output_label=label+"_pixels", duration=duration))
        else:
            keyframes = layer.crop_track or [CropKeyframe(at=0, crop=layer.source_crop or
                NormalizedRect(x=0, y=0, width=1, height=1))]
            track = compile_crop_track(keyframes, source_size=source_size)
            parts.append(compile_motion_layer(track, destination=destination, fill=layer.fill,
                blur_sigma=layer.blur_sigma, input_label=label+"_source",
                output_label=label+"_pixels", duration=duration,
                preserve_regions=layer.preserve_regions, shot_id=shot.shot_id,
                layer_id=layer.layer_id))
        combined = label + "_composite"
        parts.append(f"{_label(base)}{_label(label + '_pixels')}overlay=x={destination.x}:"
                     f"y={destination.y}:shortest=1{_label(combined)}")
        base = combined
    output = prefix + "_video"
    parts.append(f"{_label(base)}setsar=1,fps=60{_label(output)}")
    return CompiledShotLayers(shot.shot_id, ";".join(parts), output,
                              tuple(layer.layer_id for layer in layers),
                              shot.source.start, shot.source.end, shot.output.start, shot.output.end,
                              _requirements_for_layers(layers))
