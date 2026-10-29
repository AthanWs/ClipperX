"""Immutable crop geometry and numeric FFmpeg expressions; never renders frames."""

from collections.abc import Sequence
from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

from pydantic import BaseModel

from .visual_contracts import (
    CropKeyframe, FillMode, NormalizedRect, _prepare_strict_visual_input,
)

if TYPE_CHECKING:
    from .filtergraph import PixelRect


def _strict_model_input(model: BaseModel) -> object:
    """Retain the shared strict boundary, including extras model_dump omits."""
    def check(value: object) -> None:
        if isinstance(value, BaseModel):
            fields = type(value).model_fields
            if value.__dict__.keys() - fields.keys() or value.model_extra:
                raise ValueError(f"Unrequested fields in {type(value).__name__}")
            for child in value.__dict__.values():
                check(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                check(child)
        elif isinstance(value, dict):
            for child in value.values():
                check(child)

    check(model)
    return _prepare_strict_visual_input(type(model), model)


@dataclass(frozen=True, slots=True)
class FloatRect:
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in self.values) or min(self.width, self.height) < 2:
            raise ValueError("Crop requires finite geometry and at least two source pixels")

    @property
    def values(self) -> tuple[float, float, float, float]:
        return self.x, self.y, self.width, self.height


@dataclass(frozen=True, slots=True)
class CropPoint:
    frame: float
    rect: FloatRect
    easing: str


def _mix(a: FloatRect, b: FloatRect, u: float) -> FloatRect:
    return FloatRect(*(x + (y - x) * u for x, y in zip(a.values, b.values, strict=True)))


def _fraction(u: float, easing: str) -> float:
    return u if easing == "linear" else 3 * u * u - 2 * u * u * u


def _within_source(rect: FloatRect, size: tuple[int, int]) -> None:
    if (rect.x < 0 or rect.y < 0 or rect.x + rect.width > size[0] + 1e-8
            or rect.y + rect.height > size[1] + 1e-8):
        raise ValueError("Crop exceeds source bounds")


@dataclass(frozen=True, slots=True)
class CompiledCropTrack:
    points: tuple[CropPoint, ...]
    source_size: tuple[int, int]
    fps: int
    mode: str

    def sample(self, frame: float) -> FloatRect:
        if not math.isfinite(frame):
            raise ValueError("Sample frame must be finite")
        if frame <= self.points[0].frame:
            return self.points[0].rect
        for a, b in zip(self.points, self.points[1:]):
            if frame <= b.frame:
                if frame == b.frame:
                    return b.rect
                return _mix(a.rect, b.rect, _fraction(
                    (frame - a.frame) / (b.frame - a.frame), a.easing))
        return self.points[-1].rect

    def sample_time(self, seconds: float) -> FloatRect:
        return self.sample(seconds * self.fps)

    def expressions(self, frame: str) -> tuple[str, str, str, str]:
        """Serialize the same piecewise data, accepting only host clock symbols."""
        if frame not in ("N", "on-1", "n"):
            raise ValueError("Unsupported host motion clock")
        result = []
        for component in range(4):
            expr = repr(self.points[-1].rect.values[component])
            for a, b in reversed(list(zip(self.points, self.points[1:]))):
                u = f"clip((({frame})-{a.frame!r})/{b.frame-a.frame!r},0,1)"
                eased = u if a.easing == "linear" else f"(3*({u})^2-2*({u})^3)"
                start, end = a.rect.values[component], b.rect.values[component]
                value = f"({start!r}+({end-start!r})*({eased}))"
                expr = f"if(lte(({frame}),{b.frame!r}),{value},{expr})"
            result.append(expr)
        return result[0], result[1], result[2], result[3]

    def crossing_frames(self, destination: tuple[int, int]) -> tuple[float, ...]:
        ratio = destination[0] / destination[1]
        frames = []
        for a, b in zip(self.points, self.points[1:]):
            denominator = b.rect.width - a.rect.width - ratio * (b.rect.height-a.rect.height)
            if denominator == 0:
                continue
            u = (ratio * a.rect.height - a.rect.width) / denominator
            if 0 < u < 1:
                lo, hi = 0.0, 1.0
                if a.easing != "linear":
                    for _ in range(54):
                        middle = (lo + hi) / 2
                        if _fraction(middle, a.easing) < u:
                            lo = middle
                        else:
                            hi = middle
                    u = (lo + hi) / 2
                frames.append(a.frame + u * (b.frame - a.frame))
        return tuple(frames)

    def validate_preservation(self, regions: Sequence[NormalizedRect], *,
                              destination: tuple[int, int], fill: FillMode,
                              shot_id: str, layer_id: str) -> None:
        checked = [NormalizedRect.model_validate(_strict_model_input(r)) for r in regions]
        if not checked:
            return
        frames = {p.frame for p in self.points}
        frames.update((a.frame+b.frame)/2 for a, b in zip(self.points, self.points[1:]))
        frames.update(self.crossing_frames(destination))
        frames.update(range(math.ceil(self.points[-1].frame) + 1))
        for frame in sorted(frames):
            crop = self.sample(frame)
            visible = crop.values
            if fill == FillMode.COVER:
                scale = max(destination[0]/crop.width, destination[1]/crop.height)
                w, h = destination[0]/scale, destination[1]/scale
                # This is an analytical visible subset, not a new authored crop;
                # source-space width may be subpixel at a valid narrow output.
                visible = crop.x+(crop.width-w)/2, crop.y+(crop.height-h)/2, w, h
            for region in checked:
                x, y = region.x*self.source_size[0], region.y*self.source_size[1]
                right = (region.x+region.width)*self.source_size[0]
                bottom = (region.y+region.height)*self.source_size[1]
                for bx, by, bw, bh in (crop.values, visible):
                    if (x < bx-2 or y < by-2 or right > bx+bw+2 or bottom > by+bh+2):
                        index = max((i for i, p in enumerate(self.points) if p.frame <= frame), default=0)
                        raise ValueError(f"shot={shot_id} layer={layer_id} keyframe={index} "
                                         f"frame={frame}: preserve_region outside crop/fill")


def compile_crop_track(keyframes: Sequence[CropKeyframe], *, source_size: Sequence[int],
                       fps: int = 60) -> CompiledCropTrack:
    """Revalidate detached input; preserve fractional moving source coordinates."""
    from .filtergraph import _validate_size, source_crop_pixels

    _validate_size(source_size)
    source_size = (source_size[0], source_size[1])
    if type(fps) is not int or fps != 60 or not keyframes:
        raise ValueError("Crop track requires keyframes and output FPS=60")
    keys = [CropKeyframe.model_validate(_strict_model_input(k)) for k in keyframes]
    if any(not math.isfinite(k.at) for k in keys):
        raise ValueError("Crop track requires finite times")
    if len(keys) > 1 and (keys[0].at != 0 or any(
            a.at >= b.at for a, b in zip(keys, keys[1:]))):
        raise ValueError("Moving crop times must start at zero and strictly increase")
    points = []
    for key in keys:
        if len(keys) == 1:
            pixel = source_crop_pixels(key.crop, source_size)
            rect = FloatRect(pixel.x, pixel.y, pixel.width, pixel.height)
        else:
            x, y = key.crop.x*source_size[0], key.crop.y*source_size[1]
            width, height = key.crop.width*source_size[0], key.crop.height*source_size[1]
            right, bottom = x+width, y+height
            if right > source_size[0]+1 or bottom > source_size[1]+1:
                raise ValueError("Crop exceeds source bounds by more than one pixel")
            rect = FloatRect(x, y, min(width, source_size[0]-x),
                             min(height, source_size[1]-y))
        _within_source(rect, source_size)
        points.append(CropPoint(key.at*fps, rect, key.easing))
    return CompiledCropTrack(tuple(points), source_size, fps,
                             "static" if len(points) == 1 else "dynamic")


def smooth_crop_observations(observations: Sequence["PixelRect | FloatRect"], *,
                             source_size: tuple[int, int]) -> tuple[FloatRect, ...]:
    """Bounded SG observation correction, never smoothing authored crop tracks."""
    from .filtergraph import _validate_size

    _validate_size(source_size)
    original = tuple(FloatRect(r.x, r.y, r.width, r.height) for r in observations)
    for rect in original:
        _within_source(rect, source_size)
    if len(original) < 5:
        return original
    # Exact five-observation, degree-two Savitzky-Golay projection. Edge rows
    # use the same quadratic fit as mode='interp'; authored endpoints stay.
    filtered = [list(r.values) for r in original]
    for index in range(1, len(original)-1):
        start = min(max(index-2, 0), len(original)-5)
        coefficients = {1: (9, 13, 12, 6, -5), 2: (-3, 12, 17, 12, -3),
                        3: (-5, 6, 12, 13, 9)}[index-start]
        filtered[index] = [sum(c * original[start+j].values[component]
                               for j, c in enumerate(coefficients))/35
                           for component in range(4)]
    limits = (source_size[0]*.015, source_size[1]*.015)*2
    if any(abs(float(value)-before) >= limit
           for row, rect in zip(filtered, original, strict=True)
           for value, before, limit in zip(row, rect.values, limits, strict=True)):
        return original
    try:
        result = tuple(FloatRect(*(float(v) for v in row)) for row in filtered)
        for rect in result:
            _within_source(rect, source_size)
    except ValueError:
        return original
    return result
