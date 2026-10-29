"""Strict, provider-free contracts for visual analysis and vertical composition."""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from enum import Enum
from fractions import Fraction
import math
import re
from types import UnionType
from typing import Literal, Union, get_args, get_origin

from pydantic import BaseModel, ConfigDict, Field, model_validator

from clipperx.planner_contracts import MAX_CLIP_SECONDS, TimeRange


_TOLERANCE_SECONDS = 0.001
_REPRESENTATION_EPSILON_SECONDS = 1e-9
_RECT_CONTAINMENT_EPSILON = 1e-9
_ASSET_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_TRACK_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


class SourceState(str, Enum):
    FULL_CAMERA = "full_camera"
    CONTENT_WITH_CAMERA = "content_with_camera"
    CONTENT_ONLY = "content_only"
    BROWSER_TEXT = "browser_text"
    REACTED_VIDEO = "reacted_video"
    MULTI_PERSON = "multi_person"
    UNKNOWN = "unknown"


class IdentityStatus(str, Enum):
    CONFIDENT = "confident"
    AMBIGUOUS = "ambiguous"
    UNAVAILABLE = "unavailable"


class FillMode(str, Enum):
    COVER = "cover"
    CONTAIN_BLUR = "contain_blur"
    BLACK = "black"
    SOLID = "solid"
    INHERIT = "inherit"


class LayerKind(str, Enum):
    SOURCE_VIDEO = "source_video"
    STREAMER_CAMERA = "streamer_camera"
    SOLID = "solid"


class NormalizedRect(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> "NormalizedRect":
        if self.x + self.width > 1.000001 or self.y + self.height > 1.000001:
            raise ValueError("El rectángulo normalizado sale del frame")
        return self


class TemporalRectObservation(BaseModel):
    at: float = Field(ge=0)
    rect: NormalizedRect
    confidence: float = Field(ge=0, le=1)
    evidence_id: str


class FaceTrackSummary(BaseModel):
    track_id: str
    first_seen: float = Field(ge=0)
    last_seen: float = Field(gt=0)
    observation_count: int = Field(ge=1)
    embedding_centroid: list[float] = Field(min_length=1)
    positions: list[NormalizedRect] = Field(min_length=1)
    recurrent_states: list[SourceState] = Field(default_factory=list)
    full_camera_seconds: float = Field(default=0.0, ge=0)
    mean_confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_seen_range(self) -> "FaceTrackSummary":
        if self.last_seen <= self.first_seen:
            raise ValueError("last_seen debe ser mayor que first_seen")
        return self


class WebcamTrack(BaseModel):
    track_id: str
    source_ranges: list[TimeRange] = Field(min_length=1)
    observations: list[TemporalRectObservation] = Field(min_length=1)
    face_track_ids: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(min_length=1)


class SourceVisualSample(BaseModel):
    sample_id: str
    at: float = Field(ge=0)
    histogram_change_score: float = Field(ge=0, le=1)
    structural_change_score: float = Field(ge=0, le=1)
    letterbox_bounds: NormalizedRect | None = None
    persistent_rectangles: list[NormalizedRect] = Field(default_factory=list)
    rectangle_candidates: list[NormalizedRect] = Field(default_factory=list)
    text_density: float = Field(ge=0, le=1)
    text_regions: list[NormalizedRect] = Field(default_factory=list)
    face_track_ids: list[str] = Field(default_factory=list)
    webcam_track_id: str | None = None
    webcam_rect: NormalizedRect | None = None
    brightness: float = Field(ge=0, le=1)
    sharpness: float = Field(ge=0)
    motion: float = Field(ge=0, le=1)
    audio_features: dict[str, float] = Field(default_factory=dict)
    audio_window: TimeRange | None = None
    transcript_window: TimeRange | None = None
    diarization_window: TimeRange | None = None
    evidence_ids: list[str] = Field(min_length=1)


class SourceVisualSpan(BaseModel):
    span_id: str
    source: TimeRange
    state: SourceState
    confidence: float = Field(ge=0, le=1)
    streamer_track_id: str | None
    identity_status: IdentityStatus
    webcam_track_id: str | None
    important_regions: list[NormalizedRect]
    text_regions: list[NormalizedRect]
    evidence_ids: list[str]
    change_reasons: list[str]

    @model_validator(mode="after")
    def validate_identity_track(self) -> "SourceVisualSpan":
        if self.identity_status == IdentityStatus.CONFIDENT and self.streamer_track_id is None:
            raise ValueError("identity_status 'confident' requiere streamer_track_id")
        if (
            self.identity_status in (IdentityStatus.AMBIGUOUS, IdentityStatus.UNAVAILABLE)
            and self.streamer_track_id is not None
        ):
            raise ValueError(
                f"identity_status '{self.identity_status.value}' prohíbe streamer_track_id"
            )
        return self


class SourceVisualTimeline(BaseModel):
    contract_version: Literal[1] = 1
    source_fingerprint: str
    duration_seconds: float = Field(gt=0)
    analyzer_version: str
    model_hashes: dict[str, str]
    sampling_config: dict[str, float]
    samples: list[SourceVisualSample]
    spans: list[SourceVisualSpan]
    face_tracks: list[FaceTrackSummary]
    webcam_tracks: list[WebcamTrack]

    @model_validator(mode="after")
    def validate_timeline(self) -> "SourceVisualTimeline":
        _ensure_unique((sample.sample_id for sample in self.samples), "sample_id")
        _ensure_unique((span.span_id for span in self.spans), "span_id")
        _ensure_unique((track.track_id for track in self.face_tracks), "face track_id")
        _ensure_unique((track.track_id for track in self.webcam_tracks), "webcam track_id")

        face_ids = {track.track_id for track in self.face_tracks}
        webcam_ids = {track.track_id for track in self.webcam_tracks}
        conflicting_track_ids = face_ids & webcam_ids
        if conflicting_track_ids:
            conflicts = ", ".join(sorted(conflicting_track_ids))
            raise ValueError(
                "track_id colisiona entre face_tracks y webcam_tracks: " f"{conflicts}"
            )
        previous_sample_at = -1.0
        for sample in self.samples:
            if sample.at <= previous_sample_at:
                raise ValueError("Los timestamps de samples deben ser crecientes")
            previous_sample_at = sample.at
            self._validate_timestamp(sample.at, "sample")
            self._validate_windows(sample)
            self._validate_face_references(sample.face_track_ids, face_ids)
            self._validate_webcam_reference(sample.webcam_track_id, webcam_ids)

        previous_span_end = 0.0
        for index, span in enumerate(self.spans):
            self._validate_range(span.source, "span")
            if index and span.source.start < previous_span_end:
                raise ValueError("Los spans deben estar ordenados y no superponerse")
            previous_span_end = span.source.end
            self._validate_face_references(
                [track_id for track_id in [span.streamer_track_id] if track_id is not None],
                face_ids,
            )
            self._validate_webcam_reference(span.webcam_track_id, webcam_ids)

        for face_track in self.face_tracks:
            self._validate_timestamp(face_track.first_seen, "face track")
            self._validate_timestamp(face_track.last_seen, "face track")
            if face_track.full_camera_seconds > self.duration_seconds:
                raise ValueError("full_camera_seconds sale de duration_seconds")
        for webcam_track in self.webcam_tracks:
            for source_range in webcam_track.source_ranges:
                self._validate_range(source_range, "webcam track")
            for observation in webcam_track.observations:
                self._validate_timestamp(observation.at, "webcam observation")
            self._validate_face_references(webcam_track.face_track_ids, face_ids)
        return self

    def _validate_timestamp(self, value: float, label: str) -> None:
        if value > self.duration_seconds:
            raise ValueError(f"{label} sale de duration_seconds")

    def _validate_range(self, value: TimeRange, label: str) -> None:
        if value.end > self.duration_seconds:
            raise ValueError(f"{label} sale de duration_seconds")

    def _validate_windows(self, sample: SourceVisualSample) -> None:
        for window in (sample.audio_window, sample.transcript_window, sample.diarization_window):
            if window is not None:
                self._validate_range(window, "window")

    @staticmethod
    def _validate_face_references(track_ids: list[str], known_ids: set[str]) -> None:
        for track_id in track_ids:
            if track_id not in known_ids:
                raise ValueError("face track desconocido")

    @staticmethod
    def _validate_webcam_reference(track_id: str | None, known_ids: set[str]) -> None:
        if track_id is not None and track_id not in known_ids:
            raise ValueError("webcam track desconocido")


class CropKeyframe(BaseModel):
    at: float = Field(ge=0, description=(
        "Seconds relative to shot duration, not global source/output time. Moving tracks "
        "start at 0 and end at shot duration; times are strictly increasing and unique."
    ))
    crop: NormalizedRect
    easing: Literal["linear", "ease_in_out"] = "ease_in_out"


class Anchor(str, Enum):
    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    TOP_RIGHT = "top_right"
    CENTER = "center"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"
    BOTTOM_RIGHT = "bottom_right"


class VisualLayer(BaseModel):
    layer_id: str
    kind: LayerKind
    source_ref: str = Field(description=(
        'Every non-streamer-camera layer uses literal source_ref="source", not '
        '"source_video", a filename, arbitrary ID, empty or null. STREAMER_CAMERA alone '
        'may use a supplied locally applicable face/webcam track ID, with confident identity '
        'and its own valid crop. Whole-source preservation without an identity claim uses '
        'kind="source_video", source_ref="source".'
    ))
    anchor: Anchor = Anchor.TOP_LEFT
    source_crop: NormalizedRect | None = None
    destination: NormalizedRect = Field(description=(
        "Normalized output rectangle. Each v2 shot needs a visible full-canvas layer "
        "at x=0,y=0,width=1,height=1; split foregrounds alone are insufficient."
    ))
    fill: FillMode = Field(description=(
        "cover scales and center-crops after source_crop; contain_blur combines a blurred "
        "cover backdrop plus sharp contained foreground; black combines black destination "
        "plus sharp contained foreground. Current director ready support is source-backed "
        "cover/contain_blur/black only; inherit/solid require review."
    ))
    solid_color: str | None = None
    blur_sigma: float | None = Field(default=None, gt=0, le=50, description=(
        "Explicit positive sigma <=50 is required by the ready host gate for contain_blur."
    ))
    z_index: int = Field(ge=0)
    opacity: float = Field(default=1.0, ge=0, le=1, description=(
        "Current director ready geometry requires opacity=1; translucency requires review."
    ))
    crop_track: list[CropKeyframe] = Field(default_factory=list)
    preserve_regions: list[NormalizedRect] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_layer(self) -> "VisualLayer":
        if self.source_ref != "source" and not _TRACK_ID_PATTERN.fullmatch(self.source_ref):
            raise ValueError("source_ref debe ser 'source' o un track ID de Fase 1")
        if self.fill == FillMode.SOLID and not self.solid_color:
            raise ValueError("solid_color es obligatorio para fill solid")
        if self.fill != FillMode.SOLID and self.solid_color is not None:
            raise ValueError("solid_color solo se permite para fill solid")
        if self.source_crop is not None and any(
            not _rectangle_contains(self.source_crop, region)
            for region in self.preserve_regions
        ):
            raise ValueError("preserve_regions debe quedar completamente contenido en source_crop")
        return self


class OverlayPlan(BaseModel):
    text: str | None = Field(default=None, description=(
        "Exactly one of text or asset_id is non-null. A shot title matches headline.overlay_text."
    ))
    asset_id: str | None = None
    font_asset_id: str | None = None
    start: float = Field(ge=0, description="Seconds relative to shot duration, not global output time.")
    end: float = Field(gt=0, description="Seconds relative to shot duration; start < end <= duration.")
    destination: NormalizedRect
    z_index: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_overlay(self) -> "OverlayPlan":
        if (self.text is None) == (self.asset_id is None):
            raise ValueError("Un overlay requiere exactamente uno de text o asset_id")
        if self.end <= self.start:
            raise ValueError("El overlay debe terminar después de iniciar")
        if self.asset_id is not None:
            _validate_asset_id(self.asset_id, "asset_id")
        if self.font_asset_id is not None:
            _validate_asset_id(self.font_asset_id, "font_asset_id")
        return self


class SubtitlePlan(BaseModel):
    destination: NormalizedRect
    font_asset_id: str = "subtitle_bold"
    font_size: int = Field(default=58, ge=36, le=84)
    primary_color: str = "#FFD51A"
    outline_color: str = "#000000"
    outline_width: int = Field(default=4, ge=0, le=8)
    max_words_per_event: int = Field(default=6, ge=1, le=10)
    max_lines: int = Field(default=2, ge=1, le=3)
    emphasize_active_word: bool = False

    @model_validator(mode="after")
    def validate_font_asset(self) -> "SubtitlePlan":
        _validate_asset_id(self.font_asset_id, "font_asset_id")
        return self


class AudioCue(BaseModel):
    asset_id: str
    start: float = Field(ge=0, description=(
        "Seconds relative to shot duration. Use only offered host-approved SFX IDs; "
        "retain exact offered asset duration and explicit safe gain/duck settings."
    ))
    end: float = Field(gt=0, description="Seconds relative to shot duration; start < end <= duration.")
    gain_db: float = Field(default=-12.0, ge=-36, le=0)
    duck_source_db: float = Field(default=-6.0, ge=-18, le=0)

    @model_validator(mode="after")
    def validate_cue(self) -> "AudioCue":
        if self.end <= self.start:
            raise ValueError("El cue de audio debe terminar después de iniciar")
        _validate_asset_id(self.asset_id, "asset_id")
        return self


class AudioMixPlan(BaseModel):
    source_gain_db: float = Field(default=0.0, ge=-24, le=12)
    normalize_dialogue: bool = True
    limiter: bool = True
    cues: list[AudioCue] = Field(default_factory=list)


class VisualShot(BaseModel):
    shot_id: str
    source: TimeRange = Field(description=(
        "Exact original editorial segment subdivision at supplied mandatory source-state "
        "events, in original order; source and output durations must match."
    ))
    output: TimeRange = Field(description=(
        "Continuous global output time, starting at 0 across shots. Nested keyframe, "
        "overlay and cue times instead use seconds relative to this shot's duration."
    ))
    primary_source: Literal["content", "streamer", "shared"]
    reason: str
    layers: list[VisualLayer] = Field(min_length=1)
    subtitle_allowed_regions: list[NormalizedRect] = Field(default_factory=list)
    forbidden_regions: list[NormalizedRect] = Field(default_factory=list)
    subtitles: SubtitlePlan | None = None
    title: OverlayPlan | None = Field(default=None, description=(
        "If present, text exactly equals editorial headline.overlay_text; centered in host "
        "title-safe region without occluding protected source geometry."
    ))
    brand: OverlayPlan | None = None
    transition_in: Literal["cut"] = "cut"
    transition_out: Literal["cut"] = "cut"
    audio: AudioMixPlan = Field(default_factory=AudioMixPlan)
    evidence_ids: list[str] = Field(min_length=1, description=(
        "Only supplied evidence IDs local to this shot: applicable span, sample or frame IDs."
    ))
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_shot(self) -> "VisualShot":
        duration = self.output.duration
        if abs(self.source.duration - duration) > _TOLERANCE_SECONDS:
            raise ValueError("Las duraciones source y output deben coincidir")
        exact_duration = Fraction(str(self.output.end)) - Fraction(str(self.output.start))
        _ensure_unique((layer.layer_id for layer in self.layers), "layer_id")
        for layer in self.layers:
            self._validate_crop_track(layer, exact_duration)
        for overlay in (self.title, self.brand):
            if overlay is not None:
                self._validate_overlay(overlay, exact_duration)
        if self.subtitles is not None:
            self._validate_destination(self.subtitles.destination, "subtitles")
            if self.subtitle_allowed_regions and not any(
                _rectangle_contains(region, self.subtitles.destination)
                for region in self.subtitle_allowed_regions
            ):
                raise ValueError(
                    "subtitles debe quedar completamente contenido en subtitle_allowed_regions"
                )
        self._validate_audio(exact_duration)
        return self

    def _validate_crop_track(self, layer: VisualLayer, duration: Fraction) -> None:
        keyframes = layer.crop_track
        previous_at = -1.0
        for keyframe in keyframes:
            if keyframe.at <= previous_at or Fraction(str(keyframe.at)) > duration:
                raise ValueError("crop_track debe estar ordenado, ser único y quedar dentro del shot")
            if any(
                not _rectangle_contains(keyframe.crop, region)
                for region in layer.preserve_regions
            ):
                raise ValueError(
                    "preserve_regions debe quedar completamente contenido en cada crop de crop_track"
                )
            previous_at = keyframe.at
        if len(keyframes) > 1 and (
            abs(keyframes[0].at) > _REPRESENTATION_EPSILON_SECONDS
            or abs(Fraction(str(keyframes[-1].at)) - duration) > Fraction(str(_REPRESENTATION_EPSILON_SECONDS))
        ):
            raise ValueError("crop_track móvil debe iniciar en 0 y terminar en la duración del shot")

    def _validate_overlay(self, overlay: OverlayPlan, duration: Fraction) -> None:
        if Fraction(str(overlay.end)) > duration:
            raise ValueError("El overlay debe quedar dentro de la duración del shot")
        self._validate_destination(overlay.destination, "overlay")

    def _validate_destination(self, destination: NormalizedRect, label: str) -> None:
        if any(_rectangles_intersect(destination, region) for region in self.forbidden_regions):
            raise ValueError(f"{label} intersecta una región forbidden")

    def _validate_audio(self, duration: Fraction) -> None:
        previous_start = -1.0
        for cue in self.audio.cues:
            if cue.start < previous_start or Fraction(str(cue.end)) > duration:
                raise ValueError("Los cues de audio deben estar ordenados y dentro del shot")
            previous_start = cue.start


class VisualCompositionPlan(BaseModel):
    contract_version: Literal[2] = 2
    clip_id: str
    width: Literal[1080] = 1080
    source_fingerprint: str
    composition_fingerprint: str
    source_timeline_fingerprint: str
    identity_fingerprint: str
    style_library_fingerprint: str
    height: Literal[1920] = 1920
    shots: list[VisualShot] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_composition(self) -> "VisualCompositionPlan":
        _ensure_unique((shot.shot_id for shot in self.shots), "shot_id")
        previous_end = 0.0
        for index, shot in enumerate(self.shots):
            if index == 0 and abs(shot.output.start) > _TOLERANCE_SECONDS:
                raise ValueError("Los shots output deben iniciar en 0")
            if index and abs(shot.output.start - previous_end) > _TOLERANCE_SECONDS:
                raise ValueError("Los shots output deben ser continuos")
            if index == 0 and any(layer.fill == FillMode.INHERIT for layer in shot.layers):
                raise ValueError("La primera toma no puede usar inherit")
            if not any(_is_visible_full_frame(layer) for layer in shot.layers):
                raise ValueError("Al menos una layer visible cubre cada shot")
            previous_end = shot.output.end
        if previous_end > MAX_CLIP_SECONDS:
            raise ValueError(f"El clip supera el máximo de {MAX_CLIP_SECONDS:.0f}s")
        return self


def validate_plan_references(
    plan: VisualCompositionPlan,
    timeline: SourceVisualTimeline,
) -> None:
    if plan.source_fingerprint != timeline.source_fingerprint:
        raise ValueError(
            "source_fingerprint no coincide: "
            f"plan='{plan.source_fingerprint}', timeline='{timeline.source_fingerprint}'"
        )
    valid_source_refs = {
        "source",
        *(track.track_id for track in timeline.face_tracks),
        *(track.track_id for track in timeline.webcam_tracks),
    }
    for shot in plan.shots:
        for layer in shot.layers:
            if layer.source_ref not in valid_source_refs:
                raise ValueError(
                    "source_ref desconocido "
                    f"en shot_id='{shot.shot_id}', layer_id='{layer.layer_id}': "
                    f"source_ref='{layer.source_ref}'"
                )


def _ensure_unique(values: Iterable[str], label: str) -> None:
    seen: set[str] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"{label} debe ser único")
        seen.add(value)


def _validate_asset_id(value: str, label: str) -> None:
    if not _ASSET_ID_PATTERN.fullmatch(value):
        raise ValueError(f"{label} debe ser un asset ID seguro, no un path o URI")


def _rectangles_intersect(left: NormalizedRect, right: NormalizedRect) -> bool:
    # Compare serialized decimal geometry exactly: .48 + .34 touches .82.
    # An epsilon would also hide genuine, arbitrarily small positive overlaps.
    lx, ly, lw, lh = (Fraction(str(v)) for v in (left.x, left.y, left.width, left.height))
    rx, ry, rw, rh = (Fraction(str(v)) for v in (right.x, right.y, right.width, right.height))
    return (
        lx < rx + rw
        and rx < lx + lw
        and ly < ry + rh
        and ry < ly + lh
    )


def _rectangle_contains(container: NormalizedRect, contained: NormalizedRect) -> bool:
    epsilon = _RECT_CONTAINMENT_EPSILON
    return (
        contained.x >= container.x - epsilon
        and contained.y >= container.y - epsilon
        and contained.x + contained.width
        <= container.x + container.width + epsilon
        and contained.y + contained.height
        <= container.y + container.height + epsilon
    )


def _is_visible_full_frame(layer: VisualLayer) -> bool:
    destination = layer.destination
    return (
        layer.opacity > 0
        and destination.x == 0
        and destination.y == 0
        and destination.x + destination.width >= 1
        and destination.y + destination.height >= 1
    )


_HEX_COLOR_PATTERN = re.compile(r"^#[0-9a-fA-F]{6}$")


def _prepare_strict_visual_input(annotation: object, value: object) -> object:
    """Validate new model boundaries without changing historical nested contracts.

    Existing nested models are copied to data and revalidated, including instances
    created with model_construct. Never mutate a provider or host payload in place.
    Normalize mappings; only list/tuple containers may supply list fields, so
    Pydantic cannot coerce uninspected iterators after this recursive boundary.
    """
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        if isinstance(value, BaseModel):
            value = value.model_dump()
        if not isinstance(value, Mapping):
            return value
        fields = annotation.model_fields
        if value.keys() - fields.keys():
            raise ValueError(f"Unrequested fields in {annotation.__name__}")
        result: dict[str, object] = {}
        for name, item in value.items():
            if name in {"solid_color", "primary_color", "outline_color"} and item is not None:
                if not isinstance(item, str) or not _HEX_COLOR_PATTERN.fullmatch(item):
                    raise ValueError(f"{name} requires a #RRGGBB color")
            result[name] = _prepare_strict_visual_input(fields[name].annotation, item)
        return result
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in (Union, UnionType):
        if value is None:
            return value
        for member in arguments:
            if member is not type(None):
                return _prepare_strict_visual_input(member, value)
    if origin is list:
        if not isinstance(value, (list, tuple)):
            raise ValueError("List fields require a list or tuple")
        return [_prepare_strict_visual_input(arguments[0], item) for item in value]
    if annotation is float:
        try:
            finite = math.isfinite(float(value))  # type: ignore[arg-type]
        except (TypeError, ValueError, OverflowError):
            return value  # Let Pydantic report invalid numeric types.
        if not finite:
            raise ValueError("Numeric values must be finite")
    return value


class _StrictVisualContract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    @model_validator(mode="before")
    @classmethod
    def validate_strict_payload(cls, value: object) -> object:
        return _prepare_strict_visual_input(cls, value)


class VisualDirectorStatus(str, Enum):
    READY = "ready"
    NEED_MORE_EVIDENCE = "need_more_evidence"
    REVIEW_REQUIRED = "review_required"


class VisualEvidenceRequest(_StrictVisualContract):
    request_id: str = Field(min_length=1)
    clip_id: str = Field(min_length=1)
    source: TimeRange
    question: str = Field(min_length=1)
    sample_interval_seconds: float = Field(ge=0.25, le=2.0)
    reason: Literal[
        "identity_ambiguous", "camera_boundary", "important_region", "state_transition", "crop_motion"
    ]
    iteration: int = Field(ge=1, le=3)


class VisualDirectorResponse(_StrictVisualContract):
    contract_version: Literal[1] = 1
    status: VisualDirectorStatus
    plan: VisualCompositionPlan | None = None
    evidence_requests: list[VisualEvidenceRequest] = Field(default_factory=list, max_length=3)
    style_card_ids: list[str] = Field(default_factory=list, description=(
        "Only offered IDs. Compatible unreviewed cards may inform visual inspiration; "
        "not human approval, narrative scoring, original-VOD pairing or source identity/geometry."
    ))
    reasoning: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_status(self) -> "VisualDirectorResponse":
        _ensure_unique((request.request_id for request in self.evidence_requests), "request_id")
        if self.status == VisualDirectorStatus.READY:
            if self.plan is None or self.evidence_requests:
                raise ValueError("ready requires a plan and forbids evidence_requests")
        elif self.status == VisualDirectorStatus.NEED_MORE_EVIDENCE:
            if self.plan is not None or not self.evidence_requests:
                raise ValueError("need_more_evidence requires 1..3 requests and forbids a plan")
        elif self.plan is not None:
            raise ValueError("review_required forbids a plan")
        return self


VisualEvidenceSignature = tuple[str, str, float, float, str]


def visual_evidence_request_signature(request: VisualEvidenceRequest) -> VisualEvidenceSignature:
    """Namespace Task 6's reason/window/question key by clip; round to milliseconds.

    Whitespace in the question is normalized. IDs, iterations and sampling cadence
    do not create a new signature, while a different question may clarify evidence.
    """
    return (
        request.clip_id, request.reason, round(request.source.start, 3),
        round(request.source.end, 3), " ".join(request.question.split()),
    )


def validate_visual_director_context(
    response: VisualDirectorResponse,
    *,
    clip_id: str,
    editorial_ranges: Sequence[TimeRange],
    source_duration: float,
    prior_request_signatures: Collection[VisualEvidenceSignature] = (),
) -> None:
    """Validate against host authority; raise ValueError and never mutate inputs.

    Call after every response and retain signatures of collected requests across
    iterations. Each request fits one editorial window expanded by at most 15s,
    clamped to [0, source_duration]. Exact final editorial coverage is Task 6's gate.
    """
    if not math.isfinite(source_duration) or source_duration <= 0:
        raise ValueError("source_duration must be finite and positive")
    if not editorial_ranges or any(
        not math.isfinite(window.start) or not math.isfinite(window.end)
        or window.start < 0 or window.end <= window.start or window.end > source_duration
        for window in editorial_ranges
    ):
        raise ValueError("editorial_ranges must be valid windows within source_duration")
    if response.plan is not None:
        if response.plan.clip_id != clip_id:
            raise ValueError("plan clip_id does not match host clip_id")
        if any(shot.source.end > source_duration for shot in response.plan.shots):
            raise ValueError("plan source window exceeds source_duration")
    signatures = set(prior_request_signatures)
    for request in response.evidence_requests:
        if request.clip_id != clip_id:
            raise ValueError("request clip_id does not match host clip_id")
        if not any(
            request.source.start >= max(0.0, window.start - 15.0)
            and request.source.end <= min(source_duration, window.end + 15.0)
            for window in editorial_ranges
        ):
            raise ValueError("request source must fit a single expanded editorial window")
        signature = visual_evidence_request_signature(request)
        if signature in signatures:
            raise ValueError("request signature already collected or repeated in batch")
        signatures.add(signature)
