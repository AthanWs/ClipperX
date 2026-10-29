"""Stateless, bounded visual direction with host-authoritative publication gates.

run_visual_director returns visual_plan.json (a validated v2 plan) on success,
or receipt.json with status=review_required. Every invocation has a unique run
directory. The receipt is the publication marker; no legacy plan is fabricated.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import Field, create_model, model_validator

from . import evidence as observed_evidence
from .evidence import associated_face_positions, collect_visual_evidence, ObservedEvidenceContext
from .jobs import JobStore
from .local_paths import atomic_write_text, source_fingerprint, safe_slug, paths as local_paths
from .render_clock import SourceTimeOrigin, StreamClock, exact, canonical
from .render_runtime import RenderRuntime
from .visual_provenance import (file_sha256, read_generation_json, visual_timeline_fingerprint,
                                load_visual_provenance, generation_file, emitted_text_size, MAX_ARTIFACT_BYTES)
from .planner_contracts import ClipPlan, MAX_CLIP_SECONDS, TimeRange
from .output_timeline import OutputTimeline
from .schema import TranscriptSegment, Word
from .platform_safety import common_vertical_safe_zones
from .style_contracts import SceneFamily, StyleCard
from .style_library import StyleLibrary
from .style_retrieval import StyleQuery, retrieve_style_cards
from .visual_contracts import (
    FillMode, IdentityStatus, LayerKind, NormalizedRect, SourceState,
    SourceVisualSpan, SourceVisualTimeline, VisualDirectorResponse,
    VisualDirectorStatus, VisualEvidenceSignature, VisualEvidenceRequest, VisualLayer, VisualShot, VisualCompositionPlan,
    _rectangle_contains, _rectangles_intersect, validate_visual_director_context,
    visual_evidence_request_signature,
)
from .visual_prompts import (
    VISUAL_DIRECTOR_SYSTEM_PROMPT, build_visual_director_prompt, creator_overlay_requirements,
)
from .sfx_authority import (capture_sfx_authority, validate_sfx_authority, sfx_offers,
                            source_audio_channels, validate_sfx_cues)

if TYPE_CHECKING:
    # Task 7 may import this module from planner_engine; no runtime cycle.
    from .planner_engine import StructuredClient


@dataclass(frozen=True, slots=True)
class VisualDirectorLimits:
    max_calls_per_clip: int = 6
    max_followups_per_segment: int = 2
    max_source_images_per_call: int = 5
    style_cards_per_segment: int = 3
    max_style_images_per_card: int = 1
    transcript_padding_seconds: float = 2.0
    max_transcript_chars: int = 8000
    require_creator_overlays: bool = False

    def __post_init__(self) -> None:
        if type(self.require_creator_overlays) is not bool:
            raise ValueError("require_creator_overlays must be a bool")
        bounds = {
            "max_calls_per_clip": (1, 6), "max_followups_per_segment": (0, 2),
            "max_source_images_per_call": (1, 5), "style_cards_per_segment": (1, 3),
            "max_style_images_per_card": (1, 1), "max_transcript_chars": (1, 8000),
        }
        for name, (minimum, maximum) in bounds.items():
            value = getattr(self, name)
            if type(value) is not int or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be in {minimum}..{maximum}")
        if (not math.isfinite(self.transcript_padding_seconds)
                or not 0 <= self.transcript_padding_seconds <= 15):
            raise ValueError("transcript_padding_seconds must be finite and in 0..15")


_FULL = NormalizedRect(x=0, y=0, width=1, height=1)
_FONTS = {"subtitle_bold", "headline_impact"}
_IMAGES = {"kick_brand"}
_ASSETS = _FONTS | _IMAGES
_USAGE_FIELDS = {"prompt_tokens", "completion_tokens", "total_tokens", "estimated_input_tokens",
                 "input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"}
_EPS = 1e-6


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _write(path: Path, value: Any) -> None:
    text=_json(value)
    if emitted_text_size(text)>MAX_ARTIFACT_BYTES:
        raise ValueError("Director JSON quota exceeded before write")
    atomic_write_text(path, text)


def _overlap(a: TimeRange, b: TimeRange) -> bool:
    return a.start < b.end and a.end > b.start


def _local_spans(timeline: SourceVisualTimeline, window: TimeRange) -> list[SourceVisualSpan]:
    return [span for span in timeline.spans if _overlap(span.source, window)]


def _required_shots(clip: ClipPlan, timeline: SourceVisualTimeline) -> list[dict[str, float]]:
    required = []
    for segment in clip.segments:
        window = segment.source
        spans = _local_spans(timeline, window)
        cursor = window.start
        for span in spans:
            a, b = max(window.start, span.source.start), min(window.end, span.source.end)
            if abs(a - cursor) > _EPS:
                raise ValueError("Source timeline has an unresolved local coverage gap")
            required.append({"start": a, "end": b})
            cursor = b
        if abs(cursor - window.end) > _EPS:
            raise ValueError("Source timeline does not cover editorial segment")
    return required


def _fingerprints(clip: ClipPlan, timeline: SourceVisualTimeline, style_fp: str) -> dict[str, str]:
    return {
        "source_fingerprint": timeline.source_fingerprint,
        "composition_fingerprint": _hash(clip.model_dump(mode="json")),
        "source_timeline_fingerprint": visual_timeline_fingerprint(timeline),
        "identity_fingerprint": _hash({
            "face_tracks": [t.model_dump(mode="json") for t in timeline.face_tracks],
            "webcam_tracks": [t.model_dump(mode="json") for t in timeline.webcam_tracks],
            "local_identity": [(s.span_id, s.identity_status.value, s.streamer_track_id,
                                s.webcam_track_id) for s in timeline.spans],
        }),
        "style_library_fingerprint": style_fp,
    }


_DIRECTOR_FINGERPRINT_FIELDS = (
    "source_fingerprint",
    "composition_fingerprint",
    "source_timeline_fingerprint",
    "identity_fingerprint",
    "style_library_fingerprint",
)


def _response_model_for_fingerprints(
    fingerprints: dict[str, str],
) -> type[VisualDirectorResponse]:
    if set(fingerprints) != set(_DIRECTOR_FINGERPRINT_FIELDS):
        raise ValueError("Invalid director fingerprint authority")
    expected = tuple((name, fingerprints[name]) for name in _DIRECTOR_FINGERPRINT_FIELDS)
    if any(type(value) is not str or not value for _, value in expected):
        raise ValueError("Invalid director fingerprint authority")

    @model_validator(mode="after")
    def validate_request_authority(plan: VisualCompositionPlan) -> VisualCompositionPlan:
        if any(getattr(plan, name) != value for name, value in expected):
            raise ValueError("Plan fingerprints must match request authority")
        return plan

    plan_fields: dict[str, Any] = {
        name: (str, Field(json_schema_extra={"const": value}))
        for name, value in expected
    }
    validators: dict[str, Any] = {"validate_request_authority": validate_request_authority}
    bound_plan = create_model(
        "RequestVisualCompositionPlan",
        __base__=VisualCompositionPlan,
        __validators__=validators,
        **plan_fields,
    )
    response_model = create_model(
        "RequestVisualDirectorResponse",
        __base__=VisualDirectorResponse,
        plan=(bound_plan | None, None),
    )
    return response_model


def _select_styles(
    clip: ClipPlan, timeline: SourceVisualTimeline, library: StyleLibrary, limits: VisualDirectorLimits,
) -> tuple[list[StyleCard], list[dict[str, Any]]]:
    groups = []
    for segment in clip.segments:
        spans = _local_spans(timeline, segment.source)
        seen = set()
        for span in spans:
            key = (span.state, bool(span.webcam_track_id))
            if key in seen:
                continue
            seen.add(key)
            # A generic CONTENT_WITH_CAMERA never implies gameplay. Each local
            # state gets a separate query so a transition keeps both references.
            family = {
                SourceState.FULL_CAMERA: SceneFamily.FULL_CAMERA,
                SourceState.CONTENT_ONLY: SceneFamily.CONTENT_ONLY,
                SourceState.MULTI_PERSON: SceneFamily.MULTI_PERSON,
            }.get(span.state, SceneFamily.UNKNOWN)
            if span.state == SourceState.BROWSER_TEXT and span.webcam_track_id:
                family = SceneFamily.BROWSER_WITH_CAMERA
            elif span.state == SourceState.REACTED_VIDEO and span.webcam_track_id:
                family = SceneFamily.REACTED_VIDEO_WITH_CAMERA
            camera = (True if span.webcam_track_id or span.state == SourceState.FULL_CAMERA
                      else False if span.state == SourceState.CONTENT_ONLY else None)
            result = retrieve_style_cards(StyleQuery(
                scene_family=family, has_camera=camera,
                preserve_content=span.state in {
                    SourceState.CONTENT_WITH_CAMERA, SourceState.CONTENT_ONLY,
                    SourceState.BROWSER_TEXT, SourceState.REACTED_VIDEO,
                } or (bool(span.important_regions or span.text_regions)
                      and span.state != SourceState.FULL_CAMERA),
                people_requirement="multiple" if family == SceneFamily.MULTI_PERSON else "any",
                description=f"{clip.headline.overlay_text}. {segment.reason[:1000]}",
            ), library, limit=limits.style_cards_per_segment)
            groups.append((segment.output_order, family, result))
    cards: list[StyleCard] = []
    per_segment: dict[int, set[str]] = {}
    # Round-robin preserves state diversity before adding alternative examples.
    for rank in range(limits.style_cards_per_segment):
        for order, _, result in groups:
            if rank >= len(result):
                continue
            card = result[rank].card
            selected = per_segment.setdefault(order, set())
            if len(selected) >= limits.style_cards_per_segment:
                continue
            if card.card_id in {c.card_id for c in cards}:
                selected.add(card.card_id)
            elif len(cards) < 3:
                cards.append(card)
                selected.add(card.card_id)
    offered = {c.card_id for c in cards}
    coverage = [{
        "output_order": order, "scene_family": family.value, "coverage": result.coverage,
        "selected_card_ids": [m.card.card_id for m in result if m.card.card_id in offered
                              and m.card.card_id in per_segment.get(order, set())],
        "approval": {m.card.card_id: m.card.approval for m in result},
        "rejections": [{"card_id": r.card_id, "reasons": list(r.reasons)} for r in result.rejections[:32]],
        "rejection_count": len(result.rejections),
    } for order, family, result in groups]
    return cards, coverage


def _transcript(text: dict[str,Any], *, require_complete: bool = False) -> list[dict[str, Any]]:
    if text["path"] is None:
        return []
    with Path(text["path"]).open("rb") as stream:
        encoded=stream.read(MAX_ARTIFACT_BYTES+1)
    if len(encoded)>MAX_ARTIFACT_BYTES or hashlib.sha256(encoded).hexdigest()!=text["sha256"]:
        raise ValueError("Captured transcript identity changed before consumption")
    value=json.loads(encoded)
    if not isinstance(value, list):
        raise ValueError("Local transcript must be a list")
    if require_complete and any(not isinstance(row, dict) for row in value):
        raise ValueError("Creator transcript contains an invalid segment")
    return [row for row in value if isinstance(row, dict)]


def _creator_mode(authority: dict[str, Any]) -> bool:
    limits = VisualDirectorLimits(**authority["limits"])
    policy = authority["host_context"].get("creator_overlays")
    if policy is None and not limits.require_creator_overlays:
        return False  # Historical authority predates this opt-in policy.
    if canonical(policy) != canonical(creator_overlay_requirements(limits.require_creator_overlays)):
        raise ValueError("Captured creator overlay policy contradicts Director limits")
    return limits.require_creator_overlays


def _validate_creator_overlays(plan: VisualCompositionPlan, clip: ClipPlan,
                               transcript: list[dict[str, Any]] | None) -> None:
    """Require creator completeness; geometry and actual rendered quality stay separate."""
    if transcript is None:
        raise ValueError("Creator READY requires a known transcript; missing is not silence")
    segments = []
    try:
        for row in transcript:
            words = row.get("words", [])
            if not isinstance(words, list):
                raise ValueError("Creator transcript requires a word list when supplied")
            segments.append(TranscriptSegment(start=row["start"], end=row["end"], text=row["text"],
                words=[Word(start=word["start"], end=word["end"], word=word["word"]) for word in words]))
    except (KeyError, TypeError) as exc:
        raise ValueError("Creator transcript contains invalid timing/text fields") from exc
    # This is the same exact, half-open occurrence mapping used by ASS captions.
    speech_shots = {event.shot_id for event in OutputTimeline.from_shots(plan.shots).remap(segments)}
    for segment in segments:
        if (segment.text.strip() and segment.words and all(not word.word.strip() for word in segment.words)
                and any(segment.start < shot.source.end and segment.end > shot.source.start
                        for shot in plan.shots)):
            raise ValueError("Creator transcript cannot align nonempty text to blank word tokens")
    for index, shot in enumerate(plan.shots):
        brand = shot.brand
        if (brand is None or brand.asset_id != "kick_brand" or brand.start != 0
                or exact(str(brand.end)) != exact(str(shot.output.end)) - exact(str(shot.output.start))):
            raise ValueError("Creator READY requires independent kick_brand throughout every shot")
        if index == 0 and (shot.title is None or shot.title.text != clip.headline.overlay_text
                           or shot.title.start != 0 or shot.title.end <= 0):
            raise ValueError("Creator READY requires the editorial title at the first shot start")
        subtitles = shot.subtitles
        if shot.shot_id in speech_shots and subtitles is None:
            raise ValueError("Creator READY requires subtitles for every shot with local transcript speech")
        if subtitles is not None and (subtitles.font_asset_id != "subtitle_bold"
                or subtitles.primary_color.upper() != "#FFFF00" or subtitles.outline_color.upper() != "#000000"
                or subtitles.font_size < 68 or subtitles.outline_width < 7):
            raise ValueError("Creator subtitles require subtitle_bold, yellow/black, font>=68 and outline>=7")


def _source_manifest(bundles: list[dict[str, Any]],limits: VisualDirectorLimits) -> list[dict[str,Any]]:
    frames: list[dict[str, Any]] = []
    unique: dict[str,dict[str,Any]] = {}
    observed_pts: dict[tuple[Any,...],tuple[str,tuple[Any,...]]]={}
    frame_aliases: dict[str,str]={}
    for bundle in bundles:
        for frame in bundle["frames"]:
            key=frame["frame_id"]
            pts=(bundle.get("origin_id"),frame["raw_pts"]*exact(frame["time_base"]))
            identity=(frame.get("raw_bgr_sha256"),frame.get("width"),frame.get("height"),
                exact(frame["observed_source_start"]) if "observed_source_start" in frame else None,
                exact(frame["observed_source_end"]) if "observed_source_end" in frame else None)
            if pts in observed_pts:
                canonical_key,known_identity=observed_pts[pts]
                if known_identity!=identity:
                    raise ValueError("Conflicting same-PTS Director images across bundles")
                frame_aliases[key]=canonical_key
                key=canonical_key
            else:
                observed_pts[pts]=(key,identity)
                frame_aliases[key]=key
            refs=[dict(ref,bundle_path=bundle["bundle_path"],receipt_path=str(
                Path(bundle["bundle_path"]).parent/Path(bundle["bundle_path"]).stem/ref["receipt"]))
                if "bundle_path" in bundle else dict(ref) for ref in frame["sample_refs"]]
            if key not in unique:
                unique[key]=dict(frame,aliases=[frame["evidence_id"]],sample_refs=refs)
            else:
                target=unique[key]
                if frame["evidence_id"] not in target["aliases"]:
                    target["aliases"].append(frame["evidence_id"])
                for ref in refs:
                    if ref not in target["sample_refs"]:
                        target["sample_refs"].append(ref)
    # Every outstanding request needs actual views in this stateless call.
    # Temporal questions need two distinct instants; metadata is not imagery.
    seen: set[str] = set()
    for bundle in bundles:
        request = bundle.get("request")
        if request is None:
            continue
        window = request["source"]
        local = {frame_aliases[frame["frame_id"]]: unique[frame_aliases[frame["frame_id"]]] for frame in bundle["frames"]
                 if any(Fraction(str(window["start"])) <= exact(ref["requested"]) < Fraction(str(window["end"]))
                        for ref in frame["sample_refs"])}
        temporal = request["reason"] in {"camera_boundary", "state_transition", "crop_motion"}
        if len(local) < (2 if temporal else 1):
            raise ValueError("Requested evidence lacks distinct temporal/local image views")
        ordered=sorted(local,key=lambda key:exact(local[key]["time_base"])*local[key]["raw_pts"])
        keys = [ordered[0],ordered[-1]] if temporal else [ordered[0]]
        for key in keys:
            if key not in seen:
                frames.append(dict(local[key], kind="source"))
                seen.add(key)
    if len(frames) > limits.max_source_images_per_call:
        raise ValueError("Required requested-window views exceed source image budget")
    # Remaining slots retain event-prioritized views and the complete local
    # evidence metadata remains available even when its images do not fit.
    for bundle in reversed(bundles):
        for frame in bundle["frames"]:
            key = frame_aliases[frame["frame_id"]]
            if key not in seen and len(frames) < limits.max_source_images_per_call:
                frames.append(dict(unique[key], kind="source"))
                seen.add(key)
    return frames


def _manifest(
    bundles: list[dict[str, Any]], cards: list[StyleCard], library: StyleLibrary,
    limits: VisualDirectorLimits,
) -> tuple[list[dict[str, Any]], list[Path]]:
    frames=_source_manifest(bundles,limits)
    for card in cards[:3]:
        path = (library.root / card.keyframes[0].path).resolve(strict=True)
        if not path.is_relative_to(library.root.resolve()) or not path.is_file():
            raise ValueError("Style keyframe escapes library authority")
        frames.append({"kind": "style", "card_id": card.card_id, "path": str(path),
                       "approval": card.approval, "at": card.keyframes[0].timestamp})
    for index, frame in enumerate(frames):
        frame["image_index"] = index
        frame["sha256"] = file_sha256(Path(frame["path"]))
    return frames, [Path(frame["path"]) for frame in frames]


def _visible_crop(layer: VisualLayer, crop: NormalizedRect, width: int, height: int) -> NormalizedRect:
    if layer.fill != FillMode.COVER:
        return crop
    source_ratio = crop.width * width / (crop.height * height)
    target_ratio = layer.destination.width * 1080 / (layer.destination.height * 1920)
    if source_ratio > target_ratio:
        visible_width = crop.width * target_ratio / source_ratio
        return NormalizedRect(x=crop.x + (crop.width - visible_width) / 2, y=crop.y,
                              width=visible_width, height=crop.height)
    visible_height = crop.height * source_ratio / target_ratio
    return NormalizedRect(x=crop.x, y=crop.y + (crop.height - visible_height) / 2,
                          width=crop.width, height=visible_height)


def _crop_probes(layer: VisualLayer, width: int, height: int) -> list[NormalizedRect]:
    """Endpoints and aspect crossovers bound the visible linear crop path."""
    crops = [layer.source_crop or _FULL, *(k.crop for k in layer.crop_track)]
    probes = list(crops)
    aspect = layer.destination.width * 1080 / (layer.destination.height * 1920) * height / width
    for first, last in zip(crops, crops[1:]):
        left = first.width - aspect * first.height
        right = last.width - aspect * last.height
        if left * right < 0:
            fraction = left / (left - right)
            probes.append(NormalizedRect(**{
                field: getattr(first, field) + fraction * (getattr(last, field) - getattr(first, field))
                for field in ("x", "y", "width", "height")
            }))
    return probes


def _project(layer: VisualLayer, visible: NormalizedRect, region: NormalizedRect,
             width: int, height: int) -> NormalizedRect:
    destination = layer.destination
    if layer.fill in (FillMode.CONTAIN_BLUR, FillMode.BLACK):
        source_ratio = visible.width * width / (visible.height * height)
        target_ratio = destination.width * 1080 / (destination.height * 1920)
        if source_ratio > target_ratio:
            h = destination.height * target_ratio / source_ratio
            destination = NormalizedRect(x=destination.x,
                y=destination.y + (destination.height - h) / 2, width=destination.width, height=h)
        else:
            w = destination.width * source_ratio / target_ratio
            destination = NormalizedRect(x=destination.x + (destination.width - w) / 2,
                y=destination.y, width=w, height=destination.height)
    return NormalizedRect(
        x=destination.x + (region.x - visible.x) / visible.width * destination.width,
        y=destination.y + (region.y - visible.y) / visible.height * destination.height,
        width=region.width / visible.width * destination.width,
        height=region.height / visible.height * destination.height,
    )


def _protected_regions(
    shot: VisualShot, timeline: SourceVisualTimeline, spans: list[SourceVisualSpan],
) -> list[NormalizedRect]:
    regions = [r for span in spans for r in [*span.important_regions, *span.text_regions]]
    webcam_ids = {span.webcam_track_id for span in spans}
    for track in timeline.webcam_tracks:
        if track.track_id in webcam_ids:
            regions.extend(o.rect for o in track.observations
                           if shot.source.start <= o.at < shot.source.end)
    for sample in timeline.samples:
        if shot.source.start <= sample.at < shot.source.end:
            regions.extend(sample.text_regions)
            regions.extend(sample.persistent_rectangles)
            if sample.webcam_rect:
                regions.append(sample.webcam_rect)
    # Face summaries do not timestamp positions. Protect all positions for each
    # locally referenced identity rather than claiming one is a local observation.
    local_faces = {span.streamer_track_id for span in spans}
    for sample in timeline.samples:
        if shot.source.start <= sample.at < shot.source.end:
            local_faces.update(sample.face_track_ids)
    for face_track in timeline.face_tracks:
        if (face_track.track_id in local_faces
                or face_track.first_seen < shot.source.end and face_track.last_seen > shot.source.start):
            positioned = associated_face_positions(timeline, face_track)
            if positioned is not None:
                regions.extend(rect for at, rect, _ in positioned if shot.source.start <= at < shot.source.end)
            else:
                # Honest conservative fallback: summary geometry is unscoped,
                # not falsely presented to the model as a local observation.
                regions.extend(face_track.positions)
    return regions


def _validate_identity(shot: VisualShot, spans: list[SourceVisualSpan],
                       timeline: SourceVisualTimeline, width: int, height: int) -> None:
    has_assertion = shot.primary_source in {"streamer", "shared"} or any(
        layer.kind == LayerKind.STREAMER_CAMERA for layer in shot.layers)
    if has_assertion and any(s.identity_status != IdentityStatus.CONFIDENT for s in spans):
        raise ValueError("Streamer identity is not confident in this local shot")
    faces = {t.track_id: t for t in timeline.face_tracks}
    webcams = {t.track_id: t for t in timeline.webcam_tracks}
    for layer in shot.layers:
        if layer.kind != LayerKind.STREAMER_CAMERA:
            if layer.source_ref != "source":
                raise ValueError("Only streamer_camera may reference a local track")
            continue
        for span in spans:
            face_id = span.streamer_track_id
            if face_id not in faces:
                raise ValueError("Unknown local streamer face track")
            face = faces[face_id]
            if face.first_seen > shot.source.start or face.last_seen < shot.source.end:
                raise ValueError("Streamer face track does not cover local shot")
            positioned = associated_face_positions(timeline, face)
            # An unscoped summary is conservative geometry, not fabricated
            # timestamps. Only a proven bijection may select local positions.
            regions = (list(face.positions) if positioned is None else
                       [rect for at, rect, _ in positioned if shot.source.start <= at < shot.source.end])
            if not regions:
                raise ValueError("Streamer layer lacks applicable local face geometry")
            if layer.source_ref == span.webcam_track_id:
                webcam = webcams.get(layer.source_ref)
                if (webcam is None or face_id not in webcam.face_track_ids
                        or not any(r.start <= shot.source.start and r.end >= shot.source.end
                                   for r in webcam.source_ranges)):
                    raise ValueError("Webcam track does not support local streamer identity")
                observations = [o.rect for o in webcam.observations
                                if shot.source.start <= o.at < shot.source.end]
                if not observations:
                    raise ValueError("Streamer layer lacks applicable local webcam geometry")
                regions.extend(observations)
            elif layer.source_ref != face_id:
                raise ValueError("Streamer layer references a nonmatching local track")
            if any(not _rectangle_contains(_visible_crop(layer, crop, width, height), region)
                   for crop in _crop_probes(layer, width, height) for region in regions):
                raise ValueError("Streamer layer visible crop does not contain its track geometry")


def _validate_geometry(
    shot: VisualShot, timeline: SourceVisualTimeline, spans: list[SourceVisualSpan],
    width: int, height: int, overlay_text: str,
) -> None:
    layers = sorted(shot.layers, key=lambda layer: layer.z_index)
    if any(layer.fill in {FillMode.INHERIT, FillMode.SOLID} or layer.kind == LayerKind.SOLID
           or layer.opacity != 1 for layer in layers):
        raise ValueError("Unresolved inherited/solid/translucent geometry requires review")
    if any(layer.fill == FillMode.CONTAIN_BLUR and layer.blur_sigma is None for layer in layers):
        raise ValueError("contain_blur requires explicit blur evidence geometry")
    for region in _protected_regions(shot, timeline, spans):
        preserved = False
        for index, layer in enumerate(layers):
            # Projection coordinates are linear-fractional within each aspect
            # branch. Their extrema occur at endpoints or a cover/contain aspect
            # crossover. Ease-in-out changes timing, not the geometric path.
            destinations = []
            for crop in _crop_probes(layer, width, height):
                visible = _visible_crop(layer, crop, width, height)
                if not _rectangle_contains(visible, region):
                    break
                destinations.append(_project(layer, visible, region, width, height))
            else:
                overlay_regions = [overlay.destination for overlay in (shot.title, shot.brand)
                                   if overlay is not None and overlay.z_index >= layer.z_index]
                if shot.subtitles is not None:
                    overlay_regions.append(shot.subtitles.destination)
                occluders = [upper.destination for upper in layers[index + 1:]] + overlay_regions
                left = min(rect.x for rect in destinations)
                top = min(rect.y for rect in destinations)
                envelope = NormalizedRect(x=left, y=top,
                    width=max(rect.x + rect.width for rect in destinations) - left,
                    height=max(rect.y + rect.height for rect in destinations) - top)
                if all(not _rectangles_intersect(envelope, occluder) for occluder in occluders):
                    preserved = True
                    break
        if not preserved:
            raise ValueError("Known important region/face/HUD/OCR is cut or occluded")
    if shot.title and shot.title.text != overlay_text:
        raise ValueError("Title must match editorial overlay_text")
    safety = common_vertical_safe_zones()
    for overlay in (shot.title, shot.brand, shot.subtitles):
        if overlay is None:
            continue
        destination = overlay.destination
        allowed = (safety.title if overlay is shot.title else safety.subtitle_allowed
                   if overlay is shot.subtitles else _FULL)
        forbidden = (safety.overlay_forbidden_right, safety.overlay_forbidden_bottom)
        if (not _rectangle_contains(allowed, destination)
                or any(_rectangles_intersect(destination, r) for r in forbidden)
                or abs(destination.x + destination.width / 2 - safety.canvas_center_x) > _EPS):
            raise ValueError("Overlays must be centered inside host overlay-safe region")
        for field in ("asset_id", "font_asset_id"):
            asset = getattr(overlay, field, None)
            allowed_assets = _IMAGES if field == "asset_id" else _FONTS
            if asset is not None and asset not in allowed_assets:
                raise ValueError("Unknown host overlay/font asset")
    if shot.brand and shot.brand.asset_id != "kick_brand":
        raise ValueError("Brand must use the host kick_brand asset")


def _validate_final(
    response: VisualDirectorResponse, clip: ClipPlan, timeline: SourceVisualTimeline,
    fingerprints: dict[str, str], cards: list[StyleCard], bundles: list[dict[str, Any]],
    width: int, height: int, *, presented_assets: list[dict[str,Any]] | None = None,
    host_sfx: dict[str,Any] | None = None, audio_channels: int | None = None,
    require_creator_overlays: bool = False, transcript: list[dict[str, Any]] | None = None,
) -> None:
    plan = response.plan
    if plan is None:
        raise ValueError("Ready response requires a plan")
    if require_creator_overlays:
        _validate_creator_overlays(plan, clip, transcript)
    validate_sfx_cues(plan, host_sfx, source_channels=audio_channels)
    for field, expected in fingerprints.items():
        if getattr(plan, field) != expected:
            raise ValueError(f"{field} differs from host authority")
    if set(response.style_card_ids) - {card.card_id for card in cards}:
        raise ValueError("Unknown or unoffered style card ID")
    shot_index = 0
    cursor = 0.0
    for segment in clip.segments:
        source_cursor = segment.source.start
        while shot_index < len(plan.shots) and source_cursor < segment.source.end - _EPS:
            shot = plan.shots[shot_index]
            if (abs(shot.source.start - source_cursor) > _EPS
                    or shot.source.end > segment.source.end + _EPS
                    or abs(shot.output.start - cursor) > _EPS):
                raise ValueError("Shots must exactly subdivide each editorial segment in order")
            spans = _local_spans(timeline, shot.source)
            if len(spans) != 1:
                raise ValueError("Shot silently spans a known source-state transition")
            known_ids = {spans[0].span_id, *spans[0].evidence_ids}
            for sample in timeline.samples:
                if shot.source.start <= sample.at < shot.source.end:
                    known_ids.update([sample.sample_id, *sample.evidence_ids])
            known_ids.difference_update(_image_evidence_ids(bundles))
            known_ids.update(_presented_local_ids(presented_assets or [],shot.source))
            if set(shot.evidence_ids) - known_ids:
                raise ValueError("Shot references unknown or nonlocal evidence IDs")
            _validate_identity(shot, spans, timeline, width, height)
            _validate_geometry(shot, timeline, spans, width, height, clip.headline.overlay_text)
            source_cursor = shot.source.end
            cursor = shot.output.end
            shot_index += 1
        if abs(source_cursor - segment.source.end) > _EPS:
            raise ValueError("Final plan lost editorial source coverage/payoff")
    if shot_index != len(plan.shots) or abs(cursor - clip.duration) > _EPS or cursor > MAX_CLIP_SECONDS:
        raise ValueError("Final plan adds source or exceeds editorial duration")
    # Model geometry cannot introduce arbitrary visual subdivisions either.
    required = _required_shots(clip, timeline)
    if [shot.source.model_dump() for shot in plan.shots] != required:
        raise ValueError("Visual subdivisions must align to source-state events")


def _image_evidence_ids(bundles: list[dict[str,Any]]) -> set[str]:
    return {frame["evidence_id"] for bundle in bundles for frame in bundle["frames"]}


def _presented_local_ids(manifest: list[dict[str,Any]],window: TimeRange) -> set[str]:
    return {alias for row in manifest if row["kind"]=="source" and any(
        Fraction(str(window.start))<=exact(ref["requested"])<Fraction(str(window.end)) for ref in row["sample_refs"])
        for alias in row["aliases"]}


def _check_final_image_citations(plan: VisualCompositionPlan,bundles: list[dict[str,Any]],
                                 manifest: list[dict[str,Any]]) -> None:
    image_ids=_image_evidence_ids(bundles)
    for shot in plan.shots:
        if (set(shot.evidence_ids)&image_ids)-_presented_local_ids(manifest,shot.source):
            raise ValueError("Plan cites an image not presented locally in the final call")


class DirectorPublicationError(RuntimeError):
    """Controlled host publication state; never carries provider error text."""

    def __init__(self, phase: str,run_path: Path,committed_path: Path | None = None,
                 expected_identities: dict[str,Any] | None = None) -> None:
        super().__init__(phase)
        self.phase=phase
        self.run_path=run_path
        self.committed_path=committed_path
        self.expected_identities=expected_identities or {}


def _known_bytes(path: Path,root: Path) -> bytes | None:
    try:
        path=generation_file(root,path.relative_to(root).as_posix())
        with path.open("rb") as stream:
            content=stream.read(MAX_ARTIFACT_BYTES+1)
        if len(content)>MAX_ARTIFACT_BYTES:
            raise ValueError("Oversized known publication target")
        return content
    except FileNotFoundError:
        return None
    except Exception as exc:
        raise DirectorPublicationError("AMBIGUOUS_PUBLICATION",root) from exc


def _publish_known(path: Path,value: Any,root: Path) -> None:
    intended=_json(value).encode("utf-8")
    before=_known_bytes(path,root)
    if before is not None:
        if before==intended:
            return
        raise DirectorPublicationError("AMBIGUOUS_PUBLICATION",root,expected_identities={path.name:hashlib.sha256(intended).hexdigest()})
    error=None
    try:
        _write(path,value)
    except Exception as exc:
        error=exc
    actual=_known_bytes(path,root)
    if actual==intended:
        return
    if actual is not None or error is None:
        raise DirectorPublicationError("AMBIGUOUS_PUBLICATION",root,expected_identities={path.name:hashlib.sha256(intended).hexdigest()})
    assert error is not None
    raise error


def _intended_record(path: Path,value: Any) -> dict[str,Any]:
    content=_json(value).encode()
    return {"path":str(path.resolve()),"bytes":len(content),"sha256":hashlib.sha256(content).hexdigest()}


def _file_record(path: Path) -> dict[str,Any]:
    if not 0 <= path.stat().st_size <= MAX_ARTIFACT_BYTES:
        raise ValueError("Host artifact byte bound exceeded")
    return {"path":str(path.resolve()),"sha256":file_sha256(path),"bytes":path.stat().st_size}


def _upstream(source: Path,timeline: SourceVisualTimeline,path: Path | None,
              context: ObservedEvidenceContext) -> dict[str,Any]:
    if path is None:
        return {"path":None,"sha256":None,"bytes":None,"context":{
            "status":"unverified_legacy","reason":"object_only_no_path",
            "timeline_fingerprint":visual_timeline_fingerprint(timeline),
            "source_fingerprint":timeline.source_fingerprint}}
    pinned=path.resolve(strict=True)
    current=SourceVisualTimeline.model_validate(read_generation_json(pinned))
    if canonical(current.model_dump(mode="json")) != canonical(timeline.model_dump(mode="json")):
        raise ValueError("Pinned timeline differs from supplied detached object")
    upstream=load_visual_provenance(pinned,source_video=source,
        expected_timeline_fingerprint=visual_timeline_fingerprint(timeline),current_origin=context.origin)
    return {**_file_record(pinned),"context":upstream.to_record()}


def _runtime_record(context: ObservedEvidenceContext) -> dict[str,Any]:
    return {k:str(v) if isinstance(v,Path) else v for k,v in asdict(context.runtime).items()}


def _context_from_authority(authority: dict[str,Any]) -> ObservedEvidenceContext:
    origin=authority["host_context"]["origin"]
    streams=[]
    for row in origin["streams"]:
        data=dict(row)
        for name in ("time_base","start","end"):
            if name in data:
                data[name]=exact(data[name])
        streams.append(StreamClock(**data))
    current=SourceTimeOrigin(origin["path"],origin["size"],origin["mtime_ns"],origin["fingerprint"],
        exact(origin["origin"]),exact(origin["duration"]),streams,origin["decoder_id"],origin["color"])
    runtime=dict(authority["runtime"])
    runtime["ffmpeg"],runtime["ffprobe"]=Path(runtime["ffmpeg"]),Path(runtime["ffprobe"])
    return ObservedEvidenceContext(RenderRuntime(**runtime),current)


def _check_dependencies(authority: dict[str,Any], records: list[dict[str,Any]],
                        context: ObservedEvidenceContext, library: StyleLibrary) -> list[dict[str,Any]]:
    source=Path(authority["source"])
    if "run_path" in authority and _known_bytes(Path(authority["run_path"])/"authority.json",Path(authority["run_path"]))!=_json(authority).encode():
        raise ValueError("Captured Director authority changed")
    if source_fingerprint(source)!=authority["fingerprints"]["source_fingerprint"]:
        raise ValueError("source changed during Director run")
    if observed_evidence.probe_source(source,context.runtime).identity!=context.origin.identity:
        raise ValueError("source origin changed during Director run")
    timeline=SourceVisualTimeline.model_validate(authority["timeline"])
    host=authority["host_context"]
    if "sfx" in host:
        validate_sfx_authority(host["sfx"])
    upstream=host["upstream"]
    if _upstream(source,timeline,Path(upstream["path"]) if upstream["path"] else None,context)!=upstream:
        raise ValueError("Pinned upstream changed during Director run")
    text=host["text"]
    if text["path"] is not None and _file_record(Path(text["path"]))!={k:text[k] for k in ("path","sha256","bytes")}:
        raise ValueError("Consumed transcript changed during Director run")
    if library.fingerprint!=authority["fingerprints"]["style_library_fingerprint"]:
        raise ValueError("style library changed during Director run")
    for row in authority.get("style_images",[]):
        if _file_record(Path(row["path"]))!=row:
            raise ValueError("Style image changed during Director run")
    return [dict(observed_evidence.load_visual_evidence_bundle(Path(row["path"]),source_video=source,
        timeline=timeline,windows=[TimeRange.model_validate(w) for w in row["windows"]],context=context,
        max_images=row["max_images"],request=VisualEvidenceRequest.model_validate(row["request"]) if row["request"] else None,
        expected_bundle_sha256=row["sha256"]),bundle_path=row["path"]) for row in records]


def _captured_style_rows(authority: dict[str,Any]) -> list[dict[str,Any]]:
    selected=authority["selected_styles"]
    images=authority["style_images"]
    if type(selected) is not list or type(images) is not list or len(selected)!=len(images) or len(selected)>3:
        raise ValueError("Invalid captured selected style authority")
    rows=[]
    cards=set()
    library=Path(authority["style_library_root"])
    for selected_style,image in zip(selected,images):
        _record_fields(selected_style,{"card","image"},"selected style")
        card=StyleCard.model_validate(selected_style["card"])
        _record_file(image)
        if (canonical(selected_style["card"])!=canonical(card.model_dump(mode="json"))
                or selected_style["image"]!=image or card.card_id in cards
                or not Path(image["path"]).is_relative_to(library)):
            raise ValueError("Captured selected card/image association mismatch")
        cards.add(card.card_id)
        rows.append({"kind":"style","card_id":card.card_id,"path":image["path"],"approval":card.approval,
                     "at":card.keyframes[0].timestamp,"sha256":image["sha256"]})
    return rows


def _expected_presented_manifest(authority: dict[str,Any],bundles: list[dict[str,Any]]) -> list[dict[str,Any]]:
    limits=VisualDirectorLimits(**authority["limits"])
    sources=_source_manifest(bundles,limits)
    if not sources:
        raise ValueError("A nonzero call requires a presented source image")
    for row in sources:
        row["sha256"]=row["jpeg"]["sha256"]
        row["upstream_context_sha256"]=_hash(authority["host_context"]["upstream"])
    rows=sources+_captured_style_rows(authority)
    for index,row in enumerate(rows):
        row["image_index"]=index
    return rows


def _check_calls(authority: dict[str,Any],records: list[dict[str,Any]],bundles: list[dict[str,Any]],
                 calls: list[dict[str,Any]],*,check_files: bool = True) -> None:
    """The same captured-prefix join runs live, on READY, and without I/O on REVIEW."""
    limits=VisualDirectorLimits(**authority["limits"])
    if len(calls)>limits.max_calls_per_clip or len(records)!=len(bundles):
        raise ValueError("Presented call/bundle budget mismatch")
    segment_counts=[0 for _ in authority["editorial_clip"]["segments"]]
    previous=0
    root=Path(authority["run_path"])
    for index,call in enumerate(calls,1):
        _record_fields(call,{"call","path","bytes","sha256","presented_assets","bundle_count","segment_indices"},"call record")
        _record_integer(call["call"],index,index)
        _record_file({key:call[key] for key in ("path","bytes","sha256")})
        if call["path"]!=str(root/f"call-{index:02d}"/"images.json") or type(call["presented_assets"]) is not list:
            raise ValueError("Captured call namespace/assets mismatch")
        prefix=call["bundle_count"]
        _record_integer(prefix,max(1,previous),len(bundles))
        if index==1 and prefix!=1:
            raise ValueError("Initial call cannot claim later bundles")
        previous=prefix
        segments=call["segment_indices"]
        if type(segments) is not list or not segments:
            raise ValueError("Missing captured call segment charges")
        for segment in segments:
            _record_integer(segment,0,len(segment_counts)-1)
            segment_counts[segment]+=1
            if segment_counts[segment]>1+limits.max_followups_per_segment:
                raise ValueError("Captured per-segment call budget exceeded")
        if segments!=sorted(set(segments)) or (index==1 and segments!=list(range(len(segment_counts)))):
            raise ValueError("Invalid captured call segment charges")
        expected=_expected_presented_manifest(authority,bundles[:prefix])
        if canonical(call["presented_assets"])!=canonical(expected):
            raise ValueError("Presented rows contradict qualified captured source/style authority")
        encoded=_json(call["presented_assets"]).encode()
        if call["bytes"]!=len(encoded) or call["sha256"]!=hashlib.sha256(encoded).hexdigest():
            raise ValueError("Captured manifest identity mismatch")
        if check_files:
            if _known_bytes(Path(call["path"]),root)!=encoded:
                raise ValueError("Presented image manifest changed")
            for row in call["presented_assets"]:
                if file_sha256(Path(row["path"]))!=row["sha256"]:
                    raise ValueError("Presented image changed")


def _bundle_record(path: Path,bundle: dict[str,Any],max_images: int) -> dict[str,Any]:
    return {**_file_record(path),"windows":bundle["windows"],"request":bundle["request"],
            "max_images":max_images,"captured":json.loads(canonical(bundle))}


def _record_fields(value: Any, fields: set[str], label: str) -> None:
    if type(value) is not dict or set(value)!=fields:
        raise ValueError(f"Invalid captured {label} fields")


def _record_integer(value: Any, minimum: int, maximum: int) -> None:
    if type(value) is not int or not minimum<=value<=maximum:
        raise ValueError("Invalid captured integer/budget")


def _record_text(value: Any) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError("Invalid captured text")


def _record_file(value: Any) -> None:
    _record_fields(value,{"path","bytes","sha256"},"file")
    _record_text(value["path"])
    path=Path(value["path"])
    if not path.is_absolute() or str(path)!=value["path"] or ".." in path.parts:
        raise ValueError("Invalid captured artifact path")
    _record_integer(value["bytes"],1,MAX_ARTIFACT_BYTES)
    sha=value["sha256"]
    if type(sha) is not str or len(sha)!=64 or any(c not in "0123456789abcdef" for c in sha):
        raise ValueError("Invalid captured SHA256")


def _snapshot_structure(record: dict[str,Any],origin: SourceTimeOrigin) -> None:
    """Validate captured JSON relationships only; never assert current receipt/media validity."""
    snapshot=record["captured"]
    root=Path(record["path"]).parent/Path(record["path"]).stem
    catalog: dict[str,dict[str,Any]]={}
    for row in snapshot["catalog"]:
        _record_fields(row,{"path","bytes","sha256"},"catalog row")
        relative=row["path"]
        if type(relative) is not str or not relative or any(c in relative for c in ("/","\\",":")) or relative in {".",".."} or relative in catalog:
            raise ValueError("Invalid captured artifact name")
        _record_file({**row,"path":str(root/relative)})
        catalog[relative]=row
    observations=set()
    used=set()
    for index,row in enumerate(snapshot["observations"]):
        _record_fields(row,{"path","bytes","sha256","observation_id"},"observation row")
        _record_file({key:row[key] for key in ("path","bytes","sha256")}|{"path":str(root/row["path"])})
        if row["path"]!=f"observation-{index:02d}.json" or {k:row[k] for k in ("path","bytes","sha256")}!=catalog.get(row["path"]):
            raise ValueError("Captured receipt/catalog contradiction")
        _record_file({"path":str(root/row["path"]),"bytes":1,"sha256":row["observation_id"]})
        observations.add(row["path"])
        used.add(row["path"])
    points=snapshot["selected_requests"]
    if not 1<=len(points)<=record["max_images"] or len(snapshot["outcomes"])!=len(points):
        raise ValueError("Captured selection/outcome count mismatch")
    selected={}
    for point in points:
        _record_fields(point,{"requested","window_indices"},"selected request")
        if type(point["requested"]) is not str or point["requested"] in selected:
            raise ValueError("Invalid captured requested coordinate")
        at=exact(point["requested"])
        indices=[i for i,w in enumerate(record["windows"]) if Fraction(str(w["start"]))<=at<Fraction(str(w["end"]))]
        if not indices or canonical(point["window_indices"])!=canonical(indices):
            raise ValueError("Captured request/window locality mismatch")
        selected[point["requested"]]=point
    video=next(stream for stream in origin.streams if stream.kind=="video")
    references: dict[tuple[Any,...],str]={}
    frame_ids=set()
    for frame in snapshot["frames"]:
        _record_fields(frame,{"evidence_id","at","requested_at","path","width","height","frame_id","raw_bgr_sha256",
            "raw_pts","time_base","observed_source_start","observed_source_end","jpeg","sample_refs"},"frame metadata")
        for key in ("frame_id","raw_bgr_sha256"):
            _record_file({"path":record["path"],"bytes":1,"sha256":frame[key]})
        if (frame["frame_id"] in frame_ids or type(frame["raw_pts"]) is not int
                or type(frame["width"]) is not int or type(frame["height"]) is not int
                or (frame["width"],frame["height"])!=(video.width,video.height)):
            raise ValueError("Invalid captured raw frame identity/dimensions")
        frame_ids.add(frame["frame_id"])
        if any(type(frame[key]) is not str for key in ("time_base","observed_source_start","observed_source_end")):
            raise ValueError("Invalid captured exact frame times")
        start,end=exact(frame["observed_source_start"]),exact(frame["observed_source_end"])
        if exact(frame["time_base"])<=0 or start!=frame["raw_pts"]*exact(frame["time_base"])-origin.origin or end<=start:
            raise ValueError("Captured raw/source interval contradiction")
        jpeg=frame["jpeg"]
        _record_fields(jpeg,{"relative_path","sha256","bytes","quality","resize","opencv_version"},"JPEG metadata")
        relative=f"frame-{frame['frame_id']}.jpg"
        _record_file({"path":str(root/relative),"bytes":jpeg["bytes"],"sha256":jpeg["sha256"]})
        _record_integer(jpeg["quality"],85,85)
        _record_text(jpeg["opencv_version"])
        if (jpeg["resize"]!="none" or jpeg["relative_path"]!=relative
                or {"path":relative,"sha256":jpeg["sha256"],"bytes":jpeg["bytes"]}!=catalog.get(relative)
                or frame["path"]!=str(root/relative)
                or frame["evidence_id"]!=f"{Path(record['path']).stem}-frame-{frame['frame_id'][:16]}"):
            raise ValueError("Captured JPEG/frame/catalog contradiction")
        used.add(relative)
        if type(frame["sample_refs"]) is not list or not frame["sample_refs"]:
            raise ValueError("Missing captured frame references")
        for ref in frame["sample_refs"]:
            _record_fields(ref,{"requested","window_indices","receipt","frame_n"},"sample reference")
            point=selected.get(ref["requested"])
            _record_integer(ref["frame_n"],0,1199)
            reference_key=(ref["requested"],ref["receipt"],ref["frame_n"])
            if (point is None or canonical(point["window_indices"])!=canonical(ref["window_indices"])
                    or ref["receipt"] not in observations or reference_key in references
                    or not start<=exact(ref["requested"])<end):
                raise ValueError("Captured frame reference contradiction")
            references[reference_key]=frame["frame_id"]
        representative=float(exact(frame["sample_refs"][0]["requested"]))
        if canonical([frame["at"],frame["requested_at"]])!=canonical([representative,representative]):
            raise ValueError("Captured presentation sample changed")
    covered=set()
    for point,outcome in zip(points,snapshot["outcomes"]):
        _record_fields(outcome,{"requested","window_indices","receipt","frame_n","status","frame_id"},"sample outcome")
        if (outcome["requested"]!=point["requested"] or canonical(outcome["window_indices"])!=canonical(point["window_indices"])
                or outcome["receipt"] not in observations):
            raise ValueError("Captured outcome/request contradiction")
        if outcome["status"]=="covered":
            _record_integer(outcome["frame_n"],0,1199)
            reference_key=(outcome["requested"],outcome["receipt"],outcome["frame_n"])
            if references.get(reference_key)!=outcome["frame_id"]:
                raise ValueError("Captured outcome/frame reference contradiction")
            covered.add(reference_key)
        elif (outcome["status"] not in {"before_video_start","missing_predecessor","unknown_terminal"}
                or outcome["frame_n"] is not None or outcome["frame_id"] is not None):
            raise ValueError("Invalid captured unavailable outcome")
    if covered!=set(references) or used!=set(catalog):
        raise ValueError("Unassociated captured frame/receipt/catalog metadata")


def _captured_structure(root: Path,receipt: dict[str,Any],companion: dict[str,Any]) -> dict[str,Any] | None:
    """Internal captured facts only. REVIEW never reopens drifted dependencies."""
    ready=receipt.get("status")=="ready"
    receipt_fields={"version","kind","run_id","status","source","source_fingerprint","clip_id",
                    "calls","host_context","companion_path","companion_sha256"}
    companion_fields={"version","kind","run_id","status","source","source_fingerprint","clip_id",
                      "host_context","authority","bundles","calls"}
    _record_fields(receipt,receipt_fields|({"plan_path","plan_sha256"} if ready else {"reason"}),"receipt")
    _record_fields(companion,companion_fields|({"plan"} if ready else {"failed_check","validation"}),"companion")
    _record_text(receipt["source"])
    _record_text(receipt["clip_id"])
    source=Path(receipt["source"])
    if not source.is_absolute() or str(source)!=receipt["source"]:
        raise ValueError("Invalid captured source path")
    for key in ("run_id","status","source","source_fingerprint","clip_id","host_context"):
        if canonical(companion[key])!=canonical(receipt[key]):
            raise ValueError("Captured terminal facts contradict companion")
    _record_integer(receipt["calls"],0,6)
    for key in ("calls","bundles"):
        if type(companion[key]) is not list:
            raise ValueError("Invalid captured fact list")
    if len(companion["calls"])!=receipt["calls"]:
        raise ValueError("Captured call count mismatch")
    if not ready:
        _record_text(receipt["reason"])
        if companion["failed_check"]!=receipt["reason"] or companion["validation"]!="last_validated_facts":
            raise ValueError("Invalid REVIEW last-validated semantics")
    authority_bytes=_known_bytes(root/"authority.json",root)
    if companion["authority"] is None:
        if (ready or authority_bytes is not None or companion["host_context"] is not None
                or receipt["calls"]!=0 or companion["calls"] or companion["bundles"]):
            raise ValueError("Mixed or incomplete initialization REVIEW")
        fingerprint=receipt["source_fingerprint"]
        if (type(fingerprint) is not str or len(fingerprint)!=16
                or any(c not in "0123456789abcdef" for c in fingerprint)
                or root.parent.name!=safe_slug(receipt["clip_id"])
                or root.parents[3].name!=f"{safe_slug(source.stem)}-{fingerprint}"):
            raise ValueError("Initialization REVIEW source/run namespace mismatch")
        return None
    _record_file(companion["authority"])
    if authority_bytes is None or companion["authority"]!=_file_record(root/"authority.json"):
        raise ValueError("Captured authority identity mismatch")
    authority=read_generation_json(root/"authority.json")
    if (authority["run_id"]!=root.name or authority["run_path"]!=str(root) or authority["source"]!=receipt["source"]
            or authority["host_context"]!=companion["host_context"]
            or authority["fingerprints"]["source_fingerprint"]!=receipt["source_fingerprint"]
            or authority["editorial_clip"]["clip_id"]!=receipt["clip_id"]):
        raise ValueError("Captured authority/terminal association mismatch")
    limits=VisualDirectorLimits(**authority["limits"])
    _record_integer(receipt["calls"],0,limits.max_calls_per_clip)
    timeline=SourceVisualTimeline.model_validate(authority["timeline"])
    clip=ClipPlan.model_validate(authority["editorial_clip"])
    context=_context_from_authority(authority)
    if (context.origin.path!=receipt["source"] or context.origin.fingerprint!=receipt["source_fingerprint"]
            or context.origin.identity!=authority["host_context"]["origin_id"]
            or timeline.source_fingerprint!=receipt["source_fingerprint"]
            or visual_timeline_fingerprint(timeline)!=authority["fingerprints"]["source_timeline_fingerprint"]
            or _hash(clip.model_dump(mode="json"))!=authority["fingerprints"]["composition_fingerprint"]):
        raise ValueError("Captured context/clip/timeline contradiction")
    if canonical(authority["fingerprints"])!=canonical(_fingerprints(clip,timeline,authority["fingerprints"]["style_library_fingerprint"])):
        raise ValueError("Captured fingerprint associations changed")
    host=authority["host_context"]
    _record_fields(host,{"origin","origin_id","upstream","text"}
                   | ({"sfx"} if type(host) is dict and "sfx" in host else set())
                   | ({"creator_overlays"} if type(host) is dict and "creator_overlays" in host else set()),"host context")
    _creator_mode(authority)
    if "sfx" in host:
        validate_sfx_authority(host["sfx"], check_files=False)
    upstream=host["upstream"]
    _record_fields(upstream,{"path","bytes","sha256","context"},"upstream")
    upstream_context=upstream["context"]
    if (type(upstream_context) is not dict or upstream_context.get("status") not in {"unverified_legacy","observed_samples"}
            or upstream_context.get("source_fingerprint")!=timeline.source_fingerprint
            or upstream_context.get("timeline_fingerprint")!=visual_timeline_fingerprint(timeline)):
        raise ValueError("Captured upstream context contradiction")
    if upstream["path"] is None:
        if (upstream["bytes"] is not None or upstream["sha256"] is not None
                or upstream_context.get("status")!="unverified_legacy"
                or upstream_context.get("reason")!="object_only_no_path"):
            raise ValueError("Invalid object-only captured upstream")
    else:
        _record_file({key:upstream[key] for key in ("path","bytes","sha256")})
    text=host["text"]
    _record_fields(text,{"status","path","bytes","sha256"},"text context")
    if text["status"]=="missing":
        if any(text[key] is not None for key in ("path","bytes","sha256")):
            raise ValueError("Missing text claims a captured artifact")
    elif text["status"]=="unverified_legacy":
        _record_file({key:text[key] for key in ("path","bytes","sha256")})
    else:
        raise ValueError("Invalid captured text status")
    if len(companion["bundles"])>1+3*(limits.max_calls_per_clip-1):
        raise ValueError("Captured bundle budget exceeded")
    paths=set()
    for index,row in enumerate(companion["bundles"]):
        _record_fields(row,{"path","bytes","sha256","windows","request","max_images","captured"},"bundle record")
        _record_file({key:row[key] for key in ("path","bytes","sha256")})
        path=Path(row["path"])
        if (not path.is_relative_to(root/"evidence") or len(path.relative_to(root/"evidence").parts)!=1
                or path.suffix!=".json" or row["path"] in paths):
            raise ValueError("Invalid captured bundle namespace")
        paths.add(row["path"])
        _record_integer(row["max_images"],1,limits.max_source_images_per_call)
        if type(row["windows"]) is not list or not row["windows"]:
            raise ValueError("Missing captured bundle windows")
        windows=[TimeRange.model_validate(w) for w in row["windows"]]
        if canonical(row["windows"])!=canonical([w.model_dump(mode="json") for w in windows]):
            raise ValueError("Captured window types changed")
        if any(w.end>timeline.duration_seconds for w in windows):
            raise ValueError("Captured bundle windows exceed timeline")
        if index==0 and (row["request"] is not None or path.name!="initial.json"):
            raise ValueError("Missing captured initial bundle")
        if row["request"] is not None:
            request=VisualEvidenceRequest.model_validate(row["request"])
            if (request.clip_id!=clip.clip_id or canonical(row["request"])!=canonical(request.model_dump(mode="json"))
                    or canonical(row["windows"])!=canonical([request.source.model_dump(mode="json")])):
                raise ValueError("Captured request/window contradiction")
        captured=row["captured"]
        _record_fields(captured,{"version","kind","source","source_fingerprint","origin","origin_id","timeline_fingerprint",
            "windows","max_images","request","selected_requests","outcomes","observations","frames","catalog"},"bundle snapshot")
        encoded=canonical(captured).encode()
        if len(encoded)!=row["bytes"] or hashlib.sha256(encoded).hexdigest()!=row["sha256"]:
            raise ValueError("Captured bundle snapshot identity mismatch")
        for key,value in {"version":1,"kind":"observed_director_images","source":receipt["source"],
            "source_fingerprint":receipt["source_fingerprint"],"origin":host["origin"],"origin_id":host["origin_id"],
            "timeline_fingerprint":visual_timeline_fingerprint(timeline),"windows":row["windows"],
            "max_images":row["max_images"],"request":row["request"]}.items():
            if canonical(captured[key])!=canonical(value):
                raise ValueError("Captured bundle/context association mismatch")
        for key in ("selected_requests","outcomes","observations","frames","catalog"):
            if type(captured[key]) is not list:
                raise ValueError("Invalid captured bundle metadata list")
        if (len(captured["frames"])>row["max_images"] or len(captured["observations"])>row["max_images"]
                or len(captured["catalog"])>2*row["max_images"]):
            raise ValueError("Captured image/receipt metadata budget exceeded")
        _snapshot_structure(row,context.origin)
    captured_bundles=[dict(row["captured"],bundle_path=row["path"]) for row in companion["bundles"]]
    _captured_style_rows(authority)
    _check_calls(authority,companion["bundles"],captured_bundles,companion["calls"],check_files=False)
    return authority


def _terminal_pair(root: Path,receipt: dict[str,Any],companion: dict[str,Any],*,
                   companion_sha256: str,allow_precommit_plan: bool = False) -> dict[str,Any] | None:
    """Shared full protocol checks; caller supplies the physical or intended-byte identity."""
    if (type(receipt.get("version")) is not int or receipt["version"]!=1
            or receipt.get("kind")!="visual_director_receipt" or receipt.get("run_id")!=root.name
            or receipt.get("status") not in {"ready","review_required"}
            or type(receipt.get("calls")) is not int or receipt["calls"]<0
            or receipt.get("companion_path")!=str(root/"provenance.json")):
        raise ValueError("Invalid terminal receipt")
    if (companion_sha256!=receipt["companion_sha256"]
            or type(companion.get("version")) is not int or companion["version"]!=1
            or companion.get("kind")!="visual_plan_provenance" or companion.get("run_id")!=root.name
            or companion.get("status")!=receipt["status"]
            or companion.get("source_fingerprint")!=receipt.get("source_fingerprint")
            or companion.get("host_context")!=receipt.get("host_context")):
        raise ValueError("Terminal companion association mismatch")
    authority=_captured_structure(root,receipt,companion)
    if receipt["status"]=="review_required" and not allow_precommit_plan and _known_bytes(root/"visual_plan.json",root) is not None:
        raise ValueError("Invalid terminal review plan association")
    return authority


def _load_terminal(root: Path,*,check_dependencies: bool = True) -> tuple[Path,dict[str,Any],dict[str,Any]] | None:
    encoded=_known_bytes(root/"receipt.json",root)
    if encoded is None:
        return None
    try:
        receipt=read_generation_json(root/"receipt.json")
        companion=read_generation_json(root/"provenance.json")
        authority=_terminal_pair(root,receipt,companion,companion_sha256=file_sha256(root/"provenance.json"))
        status=receipt["status"]
        if status=="ready":
            plan=root/"visual_plan.json"
            if receipt.get("plan_path")!=str(plan) or receipt.get("plan_sha256")!=file_sha256(plan):
                raise ValueError("Committed plan association mismatch")
            record=companion["plan"]
            if {k:record[k] for k in ("path","sha256","bytes")}!=_file_record(plan):
                raise ValueError("Companion plan identity mismatch")
            if _hash(read_generation_json(plan))!=record["fingerprint"]:
                raise ValueError("Companion canonical plan fingerprint mismatch")
            model=VisualCompositionPlan.model_validate(read_generation_json(plan))
            assert authority is not None
            validate_sfx_cues(model, authority["host_context"].get("sfx"),
                source_channels=source_audio_channels(authority["host_context"]["origin"]),
                check_files=check_dependencies)
            if (companion["authority"]!=_file_record(root/"authority.json")
                    or authority["run_id"]!=root.name or authority["host_context"]!=companion["host_context"]
                    or authority["source"]!=companion["source"]
                    or authority["fingerprints"]["source_fingerprint"]!=receipt["source_fingerprint"]
                    or authority["editorial_clip"]["clip_id"]!=receipt.get("clip_id")):
                raise ValueError("Committed authority association mismatch")
            if (model.clip_id!=receipt["clip_id"] or any(getattr(model,k)!=v for k,v in authority["fingerprints"].items())
                    or not companion["bundles"] or not companion["calls"] or len(companion["calls"])!=receipt["calls"]):
                raise ValueError("Committed plan/call authority mismatch")
            if check_dependencies:
                context=_context_from_authority(authority)
                bundles=_check_dependencies(authority,companion["bundles"],context,StyleLibrary(Path(authority["style_library_root"])))
                _check_calls(authority,companion["bundles"],bundles,companion["calls"])
            else:
                bundles=[dict(row["captured"],bundle_path=row["path"]) for row in companion["bundles"]]
            _check_final_image_citations(model,bundles,companion["calls"][-1]["presented_assets"])
            if _creator_mode(authority):
                text = authority["host_context"]["text"]
                _validate_creator_overlays(model, ClipPlan.model_validate(authority["editorial_clip"]),
                    _transcript(text, require_complete=True) if text["path"] is not None else None)
            return plan,receipt,companion
        return root/"receipt.json",receipt,companion
    except DirectorPublicationError:
        raise
    except Exception as exc:
        raise DirectorPublicationError("AMBIGUOUS_PUBLICATION",root) from exc


def _precommit_moves(root: Path,review_companion: bytes | None) -> list[tuple[str,str,bytes]]:
    """Inspect every known conflict before any move; later I/O failures stay partial/typed."""
    if _known_bytes(root/"receipt.json",root) is not None:
        raise DirectorPublicationError("AMBIGUOUS_PUBLICATION",root)
    moves=[]
    for final,candidate,retained in (("visual_plan.json","visual_plan.candidate.json","uncommitted-plan.json"),
        ("provenance.json","ready-provenance.candidate.json","uncommitted-ready-provenance.json")):
        content=_known_bytes(root/final,root)
        if content is None:
            continue
        if final=="provenance.json" and review_companion is not None and content==review_companion:
            continue
        expected=_known_bytes(root/candidate,root)
        if expected is None or content!=expected:
            raise DirectorPublicationError("AMBIGUOUS_PUBLICATION",root)
        saved=_known_bytes(root/retained,root)
        if saved is not None:
            # Never overwrite or delete a conflicting/uncommitted final.
            raise DirectorPublicationError("PRECOMMIT_PRESERVATION_FAILED",root)
        moves.append((final,retained,expected))
    return moves


def _preserve_precommit(root: Path,*,review_companion: bytes | None = None) -> None:
    for final,retained,expected in _precommit_moves(root,review_companion):
        try:
            (root/final).replace(root/retained)
        except Exception:
            if _known_bytes(root/retained,root)==expected and _known_bytes(root/final,root) is None:
                continue
            raise DirectorPublicationError("PRECOMMIT_PRESERVATION_FAILED",root) from None
        if _known_bytes(root/retained,root)!=expected or _known_bytes(root/final,root) is not None:
            raise DirectorPublicationError("PRECOMMIT_PRESERVATION_FAILED",root)


def _publish_review(root: Path,job: JobStore,stage: str,clip_id: str,reason: str,calls: int,
                    authority: dict[str,Any],records: list[dict[str,Any]],call_records: list[dict[str,Any]],
                    *, expected_authority: dict[str,Any] | None = None) -> Path:
    terminal=_load_terminal(root,check_dependencies=False)
    if terminal is not None:
        return terminal[0]
    _preserve_precommit(root)
    if _known_bytes(root/"visual_plan.json",root) is not None:
        raise DirectorPublicationError("PRECOMMIT_PRESERVATION_FAILED",root)
    captured_authority=_known_bytes(root/"authority.json",root)
    if expected_authority is not None and (captured_authority is None or _file_record(root/"authority.json")!=expected_authority):
        raise DirectorPublicationError("AMBIGUOUS_PUBLICATION",root)
    if captured_authority is not None and captured_authority!=_json(authority).encode():
        raise DirectorPublicationError("AMBIGUOUS_PUBLICATION",root)
    companion={"version":1,"kind":"visual_plan_provenance","run_id":root.name,"status":"review_required",
        "clip_id":clip_id,"host_context":authority.get("host_context") if (root/"authority.json").exists() else None,
        "source":str(job.source),"source_fingerprint":job.fingerprint,
        "authority":_file_record(root/"authority.json") if (root/"authority.json").exists() else None,
        "bundles":records,"calls":call_records,"failed_check":reason,"validation":"last_validated_facts"}
    receipt={"version":1,"kind":"visual_director_receipt","run_id":root.name,"status":"review_required",
        "reason":reason,"clip_id":clip_id,"source":str(job.source),"source_fingerprint":job.fingerprint,"calls":calls,
        "host_context":companion["host_context"],"companion_path":str(root/"provenance.json"),
        "companion_sha256":_intended_record(root/"provenance.json",companion)["sha256"]}
    try:
        _terminal_pair(root,receipt,companion,companion_sha256=_intended_record(root/"provenance.json",companion)["sha256"])
    except DirectorPublicationError:
        raise
    except Exception as exc:
        raise DirectorPublicationError("AMBIGUOUS_PUBLICATION",root) from exc
    try:
        _publish_known(root/"review-provenance.candidate.json",companion,root)
        _publish_known(root/"review-receipt.candidate.json",receipt,root)
        _publish_known(root/"provenance.json",companion,root)
        _publish_known(root/"receipt.json",receipt,root)
    except DirectorPublicationError:
        raise
    except Exception as exc:
        raise DirectorPublicationError("PRECOMMIT_PRESERVATION_FAILED",root) from exc
    committed=_load_terminal(root,check_dependencies=False)
    assert committed is not None
    try:
        job.mark_review_required(stage,reason=reason,receipt=root/"receipt.json")
    except Exception as exc:
        raise DirectorPublicationError("COMMITTED_BOOKKEEPING_FAILED",root,committed[0],receipt) from exc
    return committed[0]


def _check_ready_candidates(root: Path,authority: dict[str,Any]) -> None:
    """Inspect only named existing candidates, including after partial REVIEW publication."""
    companion_bytes=_known_bytes(root/"ready-provenance.candidate.json",root)
    receipt_bytes=_known_bytes(root/"ready-receipt.candidate.json",root)
    if companion_bytes is None:
        if receipt_bytes is not None:
            raise DirectorPublicationError("AMBIGUOUS_PUBLICATION",root)
        return
    try:
        companion=read_generation_json(root/"ready-provenance.candidate.json")
        plan=read_generation_json(root/"visual_plan.candidate.json")
        model=VisualCompositionPlan.model_validate(plan)
        validate_sfx_cues(model, authority["host_context"].get("sfx"),
            source_channels=source_audio_channels(authority["host_context"]["origin"]), check_files=False)
        row=companion["plan"]
        actual=_file_record(root/"visual_plan.candidate.json")
        if (companion.get("run_id")!=root.name or companion.get("status")!="ready"
                or companion.get("authority")!=_file_record(root/"authority.json")
                or companion.get("host_context")!=authority["host_context"]
                or companion.get("source")!=authority["source"] or row["path"]!=str(root/"visual_plan.json")
                or row["bytes"]!=actual["bytes"] or row["sha256"]!=actual["sha256"]
                or row["fingerprint"]!=_hash(model.model_dump(mode="json"))
                or any(getattr(model,k)!=v for k,v in authority["fingerprints"].items())):
            raise ValueError("Known ready candidate association mismatch")
        if receipt_bytes is not None:
            receipt=read_generation_json(root/"ready-receipt.candidate.json")
            if (receipt.get("run_id")!=root.name or receipt.get("status")!="ready"
                    or receipt.get("clip_id")!=model.clip_id or receipt.get("host_context")!=authority["host_context"]
                    or receipt.get("plan_path")!=str(root/"visual_plan.json") or receipt.get("plan_sha256")!=row["sha256"]
                    or receipt.get("companion_path")!=str(root/"provenance.json")
                    or receipt.get("companion_sha256")!=file_sha256(root/"ready-provenance.candidate.json")):
                raise ValueError("Known ready receipt candidate association mismatch")
    except DirectorPublicationError:
        raise
    except Exception as exc:
        raise DirectorPublicationError("AMBIGUOUS_PUBLICATION",root) from exc


def _recovery_namespace(source_video: Path,run_path: Path,jobs_root: Path | None) -> tuple[JobStore,Path,str]:
    """Locate ONLY the explicit existing captured namespace, without initializing a job."""
    try:
        root=run_path.resolve(strict=True)
        jobs=(jobs_root if jobs_root is not None else local_paths.jobs_dir).resolve(strict=True)
        relative=root.relative_to(jobs)
        if (len(relative.parts)!=5 or relative.parts[1:3]!=("planning","visual")
                or len(root.name)!=32 or any(c not in "0123456789abcdef" for c in root.name)):
            raise ValueError("Invalid explicit recovery run namespace")
        source=source_video.resolve()
        captured_authority=_known_bytes(root/"authority.json",root)
        locator=read_generation_json(root/("authority.json" if captured_authority is not None else "receipt.json"))
        fingerprint=locator["fingerprints"]["source_fingerprint"] if captured_authority is not None else locator["source_fingerprint"]
        clip_id=locator["editorial_clip"]["clip_id"] if captured_authority is not None else locator["clip_id"]
        _record_text(clip_id)
        if (type(fingerprint) is not str or len(fingerprint)!=16 or any(c not in "0123456789abcdef" for c in fingerprint)
                or locator["source"]!=str(source) or locator["run_id"]!=root.name
                or root.parent.name!=safe_slug(clip_id) or relative.parts[0]!=f"{safe_slug(source.stem)}-{fingerprint}"):
            raise ValueError("Captured recovery source/clip/run mismatch")
        job_root=jobs/relative.parts[0]
        if captured_authority is not None and (locator["job_root"]!=str(job_root) or locator["run_path"]!=str(root)):
            raise ValueError("Captured recovery authority namespace mismatch")
        job=JobStore(source=source,root=job_root,fingerprint=fingerprint)
        _recovery_state(job,root,clip_id)
        return job,root,clip_id
    except DirectorPublicationError:
        raise
    except Exception as exc:
        raise DirectorPublicationError("AMBIGUOUS_PUBLICATION",run_path.resolve()) from exc


def _recovery_state(job: JobStore,root: Path,clip_id: str) -> dict[str,Any]:
    state=read_generation_json(generation_file(job.root,"state.json"))
    if (type(state.get("version")) is not int or state["version"]!=1 or state.get("source")!=str(job.source)
            or state.get("fingerprint")!=job.fingerprint or type(state.get("stages")) is not dict):
        raise DirectorPublicationError("AMBIGUOUS_PUBLICATION",root)
    stage=state["stages"].get(f"visual_director:{clip_id}:{root.name}")
    if type(stage) is not dict or stage.get("status") not in {"running","complete","review_required","failed"}:
        raise DirectorPublicationError("AMBIGUOUS_PUBLICATION",root)
    if "artifacts" in stage:
        if type(stage["artifacts"]) is not list or any(type(path) is not str or ".." in Path(path).parts
            or not Path(path).is_relative_to(root) for path in stage["artifacts"]):
            raise DirectorPublicationError("AMBIGUOUS_PUBLICATION",root)
    return stage


def _review_state_matches(stage: dict[str,Any],receipt: dict[str,Any],root: Path) -> None:
    if stage["status"]=="complete" or (stage["status"]=="review_required" and (
            stage.get("reason")!=receipt["reason"] or stage.get("artifacts")!=[str(root/"receipt.json")])):
        raise DirectorPublicationError("AMBIGUOUS_PUBLICATION",root)


def recover_visual_director_publication(source_video: Path,run_path: Path,*,jobs_root: Path | None = None) -> Path:
    """Reconcile one explicitly named run; never promote a candidate to READY."""
    job,root,clip_id=_recovery_namespace(source_video,run_path,jobs_root)
    stage=f"visual_director:{clip_id}:{root.name}"
    with job.stage_lock(f"visual_director:{root.name}"):
        locked_job,locked_root,locked_clip=_recovery_namespace(source_video,root,jobs_root)
        if (locked_root!=root or locked_clip!=clip_id or locked_job.source!=job.source
                or locked_job.root!=job.root or locked_job.fingerprint!=job.fingerprint):
            raise DirectorPublicationError("AMBIGUOUS_PUBLICATION",root)
        owned_stage=_recovery_state(job,root,clip_id)
        if _known_bytes(root/"authority.json",root) is None:
            terminal=_load_terminal(root,check_dependencies=False)
            if (terminal is None or terminal[1]["status"]!="review_required"
                    or terminal[1]["source"]!=str(job.source) or terminal[1]["source_fingerprint"]!=job.fingerprint):
                raise DirectorPublicationError("AMBIGUOUS_PUBLICATION",root)
            path,receipt,_=terminal
            _review_state_matches(owned_stage,receipt,root)
            try:
                job.mark_review_required(stage,reason=receipt["reason"],receipt=path)
            except Exception as exc:
                raise DirectorPublicationError("COMMITTED_BOOKKEEPING_FAILED",root,path,receipt) from exc
            return path
        authority=read_generation_json(root/"authority.json")
        if authority["source"]!=str(job.source) or authority["job_root"]!=str(job.root) or authority["run_id"]!=root.name:
            raise ValueError("Recovery authority/source mismatch")
        terminal=_load_terminal(root)
        if terminal is not None:
            path,receipt,_=terminal
            if receipt["status"]=="review_required":
                _review_state_matches(owned_stage,receipt,root)
            try:
                if receipt["status"]=="ready":
                    job.mark_complete(stage,artifacts=[root/"receipt.json",path])
                else:
                    job.mark_review_required(stage,reason=receipt["reason"],receipt=path)
            except Exception as exc:
                raise DirectorPublicationError("COMMITTED_BOOKKEEPING_FAILED",root,path,receipt) from exc
            return path
        # Captured-namespace reconciliation bypasses current stats ONLY for final REVIEW.
        try:
            if source_fingerprint(job.source)!=job.fingerprint:
                raise ValueError("Uncommitted recovery source identity changed")
        except Exception as exc:
            raise DirectorPublicationError("AMBIGUOUS_PUBLICATION",root) from exc
        _check_ready_candidates(root,authority)
        review_candidate=root/"review-provenance.candidate.json"
        review_receipt=root/"review-receipt.candidate.json"
        companion_bytes=_known_bytes(review_candidate,root)
        receipt_bytes=_known_bytes(review_receipt,root)
        if companion_bytes is not None or receipt_bytes is not None:
            if companion_bytes is None or receipt_bytes is None:
                raise DirectorPublicationError("AMBIGUOUS_PUBLICATION",root)
            companion=read_generation_json(review_candidate)
            receipt=read_generation_json(review_receipt)
            try:
                intended_companion=_json(companion).encode()
                if (companion_bytes!=intended_companion or receipt_bytes!=_json(receipt).encode()
                        or receipt.get("status")!="review_required"):
                    raise ValueError("REVIEW candidate bytes differ from intended serialization")
                intended_sha256=hashlib.sha256(intended_companion).hexdigest()
                _terminal_pair(root,receipt,companion,companion_sha256=intended_sha256,allow_precommit_plan=True)
            except DirectorPublicationError:
                raise
            except Exception as exc:
                raise DirectorPublicationError("AMBIGUOUS_PUBLICATION",root) from exc
            # A failed REVIEW publication may already have its own companion.
            # All facts/known conflicts are checked before preserving any own finals.
            _preserve_precommit(root,review_companion=intended_companion)
            try:
                _terminal_pair(root,receipt,companion,companion_sha256=intended_sha256)
                if _known_bytes(root/"receipt.json",root) is not None:
                    raise ValueError("Terminal appeared before REVIEW publication")
            except DirectorPublicationError:
                raise
            except Exception as exc:
                raise DirectorPublicationError("AMBIGUOUS_PUBLICATION",root) from exc
            _publish_known(root/"provenance.json",companion,root)
            _publish_known(root/"receipt.json",receipt,root)
            terminal=_load_terminal(root,check_dependencies=False)
            assert terminal is not None
            try:
                job.mark_review_required(stage,reason=receipt["reason"],receipt=root/"receipt.json")
            except Exception as exc:
                raise DirectorPublicationError("COMMITTED_BOOKKEEPING_FAILED",root,terminal[0],receipt) from exc
            return terminal[0]
        # Only known candidates supply last validated facts; no model/source decode.
        candidate_path=root/"ready-provenance.candidate.json"
        candidate=read_generation_json(candidate_path) if candidate_path.exists() else {}
        if candidate and (candidate.get("run_id")!=root.name or candidate.get("authority")!=_file_record(root/"authority.json")):
            raise DirectorPublicationError("AMBIGUOUS_PUBLICATION",root)
        return _publish_review(root,job,stage,clip_id,"Explicit precommit recovery",len(candidate.get("calls",[])),
                               authority,candidate.get("bundles",[]),candidate.get("calls",[]))


def run_visual_director(
    source_video: Path, editorial_clip: ClipPlan, source_timeline: SourceVisualTimeline, *,
    client: StructuredClient, style_library: StyleLibrary,
    limits: VisualDirectorLimits = VisualDirectorLimits(), jobs_root: Path | None = None,
    source_timeline_path: Path | None = None, transcript_path: Path | None = None,
) -> Path:
    job=JobStore.for_video(source_video,jobs_root=jobs_root.resolve() if jobs_root is not None else None)
    root=job.new_visual_run(editorial_clip.clip_id)
    stage=f"visual_director:{editorial_clip.clip_id}:{root.name}"
    with job.stage_lock(f"visual_director:{root.name}"):
        return _run_visual_director(source_video,editorial_clip,source_timeline,client=client,
            style_library=style_library,limits=limits,jobs_root=jobs_root,
            source_timeline_path=source_timeline_path,transcript_path=transcript_path,
            job=job,root=root,stage=stage)


def _run_visual_director(
    source_video: Path,
    editorial_clip: ClipPlan,
    source_timeline: SourceVisualTimeline,
    *,
    client: StructuredClient,
    style_library: StyleLibrary,
    limits: VisualDirectorLimits = VisualDirectorLimits(),
    jobs_root: Path | None = None,
    source_timeline_path: Path | None = None,
    transcript_path: Path | None = None,
    job: JobStore,
    root: Path,
    stage: str,
) -> Path:
    """Return a ready v2 plan path, or an explicit per-run review receipt path.

    Requests/responses, image manifest/hash, all local evidence, usage and host
    authority are atomically recorded. Raw provider config/headers are never saved.
    Operational/validation failures become review receipts; BaseException cancels.
    """
    job.mark_started(stage)
    receipt_path = root / "receipt.json"
    usage: list[dict[str, Any]] = []
    calls = 0
    bundles: list[dict[str, Any]] = []
    records: list[dict[str,Any]]=[]
    call_records: list[dict[str,Any]]=[]
    authority: dict[str,Any]={}
    expected_authority: dict[str,Any] | None = None
    try:
        clip = ClipPlan.model_validate(editorial_clip.model_dump())
        timeline = SourceVisualTimeline.model_validate(source_timeline.model_dump())
        if job.fingerprint != timeline.source_fingerprint:
            raise ValueError("source fingerprint differs from source timeline")
        if any(s.source.end > timeline.duration_seconds for s in clip.segments):
            raise ValueError("Editorial source range exceeds source duration")
        required = _required_shots(clip, timeline)
        context=observed_evidence.prepare_visual_evidence_context(source_video)
        video=next(s for s in context.origin.streams if s.kind=="video")
        width, height = video.width,video.height
        if width <= 0 or height <= 0:
            raise ValueError("Actual source dimensions are unavailable")
        actual_duration = float(context.origin.duration)
        if not math.isfinite(actual_duration) or abs(actual_duration - timeline.duration_seconds) > 0.1:
            raise ValueError("Actual source duration differs from timeline authority")
        fingerprints = _fingerprints(clip, timeline, style_library.fingerprint)
        response_model = _response_model_for_fingerprints(fingerprints)
        text_path=(transcript_path.resolve(strict=True) if transcript_path is not None
                   else job.preferred_transcript_path)
        text={"status":"unverified_legacy",**_file_record(text_path)} if text_path.exists() else {
            "status":"missing","path":None,"sha256":None,"bytes":None}
        host_context={"origin":context.origin.record(),"origin_id":context.origin.identity,
            "upstream":_upstream(source_video,timeline,source_timeline_path,context),"text":text,
            "creator_overlays":creator_overlay_requirements(limits.require_creator_overlays)}
        captured_sfx = capture_sfx_authority()
        if captured_sfx is not None:
            host_context["sfx"] = captured_sfx
        authority={"source": str(job.source), "job_root": str(job.root),"run_id":root.name,"run_path":str(root),
            "fingerprints": fingerprints, "source_dimensions": {"width": width, "height": height},
            "limits": asdict(limits), "editorial_clip": clip.model_dump(mode="json"),
            "timeline":timeline.model_dump(mode="json"),"runtime":_runtime_record(context),
            "host_context":host_context,"style_library_root":str(style_library.root.resolve())}
        expanded = [TimeRange(start=max(0, s.source.start - 15),
                              end=min(timeline.duration_seconds, s.source.end + 15)) for s in clip.segments]
        _write(root / "evidence" / "source-context.json", {
            "source_fingerprint": job.fingerprint, "clip_id": clip.clip_id,
            "local_windows": [w.model_dump(mode="json") for w in expanded],
            "geometry_semantics": "original span aggregates; not timestamped OCR; host protection authority",
            "spans": [span.model_dump(mode="json") for span in timeline.spans
                      if any(_overlap(span.source, w) for w in expanded)],
        })
        cards, coverage = _select_styles(clip, timeline, style_library, limits)
        authority["style_images"]=[_file_record((style_library.root/card.keyframes[0].path).resolve(strict=True)) for card in cards]
        authority["selected_styles"]=[{"card":card.model_dump(mode="json"),"image":image}
            for card,image in zip(cards,authority["style_images"])]
        _write(root/"authority.json",authority)
        expected_authority=_file_record(root/"authority.json")
        _write(root / "styles.json", {"coverage": coverage,
                                      "cards": [c.model_dump(mode="json") for c in cards]})
        ranges = [s.source for s in clip.segments]
        path = collect_visual_evidence(source_video, ranges, timeline=timeline,
            output_path=root / "evidence" / "initial.json", max_images=limits.max_source_images_per_call,context=context)
        initial=read_generation_json(path)
        records.append(_bundle_record(path,initial,limits.max_source_images_per_call))
        bundles=_check_dependencies(authority,records,context,style_library)
        if not initial["frames"]:
            raise ValueError("No usable initial observed Director image")
        windows = [TimeRange(start=max(0, r.start - limits.transcript_padding_seconds),
                             end=min(timeline.duration_seconds, r.end + limits.transcript_padding_seconds))
                   for r in ranges]
        transcript = _transcript(text, require_complete=limits.require_creator_overlays)
        prior: set[VisualEvidenceSignature] = set()
        segment_calls = [0 for _ in clip.segments]
        active_segments = set(range(len(clip.segments)))
        for calls in range(1, limits.max_calls_per_clip + 1):
            if any(segment_calls[i] >= 1 + limits.max_followups_per_segment for i in active_segments):
                raise ValueError("Per-segment initial plus followup call limit reached")
            bundles=_check_dependencies(authority,records,context,style_library)
            _check_calls(authority,records,bundles,call_records)
            if any(frame["width"] != width or frame["height"] != height
                   for bundle in bundles for frame in bundle["frames"]):
                raise ValueError("Probed and decoded source dimensions disagree")
            manifest, images = _manifest(bundles, cards, style_library, limits)
            for item in manifest:
                if item["kind"]=="source":
                    item["upstream_context_sha256"]=_hash(host_context["upstream"])
            offered_sfx = sfx_offers(host_context.get("sfx"),
                                    source_channels=source_audio_channels(host_context["origin"]))
            offered_sfx_ids = {row["asset_id"] for row in offered_sfx}
            host = {"fingerprints": fingerprints, "source_dimensions": {"width": width, "height": height},
                    "required_shots": required, "authorized_assets": sorted(_ASSETS | offered_sfx_ids),
                    "local_source_image_ids_by_shot": [
                        {"source": dict(part), "image_evidence_ids": sorted(
                            _presented_local_ids(manifest, TimeRange.model_validate(part)))}
                        for part in required],
                    "supported_fills": ["cover", "contain_blur", "black"],
                    "supported_layers": ["source_video", "streamer_camera"], "opaque_layers_only": True,
                    "asset_kinds": {**dict.fromkeys(sorted(_FONTS), "font"),
                                    **dict.fromkeys(sorted(_IMAGES), "image"),
                                    **dict.fromkeys(sorted(offered_sfx_ids), "sfx")},
                    "authorized_sfx": offered_sfx, "style_coverage": coverage,"provenance":host_context,
                    "creator_overlays":host_context["creator_overlays"],
                    "evidence_bundles": [{k: v for k, v in b.items() if k in {"windows", "frames", "request"}}
                                         for b in bundles]}
            prompt = build_visual_director_prompt(clip, timeline, windows=windows,
                transcript=transcript, style_cards=cards, frame_manifest=manifest, host_context=host,
                safe_zones=common_vertical_safe_zones().model_dump(mode="json"),
                max_transcript_chars=limits.max_transcript_chars)
            call_dir = root / f"call-{calls:02d}"
            _write(call_dir / "request.json", {"system_prompt": VISUAL_DIRECTOR_SYSTEM_PROMPT,
                                               "user_payload": json.loads(prompt)})
            _write(call_dir / "images.json", manifest)
            call_records.append({"call":calls,**_file_record(call_dir/"images.json"),
                                 "presented_assets":manifest,"bundle_count":len(records),"segment_indices":sorted(active_segments)})
            _check_calls(authority,records,bundles,call_records)
            for index in active_segments:
                segment_calls[index] += 1
            try:
                result = client.generate(response_model,
                    system_prompt=VISUAL_DIRECTOR_SYSTEM_PROMPT, user_prompt=prompt,
                    images=images, temperature=0.1)
            except Exception as exc:
                # A provider's ValueError can contain its headers/key/config.
                _write(call_dir / "error.json", {"error_type": type(exc).__name__, "stage": "client"})
                raise RuntimeError("Structured client failed") from exc
            # Revalidate instances too: model_construct/mutation must not bypass
            # the strict recursive model boundary supplied by Task 1.
            _write(call_dir / "response.json", result.value.model_dump(mode="json"))
            usage.append({"call": calls, "usage": {k: v for k, v in result.usage.items()
                if k in _USAGE_FIELDS and isinstance(v, int) and not isinstance(v, bool) and v >= 0}})
            _write(root / "usage.json", usage)
            response = VisualDirectorResponse.model_validate(result.value.model_dump())
            validate_visual_director_context(response, clip_id=clip.clip_id,
                editorial_ranges=ranges, source_duration=timeline.duration_seconds,
                prior_request_signatures=prior)
            bundles=_check_dependencies(authority,records,context,style_library)
            _check_calls(authority,records,bundles,call_records)
            if set(response.style_card_ids) - {card.card_id for card in cards}:
                raise ValueError("Unknown or unoffered style card ID")
            if response.status == VisualDirectorStatus.READY:
                _validate_final(response, clip, timeline, fingerprints, cards, bundles, width, height,
                    presented_assets=manifest, host_sfx=host_context.get("sfx"),
                    audio_channels=source_audio_channels(host_context["origin"]),
                    require_creator_overlays=limits.require_creator_overlays,
                    transcript=transcript if text["path"] is not None else None)
                bundles=_check_dependencies(authority,records,context,style_library)
                assert response.plan is not None
                plan_path = root / "visual_plan.json"
                plan_value=response.plan.model_dump(mode="json")
                companion: dict[str,Any]={"version":1,"kind":"visual_plan_provenance","run_id":root.name,"status":"ready",
                    "clip_id":clip.clip_id,"host_context":host_context,"source":str(job.source),"source_fingerprint":job.fingerprint,
                    "authority":_file_record(root/"authority.json"),"bundles":records,"calls":call_records,
                    "plan":{**_intended_record(plan_path,plan_value),"fingerprint":_hash(plan_value)}}
                ready_receipt={"status": "ready", "plan_path": str(plan_path),
                    "source":str(job.source),
                    "version":1,"kind":"visual_director_receipt","run_id":root.name,
                    "plan_sha256":companion["plan"]["sha256"],"companion_path":str(root/"provenance.json"),
                    "companion_sha256":_intended_record(root/"provenance.json",companion)["sha256"],"host_context":host_context,
                    "clip_id": clip.clip_id, "source_fingerprint": job.fingerprint, "calls": calls}
                _publish_known(root/"visual_plan.candidate.json",plan_value,root)
                _publish_known(root/"ready-provenance.candidate.json",companion,root)
                _publish_known(root/"ready-receipt.candidate.json",ready_receipt,root)
                _publish_known(plan_path,plan_value,root)
                _publish_known(root/"provenance.json",companion,root)
                final_bundles=_check_dependencies(authority,records,context,style_library)
                _check_calls(authority,records,final_bundles,call_records)
                if (_known_bytes(plan_path,root)!=_json(plan_value).encode()
                        or _known_bytes(root/"provenance.json",root)!=_json(companion).encode()):
                    raise DirectorPublicationError("AMBIGUOUS_PUBLICATION",root)
                _publish_known(receipt_path,ready_receipt,root)
                _load_terminal(root)
                try:
                    job.mark_complete(stage, artifacts=[receipt_path, plan_path])
                except Exception as exc:
                    raise DirectorPublicationError("COMMITTED_BOOKKEEPING_FAILED",root,plan_path,ready_receipt) from exc
                return plan_path
            if response.status == VisualDirectorStatus.REVIEW_REQUIRED:
                raise ValueError("Director requested review")
            active_segments = set()
            for request in response.evidence_requests:
                candidates = [i for i, r in enumerate(ranges)
                              if request.source.start >= max(0, r.start - 15)
                              and request.source.end <= min(timeline.duration_seconds, r.end + 15)]
                # Prefer actual overlap, then nearest source midpoint. Ties charge
                # all equally applicable segments, avoiding a budget bypass.
                def distance(i: int) -> float:
                    return abs((ranges[i].start + ranges[i].end) / 2
                               - (request.source.start + request.source.end) / 2)
                nearest = min(distance(i) for i in candidates)
                active_segments.update(i for i in candidates if abs(distance(i) - nearest) <= _EPS)
            if calls == limits.max_calls_per_clip or any(
                    segment_calls[i] >= 1 + limits.max_followups_per_segment for i in active_segments):
                raise ValueError("Evidence iteration limit reached; unresolved response retained")
            for index, request in enumerate(response.evidence_requests):
                prior.add(visual_evidence_request_signature(request))
                path = collect_visual_evidence(source_video, [request.source], timeline=timeline,
                    output_path=root / "evidence" / f"followup-{calls:02d}-{index:02d}.json",
                    max_images=limits.max_source_images_per_call,context=context,request=request)
                bundle = read_generation_json(path)
                records.append(_bundle_record(path,bundle,limits.max_source_images_per_call))
                bundles=_check_dependencies(authority,records,context,style_library)
                windows.append(request.source)
        raise ValueError("Global visual director call limit reached")
    except DirectorPublicationError:
        raise
    except Exception as exc:
        # Never persist arbitrary exception messages: provider errors may contain
        # secrets/headers. Host ValueError messages below are controlled; external
        # client messages are represented by their type only.
        reason = str(exc) if type(exc) is ValueError else type(exc).__name__
        return _publish_review(root,job,stage,editorial_clip.clip_id,reason,len(call_records),authority,records,call_records,
                               expected_authority=expected_authority)
