from __future__ import annotations

from collections import UserDict, deque
from collections.abc import Callable, Iterable, Iterator
from math import nextafter
from typing import Any

import pytest
from pydantic import ValidationError

import clipperx.visual_contracts as contracts
from clipperx.planner_contracts import TimeRange
from clipperx.visual_contracts import (
    AudioCue,
    AudioMixPlan,
    CropKeyframe,
    FaceTrackSummary,
    FillMode,
    IdentityStatus,
    LayerKind,
    NormalizedRect,
    OverlayPlan,
    SourceState,
    SourceVisualSample,
    SourceVisualSpan,
    SourceVisualTimeline,
    SubtitlePlan,
    TemporalRectObservation,
    VisualCompositionPlan,
    VisualLayer,
    VisualShot,
    WebcamTrack,
)


FULL_FRAME = NormalizedRect(x=0, y=0, width=1, height=1)
LOWER_THIRD = NormalizedRect(x=0.1, y=0.7, width=0.8, height=0.2)


def test_serialized_director_schema_explains_required_source_ref() -> None:
    from clipperx.structured_client import _strict_json_schema
    from clipperx.visual_contracts import VisualDirectorResponse

    schema = _strict_json_schema(VisualDirectorResponse)
    layer = schema["$defs"]["VisualLayer"]
    field = layer["properties"]["source_ref"]
    assert field["type"] == "string"
    assert "source_ref" in layer["required"] and "default" not in field
    for term in ['source_ref="source"', "non-streamer-camera", "STREAMER_CAMERA alone",
                 "locally applicable", "confident identity", "own valid crop", "empty or null"]:
        assert term in field["description"]


def make_face_track(track_id: str = "face_main", **overrides: object) -> FaceTrackSummary:
    values: dict[str, object] = {
        "track_id": track_id,
        "first_seen": 0.0,
        "last_seen": 4.0,
        "observation_count": 2,
        "embedding_centroid": [0.1, 0.2],
        "positions": [FULL_FRAME],
        "recurrent_states": [SourceState.FULL_CAMERA],
        "full_camera_seconds": 2.0,
        "mean_confidence": 0.9,
        "evidence_ids": ["evidence_face"],
    }
    values.update(overrides)
    return FaceTrackSummary.model_validate(values)


def make_webcam_track(track_id: str = "webcam_main", **overrides: object) -> WebcamTrack:
    values: dict[str, object] = {
        "track_id": track_id,
        "source_ranges": [TimeRange(start=0, end=4)],
        "observations": [
            TemporalRectObservation(
                at=1,
                rect=NormalizedRect(x=0.7, y=0.05, width=0.25, height=0.25),
                confidence=0.9,
                evidence_id="evidence_webcam",
            )
        ],
        "face_track_ids": ["face_main"],
        "confidence": 0.9,
        "evidence_ids": ["evidence_webcam"],
    }
    values.update(overrides)
    return WebcamTrack.model_validate(values)


def make_sample(sample_id: str = "sample_1", **overrides: object) -> SourceVisualSample:
    values: dict[str, object] = {
        "sample_id": sample_id,
        "at": 1.0,
        "histogram_change_score": 0.1,
        "structural_change_score": 0.2,
        "text_density": 0.3,
        "face_track_ids": ["face_main"],
        "webcam_track_id": "webcam_main",
        "brightness": 0.4,
        "sharpness": 0.5,
        "motion": 0.6,
        "evidence_ids": ["evidence_sample"],
    }
    values.update(overrides)
    return SourceVisualSample.model_validate(values)


def make_span(span_id: str = "span_1", **overrides: object) -> SourceVisualSpan:
    values: dict[str, object] = {
        "span_id": span_id,
        "source": TimeRange(start=0, end=4),
        "state": SourceState.CONTENT_WITH_CAMERA,
        "confidence": 0.9,
        "streamer_track_id": "face_main",
        "identity_status": IdentityStatus.CONFIDENT,
        "webcam_track_id": "webcam_main",
        "important_regions": [FULL_FRAME],
        "text_regions": [],
        "evidence_ids": ["evidence_span"],
        "change_reasons": ["scene_change"],
    }
    values.update(overrides)
    return SourceVisualSpan.model_validate(values)


def make_timeline(**overrides: object) -> SourceVisualTimeline:
    values: dict[str, object] = {
        "source_fingerprint": "source_fp",
        "duration_seconds": 10.0,
        "analyzer_version": "visual-v1",
        "model_hashes": {"detector": "hash"},
        "sampling_config": {"interval_seconds": 1.0},
        "samples": [make_sample()],
        "spans": [make_span()],
        "face_tracks": [make_face_track()],
        "webcam_tracks": [make_webcam_track()],
    }
    values.update(overrides)
    return SourceVisualTimeline.model_validate(values)


def make_layer(layer_id: str = "content", **overrides: object) -> VisualLayer:
    values: dict[str, object] = {
        "layer_id": layer_id,
        "kind": LayerKind.SOURCE_VIDEO,
        "source_ref": "source",
        "source_crop": FULL_FRAME,
        "destination": FULL_FRAME,
        "fill": FillMode.COVER,
        "z_index": 0,
    }
    values.update(overrides)
    return VisualLayer.model_validate(values)


def make_shot(shot_id: str = "shot_1", **overrides: object) -> VisualShot:
    values: dict[str, object] = {
        "shot_id": shot_id,
        "source": TimeRange(start=10, end=12),
        "output": TimeRange(start=0, end=2),
        "primary_source": "content",
        "reason": "hook",
        "layers": [make_layer()],
        "evidence_ids": ["sample_1"],
        "confidence": 1.0,
    }
    values.update(overrides)
    return VisualShot.model_validate(values)


def make_plan(**overrides: object) -> VisualCompositionPlan:
    values: dict[str, object] = {
        "clip_id": "clip_1",
        "source_fingerprint": "source_fp",
        "composition_fingerprint": "composition_fp",
        "source_timeline_fingerprint": "timeline_fp",
        "identity_fingerprint": "identity_fp",
        "style_library_fingerprint": "style_fp",
        "shots": [make_shot()],
    }
    values.update(overrides)
    return VisualCompositionPlan.model_validate(values)


def make_overlay(**overrides: object) -> OverlayPlan:
    values: dict[str, object] = {
        "text": "Hook",
        "start": 0.0,
        "end": 1.0,
        "destination": LOWER_THIRD,
        "z_index": 2,
    }
    values.update(overrides)
    return OverlayPlan.model_validate(values)


def test_normalized_rect_rejects_out_of_frame_geometry() -> None:
    with pytest.raises(ValidationError, match="sale del frame"):
        NormalizedRect(x=0.8, y=0.1, width=0.3, height=0.4)


def test_source_span_keeps_ambiguous_identity_explicit() -> None:
    span = make_span(
        state=SourceState.MULTI_PERSON,
        streamer_track_id=None,
        identity_status=IdentityStatus.AMBIGUOUS,
    )

    assert span.streamer_track_id is None
    assert span.identity_status == IdentityStatus.AMBIGUOUS


def test_source_span_rejects_identity_status_track_mismatches() -> None:
    invalid_cases = (
        ("ambiguous_with_track", IdentityStatus.AMBIGUOUS, "face_main"),
        ("confident_without_track", IdentityStatus.CONFIDENT, None),
        ("unavailable_with_track", IdentityStatus.UNAVAILABLE, "face_main"),
    )
    accepted: list[str] = []
    for label, identity_status, track_id in invalid_cases:
        try:
            make_span(identity_status=identity_status, streamer_track_id=track_id)
        except ValidationError:
            continue
        accepted.append(label)

    assert accepted == [], f"SourceVisualSpan accepted invalid identity cases: {accepted}"


def test_validate_plan_references_uses_actual_timeline_track_ids() -> None:
    validator = getattr(contracts, "validate_plan_references", None)
    assert callable(validator), "visual_contracts.validate_plan_references must be public"

    timeline = make_timeline()
    valid_plan = make_plan(
        shots=[
            make_shot(
                layers=[
                    make_layer("source_layer", source_ref="source"),
                    make_layer("face_layer", source_ref="face_main"),
                    make_layer("webcam_layer", source_ref="webcam_main"),
                ]
            )
        ]
    )
    validator(valid_plan, timeline)

    unknown_plan = make_plan(shots=[make_shot(layers=[make_layer(source_ref="face_missing")])])
    with pytest.raises(ValueError, match="source_ref|desconocido"):
        validator(unknown_plan, timeline)


@pytest.mark.parametrize("id_scope", ["sample", "span", "face", "webcam"])
def test_timeline_rejects_duplicate_ids(id_scope: str) -> None:
    overrides: dict[str, object]
    if id_scope == "sample":
        overrides = {"samples": [make_sample(), make_sample(at=2.0)]}
    elif id_scope == "span":
        overrides = {
            "spans": [
                make_span(),
                make_span(source=TimeRange(start=4, end=6)),
            ]
        }
    elif id_scope == "face":
        overrides = {"face_tracks": [make_face_track(), make_face_track()]}
    else:
        overrides = {"webcam_tracks": [make_webcam_track(), make_webcam_track()]}

    with pytest.raises(ValidationError, match="único"):
        make_timeline(**overrides)


def test_timeline_rejects_non_increasing_sample_timestamps() -> None:
    with pytest.raises(ValidationError, match="crecientes"):
        make_timeline(
            samples=[
                make_sample("sample_1", at=2.0),
                make_sample("sample_2", at=1.0),
            ]
        )


@pytest.mark.parametrize(
    "spans",
    [
        [
            make_span("span_1", source=TimeRange(start=0, end=3)),
            make_span("span_2", source=TimeRange(start=2, end=4)),
        ],
        [
            make_span("span_1", source=TimeRange(start=4, end=6)),
            make_span("span_2", source=TimeRange(start=0, end=2)),
        ],
    ],
    ids=["overlap", "out_of_order"],
)
def test_timeline_rejects_overlapping_or_out_of_order_spans(
    spans: list[SourceVisualSpan],
) -> None:
    with pytest.raises(ValidationError, match="ordenados|superponerse"):
        make_timeline(spans=spans)


@pytest.mark.parametrize(
    "reference_case",
    ["sample_face", "sample_webcam", "span_face", "span_webcam", "webcam_face"],
)
def test_timeline_rejects_unknown_track_references(reference_case: str) -> None:
    overrides: dict[str, object]
    if reference_case == "sample_face":
        overrides = {"samples": [make_sample(face_track_ids=["face_missing"])]}
    elif reference_case == "sample_webcam":
        overrides = {"samples": [make_sample(webcam_track_id="webcam_missing")]}
    elif reference_case == "span_face":
        overrides = {"spans": [make_span(streamer_track_id="face_missing")]}
    elif reference_case == "span_webcam":
        overrides = {"spans": [make_span(webcam_track_id="webcam_missing")]}
    else:
        overrides = {"webcam_tracks": [make_webcam_track(face_track_ids=["face_missing"])]}

    with pytest.raises(ValidationError, match="desconocido"):
        make_timeline(**overrides)


@pytest.mark.parametrize(
    "duration_case",
    [
        "sample",
        "sample_window",
        "span",
        "face_timestamp",
        "face_duration",
        "webcam_range",
        "webcam_observation",
    ],
)
def test_timeline_rejects_timestamps_and_ranges_outside_duration(duration_case: str) -> None:
    overrides: dict[str, object]
    if duration_case == "sample":
        overrides = {"samples": [make_sample(at=10.1)]}
    elif duration_case == "sample_window":
        overrides = {"samples": [make_sample(audio_window=TimeRange(start=9, end=10.1))]}
    elif duration_case == "span":
        overrides = {"spans": [make_span(source=TimeRange(start=8, end=10.1))]}
    elif duration_case == "face_timestamp":
        overrides = {"face_tracks": [make_face_track(last_seen=10.1)]}
    elif duration_case == "face_duration":
        overrides = {"face_tracks": [make_face_track(full_camera_seconds=10.1)]}
    elif duration_case == "webcam_range":
        overrides = {
            "webcam_tracks": [
                make_webcam_track(source_ranges=[TimeRange(start=9, end=10.1)])
            ]
        }
    else:
        overrides = {
            "webcam_tracks": [
                make_webcam_track(
                    observations=[
                        TemporalRectObservation(
                            at=10.1,
                            rect=FULL_FRAME,
                            confidence=0.9,
                            evidence_id="evidence_webcam",
                        )
                    ]
                )
            ]
        }

    with pytest.raises(ValidationError, match="duration_seconds"):
        make_timeline(**overrides)


def test_visual_shot_allows_one_millisecond_duration_tolerance() -> None:
    make_shot(source=TimeRange(start=10, end=12.0005))

    with pytest.raises(ValidationError, match="duraciones"):
        make_shot(source=TimeRange(start=10, end=12.002))


@pytest.mark.parametrize(
    "shots",
    [
        [make_shot(output=TimeRange(start=0.1, end=2.1))],
        [
            make_shot(),
            make_shot(
                "shot_2",
                source=TimeRange(start=20, end=22),
                output=TimeRange(start=2.2, end=4.2),
            ),
        ],
        [
            make_shot(),
            make_shot(
                "shot_2",
                source=TimeRange(start=20, end=22),
                output=TimeRange(start=1.8, end=3.8),
            ),
        ],
    ],
    ids=["not_zero", "gap", "overlap"],
)
def test_visual_plan_rejects_invalid_output_timeline(shots: list[VisualShot]) -> None:
    with pytest.raises(ValidationError, match="iniciar|continuos"):
        make_plan(shots=shots)


def test_visual_plan_rejects_duplicate_shot_and_layer_ids() -> None:
    with pytest.raises(ValidationError, match="layer_id"):
        make_shot(layers=[make_layer(), make_layer()])

    with pytest.raises(ValidationError, match="shot_id"):
        make_plan(
            shots=[
                make_shot(),
                make_shot(
                    source=TimeRange(start=20, end=22),
                    output=TimeRange(start=2, end=4),
                ),
            ]
        )


@pytest.mark.parametrize(
    "crop_track",
    [
        [CropKeyframe(at=1, crop=FULL_FRAME), CropKeyframe(at=0, crop=FULL_FRAME)],
        [CropKeyframe(at=0, crop=FULL_FRAME), CropKeyframe(at=0, crop=FULL_FRAME)],
        [CropKeyframe(at=0.1, crop=FULL_FRAME), CropKeyframe(at=2, crop=FULL_FRAME)],
        [CropKeyframe(at=0, crop=FULL_FRAME), CropKeyframe(at=1.9, crop=FULL_FRAME)],
    ],
    ids=["out_of_order", "duplicate", "missing_start", "missing_end"],
)
def test_crop_track_rejects_invalid_order_or_endpoints(
    crop_track: list[CropKeyframe],
) -> None:
    with pytest.raises(ValidationError, match="crop_track"):
        make_shot(layers=[make_layer(crop_track=crop_track)])


@pytest.mark.parametrize(
    "overlay_values",
    [
        {"text": None, "asset_id": None},
        {"text": "Hook", "asset_id": "brand_logo"},
    ],
    ids=["neither", "both"],
)
def test_overlay_requires_exactly_one_text_or_asset(
    overlay_values: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="exactamente uno"):
        make_overlay(**overlay_values)


def test_overlay_times_are_relative_and_inside_shot() -> None:
    with pytest.raises(ValidationError, match="después"):
        make_overlay(start=1, end=1)

    with pytest.raises(ValidationError, match="duración"):
        make_shot(title=make_overlay(end=2.1))


@pytest.mark.parametrize("destination_kind", ["subtitles", "title", "brand"])
def test_text_destinations_cannot_intersect_forbidden_regions(
    destination_kind: str,
) -> None:
    overrides: dict[str, object] = {"forbidden_regions": [LOWER_THIRD]}
    if destination_kind == "subtitles":
        overrides["subtitles"] = SubtitlePlan(destination=LOWER_THIRD)
    else:
        overrides[destination_kind] = make_overlay()

    with pytest.raises(ValidationError, match="forbidden"):
        make_shot("shot_1", **overrides)


@pytest.mark.parametrize("destination_kind", ["subtitles", "title", "brand"])
@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize(
    ("forbidden_start", "overlaps"),
    [(0.82, False), (nextafter(0.82, 1.0), False), (nextafter(0.82, 0.0), True)],
    ids=["touching", "decimal-gap", "real-tiny-overlap"],
)
def test_decimal_boundary_overlay_intersection(
    destination_kind: str, axis: str, forbidden_start: float, overlaps: bool,
) -> None:
    # Actual Luna response: 0.48 + 0.34 was compared with 0.82 as a float.
    if overlaps:
        assert forbidden_start < 0.82
    if axis == "y":
        destination = NormalizedRect(x=0.1, y=0.48, width=0.72, height=0.34)
        forbidden = NormalizedRect(x=0, y=forbidden_start, width=1, height=0.18)
    else:
        destination = NormalizedRect(x=0.48, y=0.1, width=0.34, height=0.72)
        forbidden = NormalizedRect(x=forbidden_start, y=0, width=0.18, height=1)
    assert contracts._rectangles_intersect(destination, forbidden) is overlaps
    assert contracts._rectangles_intersect(forbidden, destination) is overlaps
    overrides: dict[str, object] = {"forbidden_regions": [forbidden]}
    if destination_kind == "subtitles":
        overrides["subtitles"] = SubtitlePlan(destination=destination)
        overrides["subtitle_allowed_regions"] = [destination]
    else:
        overrides[destination_kind] = make_overlay(destination=destination)
    if overlaps:
        with pytest.raises(ValidationError, match="forbidden"):
            make_shot(**overrides)
    else:
        make_shot(**overrides)


@pytest.mark.parametrize("audio_case", ["out_of_order", "out_of_bounds"])
def test_audio_cues_must_be_ordered_and_inside_shot(audio_case: str) -> None:
    if audio_case == "out_of_order":
        cues = [
            AudioCue(asset_id="cue_one", start=1, end=1.5),
            AudioCue(asset_id="cue_two", start=0.5, end=1),
        ]
    else:
        cues = [AudioCue(asset_id="cue_one", start=1.5, end=2.1)]

    with pytest.raises(ValidationError, match="audio"):
        make_shot(audio=AudioMixPlan(cues=cues))


def test_solid_color_and_first_shot_inherit_rules_are_strict() -> None:
    with pytest.raises(ValidationError, match="solid_color"):
        make_layer(kind=LayerKind.SOLID, fill=FillMode.SOLID, solid_color=None)

    with pytest.raises(ValidationError, match="solid_color"):
        make_layer(fill=FillMode.COVER, solid_color="#000000")

    with pytest.raises(ValidationError, match="inherit"):
        make_plan(shots=[make_shot(layers=[make_layer(fill=FillMode.INHERIT)])])


@pytest.mark.parametrize(
    "asset_kind",
    ["overlay_asset", "overlay_font", "subtitle_font", "audio_asset"],
)
def test_asset_font_and_audio_ids_reject_paths_and_uris(asset_kind: str) -> None:
    with pytest.raises(ValidationError, match="asset ID"):
        if asset_kind == "overlay_asset":
            make_overlay(text=None, asset_id="C:\\assets\\logo.png")
        elif asset_kind == "overlay_font":
            make_overlay(font_asset_id="../fonts/bold.ttf")
        elif asset_kind == "subtitle_font":
            SubtitlePlan(destination=LOWER_THIRD, font_asset_id="fonts/bold.ttf")
        else:
            AudioCue(asset_id="https://example.test/cue.wav", start=0, end=1)


def test_visual_plan_enforces_179_second_maximum() -> None:
    make_plan(
        shots=[
            make_shot(
                source=TimeRange(start=0, end=179),
                output=TimeRange(start=0, end=179),
            )
        ]
    )

    with pytest.raises(ValidationError, match="máximo"):
        make_plan(
            shots=[
                make_shot(
                    source=TimeRange(start=0, end=180),
                    output=TimeRange(start=0, end=180),
                )
            ]
        )


@pytest.mark.parametrize(
    "layer",
    [
        make_layer(destination=NormalizedRect(x=0, y=0, width=0.5, height=1)),
        make_layer(opacity=0),
    ],
    ids=["partial_destination", "transparent"],
)
def test_each_shot_requires_visible_full_frame_coverage(layer: VisualLayer) -> None:
    with pytest.raises(ValidationError, match="cubre"):
        make_plan(shots=[make_shot(layers=[layer])])


def test_unknown_layer_enum_values_are_rejected() -> None:
    with pytest.raises(ValidationError):
        make_layer(fill="stretch")

    with pytest.raises(ValidationError):
        make_layer(kind="unknown")


def test_validate_plan_references_rejects_source_fingerprint_mismatch() -> None:
    plan = make_plan(source_fingerprint="plan_fp")
    timeline = make_timeline(source_fingerprint="timeline_fp")

    with pytest.raises(
        ValueError,
        match=r"source_fingerprint.*plan_fp.*timeline_fp",
    ):
        contracts.validate_plan_references(plan, timeline)


def test_timeline_rejects_global_face_webcam_track_id_collision() -> None:
    shared_track_id = "shared_track"

    with pytest.raises(
        ValidationError,
        match=r"track_id.*face_tracks.*webcam_tracks.*shared_track",
    ):
        make_timeline(
            samples=[],
            spans=[],
            face_tracks=[make_face_track(shared_track_id)],
            webcam_tracks=[
                make_webcam_track(shared_track_id, face_track_ids=[shared_track_id])
            ],
        )


def test_crop_keyframes_reject_any_fraction_past_shot_duration() -> None:
    invalid_tracks = {
        "keyframe_after_shot": [CropKeyframe(at=2.0005, crop=FULL_FRAME)],
        "moving_endpoint_after_shot": [
            CropKeyframe(at=0, crop=FULL_FRAME),
            CropKeyframe(at=2.0005, crop=FULL_FRAME),
        ],
    }
    accepted: list[str] = []
    for label, crop_track in invalid_tracks.items():
        try:
            make_shot(layers=[make_layer(crop_track=crop_track)])
        except ValidationError:
            continue
        accepted.append(label)

    assert accepted == [], f"crop_track accepted hard-bound excesses: {accepted}"


def test_overlay_end_rejects_fraction_past_shot_duration() -> None:
    with pytest.raises(ValidationError, match="duración"):
        make_shot(title=make_overlay(end=2.0005))


def test_audio_cue_end_rejects_fraction_past_shot_duration() -> None:
    with pytest.raises(ValidationError, match="audio"):
        make_shot(
            audio=AudioMixPlan(
                cues=[AudioCue(asset_id="cue_one", start=1, end=2.0005)]
            )
        )


def test_visual_plan_rejects_fraction_past_maximum_duration() -> None:
    with pytest.raises(ValidationError, match="máximo"):
        make_plan(
            shots=[
                make_shot(
                    source=TimeRange(start=0, end=179.0005),
                    output=TimeRange(start=0, end=179.0005),
                )
            ]
        )


def test_preserve_region_must_fit_source_crop_or_full_source_frame() -> None:
    preserve_region = NormalizedRect(x=0.2, y=0.2, width=0.2, height=0.2)
    containing_crop = NormalizedRect(x=0.1, y=0.1, width=0.4, height=0.4)
    excluding_crop = NormalizedRect(x=0.25, y=0.25, width=0.4, height=0.4)

    make_layer(source_crop=None, preserve_regions=[preserve_region])
    make_layer(source_crop=containing_crop, preserve_regions=[preserve_region])

    with pytest.raises(ValidationError, match=r"preserve_regions.*source_crop"):
        make_layer(source_crop=excluding_crop, preserve_regions=[preserve_region])


def test_preserve_region_must_fit_every_crop_track_keyframe() -> None:
    preserve_region = NormalizedRect(x=0.2, y=0.2, width=0.2, height=0.2)
    containing_crop = NormalizedRect(x=0.1, y=0.1, width=0.4, height=0.4)
    excluding_crop = NormalizedRect(x=0.25, y=0.25, width=0.4, height=0.4)

    make_shot(
        layers=[
            make_layer(
                source_crop=None,
                crop_track=[
                    CropKeyframe(at=0, crop=FULL_FRAME),
                    CropKeyframe(at=2, crop=containing_crop),
                ],
                preserve_regions=[preserve_region],
            )
        ]
    )

    with pytest.raises(ValidationError, match=r"preserve_regions.*crop_track"):
        make_shot(
            layers=[
                make_layer(
                    source_crop=None,
                    crop_track=[
                        CropKeyframe(at=0, crop=FULL_FRAME),
                        CropKeyframe(at=2, crop=excluding_crop),
                    ],
                    preserve_regions=[preserve_region],
                )
            ]
        )


def test_subtitle_destination_must_fit_allowed_region_when_configured() -> None:
    allowed_region = NormalizedRect(x=0.1, y=0.65, width=0.8, height=0.3)
    inside_destination = NormalizedRect(x=0.2, y=0.7, width=0.6, height=0.2)
    outside_destination = NormalizedRect(x=0.1, y=0.1, width=0.8, height=0.2)

    make_shot(subtitles=SubtitlePlan(destination=outside_destination))
    make_shot(
        subtitle_allowed_regions=[allowed_region],
        subtitles=SubtitlePlan(destination=inside_destination),
    )

    with pytest.raises(
        ValidationError,
        match=r"subtitles.*subtitle_allowed_regions",
    ):
        make_shot(
            subtitle_allowed_regions=[allowed_region],
            subtitles=SubtitlePlan(destination=outside_destination),
        )


def make_visual_request(**overrides: object) -> contracts.VisualEvidenceRequest:
    values: dict[str, object] = {
        "request_id": "probe_1", "clip_id": "clip_1",
        "source": {"start": 5.0, "end": 35.0}, "question": "Where is the camera?",
        "sample_interval_seconds": 1.0, "reason": "camera_boundary", "iteration": 1,
    }
    values.update(overrides)
    return contracts.VisualEvidenceRequest.model_validate(values)


def make_director_response(**overrides: object) -> contracts.VisualDirectorResponse:
    values: dict[str, object] = {
        "status": "need_more_evidence", "evidence_requests": [make_visual_request()],
        "reasoning": "Camera boundary needs inspection.",
    }
    values.update(overrides)
    return contracts.VisualDirectorResponse.model_validate(values)


@pytest.mark.parametrize("overrides", [
    {"status": "ready", "plan": None, "evidence_requests": []},
    {"status": "ready", "plan": make_plan()},
    {"evidence_requests": []},
    {"plan": make_plan()},
    {"status": "review_required", "plan": make_plan()},
    {"evidence_requests": [make_plan()]},
])
def test_director_status_plan_request_coherence(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        make_director_response(**overrides)


def test_director_accepts_ready_probe_and_review_contracts() -> None:
    ready = make_director_response(status="ready", plan=make_plan(), evidence_requests=[])
    probing = make_director_response()
    review = make_director_response(status="review_required", evidence_requests=[])
    for response in (ready, probing, review):
        assert contracts.VisualDirectorResponse.model_validate_json(response.model_dump_json()) == response


def test_director_rejects_duplicate_request_ids_and_more_than_three_requests() -> None:
    with pytest.raises(ValidationError, match="request_id"):
        make_director_response(evidence_requests=[make_visual_request(), make_visual_request()])
    with pytest.raises(ValidationError):
        make_director_response(evidence_requests=[make_visual_request(request_id=str(i)) for i in range(4)])


@pytest.mark.parametrize(("field", "value"), [
    ("sample_interval_seconds", 0.2499), ("sample_interval_seconds", 2.0001),
    ("sample_interval_seconds", float("nan")), ("iteration", 0), ("iteration", 4),
    ("source", {"start": 0, "end": float("inf")}),
    ("source", {"start": 0, "end": 1, "source_path": "secret"}),
    ("reason", "guess"), ("unrequested", True),
])
def test_visual_request_is_strict_and_bounded(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        make_visual_request(**{field: value})


def test_director_rejects_nested_plan_extras_nonfinite_values_and_malformed_colors() -> None:
    for mutation in ("extra", "nonfinite", "color"):
        plan = make_plan().model_dump()
        if mutation == "extra":
            plan["shots"][0]["source"]["vod_url"] = "secret"
        elif mutation == "nonfinite":
            plan["shots"][0]["source"]["end"] = "inf"
        else:
            plan["shots"][0]["subtitles"] = SubtitlePlan(destination=LOWER_THIRD).model_dump()
            plan["shots"][0]["subtitles"]["primary_color"] = "red;movie=evil"
        with pytest.raises(ValidationError):
            make_director_response(status="ready", plan=plan, evidence_requests=[])
    with pytest.raises(ValidationError):
        make_director_response(approval="approved")


@pytest.mark.parametrize("container", [iter, deque], ids=["iterator", "deque"])
def test_director_rejects_unsupported_shot_containers_before_consuming_them(
    container: Callable[[list[dict[str, Any]]], Iterable[dict[str, Any]]],
) -> None:
    plan = make_plan().model_dump()
    shot = plan["shots"][0]
    shot["source"]["vod_url"] = "secret"
    shot["subtitles"] = SubtitlePlan(destination=LOWER_THIRD).model_dump()
    shot["subtitles"]["primary_color"] = "red;movie=evil"
    shots = container([shot])
    plan["shots"] = shots

    with pytest.raises(ValidationError, match="list or tuple"):
        make_director_response(status="ready", plan=plan, evidence_requests=[])

    assert next(iter(shots)) is shot


def test_director_rejects_nested_generator_without_running_its_body() -> None:
    plan = make_plan().model_dump()
    layer = plan["shots"][0]["layers"][0]
    layer["renderer_command"] = "unrequested"
    consumed = []

    def layers() -> Iterator[dict[str, Any]]:
        consumed.append(True)
        yield layer

    plan["shots"][0]["layers"] = layers()
    with pytest.raises(ValidationError, match="list or tuple"):
        make_director_response(status="ready", plan=plan, evidence_requests=[])
    assert consumed == []


@pytest.mark.parametrize("mapping_boundary", ["plan", "shot", "source", "subtitles"])
def test_director_recurses_through_mapping_payloads(mapping_boundary: str) -> None:
    plan: dict[str, Any] | UserDict[str, Any] = make_plan().model_dump()
    shot = plan["shots"][0]
    if mapping_boundary == "subtitles":
        subtitles = SubtitlePlan(destination=LOWER_THIRD).model_dump()
        subtitles["primary_color"] = "red;movie=evil"
        shot["subtitles"] = UserDict(subtitles)
    else:
        shot["source"]["vod_url"] = "secret"
        if mapping_boundary == "source":
            shot["source"] = UserDict(shot["source"])
        elif mapping_boundary == "shot":
            plan["shots"][0] = UserDict(shot)
        else:
            plan = UserDict(plan)

    with pytest.raises(ValidationError, match="Unrequested fields|#RRGGBB"):
        make_director_response(status="ready", plan=plan, evidence_requests=[])


def test_visual_request_recurses_through_mapping_source() -> None:
    with pytest.raises(ValidationError, match="Unrequested fields"):
        make_visual_request(source=UserDict(start=0, end=1, vod_url="secret"))


def test_director_accepts_normalized_mappings_and_tuple_sequences() -> None:
    plan = make_plan().model_dump()
    shot = plan["shots"][0]
    shot["source"] = UserDict(shot["source"])
    shot["layers"] = tuple(UserDict(layer) for layer in shot["layers"])
    shot["subtitles"] = UserDict(SubtitlePlan(destination=LOWER_THIRD).model_dump())
    plan["shots"] = (UserDict(shot),)
    response = contracts.VisualDirectorResponse.model_validate(UserDict(
        status="ready", plan=UserDict(plan), evidence_requests=(), reasoning="Ready.",
    ))
    assert response.plan is not None
    assert response.plan.shots[0].subtitles is not None
    assert response.plan.shots[0].subtitles.primary_color == "#FFD51A"


def test_standalone_composition_keeps_historical_iterable_and_color_semantics() -> None:
    plan = make_plan().model_dump()
    shot = plan["shots"][0]
    shot["source"] = UserDict(shot["source"], vod_url="secret")
    shot["subtitles"] = UserDict(
        SubtitlePlan(destination=LOWER_THIRD).model_dump(), primary_color="red;movie=evil",
    )
    plan["shots"] = iter([shot])
    standalone = VisualCompositionPlan.model_validate(plan)
    assert standalone.shots[0].subtitles is not None
    assert standalone.shots[0].subtitles.primary_color == "red;movie=evil"


def test_context_accepts_inclusive_15s_expansion_and_clamps_to_source_duration() -> None:
    response = make_director_response()
    contracts.validate_visual_director_context(
        response, clip_id="clip_1", editorial_ranges=[TimeRange(start=20, end=25)],
        source_duration=100,
    )
    edge = make_director_response(evidence_requests=[
        make_visual_request(source={"start": 0, "end": 25}),
    ])
    contracts.validate_visual_director_context(
        edge, clip_id="clip_1", editorial_ranges=[TimeRange(start=0, end=10)], source_duration=25,
    )


@pytest.mark.parametrize("source", [
    {"start": 4.9999, "end": 35}, {"start": 5, "end": 40.0001},
    {"start": 5, "end": 101}, {"start": 5, "end": 85},
])
def test_context_rejects_outside_source_or_any_single_expanded_editorial_window(
    source: dict[str, float],
) -> None:
    response = make_director_response(evidence_requests=[make_visual_request(source=source)])
    with pytest.raises(ValueError):
        contracts.validate_visual_director_context(
            response, clip_id="clip_1",
            editorial_ranges=[TimeRange(start=20, end=25), TimeRange(start=70, end=75)],
            source_duration=100,
        )


def test_context_rejects_repeated_signature_with_new_id_cadence_or_iteration() -> None:
    original = make_visual_request()
    signature = contracts.visual_evidence_request_signature(original)
    repeated = make_visual_request(request_id="probe_2", question=" Where  is the camera? ",
                                   sample_interval_seconds=0.5, iteration=2)
    assert contracts.visual_evidence_request_signature(repeated) == signature
    with pytest.raises(ValueError, match="signature"):
        contracts.validate_visual_director_context(
            make_director_response(evidence_requests=[repeated]), clip_id="clip_1",
            editorial_ranges=[TimeRange(start=20, end=25)], source_duration=100,
            prior_request_signatures={signature},
        )


def test_context_allows_new_question_and_rounds_signature_times_to_milliseconds() -> None:
    original = make_visual_request()
    signature = contracts.visual_evidence_request_signature(original)
    rounded = make_visual_request(source={"start": 5.00001, "end": 35.00001})
    assert contracts.visual_evidence_request_signature(rounded) == signature
    clarification = make_visual_request(question="Is the boundary to the left of the face?")
    assert contracts.visual_evidence_request_signature(clarification) != signature
    contracts.validate_visual_director_context(
        make_director_response(evidence_requests=[clarification]), clip_id="clip_1",
        editorial_ranges=[TimeRange(start=20, end=25)], source_duration=100,
        prior_request_signatures={signature},
    )


def test_context_rejects_same_signature_twice_in_batch() -> None:
    with pytest.raises(ValueError, match="signature"):
        contracts.validate_visual_director_context(
            make_director_response(evidence_requests=[make_visual_request(),
                                                     make_visual_request(request_id="probe_2")]),
            clip_id="clip_1", editorial_ranges=[TimeRange(start=20, end=25)], source_duration=100,
        )


def test_context_enforces_clip_association_for_requests_and_plan() -> None:
    for response in (
        make_director_response(evidence_requests=[make_visual_request(clip_id="other")]),
        make_director_response(status="ready", plan=make_plan(clip_id="other"), evidence_requests=[]),
    ):
        with pytest.raises(ValueError, match="clip_id"):
            contracts.validate_visual_director_context(
                response, clip_id="clip_1", editorial_ranges=[TimeRange(start=10, end=12)],
                source_duration=100,
            )


@pytest.mark.parametrize("duration", [0, -1, float("inf"), float("nan")])
def test_context_rejects_invalid_host_duration(duration: float) -> None:
    with pytest.raises(ValueError, match="source_duration"):
        contracts.validate_visual_director_context(
            make_director_response(), clip_id="clip_1", editorial_ranges=[TimeRange(start=20, end=25)],
            source_duration=duration,
        )


def test_context_rejects_missing_or_out_of_bounds_editorial_ranges() -> None:
    for ranges in ([], [TimeRange(start=90, end=101)]):
        with pytest.raises(ValueError, match="editorial"):
            contracts.validate_visual_director_context(
                make_director_response(), clip_id="clip_1", editorial_ranges=ranges, source_duration=100,
            )
def test_protocol_field_descriptions_explain_cross_field_rules_without_shape_changes() -> None:
    from clipperx.visual_contracts import VisualDirectorResponse

    definitions = VisualDirectorResponse.model_json_schema()["$defs"]
    expected = {
        ("VisualLayer", "fill"): "sharp contained foreground",
        ("VisualLayer", "blur_sigma"): "contain_blur",
        ("VisualLayer", "opacity"): "ready",
        ("VisualLayer", "destination"): "full-canvas",
        ("CropKeyframe", "at"): "relative to shot",
        ("OverlayPlan", "start"): "relative to shot",
        ("AudioCue", "start"): "relative to shot",
        ("VisualShot", "source"): "editorial",
        ("VisualShot", "output"): "global output",
        ("VisualShot", "title"): "headline.overlay_text",
        ("VisualShot", "evidence_ids"): "local",
        ("VisualDirectorResponse", "style_card_ids"): "visual inspiration",
    }
    for (model, field), term in expected.items():
        schema = definitions[model] if model != "VisualDirectorResponse" else (
            VisualDirectorResponse.model_json_schema())
        assert term in schema["properties"][field].get("description", "")

