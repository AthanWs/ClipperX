from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from clipperx.style_contracts import (
    BackgroundTreatment,
    SceneFamily,
    StyleCard,
    StyleCardAnalysisResponse,
    StyleKeyframe,
)


def card_values() -> dict[str, object]:
    return {
        "card_id": "style-1",
        "clip_fingerprint": "abc",
        "source_clip_ref": "clips/style-1.mp4",
        "duration": 10.0,
        "created_at": datetime(2026, 10, 14, tzinfo=timezone.utc),
        "scene_family": SceneFamily.GAMEPLAY_WITH_CAMERA,
        "keyframes": [{"timestamp": 1.0, "path": "frames/1.jpg"}],
        "camera_regions": [],
        "content_regions": [],
        "background_treatment": BackgroundTreatment.BLACK,
        "title_regions": [],
        "brand_regions": [],
        "subtitle_regions": [],
        "subtitle_palette": ["#FFD51A", "#000000"],
        "cut_rate_per_minute": 4.0,
        "zoom_count": 1,
        "special_techniques": [],
        "visual_quality": 0.9,
        "narrative_quality": None,
        "approval": "unreviewed",
        "description": "Gameplay arriba, cámara amplia abajo, espacios negros.",
        "evidence": ["frame_1"],
    }


def analysis_values() -> dict[str, object]:
    fields = StyleCardAnalysisResponse.model_fields
    values = {key: value for key, value in card_values().items() if key in fields}
    values["evidence_refs"] = ["frame_1"]
    return values


def test_style_card_requires_host_provenance_without_fake_vod_pair() -> None:
    card = StyleCard.model_validate(card_values())
    assert not hasattr(card, "source_vod_timestamp")
    assert card.source_clip_ref == "clips/style-1.mp4"
    assert StyleCard.model_validate_json(card.model_dump_json()) == card
    for required in ("source_clip_ref", "duration"):
        values = card_values()
        values.pop(required)
        with pytest.raises(ValidationError):
            StyleCard.model_validate(values)


@pytest.mark.parametrize("count", [0, 9])
def test_style_card_requires_one_to_eight_keyframes(count: int) -> None:
    values = card_values()
    values["keyframes"] = [{"timestamp": 1.0, "path": f"frames/{i}.jpg"} for i in range(count)]
    with pytest.raises(ValidationError):
        StyleCard.model_validate(values)


@pytest.mark.parametrize(
    "path",
    [
        "/frames/a.jpg",
        "../a.jpg",
        "a/../b.jpg",
        "C:/a.jpg",
        "C:a.jpg",
        "\\\\server\\a.jpg",
        "https://example.test/a.jpg",
        "file:a.jpg",
        "a\\b.jpg",
        "a//b.jpg",
        "a/./b.jpg",
        "a/.. /b.jpg",
        "a\x00.jpg",
        "a/CON.jpg",
    ],
)
def test_style_library_refs_reject_nonportable_or_escaping_paths(path: str) -> None:
    with pytest.raises(ValidationError):
        StyleKeyframe(timestamp=0, path=path)
    values = card_values()
    values["source_clip_ref"] = path
    with pytest.raises(ValidationError):
        StyleCard.model_validate(values)


@pytest.mark.parametrize("timestamp", [-1.0, 10.0001, float("inf"), float("nan")])
def test_keyframe_must_be_finite_and_within_card_duration(timestamp: float) -> None:
    values = card_values()
    values["keyframes"] = [{"timestamp": timestamp, "path": "frames/1.jpg"}]
    with pytest.raises(ValidationError):
        StyleCard.model_validate(values)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("duration", 0),
        ("duration", float("inf")),
        ("visual_quality", 1.1),
        ("narrative_quality", -0.1),
        ("cut_rate_per_minute", -1),
        ("cut_rate_per_minute", "nan"),
        ("zoom_count", -1),
        ("subtitle_palette", ["red"]),
        ("subtitle_palette", ["#12345G"]),
        ("approval", "model_approved"),
        ("source_vod_timestamp", 123.0),
    ],
)
def test_style_card_rejects_invalid_values(field: str, value: object) -> None:
    values = card_values()
    values[field] = value
    with pytest.raises(ValidationError):
        StyleCard.model_validate(values)


def test_analysis_contract_excludes_host_fields_and_serializes_only_visual_analysis() -> None:
    analysis = StyleCardAnalysisResponse.model_validate(analysis_values())
    forbidden = {
        "card_id",
        "clip_fingerprint",
        "source_clip_ref",
        "duration",
        "created_at",
        "keyframes",
        "approval",
        "narrative_quality",
        "source_vod_timestamp",
    }
    assert forbidden.isdisjoint(analysis.model_dump())
    assert forbidden.isdisjoint(StyleCardAnalysisResponse.model_json_schema()["properties"])
    for field in forbidden:
        values = analysis_values()
        values[field] = "invented"
        with pytest.raises(ValidationError):
            StyleCardAnalysisResponse.model_validate(values)


@pytest.mark.parametrize(
    "region",
    [
        {"x": 0, "y": 0, "width": 1, "height": 1, "source_path": "secret"},
        {"x": 0, "y": 0, "width": float("nan"), "height": 1},
    ],
)
def test_analysis_rejects_nested_extra_or_nonfinite_regions(region: dict[str, object]) -> None:
    values = analysis_values()
    values["camera_regions"] = [region]
    with pytest.raises(ValidationError):
        StyleCardAnalysisResponse.model_validate(values)


@pytest.mark.parametrize("ref", ["../frame.jpg", "C:/frame.jpg", "https://example.test/1"])
def test_analysis_evidence_refs_are_opaque_host_evidence_ids(ref: str) -> None:
    values = analysis_values()
    values["evidence_refs"] = [ref]
    with pytest.raises(ValidationError):
        StyleCardAnalysisResponse.model_validate(values)
