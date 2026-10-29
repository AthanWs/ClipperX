import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from clipperx.planner_contracts import (
    BoundaryReviewResponse,
    ClipPlan,
    DiscoveryBatchResponse,
    EditSegment,
    GlobalCompositionResponse,
    HeadlinePlan,
    NormalizedCrop,
    TimeRange,
    export_contract_schemas,
)
from clipperx.visual_contracts import VisualCompositionPlan


def test_clip_plan_rejects_more_than_179_seconds() -> None:
    with pytest.raises(ValidationError):
        ClipPlan(
            clip_id="clip-1",
            topic_id="topic-1",
            headline=HeadlinePlan(overlay_text="Hook", platform_title="Título"),
            segments=[
                EditSegment(
                    source=TimeRange(start=0, end=180),
                    output_order=0,
                    narrative_role="hook",
                    layout="adaptive",
                    reason="test",
                )
            ],
            reasoning="test",
        )


def test_clip_plan_normalizes_one_based_output_order() -> None:
    plan = ClipPlan(
        clip_id="clip-1",
        topic_id="topic-1",
        headline=HeadlinePlan(overlay_text="Hook", platform_title="Título"),
        segments=[
            EditSegment(
                source=TimeRange(start=10, end=12),
                output_order=2,
                narrative_role="payoff",
                layout="adaptive",
                reason="final",
            ),
            EditSegment(
                source=TimeRange(start=0, end=2),
                output_order=1,
                narrative_role="hook",
                layout="adaptive",
                reason="inicio",
            ),
        ],
        reasoning="test",
    )

    assert [segment.output_order for segment in plan.segments] == [0, 1]
    assert [segment.source.start for segment in plan.segments] == [0, 10]


def test_v1_edit_segment_fields_remain_unchanged() -> None:
    assert set(EditSegment.model_fields) == {
        "source",
        "output_order",
        "narrative_role",
        "layout",
        "camera_crop",
        "transition_in",
        "keep_source_audio",
        "reason",
    }

    segment = EditSegment(
        source=TimeRange(start=0, end=2),
        output_order=0,
        narrative_role="hook",
        layout="adaptive",
        reason="v1",
    )
    assert segment.camera_crop is None
    assert segment.transition_in == "cut"
    assert segment.keep_source_audio is True


def test_v1_normalized_crop_and_layout_remain_strict() -> None:
    crop = NormalizedCrop(x=0, y=0, width=1, height=1)
    assert crop.width == 1

    with pytest.raises(ValidationError, match="sale del frame"):
        NormalizedCrop(x=0.8, y=0, width=0.3, height=1)

    with pytest.raises(ValidationError):
        EditSegment.model_validate({
            "source": TimeRange(start=0, end=2),
            "output_order": 0,
            "narrative_role": "hook",
            "layout": "unknown",
            "reason": "invalid layout",
        })


def test_export_contract_schemas_includes_v1_and_visual_composition_v2(
    tmp_path: Path,
) -> None:
    outputs = export_contract_schemas(tmp_path)

    expected_models = (
        DiscoveryBatchResponse,
        BoundaryReviewResponse,
        GlobalCompositionResponse,
        VisualCompositionPlan,
    )
    assert [path.name for path in outputs] == [
        f"{model.__name__}.schema.json" for model in expected_models
    ]

    visual_schema = json.loads((tmp_path / "VisualCompositionPlan.schema.json").read_text())
    assert visual_schema["title"] == "VisualCompositionPlan"
    assert visual_schema["properties"]["contract_version"]["const"] == 2
