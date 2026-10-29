"""Strict JSON contracts shared by planner phases."""

from __future__ import annotations

from enum import Enum
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

MAX_CLIP_SECONDS = 179.0


class TimeRange(BaseModel):
    start: float = Field(ge=0)
    end: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_range(self) -> "TimeRange":
        if self.end <= self.start:
            raise ValueError("end debe ser mayor que start")
        return self

    @property
    def duration(self) -> float:
        return self.end - self.start


class ProbeDirection(str, Enum):
    BEFORE = "before"
    AFTER = "after"
    INSIDE = "inside"


class EvidenceKind(str, Enum):
    CONTEXT = "context"
    VISUAL_REACTION = "visual_reaction"
    INTERNAL_GAP = "internal_gap"
    TOPIC_LINK = "topic_link"
    SPEAKER_IDENTITY = "speaker_identity"
    ACTIVE_SPEAKER = "active_speaker"


class EvidenceRequest(BaseModel):
    request_id: str
    candidate_id: str
    kind: EvidenceKind
    direction: ProbeDirection
    window: TimeRange
    sample_interval_seconds: float = Field(default=1.0, ge=0.25, le=10.0)
    question: str = Field(min_length=3)
    iteration: int = Field(default=1, ge=1, le=4)

    @model_validator(mode="after")
    def validate_window(self) -> "EvidenceRequest":
        if self.window.duration > 300.0:
            raise ValueError("Una solicitud de evidencia no puede superar 300 segundos")
        return self


class CandidateSpan(BaseModel):
    candidate_id: str
    topic_id: str
    transcript_ranges: list[TimeRange] = Field(min_length=1)
    hook: str
    payoff: str
    context_risk: str
    confidence: float = Field(ge=0, le=1)
    evidence_requests: list[EvidenceRequest] = Field(default_factory=list)


class DiscoveryBatchResponse(BaseModel):
    contract_version: Literal[1] = 1
    batch_id: str
    cursor_before: int = Field(ge=0)
    cursor_after: int = Field(gt=0)
    candidates: list[CandidateSpan] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_cursor(self) -> "DiscoveryBatchResponse":
        if self.cursor_after <= self.cursor_before:
            raise ValueError("cursor_after debe avanzar")
        return self


class ReviewAction(str, Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    NEED_MORE_EVIDENCE = "need_more_evidence"
    SPLIT = "split"
    MERGE = "merge"
    REASSIGN = "reassign"


class BoundaryReviewResponse(BaseModel):
    contract_version: Literal[1] = 1
    candidate_id: str
    action: ReviewAction
    selected_ranges: list[TimeRange] = Field(default_factory=list)
    related_candidate_ids: list[str] = Field(default_factory=list)
    evidence_requests: list[EvidenceRequest] = Field(default_factory=list)
    reasoning: str
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_action(self) -> "BoundaryReviewResponse":
        if self.action == ReviewAction.NEED_MORE_EVIDENCE and not self.evidence_requests:
            raise ValueError("need_more_evidence requiere al menos una solicitud")
        if self.action == ReviewAction.ACCEPT and not self.selected_ranges:
            raise ValueError("accept requiere rangos seleccionados")
        return self


class HeadlinePlan(BaseModel):
    overlay_text: str = Field(min_length=1, max_length=80)
    platform_title: str = Field(min_length=1, max_length=160)
    caption: str = Field(default="", max_length=2200)
    keywords: list[str] = Field(default_factory=list)


class NormalizedCrop(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> "NormalizedCrop":
        if self.x + self.width > 1.000001 or self.y + self.height > 1.000001:
            raise ValueError("El crop normalizado sale del frame")
        return self


class EditSegment(BaseModel):
    source: TimeRange
    output_order: int = Field(ge=0)
    narrative_role: Literal["hook", "context", "development", "reaction", "payoff", "outro"]
    layout: Literal["full_content", "full_camera", "split", "picture_in_picture", "adaptive"]
    camera_crop: NormalizedCrop | None = None
    transition_in: Literal["cut", "none"] = "cut"
    keep_source_audio: Literal[True] = True
    reason: str


class ClipPlan(BaseModel):
    contract_version: Literal[1] = 1
    clip_id: str
    topic_id: str
    headline: HeadlinePlan
    segments: list[EditSegment] = Field(min_length=1)
    subtitle_style: str = "dynamic_clean"
    target_aspect_ratio: Literal["9:16"] = "9:16"
    reasoning: str

    @model_validator(mode="after")
    def validate_composition(self) -> "ClipPlan":
        self.segments.sort(key=lambda segment: segment.output_order)
        for index, segment in enumerate(self.segments):
            segment.output_order = index
        if self.duration > MAX_CLIP_SECONDS:
            raise ValueError(f"El clip supera el máximo de {MAX_CLIP_SECONDS:.0f}s")
        return self

    @property
    def duration(self) -> float:
        return sum(segment.source.duration for segment in self.segments)


class GlobalCompositionResponse(BaseModel):
    contract_version: Literal[1] = 1
    clips: list[ClipPlan] = Field(default_factory=list)
    rejected_candidate_ids: list[str] = Field(default_factory=list)
    unresolved_requests: list[EvidenceRequest] = Field(default_factory=list)


def export_contract_schemas(output_dir: Path) -> list[Path]:
    from clipperx.visual_contracts import VisualCompositionPlan

    output_dir.mkdir(parents=True, exist_ok=True)
    models = (
        DiscoveryBatchResponse,
        BoundaryReviewResponse,
        GlobalCompositionResponse,
        VisualCompositionPlan,
    )
    outputs: list[Path] = []
    for model in models:
        path = output_dir / f"{model.__name__}.schema.json"
        path.write_text(
            json.dumps(model.model_json_schema(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        outputs.append(path)
    return outputs
