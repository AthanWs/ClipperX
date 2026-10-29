"""Host-owned style-library cards and a minimal model-facing visual analysis."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
import re
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from clipperx.visual_contracts import NormalizedRect, _StrictVisualContract


class SceneFamily(str, Enum):
    GAMEPLAY_WITH_CAMERA = "gameplay_with_camera"
    FULL_CAMERA = "full_camera"
    BROWSER_WITH_CAMERA = "browser_with_camera"
    REACTED_VIDEO_WITH_CAMERA = "reacted_video_with_camera"
    CONTENT_ONLY = "content_only"
    MULTI_PERSON = "multi_person"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class BackgroundTreatment(str, Enum):
    COVER = "cover"
    CONTAIN_BLUR = "contain_blur"
    BLACK = "black"
    SOLID = "solid"
    MIXED = "mixed"


HexColor = Annotated[str, Field(pattern=r"^#[0-9a-fA-F]{6}$")]
EvidenceRef = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$")]
_WINDOWS_RESERVED_NAME = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])$", re.IGNORECASE)


def _validate_library_ref(value: str) -> str:
    """Allow portable manifest-relative file refs, never paths supplied by models."""
    if (
        not value
        or any(ord(char) < 32 for char in value)
        or any(char in value for char in '\\:<>"|?*')
    ):
        raise ValueError("Library ref must be a portable relative path")
    for part in value.split("/"):
        if not part or part in {".", ".."} or part.endswith((".", " ")):
            raise ValueError("Library ref cannot be absolute or contain traversal")
        if _WINDOWS_RESERVED_NAME.fullmatch(part.split(".", 1)[0]):
            raise ValueError("Library ref contains a reserved device name")
    return value


class StyleKeyframe(_StrictVisualContract):
    timestamp: float = Field(ge=0)
    path: str

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return _validate_library_ref(value)


class StyleCard(_StrictVisualContract):
    contract_version: Literal[1] = 1
    card_id: str = Field(min_length=1)
    clip_fingerprint: str = Field(min_length=1)
    source_clip_ref: str
    duration: float = Field(gt=0)
    created_at: datetime | None
    scene_family: SceneFamily
    keyframes: list[StyleKeyframe] = Field(min_length=1, max_length=8)
    camera_regions: list[NormalizedRect]
    content_regions: list[NormalizedRect]
    background_treatment: BackgroundTreatment
    title_regions: list[NormalizedRect]
    brand_regions: list[NormalizedRect]
    subtitle_regions: list[NormalizedRect]
    subtitle_palette: list[HexColor]
    cut_rate_per_minute: float = Field(ge=0)
    zoom_count: int = Field(ge=0)
    special_techniques: list[str]
    visual_quality: float = Field(ge=0, le=1)
    narrative_quality: float | None = Field(ge=0, le=1)
    approval: Literal["approved", "unreviewed", "rejected"]
    description: str = Field(min_length=1)
    evidence: list[str]

    @field_validator("source_clip_ref")
    @classmethod
    def validate_source_ref(cls, value: str) -> str:
        return _validate_library_ref(value)

    @model_validator(mode="after")
    def validate_keyframe_bounds(self) -> "StyleCard":
        if any(frame.timestamp > self.duration for frame in self.keyframes):
            raise ValueError("Keyframe timestamp exceeds card duration")
        return self


class StyleCardAnalysisResponse(_StrictVisualContract):
    contract_version: Literal[1] = 1
    scene_family: SceneFamily
    camera_regions: list[NormalizedRect]
    content_regions: list[NormalizedRect]
    background_treatment: BackgroundTreatment
    title_regions: list[NormalizedRect]
    brand_regions: list[NormalizedRect]
    subtitle_regions: list[NormalizedRect]
    subtitle_palette: list[HexColor]
    special_techniques: list[str]
    description: str = Field(min_length=1)
    evidence_refs: list[EvidenceRef]
