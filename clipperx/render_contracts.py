"""Immutable host-owned render specifications; JSON collections remain arrays."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .planner_contracts import MAX_CLIP_SECONDS


class EncoderName(str, Enum):
    H264_NVENC = "h264_nvenc"
    LIBX264 = "libx264"


class RenderProfile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False,
                              revalidate_instances="always")

    width: Literal[1080] = 1080
    height: Literal[1920] = 1920
    fps: Literal[60] = 60
    video_encoder: EncoderName = EncoderName.H264_NVENC
    audio_codec: Literal["aac"] = "aac"
    audio_bitrate: Literal["192k"] = "192k"
    audio_rate: Literal[48000] = 48000
    pixel_format: Literal["yuv420p"] = "yuv420p"


class FFmpegRenderSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    source: str
    output: str
    plan_hash: str
    command: tuple[str, ...]
    filter_complex: str
    expected_duration: float = Field(gt=0, le=MAX_CLIP_SECONDS)
    profile: RenderProfile
    assets: tuple[str, ...]
