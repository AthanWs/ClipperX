"""Conservative host defaults for title, subtitle and brand overlays.

TikTok, Reels and Shorts currently use the same approved normalized geometry;
these are design defaults, not measured vendor UI coordinates or a promise about
future interfaces or monetization rules. The common exclusions are therefore
their union. Future platform-specific geometry must preserve that union.

Only overlay placement is restricted. Source content, gameplay and streamer
panels retain the true output canvas center at x=0.5.
"""

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from clipperx.visual_contracts import NormalizedRect


class PlatformProfile(str, Enum):
    TIKTOK = "tiktok"
    REELS = "reels"
    SHORTS = "shorts"
    COMMON_VERTICAL = "common_vertical"


class FrozenNormalizedRect(NormalizedRect):
    """Reuse the visual contract's geometry with immutable safe-zone values."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


class PlatformSafeZones(BaseModel):
    """Immutable overlay regions; ``canvas_center_x`` never offsets content."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    profile: PlatformProfile
    title: FrozenNormalizedRect
    subtitle_allowed: FrozenNormalizedRect
    overlay_forbidden_right: FrozenNormalizedRect
    overlay_forbidden_bottom: FrozenNormalizedRect
    canvas_center_x: float = Field(default=0.5, ge=0.5, le=0.5)


def platform_safe_zones(profile: str) -> PlatformSafeZones:
    """Look up an exact named host profile; unknown names raise ``ValueError``.

    Every call returns fresh immutable rectangles. All four profiles intentionally
    share this baseline until verified platform-specific coordinates are adopted.
    """
    try:
        selected = PlatformProfile(profile)
    except ValueError as exc:
        raise ValueError(f"Unknown platform safety profile: {profile!r}") from exc
    return PlatformSafeZones(
        profile=selected,
        title=FrozenNormalizedRect(x=0.08, y=0.10, width=0.84, height=0.12),
        subtitle_allowed=FrozenNormalizedRect(x=0.10, y=0.48, width=0.72, height=0.34),
        overlay_forbidden_right=FrozenNormalizedRect(x=0.82, y=0.36, width=0.18, height=0.52),
        overlay_forbidden_bottom=FrozenNormalizedRect(x=0, y=0.82, width=1, height=0.18),
    )


def common_vertical_safe_zones() -> PlatformSafeZones:
    """Return the approved cross-platform overlay defaults without recentering."""
    return platform_safe_zones(PlatformProfile.COMMON_VERTICAL)
