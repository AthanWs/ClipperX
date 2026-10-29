"""Subtitle caption template system for ClipperX."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class CaptionTemplate:
    """Defines the visual style and animation for ASS subtitles."""

    name: str
    font_name: str
    font_size: int
    primary_color: str       # ASS hex &HAABBGGRR
    highlight_color: str     # For karaoke animation
    outline_color: str
    back_color: str
    bold: bool
    outline_width: int
    shadow_depth: int
    animation: str            # "none" | "reveal" | "karaoke" | "pop" | "fade"
    margin_v: int
    margin_l: int = 170
    margin_r: int = 170


# ASS color format: &HAABBGGRR (alpha, blue, green, red)
# Note: ASS uses BGR order, not RGB. &H00 alpha = fully opaque, &HFF = transparent.

CAPTION_TEMPLATES: dict[str, CaptionTemplate] = {
    "default": CaptionTemplate(
        name="default",
        font_name="Montserrat SemiBold",
        font_size=72,
        primary_color="&H0000FFFF",       # Bright yellow (#FFFF00)
        highlight_color="&H00FFFFFF",     # White highlight
        outline_color="&H00000000",       # Opaque black
        back_color="&H64000000",          # Semi-transparent black
        bold=True,
        outline_width=8,
        shadow_depth=1,
        animation="reveal",
        margin_v=890,
        margin_l=210,
        margin_r=210,
    ),
    "hormozi": CaptionTemplate(
        name="hormozi",
        font_name="Impact",
        font_size=62,
        primary_color="&H00FFFFFF",       # White
        highlight_color="&H0000FF00",     # Green (#00FF00 -> BGR: 00FF00)
        outline_color="&H00000000",       # Black
        back_color="&H00000000",
        bold=True,
        outline_width=4,
        shadow_depth=2,
        animation="karaoke",
        margin_v=620,
    ),
    "mrbeast": CaptionTemplate(
        name="mrbeast",
        font_name="Bangers",
        font_size=64,
        primary_color="&H0000D7FF",       # Gold/Yellow (#FFD700 -> BGR: 00D7FF)
        highlight_color="&H000000FF",     # Red (#FF0000 -> BGR: 0000FF)
        outline_color="&H00000000",       # Black
        back_color="&H00000000",
        bold=True,
        outline_width=4,
        shadow_depth=3,
        animation="pop",
        margin_v=600,
    ),
    "minimal": CaptionTemplate(
        name="minimal",
        font_name="Poppins",
        font_size=52,
        primary_color="&H00FFFFFF",       # White
        highlight_color="&H00FFFFFF",     # White (no highlight)
        outline_color="&H00000000",       # No outline (transparent)
        back_color="&H00000000",
        bold=False,
        outline_width=0,
        shadow_depth=1,
        animation="fade",
        margin_v=640,
    ),
    "tiktok": CaptionTemplate(
        name="tiktok",
        font_name="Arial",
        font_size=56,
        primary_color="&H00FFFFFF",       # White
        highlight_color="&H00B469FF",     # Pink (#FF69B4 -> BGR: B469FF)
        outline_color="&H00000000",       # Black
        back_color="&H00000000",
        bold=True,
        outline_width=3,
        shadow_depth=0,
        animation="karaoke",
        margin_v=620,
    ),
    "neon": CaptionTemplate(
        name="neon",
        font_name="Bebas Neue",
        font_size=60,
        primary_color="&H00FFFF00",       # Cyan (#00FFFF -> BGR: FFFF00)
        highlight_color="&H00FF00FF",     # Magenta (#FF00FF -> BGR: FF00FF)
        outline_color="&H00000000",       # Black
        back_color="&H00000000",
        bold=True,
        outline_width=3,
        shadow_depth=2,
        animation="karaoke",
        margin_v=610,
    ),
    "podcast": CaptionTemplate(
        name="podcast",
        font_name="Raleway",
        font_size=54,
        primary_color="&H0020A5DA",       # Warm gold (#DAA520 -> BGR: 20A5DA)
        highlight_color="&H0020A5DA",     # Same (no highlight change)
        outline_color="&H00000000",       # Black
        back_color="&H80000000",          # Semi-transparent black background
        bold=False,
        outline_width=2,
        shadow_depth=0,
        animation="fade",
        margin_v=640,
    ),
}


def get_template(name: str) -> CaptionTemplate:
    """Return a caption template by name, falling back to 'default'."""
    return CAPTION_TEMPLATES.get(name, CAPTION_TEMPLATES["default"])


def list_templates() -> list[str]:
    """Return all available template names."""
    return list(CAPTION_TEMPLATES.keys())
