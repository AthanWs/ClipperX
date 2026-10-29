"""Host overlay defaults are explicit, bounded and independent of content centering."""

import math

import pytest
from pydantic import ValidationError

from clipperx.platform_safety import common_vertical_safe_zones, platform_safe_zones
from clipperx.visual_contracts import NormalizedRect


PROFILES = ("tiktok", "reels", "shorts", "common_vertical")
REGIONS = ("title", "subtitle_allowed", "overlay_forbidden_right", "overlay_forbidden_bottom")


def test_common_title_band_starts_below_top_ten_percent() -> None:
    zones = common_vertical_safe_zones()
    assert zones.title.y >= 0.10


def test_right_controls_restrict_overlays_not_canvas_center() -> None:
    zones = common_vertical_safe_zones()
    assert zones.canvas_center_x == 0.5
    assert zones.overlay_forbidden_right.x >= 0.80


def test_subtitle_region_avoids_bottom_ui() -> None:
    zones = common_vertical_safe_zones()
    bottom = zones.subtitle_allowed.y + zones.subtitle_allowed.height
    assert bottom <= 0.82 or math.isclose(bottom, 0.82, rel_tol=0, abs_tol=1e-12)


def test_common_profile_matches_exact_approved_geometry() -> None:
    zones = common_vertical_safe_zones()
    assert zones.profile == "common_vertical"
    assert zones.title.model_dump() == {"x": 0.08, "y": 0.10, "width": 0.84, "height": 0.12}
    assert zones.subtitle_allowed.model_dump() == {
        "x": 0.10, "y": 0.48, "width": 0.72, "height": 0.34,
    }
    assert zones.overlay_forbidden_right.model_dump() == {
        "x": 0.82, "y": 0.36, "width": 0.18, "height": 0.52,
    }
    assert zones.overlay_forbidden_bottom.model_dump() == {
        "x": 0, "y": 0.82, "width": 1, "height": 0.18,
    }


@pytest.mark.parametrize("profile", PROFILES)
def test_named_profiles_use_the_approved_host_defaults(profile: str) -> None:
    zones = platform_safe_zones(profile)
    common = common_vertical_safe_zones()
    assert zones.profile == profile
    assert zones.canvas_center_x == common.canvas_center_x == 0.5
    for name in REGIONS:
        assert getattr(zones, name).model_dump() == getattr(common, name).model_dump()


@pytest.mark.parametrize("profile", PROFILES)
def test_all_profile_rectangles_are_normalized_and_inside_canvas(profile: str) -> None:
    zones = platform_safe_zones(profile)
    for name in REGIONS:
        rectangle = getattr(zones, name)
        assert isinstance(rectangle, NormalizedRect)
        assert 0 <= rectangle.x < rectangle.x + rectangle.width <= 1
        assert 0 <= rectangle.y < rectangle.y + rectangle.height <= 1


@pytest.mark.parametrize("profile", PROFILES[:-1])
def test_common_exclusions_cover_each_platform_exclusion(profile: str) -> None:
    zones = platform_safe_zones(profile)
    common = common_vertical_safe_zones()
    for name in ("overlay_forbidden_right", "overlay_forbidden_bottom"):
        region = getattr(zones, name)
        exclusion = getattr(common, name)
        assert exclusion.x <= region.x
        assert exclusion.y <= region.y
        assert exclusion.x + exclusion.width >= region.x + region.width
        assert exclusion.y + exclusion.height >= region.y + region.height


@pytest.mark.parametrize("profile", ["", "unknown", "TikTok", " reels", "shorts "])
def test_unknown_profile_names_fail_explicitly(profile: str) -> None:
    with pytest.raises(ValueError, match="Unknown platform safety profile"):
        platform_safe_zones(profile)


def test_profile_cannot_change_canvas_center_or_replace_overlay_regions() -> None:
    zones = common_vertical_safe_zones()
    with pytest.raises(ValidationError, match="frozen"):
        setattr(zones, "canvas_center_x", 0.4)
    with pytest.raises(ValidationError, match="frozen"):
        setattr(zones, "title", zones.subtitle_allowed)


@pytest.mark.parametrize("name", REGIONS)
def test_nested_regions_cannot_leak_mutation_between_calls(name: str) -> None:
    zones = common_vertical_safe_zones()
    region = getattr(zones, name)
    original = region.model_dump()
    with pytest.raises(ValidationError, match="frozen"):
        region.x = 0.01
    dumped = zones.model_dump()
    dumped[name]["x"] = 0.02
    assert getattr(common_vertical_safe_zones(), name).model_dump() == original
    assert getattr(platform_safe_zones("tiktok"), name).model_dump() == original
    assert region.model_dump() == original
