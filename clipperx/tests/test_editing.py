from clipperx.editing import (
    LayoutConfig,
    _camera_inner_width,
    _content_crop_without_camera,
    _fit_size,
    _resolve_render_segments,
    _split_panel_geometry,
    _top_safe_margin,
    _wrap_overlay_text,
)
from clipperx.schema import ClipCandidate
from clipperx.summaries import build_filename


def test_layout_segments_do_not_reintroduce_cut_gap() -> None:
    clip = ClipCandidate(
        start=0,
        end=6,
        text="test",
        coherence=1.0,
        time_ranges=[(0, 2), (4, 6)],
        layout_segments=[(0, 6, "FULL_CONTENT")],
    )

    assert _resolve_render_segments(clip) == [
        (0, 2, "FULL_CONTENT"),
        (4, 6, "FULL_CONTENT"),
    ]


def test_filename_has_no_hardcoded_creator() -> None:
    clip = ClipCandidate(start=1, end=2, text="", coherence=1.0)

    assert build_filename("Mi clip", clip) == "mi-clip-1-2.mp4"


def test_split_content_excludes_left_corner_camera() -> None:
    crop = _content_crop_without_camera((0, 120, 520, 420), 1920, 1080)

    assert crop is not None
    assert crop[0] > 520
    assert crop[2] < 1400


def test_overlay_title_wraps_within_safe_width() -> None:
    wrapped = _wrap_overlay_text("LE ROBARON LA SILLA EN UN SEGUNDO")

    assert wrapped == "LE ROBARON LA SILLA\nEN UN SEGUNDO"
    assert all(len(line) <= 20 for line in wrapped.splitlines())


def test_split_geometry_uses_compact_header_caption_band_and_bottom() -> None:
    cfg = LayoutConfig()

    top_y, gameplay_height, camera_y, camera_height = _split_panel_geometry(
        cfg, 934, 720, 334, 271
    )

    assert (top_y, gameplay_height, camera_y, camera_height) == (672, 1248, 0, 672)
    assert top_y + gameplay_height == cfg.height


def test_brand_logo_fits_header_without_distortion() -> None:
    assert _fit_size(791, 186, 420, 84) == (356, 84)


def test_headline_uses_compact_impact_header_style() -> None:
    cfg = LayoutConfig()

    assert cfg.title_font_file == "C:/Windows/Fonts/impact.ttf"
    assert cfg.title_font_size == 88
    assert cfg.top_bar_height >= 64
    assert _top_safe_margin(cfg) == 192
    assert cfg.safe_left == cfg.safe_right == 170


def test_camera_geometry_adapts_to_camera_aspect_ratio() -> None:
    cfg = LayoutConfig()

    square = _split_panel_geometry(cfg, 1280, 720, 400, 400)
    widescreen = _split_panel_geometry(cfg, 1280, 720, 640, 360)
    portrait = _split_panel_geometry(cfg, 1280, 720, 300, 500)

    assert square[3] == 672
    assert widescreen[3] == 608
    assert portrait[3] == 672
    assert all(camera_y == 0 for _, _, camera_y, camera_height in (
        square,
        widescreen,
        portrait,
    ))


def test_camera_width_adapts_without_losing_center_axis() -> None:
    assert _camera_inner_width(1080, 480, 334, 271) == 1014
    assert _camera_inner_width(1080, 480, 400, 400) == 928
    assert _camera_inner_width(1080, 480, 300, 500) == 670
    assert _camera_inner_width(1080, 395, 640, 360) == 1014


def test_source_resolution_does_not_change_layout_for_same_aspect() -> None:
    cfg = LayoutConfig()

    hd = _split_panel_geometry(cfg, 1280, 720, 334, 271)
    full_hd = _split_panel_geometry(cfg, 2560, 1440, 668, 542)

    assert hd == full_hd
