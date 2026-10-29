"""Creator captions stay with content, not the separated webcam panel."""
from clipperx import editing
from clipperx.caption_templates import get_template


def test_default_creator_captions_are_large_enough_for_the_content_panel():
    template=get_template("default")
    assert 68 <= template.font_size <= 78
    assert template.primary_color=="&H0000FFFF"
    assert template.outline_color=="&H00000000" and template.outline_width>=7


def test_integrated_gameplay_captions_follow_game_not_top_camera():
    cfg=editing.LayoutConfig(content_crop=(346,0,934,720),
        brand_image_path="assets/branding/KickAthanW.png")
    bottom=cfg.height-editing._caption_margin_for_segment(cfg,"SPLIT",(0,0,334,271),(1280,720))
    assert 1000 <= bottom <= 1100
    assert bottom <= cfg.height-cfg.safe_bottom


def test_bottom_camera_captions_stay_in_content_below_the_title():
    cfg=editing.LayoutConfig(camera_position="bottom",content_crop=(346,0,934,720),
        brand_image_path="assets/branding/KickAthanW.png")
    bottom=cfg.height-editing._caption_margin_for_segment(cfg,"SPLIT",(0,0,334,271),(1280,720))
    assert 580 <= bottom <= 700
    assert bottom < editing._split_panel_geometry(cfg,934,720,334,271)[2]


def test_explicit_banded_legacy_placement_is_preserved():
    cfg=editing.LayoutConfig(integrated_overlays=False,
        brand_image_path="assets/branding/KickAthanW.png")
    assert editing._caption_margin_for_segment(cfg,"SPLIT",(0,0,334,271),(1280,720))==1272
