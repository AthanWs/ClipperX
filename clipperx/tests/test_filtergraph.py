"""HOST-authored geometry fixtures; these tests do not render media."""

import pytest

from clipperx.filtergraph import (
    PixelRect, clip_source_crop, compile_layer, compile_motion_layer, compile_shot_layers, rect_to_pixels,
    source_crop_pixels,
)
from clipperx.motion_compiler import compile_crop_track
from clipperx.visual_contracts import CropKeyframe, VisualShot
from clipperx.visual_contracts import FillMode, LayerKind, NormalizedRect


def test_normalized_rect_maps_to_even_pixels() -> None:
    assert rect_to_pixels(NormalizedRect(x=.1, y=.2, width=.5, height=.4),
                          width=1920, height=1080) == PixelRect(192, 216, 960, 432)


def test_rounding_uses_shared_even_edges_and_clips_validated_destination() -> None:
    left = rect_to_pixels(NormalizedRect(x=0, y=0, width=.501, height=1),
                          width=1080, height=1920)
    right = rect_to_pixels(NormalizedRect(x=.501, y=0, width=.499, height=1),
                           width=1080, height=1920)
    assert left.x + left.width == right.x
    assert right.x + right.width == 1080
    assert all(value % 2 == 0 for value in (left.width, right.x, right.width))
    edge = NormalizedRect(x=.5, y=0, width=.5000005, height=1)
    assert rect_to_pixels(edge, width=1080, height=1920) == PixelRect(540, 0, 540, 1920)


def test_geometry_revalidates_mutated_rect_instead_of_clipping_invalid_input() -> None:
    rect = NormalizedRect(x=0, y=0, width=1, height=1)
    rect.width = 1.1
    with pytest.raises(ValueError):
        rect_to_pixels(rect, width=1080, height=1920)


@pytest.mark.parametrize("width,height", [(0, 1080), (1920, -1), (1919, 1080)])
def test_geometry_rejects_invalid_canvas(width: int, height: int) -> None:
    with pytest.raises(ValueError):
        rect_to_pixels(NormalizedRect(x=0, y=0, width=1, height=1),
                       width=width, height=height)


def test_tiny_rect_is_not_silently_enlarged() -> None:
    with pytest.raises(ValueError, match="pixel"):
        rect_to_pixels(NormalizedRect(x=0, y=0, width=.0001, height=1),
                       width=1080, height=1920)


def test_source_crop_only_clips_one_pixel_rounding_overflow() -> None:
    assert clip_source_crop(PixelRect(2, 2, 1919, 1079), (1920, 1080)) == (
        PixelRect(2, 2, 1918, 1078))
    with pytest.raises(ValueError, match="source"):
        clip_source_crop(PixelRect(2, 2, 1920, 1078), (1920, 1080))
    with pytest.raises(ValueError, match="source"):
        clip_source_crop(PixelRect(-2, 0, 1920, 1080), (1920, 1080))


def test_cover_centers_without_platform_offset() -> None:
    graph = compile_layer(source_size=(1920, 1080), destination=PixelRect(0, 0, 1080, 960),
                          fill=FillMode.COVER)
    assert "(in_w-out_w)/2" in graph
    assert "(in_h-out_h)/2" in graph
    assert "force_original_aspect_ratio=increase" in graph
    assert "right_ui" not in graph


def test_black_fill_has_no_blur_filter() -> None:
    graph = compile_layer(source_size=(1920, 1080), destination=PixelRect(0, 0, 1080, 960),
                          fill=FillMode.BLACK)
    assert "color=c=black" in graph
    assert "force_original_aspect_ratio=decrease" in graph
    assert "boxblur" not in graph
    assert "overlay=x=(main_w-overlay_w)/2:y=(main_h-overlay_h)/2" in graph


def test_contain_blur_has_separate_backdrop_and_foreground() -> None:
    graph = compile_layer(source_size=(1920, 1080), destination=PixelRect(0, 0, 1080, 960),
                          fill=FillMode.CONTAIN_BLUR, blur_sigma=12)
    assert "boxblur" in graph
    assert "split=2" in graph
    assert "force_original_aspect_ratio=decrease" in graph
    assert "force_original_aspect_ratio=increase" in graph


def test_box_strength_maps_to_explicit_luma_and_subsampled_chroma_radius() -> None:
    graph = compile_layer(source_size=(1920, 1080), destination=PixelRect(0, 0, 1080, 960),
                          fill=FillMode.CONTAIN_BLUR, blur_sigma=12.1)
    assert "boxblur=luma_radius=13:luma_power=1:chroma_radius=7:chroma_power=1" in graph
    with pytest.raises(ValueError, match="bounds"):
        compile_layer(source_size=(1920, 1080), destination=PixelRect(0, 0, 10, 10),
                      fill=FillMode.CONTAIN_BLUR, blur_sigma=12)


def test_source_overflow_is_checked_before_clipping_normalized_destination() -> None:
    rect = NormalizedRect(x=.5, y=0, width=.50000075, height=1)
    with pytest.raises(ValueError, match="source"):
        source_crop_pixels(rect, (2000000, 1080))
    assert source_crop_pixels(None, (1920, 1080)) == PixelRect(0, 0, 1920, 1080)


def test_contained_fit_must_be_representable_in_even_pixels() -> None:
    with pytest.raises(ValueError, match="[Cc]ontain"):
        compile_layer(source_size=(2000000, 2), destination=PixelRect(0, 0, 2, 2),
                      fill=FillMode.BLACK)


def test_graph_duration_preserves_fractional_precision() -> None:
    graph = compile_layer(source_size=(1920, 1080), destination=PixelRect(0, 0, 1080, 960),
                          fill=FillMode.BLACK, duration=1.123456789)
    assert "d=1.123456789" in graph


def test_explicit_source_crop_precedes_fill_scaling() -> None:
    graph = compile_layer(source_size=(1920, 1080), source_crop=PixelRect(100, 50, 800, 600),
                          destination=PixelRect(0, 0, 1080, 960), fill=FillMode.COVER)
    assert "crop=w=800:h=600:x=100:y=50:exact=1,scale=" in graph


def test_solid_fill_requires_solid_kind_and_safe_explicit_color() -> None:
    graph = compile_layer(source_size=(1920, 1080), destination=PixelRect(0, 0, 1080, 960),
                          fill=FillMode.SOLID, kind=LayerKind.SOLID, solid_color="#12abEF")
    assert "color=c=0x12abEF" in graph
    assert "[in]" not in graph
    for kind, fill, color in [(LayerKind.SOURCE_VIDEO, FillMode.SOLID, "#112233"),
                              (LayerKind.SOLID, FillMode.COVER, None),
                              (LayerKind.SOLID, FillMode.SOLID, "black;movie=evil")]:
        with pytest.raises(ValueError):
            compile_layer(source_size=(1920, 1080), destination=PixelRect(0, 0, 1080, 960),
                          fill=fill, kind=kind, solid_color=color)


@pytest.mark.parametrize("fill,sigma", [(FillMode.INHERIT, None),
    (FillMode.CONTAIN_BLUR, None), (FillMode.CONTAIN_BLUR, float("nan"))])
def test_unresolved_or_invalid_fill_fails(fill: FillMode, sigma: float | None) -> None:
    with pytest.raises(ValueError):
        compile_layer(source_size=(1920, 1080), destination=PixelRect(0, 0, 1080, 960),
                      fill=fill, blur_sigma=sigma)


def test_filter_labels_cannot_inject_graph_syntax() -> None:
    with pytest.raises(ValueError, match="label"):
        compile_layer(source_size=(1920, 1080), destination=PixelRect(0, 0, 1080, 960),
                      fill=FillMode.COVER, input_label="in];movie=evil")


@pytest.mark.parametrize("fill", [FillMode.COVER, FillMode.BLACK, FillMode.CONTAIN_BLUR])
def test_dynamic_transform_uses_native_independent_crop_and_fixed_policy(fill: FillMode) -> None:
    track = compile_crop_track([CropKeyframe(at=0, crop=NormalizedRect(
        x=.1, y=.1, width=.8, height=.6)), CropKeyframe(at=2, crop=NormalizedRect(
        x=.2, y=.2, width=.5, height=.7))], source_size=(200, 200))
    graph = compile_motion_layer(track, destination=PixelRect(0, 0, 48, 48), fill=fill,
                                  blur_sigma=2, duration=2)
    assert "libplacebo=w=48:h=48" in graph and "(n)" in graph
    assert all(f"crop_{axis}=" in graph for axis in ("x", "y", "w", "h"))
    assert "pos_w='cw*" in graph and "pos_x='(ow-pw)/2'" in graph
    assert "upscaler=ewa_lanczos:downscaler=ewa_lanczos" in graph
    assert "disable_linear=0:dithering=ordered_fixed:dither_temporal=0" in graph
    assert "fps=none" in graph and "frame_mixer=none" in graph
    assert "geq=" not in graph and "perspective=" not in graph
    assert ("fillcolor=black@0" in graph) == (fill == FillMode.CONTAIN_BLUR)
    assert ("boxblur=" in graph) == (fill == FillMode.CONTAIN_BLUR)
    assert "zoompan" not in graph


def test_single_motion_keyframe_reuses_existing_static_path() -> None:
    crop = NormalizedRect(x=.1, y=.1, width=.4, height=.4)
    track = compile_crop_track([CropKeyframe(at=0, crop=crop)], source_size=(1920, 1080))
    destination = PixelRect(0, 0, 1080, 960)
    assert compile_motion_layer(track, destination=destination, fill=FillMode.COVER) == (
        compile_layer(source_size=(1920, 1080), destination=destination,
                      source_crop=source_crop_pixels(crop, (1920, 1080)), fill=FillMode.COVER))


def test_dynamic_contained_fit_must_remain_representable() -> None:
    full = NormalizedRect(x=0, y=0, width=1, height=1)
    track = compile_crop_track([CropKeyframe(at=0, crop=full), CropKeyframe(at=2, crop=full)],
                               source_size=(2000000, 2))
    with pytest.raises(ValueError, match="[Cc]ontain"):
        compile_motion_layer(track, destination=PixelRect(0, 0, 2, 2), fill=FillMode.BLACK)


def host_motion_shot() -> VisualShot:
    return VisualShot.model_validate(dict(shot_id="HOST-shot", source=dict(start=20.125, end=22.125),
        output=dict(start=3.5, end=5.5), primary_source="content", reason="HOST synthetic test",
        evidence_ids=["HOST"], confidence=1, layers=[dict(layer_id="HOST-layer];evil",
        kind="source_video", source_ref="source", z_index=0, fill="cover",
        destination=dict(x=0, y=0, width=1, height=1), crop_track=[
            dict(at=0, crop=dict(x=0, y=0, width=.8, height=.8)),
            dict(at=2, crop=dict(x=.2, y=.2, width=.8, height=.8))])]))


def test_shot_motion_dispatch_retains_source_output_anchors_and_normalizes_before_split() -> None:
    shot = host_motion_shot()
    before = shot.model_dump_json()
    result = compile_shot_layers(shot, source_size=(1920, 1080))
    assert "trim=start=20.125:end=22.125,setpts=PTS-20.125/TB,fps=60:start_time=0:round=near" in (
        result.filter_complex)
    assert "libplacebo=" in result.filter_complex
    assert result.requirements.needs_vulkan
    assert result.requirements.runtime_id == "ffmpeg-9.0.2-libplacebo"
    assert result.source_start == 20.125 and result.source_end == 22.125
    assert result.output_start == 3.5 and result.output_end == 5.5
    assert "evil" not in result.filter_complex
    assert shot.model_dump_json() == before


def test_direct_shot_call_revalidates_mutations_and_static_fill_preservation() -> None:
    shot = host_motion_shot()
    shot.layers[0].crop_track[1].at = 3
    with pytest.raises(ValueError):
        compile_shot_layers(shot, source_size=(1920, 1080))
    shot = host_motion_shot()
    shot.layers[0].crop_track = []
    shot.layers[0].preserve_regions = [NormalizedRect(x=0, y=.1, width=.1, height=.1)]
    with pytest.raises(ValueError, match="HOST-shot.*HOST-layer.*keyframe"):
        compile_shot_layers(shot, source_size=(1920, 1080))


@pytest.mark.parametrize("mutation", ["provenance", "opacity", "unknown", "nonfinite"])
def test_direct_shot_rejects_unsupported_and_nonfinite_mutations(mutation: str) -> None:
    shot = host_motion_shot()
    if mutation == "provenance":
        shot.layers[0].source_ref = "different-source"
    elif mutation == "opacity":
        shot.layers[0].opacity = .5
    elif mutation == "nonfinite":
        shot.layers[0].crop_track[0].at = float("nan")
    else:
        shot.layers[0].__dict__["unrequested_filter"] = "movie=untrusted"
    with pytest.raises(ValueError):
        compile_shot_layers(shot, source_size=(1920, 1080))


def test_independent_source_primitive_preserves_motion_and_finite_generated_clock() -> None:
    from clipperx import filtergraph as fg
    from clipperx.output_timeline import OutputTimeline
    shot = host_motion_shot()
    shot.output.start, shot.output.end = 0, 2
    span = OutputTimeline.from_shots([shot]).raster_spans()[0]
    primitive = fg.compile_source_layer(shot, layer_index=0, raster_span=span,
        source_size=(1920, 1080), input_label="scheduled")
    assert primitive.layer_id == shot.layers[0].layer_id
    assert primitive.declaration_index == 0 and primitive.input_label == "scheduled"
    assert primitive.crop_track is not None and primitive.crop_track.points[-1].frame == 120
    assert "libplacebo=" in primitive.filter_complex and "(n)" in primitive.filter_complex
    assert "trim=start=" not in primitive.filter_complex
    shot.layers[0].kind, shot.layers[0].fill = LayerKind.SOLID, FillMode.SOLID
    shot.layers[0].solid_color, shot.layers[0].crop_track = "#112233", []
    solid = fg.compile_source_layer(shot, layer_index=0, raster_span=span,
        source_size=(1920, 1080), input_label=None)
    assert solid.crop_track is None and solid.input_label is None
    assert "trim=end_frame=120,settb=1/60,setpts=N" in solid.filter_complex
    assert "d=" not in solid.filter_complex
    for index in (True, -1, 1):
        with pytest.raises(ValueError):
            fg.compile_source_layer(shot, layer_index=index, raster_span=span,
                source_size=(1920, 1080), input_label=None)
    with pytest.raises(ValueError):
        fg.compile_source_layer(shot, layer_index=0, raster_span=span,
            source_size=(1920, 1080), input_label="unneeded")
    shot.source.start, shot.source.end = 20, 22
    with pytest.raises(ValueError, match="span"):
        fg.compile_source_layer(shot, layer_index=0, raster_span=span,
            source_size=(1920, 1080), input_label=None)


def test_finite_black_static_delegate_and_backward_default() -> None:
    from clipperx import filtergraph as fg
    destination = PixelRect(0, 0, 48, 48)
    old = compile_layer(source_size=(200, 200), destination=destination, fill=FillMode.BLACK)
    assert old == compile_layer(source_size=(200, 200), destination=destination,
                               fill=FillMode.BLACK, frame_count=None)
    track = compile_crop_track([CropKeyframe(at=0, crop=NormalizedRect(
        x=0, y=0, width=1, height=1))], source_size=(200, 200))
    finite = fg.compile_motion_layer(track, destination=destination, fill=FillMode.BLACK,
                                    frame_count=7)
    assert "trim=end_frame=7,settb=1/60,setpts=N" in finite and ":d=" not in finite
    for count in (True, 0, -1):
        with pytest.raises(ValueError):
            compile_layer(source_size=(200, 200), destination=destination,
                          fill=FillMode.BLACK, frame_count=count)


@pytest.mark.parametrize("fault", ["float_coercion", "generated_budget", "input_label", "count_bits"])
def test_source_primitives_guard_before_clone_or_expression(
        fault: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from clipperx import filtergraph as fg
    from clipperx.output_timeline import OutputTimeline
    shot = host_motion_shot()
    shot.output.start, shot.output.end = 0, 2
    span = OutputTimeline.from_shots([shot]).raster_spans()[0]
    if fault == "count_bits":
        with pytest.raises(ValueError):
            fg.compile_layer(source_size=(2, 2), destination=PixelRect(0, 0, 2, 2),
                fill=FillMode.SOLID, kind=LayerKind.SOLID, solid_color="#112233",
                frame_count=1 << 4097)
        return
    if fault == "float_coercion":
        shot.source.__dict__["start"] = "20.125"
    elif fault == "generated_budget":
        monkeypatch.setattr(fg, "MAX_GENERATED_BYTES", 100)
    elif fault == "input_label":
        monkeypatch.setattr(fg, "MAX_INPUT_TEXT_BYTES", 10)
    def forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("Source guard must precede cloning/expression work")
    monkeypatch.setattr(fg, "_strict_model_input", forbidden)
    with pytest.raises(ValueError):
        fg.compile_source_layer(shot, layer_index=0, raster_span=span, source_size=(2, 2),
                                input_label="a_very_large_scheduled_input_label")


@pytest.mark.parametrize("fault", ["boolean_label", "unrepresentable_size"])
def test_source_primitive_rejects_unrepresentable_host_scalars_as_value_error(fault: str) -> None:
    from clipperx import filtergraph as fg
    from clipperx.output_timeline import OutputTimeline
    shot = host_motion_shot()
    shot.output.start, shot.output.end = 0, 2
    span = OutputTimeline.from_shots([shot]).raster_spans()[0]
    shot.layers[0].crop_track = []
    # Use the raw host boundary deliberately; annotations alone cannot reject bool.
    kwargs = dict(layer_index=0, raster_span=span,
        source_size=(1 << 4000, 2) if fault == "unrepresentable_size" else (2, 2),
        input_label=False if fault == "boolean_label" else "scheduled")
    from typing import Any
    host_arguments: Any = kwargs
    with pytest.raises(ValueError):
        fg.compile_source_layer(shot, **host_arguments)


@pytest.mark.parametrize("forbidden_name", ["_strict_model_input", "compile_crop_track"])
def test_source_primitive_reserves_emitted_label_before_clone_and_geometry(
        forbidden_name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from clipperx import filtergraph as fg
    from clipperx.output_timeline import OutputTimeline
    shot = host_motion_shot()
    shot.output.start, shot.output.end = 0, 2
    shot.layers[0].crop_track = []
    span = OutputTimeline.from_shots([shot]).raster_spans()[0]
    monkeypatch.setattr(fg, "MAX_GENERATED_BYTES", 10000)
    short = fg.compile_source_layer(shot, layer_index=0, raster_span=span,
        source_size=(1920, 1080), input_label="scheduled")
    assert "[scheduled]" in short.filter_complex
    assert len(short.filter_complex.encode()) < 10000

    def forbidden(*args: object, **kwargs: object) -> object:
        pytest.fail("Oversized emitted label reached cloning/geometry")

    monkeypatch.setattr(fg, forbidden_name, forbidden)
    with pytest.raises(ValueError, match="byte budget"):
        fg.compile_source_layer(shot, layer_index=0, raster_span=span,
            source_size=(1920, 1080), input_label="a" * 12000)
