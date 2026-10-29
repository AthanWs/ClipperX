"""HOST-authored numerical and synthetic FFmpeg motion checks; no source media."""

from dataclasses import FrozenInstanceError
import json
import math
from pathlib import Path
import subprocess
import statistics
import time

import pytest

from clipperx.filtergraph import PixelRect
from clipperx.motion_compiler import (
    FloatRect, compile_crop_track, smooth_crop_observations,
)
from clipperx.visual_contracts import CropKeyframe, FillMode, NormalizedRect


def keyframe(at: float, crop: tuple[float, float, float, float],
             easing: str = "linear") -> CropKeyframe:
    return CropKeyframe.model_validate(dict(at=at, crop=dict(zip(
        ("x", "y", "width", "height"), crop, strict=True)), easing=easing))


def test_single_keyframe_compiles_to_truthful_static_crop() -> None:
    result = compile_crop_track([keyframe(0, (.101, .101, .401, .401))],
                                source_size=(1920, 1080), fps=60)
    assert result.mode == "static"
    assert result.sample(0) == FloatRect(194, 110, 770, 432)


@pytest.mark.parametrize("easing", ["linear", "ease_in_out"])
def test_motion_reaches_both_endpoints_without_speeding_up(easing: str) -> None:
    track = [keyframe(0, (0, 0, .5, .5), easing),
             keyframe(2, (.5, .5, .5, .5))]
    result = compile_crop_track(track, source_size=(1920, 1080), fps=60)
    assert result.sample(0) == FloatRect(0, 0, 960, 540)
    assert result.sample(120) == FloatRect(960, 540, 960, 540)
    assert result.sample(60) == FloatRect(480, 270, 960, 540)
    assert result.sample(119).x < 960
    track[0].crop.x = .25
    assert result.sample(0).x == 0
    with pytest.raises(FrozenInstanceError):
        setattr(result.points[0].rect, "x", 3)


@pytest.mark.parametrize("moving", [False, True])
def test_source_size_is_detached_from_mutable_caller_for_geometry_and_preservation(
        moving: bool) -> None:
    size = [200, 200]
    frames = [keyframe(0, (0, 0, 1, 1))]
    if moving:
        frames.append(keyframe(2, (0, 0, 1, 1)))
    track = compile_crop_track(frames, source_size=size)
    region = NormalizedRect(x=.6, y=.2, width=.2, height=.2)
    samples = tuple(track.sample(frame) for frame in (0, 60, 120))
    expressions = track.expressions("n")
    track.validate_preservation([region], destination=(200, 200), fill=FillMode.BLACK,
                                 shot_id="detached", layer_id="source")
    size[:] = [400, 100]
    track.validate_preservation([region], destination=(200, 200), fill=FillMode.BLACK,
                                 shot_id="detached", layer_id="source")
    assert track.source_size == (200, 200)
    assert track.source_size is not size
    assert tuple(track.sample(frame) for frame in (0, 60, 120)) == samples
    assert track.expressions("n") == expressions


def test_piecewise_fractional_frame_times_and_departing_easing() -> None:
    result = compile_crop_track([keyframe(0, (0, 0, .5, .5), "ease_in_out"),
        keyframe(.025, (.1, .1, .6, .7)), keyframe(.1, (.2, .2, .7, .8))],
        source_size=(100, 100))
    assert result.points[1].frame == 1.5
    assert result.sample(.75) == FloatRect(5, 5, 55, 60)
    assert result.sample_time(.025) == FloatRect(10, 10, 60, 70)
    assert result.sample(3.75) == FloatRect(15, 15, 65, 75)
    assert result.sample(-1) == result.sample(0)
    assert result.sample(999) == result.sample(6)


def test_native_zero_based_clock_serializes_same_immutable_geometry() -> None:
    result = compile_crop_track([keyframe(0, (0, 0, .5, .5), "ease_in_out"),
        keyframe(.025, (.1, .1, .6, .7)), keyframe(2, (.2, .2, .7, .8))],
        source_size=(200, 200))
    native = result.expressions("n")
    assert native == tuple(expression.replace("(N)", "(n)") for expression in result.expressions("N"))
    assert result.sample(120) == FloatRect(40, 40, 140, 160)


@pytest.mark.parametrize("easing", ["linear", "ease_in_out"])
def test_geometry_preserves_aspect_crossing_for_region_validation(easing: str) -> None:
    result = compile_crop_track([keyframe(0, (0, 0, .8, .2), easing),
        keyframe(2, (0, 0, .2, .8))], source_size=(500, 500))
    assert result.crossing_frames((100, 100)) == pytest.approx((60,))
    assert result.sample(result.crossing_frames((100, 100))[0]) == FloatRect(0, 0, 250, 250)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1])
def test_invalid_times_and_mutated_rectangles_fail(bad: float) -> None:
    first = keyframe(0, (0, 0, .5, .5))
    first.at = bad
    with pytest.raises(ValueError):
        compile_crop_track([first], source_size=(1920, 1080))


def test_invalid_tracks_and_subpixel_collapse_fail() -> None:
    for track in ([], [keyframe(1, (0, 0, .5, .5)), keyframe(2, (0, 0, .5, .5))],
                  [keyframe(0, (0, 0, .5, .5)), keyframe(0, (0, 0, .5, .5))],
                  [keyframe(0, (0, 0, .001, .5)), keyframe(1, (0, 0, .5, .5))]):
        with pytest.raises(ValueError):
            compile_crop_track(track, source_size=(100, 100))
    with pytest.raises(ValueError):
        compile_crop_track([keyframe(0, (0, 0, .5, .5))],
                           source_size=(100, 100), fps=30)


def test_preserve_regions_through_crop_and_cover_with_ids() -> None:
    track = compile_crop_track([keyframe(0, (0, 0, 1, 1)),
        keyframe(2, (0, 0, 1, .5))], source_size=(200, 100))
    region = NormalizedRect(x=.01, y=.01, width=.1, height=.1)
    with pytest.raises(ValueError, match="shot=shot-A layer=layer-B keyframe="):
        track.validate_preservation([region], destination=(100, 100),
                                    fill=FillMode.COVER, shot_id="shot-A", layer_id="layer-B")
    track.validate_preservation([region], destination=(100, 100),
                                 fill=FillMode.BLACK, shot_id="shot-A", layer_id="layer-B")


def test_smoothing_never_discards_deliberate_move_or_endpoints() -> None:
    observations = [PixelRect(x, y, 500, 400) for x, y in
                    [(100, 100), (101, 99), (100, 101), (400, 100), (700, 100)]]
    assert smooth_crop_observations(observations, source_size=(1920, 1080)) == tuple(
        FloatRect(r.x, r.y, r.width, r.height) for r in observations)
    jitter = [PixelRect(x, 100, 500, 400) for x in [100, 102, 99, 101, 100]]
    smoothed = smooth_crop_observations(jitter, source_size=(1920, 1080))
    assert smoothed[0].x == smoothed[-1].x == 100
    assert smoothed != tuple(FloatRect(r.x, r.y, r.width, r.height) for r in jitter)
    assert all(math.isfinite(r.x) for r in smoothed)


def test_sg_projection_matches_quadratic_fit_and_correction_limits() -> None:
    values = [100, 102, 99, 101, 100, 98, 101]
    observations = [PixelRect(x, 100, 500, 400) for x in values]
    result = smooth_crop_observations(observations, source_size=(1920, 1080))
    expected = [100, 3520/35, 3519/35, 3505/35, 3488/35, 3494/35, 101]
    assert [r.x for r in result] == pytest.approx(expected)
    assert result[0].x == 100 and result[-1].x == 101
    assert smooth_crop_observations(observations[:4], source_size=(1920, 1080)) == tuple(
        FloatRect(r.x, r.y, r.width, r.height) for r in observations[:4])
    # Any width/height-correction over its own 1.5% bound rejects the entire
    # smoothing proposal; intentional coordinates and endpoints stay authored.
    moves = [PixelRect(100, 100, w, 400) for w in (500, 500, 500, 800, 900)]
    assert smooth_crop_observations(moves, source_size=(1920, 1080)) == tuple(
        FloatRect(r.x, r.y, r.width, r.height) for r in moves)


def test_sg_strict_correction_threshold_and_source_bounds_are_not_relaxed() -> None:
    for deviation, allowed in ((34, True), (35, False)):
        observations = [PixelRect(x, 100, 500, 400)
                        for x in (100, 100, 100+deviation, 100, 100)]
        original = tuple(FloatRect(r.x, r.y, r.width, r.height) for r in observations)
        result = smooth_crop_observations(observations, source_size=(1200, 1000))
        assert (result != original) == allowed
    border = [PixelRect(x, 100, 500, 400) for x in (0, 0, 0, 1, 2)]
    assert smooth_crop_observations(border, source_size=(1200, 1000)) == tuple(
        FloatRect(r.x, r.y, r.width, r.height) for r in border)


@pytest.mark.parametrize("field", ["at", "x", "width"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_mutated_numeric_boundaries_are_revalidated(field: str, value: float) -> None:
    first = keyframe(0, (.1, .1, .5, .5))
    if field == "at":
        first.at = value
    else:
        setattr(first.crop, field, value)
    with pytest.raises(ValueError):
        compile_crop_track([first, keyframe(2, (.2, .2, .5, .5))], source_size=(200, 200))


def test_mutated_finite_rectangles_easing_and_sample_clocks_fail() -> None:
    first = keyframe(0, (0, 0, .5, .5))
    first.crop.width = 1.1
    with pytest.raises(ValueError):
        compile_crop_track([first], source_size=(200, 200))
    first = keyframe(0, (0, 0, .5, .5))
    setattr(first, "easing", "other")
    with pytest.raises(ValueError):
        compile_crop_track([first], source_size=(200, 200))
    track = compile_crop_track([keyframe(0, (0, 0, .5, .5))], source_size=(200, 200))
    for frame in (float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite"):
            track.sample(frame)
    with pytest.raises(ValueError, match="clock"):
        track.expressions("N);movie=other")
    first = keyframe(0, (0, 0, .5, .5))
    first.crop.__dict__["unrequested_axis"] = "X+10"
    with pytest.raises(ValueError, match="Unrequested"):
        compile_crop_track([first], source_size=(200, 200))


def test_preservation_has_exact_two_source_pixel_budget_and_detached_inputs() -> None:
    track = compile_crop_track([keyframe(0, (.1, .1, .8, .8)),
        keyframe(2, (.1, .1, .8, .8))], source_size=(200, 200))
    region = NormalizedRect(x=.09, y=.09, width=.82, height=.82)
    before = region.model_dump_json()
    track.validate_preservation([region], destination=(100, 100), fill=FillMode.BLACK,
                                 shot_id="s", layer_id="l")
    assert region.model_dump_json() == before
    region.x = .0899
    with pytest.raises(ValueError, match="shot=s layer=l keyframe=0"):
        track.validate_preservation([region], destination=(100, 100), fill=FillMode.BLACK,
                                     shot_id="s", layer_id="l")


def test_thin_cover_visible_area_uses_physical_tolerance_without_crop_growth() -> None:
    track = compile_crop_track([keyframe(0, (0, 0, 1, 1)), keyframe(2, (0, 0, 1, 1))],
                               source_size=(200, 200))
    region = NormalizedRect(x=.4975, y=.4975, width=.005, height=.005)
    track.validate_preservation([region], destination=(2, 1920), fill=FillMode.COVER,
                                 shot_id="thin", layer_id="original")
    assert track.sample(60) == FloatRect(0, 0, 200, 200)


def test_static_preservation_error_before_lone_point_keeps_required_identity() -> None:
    track = compile_crop_track([keyframe(1, (0, 0, 1, 1))], source_size=(1920, 1080))
    region = NormalizedRect(x=.01, y=.1, width=.02, height=.02)
    with pytest.raises(ValueError, match="shot=static layer=source keyframe=0"):
        track.validate_preservation([region], destination=(100, 100), fill=FillMode.COVER,
                                     shot_id="static", layer_id="source")


def render_graph(tmp_path: Path, graph: str, *, size: tuple[int, int], frames: int,
                 source: str | None = None, rate: int = 60, null: bool = False,
                 threads: int = 1, copyts: bool = False,
                 pixel_format: str = "yuv444p", timeout_seconds: int = 45) -> bytes:
    """Bounded HOST diagnostic; no encoded product/real media, no Python image rendering."""
    from clipperx.render_runtime import RenderRequirements, resolve_render_runtime

    native = "libplacebo=" in graph
    selected = resolve_render_runtime(RenderRequirements("ffmpeg-9.0.2-libplacebo", native))
    number = len(list(tmp_path.glob("diagnostic-*.json")))
    graph_file = tmp_path / f"diagnostic-{number}.ffgraph"
    graph_file.write_text(graph, encoding="utf-8")
    command = [str(selected.ffmpeg), "-hide_banner", "-loglevel", "info",
               "-filter_complex_threads", str(threads)]
    command.extend(selected.device_arguments)
    if copyts:
        command.append("-copyts")
    command += ["-f", "lavfi", "-i", source or f"testsrc2=s={size[0]}x{size[1]}:r={rate}",
               selected.graph_file_option, str(graph_file),
               "-map", "[out]",
               "-frames:v", str(frames), "-an", "-threads", "1"]
    command += (["-f", "null", "-"] if null else
                ["-pix_fmt", pixel_format, "-f", "rawvideo", "pipe:1"])
    started = time.perf_counter()
    try:
        result = subprocess.run(command, capture_output=True, timeout=timeout_seconds, check=False)
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter()-started
        (tmp_path / f"diagnostic-{number}.json").write_text(json.dumps(dict(
            command=command, elapsed=elapsed, timeout=True, stdout_bytes=len(exc.stdout or b""),
            stderr=(exc.stderr or b"").decode(errors="replace"))), encoding="utf-8")
        raise
    elapsed = time.perf_counter()-started
    print(f"diagnostic frames={frames} elapsed={elapsed:.3f}s exit={result.returncode}")
    (tmp_path / f"diagnostic-{number}.json").write_text(json.dumps(dict(
        command=command, elapsed=elapsed, exit=result.returncode, stdout_bytes=len(result.stdout),
        stderr=result.stderr.decode(errors="replace"))), encoding="utf-8")
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    return result.stdout


@pytest.mark.parametrize("native", [False, True])
def test_render_graph_uses_selected_runtime_and_file_option(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, native: bool) -> None:
    from clipperx import render_runtime

    graph = "[0:v]format=rgb24[out]" if not native else "[0:v]libplacebo=w=2:h=2[out]"
    ffmpeg = tmp_path / "pinned-ffmpeg.exe"
    ffprobe = tmp_path / "pinned-ffprobe.exe"
    selected = render_runtime.RenderRuntime(
        ffmpeg=ffmpeg, ffprobe=ffprobe, runtime_id="ffmpeg-9.0.2-libplacebo",
        ffmpeg_sha256="fake-ffmpeg-hash", ffprobe_sha256="fake-ffprobe-hash",
        expected_ffmpeg_sha256=None, expected_ffprobe_sha256=None,
        graph_file_option="-/filter_complex", device_index=0 if native else None)
    requirements_seen: list[render_runtime.RenderRequirements] = []

    def resolve(requirements: render_runtime.RenderRequirements) -> render_runtime.RenderRuntime:
        requirements_seen.append(requirements)
        return selected

    invocations: list[tuple[list[str], dict[str, object]]] = []
    stdout = b"synthetic diagnostic bytes"
    stderr = b"synthetic diagnostic stderr"

    def run(args: list[str], *, capture_output: bool, timeout: float,
            check: bool) -> subprocess.CompletedProcess[bytes]:
        invocations.append((args, dict(capture_output=capture_output, timeout=timeout, check=check)))
        return subprocess.CompletedProcess(args, 0, stdout, stderr)

    monkeypatch.setattr(render_runtime, "resolve_render_runtime", resolve)
    monkeypatch.setattr(subprocess, "run", run)

    result = render_graph(tmp_path, graph, size=(2, 2), frames=3, timeout_seconds=17)

    assert len(requirements_seen) == 1
    assert requirements_seen[0].runtime_id == "ffmpeg-9.0.2-libplacebo"
    assert requirements_seen[0].needs_vulkan is native
    assert len(invocations) == 1
    command, run_options = invocations[0]
    graph_file = tmp_path / "diagnostic-0.ffgraph"
    assert graph_file.read_text(encoding="utf-8") == graph
    assert command[0] == str(ffmpeg)
    assert command[command.index("-/filter_complex") + 1] == str(graph_file)
    assert ("-init_hw_device" in command) is native
    assert ("-filter_hw_device" in command) is native
    assert command[command.index("-frames:v") + 1] == "3"
    assert command[-5:] == ["-pix_fmt", "yuv444p", "-f", "rawvideo", "pipe:1"]
    assert run_options == {"capture_output": True, "timeout": 17, "check": False}
    diagnostic = json.loads((tmp_path / "diagnostic-0.json").read_text(encoding="utf-8"))
    assert diagnostic["exit"] == 0
    assert diagnostic["stdout_bytes"] == len(stdout)
    assert diagnostic["stderr"] == stderr.decode()
    assert result == stdout


def test_1080_multilayer_feasibility(tmp_path: Path) -> None:
    from clipperx.filtergraph import compile_motion_layer

    track = compile_crop_track([keyframe(0, (.01, .05, .95, .9)),
        keyframe(2, (.03, .1, .8, .8), "ease_in_out")], source_size=(1920, 1080))
    camera = compile_crop_track([keyframe(0, (.3, .1, .25, .5)),
        keyframe(2, (.4, .15, .2, .5))], source_size=(1920, 1080))
    graph = "[0:v]split=2[main][camera];" + compile_motion_layer(track,
        destination=PixelRect(0, 0, 1080, 1920), fill=FillMode.CONTAIN_BLUR,
        blur_sigma=12, input_label="main", output_label="mainout", duration=2)
    graph += ";" + compile_motion_layer(camera, destination=PixelRect(0, 0, 540, 960),
        fill=FillMode.COVER, input_label="camera", output_label="camout", duration=2)
    graph += ";[mainout][camout]overlay=540:960:shortest=1:format=rgb:alpha=straight,setsar=1,format=yuv420p,showinfo[out]"
    assert "libplacebo=w=1080:h=1920" in graph and "geq=" not in graph
    render_graph(tmp_path, graph, size=(1920, 1080), frames=3, null=True, threads=4)




@pytest.mark.parametrize("fill", [FillMode.COVER, FillMode.BLACK, FillMode.CONTAIN_BLUR])
@pytest.mark.parametrize("easing", ["linear", "ease_in_out"])
@pytest.mark.parametrize("aspect", [True, False])
def test_decoded_native_piecewise_tracks_match_frozen_and_independent_xy(
        tmp_path: Path, fill: FillMode, easing: str, aspect: bool) -> None:
    from clipperx.filtergraph import compile_motion_layer

    crops = ((.05, .05, .8, .35), (.08, .1, .6, .65), (.2, .02, .45, .85)) if aspect else (
        (.075, .1, .75, .6), (.15, .2, .6, .48), (.475, .475, .45, .36))
    track = compile_crop_track([keyframe(0, crops[0], easing), keyframe(.7, crops[1], easing),
                                keyframe(2, crops[2])], source_size=(200, 200))
    ratios = tuple(point.rect.width/point.rect.height for point in track.points)
    if aspect:
        assert max(ratios) > min(ratios)
        assert track.crossing_frames((200, 200))
    else:
        assert ratios == pytest.approx((1.25, 1.25, 1.25))
        assert all(track.sample(frame).width/track.sample(frame).height == pytest.approx(1.25)
                   for frame in range(121))
    # Full121-frame timing/geometry uses an upscale raster. Strong reduction
    # has separate all-fill analytical detail/color oracles; changing minifier
    # ratios on every frame exceeded the45s process bound (archived pytest190).
    source = rgb_ramp()
    graph = compile_motion_layer(track, destination=PixelRect(0, 0, 200, 200), fill=fill,
                                 blur_sigma=2, input_label="0:v", duration=3)
    actual = render_graph(tmp_path, graph, size=(200, 200), source=source,
                           frames=121, pixel_format="rgba")
    checkpoints = (0, 1, 2, 21, 42, 60, 90, 119, 120)
    parts = ["[0:v]split=9" + "".join(f"[ref{i}]" for i in range(9))]
    for i, frame in enumerate(checkpoints):
        frozen = compile_motion_layer(track, destination=PixelRect(0, 0, 200, 200), fill=fill,
            blur_sigma=2, input_label=f"ref{i}", output_label=f"frozen{i}", duration=3)
        rect = track.sample(frame)
        for expression, value in zip(track.expressions("n"), rect.values, strict=True):
            frozen = frozen.replace(expression, repr(value))
        parts.append(frozen)
    parts.append("".join(f"[frozen{i}]" for i in range(9)) + "hstack=inputs=9[out]")
    expected = render_graph(tmp_path, ";".join(parts), size=(200, 200), source=source,
                             frames=1, pixel_format="rgba")
    assert len(actual) == 121*200*200*4 and len(expected) == 9*200*200*4
    for i, frame in enumerate(checkpoints):
        offset = frame*200*200*4
        for row in range(200):
            start = (row*1800+i*200)*4
            a = offset+row*200*4
            assert actual[a:a+200*4] == expected[start:start+200*4], (frame, row)
        rect = track.sample(frame)
        s = (max if fill == FillMode.COVER else min)(200/rect.width, 200/rect.height)
        dx, dy = (200-rect.width*s)/2, (200-rect.height*s)/2
        for q, r in ((80, 80), (100, 100), (120, 120)):
            sx = rect.x+(q+.5-dx)/s-.5
            sy = rect.y+(r+.5-dy)/s-.5
            for channel, oracle in enumerate((16+sx, 16+sy, 128)):
                assert abs(actual[offset+(r*200+q)*4+channel]-oracle) <= 1.2, (
                    frame, channel, q, r, oracle)


def rgb_ramp(rate: int = 60, size: tuple[int, int] = (200, 200),
             transfer: str = "linear") -> str:
    return (f"nullsrc=s={size[0]}x{size[1]}:r={rate},format=gbrp,"
            "geq=r='16+X':g='16+Y':b=128,setparams=range=full:"
            f"color_primaries=bt709:color_trc={transfer}:colorspace=gbr")


@pytest.mark.parametrize("fill", [FillMode.COVER, FillMode.BLACK, FillMode.CONTAIN_BLUR])
def test_native_fractional_edges_do_not_reveal_unrelated_source(tmp_path: Path,
                                                              fill: FillMode) -> None:
    from clipperx.filtergraph import compile_motion_layer

    crop = (16.25/192, 16.25/144, 95.5/192, 47.5/144)
    track = compile_crop_track([keyframe(0, crop), keyframe(2, crop)], source_size=(192, 144))
    inside = "gte(X,16)*lt(X,112)*gte(Y,16)*lt(Y,64)"
    source = (f"nullsrc=s=192x144:r=60,format=gbrp,geq=r='if({inside},200,16)':"
              f"g='if({inside},128,32)':b='if({inside},64,200)',setparams=range=full:"
              "color_primaries=bt709:color_trc=linear:colorspace=gbr")
    graph = compile_motion_layer(track, destination=PixelRect(0, 0, 96, 96), fill=fill,
                                 blur_sigma=2, input_label="0:v", duration=1)
    data = render_graph(tmp_path, graph, size=(192, 144), source=source,
                         frames=3, pixel_format="rgba")
    stride = 96*96*4
    assert data[:stride] == data[stride:2*stride] == data[2*stride:]
    rows = range(96) if fill == FillMode.COVER else range(30, 66)
    for row in rows:
        for col in range(96):
            offset = (row*96+col)*4
            assert all(abs(data[offset+c]-value) <= 1 for c, value in enumerate((200, 128, 64)))
            assert data[offset+3] == 255
    if fill == FillMode.BLACK:
        # Exact native black replaces the retired CPU16-vs17 assertion with an
        # equally strict RGB oracle, not a relaxed unexplained tolerance.
        for row in (0, 1, 12, 83, 94, 95):
            assert data[(row*96+48)*4:(row*96+48)*4+4] == bytes((0, 0, 0, 255))


def test_native_transparent_alpha_and_changed_fractional_placement(tmp_path: Path) -> None:
    from clipperx.filtergraph import _motion_transform

    track = compile_crop_track([keyframe(0, (0, 0, 1, .50125)),
        keyframe(2/60, (.00125, .00125, .49875, .99875))], source_size=(200, 200))
    graph = _motion_transform(track, PixelRect(0, 0, 256, 320), cover=False,
                               transparent=True, input_label="0:v", output_label="out")
    source = ("nullsrc=s=200x200:r=60,format=gbrp,geq=r=200:g=128:b=64,"
              "setparams=range=full:color_primaries=bt709:color_trc=linear:colorspace=gbr")
    raw = render_graph(tmp_path, graph, size=(200, 200), source=source,
                        frames=3, pixel_format="rgba")
    for frame in range(3):
        r = track.sample(frame)
        scale = min(256/r.width, 320/r.height)
        dx, dy = (256-r.width*scale)/2, (320-r.height*scale)/2
        base = frame*256*320*4
        center = base+(160*256+128)*4
        assert raw[center+3] == 255
        assert all(abs(raw[center+c]-value) <= 1 for c, value in enumerate((200, 128, 64)))
        for x, y in ((0, 0), (255, 0), (0, 319), (255, 319)):
            if x+.5 < dx or x+.5 >= 256-dx or y+.5 < dy or y+.5 >= 320-dy:
                at = base+(y*256+x)*4
                assert raw[at:at+4] == bytes((0, 0, 0, 0))


@pytest.mark.parametrize("fill", [FillMode.COVER, FillMode.BLACK, FillMode.CONTAIN_BLUR])
def test_native_linear_minification_fine_detail_and_color(tmp_path: Path, fill: FillMode) -> None:
    from clipperx.filtergraph import compile_motion_layer

    track = compile_crop_track([keyframe(0, (.04, .08, .76, .52)),
        keyframe(2/60, (.16, .04, .5, .74))], source_size=(200, 200))
    source = ("nullsrc=s=200x200:r=60,format=gbrp,geq=r='16+219*mod(X+Y,2)':"
              "g='64+128*mod(X+Y,2)':b='80+80*mod(X+Y,2)',setparams=range=full:"
              "color_primaries=bt709:color_trc=linear:colorspace=gbr")
    graph = compile_motion_layer(track, destination=PixelRect(0, 0, 48, 48), fill=fill,
                                 blur_sigma=2, input_label="0:v", duration=1)
    data = render_graph(tmp_path, graph, size=(200, 200), source=source, frames=3, pixel_format="rgba")
    parts = ["[0:v]split=3[r0][r1][r2]"]
    for frame in range(3):
        frozen = compile_motion_layer(track, destination=PixelRect(0, 0, 48, 48), fill=fill,
            blur_sigma=2, input_label=f"r{frame}", output_label=f"f{frame}", duration=1)
        for expr, value in zip(track.expressions("n"), track.sample(frame).values, strict=True):
            frozen = frozen.replace(expr, repr(value))
        parts.append(frozen)
    parts.append("[f0][f1][f2]hstack=inputs=3[out]")
    references = render_graph(tmp_path, ";".join(parts), size=(200, 200), source=source,
                               frames=1, pixel_format="rgba")
    for frame in range(3):
        for row in range(48):
            actual = (frame*48*48+row*48)*4
            expected = (row*144+frame*48)*4
            assert data[actual:actual+48*4] == references[expected:expected+48*4]
        for channel, mean in enumerate((125.5, 128, 120)):
            values = [data[frame*48*48*4+(y*48+x)*4+channel]
                      for y in range(16, 32) for x in range(16, 32)]
            assert abs(statistics.mean(values)-mean) < 1
            assert statistics.pstdev(values) < 1
        assert data[frame*48*48*4+3] == 255
        if fill == FillMode.BLACK:
            rect = track.sample(frame)
            if rect.width != rect.height:
                assert data[frame*48*48*4:frame*48*48*4+4] == bytes((0, 0, 0, 255))


def test_native_minified_transparent_foreground_has_zero_alpha_outside_crop(tmp_path: Path) -> None:
    from clipperx.filtergraph import _motion_transform

    track = compile_crop_track([keyframe(0, (.05, .05, .8, .4)),
        keyframe(2/60, (.1, .1, .4, .8))], source_size=(200, 200))
    graph = _motion_transform(track, PixelRect(0, 0, 48, 48), cover=False,
        transparent=True, input_label="0:v", output_label="out")
    source = ("nullsrc=s=200x200:r=60,format=gbrp,geq=r=200:g=128:b=64,"
              "setparams=range=full:color_primaries=bt709:color_trc=linear:colorspace=gbr")
    data = render_graph(tmp_path, graph, size=(200, 200), source=source, frames=3, pixel_format="rgba")
    for frame in range(3):
        rect = track.sample(frame)
        scale = min(48/rect.width, 48/rect.height)
        dx, dy = (48-rect.width*scale)/2, (48-rect.height*scale)/2
        base = frame*48*48*4
        assert data[base+(24*48+24)*4+3] == 255
        for x, y in ((0, 0), (47, 0), (0, 47), (47, 47)):
            if x+.5 < dx or x+.5 >= 48-dx or y+.5 < dy or y+.5 >= 48-dy:
                at = base+(y*48+x)*4
                assert data[at:at+4] == bytes((0, 0, 0, 0))


def test_native_121frame_strong_minification_cover_control(tmp_path: Path) -> None:
    """One approved90s control; keeps the full2s clock and every authored move."""
    from clipperx.filtergraph import compile_motion_layer

    track = compile_crop_track([keyframe(0, (.05, .05, .8, .35)),
        keyframe(.7, (.08, .1, .6, .65)), keyframe(2, (.2, .02, .45, .85))],
        source_size=(200, 200))
    graph = compile_motion_layer(track, destination=PixelRect(0, 0, 48, 48),
        fill=FillMode.COVER, input_label="0:v", duration=3)
    source = rgb_ramp()
    data = render_graph(tmp_path, graph, size=(200, 200), source=source,
                        frames=121, pixel_format="rgba", timeout_seconds=90)
    assert len(data) == 121*48*48*4
    checkpoints = (0, 42, 60, 120)
    parts = ["[0:v]split=4[r0][r1][r2][r3]"]
    for i, frame in enumerate(checkpoints):
        frozen = compile_motion_layer(track, destination=PixelRect(0, 0, 48, 48),
            fill=FillMode.COVER, input_label=f"r{i}", output_label=f"f{i}", duration=3)
        for expr, value in zip(track.expressions("n"), track.sample(frame).values, strict=True):
            frozen = frozen.replace(expr, repr(value))
        parts.append(frozen)
    parts.append("[f0][f1][f2][f3]hstack=inputs=4[out]")
    references = render_graph(tmp_path, ";".join(parts), size=(200, 200), source=source,
                               frames=1, pixel_format="rgba")
    for i, frame in enumerate(checkpoints):
        for row in range(48):
            actual_offset, reference_offset = (frame*48*48+row*48)*4, (row*192+i*48)*4
            assert data[actual_offset:actual_offset+48*4] == references[reference_offset:reference_offset+48*4]


@pytest.mark.parametrize("transfer", ["linear", "bt709"])
@pytest.mark.parametrize("fill", [FillMode.COVER, FillMode.BLACK, FillMode.CONTAIN_BLUR])
def test_native_identity_preserves_rgb_values_and_known_tags(tmp_path: Path,
                                                            transfer: str, fill: FillMode) -> None:
    from clipperx.filtergraph import compile_motion_layer

    full = (0, 0, 1, 1)
    track = compile_crop_track([keyframe(0, full), keyframe(2, full)], source_size=(192, 144))
    source = rgb_ramp(size=(192, 144), transfer=transfer)
    graph = compile_motion_layer(track, destination=PixelRect(0, 0, 192, 144),
        fill=fill, blur_sigma=2, input_label="0:v", duration=1)
    graph = graph.replace("[out]", ",showinfo[out]")
    actual = render_graph(tmp_path, graph, size=(192, 144), source=source, frames=1, pixel_format="rgba")
    log = json.loads((tmp_path / "diagnostic-0.json").read_text(encoding="utf-8"))["stderr"]
    assert f"color_trc:{transfer}" in log
    assert "color_primaries:bt709" in log and "color_range:pc" in log
    reference = render_graph(tmp_path, "[0:v]format=rgba[out]", size=(192, 144),
                               source=source, frames=1, pixel_format="rgba")
    assert actual == reference


def test_native_unknown_tags_remain_unknown(tmp_path: Path) -> None:
    from clipperx.filtergraph import compile_motion_layer

    track = compile_crop_track([keyframe(0, (0, 0, 1, 1)), keyframe(2, (0, 0, 1, 1))],
                               source_size=(192, 144))
    graph = compile_motion_layer(track, destination=PixelRect(0, 0, 96, 96),
        fill=FillMode.COVER, input_label="0:v", duration=1).replace("[out]", ",showinfo[out]")
    render_graph(tmp_path, graph, size=(192, 144), frames=1, pixel_format="rgba")
    log = json.loads((tmp_path / "diagnostic-0.json").read_text(encoding="utf-8"))["stderr"]
    assert "color_primaries:unknown" in log and "color_trc:unknown" in log


def test_native_16bit_chroma_geometry_has_independent_xy_oracle(tmp_path: Path) -> None:
    import array
    from clipperx.filtergraph import _motion_transform

    track = compile_crop_track([keyframe(0, (16/192, 16/144, 96/192, 48/144), "ease_in_out"),
        keyframe(2, (48/192, 8/144, 96/192, 112/144))], source_size=(192, 144))
    source = ("nullsrc=s=192x144:r=60,format=yuv444p16le,"
              "geq=lum=32768:cb='20000+100*Y':cr='45000-100*X',"
              "setparams=range=full:color_primaries=bt709:color_trc=bt709:colorspace=bt709")
    graph = _motion_transform(track, PixelRect(0, 0, 96, 96), cover=True,
        input_label="0:v", output_label="out", pixel_format="yuv444p16le")
    raw = render_graph(tmp_path, graph, size=(192, 144), source=source,
                        frames=121, pixel_format="yuv444p16le")
    values = array.array("H", raw)
    assert len(values) == 121*96*96*3
    for frame in (0, 60, 120):
        crop = track.sample(frame)
        scale = max(96/crop.width, 96/crop.height)
        dx, dy = (96-crop.width*scale)/2, (96-crop.height*scale)/2
        for x in (24, 48, 72):
            for y in (24, 48, 72):
                base = frame*96*96*3+y*96+x
                actual_y = (values[base+96*96]-20000)/100
                actual_x = (45000-values[base+2*96*96])/100
                expected_x = crop.x+(x+.5-dx)/scale-.5
                expected_y = crop.y+(y+.5-dy)/scale-.5
                assert abs(actual_x-expected_x) < 1
                assert abs(actual_y-expected_y) < 1


@pytest.mark.parametrize("start,rate,offset", [(.025, 30, 0), (.0125, 30, 0),
                                             (10.025, 30, 10), (.025, 60, 0)])
def test_fractional_source_clock_keeps_initial_offset_and_half_open_interval(
        tmp_path: Path, start: float, rate: int, offset: int) -> None:
    from clipperx.filtergraph import compile_shot_layers
    from clipperx.tests.test_filtergraph import host_motion_shot

    shot = host_motion_shot()
    shot.source.start, shot.source.end = start, start+2
    compiled = compile_shot_layers(shot, source_size=(200, 200))
    source_branch = compiled.filter_complex.split(";")[1]
    # Reuse actual production clock branch, keep its exact trim/setpts. Inspect
    # its small decoded samples without rendering the 1080 composition.
    graph = source_branch[:source_branch.rindex("[")] + "[out]"
    source = (f"nullsrc=s=200x200:r={rate},format=yuv444p,geq=lum='16+N':cb=90:cr=160,"
              f"setpts=PTS+{offset}/TB")
    data = render_graph(tmp_path, graph, size=(200, 200), source=source, frames=200, rate=rate,
                        copyts=offset != 0)
    stride = 200*200*3
    assert len(data) == 120*stride
    expected = [17, 17, 17, 18] if rate == 30 else [18, 18, 19, 20]
    assert [data[f*stride] for f in (0, 1, 2, 3)] == expected
    # The frame at the two-second boundary is not appended to the editorial
    # shot. Its crop geometry remains queryable at frame120 separately.
    assert data[119*stride] == (76 if rate == 30 else 136)
    track = compile_crop_track(shot.layers[0].crop_track, source_size=(200, 200))
    assert track.sample(0) == FloatRect(0, 0, 160, 160)
    assert track.sample(120) == FloatRect(40, 40, 160, 160)


def test_fractional_clock_drives_actual_motion_and_30fps_boundary_geometry(tmp_path: Path) -> None:
    from clipperx.filtergraph import compile_motion_layer, compile_shot_layers
    from clipperx.tests.test_filtergraph import host_motion_shot

    shot = host_motion_shot()
    shot.source.start, shot.source.end = .025, 2.025
    compiled = compile_shot_layers(shot, source_size=(200, 200))
    branch = compiled.filter_complex.split(";")[1]
    clock = branch[:branch.rindex("[")] + "[clock]"
    track = compile_crop_track(shot.layers[0].crop_track, source_size=(200, 200))
    transform = compile_motion_layer(track, destination=PixelRect(0, 0, 48, 48),
        fill=FillMode.COVER, input_label="clock", duration=2)
    source = rgb_ramp(rate=30)
    data = render_graph(tmp_path, clock+";"+transform, size=(200, 200), source=source,
                         frames=200, pixel_format="rgba")
    assert len(data) == 120*48*48*4
    for frame in (0, 1, 2, 60, 119):
        rect = track.sample(frame)
        for channel, expected in enumerate((16+rect.x+24.5*rect.width/48-.5,
                                             16+rect.y+24.5*rect.height/48-.5)):
            assert abs(data[frame*48*48*4+(24*48+24)*4+channel]-expected) <= 1.2
    # A separate121-frame boundary diagnostic is explicit synthetic material,
    # not an extra frame appended to the exact half-open editorial interval.
    boundary = render_graph(tmp_path, "[0:v]fps=60:start_time=0:round=near[clock];"+transform,
                            size=(200, 200), source=source, frames=121, pixel_format="rgba")
    endpoint = track.sample(120)
    expected = 16+endpoint.x+24.5*endpoint.width/48-.5
    assert abs(boundary[120*48*48*4+(24*48+24)*4]-expected) <= 1.2


@pytest.mark.parametrize("start", [.025, .0125])
def test_fractional_static_shot_final_overlay_has_same_source_clock(tmp_path: Path,
                                                                  start: float) -> None:
    from clipperx.filtergraph import compile_shot_layers
    from clipperx.tests.test_filtergraph import host_motion_shot

    shot = host_motion_shot()
    shot.source.start, shot.source.end = start, start+2
    shot.layers[0].crop_track = []
    compiled = compile_shot_layers(shot, source_size=(200, 200))
    # Inspect a 2x2 patch of the actual full-canvas final overlay; no rewritten
    # source branch or substituted composition bypasses first-secondary PTS.
    graph = compiled.filter_complex + f";[{compiled.output_label}]crop=2:2:540:960[out]"
    source = "nullsrc=s=200x200:r=30,format=yuv444p,geq=lum='16+N':cb=90:cr=160"
    data = render_graph(tmp_path, graph, size=(200, 200), source=source, frames=4)
    assert len(data) == 4*2*2*3
    assert [data[f*12] for f in range(4)] == [17, 17, 17, 18]
