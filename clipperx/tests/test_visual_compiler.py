"""Explicit HOST synthetic v2 fixtures, never director-ready receipts."""

from pathlib import Path
import subprocess
from dataclasses import replace
from fractions import Fraction as F
import json
from typing import Any

import pytest
from pydantic import ValidationError

from clipperx import visual_compiler as compiler
from clipperx.local_paths import source_fingerprint
from clipperx.render_assets import RenderAssetRegistry, default_render_assets
from clipperx.render_contracts import EncoderName, RenderProfile
from clipperx.schema import TranscriptSegment, Word
from clipperx.visual_contracts import VisualCompositionPlan


def pure_plan(kinds: tuple[str, ...] = ("source_video",), *, duration: float = .1
              ) -> VisualCompositionPlan:
    """Declared metadata only; never resolves or fingerprints a live path."""
    shots = []
    for index, kind in enumerate(kinds):
        start, end = float(F(str(duration)) * index), float(F(str(duration)) * (index + 1))
        shots.append(dict(shot_id=f"pure{index}", source=dict(start=.02, end=float(F(1, 50) + F(str(duration)))),
            output=dict(start=start, end=end), primary_source="content", reason="HOST pure",
            evidence_ids=["HOST"], confidence=1, layers=[dict(layer_id="root", kind=kind,
                source_ref="source", destination=dict(x=0, y=0, width=1, height=1), z_index=0,
                fill="solid" if kind == "solid" else "black",
                solid_color="#112233" if kind == "solid" else None)]))
    return VisualCompositionPlan.model_validate(dict(clip_id="pure", shots=shots,
        source_fingerprint="0123456789abcdef", composition_fingerprint="HOST",
        source_timeline_fingerprint="HOST", identity_fingerprint="HOST",
        style_library_fingerprint="HOST"))


def pure_overlays(plan: VisualCompositionPlan) -> Any:
    from clipperx.filtergraph import rect_to_pixels
    from clipperx.output_timeline import OutputTimeline
    from clipperx.overlay_ass import OverlayCompilation, OverlayEvent, serialize_overlay_events
    from clipperx.render_assets import PreparedBrandAsset, place_brand_asset
    timeline = OutputTimeline.from_shots(plan.shots)
    events, brands = [], []
    for shot in plan.shots:
        if shot.title is not None:
            title = shot.title
            assert title.text is not None
            from clipperx.overlay_ass import _clean_text
            text = _clean_text(title.text)
            events.append(OverlayEvent(shot.shot_id, "title", shot.output.start + title.start,
                shot.output.start + title.end,
                rect_to_pixels(title.destination, width=1080, height=1920), (text,), "HOST Face",
                46, True, 2, "&H00FFFFFF", "&H00000000", title.z_index, 20, 20,
                text=text, line_ranges=((0, len(text)),), exact_interval=
                timeline.overlay_output_interval(shot.shot_id, title.start, title.end)))
        if shot.brand is not None:
            brand = shot.brand
            asset = PreparedBrandAsset(Path("HOST-original.png"), Path("HOST-trimmed.png"),
                "a" * 64, (4, 4), (0, 0, 4, 4), (4, 4))
            brands.append(place_brand_asset(asset,
                destination=rect_to_pixels(brand.destination, width=1080, height=1920),
                start=shot.output.start + brand.start, end=shot.output.start + brand.end,
                z_index=brand.z_index, shot_id=shot.shot_id))
    return OverlayCompilation(serialize_overlay_events(events), tuple(events), (), tuple(brands))


def assert_outer_rgb_graph(assembly: compiler.CompiledVisualAssembly) -> None:
    """Graph contract only; no assertion about negotiated native pixel values."""
    assert "format" in assembly.requirements.required_filters
    for shot in assembly.shots:
        base_index = int(shot.source_input_label is not None)
        base = shot.steps[base_index]
        assert isinstance(base, compiler.FilterFragment)
        base_label = f"shot_{shot.span.occurrence_index:04d}_base"
        assert base == compiler.FilterFragment(
            f"color=c=black:s=1080x1920:r=60,trim=end_frame={shot.frame_count},"
            f"settb=1/60,setpts=N,format=rgba,setpts=N+{shot.span.k_start}[{base_label}]",
            (), (base_label,))
        for primitive in shot.source_layers:
            primitive_index = next(i for i, step in enumerate(shot.steps)
                if isinstance(step, compiler.FilterFragment)
                and step.output_labels == (primitive.output_label,))
            global_label = primitive.output_label + "_global"
            assert shot.steps[primitive_index + 1] == compiler.FilterFragment(
                f"[{primitive.output_label}]format=rgba,settb=1/60,"
                f"setpts=N+{shot.span.k_start}[{global_label}]",
                (primitive.output_label,), (global_label,))
            paint = shot.steps[primitive_index + 2]
            assert isinstance(paint, compiler.FilterFragment)
            assert paint.input_labels[1:] == (global_label,)
            assert len(paint.output_labels) == 1
            p = primitive.destination
            assert paint.text == (f"[{paint.input_labels[0]}][{global_label}]"
                f"overlay=x={p.x}:y={p.y}:shortest=1:format=rgb[{paint.output_labels[0]}]")
        brand_labels = {b.graph_label for b in assembly.brand_inputs if b.shot_id == shot.span.shot_id}
        available = brand_labels | ({shot.source_input_label} if shot.source_input_label else set())
        for step in shot.steps:
            inputs = step.input_labels if isinstance(step, compiler.FilterFragment) else (step.input_label,)
            outputs = step.output_labels if isinstance(step, compiler.FilterFragment) else (step.output_label,)
            assert set(inputs) <= available
            available.difference_update(inputs)
            available.update(outputs)
        assert available == {shot.output_label}
        assert len(shot.steps) == (2 + base_index + 3 * len(shot.source_layers)
            + sum(isinstance(s, compiler.AssApply) for s in shot.steps) + len(brand_labels))
        final = shot.steps[-1]
        assert isinstance(final, compiler.FilterFragment)
        assert final.text == (f"[{final.input_labels[0]}]setsar=1,settb=1/60,"
                              f"setpts=N[{shot.output_label}]")


@pytest.mark.parametrize("fill", ["cover", "black", "contain_blur", "solid"])
@pytest.mark.parametrize("directive", ["base", "primitive", "paint", "requirement"])
def test_pure_assembly_explicit_outer_rgb_directives(fill: str, directive: str) -> None:
    plan = pure_plan(("solid" if fill == "solid" else "source_video",), duration=1/30)
    layer = plan.shots[0].layers[0]
    layer.fill = compiler.FillMode(fill)
    layer.blur_sigma = 2 if fill == "contain_blur" else None
    assembly = compiler.compile_visual_assembly(plan, source_size=(2, 2), overlays=pure_overlays(plan))
    shot = assembly.shots[0]
    base_index = int(fill != "solid")
    primitive = shot.source_layers[0]
    if directive == "base":
        assert shot.steps[base_index] == compiler.FilterFragment(
            "color=c=black:s=1080x1920:r=60,trim=end_frame=2,settb=1/60,setpts=N,"
            "format=rgba,setpts=N+0[shot_0000_base]", (), ("shot_0000_base",))
    elif directive == "primitive":
        assert shot.steps[base_index + 2] == compiler.FilterFragment(
            f"[{primitive.output_label}]format=rgba,settb=1/60,"
            f"setpts=N+0[{primitive.output_label}_global]",
            (primitive.output_label,), (primitive.output_label + "_global",))
    elif directive == "paint":
        assert shot.steps[base_index + 3] == compiler.FilterFragment(
            f"[shot_0000_base][{primitive.output_label}_global]"
            "overlay=x=0:y=0:shortest=1:format=rgb[shot_0000_paint_0]",
            ("shot_0000_base", primitive.output_label + "_global"), ("shot_0000_paint_0",))
    else:
        assert "format" in assembly.requirements.required_filters
    assert_outer_rgb_graph(assembly)


@pytest.mark.parametrize("fill,primitive_hash,legacy_hash", [
    ("cover", "a462f73c0db93e0b1368d2f096a4edcdf60241a59b0134c6346b5def18201350",
     "c7912160dcfc482530bce39f88b3c3eb5d31ca9833e216b7975d492225fea048"),
    ("black", "659885207566c1eb2ec57c989e7cfa689eb64dc0d1f306ef81c032df73bb166e",
     "0208bececfa0b4d055516781df0a14d4dc3384fb2b3a7c9c9464fbe2fa82434d"),
    ("contain_blur", "a2174a558b0323ce5608abd2983447aa74932e1f58ab51554052681466cbdb92",
     "d38693d1a12cb70aa78c344b750e92a1b1f8dc3b5241e12e69602f3176c297b5"),
    ("solid", "57f09a2db77f4ddf1c14c70958c1b58fce0437496a9766bd3cff66b5d8b905e5",
     "02156e7d351f50166461f89720d0d07e538c38c117bb7ac29a47cf14504dbef7"),
])
def test_shared_primitive_and_legacy_graphs_keep_pre_rgb_snapshots(
        fill: str, primitive_hash: str, legacy_hash: str) -> None:
    import hashlib
    from clipperx.filtergraph import PixelRect, _finite_color, compile_layer
    # Fixed hashes captured before the V2 outer-format edit, never regenerated in this test.
    kind = compiler.LayerKind.SOLID if fill == "solid" else compiler.LayerKind.SOURCE_VIDEO
    blur = 2 if fill == "contain_blur" else None
    graph = compile_layer(source_size=(2, 2), destination=PixelRect(0, 0, 4, 8),
        fill=compiler.FillMode(fill), kind=kind, solid_color="#112233" if fill == "solid" else None,
        blur_sigma=blur, frame_count=2)
    assert hashlib.sha256(graph.encode()).hexdigest() == primitive_hash
    assert _finite_color("black", PixelRect(0, 0, 4, 8), frame_count=2) == (
        "color=c=black:s=4x8:r=60,trim=end_frame=2,settb=1/60,setpts=N")
    plan = pure_plan((kind.value,))
    plan.shots[0].layers[0].fill = compiler.FillMode(fill)
    plan.shots[0].layers[0].blur_sigma = blur
    legacy = compiler.compile_visual_layers(plan, source_size=(2, 2))[0]
    assert hashlib.sha256(legacy.filter_complex.encode()).hexdigest() == legacy_hash


def test_pure_sparse_assembly_branches_scheduled_pixels_and_finishes_local(
        monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> Any:
        pytest.fail("Pure assembly invoked a host/old-clock operation")
    for name in ("compile_shot_layers", "compile_visual_layers", "_probe", "_inspect",
                 "resolve_render_runtime", "qualify_render_runtime", "source_fingerprint"):
        monkeypatch.setattr(compiler, name, forbidden)
    plan = pure_plan(("source_video", "solid", "source_video"), duration=1/30)
    camera = plan.shots[0].layers[0].model_copy(deep=True)
    camera.layer_id, camera.kind, camera.source_ref = "cam", compiler.LayerKind.STREAMER_CAMERA, "cam0"
    camera.z_index = 2
    plan.shots[0].layers.append(camera)
    before = plan.model_dump_json()
    assembly = compiler.compile_visual_assembly(plan, source_size=(2, 2), overlays=pure_overlays(plan))
    assert assembly.source_input is not None and assembly.source_input.frame_count == 4
    assert [(s.transport.slot_start, s.transport.slot_end) for s in assembly.shots
            if s.transport is not None] == [(0, 2), (2, 4)]
    assert [s.span.k_start for s in assembly.shots] == [0, 2, 4]
    first = assembly.shots[0]
    assert len(first.source_layers) == 2
    assert isinstance(first.steps[0], compiler.FilterFragment)
    assert "settb=1/60,setpts=N,split=2" in first.steps[0].text
    assert all(s.output_clock == "local_transport" and s.output_time_base == F(1, 60)
               and s.local_n_start == 0 and s.frame_count == 2 for s in assembly.shots)
    assert all(isinstance(s.steps[-1], compiler.FilterFragment)
               and "settb=1/60,setpts=N" in s.steps[-1].text for s in assembly.shots)
    assert "fps=" not in ";".join(step.text for s in assembly.shots for step in s.steps
                                  if isinstance(step, compiler.FilterFragment))
    metadata = json.loads(assembly.compiler_metadata_json)
    assert metadata["source_observation"] == "source_input_observed_at_binding_only"
    assert plan.model_dump_json() == before
    assert_outer_rgb_graph(assembly)
    plan.shots[0].shot_id = "changed"
    assert first.span.shot_id == "pure0"
    solid_plan = pure_plan(("solid",))
    solid = compiler.compile_visual_assembly(solid_plan, source_size=(2, 2),
                                             overlays=pure_overlays(solid_plan))
    assert solid.source_input is None and solid.shots[0].source_input_label is None
    assert solid.shots[0].frame_count == 6
    assert json.loads(solid.compiler_metadata_json)["source_observation"] == "not_applicable_no_source_consumer"
    assert_outer_rgb_graph(solid)


def interleaved_plan() -> VisualCompositionPlan:
    from clipperx.visual_contracts import NormalizedRect, OverlayPlan
    plan = pure_plan(duration=.3)
    shot = plan.shots[0]
    rect = NormalizedRect(x=.1, y=.1, width=.8, height=.1)
    shot.title = OverlayPlan(text=r"literal {\\pos(1,2)}", start=.025, end=.225,
                             destination=rect, z_index=1)
    shot.brand = OverlayPlan(asset_id="host_brand", start=.025, end=.225,
                             destination=rect, z_index=2)
    camera = shot.layers[0].model_copy(deep=True)
    camera.layer_id, camera.kind, camera.source_ref = "camera", compiler.LayerKind.STREAMER_CAMERA, "cam0"
    camera.z_index = 1
    shot.layers.append(camera)
    return plan


def test_global_order_ass_runs_png_boundary_and_literal_escape() -> None:
    from clipperx.filtergraph import rect_to_pixels
    from clipperx.overlay_ass import OverlayCompilation, serialize_overlay_events
    from clipperx.visual_contracts import NormalizedRect, SubtitlePlan
    plan = interleaved_plan()
    shot = plan.shots[0]
    shot.subtitles = SubtitlePlan(destination=NormalizedRect(x=.1, y=.7, width=.8, height=.1))
    overlays = pure_overlays(plan)
    title = overlays.events[0]
    caption = replace(title, kind="subtitle", start=.025, end=.225, text="hello, world!",
        lines=("hello, world!",), line_ranges=((0, 13),), word_ranges=((0, 5), (5, 6), (7, 13)),
        word_count=3, destination=rect_to_pixels(shot.subtitles.destination,
            width=1080, height=1920), font_size=58, outline_width=4,
        primary_color="&H001AD5FF", layer=3, exact_interval=(F(1, 40), F(9, 40)))
    events = (caption, title)  # Canonical order is intentionally not paint order.
    overlays = OverlayCompilation(serialize_overlay_events(events), events, (), overlays.brands)
    before = overlays.document
    result = compiler.compile_visual_assembly(plan, source_size=(2, 2), overlays=overlays)
    assert_outer_rgb_graph(result)
    applies = [step for step in result.shots[0].steps if isinstance(step, compiler.AssApply)]
    assert len(applies) == 2 and len(result.brand_inputs) == 1
    assert [d.event_indices for d in result.ass_documents] == [(1,), (0,)]
    assert result.ass_documents[0].windows == ((2, 14),)
    brand = result.brand_inputs[0]
    assert brand.required_asset_id == "host_brand"
    assert brand.declared_interval == (F(1, 40), F(9, 40))
    assert brand.execution_interval == (F(3, 120), F(27, 120))
    assert overlays.document == before
    assert "\\{" in result.ass_documents[0].document
    for start in range(1, 7):
        for end in range(start + 1, start + 4):
            c0, c1 = compiler._ass_boundary(start), compiler._ass_boundary(end)
            e0, e1 = compiler._png_boundary(start), compiler._png_boundary(end)
            assert [k for k in range(12) if c0 <= F(k, 60) < c1] == list(range(start, end))
            assert [k for k in range(12) if e0 <= F(k, 60) < e1] == list(range(start, end))


@pytest.mark.parametrize("fault", ["missing", "extra", "canonical", "foreign", "style", "text",
    "range", "extent", "timing", "brand", "font_injection", "active_title"])
def test_fresh_overlay_structural_association_rejects_forgery(fault: str) -> None:
    from clipperx.overlay_ass import OverlayCompilation, serialize_overlay_events
    plan = interleaved_plan()
    overlays = pure_overlays(plan)
    event = overlays.events[0]
    changes: dict[str, Any] = {"foreign": dict(shot_id="foreign"), "style": dict(font_size=47),
        "text": dict(text="other"), "range": dict(line_ranges=((1, len(event.text)),)),
        "extent": dict(measured_width=float("nan")), "timing": dict(exact_interval=(F(0), F(1, 10))),
        "font_injection": dict(font_family="HOST,Injected"), "active_title": dict(active_words=(0,))}
    events = () if fault == "missing" else (event, event) if fault == "extra" else (
        replace(event, **changes[fault]),) if fault in changes else overlays.events
    document = ("forged" if fault == "canonical" else overlays.document if fault == "active_title"
                else serialize_overlay_events(events))
    brands = () if fault == "brand" else overlays.brands
    with pytest.raises(ValueError):
        compiler.compile_visual_assembly(plan, source_size=(2, 2),
            overlays=OverlayCompilation(document, events, (), brands))


def supplied_window(shift: F = F(0)) -> tuple[Any, Any, list[bytes], Any]:
    import hashlib
    import zlib
    from clipperx.render_clock import SourceDecodeReceipt
    from clipperx.source_decode import DECODE_POLICY, DecodedWindow, DecodeLimits, VideoTag, select_frames
    from clipperx.tests.test_render_clock import make_origin, valid_payload
    origin = make_origin()
    if shift:
        origin = type(origin)(origin.path, origin.size, origin.mtime_ns, origin.fingerprint,
            origin.origin, F(4), [replace(s, end=F(9)) for s in origin.streams], origin.decoder_id,
            dict(origin.color))
    limits = DecodeLimits(preroll=F(1, 50), postroll=F(3, 100), max_window=F(1), timeout=3,
        max_frames=10, max_samples=10000, max_pixels=4, max_decoded_bytes=30000, max_retained_bytes=20000)
    pixels = [bytes([value]) * 12 for value in (20, 40, 80)]
    tags = [VideoTag(n, pts + int(shift * 1000), F(1, 1000), 2, 2, zlib.adler32(pixels[n], 0))
            for n, pts in enumerate((5015, 5050, 5130))]
    requests = tuple(shift + F(1, 50) + F(k, 60) for k in range(6))
    selection = select_frames(zip(tags, pixels), origin, requests, limits)
    payload = valid_payload()
    from clipperx.render_clock import rational
    declared_window = [rational(shift + F(1, 50)), rational(shift + F(3, 25))]
    payload["windows"] = [declared_window]
    payload["limits"] = dict(limits.record(), decoder_start=rational(shift),
                             decoder_end=rational(shift + F(3, 20)))
    payload["video"] = [json.loads(row) for row in selection.observations]
    payload["outcomes"] = [row.record() for row in selection.outcomes]
    pcm = bytes(14400)
    pcm_sha = hashlib.sha256(pcm).hexdigest()
    audio = payload["audio"][0]
    audio["input"][0]["samples"] = 6615
    audio["input"][0]["pts"] += int(shift * 44100)
    audio["output"][0].update(samples=7200, bytes=len(pcm), sha256=pcm_sha,
                              checksum=f"{zlib.adler32(pcm, 0):08X}", pts=240000 + int(shift * 48000))
    audio.update(samples=7200, bytes=len(pcm), pcm_sha256=pcm_sha,
                 sample_grid_origin=rational(F(5) + shift), source_phase=rational(shift),
                 coverage=[{"window": declared_window, "status": "covered"}])
    receipt = SourceDecodeReceipt(origin.identity, origin.decoder_id, DECODE_POLICY, payload)
    return origin, DecodedWindow(receipt, selection.frames, pcm), pixels, limits


def stream_binding_fixture(shift: F = F(0), *, partial: bool = False) -> Any:
    """Synthetic producer metadata only: no stream, child, clock or media work."""
    from clipperx import render_clock as clock
    from clipperx.source_decode import ProvisionalSelection
    origin, materialized, _, _ = supplied_window(shift)
    plan = pure_plan(duration=2.1) if shift else pure_plan(("source_video", "solid", "source_video"))
    if shift:
        from clipperx.visual_contracts import CropKeyframe, NormalizedRect
        plan.shots[0].layers[0].crop_track = [CropKeyframe(at=t,
            crop=NormalizedRect(x=0, y=0, width=1, height=1)) for t in (0, 2, 2.1)]
    assembly = compiler.compile_visual_assembly(plan, source_size=(2, 2), overlays=pure_overlays(plan))
    limits = clock.StreamDecodeLimits(preroll=F(1, 50), postroll=F(3, 100), max_window=F(1),
        timeout=3, max_frames=10, max_samples=10000, max_pixels=4, max_decoded_bytes=30000)
    start, end = (123 if partial else 120, 126) if shift else (6, 12)
    session = "00000000-0000-4000-8000-000000000001"
    cursor = compiler.begin_source_stream_binding(assembly, slot_start=start, slot_end=end,
        origin=origin, session_id=session, window=(shift + F(1, 50), shift + F(3, 25)),
        expected_audio_policy="render48k-mono", limits=limits)
    data = json.loads(materialized.receipt.payload_json)
    rows = data["video"]
    groups = ((1, tuple(range(3))),) if partial else ((0, (0, 1)), (1, (2, 3, 4, 5)))
    selections: list[Any] = []
    cursors, deliveries = [cursor], []
    for n, ordinals in groups:
        selection = ProvisionalSelection(session, len(selections), ordinals, materialized.frames[n],
            rows[n], dict(rows[n + 1], coverage_end=None), previous_chain_sha256=cursor.chain_sha256)
        selections.append(selection)
        cursor, delivery = compiler.bind_source_stream_selection(cursor, selection=selection)
        cursors.append(cursor)
        deliveries.append(delivery)
    requests = [clock.rational(r.requested_source) for r in cursor.context.requests]
    payload = dict(window=[clock.rational(t) for t in cursor.context.window], requests=requests,
        operation_limits=limits.operation_record(), delivery_limits=limits.delivery_record(),
        decoder_start=clock.rational(shift), decoder_end=clock.rational(shift + F(3, 20)),
        video=rows, audio=data["audio"][0], outcomes=data["outcomes"][3:] if partial else data["outcomes"],
        argv=[clock.video_decode_argv("fixture-ffmpeg", origin, shift, shift + F(3, 20)),
              clock.audio_decode_argv("fixture-ffmpeg", origin, shift, shift + F(3, 20), "render48k-mono")],
        deliveries=[s.delivery_record() for s in selections])
    payload["accounting"] = dict(decoded_video_frames=3, decoded_video_bytes=36,
        decoded_audio_output_bytes=14400, selected_unique_frames=len(groups),
        selected_unique_bytes=12 * len(groups), offered_deliveries=len(groups),
        offered_payload_bytes=12 * len(groups), acknowledged_deliveries=len(groups),
        acknowledged_requests=len(requests), acknowledged_request_bytes=12 * len(requests),
        reserved_payload_capacity_bytes=80000, peak_live_payload_bytes_accounted=60000,
        metadata_bytes=len(clock.canonical(payload).encode("utf-8")), video_log_bytes=1, audio_log_bytes=1)
    payload["completion"] = dict(children=[dict(kind=kind, exit_code=0, started_ns=start_ns,
        finished_ns=start_ns + 1, cleanup_complete=True) for kind, start_ns in (("video", 0), ("audio", 2))],
        source_fingerprint_after=origin.fingerprint, pins_verified_before_children=True,
        all_requests_acknowledged=True, cleanup_complete=True, elapsed_ns=4)
    finalization = clock.SourceStreamFinalization(origin, session, payload, limits=limits)
    return assembly, origin, materialized, limits, cursors, selections, deliveries, finalization


def test_stream_binding_real_roundtrip_sparse_schedule() -> None:
    from clipperx import render_clock as clock
    assembly, origin, materialized, limits, cursors, selections, deliveries, finalization = stream_binding_fixture()
    assert cursors[0].context.assembly is assembly
    assert [s.transport_slot for d in deliveries for s in d.slots] == list(range(6, 12))
    assert [s.request.global_k for d in deliveries for s in d.slots] == list(range(12, 18))
    assert [s.request.local_n for d in deliveries for s in d.slots] == list(range(6))
    assert [len(d.slots) for d in deliveries] == [2, 4]
    assert deliveries[0].frame is materialized.frames[0] is selections[0].frame
    assert deliveries[1].frame is materialized.frames[1]
    assert not hasattr(deliveries[0], "receipt_hash") and not hasattr(cursors[-1], "receipt_hash")
    checked = clock.SourceStreamFinalization.from_json(finalization.to_json(), origin=origin,
        policy=clock.STREAM_DECODE_POLICY, limits=limits, expected_session_id=cursors[0].context.session_id)
    final = compiler.finalize_source_stream_binding(cursors[-1], finalization=checked)
    assert final.finalization_sha256 == checked.sha256
    assert final.final_chain_sha256 == cursors[-1].chain_sha256
    assert [s.frame_n for s in final.slots] == [0, 0, 1, 1, 1, 1]
    assert final.selected_unique_bytes == 24 and final.request_emission_bytes == 72
    assert final.producer_pcm_bytes == 14400 and final.producer_audio_coverage == "covered"
    assert not hasattr(final, "frame") and not hasattr(final, "pcm")


@pytest.mark.parametrize("fault", ["foreign_assembly", "foreign_origin", "foreign_limits", "slot_bool",
    "cross_occurrence", "all_solid", "source_count", "source_mask", "transport_prefix", "origin_bool",
    "origin_policy", "origin_streams", "origin_tb", "timeline_bool", "timeline_huge", "timeline_fraction",
    "mutable_shots", "mutable_steps", "mutable_labels", "mutable_size", "mutable_requirements",
    "mutable_crop_points", "mutable_crop_rect", "wrong_window", "window_bool", "session", "audio_policy",
    "limit_bool", "limit_fraction", "context_request_bool", "context_local_n", "context_request_rule",
    "context_header", "context_source_size", "context_transport", "context_occurrence", "context_policy",
    "context_requests_list", "context_window_list", "context_transport_bool", "context_raster_bool",
    "context_size_float", "cursor_bool", "cursor_negative", "cursor_huge", "cursor_skip", "cursor_last_huge",
    "missing_assembly", "missing_origin", "missing_limits", "missing_cursor", "missing_context", "missing_span"])
def test_stream_binding_rejects_foreign_shapes_and_context_changes(fault: str) -> None:
    assembly, origin, _, limits, cursors, selections, _, _ = stream_binding_fixture()
    args: dict[str, Any] = dict(assembly=assembly, origin=origin, limits=limits, slot_start=6, slot_end=12,
        session_id=cursors[0].context.session_id, window=(F(1, 50), F(3, 25)),
        expected_audio_policy="render48k-mono")
    cursor = cursors[0]
    if fault.startswith("missing_"):
        if fault in ("missing_cursor", "missing_context"):
            cursor = object.__new__(type(cursor)) if fault == "missing_cursor" else replace(cursor,
                context=object.__new__(type(cursor.context)))
        elif fault == "missing_span":
            args["assembly"] = replace(assembly, shots=(replace(assembly.shots[0],
                span=object.__new__(compiler.RasterSpan)), *assembly.shots[1:]))
        else:
            name = fault.removeprefix("missing_")
            args[name] = object.__new__(type(args[name]))
    elif fault.startswith("foreign_"):
        args[fault.removeprefix("foreign_")] = object()
    elif fault == "slot_bool":
        args["slot_start"] = True
    elif fault == "cross_occurrence":
        args.update(slot_start=5, slot_end=7)
    elif fault == "all_solid":
        plan = pure_plan(("solid",))
        args["assembly"] = compiler.compile_visual_assembly(plan, source_size=(2, 2), overlays=pure_overlays(plan))
    elif fault == "source_count":
        args["assembly"] = replace(assembly, source_input=replace(assembly.source_input, frame_count=13))
    elif fault in ("source_mask", "transport_prefix"):
        shot = assembly.shots[0]
        shot = replace(shot, transport=None, source_input_label=None) if fault == "source_mask" else replace(
            shot, transport=replace(shot.transport, slot_start=1))
        args["assembly"] = replace(assembly, shots=(shot, *assembly.shots[1:]))
    elif fault.startswith("origin_"):
        origin = replace_origin(origin)
        origin_changes: dict[str, tuple[str, Any]] = {"origin_bool": ("size", True), "origin_policy": ("policy", "foreign"),
            "origin_streams": ("streams", list(origin.streams)), "origin_tb": ("streams",
            (replace(origin.streams[0], time_base=F(1)), origin.streams[1]))}
        key, value = origin_changes[fault]
        if fault == "origin_tb":
            object.__setattr__(value[0], "time_base", True)
        object.__setattr__(origin, key, value)
        args["origin"] = origin
    elif fault.startswith("timeline_"):
        span = replace(assembly.timeline.spans[0])
        object.__setattr__(span, "source_start", {"timeline_bool": True, "timeline_huge": 1 << 4095,
            "timeline_fraction": F(1, 50)}[fault])
        timeline = replace(assembly.timeline)
        object.__setattr__(timeline, "spans", (span, *assembly.timeline.spans[1:]))
        args["assembly"] = replace(assembly, timeline=timeline)
    elif fault.startswith("mutable_"):
        shot = assembly.shots[0]
        if fault == "mutable_shots":
            args["assembly"] = replace(assembly, shots=list(assembly.shots))
        elif fault == "mutable_size":
            args["assembly"] = replace(assembly, source_size=[2, 2])
        elif fault == "mutable_steps":
            args["assembly"] = replace(assembly, shots=(replace(shot, steps=list(shot.steps)), *assembly.shots[1:]))
        elif fault == "mutable_labels":
            step = shot.steps[0]
            args["assembly"] = replace(assembly, shots=(replace(shot, steps=(replace(step, input_labels=[]), *shot.steps[1:])), *assembly.shots[1:]))
        elif fault == "mutable_requirements":
            requirements = compiler.RenderRequirements()
            object.__setattr__(requirements, "required_filters", [])
            args["assembly"] = replace(assembly, requirements=requirements)
        else:
            track = compiler.CompiledCropTrack((compiler.CropPoint(0, compiler.FloatRect(0, 0, 2, 2), "linear"),), (2, 2), 60, "static")
            if fault == "mutable_crop_points":
                object.__setattr__(track, "points", list(track.points))
            else:
                object.__setattr__(track.points[0], "rect", [0, 0, 2, 2])
            layer = replace(shot.source_layers[0], crop_track=track)
            args["assembly"] = replace(assembly, shots=(replace(shot, source_layers=(layer,)), *assembly.shots[1:]))
    elif fault == "wrong_window":
        args["window"] = (F(1, 40), F(3, 25))
    elif fault == "window_bool":
        args["window"] = (True, F(3, 25))
    elif fault == "session":
        args["session_id"] = "foreign"
    elif fault == "audio_policy":
        args["expected_audio_policy"] = "foreign"
    elif fault.startswith("limit_"):
        altered = replace(limits)
        object.__setattr__(altered, "max_requests" if fault == "limit_bool" else "preroll", True)
        args["limits"] = altered
    elif fault.startswith("context_"):
        context = cursor.context
        if fault in ("context_request_bool", "context_local_n", "context_request_rule"):
            request = replace(context.requests[0])
            key, value = {"context_request_bool": ("local_n", False), "context_local_n": ("local_n", 12),
                "context_request_rule": ("request_rule", "foreign")}[fault]
            object.__setattr__(request, key, value)
            context = replace(context, requests=(request, *context.requests[1:]))
        elif fault == "context_transport_bool":
            # slot_start is zero on the first consuming transport, so bool False
            # would compare equal without a strict field-type check.
            first = compiler.begin_source_stream_binding(assembly, slot_start=0, slot_end=6, origin=origin,
                session_id=context.session_id, window=context.window, expected_audio_policy="render48k-mono", limits=limits)
            cursor, context = first, first.context
            context = replace(context, transport=replace(context.transport, slot_start=False))
        elif fault == "context_raster_bool":
            first = compiler.begin_source_stream_binding(assembly, slot_start=0, slot_end=6, origin=origin,
                session_id=context.session_id, window=context.window, expected_audio_policy="render48k-mono", limits=limits)
            cursor, context = first, first.context
            context = replace(context, occurrence=forge_stream_record(context.occurrence, occurrence_index=False))
        elif fault == "context_size_float":
            context = replace(context, source_size=(2.0, 2.0))
        else:
            key, value = {"context_header": ("chain_header_json", "{}"),
                "context_source_size": ("source_size", [2, 2]),
                "context_transport": ("transport", replace(context.transport, k_start=0)),
                "context_occurrence": ("occurrence", replace(context.occurrence, occurrence_index=0)),
                "context_policy": ("policy", "foreign"), "context_requests_list": ("requests", list(context.requests)),
                "context_window_list": ("window", list(context.window))}[fault]
            context = replace(context, **{key: value})
        cursor = replace(cursor, context=context)
    elif fault == "cursor_bool":
        cursor = replace(cursor, next_delivery_index=False)
    elif fault == "cursor_negative":
        cursor = replace(cursor, selected_unique_bytes=-1)
    elif fault == "cursor_huge":
        cursor = replace(cursor, next_delivery_index=1 << 4097)
    elif fault == "cursor_skip":
        cursor = replace(cursor, next_request_ordinal=2, request_emission_bytes=24)
        selections[0] = forge_stream_record(selections[1], delivery_index=0)
    elif fault == "cursor_last_huge":
        cursor = replace(cursors[1], last_selected_frame_n=1 << 4095)
    with pytest.raises(ValueError):
        if fault.startswith(("context_", "cursor_")) or fault in ("missing_context", "missing_cursor"):
            compiler.bind_source_stream_selection(cursor, selection=selections[0])
        else:
            compiler.begin_source_stream_binding(**args)


def replace_origin(origin: Any) -> Any:
    return type(origin)(origin.path, origin.size, origin.mtime_ns, origin.fingerprint, origin.origin,
        origin.duration, origin.streams, origin.decoder_id, dict(origin.color))


def forge_stream_record(record: Any, **changes: Any) -> Any:
    """Negative-only constructor bypass to test the public trust boundary."""
    forged = object.__new__(type(record))
    for name in record.__dataclass_fields__:
        object.__setattr__(forged, name, changes.get(name, getattr(record, name)))
    return forged


@pytest.mark.parametrize("fault", ["foreign", "session", "index", "index_bool", "ordinals_empty",
    "ordinals_gap", "ordinals_bool", "ordinals_list", "ordinals_extra", "ordinal_not_int",
    "frame_none", "tag_none", "bgr_mutable", "bgr_corrupt", "bgr_short", "tag_bool", "tag_pts",
    "tag_tb", "tag_dimensions", "tag_adler", "raw_end", "selected_sha", "selected_bytes", "selected_pts",
    "selected_missing", "selected_array", "successor_sha", "successor_pts", "successor_closed",
    "chain", "foreign_chain", "old_cursor", "same_frame", "noncovering_group", "frame_n_limit",
    "missing_selection", "missing_frame", "missing_tag"])
def test_stream_binding_rejects_selection_byte_row_chain_or_order_corruption(fault: str) -> None:
    from clipperx.render_clock import canonical
    from clipperx.source_decode import ProvisionalSelection
    _, _, _, _, cursors, selections, _, _ = stream_binding_fixture()
    cursor, selection = cursors[0], selections[0]
    if fault.startswith("missing_"):
        if fault == "missing_selection":
            selection = object.__new__(type(selection))
        elif fault == "missing_frame":
            selection = forge_stream_record(selection, frame=object.__new__(type(selection.frame)))
        else:
            selection = forge_stream_record(selection, frame=forge_stream_record(selection.frame,
                tag=object.__new__(type(selection.frame.tag))))
    elif fault == "foreign":
        selection = object()
    elif fault in ("session", "index", "index_bool", "ordinals_empty", "ordinals_gap", "ordinals_bool",
                   "ordinals_list", "ordinals_extra", "ordinal_not_int", "frame_none", "chain"):
        key, value = {"session": ("session_id", "00000000-0000-4000-8000-000000000002"),
            "index": ("delivery_index", 1), "index_bool": ("delivery_index", False),
            "ordinals_empty": ("request_ordinals", ()), "ordinals_gap": ("request_ordinals", (0, 2)),
            "ordinals_bool": ("request_ordinals", (False, True)), "ordinals_list": ("request_ordinals", [0, 1]),
            "ordinals_extra": ("request_ordinals", tuple(range(7))), "ordinal_not_int": ("request_ordinals", ("0",)),
            "frame_none": ("frame", None), "chain": ("delivery_chain_sha256", "0" * 64)}[fault]
        selection = forge_stream_record(selection, **{key: value})
    elif fault.startswith(("tag_", "bgr_")) or fault == "raw_end":
        frame = forge_stream_record(selection.frame)
        if fault == "tag_none":
            object.__setattr__(frame, "tag", None)
        elif fault.startswith("tag_"):
            key, value = {"tag_bool": ("n", False), "tag_pts": ("pts", 5016), "tag_tb": ("time_base", True),
                "tag_dimensions": ("width", 4), "tag_adler": ("checksum", 0)}[fault]
            object.__setattr__(frame, "tag", forge_stream_record(frame.tag, **{key: value}))
        else:
            key, value = {"bgr_mutable": ("bgr", bytearray(frame.bgr)), "bgr_corrupt": ("bgr", bytes(12)),
                "bgr_short": ("bgr", bytes(3)), "raw_end": ("raw_end", F(6))}[fault]
            object.__setattr__(frame, key, value)
        selection = forge_stream_record(selection, frame=frame)
    elif fault.startswith(("selected_", "successor_")):
        row = selection.selected_record() if fault.startswith("selected_") else selection.successor_record()
        key, value = {"selected_sha": ("sha256", "0" * 64), "selected_bytes": ("bytes", 3),
            "selected_pts": ("pts", 5016), "selected_missing": ("n", None), "selected_array": ("n", None),
            "successor_sha": ("sha256", "bad"), "successor_pts": ("pts", 5051),
            "successor_closed": ("coverage_end", "513/100")}[fault]
        row[key] = value
        if fault == "selected_missing":
            row.pop("n")
        selection = forge_stream_record(selection, **{
            "selected_row_json" if fault.startswith("selected_") else "successor_row_json":
            canonical([] if fault == "selected_array" else row)})
    elif fault == "foreign_chain":
        selection = ProvisionalSelection(selection.session_id, 0, (0, 1), selection.frame,
            selection.selected_record(), selection.successor_record(), previous_chain_sha256="1" * 64)
    elif fault == "old_cursor":
        selection = selections[1]
    elif fault == "same_frame":
        cursor = cursors[1]
        selection = ProvisionalSelection(selection.session_id, 1, (2, 3), selection.frame,
            selection.selected_record(), selection.successor_record(), previous_chain_sha256=cursor.chain_sha256)
    elif fault == "noncovering_group":
        selection = ProvisionalSelection(selection.session_id, 0, tuple(range(6)), selection.frame,
            selection.selected_record(), selection.successor_record(), previous_chain_sha256=cursor.chain_sha256)
    elif fault == "frame_n_limit":
        frame = replace(selection.frame, tag=replace(selection.frame.tag, n=10))
        selection = ProvisionalSelection(selection.session_id, 0, (0, 1), frame,
            dict(selection.selected_record(), n=10), dict(selection.successor_record(), n=11),
            previous_chain_sha256=cursor.chain_sha256)
    with pytest.raises(ValueError):
        compiler.bind_source_stream_selection(cursor, selection=selection)
    assert cursors[0].next_request_ordinal == 0 and cursors[1].next_request_ordinal == 2


def reseal_stream_payload(origin: Any, limits: Any, session: str, payload: Any, frames: Any) -> Any:
    """Use the real producer constructors to repair unrelated links/accounting."""
    from clipperx import render_clock as clock
    from clipperx.source_decode import ProvisionalSelection
    previous = clock.digest(clock.canonical(clock._stream_header(origin, session, payload)))
    for index, delivery in enumerate(payload["deliveries"]):
        n = delivery["frame_n"]
        selection = ProvisionalSelection(session, index, delivery["request_ordinals"], frames[n],
            payload["video"][n], dict(payload["video"][n + 1], coverage_end=None), previous_chain_sha256=previous)
        payload["deliveries"][index] = selection.delivery_record()
        previous = selection.delivery_chain_sha256
    payload["accounting"]["metadata_bytes"] = len(clock.canonical({k: v for k, v in payload.items()
        if k not in ("accounting", "completion")}).encode("utf-8"))
    return clock.SourceStreamFinalization(origin, session, payload, limits=limits)


@pytest.mark.parametrize("fault", ["incomplete", "foreign", "origin_id", "decoder_id", "session", "policy",
    "payload_none", "payload_malformed", "window", "request", "limits", "audio_policy", "terminal_chain",
    "delivery_count", "request_count", "last_n", "unique_bytes", "logical_bytes", "payload_count",
    "payload_bytes", "payload_chain", "payload_completion", "payload_audio", "changed_successor_identity", "missing_final"])
def test_stream_binding_finalization_reconciles_actual_loader_and_cursor(fault: str) -> None:
    from clipperx import render_clock as clock
    from clipperx.source_decode import ProvisionalSelection
    _, origin, materialized, limits, cursors, selections, _, finalization = stream_binding_fixture()
    cursor = cursors[-1]
    if fault == "missing_final":
        finalization = object.__new__(type(finalization))
    elif fault == "incomplete":
        cursor = cursors[1]
    elif fault == "foreign":
        finalization = object()
    elif fault in ("origin_id", "decoder_id", "session", "policy", "payload_none", "payload_malformed"):
        changes: dict[str, tuple[str, Any]] = {"origin_id": ("origin_id", "0" * 64), "decoder_id": ("decoder_id", "foreign"),
            "session": ("session_id", "00000000-0000-4000-8000-000000000002"), "policy": ("policy", "foreign"),
            "payload_none": ("payload_json", None), "payload_malformed": ("payload_json", "{")}
        key, value = changes[fault]
        finalization = forge_stream_record(finalization, **{key: value})
    elif fault in ("window", "request", "limits", "audio_policy"):
        payload = json.loads(finalization.payload_json)
        session = finalization.session_id
        if fault == "window":
            # Still covers every request and remains valid producer metadata.
            payload["window"][0] = "19/1000"
            payload["audio"]["coverage"][0]["window"] = list(payload["window"])
        elif fault == "request":
            payload["requests"][0] = "21/1000"
            payload["outcomes"][0]["requested"] = "21/1000"
        elif fault == "limits":
            limits = replace(limits, max_selected_unique_bytes=100)
            payload["delivery_limits"] = limits.delivery_record()
        else:
            # Different valid producer audio policy: same PCM format/rate here.
            payload["audio"]["policy"] = "render48k-stereo"
            # This malformed audio is owned by the real loader, before adapter join.
            payload["accounting"]["metadata_bytes"] = len(clock.canonical({k: v for k, v in payload.items()
                if k not in ("accounting", "completion")}).encode("utf-8"))
            finalization = forge_stream_record(finalization, payload_json=clock.canonical(payload))
        if fault != "audio_policy":
            finalization = reseal_stream_payload(origin, limits, session, payload, materialized.frames)
    elif fault in ("terminal_chain", "delivery_count", "request_count", "last_n", "unique_bytes", "logical_bytes"):
        key, value = {"terminal_chain": ("chain_sha256", "1" * 64), "delivery_count": ("next_delivery_index", 1),
            "request_count": ("next_request_ordinal", 5), "last_n": ("last_selected_frame_n", 2),
            "unique_bytes": ("selected_unique_bytes", 12), "logical_bytes": ("request_emission_bytes", 60)}[fault]
        cursor = replace(cursor, **{key: value})
    elif fault.startswith("payload_"):
        payload = json.loads(finalization.payload_json)
        if fault == "payload_count":
            payload["accounting"]["acknowledged_requests"] = 5
        elif fault == "payload_bytes":
            payload["accounting"]["selected_unique_bytes"] = 12
        elif fault == "payload_chain":
            payload["deliveries"][-1]["chain_sha256"] = "1" * 64
        elif fault == "payload_completion":
            payload["completion"]["cleanup_complete"] = False
        elif fault == "payload_audio":
            payload["audio"]["bytes"] = 2
        finalization = forge_stream_record(finalization, payload_json=clock.canonical(payload))
    elif fault == "changed_successor_identity":
        import hashlib
        import zlib
        frame = materialized.frames[1]
        pixels = bytes([60]) * 12
        frame = replace(frame, tag=replace(frame.tag, checksum=zlib.adler32(pixels, 0)), bgr=pixels)
        row = dict(selections[1].selected_record(), checksum=f"{frame.tag.checksum:08X}",
            sha256=hashlib.sha256(pixels).hexdigest())
        changed = ProvisionalSelection(finalization.session_id, 1, (2, 3, 4, 5), frame, row,
            selections[1].successor_record(), previous_chain_sha256=cursors[1].chain_sha256)
        cursor, _ = compiler.bind_source_stream_selection(cursors[1], selection=changed)
        # Pure provisional bind has no retained row history. The terminal loader join
        # must reject the resulting chain against the finalized producer observations.
    with pytest.raises(ValueError):
        compiler.finalize_source_stream_binding(cursor, finalization=finalization)


@pytest.mark.parametrize("fault", ["frame_length", "frame_pixels", "frame_live", "unique_limit", "emission_limit",
    "delivery_limit", "row_bytes", "row_depth", "row_entries", "row_integer", "row_utf8", "row_type",
    "final_bytes", "final_depth", "final_entries", "final_integer", "final_utf8", "final_escaped_utf8", "final_header",
    "assembly_text", "assembly_entries", "origin_bits", "header_expansion", "outgoing_metadata",
    "rational_final_exponent", "rational_final_decimal", "rational_row_exponent", "rational_successor_exponent",
    "rational_complete_exponent", "rational_complete_negative_exponent", "rational_complete_zero_exponent"])
def test_stream_binding_caps_precede_clone_hash_or_parser(fault: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from clipperx import source_decode as decoder
    from clipperx import render_clock as clock
    assembly, origin, materialized, limits, cursors, selections, _, finalization = stream_binding_fixture()
    cursor, selection = cursors[0], selections[0]

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("cap must reject before constructor/parser/hash work")

    if fault.startswith("rational_"):
        bad = "0.02" if fault == "rational_final_decimal" else "1e-1000000" if fault == "rational_complete_negative_exponent" else (
            "0e1000000" if fault == "rational_complete_zero_exponent" else "1e1000000")
        original_exact, original_clock_exact = compiler.exact, clock.exact

        def bounded_exact(value: Any) -> Any:
            if value == bad:
                pytest.fail("rational expansion must be bounded before generic exact conversion")
            return original_exact(value)

        def bounded_clock_exact(value: Any) -> Any:
            if value == bad:
                pytest.fail("stream rational syntax must be checked before generic exact conversion")
            return original_clock_exact(value)

        monkeypatch.setattr(compiler, "exact", bounded_exact)
        monkeypatch.setattr(clock, "exact", bounded_clock_exact)
        if fault.startswith("rational_complete_"):
            payload = json.loads(materialized.receipt.payload_json)
            payload["windows"][0][0] = bad
            materialized = replace(materialized, receipt=forge_stream_record(materialized.receipt,
                payload_json=clock.canonical(payload)))
            _, _, _, old_limits = supplied_window()
            with pytest.raises(ValueError):
                compiler.bind_observed_batch(assembly, slot_start=6, slot_end=12, origin=origin,
                    decoded_window=materialized, expected_audio_policy="render48k-mono", limits=old_limits)
        elif fault in ("rational_row_exponent", "rational_successor_exponent"):
            row = selection.selected_record() if fault == "rational_row_exponent" else selection.successor_record()
            row["time_base"] = bad
            selection = forge_stream_record(selection, **{
                "selected_row_json" if fault == "rational_row_exponent" else "successor_row_json": clock.canonical(row)})
            with pytest.raises(ValueError):
                compiler.bind_source_stream_selection(cursor, selection=selection)
        else:
            payload = json.loads(finalization.payload_json)
            payload["window"][0] = bad
            finalization = forge_stream_record(finalization, payload_json=clock.canonical(payload))
            with pytest.raises(ValueError):
                compiler.finalize_source_stream_binding(cursors[-1], finalization=finalization)
        return
    if fault.startswith("frame_"):
        if fault == "frame_length":
            selection = forge_stream_record(selection, frame=forge_stream_record(selection.frame, bgr=bytes(24)))
        elif fault == "frame_pixels":
            selection = forge_stream_record(selection, frame=forge_stream_record(selection.frame,
            tag=forge_stream_record(selection.frame.tag, width=4)))
        else:
            limits = replace(limits, max_live_payload_bytes=11)
            cursor = compiler.begin_source_stream_binding(cursor.context.assembly, slot_start=6, slot_end=12,
                origin=origin, session_id=cursor.context.session_id, window=cursor.context.window,
                expected_audio_policy="render48k-mono", limits=limits)
        monkeypatch.setattr(decoder.ProvisionalSelection, "__init__", forbidden)
    elif fault in ("unique_limit", "emission_limit", "delivery_limit"):
        limits = replace(limits, **{"max_selected_unique_bytes" if fault == "unique_limit" else
            "max_request_emission_bytes" if fault == "emission_limit" else "max_deliveries":
            11 if fault == "unique_limit" else 23 if fault == "emission_limit" else 1})
        initial = compiler.begin_source_stream_binding(cursor.context.assembly, slot_start=6, slot_end=12,
            origin=origin, session_id=cursor.context.session_id, window=cursor.context.window,
            expected_audio_policy="render48k-mono", limits=limits)
        cursor = initial if fault != "delivery_limit" else replace(cursors[1], context=initial.context)
        if fault == "delivery_limit":
            selection = selections[1]
        monkeypatch.setattr(decoder.ProvisionalSelection, "__init__", forbidden)
    elif fault.startswith("row_"):
        value = {"row_bytes": " " * 16385, "row_depth": "[" * 25 + "0" + "]" * 25,
            "row_entries": "[" + ",".join("0" for _ in range(40)) + "]",
            "row_integer": '{"n":' + "9" * 1236 + "}", "row_utf8": '"\ud800"', "row_type": None}[fault]
        selection = forge_stream_record(selection, selected_row_json=value)
        if fault == "row_entries":
            # Revalidate context first under normal bounds, then lower only the JSON work budget.
            monkeypatch.setattr(compiler, "_checked_stream_cursor", lambda value: value.context)
            monkeypatch.setattr(compiler, "MAX_METADATA_ENTRIES", 32)
        monkeypatch.setattr(decoder.ProvisionalSelection, "selected_record", forbidden)
        monkeypatch.setattr(compiler.json, "loads", forbidden)
    elif fault.startswith("final_"):
        cursor = cursors[-1]
        value = {"final_bytes": " " * 4097, "final_depth": "[" * 25 + "0" + "]" * 25,
            "final_entries": "[" + ",".join("0" for _ in range(40)) + "]",
            "final_integer": '{"n":' + "9" * 1236 + "}", "final_utf8": '"\ud800"',
            "final_escaped_utf8": '{"extra":"\\ud800"}', "final_header": finalization.payload_json}[fault]
        finalization = forge_stream_record(finalization, payload_json=value,
            decoder_id="x" * 4097 if fault == "final_header" else finalization.decoder_id)
        monkeypatch.setattr(compiler, "_checked_stream_cursor", lambda value: value.context)
        if fault in ("final_bytes", "final_header"):
            monkeypatch.setattr(compiler, "MAX_RECEIPT_BYTES", 4096)
        if fault == "final_entries":
            monkeypatch.setattr(compiler, "MAX_METADATA_ENTRIES", 32)
        monkeypatch.setattr(clock.SourceStreamFinalization, "to_json", forbidden)
        if fault != "final_escaped_utf8":
            monkeypatch.setattr(compiler.json, "loads", forbidden)
    elif fault.startswith("assembly_") or fault in ("origin_bits", "header_expansion"):
        assembly = cursor.context.assembly
        if fault == "assembly_text":
            assembly = replace(assembly, canonical_ass="x" * 4097)
        elif fault == "assembly_entries":
            assembly = replace(assembly, shots=(assembly.shots[0],) * 40)
        elif fault == "origin_bits":
            origin = forge_stream_record(origin, size=1 << 4097)
        else:
            text = compiler._guard_known_input(origin, record_types=(compiler.SourceTimeOrigin, compiler.StreamClock))
            monkeypatch.setattr(compiler, "MAX_INPUT_TEXT_BYTES", text * 6 + 4097)
        if fault == "assembly_text":
            monkeypatch.setattr(compiler, "MAX_INPUT_TEXT_BYTES", 4096)
        if fault == "assembly_entries":
            monkeypatch.setattr(compiler, "MAX_METADATA_ENTRIES", 32)
        monkeypatch.setattr(compiler, "_stream_header", forbidden)
        with pytest.raises(ValueError):
            compiler.begin_source_stream_binding(assembly, slot_start=6, slot_end=12,
                origin=origin, session_id=cursor.context.session_id, window=cursor.context.window,
                expected_audio_policy="render48k-mono", limits=limits)
        return
    else:
        # A tiny lowered limit at the explicit outgoing metadata gate proves that
        # result metadata is checked without walking or serializing borrowed BGR.
        guard = compiler._guard_stream_metadata

        def result_guard(value: Any) -> int:
            if type(value) is tuple and any(type(v) is compiler.ProvisionalScheduledSlot for v in value):
                raise ValueError("lowered outgoing slot budget")
            return guard(value)

        monkeypatch.setattr(compiler, "_guard_stream_metadata", result_guard)
    with pytest.raises(ValueError):
        if fault.startswith("final_"):
            compiler.finalize_source_stream_binding(cursor, finalization=finalization)
        else:
            compiler.bind_source_stream_selection(cursor, selection=selection)


def test_stream_binding_retains_no_pixels_and_has_honest_value_semantics() -> None:
    from dataclasses import fields, is_dataclass
    from clipperx.source_decode import ObservedFrame, ProvisionalSelection
    from clipperx.render_clock import SourceStreamFinalization
    _, origin, materialized, limits, cursors, selections, deliveries, finalization = stream_binding_fixture()
    context = cursors[0].context
    # Constructor inputs detach; one original frame is intentionally borrowed.
    ordinals = [0, 1]
    selected, successor = selections[0].selected_record(), selections[0].successor_record()
    detached = ProvisionalSelection(context.session_id, 0, ordinals, materialized.frames[0], selected,
        successor, previous_chain_sha256=cursors[0].chain_sha256)
    ordinals.clear()
    selected.clear()
    successor.clear()
    assert detached == selections[0]
    payload = json.loads(finalization.payload_json)
    detached_final = SourceStreamFinalization(origin, context.session_id, payload, limits=limits)
    payload.clear()
    assert detached_final.sha256 == finalization.sha256
    replay_cursor, replay_delivery = compiler.bind_source_stream_selection(cursors[0], selection=detached)
    assert replay_cursor == cursors[1] and replay_delivery == deliveries[0]
    assert replay_delivery.frame is materialized.frames[0]
    assert replay_delivery.frame.bgr is materialized.frames[0].bgr
    final = compiler.finalize_source_stream_binding(cursors[-1], finalization=finalization)
    assert compiler.finalize_source_stream_binding(cursors[-1], finalization=finalization) == final

    def metadata_only(value: Any) -> None:
        assert type(value) not in (ObservedFrame, bytes, bytearray, memoryview, list, dict)
        if is_dataclass(value):
            for field in fields(value):
                metadata_only(getattr(value, field.name))
        elif type(value) is tuple:
            for child in value:
                metadata_only(child)

    for value in (*cursors, final):
        metadata_only(value)
    assert not hasattr(final, "verified_pcm_bytes")
    assert len(deliveries[0].slots) == 2 and all(not hasattr(slot, "frame") for slot in deliveries[0].slots)
    assert not hasattr(context, "__dict__") and not hasattr(cursors[-1], "__dict__")


@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("old_time", ["101/50", "2.02", "2.02e0"])
def test_stream_binding_complete_batch_regression_and_local_n(partial: bool, old_time: str) -> None:
    from clipperx.render_clock import SourceDecodeReceipt
    assembly, origin, materialized, _, cursors, _, deliveries, finalization = stream_binding_fixture(F(2), partial=partial)
    _, _, _, old_limits = supplied_window(F(2))
    track = assembly.shots[0].source_layers[0].crop_track
    assert track is not None and track.points[1].frame == 120
    if partial:
        payload = json.loads(materialized.receipt.payload_json)
        payload["outcomes"] = payload["outcomes"][3:]
        materialized = replace(materialized, frames=(materialized.frames[1],), receipt=SourceDecodeReceipt(
            materialized.receipt.origin_id, materialized.receipt.decoder_id, materialized.receipt.policy, payload))
    # COMPLETE accepts equivalent bounded decimal/exponential window coordinates;
    # the new stream format continues to require its own canonical rational syntax.
    payload = json.loads(materialized.receipt.payload_json)
    payload["windows"][0][0] = old_time
    payload["audio"][0]["coverage"][0]["window"] = list(payload["windows"][0])
    materialized = replace(materialized, receipt=SourceDecodeReceipt(materialized.receipt.origin_id,
        materialized.receipt.decoder_id, materialized.receipt.policy, payload))
    old = compiler.bind_observed_batch(assembly, slot_start=123 if partial else 120, slot_end=126,
        origin=origin, decoded_window=materialized, expected_audio_policy="render48k-mono", limits=old_limits)
    final = compiler.finalize_source_stream_binding(cursors[-1], finalization=finalization)
    expected = list(range(123 if partial else 120, 126))
    assert [s.request.local_n for s in final.slots] == expected
    assert [s.request.global_k for s in final.slots] == expected
    assert [s.transport_slot for s in final.slots] == expected
    assert [s.frame_n for s in final.slots] == [b.frame_n for b in old.bindings]
    assert [s.request for s in final.slots] == [b.request for b in old.bindings]
    assert [d.frame.bgr for d in deliveries] == [f.bgr for f in materialized.frames]
    assert all(b.frame is next(f for f in materialized.frames if f.tag.n == b.frame_n) for b in old.bindings)
    assert all(b.receipt_hash == old.receipt_hash for b in old.bindings)
    assert old.verified_retained_bytes == (14412 if partial else 14424)
    assert old.verified_pcm_bytes == 14400


def test_complete_binder_real_receipt_bytes_and_distinct_occurrences() -> None:
    origin, window, pixels, limits = supplied_window()
    plan = pure_plan(("source_video", "solid", "source_video"))
    assembly = compiler.compile_visual_assembly(plan, source_size=(2, 2), overlays=pure_overlays(plan))
    batch = compiler.bind_observed_batch(assembly, slot_start=0, slot_end=6, origin=origin,
        decoded_window=window, expected_audio_policy="render48k-mono", limits=limits)
    assert [b.frame.bgr for b in batch.bindings] == [pixels[0]] * 2 + [pixels[1]] * 4
    assert [b.request.local_n for b in batch.bindings] == list(range(6))
    assert [b.request.global_k for b in batch.bindings] == list(range(6))
    assert batch.bindings[0].frame is window.frames[0]
    assert batch.bindings[0].frame is batch.bindings[1].frame
    assert batch.verified_retained_bytes == 14424 and batch.verified_pcm_bytes == 14400
    assert not hasattr(batch, "pcm")
    repeated = compiler.bind_observed_batch(assembly, slot_start=6, slot_end=12, origin=origin,
        decoded_window=window, expected_audio_policy="render48k-mono", limits=limits)
    assert [b.transport_slot for b in repeated.bindings] == list(range(6, 12))
    assert [b.request.global_k for b in repeated.bindings] == list(range(12, 18))
    assert [b.request.local_n for b in repeated.bindings] == list(range(6))
    assert all(b.receipt_hash == batch.receipt_hash for b in batch.bindings)


@pytest.mark.parametrize("fault", ["origin", "policy", "fingerprint", "dimensions", "pts", "tb",
    "end", "checksum", "bgr", "frame_sha", "duplicate", "extra", "missing", "bool_n", "mutable",
    "pcm", "pcm_short", "pcm_aggregate", "pcm_block_sha", "pcm_block_adler", "audio_policy",
    "outcomes_missing", "outcomes_extra", "outcomes_order", "unavailable", "limit", "slot_bool", "cross_shot"])
def test_complete_binder_rejects_independent_actual_body_or_association_corruption(fault: str) -> None:
    from clipperx.render_clock import SourceDecodeReceipt
    from clipperx.source_decode import DecodedWindow
    origin, window, _, limits = supplied_window()
    plan = pure_plan(("source_video", "source_video"))
    assembly = compiler.compile_visual_assembly(plan, source_size=(2, 2), overlays=pure_overlays(plan))
    payload = json.loads(window.receipt.payload_json)
    frames = window.frames
    pcm = window.pcm
    slot_start, slot_end = 0, 6
    if fault == "origin":
        origin = type(origin)("OTHER", origin.size, origin.mtime_ns, origin.fingerprint, origin.origin,
            origin.duration, origin.streams, origin.decoder_id, dict(origin.color))
    elif fault == "policy":
        object.__setattr__(window.receipt, "policy", "untrusted")
    elif fault == "fingerprint":
        assembly = replace(assembly, expected_source_fingerprint="other")
    elif fault in ("dimensions", "pts", "tb", "checksum", "bool_n"):
        tag = frames[0].tag
        tag_changes: dict[str, dict[str, Any]] = {"dimensions": dict(width=4), "pts": dict(pts=5016), "tb": dict(time_base=F(1, 999)),
                   "checksum": dict(checksum=0), "bool_n": dict(n=False)}
        # Deliberately bypass constructor convenience checks: the boundary must revalidate records.
        object.__setattr__(frames[0], "tag", replace(tag, **tag_changes[fault]))
    elif fault == "end":
        object.__setattr__(frames[0], "raw_end", F(6))
    elif fault == "bgr":
        object.__setattr__(frames[0], "bgr", bytes(12))
    elif fault == "mutable":
        object.__setattr__(frames[0], "bgr", bytearray(frames[0].bgr))
    elif fault == "frame_sha":
        payload["video"][0]["sha256"] = "0" * 64
    elif fault == "duplicate":
        frames = (*frames, frames[0])
    elif fault == "extra":
        frames = (*frames, replace(frames[1], tag=replace(frames[1].tag, n=2)))
    elif fault == "missing":
        frames = frames[1:]
    elif fault == "pcm":
        pcm = b"x" + pcm[1:]
    elif fault == "pcm_short":
        pcm = pcm[:-2]
    elif fault == "pcm_aggregate":
        payload["audio"][0]["pcm_sha256"] = "0" * 64
    elif fault == "pcm_block_sha":
        payload["audio"][0]["output"][0]["sha256"] = "0" * 64
    elif fault == "pcm_block_adler":
        payload["audio"][0]["output"][0]["checksum"] = "00000001"
    elif fault == "audio_policy":
        pass
    elif fault == "outcomes_missing":
        payload["outcomes"].pop()
    elif fault == "outcomes_extra":
        payload["outcomes"].append(payload["outcomes"][0])
    elif fault == "outcomes_order":
        payload["outcomes"].reverse()
    elif fault == "unavailable":
        payload["outcomes"][0].update(status="missing_predecessor", frame_n=None)
    elif fault == "limit":
        limits = replace(limits, max_retained_bytes=14423)
    elif fault == "slot_bool":
        slot_start = False
    elif fault == "cross_shot":
        slot_end = 7
    if payload != json.loads(window.receipt.payload_json):
        window = replace(window, receipt=SourceDecodeReceipt(window.receipt.origin_id,
            window.receipt.decoder_id, window.receipt.policy, payload))
    window = DecodedWindow(window.receipt, frames, pcm)
    with pytest.raises(ValueError):
        compiler.bind_observed_batch(assembly, slot_start=slot_start, slot_end=slot_end, origin=origin,
            decoded_window=window, expected_audio_policy="analysis16k-mono" if fault == "audio_policy"
            else "render48k-mono", limits=limits)


@pytest.mark.parametrize("cap,value", [("MAX_PRIMITIVES", 0), ("MAX_EVENTS", 0),
    ("MAX_BRANDS", 0), ("MAX_ASS_RESOURCES", 0), ("MAX_STEPS", 1),
    ("MAX_EVENT_CHARS", 2), ("MAX_INPUT_TEXT_BYTES", 10), ("MAX_GENERATED_BYTES", 100),
    ("MAX_METADATA_ENTRIES", 2)])
def test_host_caps_reject_before_clone_geometry_or_serialization(
        cap: str, value: int, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = interleaved_plan()
    overlays = pure_overlays(plan)
    def forbidden(*args: object, **kwargs: object) -> Any:
        pytest.fail("Cap must reject before expensive work")
    monkeypatch.setattr(compiler, cap, value)
    monkeypatch.setattr(compiler, "resolve_visual_plan", forbidden)
    monkeypatch.setattr(compiler, "serialize_overlay_events", forbidden)
    monkeypatch.setattr(compiler, "compile_source_layer", forbidden)
    with pytest.raises(ValueError, match="limit|budget|cap"):
        compiler.compile_visual_assembly(plan, source_size=(2, 2), overlays=overlays)


@pytest.mark.parametrize("fault", ["cycle", "huge_integer", "injected_nested", "path", "bool_z"])
def test_known_structure_guard_precedes_clone(fault: str, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = interleaved_plan()
    overlays = pure_overlays(plan)
    if fault == "cycle":
        plan.shots[0].__dict__["layers"].append(plan.shots[0].layers)
    elif fault == "huge_integer":
        plan.shots[0].layers[0].__dict__["z_index"] = 1 << 4097
    elif fault == "injected_nested":
        plan.shots[0].__dict__["foreign"] = {"nested": "bad"}
    elif fault == "path":
        overlays = replace(overlays, font_paths=(Path("over-limit-component"),))
        monkeypatch.setattr(compiler, "MAX_INPUT_TEXT_BYTES", 5)
    elif fault == "bool_z":
        plan.shots[0].layers[0].__dict__["z_index"] = True
    def forbidden(*args: object, **kwargs: object) -> Any:
        pytest.fail("Known shape must be checked before clone")
    monkeypatch.setattr(compiler, "resolve_visual_plan", forbidden)
    with pytest.raises(ValueError):
        compiler.compile_visual_assembly(plan, source_size=(2, 2), overlays=overlays)


@pytest.mark.parametrize("cap,value", [("MAX_RECEIPT_BYTES", 100), ("MAX_BATCH_REQUESTS", 2),
                                       ("MAX_RETAINED_BYTES", 100)])
def test_binder_caps_precede_receipt_parser_or_hash(
        cap: str, value: int, monkeypatch: pytest.MonkeyPatch) -> None:
    from clipperx.render_clock import SourceDecodeReceipt
    origin, window, _, limits = supplied_window()
    plan = pure_plan()
    assembly = compiler.compile_visual_assembly(plan, source_size=(2, 2), overlays=pure_overlays(plan))
    def forbidden(*args: object, **kwargs: object) -> Any:
        pytest.fail("Binder cap must reject before parser/hash")
    monkeypatch.setattr(compiler, cap, value)
    monkeypatch.setattr(SourceDecodeReceipt, "from_json", forbidden)
    with pytest.raises(ValueError, match="limit|budget|cap"):
        compiler.bind_observed_batch(assembly, slot_start=0, slot_end=6, origin=origin,
            decoded_window=window, expected_audio_policy="render48k-mono", limits=limits)


@pytest.mark.parametrize("fault", ["rect_bool", "line_count", "word_empty", "asset_list", "style_control"])
def test_overlay_record_scalar_and_slice_integrity(fault: str) -> None:
    from clipperx.overlay_ass import OverlayCompilation, serialize_overlay_events
    plan = interleaved_plan()
    if fault == "rect_bool":
        assert plan.shots[0].title is not None
        plan.shots[0].title.destination.x = 0
    overlays = pure_overlays(plan)
    event = overlays.events[0]
    if fault == "rect_bool":
        object.__setattr__(event.destination, "x", False)
        # Pixel bool must fail validation even where bool equals its planned integer.
    elif fault == "line_count":
        event = replace(event, text="first\nsecond", lines=("first\nsecond",), line_ranges=((0, 12),))
        assert plan.shots[0].title is not None
        plan.shots[0].title.text = "first\nsecond"
    elif fault == "word_empty":
        object.__setattr__(event, "line_ranges", None)
    elif fault == "asset_list":
        object.__setattr__(overlays.brands[0].asset, "size", [4, 4])
    elif fault == "style_control":
        event = replace(event, font_family="HOST\u0085Injected")
    document = overlays.document if fault == "word_empty" else serialize_overlay_events((event,))
    overlays = OverlayCompilation(document, (event,), (), overlays.brands)
    with pytest.raises(ValueError):
        compiler.compile_visual_assembly(plan, source_size=(2, 2), overlays=overlays)


def test_later_complete_batch_retains_original_motion_n120_and_partial_local_indices() -> None:
    from clipperx.render_clock import SourceDecodeReceipt
    from clipperx.visual_contracts import CropKeyframe, NormalizedRect
    origin, window, _, limits = supplied_window(F(2))
    plan = pure_plan(duration=2.1)
    plan.shots[0].layers[0].crop_track = [CropKeyframe(at=0,
        crop=NormalizedRect(x=0, y=0, width=1, height=1)), CropKeyframe(at=2,
        crop=NormalizedRect(x=0, y=0, width=1, height=1)), CropKeyframe(at=2.1,
        crop=NormalizedRect(x=0, y=0, width=1, height=1))]
    assembly = compiler.compile_visual_assembly(plan, source_size=(2, 2), overlays=pure_overlays(plan))
    track = assembly.shots[0].source_layers[0].crop_track
    assert track is not None and track.points[1].frame == 120
    batch = compiler.bind_observed_batch(assembly, slot_start=120, slot_end=126, origin=origin,
        decoded_window=window, expected_audio_policy="render48k-mono", limits=limits)
    assert [b.request.local_n for b in batch.bindings] == list(range(120, 126))
    assert [b.request.global_k for b in batch.bindings] == list(range(120, 126))
    data = json.loads(window.receipt.payload_json)
    data["outcomes"] = data["outcomes"][3:]
    partial = replace(window, receipt=SourceDecodeReceipt(window.receipt.origin_id,
        window.receipt.decoder_id, window.receipt.policy, data), frames=(window.frames[1],))
    tail = compiler.bind_observed_batch(assembly, slot_start=123, slot_end=126, origin=origin,
        decoded_window=partial, expected_audio_policy="render48k-mono", limits=limits)
    assert [b.request.local_n for b in tail.bindings] == [123, 124, 125]
    assert all(b.frame is window.frames[1] for b in tail.bindings)


def test_complete_pcm_ordered_blocks_and_unavailable_coverage_are_honest() -> None:
    import hashlib
    import zlib
    from clipperx.render_clock import SourceDecodeReceipt
    origin, window, _, limits = supplied_window()
    plan = pure_plan()
    assembly = compiler.compile_visual_assembly(plan, source_size=(2, 2), overlays=pure_overlays(plan))
    payload = json.loads(window.receipt.payload_json)
    audio = payload["audio"][0]
    first = dict(audio["output"][0], samples=3600, bytes=7200,
                 sha256=hashlib.sha256(bytes(7200)).hexdigest(), checksum="00000000")
    second = dict(first, n=1, pts=first["pts"] + 3600)
    audio["output"] = [first, second]
    receipt = SourceDecodeReceipt(window.receipt.origin_id, window.receipt.decoder_id,
                                  window.receipt.policy, payload)
    batch = compiler.bind_observed_batch(assembly, slot_start=0, slot_end=6, origin=origin,
        decoded_window=replace(window, receipt=receipt), expected_audio_policy="render48k-mono", limits=limits)
    assert batch.verified_pcm_sha256 == audio["pcm_sha256"]
    audio["output"][1]["checksum"] = "00000001"
    corrupt = SourceDecodeReceipt(window.receipt.origin_id, window.receipt.decoder_id,
                                  window.receipt.policy, payload)
    with pytest.raises(ValueError, match="block"):
        compiler.bind_observed_batch(assembly, slot_start=0, slot_end=6, origin=origin,
            decoded_window=replace(window, receipt=corrupt), expected_audio_policy="render48k-mono", limits=limits)
    pcm = bytes(9600)
    payload = json.loads(window.receipt.payload_json)
    audio = payload["audio"][0]
    audio["input"][0]["samples"] = 4410
    audio["output"][0].update(samples=4800, bytes=len(pcm), sha256=hashlib.sha256(pcm).hexdigest(),
                              checksum=f"{zlib.adler32(pcm, 0):08X}")
    audio.update(samples=4800, bytes=len(pcm), pcm_sha256=hashlib.sha256(pcm).hexdigest())
    audio["coverage"][0]["status"] = "unavailable"
    receipt = SourceDecodeReceipt(window.receipt.origin_id, window.receipt.decoder_id,
                                  window.receipt.policy, payload)
    unavailable = compiler.bind_observed_batch(assembly, slot_start=0, slot_end=6, origin=origin,
        decoded_window=replace(window, receipt=receipt, pcm=pcm), expected_audio_policy="render48k-mono", limits=limits)
    assert unavailable.verified_pcm_bytes == 9600
    assert json.loads(receipt.payload_json)["audio"][0]["coverage"][0]["status"] == "unavailable"


def test_exact_title_subtitle_boundary_and_stable_global_interleaving() -> None:
    from clipperx.filtergraph import rect_to_pixels
    from clipperx.overlay_ass import OverlayCompilation, serialize_overlay_events
    from clipperx.visual_contracts import NormalizedRect, SubtitlePlan
    plan = interleaved_plan()
    shot = plan.shots[0]
    assert shot.title is not None and shot.brand is not None
    shot.title.start, shot.title.end = .225, .3
    shot.layers[1].z_index, shot.brand.z_index = 2, 3
    upper = shot.layers[0].model_copy(deep=True)
    upper.layer_id, upper.z_index = "upper", 4
    shot.layers.append(upper)
    shot.subtitles = SubtitlePlan(destination=NormalizedRect(x=.1, y=.7, width=.8, height=.1))
    overlays = pure_overlays(plan)
    title = overlays.events[0]
    interval = (F(9, 40) - F(1, 10**18), F(3, 10))
    subtitle = replace(title, kind="subtitle", start=float(interval[0]), end=float(interval[1]),
        text="hello", lines=("hello",), line_ranges=((0, 5),), font_size=58, outline_width=4,
        primary_color="&H001AD5FF", word_count=1, layer=5,
        destination=rect_to_pixels(shot.subtitles.destination, width=1080, height=1920), exact_interval=interval)
    events = (subtitle, title)
    overlays = OverlayCompilation(serialize_overlay_events(events), events, (), overlays.brands)
    assembly = compiler.compile_visual_assembly(plan, source_size=(2, 2), overlays=overlays)
    assert [r.windows for r in assembly.ass_documents] == [((14, 18),), ((13, 18),)]
    assert_outer_rgb_graph(assembly)
    operations = json.loads(assembly.compiler_metadata_json)["shots"][0]["operations"]
    assert [op["kind"] for op in operations] == ["source_video", "ass", "streamer_camera", "brand", "source_video", "ass"]
    # Equal source z retains original declarations; source precedes title at equal z.
    shot.layers[1].z_index, shot.layers[2].z_index = 0, 0
    shot.title.z_index = 0
    silence = pure_overlays(plan)
    equal = compiler.compile_visual_assembly(plan, source_size=(2, 2), overlays=silence)
    ops = json.loads(equal.compiler_metadata_json)["shots"][0]["operations"]
    assert [op.get("layer_index") for op in ops[:3]] == [0, 1, 2]
    assert ops[3]["kind"] == "ass"


def test_origin_text_reservation_precedes_identity_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    from clipperx.render_clock import SourceTimeOrigin
    origin, window, _, limits = supplied_window()
    plan = pure_plan()
    assembly = compiler.compile_visual_assembly(plan, source_size=(2, 2), overlays=pure_overlays(plan))
    object.__setattr__(origin, "path", "\u2000" * 1500)
    monkeypatch.setattr(compiler, "MAX_INPUT_TEXT_BYTES", 10000)
    def forbidden(self: object) -> str:
        pytest.fail("Origin text expansion must be reserved before hashing")
    monkeypatch.setattr(SourceTimeOrigin, "identity", property(forbidden))
    with pytest.raises(ValueError, match="limit|budget"):
        compiler.bind_observed_batch(assembly, slot_start=0, slot_end=6, origin=origin,
            decoded_window=window, expected_audio_policy="render48k-mono", limits=limits)


@pytest.mark.parametrize("fault", ["layers_none", "foreign_layer", "shots_none", "size_none",
    "bound_shot_none", "bound_span_bool", "bound_clock", "origin_fraction", "origin_streams"])
def test_entrypoints_reject_malformed_known_shapes_with_value_error(fault: str) -> None:
    plan = pure_plan()
    overlays = pure_overlays(plan)
    if fault in ("layers_none", "foreign_layer", "shots_none", "size_none"):
        if fault == "layers_none":
            plan.shots[0].__dict__["layers"] = None
        elif fault == "foreign_layer":
            plan.shots[0].__dict__["layers"] = ["foreign"]
        elif fault == "shots_none":
            plan.__dict__["shots"] = None
        bad_size: Any = None if fault == "size_none" else (2, 2)
        with pytest.raises(ValueError):
            compiler.compile_visual_assembly(plan, source_size=bad_size, overlays=overlays)
        return
    origin, window, _, limits = supplied_window()
    assembly = compiler.compile_visual_assembly(plan, source_size=(2, 2), overlays=overlays)
    if fault == "bound_shot_none":
        object.__setattr__(assembly, "shots", (None,))
    elif fault == "bound_span_bool":
        object.__setattr__(assembly.shots[0].span, "k_start", False)
    elif fault == "bound_clock":
        object.__setattr__(assembly.shots[0], "output_clock", "global")
    elif fault == "origin_fraction":
        object.__setattr__(origin, "origin", "5/1")
    elif fault == "origin_streams":
        object.__setattr__(origin, "streams", None)
    with pytest.raises(ValueError):
        compiler.bind_observed_batch(assembly, slot_start=0, slot_end=6, origin=origin,
            decoded_window=window, expected_audio_policy="render48k-mono", limits=limits)


def test_assembly_reuses_A_for_repeated_out_of_order_and_leading_raster_requests() -> None:
    plan = pure_plan(("source_video", "source_video", "source_video"), duration=.024)
    plan.shots[1].source.start, plan.shots[1].source.end = .5, .524
    result = compiler.compile_visual_assembly(plan, source_size=(2, 2), overlays=pure_overlays(plan))
    assert [(s.span.k_start, s.span.k_end) for s in result.shots] == [(0, 1), (1, 3), (3, 4)]
    requests = [result.timeline.video_request(k) for k in range(4)]
    assert [r.shot_id for r in requests] == ["pure0", "pure1", "pure1", "pure2"]
    assert requests[1].requested_source == F(1, 2)
    assert requests[1].request_rule == "declared_start_for_leading_raster_cell"
    assert requests[3].requested_source == F(11, 500)


@pytest.mark.parametrize("field", ["measured_width", "measured_height", "start", "end"])
def test_overlay_large_integer_rejected_as_preflight_before_geometry_or_serializer(
        field: str, monkeypatch: pytest.MonkeyPatch) -> None:
    from clipperx.overlay_ass import OverlayCompilation
    plan = interleaved_plan()
    overlays = pure_overlays(plan)
    event = replace(overlays.events[0], **{field: 1 << 4000})
    supplied = OverlayCompilation(overlays.document, (event,), (), overlays.brands)

    def forbidden(*args: object, **kwargs: object) -> Any:
        pytest.fail("Invalid overlay integer reached geometry/serializer")

    for name in ("rect_to_pixels", "compile_source_layer", "serialize_overlay_events"):
        monkeypatch.setattr(compiler, name, forbidden)
    with pytest.raises(ValueError, match="metadata"):
        compiler.compile_visual_assembly(plan, source_size=(2, 2), overlays=supplied)


def host_plan(source: Path) -> VisualCompositionPlan:
    return VisualCompositionPlan.model_validate({
        "clip_id": "host_fixture", "source_fingerprint": source_fingerprint(source),
        "composition_fingerprint": "host", "source_timeline_fingerprint": "host",
        "identity_fingerprint": "host", "style_library_fingerprint": "host",
        "shots": [{"shot_id": "host_shot", "source": {"start": 10, "end": 12},
                   "output": {"start": 0, "end": 2}, "primary_source": "content",
                   "reason": "host-authored fixture", "evidence_ids": ["host_frame"],
                   "confidence": 1, "layers": [{"layer_id": "original", "kind": "source_video",
                   "source_ref": "source", "source_crop": {"x": 0, "y": 0, "width": 1,
                   "height": 1}, "destination": {"x": 0, "y": 0, "width": 1, "height": 1},
                   "fill": "cover", "z_index": 0}]}],
    })


def two_shot_plan(source: Path) -> VisualCompositionPlan:
    payload = host_plan(source).model_dump()
    import copy
    second = copy.deepcopy(payload["shots"][0])
    second.update(shot_id="second", source={"start": 20.125, "end": 22.125},
                  output={"start": 2, "end": 4})
    payload["shots"].append(second)
    return VisualCompositionPlan.model_validate(payload)


def test_inherit_resolves_same_layer_without_mutating_plan(source: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    fake_tools(monkeypatch)
    plan = two_shot_plan(source)
    plan.shots[0].layers[0].fill = compiler.FillMode.CONTAIN_BLUR
    plan.shots[0].layers[0].blur_sigma = 12
    plan.shots[1].layers[0].fill = compiler.FillMode.INHERIT
    before = plan.model_dump_json()
    resolved = compiler.resolve_visual_plan(plan)
    assert resolved.shots[1].layers[0].fill == compiler.FillMode.CONTAIN_BLUR
    assert resolved.shots[1].layers[0].blur_sigma == 12
    compiler.preflight_visual_plan(plan, source=source, assets=default_render_assets())
    assert plan.model_dump_json() == before
    resolved.shots[0].layers[0].destination.width = .5
    assert plan.model_dump_json() == before


def test_inherit_missing_previous_layer_fails_before_tools(source: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake_tools(monkeypatch)
    plan = two_shot_plan(source)
    plan.shots[1].layers[0].layer_id = "missing"
    plan.shots[1].layers[0].fill = compiler.FillMode.INHERIT
    with pytest.raises(compiler.RenderPreflightError, match="inherit.*missing"):
        compiler.preflight_visual_plan(plan, source=source, assets=default_render_assets())
    assert calls == []


def test_static_shot_compilation_uses_explicit_geometry_z_order_and_exact_times(source: Path) -> None:
    from clipperx.filtergraph import layer_label
    plan = two_shot_plan(source)
    import copy
    foreground = copy.deepcopy(plan.shots[1].layers[0])
    foreground.layer_id = "foreground];movie=bad"
    foreground.z_index = 5
    foreground.destination.height = .5
    plan.shots[1].layers.insert(0, foreground)
    before = plan.model_dump_json()
    result = compiler.compile_visual_layers(plan, source_size=(1920, 1080))
    shot = result[1]
    assert "trim=start=20.125:end=22.125" in shot.filter_complex
    assert "setpts=PTS-20.125/TB" in shot.filter_complex
    assert "color=c=black:s=1080x1920:r=60:d=2" in shot.filter_complex
    assert "setsar=1,fps=60" in shot.filter_complex
    assert shot.layer_ids == ("original", "foreground];movie=bad")
    assert (shot.source_start, shot.source_end, shot.output_start, shot.output_end) == (
        20.125, 22.125, 2, 4)
    assert layer_label("second", "original") in shot.filter_complex
    assert "movie=bad" not in shot.filter_complex
    assert result == compiler.compile_visual_layers(plan, source_size=(1920, 1080))
    assert before == plan.model_dump_json()


def test_labels_follow_ids_even_when_layer_list_order_changes(source: Path) -> None:
    from clipperx.filtergraph import layer_label
    assert layer_label("shot-a", "layer") != layer_label("shot-b", "layer")
    assert layer_label("shot", "a-b") != layer_label("shot", "a_b")
    assert layer_label("a", "bc") != layer_label("ab", "c")


def test_preflight_solid_is_exclusive_to_solid_kind(source: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake_tools(monkeypatch)
    payload = host_plan(source).model_dump()
    layer = payload["shots"][0]["layers"][0]
    layer.update(kind="solid", fill="solid", solid_color="#123456", source_crop=None)
    plan = VisualCompositionPlan.model_validate(payload)
    compiler.preflight_visual_plan(plan, source=source, assets=default_render_assets())
    assert calls
    assert "color=c=0x123456" in compiler.compile_visual_layers(
        plan, source_size=(1920, 1080))[0].filter_complex
    calls.clear()
    plan.shots[0].layers[0].kind = compiler.LayerKind.SOURCE_VIDEO
    with pytest.raises(compiler.RenderPreflightError, match="solid"):
        compiler.preflight_visual_plan(plan, source=source, assets=default_render_assets())
    assert not calls


def test_compilation_preserves_motion_and_propagates_native_requirements(source: Path) -> None:
    from clipperx.visual_contracts import CropKeyframe, NormalizedRect
    plan = host_plan(source)
    plan.shots[0].layers[0].crop_track = [CropKeyframe(
        at=0, crop=NormalizedRect(x=0, y=0, width=1, height=1)), CropKeyframe(
        at=2, crop=NormalizedRect(x=.1, y=.1, width=.8, height=.8))]
    compiled = compiler.compile_visual_layers(plan, source_size=(1920, 1080))[0]
    assert "libplacebo=" in compiled.filter_complex
    assert compiled.requirements.needs_vulkan
    assert "libplacebo" in compiled.requirements.required_filters


def test_tied_z_order_preserves_declared_layer_order(source: Path) -> None:
    import copy
    plan = host_plan(source)
    extra = copy.deepcopy(plan.shots[0].layers[0])
    extra.layer_id = "earlier-declared"
    plan.shots[0].layers.insert(0, extra)
    compiled = compiler.compile_visual_layers(plan, source_size=(1920, 1080))[0]
    assert compiled.layer_ids == ("earlier-declared", "original")


def test_inherit_uses_immediate_predecessor_and_can_chain(source: Path) -> None:
    import copy
    plan = two_shot_plan(source)
    third = copy.deepcopy(plan.shots[1])
    third.shot_id = "third"
    third.output.start, third.output.end = 4, 6
    plan.shots.append(third)
    plan.shots[1].layers[0].fill = compiler.FillMode.INHERIT
    plan.shots[2].layers[0].fill = compiler.FillMode.INHERIT
    resolved = compiler.resolve_visual_plan(plan)
    assert all(shot.layers[0].fill == compiler.FillMode.COVER for shot in resolved.shots)
    plan.shots[1].layers[0].layer_id = "different"
    plan.shots[1].layers[0].fill = compiler.FillMode.COVER
    with pytest.raises(compiler.RenderPreflightError, match="inherit missing"):
        compiler.resolve_visual_plan(plan)


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "source.mp4"
    path.write_bytes(b"HOST placeholder; subprocess probing is mocked, not real media")
    return path


def add_brand(plan: VisualCompositionPlan, asset_id: str = "kick_brand") -> VisualCompositionPlan:
    payload = plan.model_dump()
    payload["shots"][0]["brand"] = {
        "asset_id": asset_id, "start": 0, "end": 2, "z_index": 1,
        "destination": {"x": 0.2, "y": 0.1, "width": 0.6, "height": 0.1},
    }
    return VisualCompositionPlan.model_validate(payload)


def fake_tools(monkeypatch: pytest.MonkeyPatch, *, width: int = 1920,
               height: int = 1080, duration: str = "30", audio_count: int = 1) -> list[list[str]]:
    import json

    commands: list[list[str]] = []
    monkeypatch.setattr(compiler.shutil, "which", lambda name: name)

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(args)
        assert "-i" not in args and "-filter_complex" not in args
        if "-encoders" in args:
            output = " V....D h264_nvenc Nvidia\n V....D libx264 CPU\n A..... aac AAC\n"
        elif "-show_entries" in args:
            output = json.dumps({"streams": [{"codec_type": "video", "width": width,
                "height": height, "start_time": "0", "duration": "30"}]
                + [{"codec_type": "audio", "start_time": "0", "duration": "30"}]
                * audio_count, "format": {"start_time": "0", "duration": duration}})
        else:
            output = f"{args[0]} version host-fixture"
        return subprocess.CompletedProcess(args, 0, output, "")

    monkeypatch.setattr(compiler.subprocess, "run", run)
    return commands


def test_v2_compiler_rejects_missing_brand_before_tools(source: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        pytest.fail("invalid assets must fail before subprocess")
    monkeypatch.setattr(compiler.subprocess, "run", forbidden)
    with pytest.raises(compiler.RenderPreflightError, match="brand"):
        compiler.preflight_visual_plan(add_brand(host_plan(source)), source=source,
                                       assets=RenderAssetRegistry({}, {}, {}))


@pytest.mark.parametrize("value", ["adaptive", "split", "unknown"])
def test_v2_never_maps_unknown_layout_to_split(source: Path, value: str) -> None:
    payload = host_plan(source).model_dump()
    payload["shots"][0]["layers"][0]["fill"] = value
    with pytest.raises(ValidationError):
        VisualCompositionPlan.model_validate(payload)


def test_encoder_failure_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(compiler, "available_encoders", lambda: {"libx264"})
    with pytest.raises(compiler.RenderPreflightError, match=r"h264_nvenc.*\['libx264'\]"):
        compiler.select_encoder("h264_nvenc")


def test_profiles_are_explicit_and_have_exact_arguments() -> None:
    common = ("-r", "60", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
              "-c:a", "aac", "-b:a", "192k", "-ar", "48000")
    assert compiler.encoder_arguments(RenderProfile()) == (
        "-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr",
        "-cq", "19", "-b:v", "0", *common)
    assert compiler.encoder_arguments(RenderProfile(video_encoder=EncoderName.LIBX264)) == (
        "-c:v", "libx264", "-preset", "medium", "-crf", "18", *common)


def test_preflight_reads_metadata_only_and_does_not_mutate(source: Path,
                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake_tools(monkeypatch)
    plan = add_brand(host_plan(source))
    before = plan.model_dump_json()
    compiler.preflight_visual_plan(plan, source=source, assets=default_render_assets())
    assert plan.model_dump_json() == before
    assert len(calls) == 4
    assert any("-show_entries" in call for call in calls)


@pytest.mark.parametrize("kwargs,match", [({"width": 1919}, "even"),
    ({"height": 0}, "dimensions"), ({"duration": "nan"}, "duration"),
    ({"duration": "11"}, "source range"), ({"audio_count": 0}, "one mixed audio"),
    ({"audio_count": 2}, "one mixed audio")])
def test_preflight_rejects_invalid_source(source: Path, monkeypatch: pytest.MonkeyPatch,
                                         kwargs: dict[str, object], match: str) -> None:
    fake_tools(monkeypatch, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(compiler.RenderPreflightError, match=match):
        compiler.preflight_visual_plan(host_plan(source), source=source,
                                       assets=default_render_assets())


@pytest.mark.parametrize("mutation", ["transition", "fill", "source_ref", "duration", "extra"])
def test_preflight_revalidates_mutated_plan_before_tools(source: Path,
    monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    calls = fake_tools(monkeypatch)
    plan = host_plan(source)
    if mutation == "transition":
        plan.shots[0].__dict__["transition_in"] = "dissolve"
    elif mutation == "fill":
        plan.shots[0].layers[0].__dict__["fill"] = "adaptive"
    elif mutation == "source_ref":
        plan.shots[0].layers[0].source_ref = "C:/other.mp4"
    elif mutation == "extra":
        plan.shots[0].layers[0].__dict__["opacity"] = 0.5
    else:
        plan.shots[0].output.end = 180
        plan.shots[0].source.end = 190
    if mutation == "fill":
        with pytest.warns(UserWarning, match="Pydantic serializer warnings"):
            with pytest.raises(compiler.RenderPreflightError):
                compiler.preflight_visual_plan(plan, source=source, assets=default_render_assets())
    else:
        with pytest.raises(compiler.RenderPreflightError):
            compiler.preflight_visual_plan(plan, source=source, assets=default_render_assets())
    assert calls == []


def test_subtitles_require_valid_data_and_registered_font(source: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake_tools(monkeypatch)
    payload = host_plan(source).model_dump()
    payload["shots"][0]["subtitles"] = {"destination": {
        "x": 0.1, "y": 0.7, "width": 0.8, "height": 0.2}}
    plan = VisualCompositionPlan.model_validate(payload)
    with pytest.raises(compiler.RenderPreflightError, match="subtitle data"):
        compiler.preflight_visual_plan(plan, source=source, assets=default_render_assets())
    assert calls == []
    data = [TranscriptSegment(10, 11, "HOST text", [Word(10, 11, "HOST")])]
    compiler.preflight_visual_plan(plan, source=source, assets=default_render_assets(),
                                   transcript=data)
    data[0].words[0].end = float("nan")
    with pytest.raises(compiler.RenderPreflightError, match="subtitle"):
        compiler.preflight_visual_plan(plan, source=source, assets=default_render_assets(),
                                       transcript=data)


def test_unused_font_id_on_brand_is_rejected(source: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake_tools(monkeypatch)
    plan = add_brand(host_plan(source))
    assert plan.shots[0].brand is not None
    plan.shots[0].brand.font_asset_id = "unregistered_font"
    with pytest.raises(compiler.RenderPreflightError, match="font"):
        compiler.preflight_visual_plan(plan, source=source, assets=default_render_assets())
    assert calls == []


@pytest.mark.parametrize("command", ["ffmpeg", "ffprobe"])
def test_missing_binary_fails_explicitly(source: Path, monkeypatch: pytest.MonkeyPatch,
                                        command: str) -> None:
    fake_tools(monkeypatch)
    monkeypatch.setattr(compiler.shutil, "which", lambda name: None if name == command else name)
    with pytest.raises(compiler.RenderPreflightError, match=command):
        compiler.preflight_visual_plan(host_plan(source), source=source,
                                       assets=default_render_assets())


def test_bad_probe_and_command_timeout_have_domain_errors(source: Path,
                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    fake_tools(monkeypatch)
    monkeypatch.setattr(compiler.subprocess, "run", lambda args, **kwargs:
                        subprocess.CompletedProcess(args, 0, "not-json", ""))
    with pytest.raises(compiler.RenderPreflightError, match="invalid JSON"):
        compiler._probe(source)

    def timeout(*args: object, **kwargs: object) -> None:
        raise subprocess.TimeoutExpired("ffmpeg", 15)
    monkeypatch.setattr(compiler.subprocess, "run", timeout)
    with pytest.raises(compiler.RenderPreflightError, match="inspection failed"):
        compiler.available_encoders()


def test_source_fingerprint_is_independently_checked(source: Path,
                                                     monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake_tools(monkeypatch)
    plan = host_plan(source)
    source.write_bytes(b"different HOST source")
    with pytest.raises(compiler.RenderPreflightError, match="source_fingerprint"):
        compiler.preflight_visual_plan(plan, source=source, assets=default_render_assets())
    assert calls == []


def test_explicit_host_sfx_id_is_probed_not_interpreted_as_model_path(source: Path,
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake_tools(monkeypatch)
    payload = host_plan(source).model_dump()
    payload["shots"][0]["audio"]["cues"] = [{"asset_id": "impact", "start": 0, "end": 1}]
    plan = VisualCompositionPlan.model_validate(payload)
    with pytest.raises(compiler.RenderPreflightError, match="sfx"):
        compiler.preflight_visual_plan(plan, source=source, assets=default_render_assets())
    assert calls == []
    impact = tmp_path / "impact.wav"
    impact.write_bytes(b"HOST fixture; metadata probe mocked")
    assets = RenderAssetRegistry({}, {}, {"impact": impact})
    compiler.preflight_visual_plan(plan, source=source, assets=assets)
    assert calls[-1][-1] == str(impact)


def test_provider_free_imports_do_not_load_settings_or_models() -> None:
    import sys

    script = """
import sys
import clipperx.visual_compiler
import clipperx.diagnostics
for name in ('clipperx.config', 'clipperx.embeddings', 'clipperx.model_assets',
             'clipperx.structured_client', 'sentence_transformers', 'torch'):
    assert name not in sys.modules, name
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                            text=True, timeout=15, check=False)
    assert result.returncode == 0, result.stderr


def host_metadata() -> dict[str, object]:
    """HOST-only probe fixture; durations are independent for every component."""
    return {"streams": [
        {"codec_type": "video", "width": 1920, "height": 1080,
         "start_time": "0", "duration": "30"},
        {"codec_type": "audio", "start_time": "0", "duration": "30"},
    ], "format": {"start_time": "0", "duration": "30"}}


def source_span_tools(monkeypatch: pytest.MonkeyPatch,
                      metadata: dict[str, object]) -> list[list[str]]:
    """Exercise the actual preflight probe with in-memory FFprobe JSON only."""
    commands = fake_tools(monkeypatch)
    run_tools = compiler.subprocess.run

    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        result = run_tools(args, **kwargs)
        if "-show_entries" in args:
            return subprocess.CompletedProcess(args, 0, json.dumps(metadata), "")
        return result

    def forbidden(*args: object, **kwargs: object) -> Any:
        pytest.fail("Metadata-only source bounds must not qualify a native runtime")

    monkeypatch.setattr(compiler.subprocess, "run", run)
    monkeypatch.setattr(compiler, "qualify_render_runtime", forbidden)
    return commands


@pytest.mark.parametrize("origin", ["100", "-100", "10.1", "-0.1", "0"])
def test_preflight_maps_file_source_through_one_common_raw_origin(
    source: Path, monkeypatch: pytest.MonkeyPatch, origin: str
) -> None:
    metadata = host_metadata()
    for component in [*compiler._streams(metadata), metadata["format"]]:
        assert isinstance(component, dict)
        component["start_time"] = origin
    calls = source_span_tools(monkeypatch, metadata)
    plan = host_plan(source)
    before = plan.model_dump_json()
    assert compiler.preflight_visual_plan(plan, source=source,
                                          assets=default_render_assets()) is None
    assert plan.model_dump_json() == before
    assert len(calls) == 4
    assert sum("-show_entries" in call for call in calls) == 1


def test_preflight_rejects_raw_looking_source_outside_two_second_file(
    source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = host_metadata()
    for component in [*compiler._streams(metadata), metadata["format"]]:
        assert isinstance(component, dict)
        component.update(start_time="10", duration="2")
    calls = source_span_tools(monkeypatch, metadata)
    with pytest.raises(compiler.RenderPreflightError, match="source range.*format"):
        compiler.preflight_visual_plan(host_plan(source), source=source,
                                       assets=default_render_assets())
    assert len(calls) == 4
    assert sum("-show_entries" in call for call in calls) == 1


@pytest.mark.parametrize("kind", ["video", "audio"])
@pytest.mark.parametrize("start,end,accepted", [(.3, 2, True),
    (.29999999999999993, 2, False), (.3, 2.0000000000000004, False)])
def test_preflight_preserves_delayed_raw_stream_coverage_with_common_origin(
    source: Path, monkeypatch: pytest.MonkeyPatch, kind: str,
    start: float, end: float, accepted: bool
) -> None:
    metadata = host_metadata()
    for component in [*compiler._streams(metadata), metadata["format"]]:
        assert isinstance(component, dict)
        component["start_time"] = "100.1"
    component = next(stream for stream in compiler._streams(metadata)
                     if stream["codec_type"] == kind)
    component.update(start_time="100.4", duration="1.7")
    calls = source_span_tools(monkeypatch, metadata)
    plan = host_plan(source)
    plan.shots[0].source.start, plan.shots[0].source.end = start, end
    plan.shots[0].output.end = end - start
    before = plan.model_dump_json()
    if accepted:
        assert compiler.preflight_visual_plan(plan, source=source,
                                              assets=default_render_assets()) is None
    else:
        with pytest.raises(compiler.RenderPreflightError, match=rf"source range.*{kind}"):
            compiler.preflight_visual_plan(plan, source=source, assets=default_render_assets())
    assert plan.model_dump_json() == before
    assert len(calls) == 4
    assert sum("-show_entries" in call for call in calls) == 1


@pytest.mark.parametrize("end,accepted", [(.8, True), (.8000000000000002, False)])
def test_preflight_uses_exact_decimal_origin_and_span_sums(
    source: Path, monkeypatch: pytest.MonkeyPatch, end: float, accepted: bool
) -> None:
    metadata = host_metadata()
    metadata["format"] = {"start_time": "0.1", "duration": "1"}
    for component in compiler._streams(metadata):
        component.update(start_time="0.3", duration="0.6")
    calls = source_span_tools(monkeypatch, metadata)
    plan = host_plan(source)
    plan.shots[0].source.start, plan.shots[0].source.end = .2, end
    plan.shots[0].output.end = end - .2
    before = plan.model_dump_json()
    if accepted:
        assert compiler.preflight_visual_plan(plan, source=source,
                                              assets=default_render_assets()) is None
    else:
        with pytest.raises(compiler.RenderPreflightError, match="source range.*video"):
            compiler.preflight_visual_plan(plan, source=source, assets=default_render_assets())
    assert plan.model_dump_json() == before
    assert len(calls) == 4
    assert sum("-show_entries" in call for call in calls) == 1


@pytest.mark.parametrize("kind", ["video", "audio"])
def test_source_range_requires_each_stream_not_only_container_duration(
    source: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    fake_tools(monkeypatch)
    metadata = host_metadata()
    streams = compiler._streams(metadata)
    next(stream for stream in streams if stream["codec_type"] == kind)["duration"] = "8"
    monkeypatch.setattr(compiler, "_probe", lambda path: metadata)
    with pytest.raises(compiler.RenderPreflightError, match=rf"source range.*{kind}"):
        compiler.preflight_visual_plan(host_plan(source), source=source,
                                       assets=default_render_assets())


@pytest.mark.parametrize("kind", ["video", "audio", "format"])
@pytest.mark.parametrize("value", [None, "N/A", "", "0", "-1", "nan", "inf", True, []])
def test_source_duration_unavailable_or_invalid_is_explicit(
    source: Path, monkeypatch: pytest.MonkeyPatch, kind: str, value: object
) -> None:
    fake_tools(monkeypatch)
    metadata = host_metadata()
    component = (metadata["format"] if kind == "format" else next(
        stream for stream in compiler._streams(metadata) if stream["codec_type"] == kind))
    assert isinstance(component, dict)
    if value is None:
        component.pop("duration")  # FFprobe JSON omits invalid/unknown optional fields.
    else:
        component["duration"] = value
    monkeypatch.setattr(compiler, "_probe", lambda path: metadata)
    with pytest.raises(compiler.RenderPreflightError, match=rf"{kind}.*duration"):
        compiler.preflight_visual_plan(host_plan(source), source=source,
                                       assets=default_render_assets())


@pytest.mark.parametrize("kind", ["video", "audio", "format"])
@pytest.mark.parametrize("value", [None, "N/A", "", "nan", "inf", True])
def test_source_start_time_unavailable_or_invalid_is_explicit(
    source: Path, monkeypatch: pytest.MonkeyPatch, kind: str, value: object
) -> None:
    fake_tools(monkeypatch)
    metadata = host_metadata()
    component = (metadata["format"] if kind == "format" else next(
        stream for stream in compiler._streams(metadata) if stream["codec_type"] == kind))
    assert isinstance(component, dict)
    if value is None:
        component.pop("start_time")
    else:
        component["start_time"] = value
    monkeypatch.setattr(compiler, "_probe", lambda path: metadata)
    with pytest.raises(compiler.RenderPreflightError, match=rf"{kind}.*start_time"):
        compiler.preflight_visual_plan(host_plan(source), source=source,
                                       assets=default_render_assets())


@pytest.mark.parametrize("start", ["10", "-1", "9"])
def test_source_spans_preserve_offsets_and_accept_exact_end_without_retiming(
    source: Path, monkeypatch: pytest.MonkeyPatch, start: str
) -> None:
    fake_tools(monkeypatch)
    metadata = host_metadata()
    for component in [*compiler._streams(metadata), metadata["format"]]:
        assert isinstance(component, dict)
        component.update(start_time=start, duration="2")
    monkeypatch.setattr(compiler, "_probe", lambda path: metadata)
    plan = host_plan(source)
    plan.shots[0].source.start, plan.shots[0].source.end = 0, 2
    before = plan.model_dump_json()
    compiler.preflight_visual_plan(plan, source=source, assets=default_render_assets())
    assert plan.model_dump_json() == before


@pytest.mark.parametrize("kind", ["video", "audio", "format"])
def test_source_range_before_required_component_start_is_rejected(
    source: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    fake_tools(monkeypatch)
    metadata = host_metadata()
    component = (metadata["format"] if kind == "format" else next(
        stream for stream in compiler._streams(metadata) if stream["codec_type"] == kind))
    assert isinstance(component, dict)
    if kind == "format":
        component.update(start_time="11", duration="2")
    else:
        component["start_time"] = "11"
    monkeypatch.setattr(compiler, "_probe", lambda path: metadata)
    with pytest.raises(compiler.RenderPreflightError, match=rf"source range.*{kind}"):
        compiler.preflight_visual_plan(host_plan(source), source=source,
                                       assets=default_render_assets())


def test_probe_requests_start_time_and_duration_for_streams_and_format(
    source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands = fake_tools(monkeypatch)
    compiler._probe(source)
    entries = commands[0][commands[0].index("-show_entries") + 1]
    assert "stream=codec_type,width,height,start_time,duration" in entries
    assert "format=start_time,duration" in entries


def native_plan(source: Path) -> VisualCompositionPlan:
    from clipperx.visual_contracts import CropKeyframe, NormalizedRect

    plan = host_plan(source)
    plan.shots[0].layers[0].crop_track = [CropKeyframe(at=0, crop=NormalizedRect(
        x=0, y=0, width=1, height=1)), CropKeyframe(at=2, crop=NormalizedRect(
        x=.1, y=.1, width=.8, height=.8))]
    return plan


def test_native_preflight_uses_paired_runtime_and_qualifies_after_geometry(
        source: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from clipperx.render_runtime import (
        RenderRequirements, RuntimeQualification, requirements_for_plan, resolve_render_runtime,
    )

    plan = native_plan(source)
    selected = resolve_render_runtime(requirements_for_plan(plan))
    calls = fake_tools(monkeypatch)
    qualified: list[str] = []

    def qualify(runtime: compiler.RenderRuntime,
                requirements: RenderRequirements) -> RuntimeQualification:
        assert len(calls) == 4
        assert runtime == selected and requirements.needs_vulkan
        qualified.append("actual GPU probe boundary")
        return RuntimeQualification(runtime, "HOST9", "HOST9", ("libplacebo",), (),
                                    0, None, 0, 1024, .1, "HOST unit evidence")

    monkeypatch.setattr(compiler, "qualify_render_runtime", qualify)
    result = compiler.preflight_visual_plan(plan, source=source, assets=default_render_assets(),
                                            runtime=selected)
    assert qualified == ["actual GPU probe boundary"]
    assert result is not None and result.runtime == selected
    assert all(call[0] in (str(selected.ffmpeg), str(selected.ffprobe)) for call in calls)
    assert dict(result.source_color) == dict(color_transfer=None, color_primaries=None,
        color_space=None, color_range=None, pix_fmt=None, chroma_location=None)


@pytest.mark.parametrize("defect", ["preservation", "HDR-PQ", "HDR-HLG"])
def test_native_geometry_and_hdr_fail_before_any_vulkan_probe(
        source: Path, monkeypatch: pytest.MonkeyPatch, defect: str) -> None:
    from clipperx.render_runtime import requirements_for_plan, resolve_render_runtime
    from clipperx.visual_contracts import NormalizedRect

    plan = native_plan(source)
    selected = resolve_render_runtime(requirements_for_plan(plan))
    fake_tools(monkeypatch)
    metadata = host_metadata()
    if defect == "preservation":
        plan.shots[0].layers[0].preserve_regions = [NormalizedRect(x=.1, y=.1, width=.1, height=.1)]
    else:
        compiler._streams(metadata)[0]["color_transfer"] = (
            "smpte2084" if defect == "HDR-PQ" else "arib-std-b67")
    monkeypatch.setattr(compiler, "_probe", lambda path, runtime=None: metadata)
    monkeypatch.setattr(compiler, "qualify_render_runtime", lambda *args: pytest.fail(
        "source/geometry/HDR failure must precede GPU probe"))
    with pytest.raises(compiler.RenderPreflightError, match="preserve|HDR"):
        compiler.preflight_visual_plan(plan, source=source, assets=default_render_assets(),
                                        runtime=selected)


def test_native_bad_fingerprint_fails_before_runtime_resolution(
        source: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = native_plan(source)
    plan.source_fingerprint = "not-the-source"
    monkeypatch.setattr(compiler, "resolve_render_runtime", lambda *args, **kwargs: pytest.fail(
        "source fingerprint must precede runtime/pin inspection"))
    with pytest.raises(compiler.RenderPreflightError, match="fingerprint"):
        compiler.preflight_visual_plan(plan, source=source, assets=default_render_assets())


@pytest.mark.parametrize("transfer", ["smpte2084", "arib-std-b67"])
def test_basic_static_hdr_is_explicitly_unqualified_without_gpu(
        source: Path, monkeypatch: pytest.MonkeyPatch, transfer: str) -> None:
    fake_tools(monkeypatch)
    metadata = host_metadata()
    compiler._streams(metadata)[0]["color_transfer"] = transfer
    monkeypatch.setattr(compiler, "_probe", lambda path: metadata)
    monkeypatch.setattr(compiler, "qualify_render_runtime", lambda *args: pytest.fail(
        "static HDR must fail without probing GPU"))
    with pytest.raises(compiler.RenderPreflightError, match="HDR.*not qualified"):
        compiler.preflight_visual_plan(host_plan(source), source=source,
                                        assets=default_render_assets())


def test_basic_static_unknown_color_is_retained_without_relabeling_or_gpu(
        source: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    fake_tools(monkeypatch)
    metadata = host_metadata()
    compiler._streams(metadata)[0].update(color_transfer="unknown", color_primaries="unknown")
    before = json.dumps(metadata)
    monkeypatch.setattr(compiler, "_probe", lambda path: metadata)
    monkeypatch.setattr(compiler, "qualify_render_runtime", lambda *args: pytest.fail(
        "basic static plan must retain its system path without GPU"))
    assert compiler.preflight_visual_plan(host_plan(source), source=source,
                                           assets=default_render_assets()) is None
    assert json.dumps(metadata) == before


def test_audio_bridge_combines_native_visual_and_explicit_audio_requirements(source: Path) -> None:
    from clipperx.audio_mix import AudioFormat
    plan = native_plan(source)
    plan.shots[0].audio.normalize_dialogue = False
    result = compiler.compile_plan_audio(plan, audio_format=AudioFormat(48000, "stereo"))
    assert result.requirements.needs_vulkan
    assert {"libplacebo", "atrim", "alimiter"} <= set(result.requirements.required_filters)
    assert result.schedule.spans[0].source_start == 10


def test_exact_audio_video_clock_subset_rejected_before_host_tools(
        source: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = fake_tools(monkeypatch)
    plan = host_plan(source)
    plan.shots[0].output.end += .0001
    with pytest.raises(compiler.RenderPreflightError, match="durations must match exactly"):
        compiler.preflight_visual_plan(plan, source=source, assets=default_render_assets())
    assert calls == []


def test_probe_audio_format_explicit_fields_and_unknown_layout_rejection(
        source: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from clipperx.audio_mix import AudioFormat
    calls = fake_tools(monkeypatch)
    compiler._probe(source)
    entries = calls[0][calls[0].index("-show_entries")+1]
    assert "sample_rate,channels,channel_layout" in entries
    metadata = host_metadata()
    audio = compiler._streams(metadata)[1]
    monkeypatch.setattr(compiler, "_probe", lambda *args, **kwargs: metadata)
    with pytest.raises(compiler.RenderPreflightError, match="audio format"):
        compiler.probe_audio_format(source, runtime=None)
    audio.update(sample_rate="48000", channels=2, channel_layout="stereo")
    assert compiler.probe_audio_format(source, runtime=None) == AudioFormat(48000, "stereo")
    audio["channels"] = 1
    with pytest.raises(compiler.RenderPreflightError, match="audio format"):
        compiler.probe_audio_format(source, runtime=None)


def test_native_audio_bridge_rejects_unmatched_runtime_before_graph(source: Path) -> None:
    from clipperx.audio_mix import AudioFormat
    from clipperx.render_runtime import RenderRuntime
    plan = native_plan(source)
    plan.shots[0].audio.normalize_dialogue = False
    wrong = RenderRuntime(source, source, "system", "host", "host", None, None,
                          "-filter_complex_script", None)
    with pytest.raises(compiler.RenderPreflightError, match="runtime"):
        compiler.compile_plan_audio(plan, audio_format=AudioFormat(48000, "stereo"), runtime=wrong)
