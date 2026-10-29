"""Sequential source-slot materialization and unchanged ordered spatial graphs."""

from pathlib import Path
from dataclasses import fields, replace
from typing import Any, Literal

from . import source_decode as sd
from .render_clock import SourceTimeOrigin, StreamDecodeLimits
from .render_runtime import RenderRuntime
from .visual_compiler import (
    AssApply, BrandInputResource, CompiledShotVisual, CompiledVisualAssembly, FilterFragment,
    begin_source_stream_binding, bind_source_stream_selection, finalize_source_stream_binding,
)


def source_windows(assembly: CompiledVisualAssembly, shot: CompiledShotVisual
        ) -> tuple[tuple[int, int, sd.Window], ...]:
    transport = shot.transport
    if transport is None:
        return ()
    windows = []
    first = transport.slot_start
    while first < transport.slot_end:
        request = assembly.timeline.video_request(transport.k_start+first-transport.slot_start)
        end = min(shot.span.source_end, request.requested_source+2)
        last = first+1
        while last < min(transport.slot_end, first+120):
            upcoming = assembly.timeline.video_request(transport.k_start+last-transport.slot_start)
            if upcoming.requested_source >= end:
                break
            last += 1
        windows.append((first, last, sd.Window(request.requested_source, end)))
        first = last
    return tuple(windows)


def serialize_shot_graph(shot: CompiledShotVisual, brands: tuple[BrandInputResource, ...],
                         ass_paths: dict[str, str]) -> str:
    parts = []
    input_index = 0
    if shot.transport is not None:
        parts.append(f"[0:v]null[{shot.source_input_label}]")
        input_index = 1
    for brand in brands:
        parts.append(f"[{input_index}:v]settb=1/60,setpts=N+{shot.span.k_start}[{brand.graph_label}]")
        input_index += 1
    for step in shot.steps:
        if type(step) is FilterFragment:
            parts.append(step.text)
        elif type(step) is AssApply:
            path = ass_paths[step.document_key]
            parts.append(f"[{step.input_label}]ass=filename={path}:fontsdir=fonts[{step.output_label}]")
        else:
            raise ValueError("unknown ordered spatial step")
    return ";".join(parts)


def materialize_shot(host: Any, assembly: CompiledVisualAssembly, shot: CompiledShotVisual,
                     runtime: RenderRuntime, origin: SourceTimeOrigin, path: Path) -> list[Any]:
    limits = StreamDecodeLimits(max_requests=120)
    observation_runtime = replace(runtime, device_index=None)
    policy: Literal["render48k-mono", "render48k-stereo"] = (
        "render48k-mono" if next(s for s in origin.streams if s.kind == "audio").channels == 1 else "render48k-stereo")
    expected = 3*assembly.source_size[0]*assembly.source_size[1]*shot.frame_count
    host.register(path, "raw", expected)
    evidence = []
    with host.sink(path) as sink:
        for first, last, window in source_windows(assembly, shot):
            host.require_log_capacity(2*limits.max_log_bytes)
            transport = shot.transport
            assert transport is not None
            requests = tuple(assembly.timeline.video_request(
                transport.k_start+slot-transport.slot_start).requested_source for slot in range(first, last))
            control = host.job.new_observation("streamed_s16", deadline_ns=host.deadline(45))
            with sd.stream_observed_window(observation_runtime, origin, window, requests, audio_policy=policy,
                    limits=limits, _host=control, _runtime_files=host.files) as stream:
                cursor = begin_source_stream_binding(assembly, slot_start=first, slot_end=last,
                    origin=origin, session_id=stream.session_id, window=(window.start, window.end),
                    expected_audio_policy=policy, limits=limits)
                while True:
                    selection = stream.next_selection()
                    if selection is None:
                        break
                    cursor, delivery = bind_source_stream_selection(cursor, selection=selection)
                    index = delivery.delivery_index
                    view = memoryview(delivery.frame.bgr)
                    try:
                        for _ in delivery.slots:
                            host.write_all(sink, view)
                    finally:
                        view.release()
                    del view, delivery, selection
                    stream.acknowledge(index)
                final = stream.finish()
                binding = finalize_source_stream_binding(cursor, finalization=final)
            status = control.ownership_snapshot()
            if status.phase != "retired_ok" or not status.window_resources_zero or status.cleanup_required:
                raise ValueError("source stream not certainly retired")
            projection = {f.name: getattr(binding, f.name) for f in fields(binding)}
            projection["source_control"] = status
            projection["source_finalization"] = final.record()
            evidence.append(projection)
            host.account_logs(sum(d.log_bytes for d in status.diagnostics))
            host.record_auxiliary("source", projection)
    host.seal(path, expected)
    return evidence


def render_shots(host: Any, assembly: CompiledVisualAssembly, runtime: RenderRuntime,
                 origin: SourceTimeOrigin) -> tuple[tuple[Path, ...], list[Any], tuple[str, ...]]:
    ass_paths: dict[str, str] = {}
    for index, document in enumerate(assembly.ass_documents):
        leaf = host.root / "assets" / f"ass{index:06d}.ass"
        host.publish(leaf, document.document.encode("utf-8-sig"), role="ass")
        ass_paths[document.key] = f"assets/{leaf.name}"
    if len(ass_paths) != len(assembly.ass_documents):
        raise ValueError("ASS resource mapping must be bijective")
    outputs, bindings, graphs = [], [], []
    for index, shot in enumerate(assembly.shots):
        host.recheck_space()
        source = host.root / "shots" / f"{index:06d}.bgr"
        rgba = host.root / "shots" / f"{index:06d}.rgba"
        inputs: list[Path] = []
        argv = [str(runtime.ffmpeg), "-hide_banner", "-nostats", "-loglevel", "info", *runtime.device_arguments]
        if shot.transport is not None:
            bindings.extend(materialize_shot(host, assembly, shot, runtime, origin, source))
            w, h = assembly.source_size
            argv.extend(("-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{w}x{h}",
                "-framerate", "60", "-i", str(source)))
            inputs.append(source)
        brands = tuple(b for b in assembly.brand_inputs if b.shot_id == shot.span.shot_id)
        for brand in brands:
            argv.extend(("-i", str(brand.asset.path)))
            inputs.append(brand.asset.path)
        graph = serialize_shot_graph(shot, brands, ass_paths)
        graphs.append(graph)
        graph_path = host.root / "assets" / f"graph{index:06d}.txt"
        host.publish(graph_path, graph.encode(), role="graph")
        inputs.extend([graph_path, *(host.root / p for p in ass_paths.values()), *host.font_paths])
        argv.extend((runtime.graph_file_option, f"assets/{graph_path.name}", "-map", f"[{shot.output_label}]",
            "-an", "-fps_mode", "passthrough", "-pix_fmt", "rgba", "-c:v", "rawvideo", "-f", "rawvideo", "pipe:1"))
        expected = 8294400*shot.frame_count
        host.register(rgba, "raw", expected)
        with host.sink(rgba) as sink:
            host.run(tuple(argv), kind="spatial", timeout=900, expected_stdout_bytes=expected,
                max_stdout_bytes=expected, sink=sink, inputs=tuple(dict.fromkeys(inputs)))
        host.seal(rgba, expected)
        if shot.transport is not None:
            with host.io_scope(120):
                host.files.path_action("delete", source)
        outputs.append(rgba)
    return tuple(outputs), bindings, tuple(graphs)
