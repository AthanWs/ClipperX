"""Ordered transport and graph adapters."""

import importlib.util
from fractions import Fraction
from typing import Any

import pytest

from clipperx.tests.test_visual_compiler import pure_plan, pure_overlays
from clipperx.tests.test_retained_files import memory as memory, ROOT
from clipperx.tests.test_source_host_ownership import closed_boundaries as closed_boundaries
from clipperx.tests.test_render_services import REAL_PROBE_SOURCE, _QUEUE_GET, _QUEUE_PUT


def test_video_executor_exists() -> None:
    assert importlib.util.find_spec("clipperx._visual_render_video") is not None


def test_whole_shot_graph_and_exact_window_partition() -> None:
    from clipperx.visual_compiler import compile_visual_assembly, FilterFragment
    from clipperx._visual_render_video import source_windows, serialize_shot_graph
    plan = pure_plan()
    assembly = compile_visual_assembly(plan, source_size=(2, 2), overlays=pure_overlays(plan))
    for shot in assembly.shots:
        windows = source_windows(assembly, shot)
        assert shot.transport is not None
        for first, last, window in windows:
            assert last-first <= 120 and window.end-window.start <= 2
            assert all(window.start <= assembly.timeline.video_request(
                shot.transport.k_start+slot-shot.transport.slot_start).requested_source < window.end
                for slot in range(first, last))
        graph = serialize_shot_graph(shot, (), {})
        fragments = [s.text for s in shot.steps if isinstance(s, FilterFragment)]
        position = 0
        for fragment in fragments:
            position = graph.index(fragment, position)+len(fragment)
        assert shot.output_time_base == Fraction(1, 60)


def test_n120_stays_in_one_spatial_graph_across_observer_windows() -> None:
    from clipperx._visual_render_video import source_windows, serialize_shot_graph
    from clipperx.visual_contracts import CropKeyframe, NormalizedRect
    from clipperx.visual_compiler import compile_visual_assembly
    plan = pure_plan(duration=2.1)
    layer = plan.shots[0].layers[0]
    layer.crop_track = [CropKeyframe(at=t, crop=NormalizedRect(x=0, y=0, width=1, height=1)) for t in (0, 2, 2.1)]
    assembly = compile_visual_assembly(plan, source_size=(2, 2), overlays=pure_overlays(plan))
    shot = assembly.shots[0]
    assert [(a, b) for a, b, _ in source_windows(assembly, shot)] == [(0, 120), (120, 126)]
    graph = serialize_shot_graph(shot, (), {})
    assert len(assembly.shots) == 1 and shot.frame_count == 126 and graph.count("setpts=N[") == 1
    track = shot.source_layers[0].crop_track
    assert track is not None and track.points[1].frame == 120


def test_actual_two_source_windows_feed_one_full_n120_spatial_child(memory: Any,
        monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise actual orchestration; only trusted external IO is synthetic."""
    import hashlib
    from dataclasses import replace
    import json
    import math
    from pathlib import Path
    import queue
    from types import SimpleNamespace
    import zlib

    from clipperx import render_runtime as rr, source_decode as sd
    from clipperx._visual_render_host import HostContext
    from clipperx._visual_render_video import render_shots, serialize_shot_graph
    from clipperx.local_paths import source_fingerprint
    from clipperx.tests.test_source_decode import video_line
    from clipperx.tests.test_source_host_ownership import ScriptProcess
    from clipperx.visual_compiler import compile_visual_assembly
    from clipperx.visual_contracts import CropKeyframe, NormalizedRect

    c = memory
    files, job = c.files, c.h.job
    host = HostContext(job, files, ROOT)
    files.register(ROOT, role="checked", owned=False, max_bytes=0)
    for name in ("assets", "shots", "diagnostics"):
        path = ROOT / name
        files.register(path, role="directory", owned=True, max_bytes=0)
        files.path_action("mkdir", path)
    action = files._path_action
    files._path_action = lambda op, p, d: (10**12, 1, 10**12-1) if op == "disk_usage" else action(op, p, d)

    source = Path("O:/HOST/n120-source.mp4")
    c.fs.data[source] = b"media"
    files.register(source, role="original", owned=False, max_bytes=100)
    plan = pure_plan(duration=2.1)
    plan.source_fingerprint = source_fingerprint(source, _files=files)
    plan.shots[0].layers[0].crop_track = [
        CropKeyframe(at=0, crop=NormalizedRect(x=0, y=0, width=1, height=1)),
        CropKeyframe(at=2, crop=NormalizedRect(x=.25, y=0, width=.75, height=1)),
        CropKeyframe(at=2.1, crop=NormalizedRect(x=.25, y=0, width=.75, height=1)),
    ]
    assembly = compile_visual_assembly(plan, source_size=(4, 4), overlays=pure_overlays(plan))
    assert len(assembly.shots) == 1 and assembly.shots[0].frame_count == 126
    track = assembly.shots[0].source_layers[0].crop_track
    assert track is not None and track.points[1].frame == 120
    requirements = rr.combine_render_requirements(sd.decode_requirements(), assembly.requirements)
    pair = rr._runtime_paths(requirements)
    for module in (rr, sd):
        monkeypatch.setattr(module, "FFMPEG_SHA256", hashlib.sha256(b"ffmpeg").hexdigest().upper())
        monkeypatch.setattr(module, "FFPROBE_SHA256", hashlib.sha256(b"ffprobe").hexdigest().upper())
    for path, raw in zip(pair, (b"ffmpeg", b"ffprobe")):
        c.fs.data[path] = raw
        files.register(path, role="runtime", owned=False, max_bytes=268435456)
    monkeypatch.setattr(sd, "_verify_pins", rr._verify_pins)
    monkeypatch.setattr(sd, "source_fingerprint", source_fingerprint)
    monkeypatch.setattr(sd.time, "monotonic_ns", lambda: c.h.now)
    monkeypatch.setattr(queue.Queue, "get", lambda self, *a, **k: _QUEUE_GET(self, False))
    monkeypatch.setattr(queue.Queue, "put", lambda self, item, *a, **k: _QUEUE_PUT(self, item, False))
    runtime = rr.resolve_render_runtime(requirements, device_index=0, _files=files)
    assert runtime.device_index == 0

    # Sparse external file IO still accepts, hashes and reads every logical
    # zero byte. No BGR/observer/mapper/spatial orchestration validator is replaced.
    sparse: dict[Path, Any] = {}
    largest_transfer = [0]
    original_open, original_stamp = files._open, c.fs.stamp

    class SparseFile:
        def __init__(self, path: Path, mode: str) -> None:
            self.path, self.mode, self.position, self.closed = path, mode, 0, False
            if mode == "xb":
                assert path not in c.fs.data
                c.fs.data[path] = b""
                sparse[path] = SimpleNamespace(size=0, digest=hashlib.sha256(), writes=0, reads=0,
                    handles=[])
            sparse[path].handles.append(self)
            c.fs.event("open", path)

        def read(self, size: int) -> bytes:
            assert self.mode == "rb" and not self.closed
            count = min(size, sparse[self.path].size-self.position)
            self.position += count
            sparse[self.path].reads += count
            largest_transfer[0] = max(largest_transfer[0], count)
            c.fs.event("read", self.path)
            return bytes(count)

        def write(self, view: memoryview) -> int:
            assert self.mode == "xb" and not self.closed
            raw = bytes(view)
            assert raw == bytes(len(raw)) and 0 < len(raw) <= 65536
            row = sparse[self.path]
            row.size += len(raw)
            row.digest.update(raw)
            row.writes += 1
            largest_transfer[0] = max(largest_transfer[0], len(raw))
            c.fs.event("write", self.path)
            return len(raw)

        def flush(self) -> None:
            c.fs.event("flush", self.path)

        def fileno(self) -> int:
            return 100

        def close(self) -> None:
            assert not self.closed
            self.closed = True
            c.fs.event("close", self.path)

    def open_file(path: Path, mode: str) -> Any:
        return SparseFile(path, mode) if path.suffix == ".rgba" else original_open(path, mode)

    def stamp(path: Path) -> Any:
        result = original_stamp(path)
        if path in sparse:
            result.st_size = sparse[path].size
        return result

    files._open = open_file
    c.fs.stamp = stamp

    class ZeroPipe:
        closed = False

        def __init__(self, size: int) -> None:
            self.remaining = size

        def read(self, size: int) -> bytes:
            assert not self.closed
            count = min(size, self.remaining)
            self.remaining -= count
            largest_transfer[0] = max(largest_transfer[0], count)
            return bytes(count)

        def close(self) -> None:
            self.closed = True

    commands: list[tuple[str, ...]] = []
    source_video_commands: list[tuple[str, ...]] = []
    source_audio_commands: list[tuple[str, ...]] = []
    active = [0, 0]
    expected_bytes = 8294400*126
    expected_bgr = bytes([11])*48*59+bytes([12])*48*60+bytes([13])*48*7
    metadata = {"format": {"start_time": "0", "duration": "3"}, "streams": [
        {"index": 0, "codec_type": "video", "time_base": "1/48000", "start_pts": 0,
         "duration_ts": 144000, "width": 4, "height": 4},
        {"index": 1, "codec_type": "audio", "time_base": "1/48000", "start_pts": 0,
         "duration_ts": 144000, "sample_rate": "48000", "channels": 1, "sample_fmt": "flt"}]}

    def process(argv: tuple[str, ...]) -> Any:
        assert job._registered and active[0] == 0
        active[0] += 1
        active[1] = max(active[1], active[0])
        commands.append(argv)
        raw, log, spatial = b"", b"", False
        if argv[0] == str(pair[1]):
            raw = json.dumps(metadata).encode()
        elif "-vf" in argv:
            source_video_commands.append(argv)
            pts = (0, 48000, 96000, 105600) if "-ss" not in argv else (72960, 96000, 105600)
            values = {0: 11, 48000: 12, 72960: 99, 96000: 13, 105600: 14}
            pixels = [bytes([values[p]])*48 for p in pts]
            raw = b"".join(pixels)
            log = ("[showinfo@cxv @ 000] config in time_base: 1/48000\n"+
                "\n".join(video_line(n, p, body, size="4x4") for n, (p, body) in enumerate(zip(pts, pixels)))+"\n").encode()
        elif "-af" in argv:
            source_audio_commands.append(argv)
            first = int(Fraction(argv[argv.index("-ss")+1])*48000) if "-ss" in argv else 0
            count = math.ceil(Fraction(argv[argv.index("-t")+1])*48000)
            blocks, lines = [], []
            for n, offset in enumerate(range(0, count, 65536)):
                size = min(65536, count-offset)
                body = bytes(size*2)
                blocks.append(body)
                checksum = zlib.adler32(body, 0)
                for tag, fmt in (("cxain", "flt"), ("cxaout", "s16")):
                    lines.append(f"[ashowinfo@{tag} @ 000] n:{n} pts:{first+offset} fmt:{fmt} channels:1 rate:48000 nb_samples:{size} checksum:{checksum:08X}")
            raw, log = b"".join(blocks), ("\n".join(lines)+"\n").encode()
        else:
            assert "-c:v" in argv and argv[argv.index("-c:v")+1] == "rawvideo"
            assert c.fs.data[ROOT / "shots/000000.bgr"] == expected_bgr
            spatial = True
        child = ScriptProcess(raw, log, pid=80+len(c.h.children))
        if spatial:
            setattr(child, "stdout", ZeroPipe(expected_bytes))
        wait = child.wait
        def waited(timeout: float) -> int:
            active[0] -= 1
            return wait(timeout)
        setattr(child, "wait", waited)
        c.h.children.append(child)
        return child

    job._processes = process
    observation_runtime = replace(runtime, device_index=None)
    assert sd._runtime_record(runtime) == sd._runtime_record(observation_runtime)
    origin = REAL_PROBE_SOURCE(source, observation_runtime, _files=files, _runner=host.tools)
    host.admission(assembly, 1)
    outputs, bindings, graphs = render_shots(host, assembly, runtime, origin)

    assert len(source_video_commands) == len(source_audio_commands) == len(bindings) == 2
    assert [tuple(b[k] for k in ("slot_start", "slot_end")) for b in bindings] == [(0, 120), (120, 126)]
    assert [b["source_finalization"]["payload"]["window"] for b in bindings] == [["1/50", "101/50"], ["101/50", "53/25"]]
    slots = [s for b in bindings for s in b["slots"]]
    assert [s.transport_slot for s in slots] == list(range(126))
    assert [s.request.global_k for s in slots] == [s.request.local_n for s in slots] == list(range(126))
    assert all(b["source_control"].phase == "retired_ok" and b["source_control"].window_resources_zero for b in bindings)
    spatial_results = [r for r in host.tools.evidence if r["kind"] == "spatial"]
    assert len(spatial_results) == len(outputs) == len(graphs) == 1
    spatial = spatial_results[0]
    assert spatial["stdout_bytes"] == expected_bytes and spatial["natural_eof"] and spatial["retired"]
    assert spatial["argv"].count("-i") == 1 and spatial["argv"][spatial["argv"].index("-i")+1].endswith("000000.bgr")
    assert spatial["argv"][spatial["argv"].index("-init_hw_device")+1] == "vulkan=vk:0"
    expected_graph = serialize_shot_graph(assembly.shots[0], (), {})
    assert graphs == (expected_graph,) and "120" in expected_graph and expected_graph.count("setpts=N[") == 1
    assert c.fs.data[ROOT / "assets/graph000000.txt"].decode() == expected_graph
    row = sparse[outputs[0]]
    assert row.size == row.reads == expected_bytes
    assert row.digest.hexdigest() == spatial["stdout_sha256"] == host.seals[str(outputs[0].relative_to(ROOT))]["sha256"]
    assert all(handle.closed for handle in row.handles)
    assert c.fs.data[outputs[0]] == b"" and largest_transfer[0] <= 1024**2
    assert active == [0, 1] and len(commands) == 6
    assert ROOT / "shots/000000.bgr" not in c.fs.data
    assert files.retire().files_clear
    assert not any(p.suffix in (".rgba", ".bgr") for p in c.fs.data)
    job.driver_done()
    c.h.driver.alive, c.h.current, c.h.workers[0].alive = False, None, False
    assert job.service_cleanup().own0


def test_global_png_adapter_and_ass_interleaving_preserve_fragments() -> None:
    from clipperx.tests.test_visual_compiler import interleaved_plan
    from clipperx.visual_compiler import compile_visual_assembly, AssApply, FilterFragment
    from clipperx._visual_render_video import serialize_shot_graph
    plan = interleaved_plan()
    assembly = compile_visual_assembly(plan, source_size=(2, 2), overlays=pure_overlays(plan))
    shot = assembly.shots[0]
    names = {d.key: f"assets/ass{i:06d}.ass" for i, d in enumerate(assembly.ass_documents)}
    graph = serialize_shot_graph(shot, assembly.brand_inputs, names)
    assert "[1:v]settb=1/60,setpts=N+0[" in graph
    position = 0
    for step in shot.steps:
        text = step.text if isinstance(step, FilterFragment) else (
            f"[{step.input_label}]ass=filename={names[step.document_key]}:fontsdir=fonts[{step.output_label}]")
        assert isinstance(step, (FilterFragment, AssApply))
        position = graph.index(text, position)+len(text)
