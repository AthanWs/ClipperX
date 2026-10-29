"""Concrete default HOST must exist before public v2 activation."""

import importlib.util
from typing import Any

import pytest

from clipperx.tests.test_retained_files import memory as memory, ROOT
from clipperx.tests.test_source_host_ownership import closed_boundaries as closed_boundaries, finish_job
from clipperx.tests.test_render_services import REAL_PROBE_SOURCE, _QUEUE_GET, _QUEUE_PUT


@pytest.fixture
def default_host(memory: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """Only OS/file/process/font byte boundaries are substituted before entry."""
    from array import array
    from fractions import Fraction as F
    import hashlib
    import json
    import queue
    import struct
    from pathlib import Path
    from types import SimpleNamespace
    import clipperx._visual_render_host as module
    from clipperx import source_decode as sd, render_runtime as rr, font_staging as fs
    from clipperx._source_host_ownership import SourceHostJob
    from clipperx._retained_files import RetainedFiles
    from clipperx._normalization_files import NormalizationFiles
    import clipperx._visual_render_audio as audio
    from clipperx.local_paths import source_fingerprint
    from clipperx.render_assets import RenderAssetRegistry
    from clipperx.render_contracts import RenderProfile, EncoderName
    from clipperx.tests.test_visual_compiler import pure_plan
    from clipperx.tests.test_source_host_ownership import ScriptWorker, ScriptProcess, scripted_host
    from clipperx.tests.test_source_decode import video_line
    from clipperx.tests.test_normalization_runner import OBSERVATION
    c = memory
    h = scripted_host()
    c.fs.dirs.update({Path("O:/HOST"), ROOT / "output"})
    pair = rr._runtime_paths(sd.decode_requirements())
    source, plan_path = Path("O:/HOST/source.mp4"), Path("O:/HOST/plan.json")
    c.fs.data[source] = b"media"
    fingerprints = source_fingerprint
    # Stat fingerprint is computed by the real formula, no source-body read.
    expected_fp = hashlib.sha256(f"{source}|5|100".encode()).hexdigest()[:16]
    plan = pure_plan(("source_video", "solid", "source_video"), duration=1/30)
    plan.source_fingerprint = expected_fp
    plan.shots[2].source.start = .4
    plan.shots[2].source.end = .4+1/30
    for shot in plan.shots:
        shot.audio.normalize_dialogue = False
    c.fs.data[plan_path] = plan.model_dump_json().encode()
    c.fs.dirs.add(Path("O:/HOST/original_fonts"))
    fonts = {key: Path("O:/HOST/original_fonts") / (key+".ttf") for key in fs._IDS}
    for key, path in fonts.items():
        c.fs.data[path] = key.encode()
    monkeypatch.setattr(fs, "_PINS", {key: hashlib.sha256(key.encode()).hexdigest() for key in fs._IDS})
    for path, raw in zip(pair, (b"ffmpeg", b"ffprobe")):
        c.fs.data[path] = raw
    hashes = [hashlib.sha256(c.fs.data[p]).hexdigest().upper() for p in pair]
    for mod in (rr, sd):
        monkeypatch.setattr(mod, "FFMPEG_SHA256", hashes[0])
        monkeypatch.setattr(mod, "FFPROBE_SHA256", hashes[1])
    monkeypatch.setattr(sd, "_verify_pins", rr._verify_pins)
    monkeypatch.setattr(sd, "source_fingerprint", fingerprints)
    monkeypatch.setattr(sd, "probe_source", REAL_PROBE_SOURCE)
    monkeypatch.setattr(module.time, "monotonic_ns", lambda: h.now)
    original_get, original_put = queue.Queue.get, queue.Queue.put
    # The shared fixture blocks queue waits. These exact original functions are
    # saved by a pre-imported test module before that guard was installed.
    monkeypatch.setattr(queue.Queue, "get", lambda self, *a, **k: _QUEUE_GET(self, False))
    monkeypatch.setattr(queue.Queue, "put", lambda self, item, *a, **k: _QUEUE_PUT(self, item, False))
    del original_get, original_put
    class Worker(ScriptWorker):
        def is_alive(self) -> bool:
            if self.name == "source-host-watchdog" and h.job._watch_exit:
                self.alive = False
            return self.alive
    def worker(target: Any, name: str, daemon: bool = False) -> Any:
        result = Worker(h, name, target)
        h.workers.append(result)
        return result
    monkeypatch.setattr(module.threading, "Thread", worker)
    calls: list[tuple[str, ...]] = []
    fault = {"kind": ""}
    counts = {"samples": 4800}
    active = [0, 0]
    source_meta = {"format": {"start_time": "0", "duration": "1"}, "streams": [
        {"index": 0, "codec_type": "video", "time_base": "1/48000", "start_pts": 0,
         "duration_ts": 48000, "start_time": "0", "duration": "1", "width": 2, "height": 2},
        {"index": 1, "codec_type": "audio", "time_base": "1/48000", "start_pts": 0,
         "duration_ts": 48000, "start_time": "0", "duration": "1", "sample_rate": "48000",
         "channels": 1, "channel_layout": "mono", "sample_fmt": "flt"}]}
    output_meta: dict[str, Any] = {"format": {"start_time": "0", "duration": "0.1"}, "streams": [
        {"codec_type": "video", "codec_name": "h264", "width": 1080, "height": 1920,
         "pix_fmt": "yuv420p", "avg_frame_rate": "60/1", "nb_read_frames": "6"},
        {"codec_type": "audio", "codec_name": "aac", "sample_rate": "48000", "channels": 1}]}
    def process(argv: tuple[str, ...]) -> Any:
        assert h.job._registered
        assert active[0] == 0
        calls.append(argv)
        active[0] += 1
        active[1] = max(active)
        raw, log = b"", b""
        if "-version" in argv:
            raw = b"ffmpeg version 9.0.2\n"
        elif "-encoders" in argv:
            raw = b" V..... libx264 video\n"
        elif "-filters" in argv:
            names = "anull atrim asetpts asplit concat volume alimiter color null settb setpts trim scale crop pad format overlay setsar split fps aformat aresample asettb ashowinfo showinfo adelay aeval amix apad libplacebo hwupload hwdownload"
            raw = "".join(f" ... {n} filter\n" for n in names.split()).encode()
        elif "-h" in argv and "filter=libplacebo" in argv:
            raw = "".join(f"  {name} <float> fixture\n" for name in rr.NATIVE_OPTIONS).encode()
        elif "testsrc2=s=16x16:r=60" in argv:
            raw, log = bytes(1024), b"Using device: synthetic Vulkan0\n"
        elif "-show_frames" in argv:
            if "v:0" in argv:
                raw = json.dumps({"frames": [{"best_effort_timestamp_time": str(float(F(k, 60)))} for k in range(6)]}).encode()
            else:
                raw = json.dumps({"frames": [{"best_effort_timestamp_time": "0", "nb_samples": 4800}]}).encode()
        elif argv[0] == str(pair[1]):
            raw = json.dumps(output_meta if "-count_frames" in argv else source_meta).encode()
        elif "-vf" in argv:
            # Fifteen source fps: each requested pair borrows one frame.
            pts_rows = (0, 19200, 28800) if float(argv[argv.index("-t")+1]) > .5 else (0, 3200, 6400)
            raw = b"".join(bytes([n])*12 for n in range(3))
            log = ("[showinfo@cxv @ 000] config in time_base: 1/48000\n"+
                "\n".join(video_line(n, pts, bytes([n])*12, size="2x2") for n, pts in enumerate(pts_rows))+"\n").encode()
        elif "-af" in argv and "ashowinfo" in argv[argv.index("-af")+1]:
            start = F(argv[argv.index("-ss")+1]) if "-ss" in argv else F(0)
            count = math.ceil(F(argv[argv.index("-t")+1])*48000)
            floating = "pcm_f32le" in argv
            raw = array("f", [.125]*count).tobytes() if floating else bytes(count*2)
            checksum = __import__("zlib").adler32(raw, 0)
            pts = int(start*48000)
            log = (f"[ashowinfo@cxain @ 000] n:0 pts:{pts} fmt:flt channels:1 rate:48000 nb_samples:{count} checksum:{checksum:08X}\n"
                f"[ashowinfo@cxaout @ 000] n:0 pts:{pts} fmt:{'flt' if floating else 's16'} channels:1 rate:48000 nb_samples:{count} checksum:{checksum:08X}\n").encode()
        elif "-af" in argv and "loudnorm" in argv[argv.index("-af")+1]:
            raw = b"" if "null" in argv else c.fs.data[Path(argv[argv.index("-i")+1])]
            log = OBSERVATION
        elif "-c:v" in argv and argv[argv.index("-c:v")+1] == "rawvideo":
            raw = bytes(8294400*2+(1 if fault["kind"] == "extra" else -1 if fault["kind"] == "short" else 0))
        elif "-c:a" in argv and "pcm_f32le" in argv:
            raw = array("f", [.125]*counts["samples"]).tobytes()
        elif argv[-1].endswith(".mp4"):
            c.fs.data[Path(argv[-1])] = struct.pack(">I4s", 8, b"ftyp")+struct.pack(">I4s", 8, b"moov")+struct.pack(">I4s", 8, b"mdat")
            if fault["kind"] == "font":
                c.fs.data[next(iter(fonts.values()))] = b"changed"
            elif fault["kind"] == "pin":
                c.fs.data[pair[0]] = b"changed"
            elif fault["kind"] == "count":
                output_meta["streams"][0]["nb_read_frames"] = "5"
            elif fault["kind"] == "truncated":
                c.fs.data[Path(argv[-1])] = b"broken"
            elif fault["kind"] == "cap":
                original_stamp = c.fs.stamp
                def oversized(path: Path) -> Any:
                    observed = original_stamp(path)
                    if path.suffix == ".mp4" and path != source:
                        observed.st_size = 2147483649
                    return observed
                c.fs.stamp = oversized
        elif "testsrc2=s=16x16:r=1" in argv:
            raw = bytes(768)
        child = ScriptProcess(raw, log, pid=80+len(h.children))
        finish = child.wait
        def waited(timeout: float) -> int:
            active[0] -= 1
            result = finish(timeout)
            if "-af" in argv and "ashowinfo" in argv[argv.index("-af")+1]:
                if fault["kind"] == "post_eof_pin":
                    c.fs.data[pair[0]] = b"changed"
                elif fault["kind"] == "post_eof_source":
                    c.fs.data[source] = b"changed"
            return result
        setattr(child, "wait", waited)
        h.children.append(child)
        return child
    def make_job(driver: Any, **kwargs: Any) -> Any:
        kwargs["process_factory"] = process
        h.driver = driver
        h.job = SourceHostJob(driver, **kwargs, io_factory=h.io_factory, worker_factory=worker,
            event_factory=h.job._events, current_worker=lambda: h.current)
        return h.job
    monkeypatch.setattr(module, "SourceHostJob", make_job)
    def make_files(job: Any, **kwargs: Any) -> Any:
        result = RetainedFiles(job, **kwargs, _open=c.fs.open, _stat=c.fs.stat,
            _path_action=lambda op, p, d: (10**12, 1, 10**12-1) if op == "disk_usage" else c.fs.action(op, p, d),
            _fsync=lambda _: None)
        c.files = result
        return result
    monkeypatch.setattr(module, "RetainedFiles", make_files)
    def file_stamp(file: Any) -> tuple[int, int, int]:
        return (1, c.fs.stamp(file.path).st_ino, len(c.fs.data[file.path]))
    monkeypatch.setattr(audio, "NormalizationFiles", lambda job, runtime, scratch, **kwargs:
        NormalizationFiles(job, runtime, scratch, **kwargs, _open=c.fs.open,
            _stat=file_stamp,
            _unlink=lambda p: c.fs.action("unlink", p, None)))
    import math
    return SimpleNamespace(c=c, h=h, module=module, source=source, plan_path=plan_path, plan=plan,
        assets=RenderAssetRegistry({}, fonts, {}), profile=RenderProfile(video_encoder=EncoderName.LIBX264),
        calls=calls, active=active, output=ROOT / "output", fault=fault, process=process, counts=counts, pair=pair)


def test_default_public_trace_source_solid_source_and_five_sidecars(default_host: Any,
        monkeypatch: pytest.MonkeyPatch) -> None:
    from clipperx.plan_render import render_visual_composition_plan
    from clipperx.source_decode import ObservedWindowStream
    import json
    import inspect
    h = default_host
    actual_ack = ObservedWindowStream.acknowledge
    acknowledgements = []
    def acknowledge(stream: Any, index: int) -> None:
        lease = stream._owner.lease
        assert len(lease.request_ordinals) == 2
        frame = inspect.currentframe()
        assert frame is not None
        caller = frame.f_back
        assert caller is not None
        assert not {"selection", "delivery", "view"} & set(caller.f_locals)
        bgr = [p for p in h.c.fs.data if p.suffix == ".bgr"]
        assert len(bgr) == 1 and len(h.c.fs.data[bgr[0]]) == 24
        assert sum(op == "write" and p == bgr[0] for op, p in h.c.fs.events) == 2
        acknowledgements.append((stream.session_id, index))
        actual_ack(stream, index)
    monkeypatch.setattr(ObservedWindowStream, "acknowledge", acknowledge)
    result = render_visual_composition_plan(h.source, h.plan_path, h.output, assets=h.assets, profile=h.profile)
    assert len(result) == 1 and result[0].name == "clip.mp4"
    files = [p for p in h.c.fs.data if p.parent == result[0].parent]
    assert {p.name for p in files} == {"clip.mp4", "clip.visual-plan.json", "clip.render-spec.json",
        "clip.metadata.json", "clip.render-metrics.json", "clip.ffmpeg.log"}
    assert h.active == [0, 1]
    assert len([a for a in h.calls if "-/filter_complex" in a and "-c:v" in a]) == 3
    assert len([a for a in h.calls if "-c:v" in a and "libx264" in a]) == 1
    metadata = json.loads(h.c.fs.data[result[0].with_suffix(".metadata.json")])
    assert [s["transport_slot"] for b in metadata["source_bindings"] for s in b["slots"]] == [0, 1, 2, 3]
    assert [s["request"]["global_k"] for b in metadata["source_bindings"] for s in b["slots"]] == [0, 1, 4, 5]
    assert all(len(b["slots"]) == 2 for b in metadata["source_bindings"])
    assert len(acknowledgements) == len(set(acknowledgements)) == 2
    assert metadata["process_count"] == len(h.calls)
    assert h.c.files.snapshot().generations <= metadata["limits"]["generation_reservation"]
    assert len(h.c.files.snapshot().paths) <= metadata["limits"]["path_reservation"]
    assert h.c.files.snapshot().files_clear and h.h.job.snapshot().own0


@pytest.mark.parametrize("mode", ["all", "mixed"])
def test_default_host_uses_real_two_pass_normalization_once(default_host: Any, mode: str) -> None:
    from clipperx._visual_render_audio import prepare_audio
    from clipperx._visual_render_host import HostContext
    from clipperx import render_runtime as rr, source_decode as sd
    from clipperx.tests.test_visual_compiler import pure_plan
    from clipperx.render_assets import RenderAssetRegistry
    from clipperx.visual_contracts import AudioCue
    from dataclasses import replace
    h = default_host
    plan = pure_plan(("solid", "solid", "solid"), duration=.4)
    plan.source_fingerprint = h.plan.source_fingerprint
    for index, shot in enumerate(plan.shots):
        shot.audio.normalize_dialogue = mode == "all" or index == 1
    h.counts["samples"] = 57600
    h.h.job = h.c.h.job
    h.c.h.job._processes = h.process
    files = h.c.files
    original_action = files._path_action
    files._path_action = lambda op, p, d: (10**12, 1, 10**12-1) if op == "disk_usage" else original_action(op, p, d)
    files.register(ROOT, role="checked", owned=False, max_bytes=0)
    files.register(h.source, role="original", owned=False, max_bytes=100)
    for p in h.pair:
        files.register(p, role="runtime", owned=False, max_bytes=268435456)
    observation_runtime = rr.resolve_render_runtime(sd.decode_requirements(), _files=files)
    runtime = replace(observation_runtime, device_index=0)
    assert sd._runtime_record(runtime) == sd._runtime_record(observation_runtime)
    origin = sd.probe_source(h.source, observation_runtime, _files=files,
        _runner=HostContext(h.c.h.job, files, ROOT).tools)
    host = HostContext(h.c.h.job, files, ROOT)
    cue = ROOT / "impact.wav"
    h.c.fs.data[cue] = b"registered-cue"
    files.register(cue, role="original", owned=False, max_bytes=100)
    plan.shots[0].audio.cues = [AudioCue(asset_id="impact", start=0, end=.1)]
    assets = RenderAssetRegistry({}, {}, {"impact": cue})
    requirements = rr.combine_render_requirements(sd.decode_requirements(), rr.RenderRequirements(
        rr.NATIVE_RUNTIME, needs_vulkan=True, required_filters=("libplacebo",)))
    finalmix, fmt, data = prepare_audio(host, plan, origin, runtime, assets, requirements)
    norm = [a for a in h.calls if "-af" in a and "loudnorm" in a[a.index("-af")+1]]
    assert len(norm) == 2
    assert data["normalization"]["receipt"].schedule_identity == data["schedule_identity"]
    assert data["normalization"]["receipt"].runtime_identity[-1] == "0"
    assert data["normalization"]["receipt"].runtime_identity == __import__(
        "clipperx.audio_mix", fromlist=["_runtime_identity"])._runtime_identity(runtime)
    assert data["qualification"].runtime.device_index == 0
    assert len(h.c.fs.data[finalmix]) == 57600*4 and fmt.channels == 1
    assert data["cue_inputs"][0][2] == ("1:a" if mode == "all" else "2:a")
    assert data["graph"].count("loudnorm") == 0
    assert h.c.files.retire().files_clear and finish_job(h.c.h).own0
    assert not any(p.suffix in (".bgr", ".rgba", ".f32") for p in h.c.fs.data)


def test_default_public_native_track_keeps_cpu_observation_and_vulkan_spatial_roles(default_host: Any) -> None:
    from clipperx.plan_render import render_visual_composition_plan
    from clipperx.visual_contracts import CropKeyframe, NormalizedRect
    import json
    h = default_host
    for shot in h.plan.shots:
        if shot.layers[0].kind.value != "solid":
            shot.layers[0].crop_track = [CropKeyframe(at=at,
                crop=NormalizedRect(x=0, y=0, width=1, height=1)) for at in (0, shot.source.duration)]
    h.c.fs.data[h.plan_path] = h.plan.model_dump_json().encode()
    result = render_visual_composition_plan(h.source, h.plan_path, h.output,
        assets=h.assets, profile=h.profile)[0]
    metadata = json.loads(h.c.fs.data[result.with_suffix(".metadata.json")])
    assert metadata["runtime"]["device_index"] == 0
    assert metadata["observation_runtime"]["device_index"] is None
    assert {k: v for k, v in metadata["runtime"].items() if k != "device_index"} == {
        k: v for k, v in metadata["observation_runtime"].items() if k != "device_index"}
    assert metadata["audio"]["qualification"]["runtime"]["device_index"] == 0
    spatial = [a for a in h.calls if "-/filter_complex" in a and "-c:v" in a]
    assert len(spatial) == 3 and all(a[a.index("-init_hw_device")+1] == "vulkan=vk:0" for a in spatial)
    assert len([a for a in h.calls if "testsrc2=s=16x16:r=60" in a]) == 2
    assert h.h.job.snapshot().own0 and h.c.files.snapshot().files_clear


@pytest.mark.parametrize("fault", ["short", "extra", "font", "pin", "count", "truncated", "cap", "post_eof_pin", "post_eof_source"])
def test_default_failures_never_publish(default_host: Any, fault: str) -> None:
    from clipperx.plan_render import render_visual_composition_plan
    h = default_host
    h.fault["kind"] = fault
    with pytest.raises(h.module.VisualRenderFailure) as failed:
        render_visual_composition_plan(h.source, h.plan_path, h.output, assets=h.assets, profile=h.profile)
    assert failed.value.host is not None
    assert h.h.job.snapshot().own0
    assert h.active == [0, 1]
    assert any("diagnostics" in p.parts for p in h.c.fs.data)


def test_all_solid_never_opens_video_observer_or_bgr(default_host: Any) -> None:
    from clipperx.plan_render import render_visual_composition_plan
    from clipperx.visual_contracts import LayerKind, FillMode
    h = default_host
    for shot in h.plan.shots:
        shot.layers[0].kind = LayerKind.SOLID
        shot.layers[0].fill = FillMode.SOLID
        shot.layers[0].solid_color = "#112233"
    h.c.fs.data[h.plan_path] = h.plan.model_dump_json().encode()
    render_visual_composition_plan(h.source, h.plan_path, h.output, assets=h.assets, profile=h.profile)
    assert not any("-vf" in a for a in h.calls)
    assert not any(p.suffix == ".bgr" for op, p in h.c.fs.events)


def test_explicit_v2_version_is_required_before_any_child(default_host: Any) -> None:
    from clipperx.plan_render import render_visual_composition_plan
    import json
    h = default_host
    payload = h.plan.model_dump(mode="json")
    del payload["contract_version"]
    h.c.fs.data[h.plan_path] = json.dumps(payload).encode()
    with pytest.raises(h.module.VisualRenderFailure, match="explicit contract_version 2"):
        render_visual_composition_plan(h.source, h.plan_path, h.output, assets=h.assets, profile=h.profile)
    assert h.calls == [] and h.h.job.snapshot().own0


def test_concrete_host_exists() -> None:
    assert importlib.util.find_spec("clipperx._visual_render_host") is not None


def test_bounded_retained_io_and_space_admission(memory: Any) -> None:
    from clipperx._visual_render_host import HostContext
    from clipperx.tests.test_visual_compiler import pure_plan, pure_overlays
    from clipperx.visual_compiler import compile_visual_assembly
    c = memory
    host = HostContext(c.h.job, c.files, ROOT)
    c.files.register(ROOT, role="checked", owned=False, max_bytes=0)
    path = ROOT / "asset.bin"
    c.fs.write_chunk = 2
    host.publish(path, b"abcdef", role="sidecar")
    assert c.fs.data[path] == b"abcdef"
    assert host.seal(path, 6)["bytes"] == 6
    assembly = compile_visual_assembly(pure_plan(), source_size=(2, 2), overlays=pure_overlays(pure_plan()))
    budget = host.admission(assembly, 1)
    assert budget["raw_peak"] == 8294400*6+12*6
    assert budget["generation_reservation"] < 262144
    action = c.files._path_action
    c.files._path_action = lambda op, p, d: (10**12, 1, 10**12-1) if op == "disk_usage" else action(op, p, d)
    host.recheck_space()
    c.files._path_action = lambda op, p, d: (100, 99, 1) if op == "disk_usage" else action(op, p, d)
    with pytest.raises(ValueError, match="free disk"):
        host.recheck_space()
    assert c.files.retire().files_clear
    assert finish_job(c.h).own0


def test_unknown_short_write_never_fabricates_success(memory: Any) -> None:
    from clipperx._visual_render_host import HostContext
    c = memory
    host = HostContext(c.h.job, c.files, ROOT)
    path = ROOT / "bad.raw"
    host.register(path, "raw", 8)
    opened = c.files._open
    def corrupt(p: Any, mode: str) -> Any:
        stream = opened(p, mode)
        stream.write = lambda _: None
        return stream
    c.files._open = corrupt
    with pytest.raises(ValueError):
        with c.files.open_file(path, "xb") as sink:
            host.write_all(sink, memoryview(b"abcd"))
    assert c.files.snapshot().rows[0].uncertain
    assert not c.files.retire().files_clear
    assert finish_job(c.h).own0


def test_sink_open_flush_close_have_owned_preparation_deadline(memory: Any) -> None:
    from clipperx._visual_render_host import HostContext
    c = memory
    host = HostContext(c.h.job, c.files, ROOT)
    path = ROOT / "clocked.raw"
    host.register(path, "raw", 4)
    seen = []
    def hook(op: str, _: Any) -> None:
        if op in ("open", "flush", "close"):
            assert c.h.job._driver_io_deadline is not None
            seen.append(op)
    c.fs.hook = hook
    with host.sink(path) as sink:
        with c.files.stage_deadline(host.deadline(120)):
            host.write_all(sink, memoryview(b"abcd"))
    assert seen == ["open", "flush", "close"]
    assert c.files.retire().files_clear and finish_job(c.h).own0


@pytest.mark.parametrize("fault", ["close", "replace", "late_open"])
def test_publication_uncertainty_retains_custody(memory: Any, fault: str) -> None:
    from clipperx._visual_render_host import HostContext
    c = memory
    host = HostContext(c.h.job, c.files, ROOT)
    path = ROOT / "evidence.json"
    if fault == "replace":
        action = c.files._path_action
        def replaced(op: str, p: Any, destination: Any) -> Any:
            result = action(op, p, destination)
            if op == "replace":
                raise OSError("replace completion uncertain")
            return result
        c.files._path_action = replaced
    else:
        def hook(op: str, p: Any) -> None:
            if op == "close" and fault == "close":
                raise OSError("close uncertain")
            if op == "open" and fault == "late_open":
                c.h.now = c.h.job._deadline
        c.fs.hook = hook
    with pytest.raises((ValueError, OSError)):
        host.publish(path, b"{}")
    before = sum(op == "close" for op, _ in c.fs.events)
    status = c.files.retire()
    if fault != "late_open":
        assert not status.files_clear
        assert before == sum(op == "close" for op, _ in c.fs.events)
    assert c.h.job.snapshot().failure is not None
