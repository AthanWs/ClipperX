"""Source-only candidate component tests: every native process escape is trapped."""

import builtins
import importlib.util
import io
import hashlib
import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from clipperx.tests.test_retained_files import memory, ROOT
from clipperx.tests.test_source_host_ownership import ScriptProcess, closed_boundaries


SOURCE = Path(__file__).resolve().parents[2] / "scripts/qualify_production_runtime.py"


def load_controller():
    assert SOURCE.is_file(), "Task112 qualification controller is not implemented"
    spec = importlib.util.spec_from_file_location("task112_controller", SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def native_escape_trap(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("native process escape")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "call", forbidden)
    monkeypatch.setattr(subprocess, "check_output", forbidden)


def test_true_default_cli_not_run_has_no_file_or_process_actions(monkeypatch, capsys):
    module = load_controller()
    # Read the source before trapping all path/open entry points. Compile/run the
    # actual __main__ branch, rather than substituting a call to a helper.
    code = compile(SOURCE.read_bytes(), str(SOURCE), "exec")
    def forbidden(*args, **kwargs):
        raise AssertionError("default CLI attempted a file/path action")
    with monkeypatch.context() as trapped:
        trapped.setattr(sys, "argv", [str(SOURCE)])
        trapped.setattr(builtins, "open", forbidden)
        trapped.setattr(io, "open", forbidden)
        for name in ("open", "resolve", "stat", "lstat", "mkdir", "exists", "is_file"):
            trapped.setattr(Path, name, forbidden)
        with pytest.raises(SystemExit) as stopped:
            exec(code, {"__name__": "__main__", "__file__": str(SOURCE)})
    assert stopped.value.code == 0
    assert capsys.readouterr().out == "NOT_RUN\n"
    assert module.main([]) == 0


@pytest.fixture
def q():
    return load_controller()


def test_fixed_hbft_build_identity(q):
    assert q.BUILD_ID == "ffmpeg-9.0.2-clipperx-production-v9-hbft"


def test_compact_admission_requires_external_identity(compact_case, q):
    case = compact_case
    bad = q.Pin(case.cli_pin.path, "0" * 64, case.cli_pin.bytes)
    with pytest.raises(ValueError, match="pin|identity|hash"):
        q.validate_inputs(bad, io=case.io, clock=case.clock)
    assert case.io.created == []
    assert case.native_attempts == []


def test_compact_recovery_mapping_is_one_exact_target(q):
    root = "O:/Escritorio/ClipperX/.superpowers/sdd/upstream-motion/production-build-v9-hbft"
    workspace = "O:/Escritorio/ClipperX"
    assert q.O9 == root
    assert not hasattr(q, "O8")  # No compatibility alias or old-target fallback.
    assert q.BUILD_ID == "ffmpeg-9.0.2-clipperx-production-v9-hbft"
    assert q.RECORD_PATHS == {
        "publisher_source": root + "/recovery.py", "publisher_lock": root + "/recovery-lock.json",
        "publisher_review": workspace + "/.superpowers/sdd/task-125-source-review.md",
        "inspection": root + "/evidence/inspect/inspection.json",
        "closure_review": root + "/reviewed/publication-closure.json",
        "source_notice": root + "/publish/SOURCE-NOTICE.json",
        "ffmpeg_map": root + "/build/ffmpeg/ffmpeg-link.map",
        "ffprobe_map": root + "/build/ffmpeg/ffprobe-link.map",
    }
    assert q.SOURCE_PATHS == {
        "controller": workspace + "/scripts/qualify_production_runtime.py",
        "tests": workspace + "/clipperx/tests/test_production_runtime_qualification.py",
        "design": workspace + "/.superpowers/sdd/task-116-compact-component-design.md",
        "design_review": workspace + "/.superpowers/sdd/task-116-review.md",
    }
    assert q.LINEAGE_PATHS == tuple(workspace + "/" + path for path in (
        ".superpowers/sdd/task-106-component-qualification-design.md", ".superpowers/sdd/task-112-brief.md",
        ".superpowers/sdd/task-120-ffmpeg-recovery-design.md", ".superpowers/sdd/task-120-design-review.md",
        "docs/superpowers/specs/2026-10-21-hbft-recovery-design.md", ".superpowers/sdd/task-125-design-review.md"))
    assert q.LINEAGE_PINS[q.LINEAGE_PATHS[4]] == (
        17969, "FF9A593DD3C7067ED25FE04590568DADF945867F3C4C1A9FDE09CBE9A22DB061")
    assert q.LINEAGE_PINS[q.LINEAGE_PATHS[5]] == (
        15345, "D66FE38156E801398A9FD0B4A724C0C106BD96D82AEF361DA702C20315FEE217")


@pytest.fixture
def compact_case(q, request):
    """Detached, synthetic v2 chain; no published candidate or native identity."""
    from copy import deepcopy
    root = q.WORKSPACE
    runtime_root, build_id, publisher_review = getattr(request, "param", (
        q.BASE + "/production-build-v9-hbft", "ffmpeg-9.0.2-clipperx-production-v9-hbft",
        root + "/.superpowers/sdd/task-125-source-review.md"))
    class FakeIO:
        def __init__(self):
            self.raws = {}
            self.created = []
            self.native_attempts = []
            self.streams = []
            self.identities = {}
            self.opened_identities = {}
            self.directories = {root + "/.superpowers/sdd/fake-only"}

        def check_path(self, path, *, absent=False, directory=False):
            assert type(path) is str
            if absent:
                if path in self.raws or path in self.directories:
                    raise ValueError("consumed evidence namespace")
            elif directory:
                if path not in self.directories:
                    raise ValueError("missing parent")
            elif path not in self.raws:
                raise ValueError("missing input")
            return path

        def open_read(self, path):
            self.check_path(path)
            class FakeStream(io.BytesIO):
                pass
            stream = FakeStream(self.raws[path])
            stream.identity = self.opened_identities.get(path, path.casefold())
            self.streams.append(stream)
            return stream

        def file_identity(self, path):
            return self.identities.get(path, path.casefold())

        def stream_identity(self, stream):
            return stream.identity

    fake = FakeIO()
    def put(path, raw):
        fake.raws[path] = raw
        return fake_pin(path, raw)
    pair = {name: put(runtime_root + "/publish/bin/" + name + ".exe", name.encode()) for name in ("ffmpeg", "ffprobe")}
    record_paths = {
        "publisher_source": runtime_root + "/recovery.py", "publisher_lock": runtime_root + "/recovery-lock.json",
        "publisher_review": publisher_review,
        "inspection": runtime_root + "/evidence/inspect/inspection.json",
        "closure_review": runtime_root + "/reviewed/publication-closure.json",
        "source_notice": runtime_root + "/publish/SOURCE-NOTICE.json",
        "ffmpeg_map": runtime_root + "/build/ffmpeg/ffmpeg-link.map",
        "ffprobe_map": runtime_root + "/build/ffmpeg/ffprobe-link.map",
    }
    records = {role: put(path, b"\xffopaque:" + role.encode()) for role, path in record_paths.items()}
    names = ("ffmpeg-COPYING.GPLv2", "libass-COPYING", "harfbuzz-COPYING", "freetype-LICENSE.TXT",
        "fribidi-COPYING", "x264-COPYING", "dav1d-COPYING", "nv-codec-headers-README",
        "libiconv-COPYING", "libiconv-COPYING.LIB")
    notices = [put(runtime_root + "/publish/notices/" + name, name.encode()) for name in names]
    iconv = {name: fake_pin(root + "/synthetic/iconv/" + name, name.encode()) for name in
        ("archive", "rejected_import", "header", "package", "copying", "copying_lib")}
    iconv["provenance_limit"] = "existing local MSYS2 package 1.17-4 metadata, not independently reverified distribution authenticity"
    candidate = dict(build_id=build_id, source_sha256="a" * 64,
        recovery_sha256="b" * 64, reuse_sha256="c" * 64, continuation_sha256="d" * 64,
        parent_source_sha256="e" * 64, reuse_manifest_path=root + "/synthetic/reuse.json",
        iconv=iconv, harfbuzz_source=root + "/synthetic/harfbuzz", source_commit="f" * 40,
        source_contract=deepcopy(q.SOURCE_CONTRACT), publisher_source_sha256=records["publisher_source"]["sha256"],
        publisher_lock_sha256=records["publisher_lock"]["sha256"],
        license_mode="GPL due to libx264; complete source/license publication needs review", status="candidate_published",
        pair=deepcopy(pair), notices=deepcopy(notices), inspection_sha256=records["inspection"]["sha256"],
        closure_review_sha256=records["closure_review"]["sha256"],
        qualification="pending separate original media/90-second/production gates")
    publication_path = runtime_root + "/publish/candidate.json"
    acceptance = dict(schema="clipperx-component-runtime-acceptance-v1", status="accepted_for_component_qualification",
        build_id=candidate["build_id"], runtime_root=runtime_root, publication=None, pair=deepcopy(pair),
        records=deepcopy(records), notices=deepcopy(notices))
    source_paths = {
        "controller": root + "/scripts/qualify_production_runtime.py",
        "tests": root + "/clipperx/tests/test_production_runtime_qualification.py",
        "design": root + "/.superpowers/sdd/task-116-compact-component-design.md",
        "design_review": root + "/.superpowers/sdd/task-116-review.md",
    }
    source = {role: put(path, (SOURCE.parents[1] / ".superpowers/sdd" /
        ("task-116-compact-component-design.md" if role == "design" else "task-116-review.md")).read_bytes()
        if role in ("design", "design_review") else ("synthetic:" + role).encode())
        for role, path in source_paths.items()}
    dependencies = [put(root + "/clipperx/" + module + ".py", module.encode()) for module in
        ("_render_services", "_retained_files", "_source_host_ownership", "_windows_pipe_io", "audio_mix",
         "filtergraph", "font_staging", "local_paths", "motion_compiler", "output_timeline", "overlay_ass",
         "planner_contracts", "render_assets", "render_clock", "render_contracts", "render_metrics",
         "render_runtime", "schema", "source_decode", "visual_compiler", "visual_contracts")]
    dependencies.extend([put(root + "/.superpowers/sdd/" + name,
        (SOURCE.parents[1] / ".superpowers/sdd" / name).read_bytes() if name.startswith("task-120") else name.encode())
        for name in ("task-106-component-qualification-design.md", "task-112-brief.md",
            "task-120-ffmpeg-recovery-design.md", "task-120-design-review.md")])
    dependencies.extend(put(root + "/" + name, (SOURCE.parents[1] / name).read_bytes()) for name in (
        "docs/superpowers/specs/2026-10-21-hbft-recovery-design.md", ".superpowers/sdd/task-125-design-review.md"))
    dependencies.sort(key=lambda row: row["path"])
    source["dependencies"] = dependencies
    assets = {role: put(path, role.encode()) for role, path in {
        "headline_impact": "C:/Windows/Fonts/impact.ttf", "subtitle_bold": "C:/Windows/Fonts/arialbd.ttf",
        "kick_brand": root + "/assets/branding/KickAthanW.png"}.items()}
    admission_path = root + "/.superpowers/sdd/fake-only/admission.json"
    acceptance_path = runtime_root + "/reviewed/component-runtime-acceptance.json"
    admission = dict(schema="clipperx-candidate-components-admission-v2", run_id="sample",
        evidence_parent=root + "/.superpowers/sdd/fake-only",
        evidence_root=root + "/.superpowers/sdd/fake-only/sample", runtime_acceptance=None,
        source_pins=source, assets=assets, limits=q.limits())
    class Case:
        io = fake
        clock = staticmethod(lambda: 100)
        native_attempts = fake.native_attempts
        def refresh(self):
            self.acceptance["publication"] = put(publication_path, q.canonical_json(self.candidate))
            self.admission["runtime_acceptance"] = put(acceptance_path, q.canonical_json(self.acceptance))
            self.cli_pin = q.Pin.parse(put(admission_path, q.canonical_json(self.admission)))
    case = Case()
    case.admission = admission
    case.acceptance = acceptance
    case.candidate = candidate
    case.refresh()
    return case


@pytest.mark.parametrize("compact_case", [
    ("O:/Escritorio/ClipperX/.superpowers/sdd/upstream-motion/production-build-v8-nvenc",
     "ffmpeg-9.0.2-clipperx-production-v8-nvenc",
     "O:/Escritorio/ClipperX/.superpowers/sdd/task-120-source-review.md"),
    ("O:/Escritorio/ClipperX/.superpowers/sdd/upstream-motion/production-build-foreign",
     "ffmpeg-foreign", "O:/Escritorio/ClipperX/.superpowers/sdd/foreign-source-review.md"),
], indirect=True, ids=["old-o8", "foreign"])
def test_hbft_mapping_rejects_old_or_foreign_producer(compact_case, q):
    with pytest.raises(ValueError, match="exact role path|fixed Root acceptance identity"):
        q.validate_inputs(compact_case.cli_pin, io=compact_case.io, clock=compact_case.clock)
    assert compact_case.io.created == [] and compact_case.native_attempts == []


@pytest.mark.parametrize("producer", ["ffmpeg-9.0.2-clipperx-production-v8-nvenc", "ffmpeg-foreign"])
@pytest.mark.parametrize("document", ["acceptance", "candidate"])
def test_hbft_mapping_rejects_old_or_foreign_build_in_o9(compact_case, q, producer, document):
    case = compact_case
    getattr(case, document)["build_id"] = producer
    case.refresh()
    with pytest.raises(ValueError, match="build identity|fixed Root acceptance identity"):
        q.validate_inputs(case.cli_pin, io=case.io, clock=case.clock)
    assert case.io.created == [] and case.native_attempts == []


@pytest.mark.parametrize("role", ["publisher_source", "publisher_lock", "publisher_review"])
@pytest.mark.parametrize("producer", ["old-o8", "foreign"])
def test_hbft_mapping_rejects_mixed_publisher_paths(compact_case, q, role, producer):
    case = compact_case
    if role == "publisher_review":
        path = q.WORKSPACE + "/.superpowers/sdd/" + (
            "task-120-source-review.md" if producer == "old-o8" else "foreign-source-review.md")
    else:
        root = q.BASE + ("/production-build-v8-nvenc" if producer == "old-o8" else "/production-build-foreign")
        path = root + ("/recovery.py" if role == "publisher_source" else "/recovery-lock.json")
    case.acceptance["records"][role]["path"] = path
    case.refresh()
    with pytest.raises(ValueError, match="exact role path"):
        q.validate_inputs(case.cli_pin, io=case.io, clock=case.clock)
    assert case.io.created == [] and case.native_attempts == []


@pytest.mark.parametrize("path", ["docs/superpowers/specs/2026-10-21-hbft-recovery-design.md",
    ".superpowers/sdd/task-125-design-review.md"])
@pytest.mark.parametrize("mutation", ["missing", "bytes", "sha256"])
def test_hbft_mapping_requires_reviewed_design_dependencies(compact_case, q, path, mutation):
    case = compact_case
    dependencies = case.admission["source_pins"]["dependencies"]
    pin = next(row for row in dependencies if row["path"] == q.WORKSPACE + "/" + path)
    if mutation == "missing":
        dependencies.remove(pin)
    elif mutation == "bytes":
        pin["bytes"] += 1
    else:
        pin["sha256"] = "0" * 64
    case.refresh()
    with pytest.raises(ValueError, match="required .*lineage|accepted .*dependency pin drift"):
        q.validate_inputs(case.cli_pin, io=case.io, clock=case.clock)
    assert case.io.created == [] and case.native_attempts == []


@pytest.mark.parametrize("role", ["inspection", "closure_review"])
def test_hbft_mapping_rejects_changed_proof_bytes_against_external_pin(compact_case, q, role):
    case = compact_case
    pin = case.acceptance["records"][role]
    # The consumer deliberately treats both payloads as opaque pinned bytes.
    # Producer/Root, rather than a second raw parser here, decide proof meaning.
    raw = q.canonical_json(dict(hbft_postcheck_sha256="1" * 64))
    case.io.raws[pin["path"]] = raw
    case.acceptance["records"][role] = fake_pin(pin["path"], raw)
    field = "inspection_sha256" if role == "inspection" else "closure_review_sha256"
    case.candidate[field] = case.acceptance["records"][role]["sha256"]
    case.refresh()
    inputs = q.validate_inputs(case.cli_pin, io=case.io, clock=case.clock)
    assert len(inputs.records) == 8 and len(inputs.notices) == 10
    assert len(case.candidate) == 20
    external_pin = case.acceptance["records"][role].copy()
    external_acceptance = case.io.raws[case.admission["runtime_acceptance"]["path"]]
    case.io.raws[pin["path"]] = raw.replace(b"1" * 64, b"2" * 64)
    with pytest.raises(ValueError, match="input pin hash/size mismatch"):
        q.validate_inputs(case.cli_pin, io=case.io, clock=case.clock)
    assert case.acceptance["records"][role] == external_pin
    assert case.io.raws[case.admission["runtime_acceptance"]["path"]] == external_acceptance
    assert all(stream.closed for stream in case.io.streams)
    assert case.io.created == [] and case.native_attempts == []


def test_compact_admission_validates_synthetic_chain(compact_case, q):
    inputs = q.validate_inputs(compact_case.cli_pin, io=compact_case.io, clock=compact_case.clock)
    assert inputs.pair.ffmpeg.path.endswith("/publish/bin/ffmpeg.exe")
    assert inputs.started_ns == 100
    assert all(compact_case.io.raws[pin.path].startswith(b"\xffopaque:") for _, pin in inputs.records)
    assert all(stream.closed for stream in compact_case.io.streams)
    assert compact_case.io.created == [] and compact_case.native_attempts == []


@pytest.mark.parametrize("failure,attempts", [
    (None, 24), ("guard-open", 0), ("brand-drift", 0), ("brand-valid", 24),
    ("controller-start", 0), ("controller-cleanup", 0), ("controller-join", 0),
    ("generic-device", 18), ("pin-drift", 1), ("free-space", 0), ("failed-spawn", 1), ("bad-source-probe", 10),
    ("bad-video", 11), ("bad-s16", 12), ("bad-float", 13), ("bad-mix", 14),
    ("bad-png", 15), ("bad-raster", 17), ("bad-vulkan", 18),
    ("bad-delivery", 20), ("direct-oversize", 9), ("direct-over-cap", 9),
    ("cleanup-uncertain", 24),
    ("terminal-short", 24), ("terminal-exception", 24), ("terminal-overrun", 24)],
    ids=lambda value: str(value))
def test_component_driver_exposes_complete_owned_campaign(compact_case, q, monkeypatch, failure, attempts):
    """Run all original owner/tools operations against detached fake pipes/files."""
    import struct
    from clipperx._source_host_ownership import SourceHostJob
    from clipperx._retained_files import RetainedFiles
    from clipperx.tests.test_source_host_ownership import scripted_host
    from clipperx.tests.test_retained_files import MemoryFS
    from PIL import Image
    from types import SimpleNamespace
    from clipperx import _retained_files as retained

    case = compact_case
    brand_rgba = bytes((240, 80, 40, 255)) * 16
    def png_bytes(pixels):
        with io.BytesIO() as stream:
            Image.frombytes("RGBA", (4, 4), pixels).save(stream, format="PNG")
            return stream.getvalue()
    admitted_brand = png_bytes(brand_rgba)
    changed_brand = png_bytes(bytes((240, 80, 41, 255)) * 16)
    assert len(admitted_brand) == len(changed_brand) and admitted_brand != changed_brand
    if failure in ("brand-drift", "brand-valid"):
        brand_path = case.admission["assets"]["kick_brand"]["path"]
        case.io.raws[brand_path] = admitted_brand
        case.admission["assets"]["kick_brand"] = fake_pin(brand_path, admitted_brand)
        case.refresh()
    host, fs = scripted_host(), MemoryFS()
    root = Path(case.admission["evidence_root"])
    fs.data.update({Path(path): raw for path, raw in case.io.raws.items()})
    fs.dirs.add(root.parent)
    original_action = fs.action
    def action(operation, path, destination):
        if failure == "cleanup-uncertain" and operation == "unlink" and path.name == "video.bgr":
            raise OSError("synthetic uncertain retirement delete")
        return ((2**40, 2**40, 1) if failure == "free-space" else (2**40, 2**40, 2**40)) \
            if operation == "disk_usage" else original_action(operation, path, destination)
    def owner(driver, *, deadline, clock, process):
        host.driver.target = driver._target
        previous = host.job
        host.job = SourceHostJob(host.driver, deadline_ns=deadline, clock=clock,
            io_factory=previous._io_factory, worker_factory=previous._workers,
            event_factory=previous._events, current_worker=previous._current_worker,
            process_factory=process)
        controller_interrupt = failure in ("controller-start", "controller-cleanup", "controller-join")
        interrupted = []
        def interrupt_once(seam):
            if failure == "controller-" + seam and not interrupted:
                interrupted.append(seam)
                raise KeyboardInterrupt("synthetic controller interrupt")
        if controller_interrupt:
            def deferred_start():
                host.driver.started += 1
                host.driver.ident, host.driver.alive = 1, True
            host.driver.start = deferred_start
            start = host.job.start_driver
            def watched_start():
                start()
                interrupt_once("start")
            host.job.start_driver = watched_start
            join = host.driver.join
            def watched_join(timeout):
                interrupt_once("join")
                return join(timeout)
            host.driver.join = watched_join
        service = host.job.service_cleanup
        def watched_service():
            interrupt_once("cleanup")
            if controller_interrupt and host.job.snapshot().failure is not None and host.driver.alive:
                prior, host.current = host.current, host.driver
                try:
                    host.driver.target()
                finally:
                    host.current, host.driver.alive = prior, False
            if host.job.snapshot().driver_done and host.workers:
                host.workers[0].alive = False  # scripted watchdog's observed exit
            return service()
        host.job.service_cleanup = watched_service
        return host.job
    original_init = RetainedFiles.__init__
    def files_init(self, *args, **kwargs):
        kwargs.update(_open=fs_open, _stat=fs.stat, _fsync=lambda _: None)
        kwargs.setdefault("_path_action", action)
        original_init(self, *args, **kwargs)
    def fs_open(path, mode):
        if failure == "guard-open" and path == root / "guard.json" and mode == "xb":
            raise OSError("synthetic guard open failure before leaf creation")
        return fs.open(path, mode)
    monkeypatch.setattr(RetainedFiles, "__init__", files_init)
    monkeypatch.setattr(retained, "_path_action", action)
    monkeypatch.setattr(q, "_new_owner", owner)

    def fake_fonts(registry, *, stage_dir, _files):
        files, fonts = _files, stage_dir
        files.register(fonts, role="directory", owned=True, max_bytes=0)
        files.path_action("mkdir", fonts)
        impact = fonts / "headline-impact.ttf"
        bold = fonts / "subtitle-bold.ttf"
        for path, raw in ((impact, b"tiny-fake-font"), (bold, b"tiny-fake-font-2")):
            files.register(path, role="font", owned=True, max_bytes=16384)
            with files.open_file(path, "xb") as stream:
                assert stream.write(memoryview(raw)) == len(raw)
                stream.flush()
                stream.fsync()
        if failure == "brand-drift":
            fs.data[Path(case.admission["assets"]["kick_brand"]["path"])] = changed_brand
        return SimpleNamespace(staged_registry=registry, fonts=(
            SimpleNamespace(staged_path=impact, asset_id="headline_impact"),
            SimpleNamespace(staged_path=bold, asset_id="subtitle_bold")))
    def fake_assets(files, root, assets, *, deadline):
        with files.stage_deadline(deadline):
            stage = fake_fonts(None, stage_dir=root / "fonts", _files=files)
        impact, bold = (font.staged_path for font in stage.fonts)
        prepared = root / "brand.png"
        q._owned_write(files, prepared, admitted_brand, role="asset", cap=16 * q.MiB, deadline=deadline)
        return prepared.name, impact.name, brand_rgba, (4, 4), (impact, bold, prepared)
    if failure not in ("brand-drift", "brand-valid"):
        monkeypatch.setattr(q, "_stage_candidate_assets", fake_assets)

    base = bytes((16, 32, 48, 255))
    inactive = bytearray(base * (320 * 180))
    def pixel(buf, x, y, color):
        buf[(y * 320 + x) * 4:(y * 320 + x + 1) * 4] = bytes(color)
    for y in range(16):
        for x in range(16):
            alpha = 0 if x < 5 else 128 if x < 11 else 255
            pixel(inactive, 8 + x, 8 + y,
                tuple(round(a * alpha / 255 + b * (1 - alpha / 255)) for a, b in zip((240, 40, 80), base[:3])) + (255,))
    for y in range(32):
        for x in range(64):
            pixel(inactive, 248 + x, 8 + y, (240, 80, 40, 255))
    active = bytearray(inactive)
    for x in range(8, 40): pixel(active, x, 50, (255, 255, 255, 255))
    for x in range(10, 50): pixel(active, x, 140, (255, 255, 255, 255))
    raster = bytes(inactive) * 30 + bytes(active) + bytes(inactive) * 29
    vulkan = b"".join(bytes((64 + n, 32 + y, 16 + x, 255))
        for n in range(4) for y in range(48) for x in range(48))
    fixture = q.fixture_bytes()
    s16 = b"".join(struct.pack("<h", max(-32768, min(32767, round(value * 32768))))
        for (value,) in struct.iter_unpack("<f", fixture["source.f32"]))
    mix = b"".join(struct.pack("<ff", .125 if j < 24000 else .1875,
        -.125 if j < 24000 else -.1875) for j in range(48000))
    with io.BytesIO() as stream:
        Image.frombytes("RGBA", (16, 16), fixture["alpha.rgba"]).save(stream, format="PNG")
        alpha_png = stream.getvalue()
    box = lambda tag: struct.pack(">I4s", 8, tag)
    mp4 = box(b"ftyp") + box(b"moov") + box(b"mdat")
    delivery = dict(streams=[dict(index=0, codec_type="video", codec_name="h264", width=1080,
        height=1920, pix_fmt="yuv420p", avg_frame_rate="60/1", nb_read_frames="60"),
        dict(index=1, codec_type="audio", codec_name="aac", sample_rate="48000", channels=2)],
        format=dict(start_time="0", duration="1"),
        frames=[dict(media_type="audio", stream_index=1, best_effort_timestamp_time="0", nb_samples=48000)] +
            [dict(media_type="video", stream_index=0, best_effort_timestamp_time=f"{i/60:.6f}")
             for i in range(60)])
    library_names = ("libavutil", "libavcodec", "libavformat", "libavdevice", "libavfilter",
        "libswscale", "libswresample")
    version = ("ffmpeg version 9.0.2 fake\nconfiguration: --enable-gpl --enable-libx264\n" +
        "".join(f"{name} 61. 19.100 / 61. 19.100\n" for name in library_names)).encode()
    filters = ("adelay aeval aformat alimiter amix anull apad aresample asetpts asettb "
        "ashowinfo asplit ass atrim boxblur color concat crop format fps libplacebo loudnorm null "
        "overlay scale setpts setsar settb showinfo split trim volume drawtext geq drawbox hstack "
        "select testsrc testsrc2 nullsrc").split()
    codecs = "h264 hevc vp8 vp9 mpeg1video mpeg2video png qtrle pcm_f32le pcm_s16le aac opus vorbis mp3 flac av1".split()
    formats = "mpegts flv avi rawvideo f32le s16le wav null image2pipe".split()
    encoders = "h264_nvenc libx264 aac qtrle pcm_f32le pcm_s16le png".split()
    video_log = b"".join(f"[showinfo@cxv @ 0] n: {i} pts: {75+i} fmt: bgr24 s: 32x32 checksum: 00000000\n".encode()
        for i in range(60))
    def audio_log(fmt):
        return (f"[ashowinfo@cxain @ 0] n: 0 pts: 60000 rate: 48000 channels: 2 nb_samples: 48000 fmt: flt checksum: 00000000\n"
            f"[ashowinfo@cxaout @ 0] n: 0 pts: 60000 rate: 48000 channels: 2 nb_samples: 48000 fmt: {fmt} checksum: 00000000\n").encode()
    payloads = [version, version.replace(b"ffmpeg", b"ffprobe"),
        b" ".join(flag.encode() for flag in ("--enable-gpl", "--enable-libx264", "--enable-libplacebo",
            "--enable-libass", "--enable-libfreetype", "--enable-libharfbuzz",
            "--enable-libfribidi", "--enable-libdav1d")),
        "".join(f"... {name} filter\n" for name in filters).encode(),
        "".join(f"D..... {name} codec" + (" (decoders: libdav1d av1)" if name == "av1" else "") + "\n"
            for name in codecs).encode(),
        ("DE  mov,mp4,m4a QuickTime\nDE  matroska,webm Matroska\n" +
            "".join(f"DE  {name} format\n" for name in formats)).encode(),
        "".join(f"...... {name} encoder\n" for name in encoders).encode(),
        "".join(f"  {name} <string> option\n" for name in
            __import__("clipperx.render_runtime", fromlist=["NATIVE_OPTIONS"]).NATIVE_OPTIONS).encode(),
        b"", q.canonical_json(source_probe_payload()), fixture["source.bgr"], s16,
        fixture["source.f32"], mix, alpha_png, fixture["alpha.rgba"], raster, vulkan,
        b"", q.canonical_json(delivery), b"", b"", q.canonical_json(delivery), b""]
    device_log = b"[Vulkan @ 0] Device 0 selected: FakeVulkan GPU (discrete) (0x1234)\n[Vulkan @ 0] Using device: FakeVulkan GPU\n"
    logs = {11: video_log, 12: audio_log("s16"), 13: audio_log("flt"), 18: device_log}
    if failure == "generic-device":
        logs[18] = b"device extensions enabled\n"
    if failure == "bad-source-probe":
        broken = source_probe_payload()
        broken["frames"][0]["pts"] = 0
        payloads[9] = q.canonical_json(broken)
    elif failure == "bad-video":
        payloads[10] = b"X" + payloads[10][1:]
    elif failure == "bad-s16":
        payloads[11] = b"X" + payloads[11][1:]
    elif failure == "bad-float":
        payloads[12] = b"X" + payloads[12][1:]
    elif failure == "bad-mix":
        payloads[13] = bytes(384000)
    elif failure == "bad-png":
        payloads[14] = b"invalid-png"
    elif failure == "bad-raster":
        broken = bytearray(payloads[16])
        broken[30 * 230400] ^= 1
        payloads[16] = bytes(broken)
    elif failure == "bad-vulkan":
        payloads[17] = bytes((64, 64, 64, 255)) * (48 * 48 * 4)
    elif failure == "bad-delivery":
        broken = json.loads(q.canonical_json(delivery))
        broken["frames"][0]["best_effort_timestamp_time"] = "0.1"
        payloads[19] = q.canonical_json(broken)
    maximum_live = [0]
    def process(argv, root):
        index = len(case.native_attempts) + 1
        maximum_live[0] = max(maximum_live[0], sum(child.returncode is None for child in host.children))
        assert maximum_live[0] == 0
        case.native_attempts.append(argv)
        if failure == "failed-spawn":
            raise OSError("synthetic process creation failed")
        if failure == "pin-drift" and index == 1:
            fs.data[Path(case.candidate["pair"]["ffmpeg"]["path"])] = b"changed"
        if index in (9, 19, 22):
            fs.data[Path(root / {9: "source.mov", 19: "cpu.mp4", 22: "gpu.mp4"}[index])] = (
                b"x" * (16 * q.MiB + 1) if failure == "direct-over-cap" and index == 9 else
                b"x" * 8388608 if failure == "direct-oversize" and index == 9 else
                b"synthetic-mov" if index == 9 else mp4)
        child = ScriptProcess(payloads[index - 1], logs.get(index, b""), pid=70 + index)
        host.children.append(child)
        return child
    monkeypatch.setattr(q, "_default_process", process)
    clock_state = [100]
    def terminal_write(path, raw):
        assert host.job.snapshot().own0 and holder["files"].snapshot().files_clear
        case.io.created.append((path, raw))
        if failure == "terminal-exception":
            raise OSError("synthetic terminal write failure")
        if failure == "terminal-overrun":
            clock_state[0] = 1200 * 10**9 + 100
        return len(raw) - 1 if failure == "terminal-short" else len(raw)
    holder = {}
    original_files = q._new_files
    def watched_files(job, owner_root, clock):
        made = original_files(job, owner_root, clock)
        holder["files"] = made
        return made
    monkeypatch.setattr(q, "_new_files", watched_files)
    case.io.write_exclusive = terminal_write
    def campaign_clock():
        if (host.job._failure is not None and not host.driver.alive
                and failure not in ("controller-start", "controller-cleanup", "controller-join")):
            return host.job._failure.cleanup_deadline_ns + 1
        return clock_state[0]
    case.clock = campaign_clock
    campaign = q.admit(case.cli_pin, io=case.io, clock=case.clock)
    if failure in ("brand-drift", "brand-valid"):
        from clipperx import font_staging
        monkeypatch.setattr(font_staging, "stage_original_fonts", fake_fonts)
        monkeypatch.setattr(font_staging, "verify_font_stage", lambda *args, **kwargs: None)
    try:
        result = campaign.execute()
    except KeyboardInterrupt:
        pytest.fail("controller interrupt escaped original-owner cancellation/cleanup")
    assert len(case.native_attempts) == attempts, result
    assert result["attempts"] == attempts
    if failure not in (None, "brand-valid"):
        assert result["candidate_component_passed"] is False, result
        assert result["status"] in ("STOPPED", "UNCERTAIN")
        assert result["runtime_integrated"] is False
        assert result["production_accepted"] is False
        if failure == "guard-open":
            assert root in fs.dirs, "successful mkdir permanently consumes the exact namespace"
            assert root / "guard.json" not in fs.data and not case.io.created
            assert host.job.snapshot().driver_done
            assert not holder["files"].snapshot().files_clear
        if failure.startswith("controller-"):
            status = host.job.snapshot()
            assert status.failure.code == "cancelled" and status.driver_done and status.own0
            assert status.failure.cleanup_deadline_ns == clock_state[0] + 5 * 10**9
            assert holder["files"].snapshot().files_clear and not case.io.created
        return
    assert result["candidate_component_passed"] is True, result
    pair, origin = campaign.inputs.pair, q.candidate_origin(str(root / "source.mov"),
        len(b"synthetic-mov"), 100, pair=campaign.inputs.pair)
    observed_argv = case.native_attempts
    brand_name = "brand-v1-" + q.digest(admitted_brand) + ".png" if failure == "brand-valid" else "brand.png"
    assert observed_argv == list(q.operation_argv(pair, origin=origin, brand=brand_name, impact="headline-impact.ttf"))
    assert len(observed_argv) == 24
    assert maximum_live == [0]
    assert host.job.snapshot().own0 and holder["files"].snapshot().files_clear
    assert all(root / f"capture-{i:02}.bin" in fs.data for i in (*range(1, 9), 10, 20, 23))
    raster_record = json.loads(fs.data[root / "operation-17.json"])
    reviews = raster_record["oracle"]["review_frames"]
    assert [frame["frame"] for frame in reviews] == [0, 30, 59]
    paths = {path.path: path for path in holder["files"].snapshot().paths}
    for frame in reviews:
        image_path = Path(frame["path"])
        image_raw = fs.data[image_path]
        assert frame["width"] == 320 and frame["height"] == 180
        assert frame["bytes"] == len(image_raw) and frame["sha256"] == q.digest(image_raw)
        assert paths[str(image_path)].role == "asset" and paths[str(image_path)].retained
        with io.BytesIO(image_raw) as stream, Image.open(stream) as image:
            assert image.format == "PNG" and image.mode == "RGBA" and image.size == (320, 180)
            assert image.tobytes() == raster[frame["frame"] * 230400:(frame["frame"] + 1) * 230400]
    assert all(root / name not in fs.data for name in ("raster.rgba", "video.bgr", "audio.s16", "audio.f32", "vulkan.rgba"))
    vulkan_record = json.loads(fs.data[root / "operation-18.json"])
    observation = vulkan_record["oracle"]
    assert observation["selected_device"] == "FakeVulkan GPU"
    log = observation["device_log"]
    assert fs.data[Path(log["path"])] == device_log
    assert log["bytes"] == len(device_log) and log["sha256"] == q.digest(device_log)
    assert paths[log["path"]].role == "sidecar" and paths[log["path"]].retained
    assert result["runtime_integrated"] is False
    assert result["production_accepted"] is False


def test_codec_listing_requires_av1_with_libdav1d_decoder(q):
    names = "h264 hevc vp8 vp9 mpeg1video mpeg2video png qtrle pcm_f32le pcm_s16le aac opus vorbis mp3 flac av1".split()
    raw = "".join(f"D.V.L. {name} codec" + (" (decoders: libdav1d av1)" if name == "av1" else "") + "\n"
        for name in names).encode()
    q._probe_oracle(5, raw, None)
    with pytest.raises(ValueError, match="codec"):
        q._probe_oracle(5, raw.replace(b"libdav1d", b"native_av1"), None)
    with pytest.raises(ValueError, match="codec"):
        q._probe_oracle(5, raw.replace(b"D.V.L. av1", b"E.V.L. av1"), None)


def test_libplacebo_listing_requires_exact_option_rows(q):
    from clipperx.render_runtime import NATIVE_OPTIONS
    raw = "".join(f"  {name} <string> option\n" for name in NATIVE_OPTIONS).encode()
    q._probe_oracle(8, raw, None)
    with pytest.raises(ValueError, match="option"):
        q._probe_oracle(8, raw.replace(b"  w <", b"  wider <", 1), None)


def test_version_probe_requires_complete_unique_libraries_and_matching_configuration(q):
    names = ("libavutil", "libavcodec", "libavformat", "libavdevice", "libavfilter",
        "libswscale", "libswresample")
    def listing(tool, libraries=names, configurations=("--enable-gpl --enable-libx264",)):
        return (f"{tool} version 9.0.2 synthetic\n" +
            "".join(f"configuration: {value}\n" for value in configurations) +
            "".join(f"{name} 61. 19.100 / 61. 19.100\n" for name in libraries)).encode()
    ffmpeg = q._probe_oracle(1, listing("ffmpeg"), None)
    q._probe_oracle(2, listing("ffprobe"), ffmpeg)
    for raw in (listing("ffprobe", names[:-1]), listing("ffprobe", names + (names[0],)),
            listing("ffprobe", names + ("libpostproc",)),
            listing("ffprobe", configurations=()),
            listing("ffprobe", configurations=("--enable-gpl", "--enable-libx264")),
            listing("ffprobe", configurations=("--enable-gpl --disable-libx264",)),
            listing("ffprobe") + b"libavcodec malformed duplicate\n"):
        with pytest.raises(ValueError, match="librar|configuration"):
            q._probe_oracle(2, raw, ffmpeg)


def test_version_probe_accepts_exact_seven_pinned_ffmpeg9_libraries(q):
    names = ("libavutil", "libavcodec", "libavformat", "libavdevice", "libavfilter",
        "libswscale", "libswresample")
    def listing(tool):
        return (f"{tool} version 9.0.2 synthetic\nconfiguration: --enable-gpl\n" +
            "".join(f"{name} 61. 19.100 / 61. 19.100\n" for name in names)).encode()
    identity = q._probe_oracle(1, listing("ffmpeg"), None)
    q._probe_oracle(2, listing("ffprobe"), identity)


def test_format_probe_requires_complete_comma_aliases(q):
    rows = (" D  mov,mp4,m4a QuickTime\n"
        " D  matroska,webm Matroska\n" +
        "".join(f"DE  {name} format\n" for name in
            "mpegts flv avi rawvideo f32le s16le wav null image2pipe".split())).encode()
    q._probe_oracle(6, rows, None)
    for missing in (rows.replace(b"mov,mp4,m4a", b"mov,m4a"),
            rows.replace(b"matroska,webm", b"matroska"),
            rows.replace(b"mov,mp4,m4a", b"mov,m4a") + b"DE  mp4 unrelated\n",
            rows.replace(b"matroska,webm", b"matroska") + b"DE  webm unrelated\n"):
        with pytest.raises(ValueError, match="mux/demux|format"):
            q._probe_oracle(6, missing, None)


@pytest.mark.parametrize("passed,expected", [(True, 0), (False, 2)])
def test_activated_cli_uses_campaign_status_for_exit(q, monkeypatch, capsys, passed, expected):
    class Campaign:
        def execute(self):
            return dict(status="candidate_component_passed" if passed else "STOPPED",
                candidate_component_passed=passed)
    monkeypatch.setattr(q, "admit", lambda pin: Campaign())
    args = ["--native", "--confirm-host", "--admission", "O:/fake/admission.json",
        "--admission-sha256", "a" * 64, "--admission-bytes", "2"]
    assert q.main(args) == expected
    assert capsys.readouterr().out == ("candidate_component_passed\n" if passed else "STOPPED\n")


def test_compact_admission_refuses_source_alias_and_controller_origin(compact_case, q, monkeypatch):
    case = compact_case
    path = case.admission["source_pins"]["dependencies"][0]["path"]
    alias = path.replace("/", "\\")
    case.io.raws[alias] = case.io.raws[path]
    case.admission["source_pins"]["dependencies"][0]["path"] = alias
    case.refresh()
    with pytest.raises(ValueError, match="alias|canonical"):
        q.validate_inputs(case.cli_pin, io=case.io, clock=case.clock)
    case.admission["source_pins"]["dependencies"][0]["path"] = path
    case.refresh()
    monkeypatch.setattr(q, "__file__", "O:/different/controller.py")
    with pytest.raises(ValueError, match="controller.*origin"):
        q.validate_inputs(case.cli_pin, io=case.io, clock=case.clock)


def test_compact_admission_refuses_loaded_helper_origin_drift(compact_case, q, monkeypatch):
    import clipperx.render_clock as helper
    monkeypatch.setattr(helper.__spec__, "origin", "O:/different/render_clock.py")
    with pytest.raises(ValueError, match="helper spec origin"):
        q.validate_inputs(compact_case.cli_pin, io=compact_case.io, clock=compact_case.clock)
    assert compact_case.io.created == [] and compact_case.native_attempts == []


def test_compact_admission_refuses_iconv_slash_alias_without_dereference(compact_case, q):
    case = compact_case
    archive = case.candidate["iconv"]["archive"].copy()
    alias = archive["path"].replace("/", "\\")
    case.candidate["iconv"]["header"] = {**archive, "path": alias}
    assert archive["path"] not in case.io.raws and alias not in case.io.raws
    case.refresh()
    with pytest.raises(ValueError, match="iconv.*canonical|canonical.*iconv"):
        q.validate_inputs(case.cli_pin, io=case.io, clock=case.clock)
    assert archive["path"] not in case.io.raws and alias not in case.io.raws
    assert case.io.created == [] and case.native_attempts == []


def test_compact_admission_uses_hashed_stream_identity_after_path_lookup(compact_case, q):
    case = compact_case
    first = case.acceptance["records"]["ffmpeg_map"]["path"]
    second = case.acceptance["records"]["ffprobe_map"]["path"]
    assert case.io.file_identity(first) != case.io.file_identity(second)
    case.io.opened_identities[second] = first.casefold()
    with pytest.raises(ValueError, match="file-identity alias"):
        q.validate_inputs(case.cli_pin, io=case.io, clock=case.clock)
    assert all(stream.closed for stream in case.io.streams)
    assert case.io.created == [] and case.native_attempts == []


@pytest.mark.parametrize("mutation", [
    "v1", "missing", "extra", "run", "limit_bool", "limit_tight", "limit_loose",
    "acceptance_schema", "acceptance_status", "acceptance_root", "acceptance_missing",
    "acceptance_extra", "publication_role", "pair_role", "pair_oversize",
    "record_role", "record_bool", "record_missing", "map_oversize", "notice_order", "notice_missing",
    "notice_duplicate", "notice_oversize",
    "candidate_field", "candidate_oversize", "candidate_pair", "candidate_notice", "candidate_digest",
    "iconv_extra", "iconv_duplicate", "source_contract_extra", "source_contract_bool", "source_commit",
    "source_origin_path", "source_oversize", "dependency_missing", "dependency_unsorted", "helper_pin_drift",
    "task120_pin_bad", "asset_path", "asset_oversize", "dependency_conflict", "overlap",
])
def test_compact_admission_rejects_closed_contract_mutations(compact_case, q, mutation):
    case = compact_case
    a, r, c = case.admission, case.acceptance, case.candidate
    if mutation == "v1": a["schema"] = "clipperx-candidate-components-admission-v1"
    elif mutation == "missing": del a["assets"]
    elif mutation == "extra": a["other"] = True
    elif mutation == "run": a["run_id"] = "BAD"
    elif mutation == "limit_bool": a["limits"]["operations"][0]["index"] = True
    elif mutation == "limit_tight": a["limits"]["children"] = 23
    elif mutation == "limit_loose": a["limits"]["children"] = 25
    elif mutation == "acceptance_schema": r["schema"] = "old"
    elif mutation == "acceptance_status": r["status"] = "candidate_published"
    elif mutation == "acceptance_root": r["runtime_root"] = "O:/different"
    elif mutation == "acceptance_missing": del r["status"]
    elif mutation == "acceptance_extra": r["extra"] = 1
    elif mutation == "publication_role": pass
    elif mutation == "pair_role": r["pair"]["ffprobe"]["path"] = "O:/different/ffprobe.exe"
    elif mutation == "pair_oversize": r["pair"]["ffprobe"]["bytes"] = 268435457
    elif mutation == "record_role": r["records"]["ffmpeg_map"]["path"] = "O:/different/map"
    elif mutation == "record_bool": r["records"]["inspection"]["bytes"] = True
    elif mutation == "record_missing": del r["records"]["inspection"]
    elif mutation == "map_oversize": r["records"]["ffprobe_map"]["bytes"] = 67108865
    elif mutation == "notice_order": r["notices"][0], r["notices"][1] = r["notices"][1], r["notices"][0]
    elif mutation == "notice_missing": r["notices"].pop()
    elif mutation == "notice_duplicate": r["notices"][1] = r["notices"][0]
    elif mutation == "notice_oversize": r["notices"][0]["bytes"] = 1048577
    elif mutation == "candidate_field": c["other"] = 1
    elif mutation == "candidate_oversize": c["padding"] = "x" * 16384
    elif mutation == "candidate_pair": c["pair"]["ffprobe"]["sha256"] = "a" * 64
    elif mutation == "candidate_notice": c["notices"][0], c["notices"][1] = c["notices"][1], c["notices"][0]
    elif mutation == "candidate_digest": c["publisher_lock_sha256"] = "a" * 64
    elif mutation == "iconv_extra": c["iconv"]["other"] = 1
    elif mutation == "iconv_duplicate": c["iconv"]["header"] = c["iconv"]["archive"]
    elif mutation == "source_contract_extra": c["source_contract"]["other"] = 1
    elif mutation == "source_contract_bool": c["source_contract"]["local_registered_file"] = 1
    elif mutation == "source_commit": c["source_commit"] = "A" * 40
    elif mutation == "source_origin_path": a["source_pins"]["controller"]["path"] = "O:/other/controller.py"
    elif mutation == "source_oversize": a["source_pins"]["controller"]["bytes"] = 1048577
    elif mutation == "dependency_missing": a["source_pins"]["dependencies"].pop(0)
    elif mutation == "dependency_unsorted": a["source_pins"]["dependencies"].reverse()
    elif mutation == "helper_pin_drift": next(row for row in a["source_pins"]["dependencies"]
        if row["path"].endswith("render_clock.py"))["sha256"] = "a" * 64
    elif mutation == "task120_pin_bad": next(row for row in a["source_pins"]["dependencies"]
        if row["path"].endswith("task-120-design-review.md"))["sha256"] = "a" * 64
    elif mutation == "asset_path": a["assets"]["kick_brand"]["path"] = "O:/different/brand.png"
    elif mutation == "asset_oversize": a["assets"]["kick_brand"]["bytes"] = 16777217
    elif mutation == "dependency_conflict":
        a["source_pins"]["dependencies"].append(a["assets"]["kick_brand"].copy())
        a["source_pins"]["dependencies"].sort(key=lambda row: row["path"])
    elif mutation == "overlap":
        a["evidence_parent"] = q.O9
        a["evidence_root"] = q.O9 + "/sample"
        case.io.directories.add(q.O9)
    case.refresh()
    if mutation == "publication_role":
        r["publication"]["path"] = "O:/different/candidate.json"
        acceptance_path = a["runtime_acceptance"]["path"]
        raw = q.canonical_json(r)
        case.io.raws[acceptance_path] = raw
        a["runtime_acceptance"] = fake_pin(acceptance_path, raw)
        raw = q.canonical_json(a)
        case.io.raws[case.cli_pin.path] = raw
        case.cli_pin = q.Pin.parse(fake_pin(case.cli_pin.path, raw))
    with pytest.raises(ValueError):
        q.validate_inputs(case.cli_pin, io=case.io, clock=case.clock)
    assert case.io.created == [] and case.native_attempts == []


@pytest.mark.parametrize("mutation", ["duplicate", "nonfinite", "oversize_admission", "oversize_acceptance", "raw_drift", "file_alias", "consumed_root", "reparse"])
def test_compact_admission_rejects_raw_and_filesystem_mutations(compact_case, q, mutation):
    case = compact_case
    case.refresh()
    if mutation == "duplicate":
        raw = case.io.raws[case.cli_pin.path].replace(b'"schema":', b'"schema":"duplicate","schema":', 1)
        case.io.raws[case.cli_pin.path] = raw
        case.cli_pin = q.Pin(case.cli_pin.path, hashlib.sha256(raw).hexdigest(), len(raw))
    elif mutation == "nonfinite":
        raw = case.io.raws[case.cli_pin.path].replace(b'"children":24', b'"children":NaN', 1)
        case.io.raws[case.cli_pin.path] = raw
        case.cli_pin = q.Pin(case.cli_pin.path, hashlib.sha256(raw).hexdigest(), len(raw))
    elif mutation == "oversize_admission":
        raw = case.io.raws[case.cli_pin.path] + b" " * 1048576
        case.io.raws[case.cli_pin.path] = raw
        case.cli_pin = q.Pin(case.cli_pin.path, hashlib.sha256(raw).hexdigest(), len(raw))
    elif mutation == "oversize_acceptance":
        pin = case.admission["runtime_acceptance"]
        raw = case.io.raws[pin["path"]] + b" " * 16384
        case.io.raws[pin["path"]] = raw
        case.admission["runtime_acceptance"] = fake_pin(pin["path"], raw)
        case.cli_pin = q.Pin.parse(fake_pin(case.cli_pin.path, q.canonical_json(case.admission)))
        case.io.raws[case.cli_pin.path] = q.canonical_json(case.admission)
    elif mutation == "raw_drift":
        case.io.raws[case.acceptance["records"]["ffmpeg_map"]["path"]] = b"changed map"
    elif mutation == "file_alias":
        path = case.acceptance["records"]["ffprobe_map"]["path"]
        case.io.opened_identities[path] = case.acceptance["records"]["ffmpeg_map"]["path"].casefold()
    elif mutation == "consumed_root": case.io.directories.add(case.admission["evidence_root"])
    else:
        def reparse(path, *, absent=False, directory=False):
            if path == case.acceptance["records"]["inspection"]["path"]:
                raise ValueError("reparse ancestry")
            return path
        case.io.check_path = reparse
    with pytest.raises(ValueError):
        q.validate_inputs(case.cli_pin, io=case.io, clock=case.clock)
    assert case.io.created == [] and case.native_attempts == []


def test_compact_admission_guard_exact_projection_and_immutable_operands(compact_case, q):
    case = compact_case
    inputs = q.validate_inputs(case.cli_pin, io=case.io, clock=case.clock)
    guard = json.loads(q.guard_bytes(inputs, case.cli_pin))
    assert set(guard) == set("schema admission runtime_acceptance publication pair run_id evidence_parent evidence_root build_id limits".split())
    assert guard["schema"] == "clipperx-candidate-components-guard-v2"
    assert guard["admission"] == case.cli_pin.record()
    assert guard["runtime_acceptance"] == case.admission["runtime_acceptance"]
    assert guard["publication"] == case.acceptance["publication"]
    assert guard["pair"] == case.acceptance["pair"]
    assert guard["run_id"] == case.admission["run_id"]
    assert guard["evidence_parent"] == case.admission["evidence_parent"]
    assert guard["evidence_root"] == case.admission["evidence_root"]
    assert guard["build_id"] == case.acceptance["build_id"]
    assert guard["limits"] == case.admission["limits"]
    with pytest.raises((AttributeError, TypeError)):
        inputs.run_id = "other"
    assert type(inputs.records) is tuple and type(inputs.notices) is tuple
    with pytest.raises(ValueError):
        q.guard_bytes(inputs, q.Pin(case.cli_pin.path, "0" * 64, case.cli_pin.bytes))
    campaign = q.admit(case.cli_pin, io=case.io, clock=case.clock)
    assert type(campaign) is q.ComponentCampaign and campaign.inputs == inputs
    assert case.io.created == [] and case.native_attempts == []


def test_compact_admission_rejects_oversized_guard_before_namespace(compact_case, q):
    case = compact_case
    case.admission["evidence_parent"] = q.WORKSPACE + "/.superpowers/sdd/" + "😀" * 900
    case.admission["evidence_root"] = case.admission["evidence_parent"] + "/sample"
    case.io.directories.add(case.admission["evidence_parent"])
    case.refresh()
    with pytest.raises(ValueError, match="guard"):
        q.validate_inputs(case.cli_pin, io=case.io, clock=case.clock)
    assert case.admission["evidence_root"] not in case.io.directories
    assert case.io.created == [] and case.native_attempts == []


def test_compact_admission_guard_serialization_failure_leaves_namespace_absent(compact_case, q, monkeypatch):
    case = compact_case
    original = q.canonical_json
    def fail_guard(value):
        if type(value) is dict and value.get("schema") == "clipperx-candidate-components-guard-v2":
            raise TypeError("synthetic JSON encoder failure")
        return original(value)
    monkeypatch.setattr(q, "canonical_json", fail_guard)
    with pytest.raises(ValueError, match="guard serialization"):
        q.validate_inputs(case.cli_pin, io=case.io, clock=case.clock)
    assert case.admission["evidence_root"] not in case.io.directories
    assert case.io.created == [] and case.native_attempts == []


def test_pins_reject_bool_size(q):
    pin = dict(path="O:/fake/receipt.json", sha256="a" * 64, bytes=2)
    assert q.Pin.parse(pin).bytes == 2
    with pytest.raises(ValueError):
        q.Pin.parse({**pin, "bytes": True})


@pytest.mark.parametrize("path", ["relative.json", "O:/fake/../r", "O:/fake/r:ads", "//host/share/r", "O:/fake/r.", "O:/fake/R~1"])
def test_noncanonical_lexical_paths_refused(q, path):
    with pytest.raises(ValueError):
        q.canonical_path(path)


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b'{"x":true} trailing', b'\xff'])
def test_strict_json(q, raw):
    with pytest.raises(ValueError):
        q.parse_document(raw)


def test_all_24_operation_caps_and_total(q):
    operations = q.operation_limits()
    assert len(operations) == 24
    assert sum(row["timeout_seconds"] for row in operations) == 1320
    assert operations[16] == dict(index=17, kind="spatial", timeout_seconds=90, stdout_bytes=13824000, artifact_bytes=13824000)
    assert q.limits()["campaign_seconds"] == 1200
    assert (q.limits()["work_seconds"], q.limits()["cleanup_seconds"], q.limits()["settlement_seconds"]) == (1185, 5, 10)


def test_exact_fixture_bytes(q):
    f = q.fixture_bytes()
    assert set(f) == {"source.bgr", "source.f32", "tone.f32", "alpha.rgba", "shots.list", "text.ass"}
    assert len(f["source.bgr"]) == 184320
    assert f["source.bgr"][:6] == bytes((16, 32, 64, 17, 32, 64))
    assert f["source.bgr"][-3:] == bytes((47, 63, 123))
    assert f["source.f32"][:16] == bytes.fromhex("00000000000000800100000001000080")
    assert len(f["source.f32"]) == len(f["tone.f32"]) == 384000
    assert len(f["alpha.rgba"]) == 1024
    assert f["alpha.rgba"][3:48:4] == bytes([0] * 5 + [128] * 6 + [255])
    assert f["shots.list"] == b"raster.rgba\n"
    assert "Ágil مرحبا".encode() in f["text.ass"]


def test_float_bit_identity_and_s16_clipping_oracles(q):
    fixtures = q.fixture_bytes()
    q.assert_float_identity(fixtures["source.f32"], fixtures["source.f32"])
    changed = bytearray(fixtures["source.f32"])
    changed[7] = 0  # -0.0 silently coerced to +0.0
    with pytest.raises(ValueError):
        q.assert_float_identity(bytes(changed), fixtures["source.f32"])
    words = [0, 0, 0, 0, 8192, -8192, 32767, -32768]
    import struct
    s16 = struct.pack("<8h", *words) * 12000
    q.assert_s16(s16, fixtures["source.f32"])
    with pytest.raises(ValueError):
        q.assert_s16(b"\0" * 192000, fixtures["source.f32"])


def test_mix_oracle_checks_delay_amplitude_and_finiteness(q):
    import struct
    good = b"".join(struct.pack("<ff", (a := (.125 if j < 24000 else .1875) * (1 if j % 96 < 48 else -1)), -a) for j in range(48000))
    q.assert_mix(good)
    with pytest.raises(ValueError):
        q.assert_mix(b"\0" * 384000)
    with pytest.raises(ValueError):
        q.assert_mix(struct.pack("<f", float("nan")) + good[4:])


def test_vulkan_constant_raster_fails(q):
    with pytest.raises(ValueError):
        q.assert_vulkan(bytes((32, 48, 16, 255)) * (48 * 48 * 4), b"device Vulkan adapter")


@pytest.mark.parametrize("log", [
    b"device extensions enabled\n", b"adapter discovery: FakeVulkan GPU\n",
    b"Error selecting device FakeVulkan GPU\n",
    b"[Vulkan @ 0] Using device: FakeVulkan GPU\n",
    b"[Vulkan @ 0] Device 0 selected: FakeVulkan GPU (discrete) (0x1234)\n",
    b"[Vulkan @ 0] Device 0 selected: Other GPU (discrete) (0x1234)\n[Vulkan @ 0] Using device: FakeVulkan GPU\n",
])
def test_vulkan_valid_ramp_rejects_generic_or_inconsistent_device_evidence(q, log):
    raw = b"".join(bytes((64 + n, 32 + y, 16 + x, 255))
        for n in range(4) for y in range(48) for x in range(48))
    with pytest.raises(ValueError, match="selected-device"):
        q.assert_vulkan(raw, log)


@pytest.mark.parametrize("newline", [b"\n", b"\r\n"])
def test_vulkan_valid_ramp_returns_original_selected_device_identity(q, newline):
    raw = b"".join(bytes((64 + n, 32 + y, 16 + x, 255))
        for n in range(4) for y in range(48) for x in range(48))
    log = b"[AVHWDeviceContext @ 0] Device 0 selected: FakeVulkan GPU (discrete) (0x1234)\n[AVHWDeviceContext @ 0] Using device: FakeVulkan GPU\n"
    log = log.replace(b"\n", newline)
    assert q.assert_vulkan(raw, log) == {"selected_device": "FakeVulkan GPU"}


def test_png_alpha_independent_pillow_decode(q):
    from PIL import Image
    raw = q.fixture_bytes()["alpha.rgba"]
    with io.BytesIO() as output:
        Image.frombytes("RGBA", (16, 16), raw).save(output, format="PNG")
        q.assert_png(output.getvalue(), raw)
    with pytest.raises(ValueError):
        q.assert_png(b"not png", raw)


def test_mp4_box_order_is_independent(q):
    import struct
    box = lambda tag: struct.pack(">I4s", 8, tag)
    assert q.mp4_faststart(box(b"ftyp") + box(b"moov") + box(b"mdat"))
    assert not q.mp4_faststart(box(b"mdat") + box(b"moov"))
    with pytest.raises(ValueError):
        q.mp4_faststart(b"\0" * 7)


def test_operation_argv_exact_pair_profiles_and_no_exploration(q):
    pair = q.CandidatePair(q.Pin.parse(dict(path="O:/fake/publish/bin/ffmpeg.exe", bytes=1, sha256="a" * 64), 268435456),
        q.Pin.parse(dict(path="O:/fake/publish/bin/ffprobe.exe", bytes=1, sha256="b" * 64), 268435456))
    origin = q.candidate_origin("O:/fake/evidence/source.mov", 100, 1, pair=pair)
    argv = q.operation_argv(pair, origin=origin, brand="brand.png", impact="impact.ttf")
    assert len(argv) == 24
    assert argv[0] == (pair.ffmpeg.path, "-version")
    assert argv[1] == (pair.ffprobe.path, "-version")
    assert argv[8][-1] == "source.mov"
    assert "-crf" in argv[18] and "medium" in argv[18]
    assert "-cq" in argv[21] and "p5" in argv[21]
    assert argv[23][-3:] == ("-f", "null", "-")
    assert all(a[0] in (pair.ffmpeg.path, pair.ffprobe.path) for a in argv)


def test_operation_argv_only_row18_uses_adjudicated_verbose_logging(q):
    pair = q.CandidatePair(q.Pin.parse(dict(path="O:/fake/publish/bin/ffmpeg.exe", bytes=1, sha256="a" * 64), 268435456),
        q.Pin.parse(dict(path="O:/fake/publish/bin/ffprobe.exe", bytes=1, sha256="b" * 64), 268435456))
    origin = q.candidate_origin("O:/fake/evidence/source.mov", 100, 1, pair=pair)
    plan = q.operation_argv(pair, origin=origin, brand="brand.png", impact="impact.ttf")
    assert plan[17][4:6] == ("-v", "verbose")
    original = list(plan)
    original[17] = plan[17][:5] + ("info",) + plan[17][6:]
    # The pre-amendment complete plan hash covers all 23 other rows and every
    # other row18 token, without copying a second argv table into the tests.
    assert q.digest(q.canonical_json(original)) == "11cf3c448ef337554ec66f86acb5fa11a06ff87b27073a403ea5e09bd2cc9b97"


def producer_operation_argv(module, root):
    pair = module.CandidatePair(*(module.Pin.parse(dict(path=root + "/publish/bin/" + name + ".exe",
        bytes=1, sha256=sha * 64), 268435456) for name, sha in (("ffmpeg", "a"), ("ffprobe", "b"))))
    origin = module.candidate_origin("O:/fake/evidence/source.mov", 100, 1, pair=pair)
    return module.operation_argv(pair, origin=origin, brand="brand.png", impact="impact.ttf")


def test_hbft_mapping_preserves_complete_task122_argv_matrix_and_caps(q):
    baseline_path = SOURCE.parents[1] / ".superpowers/sdd/snapshots/task-125/consumer-baseline/qualify_production_runtime.py"
    raw = baseline_path.read_bytes()
    assert len(raw) == 88020
    assert hashlib.sha256(raw).hexdigest().upper() == "7F0F8E76F5DA864EB8B87700DF6433A73243FC73A67310E6E989153685620075"
    # Import only the accepted inert qualifier snapshot, never the producer.
    spec = importlib.util.spec_from_file_location("task125_task122_qualifier_baseline", baseline_path)
    baseline = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = baseline
    spec.loader.exec_module(baseline)
    expected = producer_operation_argv(baseline, baseline.O8)
    actual = producer_operation_argv(q, q.O9)
    identities = {q.O9 + "/publish/bin/" + name + ".exe": baseline.O8 + "/publish/bin/" + name + ".exe"
        for name in ("ffmpeg", "ffprobe")}
    normalized = tuple(tuple(identities.get(token, token) for token in row) for row in actual)
    assert len(actual) == len(expected) == 24
    assert normalized == expected  # Every token in row18 and all other23 rows.
    assert actual[17][4:6] == expected[17][4:6] == ("-v", "verbose")
    assert q.limits() == baseline.limits()
    for name in ("PUBLICATION_FIELDS", "NOTICE_NAMES", "SOURCE_CONTRACT", "RECORD_CAPS", "SOURCE_PATHS",
            "ACCEPTED_DESIGN_PINS", "HELPER_SYMBOLS", "ASSET_PATHS"):
        assert getattr(q, name) == getattr(baseline, name)
    assert q.LINEAGE_PATHS[:4] == baseline.LINEAGE_PATHS
    assert {path: q.LINEAGE_PINS[path] for path in baseline.LINEAGE_PINS} == baseline.LINEAGE_PINS


def test_raw_metadata_tags_must_be_complete_and_preserve_offset(q):
    from fractions import Fraction
    lines = "\n".join(f"[showinfo@cxv @ 0] n: {i} pts: {75+i} fmt: bgr24 s: 32x32 checksum: 00000000" for i in range(60)).encode()
    q.assert_video_tags(lines, Fraction(1, 60))
    with pytest.raises(ValueError):
        q.assert_video_tags(lines.replace(b"pts: 75", b"pts: 0"), Fraction(1, 60))
    with pytest.raises(ValueError):
        q.assert_video_tags(lines.split(b"\n", 1)[1], Fraction(1, 60))


def fake_pin(path="O:/fake/a.json", raw=b"{}"):
    return dict(path=path, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


@pytest.mark.parametrize("argv", [["--native"], ["--confirm-host"], ["--admission", "missing"], ["--native", "--confirm-host", "--admission", "missing", "--admission-sha256", "a" * 64]])
def test_partial_activation_refused_before_files(q, argv, monkeypatch):
    def forbidden(*a, **k): raise AssertionError("partial activation attempted setup")
    monkeypatch.setattr(q, "admit", forbidden)
    assert q.main(argv) == 2


def sequence(q, c, payload=b"ok", log=b"tag\n", factory=None, plan=None):
    from clipperx._render_services import OwnedRenderTools
    calls = []
    def fake(argv):
        calls.append(argv)
        if factory is not None: return factory(argv)
        process = ScriptProcess(payload, log)
        c.h.children.append(process)
        return process
    tools = OwnedRenderTools(c.h.job, c.files, clock=c.h.job._clock)
    plan = (("fake", "-version"),) + (("fake",),) * 23 if plan is None else plan
    seq = q.OwnedAttemptSequence(c.h.job, c.files, tools, started_ns=10, plan=plan, guard=lambda *a: None)
    c.h.job._processes = seq.process_factory(fake)
    return seq, calls


def test_original_owner_serial_sequence_charges_factory_only(q, memory, closed_boundaries):
    seq, calls = sequence(q, memory)
    result = seq.run(1, ("fake", "-version"))
    assert calls == [("fake", "-version")]
    assert seq.attempts == 1 and result.retired and result.natural_eof
    assert result.generation == memory.h.job.snapshot().current.diagnostics[-1].generation
    assert not memory.files.snapshot().consumers
    with pytest.raises(ValueError): seq.run(3, ("fake", "-filters"))
    assert seq.attempts == 1


@pytest.mark.parametrize("field", ["exit_code", "natural_eof", "control_id", "generation",
    "reused_control", "reused_generation"])
def test_original_sequence_rejects_corrupt_child_association(q, memory, closed_boundaries, monkeypatch, field):
    from dataclasses import replace
    seq, calls = sequence(q, memory)
    original_run = seq.tools.run
    def corrupt(*args, **kwargs):
        result = original_run(*args, **kwargs)
        if field == "reused_control":
            seq.seen_controls.add(result.control_id)
            return result
        if field == "reused_generation":
            seq.seen_generations.add(result.generation)
            return result
        return replace(result, **{field: False if field == "natural_eof" else
            1 if field == "exit_code" else getattr(result, field) + 1})
    monkeypatch.setattr(seq.tools, "run", corrupt)
    with pytest.raises(ValueError, match="association|EOF|generation|control"):
        seq.run(1, ("fake", "-version"))
    assert seq.attempts == 1 and len(calls) == 1 and seq.failed


def test_original_sequence_refuses_swapped_argv_and_sidecar_reservation_before_factory(q, memory, closed_boundaries):
    seq, calls = sequence(q, memory)
    with pytest.raises(ValueError, match="exact next"):
        seq.run(1, ("fake", "-filters"))
    assert not calls and seq.attempts == 0
    seq.sidecar_reservation = 33554432
    with pytest.raises(ValueError, match="sidecar reservation"):
        seq.run(1, ("fake", "-version"))
    assert not calls and seq.attempts == 0


@pytest.mark.parametrize("index", range(1, 25))
def test_hbft_mapping_refuses_altered_operation_before_factory(q, memory, closed_boundaries, index):
    plan = producer_operation_argv(q, q.O9)
    seq, calls = sequence(q, memory, plan=plan)
    # Synthetic prior-count state reaches each row without executing predecessors.
    seq.completed = seq.attempts = index - 1
    altered = plan[index - 1] + ("--altered-operation",)
    with pytest.raises(ValueError, match="exact next sequential operation"):
        seq.run(index, altered)
    assert not calls and seq.attempts == seq.completed == index - 1
    assert seq.sidecar_reservation == 0 and seq.active is None


def test_failed_spawn_is_consumed_once(q, memory, closed_boundaries):
    def failed(argv): raise OSError("synthetic spawn failure")
    seq, calls = sequence(q, memory, factory=failed)
    with pytest.raises(ValueError): seq.run(1, ("fake", "-version"))
    assert seq.attempts == 1 and len(calls) == 1
    with pytest.raises(ValueError): seq.run(1, ("fake", "-version"))
    assert seq.attempts == 1


@pytest.mark.parametrize("payload,log", [(b"x" * 1000001, b""), (b"ok", b"x" * 16385), (b"ok", b"a\n" * 32769)], ids=["stdout-overflow", "stderr-line", "stderr-total"])
def test_original_tools_boundaries_stop_sequence(q, memory, closed_boundaries, payload, log):
    seq, calls = sequence(q, memory, payload=payload, log=log)
    with pytest.raises(ValueError): seq.run(1, ("fake", "-version"))
    assert seq.attempts == 1 and len(calls) == 1


def test_attempt25_and_factory_escape_are_blocked(q, memory, closed_boundaries):
    seq, calls = sequence(q, memory)
    with pytest.raises(ValueError): memory.h.job._processes(("undeclared",))
    seq.attempts = 24
    with pytest.raises(ValueError): seq.run(25, ("fake",))
    assert not calls


def test_work_deadline_preserves_cleanup_settlement_reserves(q, memory, closed_boundaries):
    seq, calls = sequence(q, memory)
    memory.h.now = 10 + 1185 * 10**9
    with pytest.raises(ValueError): seq.run(1, ("fake",))
    assert seq.attempts == 0 and not calls


def test_raster_oracle_rejects_inactive_text_or_missing_active_glyphs(q):
    background = bytes((16, 32, 48, 255)) * (320 * 180 * 60)
    with pytest.raises(ValueError):
        q.assert_raster(background, bytes((255, 255, 255, 255)) * 16, (4, 4))


def test_audio_tags_reject_unpaired_missing_blocks(q):
    from fractions import Fraction
    log = b"[ashowinfo@cxain @ 0] n: 0 pts: 60000 rate: 48000 channels: 2 nb_samples: 48000 fmt: flt checksum: 00000000\n"
    with pytest.raises(ValueError): q.assert_audio_tags(log, floating=True)


def test_pin_roundtrip_preserves_producer_recorded_path_and_hash_case(q):
    record = dict(path="O:\\fake\\publisher.json", bytes=2, sha256="A" * 64)
    assert q.Pin.parse(record).record() == record


def source_probe_payload():
    return dict(streams=[dict(index=0, codec_type="video", codec_name="qtrle", width=32, height=32,
            time_base="1/60", start_pts=75, duration_ts=60, avg_frame_rate="60/1", nb_read_frames="60"),
        dict(index=1, codec_type="audio", codec_name="pcm_f32le", sample_rate="48000", channels=2,
            sample_fmt="flt", time_base="1/48000", start_pts=60000, duration_ts=48000)],
        format=dict(start_time="1.25", duration="2.25"),
        frames=[dict(media_type="video", stream_index=0, pts=75+i,
                best_effort_timestamp_time=str((75+i)/60)) for i in range(60)] +
            [dict(media_type="audio", stream_index=1, pts=60000,
                best_effort_timestamp_time="1.25", nb_samples=48000)])


def test_source_probe_observes_raw_rational_clocks_without_resetting(q):
    pair = q.CandidatePair(q.Pin.parse(fake_pin("O:/fake/bin/ffmpeg.exe", b"F")),
        q.Pin.parse(fake_pin("O:/fake/bin/ffprobe.exe", b"P")))
    raw = q.canonical_json(source_probe_payload())
    origin, observations = q.parse_source_probe(raw, path="O:/fake/source.mov", size=100, mtime_ns=1, pair=pair)
    assert str(origin.streams[0].start) == "5/4"
    assert observations["container_duration"] == "9/4"
    assert observations["audio_samples"] == 48000
    changed = source_probe_payload()
    changed["frames"][0]["pts"] = 0
    with pytest.raises(ValueError):
        q.parse_source_probe(q.canonical_json(changed), path="O:/fake/source.mov", size=100, mtime_ns=1, pair=pair)
    changed = source_probe_payload()
    changed["frames"][-1]["stream_index"] = 0
    with pytest.raises(ValueError):
        q.parse_source_probe(q.canonical_json(changed), path="O:/fake/source.mov", size=100, mtime_ns=1, pair=pair)


def test_delivery_probe_partitions_observed_media_before_original_validator(q):
    import struct
    box = lambda tag: struct.pack(">I4s", 8, tag)
    mp4 = box(b"ftyp") + box(b"moov") + box(b"mdat")
    payload = dict(streams=[dict(index=0, codec_type="video", codec_name="h264", width=1080, height=1920,
            pix_fmt="yuv420p", avg_frame_rate="60/1", nb_read_frames="60"),
        dict(index=1, codec_type="audio", codec_name="aac", sample_rate="48000", channels=2)],
        format=dict(start_time="0", duration="1"), frames=[])
    # Deliberately audio first: show_entries order must not define the partition.
    payload["frames"] = [dict(media_type="audio", stream_index=1, best_effort_timestamp_time="0", nb_samples=48000)] + [
        dict(media_type="video", stream_index=0, best_effort_timestamp_time=f"{i/60:.6f}") for i in range(60)]
    observation = q.assert_delivery(q.canonical_json(payload), mp4)
    assert observation["audio_sample_delta"] == 0
    payload["frames"][0]["best_effort_timestamp_time"] = "0.1"
    with pytest.raises(ValueError): q.assert_delivery(q.canonical_json(payload), mp4)


def test_terminal_own0_cannot_write_through_retired_original_file_owner(memory, closed_boundaries):
    """Diagnostic of the binding settlement/interface incompatibility."""
    from clipperx.tests.test_source_host_ownership import finish_job
    path = ROOT / "task112-terminal.json"
    memory.files.register(path, role="sidecar", owned=True, max_bytes=2097152)
    assert not memory.h.job.snapshot().own0
    assert finish_job(memory.h).own0
    before = list(memory.fs.events)
    with pytest.raises(ValueError, match="unadmitted source driver"):
        with memory.files.open_file(path, "xb"):
            raise AssertionError("retired owner admitted a terminal write")
    assert memory.fs.events == before


@pytest.mark.parametrize("mode", ["complete", "short", "exception", "overrun"])
def test_detached_terminal_only_accepts_completed_write_after_observed_retirement(q, memory, closed_boundaries, mode):
    from clipperx.tests.test_source_host_ownership import finish_job
    status = finish_job(memory.h)
    writes = []
    now = [100]
    class TerminalIO:
        def write_exclusive(self, path, raw):
            writes.append((path, raw))
            if mode == "exception": raise OSError("synthetic terminal write")
            if mode == "overrun": now[0] = 1200 * 10**9
            return len(raw) - 1 if mode == "short" else len(raw)
    result = q.publish_terminal("O:/fake/evidence/terminal.json", dict(operations=24, attempts=24),
        owner=status, files=memory.files.snapshot(), started_ns=0, clock=lambda: now[0], io=TerminalIO(),
        reserved_bytes=2097152, charged_bytes=100, reserved_paths=1, used_paths=10)
    assert len(writes) == 1
    assert result["candidate_component_passed"] is (mode == "complete")
    assert result["runtime_integrated"] is False and result["production_accepted"] is False
    if mode == "complete":
        assert json.loads(writes[0][1])["provisional"] is True
    else:
        assert result["status"] in ("STOPPED", "UNCERTAIN")


