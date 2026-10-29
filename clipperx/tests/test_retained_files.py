"""Finite file custody with every external boundary replaced before entry."""

import io
import builtins
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from clipperx.tests.test_source_host_ownership import (
    admitted, closed_boundaries as closed_boundaries, finish_job, forbidden,
)

ROOT = Path("O:/HOST/scratch")


@pytest.mark.parametrize("operation", ["close", "unlink", "rmdir"])
def test_cleanup_action_cannot_receive_driver_cancel(memory: Any, operation: str) -> None:
    c, path = memory, ROOT / "cleanup"
    c.files.register(path, role="directory" if operation == "rmdir" else "raw", owned=True, max_bytes=8)
    if operation == "rmdir":
        c.files.path_action("mkdir", path)
    else:
        stream = c.files.open_file(path, "xb")
        stream.__enter__()
        stream.write(memoryview(b"abcd"))
        if operation != "close":
            stream.close()
    c.h.job.fail("validation", "driver")
    before = len(c.h.kernel.CancelSynchronousIo.calls)
    def race(op: str, _: Path) -> None:
        assert not c.h.job._lock._is_owned()
        if op == operation:
            c.h.current = None
            try:
                c.h.job.service_cleanup()
            finally:
                c.h.current = c.h.driver
            if len(c.h.kernel.CancelSynchronousIo.calls) != before:
                error = OSError(22, "synthetic operation aborted")
                setattr(error, "winerror", 995)
                raise error
    c.fs.hook = race
    assert c.files.retire().files_clear
    assert len(c.h.kernel.CancelSynchronousIo.calls) == before
    assert finish_job(c.h).own0


def test_cleanup_stat_keeps_its_existing_unshielded_path(memory: Any) -> None:
    c, path = memory, ROOT/"consumer.raw"
    c.files.register(path, role="raw", owned=True, max_bytes=8)
    with c.files.open_file(path, "xb") as stream:
        stream.write(memoryview(b"abcd"))
    token = c.files.acquire_consumer((path,))
    c.h.job.fail("validation", "driver")
    def stat(operation: str, path: Path) -> None:
        if operation == "stat":
            assert c.h.job._cleanup_token is None
    c.fs.hook = stat
    c.files.release_consumer(token, retired=True)
    assert c.files.retire().files_clear
    assert finish_job(c.h).own0


def test_successful_cleanup_commits_before_expired_postcheck(memory: Any) -> None:
    c, path = memory, ROOT/"late.raw"
    c.files.register(path, role="raw", owned=True, max_bytes=8)
    with c.files.open_file(path, "xb"):
        pass
    failure = c.h.job.fail("validation", "driver")
    def late(operation: str, _: Path) -> None:
        if operation == "unlink":
            c.h.now = failure.cleanup_deadline_ns
    c.fs.hook = late
    assert c.files.retire().files_clear
    assert path not in c.fs.data and c.h.job._cleanup_token is None
    assert finish_job(c.h).own0


class MemoryFS:
    def __init__(self) -> None:
        self.data: dict[Path, bytes] = {}
        self.dirs = {ROOT, ROOT.parent, Path(ROOT.anchor)}
        self.events: list[tuple[str, Path]] = []
        self.hook: Any = None
        self.write_chunk = 65536

    def event(self, op: str, path: Path) -> None:
        self.events.append((op, path))
        if self.hook:
            self.hook(op, path)

    def stamp(self, path: Path) -> Any:
        if path not in self.data and path not in self.dirs:
            raise FileNotFoundError(path)
        return SimpleNamespace(st_dev=1, st_ino=abs(hash(path)) % 100000 + 1,
            st_size=len(self.data.get(path, b"")), st_mtime_ns=100, st_ctime_ns=100,
            st_mode=0o40755 if path in self.dirs else 0o100644, st_file_attributes=0, st_nlink=1)

    def open(self, path: Path, mode: str) -> Any:
        if mode == "xb":
            if path in self.data:
                raise FileExistsError(path)
            self.data[path] = b""
        elif path not in self.data:
            raise FileNotFoundError(path)
        fs = self

        class File(io.BytesIO):
            path: Path

            def read(self, size: int | None = -1) -> bytes:
                fs.event("read", path)
                return super().read(size)

            def write(self, raw: Any) -> int:
                fs.event("write", path)
                count = super().write(raw[:fs.write_chunk])
                fs.data[path] = self.getvalue()
                return count

            def close(self) -> None:
                fs.event("close", path)
                super().close()

            def flush(self) -> None:
                fs.event("flush", path)

            def fileno(self) -> int:
                return 100

        file = File(self.data[path])
        file.path = path
        self.event("open", path)
        return file

    def stat(self, file: Any) -> Any:
        self.event("fstat", file.path)
        return self.stamp(file.path)

    def action(self, operation: str, path: Path, destination: Path | None) -> Any:
        self.event(operation, path)
        if operation in ("stat", "lstat"):
            return self.stamp(path)
        if operation == "resolve":
            return path
        if operation == "mkdir":
            if path in self.dirs:
                raise FileExistsError(path)
            self.dirs.add(path)
        elif operation == "replace":
            assert destination is not None
            self.data[destination] = self.data.pop(path)
        elif operation == "unlink":
            del self.data[path]
        elif operation == "rmdir":
            self.dirs.remove(path)
        else:
            raise AssertionError(operation)


@pytest.fixture
def memory(monkeypatch: pytest.MonkeyPatch, closed_boundaries: None) -> Any:
    h, fs = admitted(), MemoryFS()
    def blocked(path: Any) -> bool:
        return isinstance(path, (str, bytes, os.PathLike)) and os.fsdecode(path).replace("\\", "/").casefold().startswith(
            ("o:/host", "o:/escritorio/clipperx/data/"))
    for name in ("open", "stat", "lstat", "resolve", "mkdir", "unlink", "rmdir", "replace",
                 "read_bytes", "write_bytes", "is_file", "is_dir", "exists"):
        original = getattr(Path, name)
        def guarded(path: Path, *args: Any, _original: Any = original, **kwargs: Any) -> Any:
            if blocked(path):
                forbidden()
            return _original(path, *args, **kwargs)
        monkeypatch.setattr(Path, name, guarded)
    for module, name in ((builtins, "open"), (io, "open"), (os, "open"), (os, "stat"),
                         (os, "lstat"), (os, "listdir"), (os, "scandir")):
        original_call = getattr(module, name)
        def guarded_call(path: Any, *args: Any, _original: Any = original_call, **kwargs: Any) -> Any:
            if blocked(path):
                forbidden()
            return _original(path, *args, **kwargs)
        monkeypatch.setattr(module, name, guarded_call)
    monkeypatch.setattr(os, "fstat", forbidden)
    monkeypatch.setattr(os, "fsync", forbidden)
    monkeypatch.setattr(shutil, "disk_usage", forbidden)
    from clipperx._retained_files import RetainedFiles
    files = RetainedFiles(h.job, root=ROOT, clock=h.job._clock,
        _open=fs.open, _stat=fs.stat, _path_action=fs.action, _fsync=lambda _: None)
    return SimpleNamespace(h=h, fs=fs, files=files)


def test_inert_registration_and_unknown_path(memory: Any) -> None:
    c = memory
    assert c.files.snapshot().files_clear and c.fs.events == []
    c.files.register(ROOT / "a.raw", role="raw", owned=True, max_bytes=8)
    assert c.fs.events == []
    with pytest.raises(ValueError):
        c.files.open_binary(ROOT / "unknown").__enter__()
    with pytest.raises(ValueError):
        c.files.register(ROOT.parent / "unsafe", role="raw", owned=True, max_bytes=1)
    with pytest.raises(ValueError):
        c.files.register(ROOT, role="directory", owned=True, max_bytes=0)
    assert finish_job(c.h).own0


def test_known_short_writes_claims_and_exact_cleanup(memory: Any) -> None:
    c, path = memory, ROOT / "a.raw"
    c.files.register(path, role="raw", owned=True, max_bytes=8)
    c.fs.write_chunk = 2
    with c.files.open_file(path, "xb") as file:
        assert file.write(memoryview(b"abcd")) == 2
        assert file.write(memoryview(b"cd")) == 2
        assert file.fstat().st_size == 4
        file.flush()
        file.fsync()
    token = c.files.acquire_consumer((path,))
    with pytest.raises(ValueError):
        c.files.path_action("delete", path)
    c.files.release_consumer(token, retired=False)
    assert not c.files.retire().files_clear
    c.files.release_consumer(token, retired=True)
    assert c.files.retire().files_clear
    assert path not in c.fs.data
    assert finish_job(c.h).own0


def test_uncertain_close_never_retried(memory: Any) -> None:
    c, path = memory, ROOT / "a.raw"
    c.files.register(path, role="raw", owned=True, max_bytes=8)
    def hook(op: str, _: Path) -> None:
        if op == "close":
            raise OSError("unknown close")
    c.fs.hook = hook
    with pytest.raises(OSError):
        with c.files.open_file(path, "xb"):
            pass
    c.files.retire()
    assert len([op for op, _ in c.fs.events if op == "close"]) == 1
    assert c.files.snapshot().rows[0].uncertain
    assert not c.files.snapshot().files_clear
    assert finish_job(c.h).own0


def test_late_open_object_retained_before_deadline_check(memory: Any) -> None:
    c, path = memory, ROOT / "late.raw"
    c.files.register(path, role="raw", owned=True, max_bytes=8)
    def hook(op: str, _: Path) -> None:
        if op == "open":
            c.h.now = c.h.job._deadline
    c.fs.hook = hook
    with pytest.raises(ValueError):
        c.files.open_file(path, "xb").__enter__()
    assert not c.files.snapshot().files_clear
    assert c.files.retire().files_clear
    assert finish_job(c.h).own0


def test_external_producer_absent_before_launch_and_retired_observation(memory: Any) -> None:
    c, path = memory, ROOT / "candidate.mp4"
    c.files.register(path, role="candidate", owned=True, max_bytes=10)
    token = c.files.acquire_consumer((), outputs=(path,))
    c.fs.data[path] = b"partial"
    assert not c.files.retire().files_clear
    c.files.release_consumer(token, retired=False)
    assert path in c.fs.data
    c.files.release_consumer(token, retired=True)
    assert c.files.retire().files_clear
    assert finish_job(c.h).own0


@pytest.mark.parametrize("short", [False, True])
def test_actual_font_stage_checks_identity_using_retained_io(memory: Any, short: bool) -> None:
    import hashlib
    from clipperx import font_staging as staging
    from clipperx.render_assets import RenderAssetRegistry
    c = memory
    sources = Path("O:/HOST/originals")
    c.fs.dirs.add(sources)
    raw = {"headline_impact": b"impact-fixture", "subtitle_bold": b"bold-fixture"}
    paths = {key: sources / f"{key}.ttf" for key in raw}
    for key, path in paths.items():
        c.fs.data[path] = raw[key]
        c.files.register(path, role="original", owned=False, max_bytes=16 * 1024**2)
    registry = RenderAssetRegistry({}, paths, {})
    before = dict(registry.fonts)
    if short:
        c.fs.write_chunk = 2
    def stage() -> Any:
        return staging._stage_fonts(registry, stage_dir=ROOT / "fonts",
            expected_sha256={key: hashlib.sha256(data).hexdigest() for key, data in raw.items()},
            identity_authority="test-fixture", _files=c.files)
    if short:
        with pytest.raises(OSError, match="Incomplete font write"):
            stage()
    else:
        result = stage()
        staging.verify_font_stage(result, _files=c.files)
        assert all(font.measurement_path == font.staged_path for font in result.fonts)
        assert result.identity_authority == "test-fixture"
    assert dict(registry.fonts) == before
    assert c.files.retire().files_clear
    assert all(c.fs.data[path] == raw[key] for key, path in paths.items())
    assert finish_job(c.h).own0


def test_atomic_content_asset_is_exclusive_and_registered(memory: Any) -> None:
    from clipperx.render_assets import publish_render_bytes
    c = memory
    result = publish_render_bytes(b"ass text", output_dir=ROOT / "assets", suffix=".ass", _files=c.files)
    assert c.fs.data[result] == b"ass text"
    assert publish_render_bytes(b"ass text", output_dir=ROOT / "assets", suffix=".ass", _files=c.files) == result
    assert ("replace", result.with_suffix(".ass.tmp")) in c.fs.events
    assert c.files.retire().files_clear
    assert finish_job(c.h).own0


def test_font_glyph_and_measurement_load_memory_face_zero(memory: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from clipperx import overlay_ass as overlay
    from clipperx.tests.test_visual_compiler import pure_plan
    from clipperx.render_assets import RenderAssetRegistry
    from clipperx.visual_contracts import OverlayPlan, NormalizedRect
    c, path = memory, ROOT / "font.ttf"
    c.files.register(path, role="original", owned=False, max_bytes=16 * 1024**2)
    c.fs.data[path] = b"fake-font-bytes"
    seen = []
    class Face:
        def __enter__(self) -> Any:
            return self
        def __exit__(self, *args: Any) -> None:
            pass
        def getBestCmap(self) -> dict[int, str]:
            return {ord(char): char for char in "Hello "}
        def getGlyphID(self, glyph: str) -> int:
            return 1
    class Font:
        def getname(self) -> tuple[str, str]:
            return "Fixture", "Bold"
        def getbbox(self, text: str, **kwargs: Any) -> tuple[int, int, int, int]:
            return 0, 0, 20, 20
        def getlength(self, text: str) -> int:
            return 20
        def getmetrics(self) -> tuple[int, int]:
            return 16, 4
    def face(file: Any, **kwargs: Any) -> Face:
        assert isinstance(file, io.BytesIO) and file.getvalue() == b"fake-font-bytes"
        assert kwargs == {"fontNumber": 0, "lazy": True}
        seen.append(file)
        return Face()
    def font(file: Any, size: int, **kwargs: Any) -> Font:
        assert isinstance(file, io.BytesIO) and file.getvalue() == b"fake-font-bytes"
        assert kwargs == {"index": 0}
        seen.append(file)
        return Font()
    monkeypatch.setattr(overlay, "import_module", lambda _: SimpleNamespace(TTFont=face))
    monkeypatch.setattr(overlay.ImageFont, "truetype", font)
    plan = pure_plan()
    plan.shots[0].title = OverlayPlan(text="Hello", start=0, end=.1, z_index=1,
        destination=NormalizedRect(x=.1, y=.1, width=.8, height=.1))
    result = overlay.compile_overlay_ass(plan, None,
        assets=RenderAssetRegistry({}, {"headline_impact": path}, {}), _files=c.files)
    assert result.events[0].font_family == "Fixture"
    assert all(stream.closed for stream in seen)
    with pytest.raises(ValueError, match="Unsupported glyph"):
        overlay._validate_glyph_coverage(path, "Z", _files=c.files)
    assert finish_job(c.h).own0


def test_registered_disk_usage_facts(memory: Any) -> None:
    c = memory
    c.files.register(ROOT, role="checked", owned=False, max_bytes=0)
    action = c.files._path_action
    c.files._path_action = lambda op, path, dest: (100, 40, 60) if op == "disk_usage" else action(op, path, dest)
    assert c.files.path_action("disk_usage", ROOT) == (100, 40, 60)
    assert c.files.snapshot().files_clear
    assert finish_job(c.h).own0


def test_stage_deadline_reaches_owner_and_retains_late_acquisition(memory: Any) -> None:
    c, path = memory, ROOT / "late.raw"
    c.files.register(path, role="raw", owned=True, max_bytes=10)
    ids = (c.h.job._next_id, c.h.job._next_generation, c.h.job._next_observation)
    def hook(op: str, _: Path) -> None:
        if op == "open":
            c.h.now = 100
    c.fs.hook = hook
    scope = c.files.stage_deadline(100)
    with pytest.raises(ValueError):
        with scope:
            assert c.h.job._effective_deadline() == (100, "driver")
            with pytest.raises(ValueError):
                with c.files.stage_deadline(110):
                    pass
            c.files.open_file(path, "xb").__enter__()
    assert ids == (c.h.job._next_id, c.h.job._next_generation, c.h.job._next_observation)
    failure = c.h.job.snapshot().failure
    assert failure.first_ns == 100 and failure.cleanup_deadline_ns == 200
    assert c.files.retire().files_clear
    assert c.h.job.snapshot().failure.cleanup_deadline_ns == 200
    assert finish_job(c.h).own0


def test_replace_never_overwrites_unowned_existing_destination(memory: Any) -> None:
    c, source, target = memory, ROOT / "temp", ROOT / "final"
    for path in (source, target):
        c.files.register(path, role="asset", owned=True, max_bytes=10)
    with c.files.open_file(source, "xb") as file:
        file.write(memoryview(b"ours"))
    c.fs.data[target] = b"other"
    with pytest.raises(ValueError):
        c.files.path_action("replace", source, destination=target)
    assert c.fs.data[target] == b"other"
    assert c.files.retire().files_clear
    assert c.fs.data[target] == b"other"
    assert finish_job(c.h).own0


def test_uncertain_replace_retains_both_endpoints(memory: Any) -> None:
    c, source, target = memory, ROOT / "temp", ROOT / "final"
    for path in (source, target):
        c.files.register(path, role="asset", owned=True, max_bytes=10)
    with c.files.open_file(source, "xb") as file:
        file.write(memoryview(b"ours"))
    action = c.files._path_action
    def uncertain(op: str, path: Path, dest: Path | None) -> Any:
        result = action(op, path, dest)
        if op == "replace":
            raise OSError("returned state unknown")
        return result
    c.files._path_action = uncertain
    with pytest.raises(OSError):
        c.files.path_action("replace", source, destination=target)
    assert c.files._busy(target)
    assert not c.files.retire().files_clear
    assert c.fs.data[target] == b"ours"
    assert finish_job(c.h).own0


@pytest.mark.parametrize("oversize", [False, True])
def test_brand_trim_cache_and_pixel_cap_use_memory_only(memory: Any,
        monkeypatch: pytest.MonkeyPatch, oversize: bool) -> None:
    from clipperx import render_assets as assets
    c, source = memory, Path("O:/HOST/brand.png")
    c.fs.data[source] = b"original"
    c.files.register(source, role="original", owned=False, max_bytes=16 * 1024**2)
    images: list[Any] = []
    encoded_metadata: dict[str, str] = {}
    class Image:
        format, mode = "PNG", "RGBA"
        def __init__(self, trimmed: bool = False) -> None:
            self.trimmed = trimmed
            self.size = (2, 2) if trimmed else (5000, 5000) if oversize else (4, 4)
            self.width, self.height = self.size
            self.info = encoded_metadata
            self.loaded = False
        def __enter__(self) -> Any:
            return self
        def __exit__(self, *args: Any) -> None:
            pass
        def load(self) -> None:
            assert not oversize, "oversized image decoded before cap"
            self.loaded = True
        def convert(self, mode: str) -> Any:
            return self
        def getchannel(self, _: str) -> Any:
            return self
        def getbbox(self) -> tuple[int, int, int, int]:
            return (0, 0, 2, 2) if self.trimmed else (1, 1, 3, 3)
        def crop(self, bbox: Any) -> Any:
            return Image(True)
        def tobytes(self) -> bytes:
            return b"pixels"
        def save(self, buffer: Any, **kwargs: Any) -> None:
            chunk = kwargs["pnginfo"].chunks[0][1]
            encoded_metadata["clipperx_brand_v1"] = chunk.split(b"\0", 1)[1].decode()
            buffer.write(b"encoded")
    def load(stream: Any) -> Image:
        assert isinstance(stream, io.BytesIO)
        image = Image(stream.getvalue() == b"encoded")
        images.append(image)
        return image
    monkeypatch.setattr(assets.Image, "open", load)
    registry = assets.RenderAssetRegistry({"brand": source}, {}, {})
    if oversize:
        with pytest.raises(ValueError, match="pixel cap"):
            assets.prepare_brand_asset(registry, "brand", cache_dir=ROOT / "assets", _files=c.files)
        assert not any(image.loaded for image in images)
    else:
        first = assets.prepare_brand_asset(registry, "brand", cache_dir=ROOT / "assets", _files=c.files)
        second = assets.prepare_brand_asset(registry, "brand", cache_dir=ROOT / "assets", _files=c.files)
        assert first == second and first.size == (2, 2) and first.source_bbox == (1, 1, 3, 3)
        assert c.fs.data[source] == b"original" and len(images) == 2
    assert c.files.retire().files_clear
    assert finish_job(c.h).own0


def test_frozen_status_rejects_unbounded_metadata(memory: Any) -> None:
    from clipperx._retained_files import RetainedActionStatus
    with pytest.raises(ValueError):
        RetainedActionStatus(True, str(ROOT), "read", "returned")
    with pytest.raises(ValueError):
        RetainedActionStatus(1, "x" * 2049, "read", "returned")
    assert finish_job(memory.h).own0


@pytest.mark.parametrize("fault", ["read", "write", "flush", "fstat", "fsync", "delete"])
def test_uncertain_file_action_remains_owned(memory: Any, fault: str) -> None:
    c, path = memory, ROOT / "data"
    c.files.register(path, role="raw", owned=True, max_bytes=10)
    with c.files.open_file(path, "xb") as writer:
        writer.write(memoryview(b"abc"))
    def fail(op: str, _: Path) -> None:
        if op == fault or fault == "delete" and op == "unlink":
            raise OSError("uncertain action")
    c.fs.hook = fail
    if fault == "fsync":
        c.files._fsync = lambda _: (_ for _ in ()).throw(OSError("uncertain action"))
    with pytest.raises(OSError):
        if fault == "delete":
            c.files.path_action("delete", path)
        elif fault == "write":
            other = ROOT / "new"
            c.files.register(other, role="raw", owned=True, max_bytes=10)
            with c.files.open_file(other, "xb") as file:
                file.write(memoryview(b"abc"))
        else:
            with c.files.open_binary(path) as file:
                getattr(file, fault)(1) if fault == "read" else getattr(file, fault)()
    assert not c.files.retire().files_clear
    assert any(row.uncertain for row in c.files.snapshot().rows)
    assert finish_job(c.h).own0


def test_retire_transfers_only_exact_settled_outputs_and_preserves_parent(memory: Any) -> None:
    c, directory = memory, ROOT / "package"
    c.files.register(directory, role="directory", owned=True, max_bytes=0)
    c.files.path_action("mkdir", directory)
    output, scratch = directory / "candidate.mp4", directory / "raw"
    for path, role in ((output, "candidate"), (scratch, "raw")):
        c.files.register(path, role=role, owned=True, max_bytes=10)
        with c.files.open_file(path, "xb") as file:
            file.write(memoryview(b"bytes"))
    status = c.files.retire(retain=(output,))
    assert status.files_clear and not status.cleanup_required
    assert c.fs.data[output] == b"bytes" and scratch not in c.fs.data and directory in c.fs.dirs
    assert next(p for p in status.paths if p.path == str(output)).retained
    assert c.files.retire().files_clear and output in c.fs.data
    with pytest.raises(ValueError):
        c.files.path_action("delete", output)
    assert finish_job(c.h).own0


def test_invalid_retention_is_atomic_and_does_not_transfer_live_or_raw(memory: Any) -> None:
    c, output, raw = memory, ROOT / "candidate.mp4", ROOT / "raw"
    for path, role in ((output, "candidate"), (raw, "raw")):
        c.files.register(path, role=role, owned=True, max_bytes=10)
        with c.files.open_file(path, "xb"):
            pass
    with pytest.raises(ValueError):
        c.files.retire(retain=(output, raw))
    assert not any(p.retained for p in c.files.snapshot().paths)
    token = c.files.acquire_consumer((output,))
    with pytest.raises(ValueError):
        c.files.retire(retain=(output,))
    assert output in c.fs.data and raw in c.fs.data
    c.files.release_consumer(token, retired=True)
    assert c.files.retire().files_clear
    assert finish_job(c.h).own0


def test_deadline_scope_exit_cannot_be_replayed(memory: Any) -> None:
    c = memory
    scope = c.files.stage_deadline(100)
    with scope:
        pass
    with c.files.stage_deadline(200):
        with pytest.raises(ValueError):
            scope.__exit__(None, None, None)
        assert c.h.job._effective_deadline() == (200, "driver")
    assert finish_job(c.h).own0


@pytest.mark.parametrize("role", ["original", "runtime", "checked"])
def test_readonly_retention_rejection_preserves_all_flags_and_custody(memory: Any, role: str) -> None:
    c = memory
    retained, candidate, readonly = ROOT / "retained.mp4", ROOT / "candidate.mp4", ROOT / role
    c.files.register(retained, role="candidate", owned=True, max_bytes=10)
    with c.files.open_file(retained, "xb") as file:
        file.write(memoryview(b"kept"))
    assert c.files.retire(retain=(retained,)).files_clear
    c.files.register(candidate, role="candidate", owned=True, max_bytes=10)
    with c.files.open_file(candidate, "xb") as file:
        file.write(memoryview(b"candidate"))
    c.files.register(readonly, role=role, owned=False, max_bytes=10)
    c.fs.data[readonly] = b"original"
    with c.files.open_binary(readonly) as file:
        assert file.read(10) == b"original"
    before, data, events = c.files.snapshot(), dict(c.fs.data), tuple(c.fs.events)
    with pytest.raises(ValueError, match="certainly settled owned output leaves"):
        c.files.retire(retain=(candidate, readonly))
    assert c.files.snapshot() == before
    assert c.fs.data == data and tuple(c.fs.events) == events
    assert [path.path for path in before.paths if path.retained] == [str(retained)]
    assert c.files.retire().files_clear
    assert candidate not in c.fs.data
    assert c.fs.data[readonly] == b"original" and c.fs.data[retained] == b"kept"
    assert finish_job(c.h).own0


@pytest.mark.parametrize("fault", ["close", "unlink"])
def test_uncertain_retention_rejection_preserves_descriptor_or_action_custody(memory: Any, fault: str) -> None:
    c = memory
    retained, candidate, uncertain = ROOT / "retained.mp4", ROOT / "candidate.mp4", ROOT / "uncertain.mp4"
    c.files.register(retained, role="candidate", owned=True, max_bytes=10)
    with c.files.open_file(retained, "xb") as file:
        file.write(memoryview(b"kept"))
    assert c.files.retire(retain=(retained,)).files_clear
    for path in (candidate, uncertain):
        c.files.register(path, role="candidate", owned=True, max_bytes=10)
    with c.files.open_file(candidate, "xb") as file:
        file.write(memoryview(b"candidate"))
    def fail(operation: str, path: Path) -> None:
        if operation == fault and path == uncertain:
            raise OSError("uncertain retained leaf")
    if fault == "close":
        c.fs.hook = fail
        with pytest.raises(OSError, match="uncertain retained leaf"):
            with c.files.open_file(uncertain, "xb") as file:
                file.write(memoryview(b"uncertain"))
    else:
        with c.files.open_file(uncertain, "xb") as file:
            file.write(memoryview(b"uncertain"))
        c.fs.hook = fail
        with pytest.raises(OSError, match="uncertain retained leaf"):
            c.files.path_action("delete", uncertain)
    before, data, events = c.files.snapshot(), dict(c.fs.data), tuple(c.fs.events)
    unresolved = tuple(row for row in before.rows if row.uncertain)
    assert len(unresolved) == 1 and unresolved[0].path == str(uncertain)
    with pytest.raises(ValueError, match="certainly settled owned output leaves"):
        c.files.retire(retain=(candidate, uncertain))
    assert c.files.snapshot() == before
    assert c.fs.data == data and tuple(c.fs.events) == events
    assert [path.path for path in before.paths if path.retained] == [str(retained)]
    after = c.files.retire()
    assert not after.files_clear and after.cleanup_required
    assert tuple(row for row in after.rows if row.uncertain) == unresolved
    assert c.files._busy(uncertain) and c.fs.data[uncertain] == b"uncertain"
    assert candidate not in c.fs.data and c.fs.data[retained] == b"kept"
    assert sum(operation == fault and path == uncertain for operation, path in c.fs.events) == 1
    assert finish_job(c.h).own0


def test_retention_preserves_all_registered_ancestors_and_deletes_disposable_siblings(memory: Any) -> None:
    from dataclasses import FrozenInstanceError
    c = memory
    ancestors = (ROOT / "package", ROOT / "package" / "assets", ROOT / "package" / "assets" / "nested")
    disposable_directory = ancestors[0] / "disposable"
    for directory in (*ancestors, disposable_directory):
        c.files.register(directory, role="directory", owned=True, max_bytes=0)
        c.files.path_action("mkdir", directory)
    retained = ancestors[-1] / "candidate.mp4"
    siblings = tuple(directory / "scratch.raw" for directory in ancestors) + (disposable_directory / "temp.raw",)
    for path, role in ((retained, "candidate"), *((path, "raw") for path in siblings)):
        c.files.register(path, role=role, owned=True, max_bytes=10)
        with c.files.open_file(path, "xb") as file:
            file.write(memoryview(b"bytes"))
    before = c.files.snapshot()
    status = c.files.retire(retain=(retained,))
    assert status.files_clear and not status.cleanup_required
    assert c.fs.data == {retained: b"bytes"}
    assert all(directory in c.fs.dirs for directory in ancestors)
    assert disposable_directory not in c.fs.dirs and ROOT in c.fs.dirs
    assert {path for operation, path in c.fs.events if operation == "unlink"} == set(siblings)
    assert [path for operation, path in c.fs.events if operation == "rmdir"] == [disposable_directory]
    by_path = {path.path: path for path in status.paths}
    assert by_path[str(retained)].retained
    assert all(by_path[str(directory)].present and not by_path[str(directory)].retained for directory in ancestors)
    assert not any(path.retained for path in before.paths)
    with pytest.raises(FrozenInstanceError):
        setattr(by_path[str(retained)], "retained", False)
    assert finish_job(c.h).own0
