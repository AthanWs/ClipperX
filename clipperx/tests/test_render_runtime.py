"""HOST runtime tests; actual local Vulkan qualification is required, never skipped."""

from dataclasses import FrozenInstanceError
from pathlib import Path
import subprocess
import sys

import pytest

from clipperx import render_runtime as runtime
from clipperx.tests.test_filtergraph import host_motion_shot
from clipperx.tests.test_visual_compiler import host_plan


def test_pinned_cpu_is_independent_of_vulkan_and_combines() -> None:
    cpu = runtime.RenderRequirements(runtime.NATIVE_RUNTIME, False, ("showinfo",))
    assert not cpu.needs_vulkan
    combined = runtime.combine_render_requirements(cpu, runtime.RenderRequirements())
    assert combined.runtime_id == runtime.NATIVE_RUNTIME and not combined.needs_vulkan
    native = runtime.RenderRequirements(runtime.NATIVE_RUNTIME, True, ("libplacebo",))
    assert runtime.combine_render_requirements(cpu, native).needs_vulkan
    with pytest.raises(runtime.RenderRuntimeError):
        runtime.RenderRequirements("system", True)


def test_pinned_cpu_resolves_and_qualifies_without_device() -> None:
    requirements = runtime.RenderRequirements(runtime.NATIVE_RUNTIME, False,
                                               ("showinfo", "ashowinfo", "format", "aresample"))
    selected = runtime.resolve_render_runtime(requirements, device_index=None)
    assert selected.device_index is None and selected.device_arguments == ()
    assert selected.ffmpeg_sha256 == runtime.FFMPEG_SHA256
    assert selected.ffprobe_sha256 == runtime.FFPROBE_SHA256
    qualified = runtime.qualify_render_runtime(selected, requirements)
    assert qualified.probe_exit == 0 and qualified.probe_bytes == 16 * 16 * 3
    assert qualified.adapter_name is None and qualified.native_options == ()
    assert "vulkan" not in qualified.probe_log.lower()


def test_requirements_are_detached_immutable_and_combine_native(source: Path) -> None:
    plan = host_plan(source)
    basic = runtime.requirements_for_plan(plan)
    assert basic.runtime_id == "system" and not basic.needs_vulkan
    plan.shots[0] = host_motion_shot()
    native = runtime.requirements_for_plan(plan)
    assert native.runtime_id == "ffmpeg-9.0.2-libplacebo" and native.needs_vulkan
    assert "libplacebo" in native.required_filters
    combined = runtime.combine_render_requirements(basic, native)
    assert combined.needs_vulkan
    assert set(combined.required_filters) >= set(basic.required_filters)
    plan.shots[0].layers[0].crop_track.clear()
    assert native.needs_vulkan
    with pytest.raises(FrozenInstanceError):
        setattr(native, "needs_vulkan", False)
    filters = ["crop"]
    detached = runtime.RenderRequirements(required_filters=filters)
    filters.append("untrusted")
    assert detached.required_filters == ("crop",)


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "HOST-source.mp4"
    path.write_bytes(b"HOST; no media decoding")
    return path


def test_native_runtime_is_exact_paired_pin_with_explicit_device() -> None:
    requirements = runtime.RenderRequirements("ffmpeg-9.0.2-libplacebo", True, ("libplacebo",))
    selected = runtime.resolve_render_runtime(requirements, device_index=0)
    assert selected.ffmpeg.name == "ffmpeg.exe" and selected.ffprobe.name == "ffprobe.exe"
    assert selected.ffmpeg.parent == selected.ffprobe.parent
    assert selected.ffmpeg_sha256 == runtime.FFMPEG_SHA256
    assert selected.ffprobe_sha256 == runtime.FFPROBE_SHA256
    assert selected.graph_file_option == "-/filter_complex"
    assert selected.device_arguments == ("-init_hw_device", "vulkan=vk:0", "-filter_hw_device", "vk")


@pytest.mark.parametrize("index", [-1, True, 1.5])
def test_device_index_is_host_integer_without_retry(index: object) -> None:
    with pytest.raises(ValueError, match="device"):
        runtime.resolve_render_runtime(runtime.RenderRequirements(
            "ffmpeg-9.0.2-libplacebo", True), device_index=index)


@pytest.mark.parametrize("state", ["absent", "wrong-ffmpeg", "wrong-ffprobe"])
def test_missing_or_mismatched_native_pin_is_actionable_without_fallback(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, state: str) -> None:
    from types import SimpleNamespace

    monkeypatch.setattr(runtime, "Paths", lambda: SimpleNamespace(base_dir=tmp_path))
    folder = tmp_path / "data/tools/ffmpeg/9.0.2/bin"
    if state != "absent":
        folder.mkdir(parents=True)
        (folder / "ffmpeg.exe").write_bytes(b"not the pinned ffmpeg")
        (folder / "ffprobe.exe").write_bytes(b"not the pinned ffprobe")
        if state == "wrong-ffprobe":
            import hashlib
            monkeypatch.setattr(runtime, "FFMPEG_SHA256", hashlib.sha256(
                (folder / "ffmpeg.exe").read_bytes()).hexdigest().upper())
    monkeypatch.setattr(runtime.shutil, "which", lambda name: pytest.fail("no PATH fallback"))
    with pytest.raises(runtime.RenderRuntimeError, match="bundle|SHA256"):
        runtime.resolve_render_runtime(runtime.RenderRequirements("ffmpeg-9.0.2-libplacebo", True))


def test_actual_pinned_vulkan_qualification_returns_immutable_evidence() -> None:
    requirements = runtime.RenderRequirements("ffmpeg-9.0.2-libplacebo", True,
                                              ("libplacebo", "format"))
    selected = runtime.resolve_render_runtime(requirements)
    qualified = runtime.qualify_render_runtime(selected, requirements)
    assert "9.0.2" in qualified.ffmpeg_version
    assert "9.0.2" in qualified.ffprobe_version
    assert "libplacebo" in qualified.filters
    assert qualified.probe_bytes == 16*16*4 and qualified.probe_exit == 0
    assert qualified.runtime == selected
    assert qualified.source_color == ()
    with pytest.raises(FrozenInstanceError):
        setattr(qualified, "probe_exit", 1)


@pytest.mark.parametrize("failure", ["filter", "option", "device", "timeout", "format"])
def test_qualification_fails_explicitly_and_never_chooses_another_device(
        monkeypatch: pytest.MonkeyPatch, failure: str) -> None:
    requirements = runtime.RenderRequirements("ffmpeg-9.0.2-libplacebo", True, ("libplacebo",))
    selected = runtime.resolve_render_runtime(requirements)
    calls: list[tuple[str, ...]] = []

    def run(toolchain: runtime.RenderRuntime, tool: str, args: tuple[str, ...],
            *, timeout: int = 15) -> subprocess.CompletedProcess[bytes]:
        calls.append(args)
        if "-version" in args:
            output = b"ffmpeg version 9.0.2 HOST"
        elif "-filters" in args:
            output = b" .. libplacebo V->V native" if failure != "filter" else b" .. crop V->V"
        elif "filter=libplacebo" in args:
            output = "\n".join(f" {name} <string>" for name in runtime.NATIVE_OPTIONS).encode()
            if failure == "option":
                output = b" crop_x <string>"
        else:
            if failure == "timeout":
                raise runtime.RenderRuntimeError("Vulkan probe timeout")
            if failure == "device":
                raise runtime.RenderRuntimeError("Vulkan device failed")
            output = b"short invalid frame"
        return subprocess.CompletedProcess(args, 0, output, b"")

    monkeypatch.setattr(runtime, "_run", run)
    with pytest.raises(runtime.RenderRuntimeError):
        runtime.qualify_render_runtime(selected, requirements)
    assert sum("-i" in call for call in calls) <= 1


def test_import_and_pure_compilation_never_probe_or_write() -> None:
    script = """
from pathlib import Path
import subprocess, shutil, sys
def forbidden(*args, **kwargs):
    raise AssertionError('pure compiler attempted host operation')
subprocess.run = forbidden
shutil.which = forbidden
Path.read_bytes = forbidden
Path.write_text = forbidden
from clipperx.tests.test_filtergraph import host_motion_shot
from clipperx.filtergraph import compile_shot_layers
from clipperx.render_runtime import combine_render_requirements
Path.open = forbidden
Path.is_file = forbidden
result = compile_shot_layers(host_motion_shot(), source_size=(1920,1080))
assert result.requirements.needs_vulkan
assert combine_render_requirements(result.requirements).needs_vulkan
for name in ('clipperx.config','torch','clipperx.structured_client'):
    assert name not in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True,
                            text=True, timeout=15, check=False)
    assert result.returncode == 0, result.stderr
