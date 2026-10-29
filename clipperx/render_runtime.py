"""Explicit provider-free render requirements, pinned host runtime and qualification.

Imports/compilation never inspect files, run processes or initialize a device.
Only resolve/qualify are host operations. No downloads, fallback or PATH mutation.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import ContextManager, Protocol, TypedDict, cast

from .local_paths import Paths, paths as _local_paths
from .visual_contracts import FillMode, VisualCompositionPlan, VisualLayer

NATIVE_RUNTIME = "ffmpeg-9.0.2-libplacebo"
FFMPEG_SHA256 = "589E50B766D251AFDF181DD664D40BD94407E200B019989FD7468C7D118A28D0"
FFPROBE_SHA256 = "02264DFA4F56BAAA31E443ECE8AA16D397F6B008200E7FD184D22B9009E75FD3"
NATIVE_POLICY = ("upscaler=ewa_lanczos:downscaler=ewa_lanczos:frame_mixer=none:fps=none:"
                 "apply_filmgrain=0:apply_dolbyvision=0:deband=0:peak_detect=0:"
                 "tonemapping=clip:gamut_mode=clip:contrast_recovery=0:sigmoid=0:"
                 "disable_linear=0:dithering=ordered_fixed:dither_temporal=0")
NATIVE_OPTIONS = ("w", "h", "format", "crop_x", "crop_y", "crop_w", "crop_h",
                  "pos_x", "pos_y", "pos_w", "pos_h", "fillcolor", "alpha_mode", "reset_sar",
                  *(option.split("=")[0] for option in NATIVE_POLICY.split(":")))


class RenderRuntimeError(ValueError):
    """Actionable host capability failure; never authorizes another backend."""


@dataclass(frozen=True, slots=True, init=False)
class RenderRequirements:
    runtime_id: str
    needs_vulkan: bool
    required_filters: tuple[str, ...]

    def __init__(self, runtime_id: str = "system", needs_vulkan: bool = False,
                 required_filters: Sequence[str] = ()) -> None:
        if (runtime_id not in ("system", NATIVE_RUNTIME) or type(needs_vulkan) is not bool
                or (needs_vulkan and runtime_id != NATIVE_RUNTIME)):
            raise RenderRuntimeError("Invalid explicit render runtime requirements")
        if any(not re.fullmatch(r"[a-z0-9_]+", name) for name in required_filters):
            raise RenderRuntimeError("Invalid required filter name")
        object.__setattr__(self, "runtime_id", runtime_id)
        object.__setattr__(self, "needs_vulkan", needs_vulkan)
        object.__setattr__(self, "required_filters", tuple(sorted(set(required_filters))))


def _requirements_for_layers(layers: Iterable[VisualLayer]) -> RenderRequirements:
    layers = tuple(layers)
    moving = any(len(layer.crop_track) > 1 for layer in layers)
    filters = {"color", "trim", "setpts", "settb", "fps", "split", "null", "setsar", "overlay"}
    if any(len(layer.crop_track) <= 1 for layer in layers):
        filters.update(("crop", "scale"))
    if any(layer.fill == FillMode.CONTAIN_BLUR for layer in layers):
        filters.update(("format", "boxblur"))
    if moving:
        filters.update(("libplacebo", "format"))
    return RenderRequirements(NATIVE_RUNTIME if moving else "system", moving, tuple(filters))


def requirements_for_plan(plan: VisualCompositionPlan) -> RenderRequirements:
    """Pure conservative requirements from explicit layers; no availability choice."""
    return _requirements_for_layers(layer for shot in plan.shots for layer in shot.layers)


def combine_render_requirements(*requirements: RenderRequirements) -> RenderRequirements:
    moving = any(req.needs_vulkan for req in requirements)
    pinned = any(req.runtime_id == NATIVE_RUNTIME for req in requirements)
    return RenderRequirements(NATIVE_RUNTIME if pinned else "system", moving,
        tuple(name for req in requirements for name in req.required_filters))


@dataclass(frozen=True, slots=True)
class RenderRuntime:
    ffmpeg: Path
    ffprobe: Path
    runtime_id: str
    ffmpeg_sha256: str
    ffprobe_sha256: str
    expected_ffmpeg_sha256: str | None
    expected_ffprobe_sha256: str | None
    graph_file_option: str
    device_index: int | None

    @property
    def device_arguments(self) -> tuple[str, ...]:
        if self.device_index is None:
            return ()
        return "-init_hw_device", f"vulkan=vk:{self.device_index}", "-filter_hw_device", "vk"


@dataclass(frozen=True, slots=True)
class RuntimeQualification:
    runtime: RenderRuntime
    ffmpeg_version: str
    ffprobe_version: str
    filters: tuple[str, ...]
    native_options: tuple[str, ...]
    device_index: int | None
    adapter_name: str | None
    probe_exit: int | None
    probe_bytes: int
    probe_seconds: float | None
    probe_log: str
    source_color: tuple[tuple[str, str | None], ...] = ()


class RuntimeHashReader(Protocol):
    def read(self, size: int) -> bytes: ...


class RuntimeHashFiles(Protocol):
    def open_binary(self, path: Path) -> ContextManager[RuntimeHashReader]: ...


class RuntimeToolRunner(Protocol):
    def __call__(self, argv: tuple[str, ...], *, timeout: int,
                 max_stdout_bytes: int, max_stderr_bytes: int) -> subprocess.CompletedProcess[bytes]: ...


class RuntimePathResolver(Protocol):
    def resolve_runtime_path(self, path: Path) -> Path: ...


class _RunnerOptions(TypedDict, total=False):
    _runner: RuntimeToolRunner


def _hash(path: Path, *, _files: RuntimeHashFiles | None = None) -> str:
    try:
        with (path.open("rb") if _files is None else _files.open_binary(path)) as stream:
            digest = hashlib.sha256()
            for chunk in iter(lambda: stream.read(1024*1024), b""):
                digest.update(chunk)
        return digest.hexdigest().upper()
    except OSError as exc:
        raise RenderRuntimeError(f"Render runtime bundle missing/unreadable: {path}") from exc


def _verify_pins(runtime: RenderRuntime, *, _files: RuntimeHashFiles | None = None) -> None:
    if runtime.runtime_id != NATIVE_RUNTIME:
        return
    for path, expected in ((runtime.ffmpeg, FFMPEG_SHA256), (runtime.ffprobe, FFPROBE_SHA256)):
        if (_hash(path) if _files is None else _hash(path, _files=_files)) != expected:
            raise RenderRuntimeError(f"Pinned runtime SHA256 mismatch: {path}; restore the qualified bundle")


def _runtime_paths(requirements: RenderRequirements, *, _base_dir: Path | None = None,
                   _resolver: RuntimePathResolver | None = None) -> tuple[Path, Path]:
    """The single existing selector; HOST can declare these paths before hashing."""
    if requirements.runtime_id == NATIVE_RUNTIME:
        folder = (_local_paths.base_dir if _base_dir is None else _base_dir) / "data/tools/ffmpeg/9.0.2/bin"
        return folder / "ffmpeg.exe", folder / "ffprobe.exe"
    paths = tuple(shutil.which(name) for name in ("ffmpeg", "ffprobe"))
    if any(path is None for path in paths):
        raise RenderRuntimeError("System ffmpeg/ffprobe not found in PATH")
    ffmpeg, ffprobe = ((Path(path).resolve() if _resolver is None else _resolver.resolve_runtime_path(Path(path)))
                      for path in paths if path is not None)
    return ffmpeg, ffprobe


def resolve_render_runtime(requirements: RenderRequirements, *, device_index: object = 0,
                           _files: RuntimeHashFiles | None = None) -> RenderRuntime:
    """Explicit host selection, paired pins checked before any native invocation."""
    if requirements.needs_vulkan and (type(device_index) is not int or device_index < 0):
        raise RenderRuntimeError("Vulkan device index requires a nonnegative host integer")
    if _files is None:
        ffmpeg, ffprobe = (_runtime_paths(requirements, _base_dir=Paths().base_dir)
            if requirements.runtime_id == NATIVE_RUNTIME else _runtime_paths(requirements))
    else:
        ffmpeg, ffprobe = (_runtime_paths(requirements) if requirements.runtime_id == NATIVE_RUNTIME else
                          _runtime_paths(requirements, _resolver=cast(RuntimePathResolver, _files)))
    options = {} if _files is None else {"_files": _files}
    if requirements.runtime_id == NATIVE_RUNTIME:
        selected = RenderRuntime(ffmpeg, ffprobe, NATIVE_RUNTIME, _hash(ffmpeg, **options), _hash(ffprobe, **options),
                                  FFMPEG_SHA256, FFPROBE_SHA256, "-/filter_complex",
                                  device_index if isinstance(device_index, int)
                                  and requirements.needs_vulkan else None)
        _verify_pins(selected, **options)
        return selected
    return RenderRuntime(ffmpeg, ffprobe, "system", _hash(ffmpeg, **options), _hash(ffprobe, **options),
                          None, None, "-filter_complex_script", None)


def _run(runtime: RenderRuntime, tool: str, args: tuple[str, ...], *,
         timeout: int = 15, _runner: RuntimeToolRunner | None = None) -> subprocess.CompletedProcess[bytes]:
    if tool not in ("ffmpeg", "ffprobe"):
        raise RenderRuntimeError("Unknown paired runtime tool")
    try:
        if _runner is None:
            result = subprocess.run([str(getattr(runtime, tool)), *args], capture_output=True,
                                    timeout=timeout, check=False)
        else:
            result = _runner((str(getattr(runtime, tool)), *args), timeout=timeout,
                             max_stdout_bytes=1_000_000, max_stderr_bytes=1_000_000)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RenderRuntimeError(f"{tool} runtime inspection/probe failed: {exc}") from exc
    if result.returncode:
        raise RenderRuntimeError(f"{tool} runtime/device/format failed ({result.returncode}): "
                                 + result.stderr.decode(errors="replace").strip())
    return result


def qualify_render_runtime(runtime: RenderRuntime,
                           requirements: RenderRequirements, *, _files: RuntimeHashFiles | None = None,
                           _runner: RuntimeToolRunner | None = None) -> RuntimeQualification:
    """Bounded CPU or Vulkan frame; neither qualifies a delivery encoder."""
    if (_files is None) != (_runner is None):
        raise RenderRuntimeError("Owned qualification requires paired file and tool services")
    if requirements.runtime_id == NATIVE_RUNTIME and runtime.runtime_id != NATIVE_RUNTIME:
        raise RenderRuntimeError("Pinned requirements need the explicit paired runtime")
    if not requirements.needs_vulkan and runtime.device_index is not None:
        raise RenderRuntimeError("CPU qualification requires a runtime without a device")
    if requirements.needs_vulkan and (runtime.runtime_id != NATIVE_RUNTIME
            or type(runtime.device_index) is not int or runtime.device_index < 0):
        raise RenderRuntimeError("Native requirements need the explicit pinned runtime/device")
    _verify_pins(runtime, **({} if _files is None else {"_files": _files}))
    run_options: _RunnerOptions = {} if _runner is None else {"_runner": _runner}
    versions = []
    for tool in ("ffmpeg", "ffprobe"):
        output = _run(runtime, tool, ("-version",), **run_options).stdout.decode(errors="replace")
        if not output.strip() or (requirements.runtime_id == NATIVE_RUNTIME
                                 and "9.0.2" not in output.splitlines()[0]):
            raise RenderRuntimeError(f"Unexpected {tool} runtime version")
        versions.append(output.splitlines()[0])
    listing = _run(runtime, "ffmpeg", ("-hide_banner", "-filters"), **run_options).stdout.decode(errors="replace")
    filters = tuple(sorted(set(re.findall(r"^\s*[A-Z.]+\s+(\w+)\s", listing, re.MULTILINE))))
    missing = set(requirements.required_filters) - set(filters)
    if missing:
        raise RenderRuntimeError(f"Runtime missing required filters: {sorted(missing)}")
    options: tuple[str, ...] = ()
    probe_exit, probe_bytes, seconds, log, adapter = None, 0, None, "", None
    if requirements.needs_vulkan:
        result = _run(runtime, "ffmpeg", ("-hide_banner", "-h", "filter=libplacebo"), **run_options)
        help_text = (result.stdout+result.stderr).decode(errors="replace")
        options = tuple(sorted(set(re.findall(r"^\s+(\w+)\s+<", help_text, re.MULTILINE))))
        if set(NATIVE_OPTIONS)-set(options):
            raise RenderRuntimeError(f"Native runtime missing libplacebo options: "
                                     f"{sorted(set(NATIVE_OPTIONS)-set(options))}")
        graph = ("libplacebo=w=16:h=16:format=rgba:crop_x='0*n':crop_y=0:crop_w=iw:crop_h=ih:"
                 "pos_w=ow:pos_h=oh:pos_x=0:pos_y=0:fillcolor=black:alpha_mode=straight:reset_sar=1:"
                 + NATIVE_POLICY)
        started = time.perf_counter()
        result = _run(runtime, "ffmpeg", ("-hide_banner", "-loglevel", "info",
            *runtime.device_arguments, "-f", "lavfi", "-i", "testsrc2=s=16x16:r=60",
            "-vf", graph, "-frames:v", "1", "-an", "-threads", "1", "-pix_fmt", "rgba",
            "-f", "rawvideo", "pipe:1"), timeout=45, **run_options)
        seconds = time.perf_counter()-started
        probe_bytes, probe_exit = len(result.stdout), result.returncode
        log = result.stderr.decode(errors="replace")
        if probe_bytes != 16*16*4:
            raise RenderRuntimeError(f"Native probe returned invalid RGBA frame: {probe_bytes} bytes")
        match = re.search(r"Using device:\s*([^\r\n]+)", log)
        adapter = match.group(1).strip() if match else None
    elif requirements.runtime_id == NATIVE_RUNTIME:
        started = time.perf_counter()
        result = _run(runtime, "ffmpeg", ("-hide_banner", "-loglevel", "error",
            "-f", "lavfi", "-i", "testsrc2=s=16x16:r=1", "-frames:v", "1", "-an",
            "-threads", "1", "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"), **run_options)
        seconds = time.perf_counter() - started
        probe_bytes, probe_exit = len(result.stdout), result.returncode
        log = result.stderr.decode(errors="replace")
        if probe_bytes != 16 * 16 * 3:
            raise RenderRuntimeError("CPU probe returned invalid BGR frame")
    return RuntimeQualification(runtime, versions[0], versions[1], filters, options,
                                runtime.device_index, adapter, probe_exit, probe_bytes, seconds, log)
