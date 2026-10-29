"""Opt-in finite native qualification. Default mode performs no setup IO.

Run as ``python -B -m scripts.render_visual_smoke``. Root must adopt and pin
the source manifest before using --native --confirm-host. Controller setup and
post-return summaries are bounded IO outside media jobs, never driver own0 proof.
"""

import argparse
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass, replace
from fractions import Fraction as F
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Generator, Iterator, NoReturn, Sequence, cast

if TYPE_CHECKING:
    from clipperx._source_host_ownership import SourceHostJob, SourceJobStatus

from clipperx.tests import media_factory as recipe
from clipperx.tests import native_render_oracles as oracle

MiB, GiB = 1024**2, 1024**3
STAGES = ("prerequisites", "synthetic", "gameplay", "camera", "transition")
LIMITS = dict(probe_seconds=45, probe_bytes=1_000_000, job_seconds=7200,
    metadata_seconds=15, source_seconds=45, float_seconds=45, spatial_seconds=900,
    finalmix_seconds=120, encoder_seconds=900, preparation_seconds=120,
    cleanup_seconds=5, candidate_bytes=2147483648, reference_bytes=16*MiB,
    fixture_bytes=256*MiB, image_bytes=128*MiB, oracle_bytes=8*MiB,
    stderr_bytes=16*MiB, stderr_tail=65536, stderr_line=16384,
    log_bytes=256*MiB, max_paths=32768, max_live=16, max_generations=262144,
    campaign_children=352, campaign_seconds=17100)
BUDGET_SEMANTICS = dict(public_b="admission plus post-return elapsed/count failure ceilings; no hard preemption or OS child quota",
    public_b_job="unchanged enforced7200s plus existing smaller child/stage deadlines and5s cleanup grace",
    fixture_oracle="owned job deadline min(campaign,job), child min(parent,stage)")
EXTERNAL_PINS = dict(
    ffmpeg="589e50b766d251afdf181dd664d40bd94407e200b019989fd7468c7d118a28d0",
    ffprobe="02264dfa4f56baaa31e443ece8aa16d397f6b008200e7fd184d22b9009e75fd3",
    headline_impact="d8bbdb72e3f930157824b2e59677d3a912d4512c14a948e6656a2e3fb0c935ad",
    subtitle_bold="e8f4e3baf6cc35fed6fcce3a540e8b39e8f6cda1d22a28f2ec8f526fef7a43f5")
LINEAGE = {
    ".superpowers/sdd/snapshots/task-56/final/phase3-task-56-report.md":
        "f84e12828f92c4601a127899110b73654f88bcc9817fb672fa261ed55181d178",
    ".superpowers/sdd/snapshots/task-56/final/phase3-task-56-review.md":
        "8b100b09a91c4f94e1840993006c56efd89b236dd216fe9ab217df998bc68524",
    ".superpowers/sdd/snapshots/task-56/final/review-task-56.diff":
        "c822e102812576aa69692ccb34a587448cce17399749776f4c374b3db4402a75"}
UNIT_PATHS = ("clipperx/tests/media_factory.py", "clipperx/tests/native_render_oracles.py",
    "clipperx/tests/test_native_render_qualification.py",
    "clipperx/tests/test_visual_render_integration.py", "scripts/render_visual_smoke.py")
AUTHORITY_PATHS = (
    "clipperx/plan_render.py", "clipperx/_visual_render_host.py",
    "clipperx/_visual_render_audio.py", "clipperx/_visual_render_video.py",
    "clipperx/_source_host_ownership.py", "clipperx/_windows_pipe_io.py",
    "clipperx/_retained_files.py", "clipperx/_render_services.py",
    "clipperx/_normalization_files.py", "clipperx/_normalization_transport.py",
    "clipperx/audio_selection.py", "clipperx/audio_mix.py", "clipperx/audio.py",
    "clipperx/source_decode.py", "clipperx/render_clock.py", "clipperx/render_runtime.py",
    "clipperx/visual_compiler.py", "clipperx/visual_contracts.py", "clipperx/output_timeline.py",
    "clipperx/overlay_ass.py", "clipperx/font_staging.py", "clipperx/render_assets.py",
    "clipperx/render_contracts.py", "clipperx/render_metrics.py", "clipperx/filtergraph.py",
    "clipperx/local_paths.py", "clipperx/schema.py",
    ".superpowers/sdd/phase3-task-57-brief.md",
    ".superpowers/sdd/phase3-native-render-qualification-plan.md",
    ".superpowers/sdd/phase3-native-render-gate-root-notes.md")


def json_bytes(value: Any) -> bytes:
    def scalar(item: Any) -> Any:
        if is_dataclass(item) and not isinstance(item, type):
            return asdict(item)
        if isinstance(item, (Path, F)):
            return str(item)
        if isinstance(item, bytes):
            return dict(bytes=len(item), sha256=hashlib.sha256(item).hexdigest())
        raise TypeError(f"not bounded JSON scalar: {type(item)}")
    raw = json.dumps(value, default=scalar, sort_keys=True, allow_nan=False).encode()
    if len(raw) > 8*MiB:
        raise ValueError("oracle/controller JSON ceiling")
    return raw


def validate_activation(native: bool, confirmed: bool) -> None:
    if not native:
        raise ValueError("explicit --native opt-in required")
    if not confirmed:
        raise ValueError("explicit --confirm-host required")


def example_manifest() -> dict[str, Any]:
    """Schema example ONLY. Zero source hashes deliberately fail live identity checks."""
    return dict(version=1, accepted_b="snapshots/task-56/final", recipe=recipe.RECIPE_VERSION,
        cases=list(STAGES), limits=dict(LIMITS), lineage=dict(LINEAGE),
        budget_semantics=dict(BUDGET_SEMANTICS),
        runs={stage: "ROOT_ASSIGN_"+stage for stage in STAGES},
        pins={path: "0"*64 for path in (*UNIT_PATHS, *AUTHORITY_PATHS)},
        external={key: dict(path=f"PIN_ABSOLUTE_{key}", sha256=value)
                  for key, value in EXTERNAL_PINS.items()}, real={}, gate_acceptance={})


def safe_relative(text: str) -> Path:
    if (type(text) is not str or not text.isascii() or "\\" in text
            or not re.fullmatch(r"[A-Za-z0-9_./-]{1,2048}", text)):
        raise ValueError("fixed ASCII relative manifest path required")
    path = Path(text)
    if path.is_absolute() or ".." in path.parts or ":" in text:
        raise ValueError("manifest path traversal")
    return path


def validate_manifest(data: Any) -> None:
    if type(data) is not dict or set(data) != {
            "version", "accepted_b", "recipe", "cases", "limits", "pins", "external",
            "real", "gate_acceptance", "lineage", "runs", "budget_semantics"}:
        raise ValueError("exact finite manifest schema required")
    if (data["version"] != 1 or data["accepted_b"] != "snapshots/task-56/final"
            or data["recipe"] != recipe.RECIPE_VERSION or data["cases"] != list(STAGES)
            or data["limits"] != LIMITS or data["lineage"] != LINEAGE
            or data["budget_semantics"] != BUDGET_SEMANTICS):
        raise ValueError("manifest authority/recipe/cases/frozen limits mismatch")
    runs = data["runs"]
    if (type(runs) is not dict or set(runs) != set(STAGES)
            or len(set(runs.values())) != 5
            or any(type(value) is not str or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value)
                   for value in runs.values())):
        raise ValueError("five unique Root-pinned run IDs required; one consumption per manifest case")
    pins = data["pins"]
    if type(pins) is not dict or not set((*UNIT_PATHS, *AUTHORITY_PATHS)) <= set(pins) or len(pins) > 256:
        raise ValueError("manifest missing source57/B/55/compiler/clock/asset pins")
    for path, digest in pins.items():
        safe_relative(path)
        if type(digest) is not str or not re.fullmatch("[0-9a-f]{64}", digest):
            raise ValueError("manifest SHA256 schema")
    if type(data["external"]) is not dict or set(data["external"]) != set(EXTERNAL_PINS):
        raise ValueError("exact original font/runtime pins required")
    for key, expected in EXTERNAL_PINS.items():
        row = data["external"][key]
        if type(row) is not dict or set(row) != {"path", "sha256"} or row["sha256"] != expected:
            raise ValueError("runtime/original font pin mismatch")
    if type(data["real"]) is not dict or len(data["real"]) > 3:
        raise ValueError("exactly three bounded real case slots")
    for stage, row in data["real"].items():
        if stage not in STAGES[2:] or type(row) is not dict or set(row) != {
                "source", "plan", "transcript", "source_fingerprint", "windows", "evidence"}:
            raise ValueError("real case evidence/plan/source schema")
        for name in ("plan", "transcript", "evidence"):
            if row[name] is not None and row[name] not in pins:
                raise ValueError("real plan/transcript/evidence must be pinned")
        windows = row["windows"]
        if (type(windows) is not list or not 1 <= len(windows) <= 2
                or any(type(w) is not list or len(w) != 2 for w in windows)):
            raise ValueError("bounded evidence windows required")
        ranges = [(F(a), F(b)) for a, b in windows]
        if (any(a < 0 or b <= a for a, b in ranges)
                or sum(b-a for a, b in ranges) > 8
                or any(b != c for (_, b), (c, _) in zip(ranges, ranges[1:]))):
            raise ValueError("real windows must be contiguous and <=8 seconds")
    if type(data["gate_acceptance"]) is not dict or len(data["gate_acceptance"]) > 2:
        raise ValueError("Root gate acceptance schema")


def require_absence(directory_exists: bool, guard_exists: bool) -> None:
    if directory_exists or guard_exists:
        raise ValueError("consumed/replayed output directory or guard; new run-id required")


def disk_budget(width: int, height: int, frames: Sequence[int], consuming: Sequence[bool]
                ) -> dict[str, int]:
    if min(width, height) <= 0 or len(frames) != len(consuming) or not frames:
        raise ValueError("actual geometry/frame bounds")
    raw = 8294400*sum(frames)+max((3*width*height*n for n, use in zip(frames, consuming) if use), default=0)
    subtotal = raw+4*68736000+512*MiB+512*MiB+256*MiB+4*GiB
    margin = max(8*GiB, (subtotal+9)//10)
    return dict(raw=raw, subtotal=subtotal, margin=margin, required=subtotal+margin)


def admit(stage: str, *, children: int = 0, seconds: float = 0,
          generations: int = 0, paths: int = 0, live: int = 0,
          free: int = 1 << 60, required: int = 1) -> None:
    if stage not in STAGES:
        raise ValueError("unknown finite stage")
    child_cap, wall = (96, 2700) if stage == "prerequisites" else (64, 3600)
    if (not 0 <= children <= child_cap or not 0 <= seconds <= wall or free < required
            or not 0 <= paths <= 32768 or not 0 <= live <= 16
            or not 0 <= generations <= 262144):
        raise ValueError("qualification process/wall/disk/path/generation admission failed")


def fs_rejected(size: int, natural_frames: int) -> bool:
    return size > 1024 or natural_frames != 6


def cancelled_result(*, admitted: bool, acquired: bool, before_eof: bool,
                     stop_code: str, own0: bool, files_clear: bool) -> None:
    if not (admitted and acquired and before_eof and stop_code == "cancelled" and own0 and files_clear):
        raise ValueError("missed cancel window/unexpected cause/uncertain total custody")


def sequence(cases: Sequence[str], run: Callable[[str], Any]) -> list[Any]:
    return [run(case) for case in cases]  # exceptions stop dependent gates immediately


def bounded_read(path: Path, cap: int) -> bytes:
    with path.open("rb") as stream:
        result = stream.read(cap+1)
    if len(result) > cap:
        raise ValueError("bounded controller input exceeded")
    return result


def file_hash(path: Path, cap: int, deadline: int) -> str:
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        while True:
            if time.monotonic_ns() >= deadline:
                raise ValueError("controller identity preparation deadline")
            raw = stream.read(MiB)
            if not raw:
                break
            size += len(raw)
            if size > cap:
                raise ValueError("source/runtime/font identity byte ceiling")
            digest.update(raw)
    return digest.hexdigest()


def load_manifest(path: Path, root: Path) -> tuple[dict[str, Any], str]:
    raw = bounded_read(path, 1_000_000)
    data = json.loads(raw)
    validate_manifest(data)
    deadline = time.monotonic_ns()+120*10**9
    check_source_pins(data, root, deadline)
    for row in data["external"].values():
        external = Path(row["path"])
        if not external.is_absolute() or not str(external).isascii():
            raise ValueError("fixed absolute ASCII original runtime/font path required")
        if file_hash(external, 256*MiB, deadline) != row["sha256"]:
            raise ValueError("original runtime/font byte identity mismatch")
    return data, hashlib.sha256(raw).hexdigest()


def check_source_pins(data: dict[str, Any], root: Path, deadline: int) -> None:
    """Root's newly adopted current pinset is execution authority; immutable
    accepted56 evidence is provenance. Approved later fixes require new manifest
    pins/review, never recomputation or a bypass of equality with live bytes.
    """
    for name, expected in {**data["pins"], **data["lineage"]}.items():
        if file_hash(root/safe_relative(name), 32*MiB, deadline) != expected:
            raise ValueError(f"current source pin mismatch: {name}")


def public_render(source: Path, plan: Path, output: Path, *, assets: Any,
                  transcript: Path | None) -> list[Path]:
    from clipperx.plan_render import render_visual_composition_plan
    from clipperx.render_contracts import EncoderName, RenderProfile
    return render_visual_composition_plan(source, plan, output, transcript_path=transcript,
        assets=assets, profile=RenderProfile(video_encoder=EncoderName.H264_NVENC))


class Campaign:
    """Controller ledger. Public B reserves a finite case allowance before entry."""

    def __init__(self, stage: str) -> None:
        self.stage, self.start, self.children = stage, time.monotonic_ns(), 0
        self.records: list[Any] = []

    def charge(self, count: int = 1) -> None:
        admit(self.stage, children=self.children+count,
              seconds=(time.monotonic_ns()-self.start)/1e9)
        self.children += count


class QualificationFailure(ValueError):
    def __init__(self, cause: BaseException, evidence: Any) -> None:
        super().__init__(str(cause))
        self.evidence = evidence


def error_details(error: BaseException) -> dict[str, Any]:
    """Bounded traceback objects only: never format_exception/linecache/source IO."""
    frames = []
    trace = error.__traceback__
    for _ in range(32):
        if trace is None:
            break
        code = trace.tb_frame.f_code
        frames.append(dict(filename=code.co_filename[:512], function=code.co_name[:128],
                           lineno=trace.tb_lineno))
        trace = trace.tb_next
    cause = error.__cause__
    return dict(type=type(error).__name__[:128], message=str(error)[:2048],
        frames=frames[-8:], trace_truncated=len(frames) > 8 or trace is not None,
        cause=None if cause is None else dict(type=type(cause).__name__[:128], message=str(cause)[:2048]))


def record_public_return(campaign: Campaign, actual: int, reservation: int) -> None:
    campaign.records.append(dict(scope="public B returned; budget observation, not OS enforcement",
        actual_children=actual, reserved_children=reservation,
        elapsed_ns=time.monotonic_ns()-campaign.start, budget_semantics=BUDGET_SEMANTICS))
    if type(actual) is not int or not 0 < actual <= reservation:
        raise QualificationFailure(ValueError("public B process prediction exceeded or invalid"), campaign.records)
    campaign.children -= reservation-actual
    try:
        campaign.charge(0)
    except ValueError as error:
        raise QualificationFailure(error, campaign.records) from error


def validate_public_evidence(meta: dict[str, Any], frames: int,
                             shot_frames: Sequence[int] | None = None, *,
                             synthetic: bool = False, all_solid: bool = False) -> None:
    if meta["maximum_active_media_children"] != 1:
        raise ValueError("serial child maximum mismatch")
    spatial = [row for row in meta["stages"] if row["kind"] == "spatial"]
    encoders = [row for row in meta["stages"] if row["kind"] == "encoder"]
    counts = (tuple(shot_frames) if shot_frames is not None else
              (150, 120, 210) if frames == 480 else (2, 2, 2) if frames == 6 else (frames,))
    if ([row["stdout_bytes"] for row in spatial] != [n*8294400 for n in counts]
            or len(encoders) != 1 or any(not row["natural_eof"] or not row["retired"]
                                       for row in (*spatial, *encoders))):
        raise ValueError("whole-shot spatial/one delivery encode/EOF/retirement mismatch")
    if all_solid:
        source_stream = meta.get("process_counts", {}).get("source_stream")
        if (meta["source_bindings"] or type(source_stream) is not int or source_stream != 0
                or any(token == "bgr24" for row in spatial for token in row.get("argv", ()))):
            raise ValueError("all-SOLID unexpectedly admitted video observer/BGR inlet")
    if synthetic:
        bindings = meta["source_bindings"]
        if [row["slot_end"]-row["slot_start"] for row in bindings] != [120, 30, 120, 90]:
            raise ValueError("expected120-bound source windows with whole150/210-frame spatial shots")
        slots = [slot for row in bindings for slot in row["slots"]]
        if [slot["request"]["global_k"] for slot in slots] != [*range(150), *range(270, 480)]:
            raise ValueError("noncontiguous source slots/global K mismatch")
        for slot in slots:
            request = slot["request"]
            k = request["global_k"]
            ordinal = recipe.synthetic_source_ordinal(k)
            expected_local = k if k < 150 else k-270
            if (ordinal is None or F(request["requested_source"]) != F(ordinal, 60)
                    or request["local_n"] != expected_local):
                raise ValueError("actual source request/local n120 association mismatch")
        norm = meta["audio"]["normalization"]
        if norm is None or len(norm["passes"]) != 2:
            raise ValueError("one actual two-pass normalizer required for mixed T/E schedule")


class FixtureSession:
    """Test-owned fixture/oracle adapter over the existing 55 services, one driver.

    It does not execute public composition rendering or import private B state.
    All file operations after admission use RetainedFiles.
    """

    def __init__(self, root: Path, manifest: dict[str, Any], campaign: Campaign) -> None:
        self.root, self.manifest, self.campaign = root, manifest, campaign
        self.clock = time.monotonic_ns
        self.records: list[Any] = []
        self.keep: list[Path] = []
        self.cancel_ready = False
        self.cancel_observed = False
        self.state: dict[str, Any] = {}
        self.job: SourceHostJob
        self.files: Any = None
        self.tools: Any = None
        self.runtime: Any = None
        self.generated_bytes = 0
        self._io_deadline: int | None = None
        self._retiring = False

    def deadline(self, seconds: int) -> int:
        return min(self.job._deadline, self.clock()+seconds*10**9)

    @contextmanager
    def io_scope(self, seconds: int = 120) -> Iterator[None]:
        """Borrow only this adapter's already tighter scope, never nest owner scopes."""
        deadline = self.deadline(seconds)
        if self._io_deadline is not None:
            if self._io_deadline > deadline:
                raise ValueError("cannot borrow a broader adapter IO deadline")
            yield
            return
        with self.files.stage_deadline(deadline):
            self._io_deadline = deadline
            try:
                yield
            finally:
                self._io_deadline = None

    @contextmanager
    def sink(self, path: Path) -> Iterator[Any]:
        # Only acquire/close carry preparation scopes. Writes occur within the
        # real owned tool control's child/stage deadline while its sink is live.
        with self.io_scope():
            stream = self.files.open_file(path, "xb")
            stream.__enter__()
        try:
            yield stream
        except BaseException as error:
            # The retained adapter uses the original failure's cleanup grace;
            # a stopped owner must not be re-entered for a fresh IO deadline.
            stream.__exit__(type(error), error, error.__traceback__)
            raise
        else:
            with self.io_scope():
                stream.__exit__(None, None, None)

    def space(self) -> None:
        with self.io_scope():
            disk = self.files.path_action("disk_usage", self.root)
        retained = sum(path.size_bytes or 0 for path in self.files.snapshot().paths
                       if path.owned and path.present)
        required = disk_budget(1920, 1080, (150, 120, 210), (True, False, True))["required"]
        admit(self.campaign.stage, free=disk[2], required=max(0, required-retained),
              children=self.campaign.children, seconds=(self.clock()-self.campaign.start)/1e9)

    def register(self, path: Path, role: str, cap: int, *, owned: bool = True) -> None:
        self.files.register(path, role=role, owned=owned, max_bytes=cap)
        for parent in path.parents:
            self.files._register_checked(parent)

    def put(self, leaf: str, raw: bytes, *, role: str | None = None, cap: int = 16*MiB,
            retain: bool = False) -> Path:
        if len(raw) > cap:
            raise ValueError("fixture leaf byte ceiling")
        path = self.root/safe_relative(leaf)
        if role is None:
            role = "sidecar" if retain and path.suffix == ".json" else "temporary"
        if retain and role not in ("candidate", "sidecar", "graph", "ass", "asset", "font"):
            raise ValueError("retained fixture purpose must be declared transferable before creation")
        self.register(path, role, cap)
        with self.io_scope(), self.files.open_file(path, "xb") as stream:
            view = memoryview(raw)
            try:
                offset = 0
                while offset < len(view):
                    written = stream.write(view[offset:offset+65536])
                    if type(written) is not int or written <= 0:
                        raise ValueError("uncertain fixture write")
                    offset += written
                stream.flush()
            finally:
                view.release()
        if retain:
            self.keep.append(path)
        return path

    def read(self, path: Path, cap: int) -> bytes:
        with self.files.stage_deadline(self.deadline(120)), self.files.open_binary(path) as stream:
            output = bytearray()
            while True:
                raw = stream.read(min(MiB, cap+1-len(output)))
                if not raw:
                    break
                output.extend(raw)
                if len(output) > cap:
                    raise ValueError("retained fixture read cap")
        return bytes(output)

    def run(self, argv: Sequence[str], *, kind: str = "encoder", inputs: tuple[Path, ...] = (),
            outputs: tuple[Path, ...] = (), expected: int = 0, sink: Any = None,
            timeout: int | None = None) -> Any:
        result = self.tools.run(tuple(argv), kind=kind,
            timeout={"probe": 45, "spatial": 900, "finalmix": 120, "encoder": 900}[kind] if timeout is None else timeout,
            max_stdout_bytes=1_000_000 if kind == "probe" else max(1, expected),
            expected_stdout_bytes=None if kind == "probe" else expected,
            max_stderr_bytes=16*MiB, sink=sink, inputs=inputs, outputs=outputs)
        status = self.files.snapshot()
        admit(self.campaign.stage, paths=len(status.paths), generations=status.generations,
              live=len(status.rows), children=self.campaign.children,
              seconds=(self.clock()-self.campaign.start)/1e9)
        return result

    def probe(self, path: Path, *options: str) -> dict[str, Any]:
        result = self.run((str(self.runtime.ffprobe), "-v", "error", *options,
                           "-of", "json", str(path)), kind="probe", inputs=(path,))
        return cast(dict[str, Any], json.loads(result.stdout))

    def file_output(self, leaf: str, argv: Sequence[str], inputs: tuple[Path, ...],
                    *, cap: int = 256*MiB, retain: bool = False,
                    timeout: int = 900) -> tuple[Path, Any]:
        path = self.root/leaf
        # Explicit handoff intent plus a known media purpose is classified before
        # registration/producer admission. Negative, PCM and other scratch stay
        # ephemeral; no existing row is relabelled to obtain a transfer.
        role = "asset" if retain and path.suffix.lower() in (".mov", ".mp4", ".png") else "temporary"
        if retain and role == "temporary":
            raise ValueError("retained file-output purpose must be MOV/MP4 fixture or diagnostic PNG")
        self.register(path, role, cap)
        result = self.run((*argv, str(path)), inputs=inputs, outputs=(path,), timeout=timeout)
        with self.files.stage_deadline(self.deadline(120)):
            size = self.files.stat_path(path).st_size
        if not 0 < size <= cap:
            raise ValueError("fixture compressed/file-output byte ceiling")
        self.generated_bytes += size
        if self.generated_bytes > (128*MiB if leaf.endswith(".png") else 256*MiB):
            raise ValueError("aggregate generated fixture/image byte ceiling")
        if retain:
            self.keep.append(path)
        return path, result

    def raw_output(self, leaf: str, argv: Sequence[str], expected: int,
                   inputs: tuple[Path, ...]) -> Path:
        path = self.root/leaf
        self.register(path, "raw", expected)
        with self.sink(path) as sink:
            self.run(argv, kind="spatial", expected=expected, sink=sink, inputs=inputs)
        return path

    def make_tools(self) -> Any:
        from clipperx._render_services import OwnedRenderTools
        session = self
        class SessionTools(OwnedRenderTools):
            def run(self, argv: tuple[str, ...], **kwargs: Any) -> Any:
                session.space()
                began = session.clock()
                session.state["active_command"] = dict(argv=argv, kind=kwargs["kind"], start_ns=began)
                result = super().run(argv, **kwargs)
                session.state["active_command"] = None
                record = dict(kind=result.kind, elapsed_ns=session.clock()-began,
                    result=result, stderr_tail_text=result.stderr_tail.decode(errors="replace"),
                    scope="child retirement while driver alive; no global own0 assertion")
                session.records.append(record)
                session.put(f"retirement{len(session.records):03}.json", json_bytes(record),
                            role="sidecar", cap=8*MiB, retain=True)
                return result
        return SessionTools(self.job, self.files, clock=self.clock)

    def source_probe(self, source: Path) -> Any:
        from clipperx import source_decode as sd
        with self.io_scope():
            return sd.probe_source(source, replace(self.runtime, device_index=None),
                                   _files=self.files, _runner=self.tools)

    def paced_cancellation(self, paced: Path) -> NoReturn:
        sink_path = self.root/"cancel.rgba"
        self.register(sink_path, "raw", 120*16*16*4)
        with self.sink(sink_path) as sink:
            self.cancel_ready = True
            self.run((str(self.runtime.ffmpeg), "-hide_banner", "-nostats", "-v", "error", "-readrate", "1",
                "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", "16x16", "-framerate", "60",
                "-i", str(paced), "-an", "-c:v", "rawvideo", "-pix_fmt", "rgba", "-f", "rawvideo", "pipe:1"),
                kind="spatial", expected=120*16*16*4, sink=sink, inputs=(paced,))
        raise ValueError("missed finite cancellation window; natural EOF is not cancellation evidence")

    def run_driver(self, work: Callable[[], Any]) -> None:
        try:
            self.state["result"] = work()
        except Exception as error:
            self.state["error"] = error
            self.state["driver_error"] = error_details(error)
            if self.job.snapshot().failure is None:
                self.job.fail("validation", "driver")
        finally:
            try:
                self._retiring = True
                self.state["retirement"] = self.files.retire(retain=tuple(self.keep))
            except Exception as error:
                self.state["cleanup_error"] = error_details(error)
                self.state.setdefault("cleanup_exception", error)
                if self.job.snapshot().failure is None:
                    self.job.fail("validation", "driver")
            finally:
                self._retiring = False
                self.job.driver_done()

    def record_path_action(self, action: Callable[[str, Path, Path | None], object]
                           ) -> Callable[[str, Path, Path | None], object]:
        def recorded(operation: str, path: Path, destination: Path | None) -> object:
            try:
                return action(operation, path, destination)
            except Exception as error:
                # Observe only real retirement deletions. Expected missing stat
                # checks remain the original retained adapter's own contract.
                if self._retiring and operation in ("unlink", "rmdir"):
                    self.state.setdefault("cleanup_exception", error)
                    count = min(32769, self.state.get("cleanup_action_count", 0)+1)
                    self.state["cleanup_action_count"] = count
                    records = self.state.setdefault("cleanup_actions", [])
                    if len(records) < 8:
                        errno, winerror = getattr(error, "errno", None), getattr(error, "winerror", None)
                        records.append(dict(operation=operation, path=str(path)[:1024],
                            errno=errno if type(errno) is int and -2**31 <= errno < 2**32 else None,
                            winerror=winerror if type(winerror) is int and -2**31 <= winerror < 2**32 else None,
                            error=error_details(error)))
                raise
        return recorded

    def finish_result(self, *, cancel: bool) -> Any:
        from clipperx._source_host_ownership import HostedObservationFailure
        status = self.job.snapshot()
        files = self.files.snapshot()
        error = self.state.get("error")
        window, terminal = self.state.get("cancellation_window"), status.current
        incomplete_payload = False
        if (window is not None and terminal is not None
                and window.observation_id == terminal.observation_id
                and window.kind == terminal.kind == "spatial"
                and len(window.diagnostics) == len(terminal.diagnostics) == 1):
            captured, final = window.diagnostics[0], terminal.diagnostics[0]
            # Conservative eligibility for this finite control, not atomic EOF proof.
            incomplete_payload = (captured.kind == final.kind == "spatial"
                and captured.generation == final.generation and captured.exit_code is None
                and not captured.natural_eof and not captured.transfer_uncertain
                and captured.stdout_bytes < 120*16*16*4 and final.stdout_bytes < 120*16*16*4
                and not final.natural_eof)
        hosted_cancel = isinstance(error, HostedObservationFailure) and error.failure.code == "cancelled"
        expected_cancel = (cancel and self.cancel_ready and self.cancel_observed
                           and hosted_cancel and incomplete_payload)
        classification = ("not_requested" if not cancel else "not_reached" if not self.cancel_ready else
                          "window_not_observed" if not self.cancel_observed else
                          "observed_cancelled" if expected_cancel else
                          "ineligible_window" if hosted_cancel else "unexpected_cause")
        self.campaign.records.append(dict(scope="completed fixture job controller observation",
            owner=status, files=files, records=self.records, active_command=self.state.get("active_command"),
            driver_error=self.state.get("driver_error"), cleanup_error=self.state.get("cleanup_error"),
            cleanup_actions=self.state.get("cleanup_actions", []),
            cleanup_actions_truncated=self.state.get("cleanup_action_count", 0) > 8,
            cancellation=dict(requested=cancel, ready=self.cancel_ready, observed=self.cancel_observed,
                              classification=classification, window=window, terminal=terminal,
                              incomplete_payload=incomplete_payload),
            last_diagnostic_tails=[] if status.current is None else [
                d.stderr_tail.decode(errors="replace") for d in status.current.diagnostics]))
        if error is not None and not expected_cancel:
            raise QualificationFailure(error, self.campaign.records) from error
        cleanup = self.state.get("cleanup_exception")
        if cleanup is not None:
            raise QualificationFailure(cleanup, self.campaign.records) from cleanup
        if cancel:
            if not expected_cancel:
                raise QualificationFailure(ValueError(f"cancellation {classification}"), self.campaign.records)
            try:
                cancelled_result(admitted=self.cancel_observed, acquired=self.cancel_ready,
                    before_eof=incomplete_payload, stop_code="" if status.failure is None else status.failure.code,
                    own0=status.own0, files_clear=files.files_clear)
            except ValueError as uncertain:
                raise QualificationFailure(uncertain, self.campaign.records) from uncertain
        elif status.failure is not None or not status.own0 or not files.files_clear:
            raise QualificationFailure(ValueError("fixture total retirement failure"),
                                       self.campaign.records)
        return self.state.get("result")

    def _observe_paced_cancel(self, status: "SourceJobStatus") -> None:
        current = status.current
        if (not self.cancel_ready or self.cancel_observed or status.failure is not None
                or current is None or current.kind != "spatial" or current.phase != "active"
                or not current.scope_open or not current.consumer_active or len(current.diagnostics) != 1):
            return
        diagnostic = current.diagnostics[0]
        if (diagnostic.kind != "spatial" or diagnostic.exit_code is not None
                or diagnostic.natural_eof or diagnostic.transfer_uncertain
                or diagnostic.stdout_bytes >= 120*16*16*4):
            return
        admitted = any(row.role == "process" and row.generation == diagnostic.generation
                       and row.pid is not None and row.phase in ("live", "in_call") and not row.uncertain
                       for row in current.resources)
        if admitted:
            # Admission metadata is detached owner evidence, not an OS liveness probe.
            self.state["cancellation_window"] = current
            self.cancel_observed = True
            self.job.request_stop("cancelled")

    def execute(self, action: Callable[["FixtureSession"], Any], *, cancel: bool = False) -> Any:
        from clipperx._source_host_ownership import Process, SourceHostJob
        from clipperx._retained_files import RetainedFiles, _path_action
        from clipperx import render_runtime as rr, source_decode as sd

        def process(argv: tuple[str, ...]) -> Any:
            self.campaign.charge()
            return cast(Process, subprocess.Popen(list(argv), stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0, text=False,
                shell=False, close_fds=True, cwd=self.root))

        def driver_work() -> Any:
            self.job.register_current_driver()
            self.register(self.root, "directory", 0)
            with self.files.stage_deadline(self.deadline(120)):
                self.files.path_action("mkdir", self.root)
            for row in self.manifest["external"].values():
                self.register(Path(row["path"]), "original", 256*MiB, owned=False)
            requirements = rr.combine_render_requirements(sd.decode_requirements(),
                rr.RenderRequirements("ffmpeg-9.0.2-libplacebo", needs_vulkan=True,
                                      required_filters=("libplacebo",)))
            with self.files.stage_deadline(self.deadline(120)):
                self.runtime = rr.resolve_render_runtime(requirements, device_index=0,
                                                         _files=self.files)
                disk = self.files.path_action("disk_usage", self.root)
            provision = disk_budget(1920, 1080, (150, 120, 210), (True, False, True))
            admit(self.campaign.stage, free=disk[2], required=provision["required"],
                  paths=512, generations=32768, live=3)
            if (str(self.runtime.ffmpeg) != self.manifest["external"]["ffmpeg"]["path"]
                    or str(self.runtime.ffprobe) != self.manifest["external"]["ffprobe"]["path"]):
                raise ValueError("actual runtime selector differs from manifest paths")
            return action(self)

        def driver_main() -> None:
            self.run_driver(driver_work)

        driver = threading.Thread(target=driver_main, name="native57-fixture-driver", daemon=False)
        wall = 2700 if self.campaign.stage == "prerequisites" else 3600
        self.job = SourceHostJob(driver, deadline_ns=min(self.clock()+7200*10**9,
                                 self.campaign.start+wall*10**9),
                                 clock=self.clock, process_factory=process)
        self.files = RetainedFiles(self.job, root=self.root.parent, clock=self.clock,
                                   _path_action=self.record_path_action(_path_action))
        self.tools = self.make_tools()
        self.job.start_driver()
        while True:
            status = self.job.service_cleanup()
            if cancel:
                self._observe_paced_cancel(status)
            if status.own0:
                break
            if status.failure is not None and self.clock() >= status.failure.cleanup_deadline_ns:
                cause = self.state.get("error", self.state.get("cleanup_exception",
                    ValueError("cleanup grace exhausted; custody retained")))
                raise QualificationFailure(cause, dict(owner=status, files=self.files.snapshot(),
                    records=self.records, driver_error=self.state.get("driver_error"),
                    cleanup_error=self.state.get("cleanup_error"), cleanup_actions=self.state.get("cleanup_actions", []),
                    scope="controller cleanup deadline observation; no final clearance asserted")) from cause
            driver.join(.01)
        return self.finish_result(cancel=cancel)


def assets_for(manifest: dict[str, Any], brand: Path, cue: Path | None = None) -> Any:
    from clipperx.render_assets import RenderAssetRegistry
    return RenderAssetRegistry({"alpha_brand": brand},
        {key: Path(manifest["external"][key]["path"]) for key in ("headline_impact", "subtitle_bold")},
        {} if cue is None else {"impact": cue})


def fixture_float(s: FixtureSession, rate: int, channels: int) -> tuple[Path, dict[str, Any]]:
    from clipperx import source_decode as sd
    from clipperx.audio_selection import (AudioSelectionLimits, plan_float_audio_selection,
                                        map_float_audio_group, apply_float_audio_group)
    name = f"float{rate}-{channels}"
    raw = recipe.float48_pcm(channels) if rate == 48000 else recipe.float44_pcm()
    pcm = s.put(name+".f32", raw, role="pcm")
    video = s.put(name+".rgb", recipe.tiny_rgb(120 if rate == 48000 else 180), role="raw")
    prepare_aac = rate == 44100
    source, _ = s.file_output(name+(".zero.mov" if prepare_aac else ".mov"), (str(s.runtime.ffmpeg), "-hide_banner", "-nostats",
        "-loglevel", "info", "-n", "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", "16x16",
        "-framerate", "60", "-i", str(video), "-f", "f32le", "-ar", str(rate), "-ac", str(channels),
        "-i", str(pcm), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "qtrle", "-c:a",
        "pcm_f32le" if rate == 48000 else "aac", "-output_ts_offset", "0" if prepare_aac else "1.25", "-use_editlist", "1",
        "-avoid_negative_ts", "disabled"), (pcm, video), cap=16*MiB if prepare_aac else 256*MiB,
        retain=prepare_aac or (channels == 2 and rate == 48000))
    if prepare_aac:
        shifted, preparation = recipe.prepare_aac_fixture_mov(s.read(source, 16*MiB))
        source = s.put(name+".mov", shifted, role="asset", cap=16*MiB, retain=True)
        s.put(name+".preparation.json", json_bytes(preparation), role="sidecar", cap=MiB, retain=True)
    cpu = replace(s.runtime, device_index=None)
    origin = s.source_probe(source)
    s.put(name+".origin.json", json_bytes(origin.record()), role="sidecar", cap=MiB, retain=True)
    video_clock = next(c for c in origin.streams if c.kind == "video")
    audio = next(c for c in origin.streams if c.kind == "audio")
    if (len(origin.streams) != 2 or origin.origin != F(5, 4)
            or audio.sample_format != ("flt" if rate == 48000 else "fltp")
            or audio.sample_rate != rate or audio.channels != channels
            or audio.start != F(5, 4) or video_clock.start != F(5, 4)
            or audio.end is None or video_clock.end is None):
        raise ValueError("UNVERIFIED MOV recipe failed actual format/O/start metadata; stop without substitution")
    plan = recipe.float_plan(origin.fingerprint, rate=rate)
    limits, decode = AudioSelectionLimits(), sd.FloatAudioDecodeLimits()
    record = sd._runtime_record(cpu)
    planning = plan_float_audio_selection(plan, origin=origin, runtime=record, limits=limits)
    group = planning.groups[0]
    reference = s.root/(name+".reference.f32")
    s.register(reference, "pcm", 16*MiB)
    control = s.run(recipe.reference_argv(str(s.runtime.ffmpeg), str(source), str(reference),
        channels=channels, input_rate=rate), inputs=(source,), outputs=(reference,))
    ref = s.read(reference, 16*MiB)
    grid, count = oracle.reference_tags(control.stderr_tail, truncated=control.stderr_truncated,
                                        channels=channels, pcm=ref)
    if rate == 48000:
        oracle.bit_identity(ref, raw, channels=channels, samples=96000)
    s.space()
    observed = sd.observe_float_audio_window(cpu, origin, sd.Window(group.source_start, group.source_end),
        audio_policy="render48k-f32-"+("mono" if channels == 1 else "stereo")+"-v1", limits=decode,
        _host=s.job.new_observation("materialized_f32", deadline_ns=s.deadline(decode.timeout)),
        _runtime_files=s.files)
    observer_control = s.job.snapshot().current
    if observer_control is None or observer_control.phase != "retired_ok" or not observer_control.window_resources_zero:
        raise ValueError("float observer not certainly retired")
    mapped = map_float_audio_group(plan, 0, observed.receipt, origin=origin, runtime=record,
                                  decode_limits=decode, limits=limits)
    selected = apply_float_audio_group(plan, mapped, observed, origin=origin, runtime=record,
                                      decode_limits=decode, limits=limits)
    receipt = observed.receipt.record()
    observed_audio = receipt["payload"]["audio"][0]
    observed_grid = F(observed_audio["sample_grid_origin"])
    offset = oracle.grid_offset(grid, observed_grid)
    indices = oracle.reference_indices(origin.origin, group.source_start, group.start,
                                       group.first, group.last-group.first, grid)
    expected = oracle.indexed_pcm(ref, indices, channels)
    selected_samples = 36000 if rate == 48000 else 33600
    oracle.finite_pcm(selected.pcm, channels, selected_samples)
    whole_expected = oracle.indexed_pcm(ref, tuple(range(offset, offset+len(observed.pcm)//(4*channels))), channels)
    stride = 4*channels
    first_selected = indices[0]-offset
    last_selected = indices[-1]-offset+1
    observer_regions = dict(
        prefix_samples=first_selected, suffix_samples=len(observed.pcm)//stride-last_selected,
        full_errors=oracle.numeric_errors(observed.pcm, whole_expected, channels=channels),
        prefix_errors=oracle.numeric_errors(observed.pcm[:first_selected*stride],
            whole_expected[:first_selected*stride], channels=channels),
        suffix_errors=oracle.numeric_errors(observed.pcm[last_selected*stride:],
            whole_expected[last_selected*stride:], channels=channels))
    if rate == 48000:
        oracle.bit_identity(observed.pcm, whole_expected, channels=channels,
                            samples=len(observed.pcm)//(4*channels))
        oracle.bit_identity(selected.pcm, expected, channels=channels, samples=selected_samples)
        errors: Any = {"bit_identity": True}
    else:
        errors = oracle.phase_envelope(selected.pcm, ref, indices, channels=channels)
        centers = (origin.origin+F(11, 10), origin.origin+F(8, 5))
        ref_events = oracle.burst_centers(ref, channels, centers, grid=grid)
        selected_grid = origin.origin+group.source_start
        obs_events = oracle.burst_centers(selected.pcm, channels, centers, grid=selected_grid)
        for wanted, actual in zip(ref_events, obs_events):
            oracle.events(wanted, actual, F(1, 48000))
    facts = dict(recipe=recipe.RECIPE_VERSION, origin=origin.record(), reference_grid=str(grid),
        observer_grid=str(observed_grid), reference_count=count, reference_sha256=hashlib.sha256(ref).hexdigest(),
        receipt=receipt, mapping_sha256=mapped.sha256, selected_sha256=selected.pcm_sha256,
        indices_first=indices[0], indices_last=indices[-1], selected_samples=selected_samples,
        errors=errors, reference_control="encoder-kind seekable file-output, not H264 delivery encode")
    facts["observer_control"] = observer_control
    facts["full_observer_regions"] = observer_regions
    s.state[name] = facts
    s.put(name+".oracle.json", json_bytes(facts), role="sidecar", cap=8*MiB, retain=True)
    return source, facts


def tiny_component(s: FixtureSession, source: Path, brand: Path) -> dict[str, Any]:
    from clipperx import font_staging
    from clipperx.local_paths import source_fingerprint
    from clipperx.overlay_ass import compile_overlay_ass
    from clipperx.visual_compiler import compile_visual_assembly
    from clipperx._visual_render_video import serialize_shot_graph

    with s.files.stage_deadline(s.deadline(120)):
        fingerprint = source_fingerprint(source, _files=s.files)
    plan = recipe.tiny_plan(fingerprint)
    fonts = s.root/"fonts"
    s.register(fonts, "directory", 0)
    with s.files.stage_deadline(s.deadline(120)):
        stage = font_staging.stage_original_fonts(assets_for(s.manifest, brand), stage_dir=fonts, _files=s.files)
        overlays = compile_overlay_ass(plan, None, assets=stage.registry,
                                        brand_cache_dir=s.root, _files=s.files)
        assembly = compile_visual_assembly(plan, source_size=(16, 16), overlays=overlays)
    ass = {document.key: s.put(f"ass{i:02}.ass", document.document.encode("utf-8-sig"), role="ass")
           for i, document in enumerate(assembly.ass_documents)}
    decoded = s.raw_output("tiny-source.bgr", (str(s.runtime.ffmpeg), "-hide_banner", "-nostats", "-v", "error",
        "-i", str(source), "-map", "0:v:0", "-an", "-pix_fmt", "bgr24", "-fps_mode", "passthrough",
        "-f", "rawvideo", "pipe:1"), 16*16*3*120, (source,))
    bgr = s.read(decoded, MiB)
    rgba = []
    for i, item in enumerate(assembly.shots):
        argv = [str(s.runtime.ffmpeg), "-hide_banner", "-nostats", "-loglevel", "info", *s.runtime.device_arguments]
        inputs = []
        if item.transport is not None:
            ordinals = (0, 1) if i == 0 else (8, 9)
            raw = b"".join(bgr[n*768:(n+1)*768] for n in ordinals)
            inlet = s.put(f"shot{i}.bgr", raw, role="raw")
            argv.extend(("-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", "16x16",
                         "-framerate", "60", "-i", str(inlet)))
            inputs.append(inlet)
        brands = tuple(b for b in assembly.brand_inputs if b.shot_id == item.span.shot_id)
        for image in brands:
            argv.extend(("-i", str(image.asset.path)))
            inputs.append(image.asset.path)
        graph = serialize_shot_graph(item, brands, {key: path.name for key, path in ass.items()})
        graph_path = s.put(f"graph{i}.txt", graph.encode(), role="graph")
        inputs.extend((graph_path, *ass.values(), *(f.staged_path for f in stage.fonts)))
        argv.extend((s.runtime.graph_file_option, graph_path.name, "-map", f"[{item.output_label}]", "-an",
                     "-fps_mode", "passthrough", "-pix_fmt", "rgba", "-c:v", "rawvideo", "-f", "rawvideo", "pipe:1"))
        rgba.append(s.raw_output(f"shot{i}.rgba", argv, 8294400*2, tuple(inputs)))
    listing = s.put("shots.list", "".join(p.name+"\n" for p in rgba).encode("ascii"), role="graph")
    # Physical concatf readback discards its stdout after exact owned count/hash;
    # no second concatenated RAW copy. Pixel checks stream original finite spools.
    concat = s.run((str(s.runtime.ffmpeg), "-hide_banner", "-nostats", "-v", "error", "-f", "rawvideo",
        "-pixel_format", "rgba", "-video_size", "1080x1920", "-framerate", "60", "-protocol_whitelist",
        "file,concatf", "-i", "concatf:shots.list", "-an", "-c:v", "copy", "-f", "rawvideo", "pipe:1"),
        kind="spatial", expected=49766400, inputs=(*rgba, listing))
    pixels = []
    digest = hashlib.sha256()
    for i, path in enumerate(rgba):
        with s.files.stage_deadline(s.deadline(900)), s.files.open_binary(path) as stream:
            for local in range(2):
                frame = bytearray()
                while len(frame) < 8294400:
                    raw = stream.read(min(MiB, 8294400-len(frame)))
                    if not raw:
                        raise ValueError("short component RGBA frame")
                    frame.extend(raw)
                digest.update(frame)
                ordinal = recipe.tiny_frame_ordinals()[i*2+local]
                expected = (32, 48, 64) if ordinal is None else recipe.tiny_color(ordinal)
                def pixel(x: int, y: int) -> tuple[int, ...]:
                    j = (y*1080+x)*4
                    return tuple(frame[j:j+4])
                oracle.color(pixel(500, 1000), expected, tolerance=2,
                             context=f"K={i*2+local} x=500 y=1000")
                if any(alpha != 255 for alpha in frame[3::4]):
                    raise ValueError("nonopaque component canvas alpha")
                if ordinal is not None and oracle.barcode(
                        [pixel(120*(bit+1), 1500) for bit in range(8)], bits=8) != ordinal:
                    raise ValueError("tiny source ordinal barcode mismatch")
                for x, y, fore in ((702, 714, (255, 0, 0, 255)),
                                   (810, 714, (0, 255, 0, 128)),
                                   (702, 822, (0, 0, 255, 0)),
                                   (810, 822, (255, 255, 255, 255))):
                    # 32-square image is aspect-preserving inside .2x.2 rect:
                    # x648..864/y660..876, not a stretched 216x384 image.
                    oracle.color(pixel(x, y), oracle.blend(fore, expected) if i == 2 else expected,
                                 tolerance=2, context=f"K={i*2+local} x={x} y={y}")
                ink = any(max(pixel(x, y)[:3])-min(pixel(x, y)[:3]) < 20
                          and min(pixel(x, y)[:3]) > 180
                          for y in range(90, 290, 2) for x in range(120, 960, 2))
                if not ink:
                    raise ValueError("scheduled title has no actual glyph ink")
                pixels.append(dict(k=i*2+local, source_ordinal=ordinal, rgba_sha256=hashlib.sha256(frame).hexdigest()))
            if stream.read(1):
                raise ValueError("extra component RGBA byte")
    if digest.hexdigest() != concat.stdout_sha256:
        raise ValueError("physical concatf order/hash mismatch")
    candidate, _ = s.file_output("fs-negative.mp4", (str(s.runtime.ffmpeg), "-hide_banner", "-nostats",
        "-v", "error", "-n", "-f", "rawvideo", "-pixel_format", "rgba", "-video_size", "1080x1920",
        "-framerate", "60", "-protocol_whitelist", "file,concatf", "-i", "concatf:shots.list",
        "-an", "-c:v", "h264_nvenc", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-fs", "1024"),
        (*rgba, listing), cap=4*GiB)
    with s.files.stage_deadline(s.deadline(120)):
        size = s.files.stat_path(candidate).st_size
    meta = s.probe(candidate, "-count_frames", "-select_streams", "v:0", "-show_streams")
    frames = int(meta["streams"][0]["nb_read_frames"])
    if not fs_rejected(size, frames):
        raise ValueError("-fs1024 expected-negative candidate unexpectedly eligible")
    # This finite paced input is the ONE cancellation control. It uses an already
    # prepared source spool and keeps the sink claimed while controller observes admission.
    paced = s.put("paced.bgr", bgr, role="raw")
    s.state["component"] = dict(pixels=pixels, concat=concat, fs=dict(size=size, natural_frames=frames),
        fonts_selected_file_identity="unverified", scope="component transport; not public B qualification")
    s.put("component.oracle.json", json_bytes(s.state["component"]), role="sidecar", cap=8*MiB, retain=True)
    s.paced_cancellation(paced)


def prepare_prerequisites(s: FixtureSession) -> None:
    fixture_float(s, 48000, 1)
    source, _ = fixture_float(s, 48000, 2)
    fixture_float(s, 44100, 2)
    brand = s.put("alpha.png", recipe.alpha_png(), role="asset", retain=True)
    from clipperx.local_paths import source_fingerprint
    with s.files.stage_deadline(s.deadline(120)):
        fingerprint = source_fingerprint(source, _files=s.files)
    s.put("tiny.json", recipe.tiny_plan(fingerprint).model_dump_json().encode(), role="sidecar", retain=True)
    s.put("solid.json", recipe.tiny_plan(fingerprint, all_solid=True).model_dump_json().encode(), role="sidecar", retain=True)
    tiny_component(s, source, brand)


def prepare_synthetic(s: FixtureSession) -> dict[str, Path]:
    from clipperx.local_paths import source_fingerprint
    pcm = s.put("synthetic.f32", recipe.tonal_pcm(48000, 12, recipe.SOURCE_PULSES), role="pcm")
    brand = s.put("alpha.png", recipe.alpha_png(), role="asset", retain=True)
    # Finite lavfi color plus integer-N marker/barcodes; no full source RAW spool.
    graph = ["[0:v]drawbox=x=0:y=0:w=480:h=270:color=0xC04020:t=fill[base]",
             "[1:v]format=rgba[marker]",
             "[base][marker]overlay=x='960+mod((n-1)*7,200)-16':y='400+mod((n-1)*3,240)-16':eval=frame[v0]"]
    label = "v0"
    for row in range(3):
        for bit in range(10):
            next_label = f"v{row*10+bit+1}"
            graph.append(f"[{label}]drawbox=x={1000+bit*40}:y={300+row*240}:w=28:h=28:"
                f"color=white:t=fill:enable='bitand(n,{1<<bit})'[{next_label}]")
            label = next_label
    for bit in range(10):
        next_label = f"c{bit}"
        graph.append(f"[{label}]drawbox=x={20+bit*40}:y=180:w=28:h=28:"
                     f"color=white:t=fill:enable='bitand(n,{1<<bit})'[{next_label}]")
        label = next_label
    for i, center in enumerate(recipe.SOURCE_PULSES):
        next_label = f"flash{i}"
        graph.append(f"[{label}]drawbox=x=360:y=210:w=80:h=40:color=white:t=fill:"
                     f"enable='eq(n,{int(center*60)})'[{next_label}]")
        label = next_label
    font = Path(s.manifest["external"]["headline_impact"]["path"])
    # Use staged font bytes under the registered fixture driver, then drawtext by
    # a relative fixed ASCII path so no colon/drive parsing escapes are needed.
    font_path = s.put("fixture.ttf", s.read(font, 16*MiB), role="font")
    graph.append(f"[{label}]drawtext=fontfile=fixture.ttf:text='CAMARA %{{n}} %{{pts}}':"
                 "x=12:y=20:fontsize=36:fontcolor=white[v]")
    graph_path = s.put("synthetic.txt", ";".join(graph).encode(), role="graph")
    source, _ = s.file_output("synthetic.mp4", (str(s.runtime.ffmpeg), "-hide_banner", "-nostats", "-v", "info",
        "-n", "-f", "lavfi", "-i", "color=c=0x204080:s=1920x1080:r=60:d=12",
        "-f", "lavfi", "-i", "color=c=0xFFFF00:s=32x32:r=60:d=12", "-f", "f32le", "-ar", "48000",
        "-ac", "2", "-i", str(pcm), s.runtime.graph_file_option, graph_path.name,
        "-map", "[v]", "-map", "2:a:0", "-c:v", "h264_nvenc", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "48000", "-movflags", "+faststart"), (pcm, graph_path, font_path), retain=True)
    # The fixed cue declares its IEEE-float stereo layout without inferring it.
    cue = s.put("impact.wav", recipe.impact_wav(), role="asset", retain=True)
    source_meta = s.probe(source, "-count_frames", "-show_streams")
    video_stream = next(row for row in source_meta["streams"] if row["codec_type"] == "video")
    if int(video_stream["nb_read_frames"]) != 720 or video_stream["width"] != 1920 or video_stream["height"] != 1080:
        raise ValueError("synthetic fixture source is not natural720/1920x1080")
    with s.files.stage_deadline(s.deadline(120)):
        plan = recipe.synthetic_plan(source_fingerprint(source, _files=s.files))
    plan_path = s.put("synthetic.json", plan.model_dump_json().encode(), role="sidecar", retain=True)
    transcript = s.put("transcript.json", json_bytes(recipe.transcript()), role="sidecar", retain=True)
    return dict(source=source, brand=brand, cue=cue, plan=plan_path, transcript=transcript)


def verify_delivery(s: FixtureSession, candidate: Path, expected_frames: int,
                    diagnostic_k: Sequence[int], *, synthetic: bool = False,
                    tiny: bool = False, all_solid: bool = False) -> dict[str, Any]:
    s.register(candidate, "original", 2*GiB, owned=False)
    metadata = s.probe(candidate, "-count_frames", "-show_streams", "-show_format")
    video = [row for row in metadata["streams"] if row["codec_type"] == "video"]
    audio = [row for row in metadata["streams"] if row["codec_type"] == "audio"]
    if (len(video) != 1 or len(audio) != 1 or video[0]["codec_name"] != "h264"
            or video[0]["width"] != 1080 or video[0]["height"] != 1920
            or video[0]["pix_fmt"] != "yuv420p" or audio[0]["codec_name"] != "aac"
            or int(audio[0]["sample_rate"]) != 48000
            or int(video[0]["nb_read_frames"]) != expected_frames):
        raise ValueError("independent delivery codec/format/natural count mismatch")
    pts = s.probe(candidate, "-select_streams", "v:0", "-show_frames", "-show_entries",
                  "frame=best_effort_timestamp_time")
    oracle.frame_grid([row["best_effort_timestamp_time"] for row in pts["frames"]], expected_frames)
    packets = s.probe(candidate, "-select_streams", "a:0", "-show_packets", "-show_entries",
        "packet=pts,duration:packet_side_data=side_data_type,skip_samples,discard_padding")
    s.put("delivery-aac-packets.json", json_bytes(packets), role="sidecar", cap=1_000_000, retain=True)
    rows = packets["packets"]
    skip = sum(int(side.get("skip_samples", 0)) for row in rows for side in row.get("side_data_list", []))
    padding = sum(int(side.get("discard_padding", 0)) for row in rows for side in row.get("side_data_list", []))
    presentation = oracle.aac_packet_presentation([row["pts"] for row in rows],
        [row["duration"] for row in rows], skip, padding, expected_frames*800)
    # Complete natural decode independently, still bounded by probe45s, not900s.
    s.run((str(s.runtime.ffmpeg), "-hide_banner", "-nostats", "-v", "error", "-xerror", "-i", str(candidate),
        "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"), kind="probe", inputs=(candidate,))
    # Stream atom headers/bodies through registered readback without materializing
    # a potentially 2GiB candidate in Python memory.
    moov = mdat = None
    with s.files.stage_deadline(s.deadline(900)), s.files.open_binary(candidate) as stream:
        size = stream.fstat().st_size
        if not 0 < size <= 2*GiB:
            raise ValueError("actual candidate cap")
        position = 0
        while position < size:
            head = stream.read(8)
            if len(head) != 8:
                raise ValueError("truncated MP4 header")
            atom_size, tag = struct.unpack(">I4s", head)
            consumed = 8
            if atom_size == 1:
                extension = stream.read(8)
                if len(extension) != 8:
                    raise ValueError("truncated extended atom")
                atom_size, consumed = struct.unpack(">Q", extension)[0], 16
            if atom_size == 0:
                atom_size = size-position
            if not consumed <= atom_size <= size-position:
                raise ValueError("MP4 atom bounds")
            if tag == b"moov":
                moov = position
            if tag == b"mdat" and mdat is None:
                mdat = position
            left = atom_size-consumed
            while left:
                raw = stream.read(min(MiB, left))
                if not raw:
                    raise ValueError("truncated MP4 atom body")
                left -= len(raw)
            position += atom_size
        if stream.read(1) or moov is None or mdat is None or moov >= mdat:
            raise ValueError("MP4 faststart/EOF")
    diagnostics = []
    if len(diagnostic_k) > 10:
        raise ValueError("diagnostic PNG count cap")
    for k in diagnostic_k:
        path, _ = s.file_output(f"frame{k:04}.png", (str(s.runtime.ffmpeg), "-hide_banner", "-nostats",
            "-v", "error", "-n", "-i", str(candidate), "-map", "0:v:0", "-an", "-vf",
            f"select=eq(n\\,{k})", "-fps_mode", "passthrough", "-c:v", "png", "-f", "image2"),
            (candidate,), cap=12*MiB, retain=True)
        diagnostics.append(path)
    numerical: Any = None
    if synthetic:
        numerical = synthetic_measurements(s, candidate, diagnostics)
    elif tiny:
        numerical = tiny_delivery_measurements(s, diagnostics, all_solid=all_solid)
    return dict(metadata=metadata, natural_frames=expected_frames, presentation=tuple(map(str, presentation)),
        aac_skip=skip, aac_discard_padding=padding, actual_size=size, moov=moov, mdat=mdat,
        diagnostics=diagnostics, numerical_oracles=numerical, pixel_oracle=None,
        pixel_oracle_reason="Root must inspect decoded diagnostic pixels/title glyph overlap/safe areas; unmeasured is not acceptance",
        selected_font_file_identity="unverified")


def tiny_delivery_measurements(s: FixtureSession, diagnostics: Sequence[Path], *,
                               all_solid: bool = False) -> list[dict[str, Any]]:
    from contextlib import closing
    from io import BytesIO
    from PIL import Image
    def observations() -> Generator[Any, None, None]:
        for path in diagnostics:
            k = int(path.stem.removeprefix("frame"))
            with Image.open(BytesIO(s.read(path, 12*MiB))) as decoded:
                if decoded.size != (1080, 1920):
                    raise ValueError("tiny delivered diagnostic dimensions")
                image = decoded.convert("RGB")
            try:
                yield k, lambda x, y: image.getpixel((x, y))
            finally:
                image.close()
    with closing(observations()) as measured:
        return oracle.tiny_encoded_pixels(measured, all_solid=all_solid)


def overlay_crop_graph(batch: recipe.OverlayBoundaryBatch) -> str:
    x, y, width, height = batch.region
    selection = "+".join(f"eq(n\\,{k})" for k in batch.frames)
    return (f"select={selection},crop={width}:{height}:{x}:{y},format=rgb24,showinfo@qoverlay,"
            "tile=layout=6x1:nb_frames=6:padding=0:margin=0")


def overlay_boundary_measurements(s: FixtureSession, candidate: Path) -> list[dict[str, Any]]:
    from io import BytesIO
    from PIL import Image
    batches = []
    for batch in recipe.overlay_boundary_batches():
        # One seekable PNG FILE-output control for six decoded region cells;
        # stdout0/cap1, not a delivery encode, no added full-frame PNG or repair.
        path, result = s.file_output(batch.name+"-boundaries.png", (str(s.runtime.ffmpeg),
            "-hide_banner", "-nostats", "-loglevel", "info", "-n", "-i", str(candidate),
            "-map", "0:v:0", "-an", "-vf", overlay_crop_graph(batch),
            "-fps_mode", "passthrough", "-c:v", "png", "-pix_fmt", "rgb24", "-f", "image2"),
            (candidate,), cap=4*MiB, retain=True, timeout=45)
        with Image.open(BytesIO(s.read(path, 4*MiB))) as decoded:
            expected_size = batch.region[2]*6, batch.region[3]
            if decoded.size != expected_size:
                raise ValueError("overlay six-cell PNG dimensions/count")
            with decoded.convert("RGB") as image:
                pixels = image.tobytes()
        observations = oracle.overlay_contact_sheet(batch.name, pixels, *expected_size,
            result.stderr_tail, truncated=result.stderr_truncated)
        if [row["expected_active"] for row in observations] != list(batch.active):
            raise ValueError("authored overlay recipe/independent label association mismatch")
        batches.append(dict(sheet=path, observations=observations,
            control="encoder-kind PNG file-output, not H264 delivery encode", deadline_seconds=45,
            scope="30 bounded decoded boundary crops; glyph/finish acceptance still Root-owned"))
    return batches


def oracle_child_reservation(scenario: str, full_pngs: int) -> int:
    if type(full_pngs) is not int or not 0 <= full_pngs <= 10:
        raise ValueError("ten full diagnostic PNG ceiling")
    # Metadata, video PTS, AAC packets, complete natural content decode.
    base = 4+full_pngs
    # Synthetic adds one audio FILE-output reference, two flash patch probes
    # and exactly five six-cell overlay crop FILE-output controls.
    return base+8 if scenario == "synthetic" else base


def synthetic_measurements(s: FixtureSession, candidate: Path, diagnostics: Sequence[Path]
                           ) -> dict[str, Any]:
    """Read actual decoded pixels/audio; authored inverse geometry is independent."""
    from io import BytesIO
    from PIL import Image
    marker_records = []
    for path in diagnostics:
        k = int(path.stem.removeprefix("frame"))
        with Image.open(BytesIO(s.read(path, 12*MiB))) as decoded:
            image = decoded.convert("RGB")
        def rgb(x: int, y: int) -> tuple[int, int, int]:
            pixel = image.getpixel((x, y))
            if not isinstance(pixel, tuple) or len(pixel) != 3:
                raise ValueError("decoded RGB pixel shape")
            return cast(tuple[int, int, int], pixel)
        ordinal = recipe.synthetic_source_ordinal(k)
        if ordinal is None:
            continue
        if k < 150:
            crop = recipe.synthetic_crop(k)
            x0, y0 = float(crop[0]*1920), 270.
            scale, offset_y = 1.125, 656.25
            pixels = [rgb(round((1014+bit*40-x0)*scale),
                          round(offset_y+(314-y0)*scale)) for bit in range(10)]
            source_x, source_y = recipe.marker_center(ordinal)
            expected_center = ((source_x-x0)*scale, offset_y+(source_y-y0)*scale)
            cx, cy = map(round, expected_center)
            yellow = [(x, y) for y in range(cy-24, cy+25) for x in range(cx-24, cx+25)
                      if (lambda p: p[0] > 180 and p[1] > 180 and p[2] < 80)(rgb(x, y))]
            if not yellow:
                raise ValueError("missing actual projected moving marker")
            observed_center = (sum(x for x, _ in yellow)/len(yellow), sum(y for _, y in yellow)/len(yellow))
            oracle.marker(oracle.barcode(pixels), ordinal, observed_center, expected_center)
            black = [rgb(x, 1320) for x in range(50, 1030, 10)]
            if max(channel for pixel in black for channel in pixel) > 12:
                raise ValueError("BLACK fill independent ROI not near black")
            marker_records.append(dict(k=k, ordinal=ordinal, expected=expected_center, observed=observed_center))
        else:
            pixels = [rgb(round((34+bit*40)*2.25), round(656.25+194*2.25))
                      for bit in range(10)]
            if oracle.barcode(pixels) != ordinal:
                raise ValueError("full-camera source ordinal mismatch")
            luma = [sum(rgb(x, y))/3 for y in range(1268, 1360, 4)
                    for x in range(50, 1030, 8)]
            if max(luma)-min(luma) < 20:
                raise ValueError("CONTAIN_BLUR fixed ROI lacks nonconstant source content")
            marker_records.append(dict(k=k, ordinal=ordinal, blur_luma_range=max(luma)-min(luma)))
        image.close()
    reference = s.root/"delivery.reference.f32"
    s.register(reference, "pcm", 16*MiB)
    result = s.run(recipe.reference_argv(str(s.runtime.ffmpeg), str(candidate), str(reference),
        channels=2, input_rate=48000, delivery_tags=True), inputs=(candidate,), outputs=(reference,))
    pcm = s.read(reference, 16*MiB)
    grid, count = oracle.reference_tags(result.stderr_tail, truncated=result.stderr_truncated, channels=2, pcm=pcm)
    pulses = oracle.pulse_peaks(pcm, recipe.EXPECTED_PULSES, grid=grid)
    tolerance = F(1, 60)+F(1024, 48000)
    for measured in pulses:
        oracle.events(recipe.EXPECTED_PULSES, measured, tolerance)
    flashes = []
    for first, last, x, y, expected in ((0, 149, 820, 600, (30, 120)),
                                      (270, 479, 890, 1165, (30, 120))):
        result = s.run((str(s.runtime.ffmpeg), "-hide_banner", "-nostats", "-v", "error", "-i", str(candidate),
            "-map", "0:v:0", "-an", "-vf", f"select=between(n\\,{first}\\,{last}),crop=8:8:{x}:{y}",
            "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"),
            kind="probe", inputs=(candidate,))
        actual = oracle.flash_ordinals(result.stdout, last-first+1)
        if actual != expected:
            raise ValueError("flash/source pulse exact frame ordinal mismatch")
        flashes.append(dict(first=first, ordinals=actual))
    return dict(markers=marker_records, audio_grid=str(grid), decoded_audio_samples=count,
                pulse_times=pulses, flashes=flashes, tolerance=str(tolerance),
                overlay_boundary_crops=overlay_boundary_measurements(s, candidate),
                text_glyph_z_overlap=None, text_glyph_z_overlap_reason="Root pixel inspection still required")


def render_and_verify(root: Path, manifest: dict[str, Any], campaign: Campaign, *,
        source: Path, plan: Path, brand: Path, output: Path, frames: int,
        transcript: Path | None = None, cue: Path | None = None, diagnostics: Sequence[int] = (),
        scenario: str = "real"
        ) -> dict[str, Any]:
    # Finite conservative B child reservation made before public entry. The
    # public result's metadata must subsequently fit it; no fake engine is used.
    # Fixed synthetic source predicts39 B children (breakdown retained below).
    # This is an admission prediction checked against actual returned metadata,
    # never an OS quota or a universal bound for later arbitrary source plans.
    reservation = 39 if scenario == "synthetic" else 32 if frames <= 6 else 48
    oracle_children = oracle_child_reservation(scenario, len(diagnostics))
    admit(campaign.stage, children=campaign.children+reservation+oracle_children,
          seconds=(time.monotonic_ns()-campaign.start)/1e9)
    campaign.records.append(dict(scope="predeclared public plus independent oracle child admission",
        public_reservation=reservation, oracle_reservation=oracle_children,
        additional_boundary_file_output_controls=5 if scenario == "synthetic" else 0,
        public_synthetic_breakdown=None if scenario != "synthetic" else dict(
            source_probe=1, preflight=5, preflight_qualification=5, float=3, normalizer=2,
            cue_probe=1, audio_qualification=5, finalmix=1, combined_source=8,
            spatial=3, delivery=1, verification=4)))
    campaign.charge(reservation)
    # Bounded controller configuration read; expected frame partitions come
    # from the Root-pinned authored plan, not compiler/binder output.
    payload = json.loads(bounded_read(plan, 8*MiB))
    shot_frames = [math.floor(F(str(shot["output"]["end"]))*60+F(1, 2))
                   - math.floor(F(str(shot["output"]["start"]))*60+F(1, 2))
                   for shot in payload["shots"]]
    if sum(shot_frames) != frames:
        raise ValueError("independent authored frame partition/count mismatch")
    from clipperx._visual_render_host import VisualRenderFailure
    try:
        outputs = public_render(source, plan, output, assets=assets_for(manifest, brand, cue), transcript=transcript)
    except VisualRenderFailure as error:
        # Observe the failed public host without re-entering its lifecycle.
        # Each port is independent: one unavailable snapshot must not hide the
        # original render failure or the other available observations.
        snapshots: dict[str, Any] = {}
        snapshot_errors: dict[str, Any] = {}
        host = error.host
        captures: list[tuple[str, Callable[[], Any]]] = [
            ("owner", lambda: host.job.snapshot()),
            ("files", lambda: host.files.snapshot())]
        if host.normalization_files is not None:
            captures.append(("normalization_files", lambda: host.normalization_files.snapshot()))
        # RecordedTools already caps this finite ledger at 32768 records.
        captures.append(("stages", lambda: tuple(host.tools.evidence[:32768])))
        for name, capture in captures:
            try:
                snapshots[name] = capture()
            except Exception as snapshot_error:
                snapshot_errors[name] = error_details(snapshot_error)
        campaign.records.append(dict(scope="public B failed; controller observation, not delivery success",
            snapshots=snapshots, snapshot_errors=snapshot_errors))
        raise QualificationFailure(error, campaign.records) from error
    if len(outputs) != 1:
        raise ValueError("public default route must return exactly one MP4")
    candidate = outputs[0]
    # Controller metadata read is explicitly outside completed B, <=8MiB.
    meta = json.loads(bounded_read(candidate.with_suffix(".metadata.json"), 8*MiB))
    actual_children = meta["process_count"]
    record_public_return(campaign, actual_children, reservation)
    suffixes = (".visual-plan.json", ".render-spec.json", ".metadata.json", ".render-metrics.json", ".ffmpeg.log")
    if sorted(p.name for p in candidate.parent.iterdir() if p.is_file()) != sorted(
            [candidate.name, *(candidate.with_suffix(x).name for x in suffixes)]):
        # B also retains bounded diagnostics in a subdirectory, not extra clip sidecars.
        raise ValueError("exactly one MP4 and five sidecars required")
    try:
        validate_public_evidence(meta, frames, shot_frames, synthetic=scenario == "synthetic",
                                 all_solid=scenario == "all-solid")
    except Exception as error:
        raise QualificationFailure(error, campaign.records) from error
    admit(campaign.stage, children=campaign.children+oracle_children,
          seconds=(time.monotonic_ns()-campaign.start)/1e9)
    verification = FixtureSession(root, manifest, campaign).execute(
        lambda s: verify_delivery(s, candidate, frames, diagnostics, synthetic=scenario == "synthetic",
                                 tiny=scenario in ("tiny", "all-solid"), all_solid=scenario == "all-solid"))
    return dict(output=str(candidate), process_count=actual_children, independent=verification,
                decision="MEASURED; visual acceptance pending Root inspection")


def run_stage(base: Path, manifest: dict[str, Any], stage: str) -> dict[str, Any]:
    campaign = Campaign(stage)
    if stage == "prerequisites":
        session = FixtureSession(base/"fixtures", manifest, campaign)
        session.execute(prepare_prerequisites, cancel=True)
        source = session.root/"float48000-2.mov"
        brand = session.root/"alpha.png"
        tiny = render_and_verify(base/"tiny-oracles", manifest, campaign, source=source,
            plan=session.root/"tiny.json", brand=brand, output=base/"tiny-public", frames=6,
            diagnostics=tuple(range(6)), scenario="tiny")
        solid = render_and_verify(base/"solid-oracles", manifest, campaign, source=source,
            plan=session.root/"solid.json", brand=brand, output=base/"solid-public", frames=2,
            diagnostics=(0, 1), scenario="all-solid")
        result: Any = dict(float_cases={k: v for k, v in session.state.items() if k.startswith("float")},
                           component=session.state["component"], tiny=tiny, solid=solid)
    elif stage == "synthetic":
        prepared = FixtureSession(base/"fixtures", manifest, campaign).execute(prepare_synthetic)
        result = render_and_verify(base/"oracles", manifest, campaign,
            source=prepared["source"], plan=prepared["plan"], brand=prepared["brand"], cue=prepared["cue"],
            transcript=prepared["transcript"], output=base/"public", frames=480, diagnostics=recipe.DIAGNOSTIC_K,
            scenario="synthetic")
    else:
        row = manifest["real"].get(stage)
        if row is None:
            raise ValueError("NEEDS_CONTEXT: missing evidence-backed Root-pinned real manual window")
        from clipperx.visual_contracts import VisualCompositionPlan
        from clipperx.local_paths import source_fingerprint
        repository = Path(__file__).resolve().parents[1]
        plan_path = repository/safe_relative(row["plan"])
        plan = VisualCompositionPlan.model_validate(json.loads(bounded_read(plan_path, 8*MiB)))
        if ([[str(F(str(item.source.start))), str(F(str(item.source.end)))] for item in plan.shots]
                != [[str(F(a)), str(F(b))] for a, b in row["windows"]]):
            raise ValueError("real plan differs from evidence-backed frozen windows")
        # Stat fingerprint only; never body-hash original R/SVOD.
        frames = math.ceil(F(str(plan.shots[-1].output.end))*60)
        brand = base/"fixtures"/"alpha.png"
        def prepare_real(s: FixtureSession) -> None:
            source = Path(row["source"])
            s.register(source, "original", 2**48, owned=False)
            with s.files.stage_deadline(s.deadline(120)):
                if source_fingerprint(source, _files=s.files) != row["source_fingerprint"]:
                    raise ValueError("real original source stat fingerprint changed")
            s.put("alpha.png", recipe.alpha_png(), role="asset", retain=True)
        FixtureSession(base/"fixtures", manifest, campaign).execute(prepare_real)
        diagnostics = {0, frames//2, frames-1}
        for item in plan.shots[1:]:
            boundary = math.floor(F(str(item.output.start))*60+F(1, 2))
            diagnostics.update(k for k in (boundary-1, boundary, boundary+1) if 0 <= k < frames)
        result = render_and_verify(base/"oracles", manifest, campaign,
            source=Path(row["source"]), plan=plan_path, brand=brand, output=base/"public", frames=frames,
            transcript=None if row["transcript"] is None else repository/safe_relative(row["transcript"]),
            diagnostics=tuple(sorted(diagnostics)))
    admit(stage, children=campaign.children, seconds=(time.monotonic_ns()-campaign.start)/1e9)
    return dict(stage=stage, status="MEASURED_VISUAL_ACCEPTANCE_PENDING", results=result,
        children=campaign.children, elapsed_ns=time.monotonic_ns()-campaign.start,
        resources=campaign.records, budget_semantics=BUDGET_SEMANTICS,
        scope="controller after completed returns; no native acceptance inferred")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--confirm-host", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--stage", choices=STAGES)
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    if not args.native:
        print("actual NOT RUN: native opt-in and Root-pinned manifest/absent guard required")
        return 0
    validate_activation(args.native, args.confirm_host)
    if args.manifest is None or args.stage is None or not args.run_id or not re.fullmatch(
            "[A-Za-z0-9_-]{1,64}", args.run_id):
        raise ValueError("--manifest --stage and fresh ASCII --run-id required")
    repository = Path(__file__).resolve().parents[1]
    expected_python = repository/".venv-whisperx/Scripts/python.exe"
    if Path(sys.executable).resolve() != expected_python.resolve():
        raise ValueError("exact approved .venv-whisperx/Scripts/python.exe required")
    manifest, digest = load_manifest(args.manifest, repository)
    if args.run_id != manifest["runs"][args.stage]:
        raise ValueError("run-id differs from Root-pinned single-consumption case ID")
    if args.stage != "prerequisites":
        preceding = "prerequisites" if args.stage == "synthetic" else "synthetic"
        acceptance = manifest["gate_acceptance"].get(preceding)
        if type(acceptance) is not dict or set(acceptance) != {"path", "sha256", "accepted_by_root"}:
            raise ValueError("Root acceptance of preceding actual gate required")
        prior = bounded_read(repository/safe_relative(acceptance["path"]), 8*MiB)
        if not acceptance["accepted_by_root"] or hashlib.sha256(prior).hexdigest() != acceptance["sha256"]:
            raise ValueError("preceding actual gate acceptance identity mismatch")
        prior_record = json.loads(prior)
        if prior_record.get("stage") != preceding or prior_record.get("status") != "MEASURED_VISUAL_ACCEPTANCE_PENDING":
            raise ValueError("preceding summary stage/status mismatch")
    parent = repository/"data/processed/phase3-native-qualification"
    if not parent.is_dir():
        raise ValueError("Root must provision only campaign parent before actual activation")
    base = parent/args.run_id
    guard = base/".guard"
    require_absence(base.exists(), guard.exists())
    # Explicit bounded controller setup, outside any registered-driver custody.
    base.mkdir(exist_ok=False)
    with guard.open("xb") as stream:
        stream.write(json_bytes(dict(consumed=True, stage=args.stage, manifest_sha256=digest)))
    try:
        summary = run_stage(base, manifest, args.stage)
        summary.update(manifest_sha256=digest, guard=str(guard), run_id=args.run_id)
        leaf = base/"controller-summary.json"
        with leaf.open("xb") as stream:
            stream.write(json_bytes(summary))
        print(f"MEASURED; Root inspection/acceptance pending: {leaf}")
        return 0
    except Exception as error:
        # First failure record only. Never reset/replay the consumed guard.
        with (base/"controller-error.json").open("xb") as stream:
            stream.write(json_bytes(dict(stage=args.stage, manifest_sha256=digest,
                first_error=repr(error)[:4096], evidence=getattr(error, "evidence", None),
                status="STOPPED; custody requires exact state inspection")))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
