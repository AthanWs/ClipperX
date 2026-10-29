"""Root opt-in bounded current-owned source observations; default performs no IO.

This adapter transfers PNG and metadata evidence only. Finalization checks metadata
consistency; current completed owner/file/control records are required alongside it.
"""

import argparse
from dataclasses import dataclass, replace
from fractions import Fraction as F
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Sequence

from clipperx.local_paths import source_fingerprint
from scripts.render_visual_smoke import (
    Campaign, FixtureSession, MiB, bounded_read, error_details, json_bytes,
    load_manifest, require_absence, safe_relative,
)


def rational(value: Any) -> F:
    if (type(value) is not str or not 1 <= len(value) <= 128 or not value.isascii()
            or re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:/[0-9]+)?", value) is None):
        raise ValueError("bounded exact rational string required")
    try:
        return F(value)
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("finite rational required") from error


def exact_keys(value: Any, keys: set[str]) -> None:
    if type(value) is not dict or set(value) != keys:
        raise ValueError("exact observation request schema required")


def parse_requests(data: Any, run_id: str) -> dict[str, Any]:
    """Pure bounded validation: source bodies and source filesystem are untouched."""
    exact_keys(data, {"version", "run_id", "cases"})
    if (type(data["version"]) is not int or data["version"] != 1
            or type(data["run_id"]) is not str
            or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", data["run_id"]) is None
            or data["run_id"] != run_id):
        raise ValueError("Root-pinned fresh ASCII run-id/version required")
    cases = data["cases"]
    if type(cases) is not list or len(cases) != 3:
        raise ValueError("exactly three source cases required")
    ids: list[str] = []
    batch_count = request_count = 0
    for case in cases:
        exact_keys(case, {"id", "source", "source_fingerprint", "batches"})
        if type(case["id"]) is not str or case["id"] not in ("gameplay", "camera", "transition"):
            raise ValueError("unknown observation case")
        ids.append(case["id"])
        source = case["source"]
        if (type(source) is not str or not source.isascii() or not 1 <= len(source) <= 2048
                or not Path(source).is_absolute() or any(ord(c) < 32 for c in source)):
            raise ValueError("fixed absolute ASCII source path required")
        if (type(case["source_fingerprint"]) is not str
                or re.fullmatch(r"[0-9a-f]{16}", case["source_fingerprint"]) is None):
            raise ValueError("source stat fingerprint required")
        batches = case["batches"]
        if type(batches) is not list or not 1 <= len(batches) <= 7:
            raise ValueError("bounded batch list required")
        batch_count += len(batches)
        for batch in batches:
            exact_keys(batch, {"window", "requests"})
            window, requests = batch["window"], batch["requests"]
            if type(window) is not list or len(window) != 2:
                raise ValueError("exact two-point window required")
            start, end = (rational(v) for v in window)
            if start < 0 or end <= start or end+F(1, 4)-max(F(0), start-F(1, 2)) > 2:
                raise ValueError("effective window including default margins exceeds two seconds")
            if type(requests) is not list or not 1 <= len(requests) <= 5:
                raise ValueError("one to five requests required")
            points = [rational(v) for v in requests]
            if any(not start <= p < end for p in points) or any(
                    b < a for a, b in zip(points, points[1:])):
                raise ValueError("ordered half-open source requests required")
            request_count += len(points)
    if len(set(ids)) != 3 or batch_count != 7 or request_count != 12:
        raise ValueError("exact three unique cases/seven batches/twelve points required")
    return data


def check_authority(repository: Path, manifest: dict[str, Any], requests: str,
                    run_id: str) -> tuple[dict[str, Any], str]:
    path = safe_relative(requests)
    if requests not in manifest["pins"]:
        raise ValueError("requests must be an exact repository-relative manifest pin")
    raw = bounded_read(repository/path, 100000)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != manifest["pins"][requests]:
        raise ValueError("Root request identity mismatch")
    data = parse_requests(json.loads(raw), run_id)
    for stage in ("prerequisites", "synthetic"):
        entry = manifest["gate_acceptance"].get(stage)
        exact_keys(entry, {"path", "sha256", "accepted_by_root"})
        if entry["accepted_by_root"] is not True:
            raise ValueError("literal Root acceptance required for both actual gates")
        prior = bounded_read(repository/safe_relative(entry["path"]), 8*MiB)
        if hashlib.sha256(prior).hexdigest() != entry["sha256"]:
            raise ValueError("accepted actual gate summary SHA mismatch")
        record = json.loads(prior)
        if (type(record) is not dict or record.get("stage") != stage
                or record.get("status") != "MEASURED_VISUAL_ACCEPTANCE_PENDING"
                or record.get("run_id") != manifest["runs"][stage]):
            raise ValueError("actual gate stage/status/run identity mismatch")
    return data, digest


class ObservationCampaign(Campaign):
    """Existing prerequisite wall admission with a tighter pre-charge child cap."""

    def __init__(self) -> None:
        super().__init__("prerequisites")

    def charge(self, count: int = 1) -> None:
        if type(count) is not int or count < 0 or self.children+count > 32:
            raise ValueError("observation campaign child admission ceiling32")
        super().charge(count)


@dataclass
class EvidenceBudget:
    raw_bytes: int = 0
    image_bytes: int = 0
    images: int = 0

    def add_raw(self, size: int) -> None:
        if size < 0 or self.raw_bytes+size > 128*MiB:
            raise ValueError("aggregate selected RAW scratch ceiling")
        self.raw_bytes += size

    def add_image(self, size: int) -> None:
        if size < 0 or self.images+1 > 12 or self.image_bytes+size > 128*MiB:
            raise ValueError("aggregate image/count ceiling")
        self.images += 1
        self.image_bytes += size


def verify_png(raw: bytes, width: int, height: int, expected: str) -> None:
    from PIL import Image
    decoded = Image.open(BytesIO(raw), formats=("PNG",))
    try:
        if (decoded.size != (width, height) or min(width, height) <= 0
                or width*height > 1920*1080):
            raise ValueError("PNG exact dimensions/pixel ceiling mismatch")
        converted = decoded.convert("RGB")
        try:
            bgr = converted.tobytes("raw", "BGR")
            if len(bgr) != 3*width*height or hashlib.sha256(bgr).hexdigest() != expected:
                raise ValueError("PNG selected BGR pixel identity mismatch")
        finally:
            converted.close()
    finally:
        decoded.close()


def observe(s: FixtureSession, cases: list[dict[str, Any]]) -> list[Any]:
    from clipperx import source_decode as sd
    from clipperx.render_clock import StreamDecodeLimits
    budget = EvidenceBudget()
    results = []
    runtime = replace(s.runtime, device_index=None)
    for case in cases:
        source = Path(case["source"])
        s.register(source, "original", 2**48, owned=False)
        with s.io_scope():
            if source_fingerprint(source, _files=s.files) != case["source_fingerprint"]:
                raise ValueError("original source stat fingerprint changed")
            origin = s.source_probe(source)
        videos = [v for v in origin.streams if v.kind == "video"]
        audios = [a for a in origin.streams if a.kind == "audio"]
        if (len(videos) != 1 or len(audios) != 1 or audios[0].channels not in (1, 2)
                or min(videos[0].width, videos[0].height) <= 0
                or videos[0].width*videos[0].height > 1920*1080):
            raise ValueError("one bounded video and one mono/stereo audio stream required")
        policy = "render48k-mono" if audios[0].channels == 1 else "render48k-stereo"
        row: dict[str, Any] = dict(id=case["id"], origin=origin.record(), batches=[])
        for batch_index, batch in enumerate(case["batches"]):
            start, end = (F(v) for v in batch["window"])
            requests = tuple(F(v) for v in batch["requests"])
            control = s.job.new_observation("streamed_s16", deadline_ns=s.deadline(45))
            saved = []
            with sd.stream_observed_window(runtime, origin, sd.Window(start, end), requests,
                    audio_policy=policy, limits=StreamDecodeLimits(max_window=F(2), max_requests=5,
                        max_pixels=1920*1080), _host=control, _runtime_files=s.files) as stream:
                while True:
                    selection = stream.next_selection()
                    if selection is None:
                        break
                    frame = selection.frame
                    width, height = frame.tag.width, frame.tag.height
                    index = selection.delivery_index
                    budget.add_raw(len(frame.bgr))
                    selected_sha = frame.sha256
                    metadata: dict[str, Any] = dict(selected=selection.selected_record(),
                        successor=selection.successor_record(), delivery=selection.delivery_record(),
                        frame_tag=frame.tag.record(), selected_bgr_sha256=selected_sha,
                        requested_sources=[batch["requests"][n] for n in selection.request_ordinals])
                    leaf = f"{case['id']}-{batch_index:02}-{index:02}"
                    path = s.put(leaf+".bgr", frame.bgr, role="raw", retain=False,
                                 cap=3*width*height)
                    saved.append((path, leaf, width, height, metadata))
                    del frame, selection
                    stream.acknowledge(index)
                final = stream.finish()
                final_record = final.record()
            status = control.ownership_snapshot()
            if status.phase != "retired_ok" or not status.window_resources_zero or status.cleanup_required:
                raise ValueError("source stream not certainly retired")
            images = []
            for path, leaf, width, height, metadata in saved:
                if budget.images >= 12:
                    raise ValueError("aggregate image/count ceiling")
                argv = (str(runtime.ffmpeg), "-hide_banner", "-nostdin", "-nostats", "-v", "error",
                    "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{width}x{height}",
                    "-framerate", "1", "-i", str(path), "-an", "-frames:v", "1", "-c:v", "png",
                    "-pix_fmt", "rgb24", "-f", "image2")
                image_path, _ = s.file_output(leaf+".png", argv, (path,), timeout=45,
                    cap=12*MiB, retain=True)
                image = s.read(image_path, 12*MiB)
                budget.add_image(len(image))
                verify_png(image, width, height, metadata["selected_bgr_sha256"])
                images.append(dict(metadata, image_path=str(image_path), image_bytes=len(image),
                                   image_sha256=hashlib.sha256(image).hexdigest()))
                del image
            row["batches"].append(dict(window=batch["window"], requests=batch["requests"],
                images=images, finalization=final_record, source_control=status))
        results.append(row)
    return results


def run_observations(base: Path, manifest: dict[str, Any], data: dict[str, Any],
                     campaign: ObservationCampaign) -> dict[str, Any]:
    results = FixtureSession(base/"observations", manifest, campaign).execute(
        lambda session: observe(session, data["cases"]))
    campaign.charge(0)
    return dict(status="MEASURED_VISUAL_ACCEPTANCE_PENDING", results=results,
        children=campaign.children, elapsed_ns=time.monotonic_ns()-campaign.start,
        resources=campaign.records,
        budget_semantics="admission plus post-return elapsed/count failure ceilings; no hard preemption or OS child quota",
        scope="bounded current-owned source observations; Root visual inspection pending; no C render acceptance")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", action="store_true")
    parser.add_argument("--confirm-host", action="store_true")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--requests")
    parser.add_argument("--run-id")
    args = parser.parse_args(argv)
    if not args.native:
        print("actual NOT RUN: native opt-in and Root-pinned observation authority required")
        return 0
    if not args.confirm_host:
        raise ValueError("explicit --confirm-host required")
    if (args.manifest is None or not args.requests or type(args.run_id) is not str
            or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.run_id) is None):
        raise ValueError("--manifest --requests and fresh ASCII --run-id required")
    repository = Path(__file__).resolve().parents[1]
    if Path(sys.executable).resolve() != (repository/".venv-whisperx/Scripts/python.exe").resolve():
        raise ValueError("exact approved .venv-whisperx/Scripts/python.exe required")
    manifest, manifest_sha = load_manifest(args.manifest, repository)
    data, request_sha = check_authority(repository, manifest, args.requests, args.run_id)
    parent = repository/"data/processed/phase3-native-qualification"
    if not parent.is_dir():
        raise ValueError("Root must provision only campaign parent before actual activation")
    base = parent/args.run_id
    guard = base/".guard"
    require_absence(base.exists(), guard.exists())
    base.mkdir(exist_ok=False)
    identity = dict(run_id=args.run_id, manifest_sha256=manifest_sha, requests_sha256=request_sha,
                    scope="bounded current-owned observation controller setup; no C rendering")
    with guard.open("xb") as stream:
        stream.write(json_bytes(dict(identity, consumed=True)))
    campaign = ObservationCampaign()
    try:
        summary = run_observations(base, manifest, data, campaign)
        summary.update(identity, guard=str(guard))
        summary_bytes = json_bytes(summary)
        leaf = base/"controller-summary.json"
        with leaf.open("xb") as stream:
            stream.write(summary_bytes)
        print(f"MEASURED; Root image inspection pending: {leaf}")
        return 0
    except Exception as error:
        error_bytes = json_bytes(dict(identity, first_error=error_details(error),
            evidence=getattr(error, "evidence", None), resources=campaign.records,
            status="STOPPED; custody requires exact state inspection"))
        with (base/"controller-error.json").open("xb") as stream:
            stream.write(error_bytes)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
