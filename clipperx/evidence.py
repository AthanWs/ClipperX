"""Execute planner evidence requests against persisted local artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from collections.abc import Sequence
from typing import Any
import zlib

from .jobs import JobStore
from .planner_contracts import EvidenceRequest, TimeRange
from .local_paths import atomic_write_text, source_fingerprint
from .visual_contracts import FaceTrackSummary, NormalizedRect, SourceVisualTimeline, VisualEvidenceRequest
from .render_clock import SourceDecodeReceipt, SourceTimeOrigin, canonical, digest, exact, rational
from .render_runtime import RenderRuntime, resolve_render_runtime, qualify_render_runtime
from .source_decode import DECODE_POLICY, DecodeLimits, Window, ObservedFrame, coalesce, decode_requirements, observe_windows, probe_source
from .visual_provenance import (MAX_ARTIFACT_BYTES, emitted_text_size, file_sha256,
                                generation_file, read_generation_json, visual_timeline_fingerprint)


def collect_evidence(
    video_path: Path,
    request: EvidenceRequest,
    *,
    transcript_path: Path | None = None,
    force: bool = False,
) -> Path:
    from .timeline import inspect_window

    job = JobStore.for_video(video_path)
    output_path = job.evidence_dir / (
        f"{_safe_name(request.request_id)}-{_request_fingerprint(request)}.json"
    )
    if output_path.exists() and not force:
        return output_path
    probe_path = inspect_window(
        video_path,
        start=request.window.start,
        end=request.window.end,
        sample_interval=request.sample_interval_seconds,
        include_audio=True,
        force=force,
    )
    probe = json.loads(probe_path.read_text(encoding="utf-8"))
    evidence_transcript_path = transcript_path or _best_transcript(job)
    transcript = _window_transcript(
        evidence_transcript_path,
        request.window.start,
        request.window.end,
    )
    bundle = {
        "version": 1,
        "request": request.model_dump(mode="json"),
        "source": str(job.source),
        "fingerprint": job.fingerprint,
        "probe_path": str(probe_path),
        "visual_summary": probe.get("visual_summary", {}),
        "samples": probe.get("samples", []),
        "transcript_path": str(evidence_transcript_path) if evidence_transcript_path else None,
        "transcript": transcript,
        "speaker_analysis": _load_optional(job.speaker_analysis_path),
        "streamer_profile": _load_optional(job.speaker_profile_path),
    }
    _write_json(output_path, bundle)
    return output_path


def visual_frame_times(
    windows: Sequence[TimeRange], timeline: SourceVisualTimeline, *, max_images: int = 5,
) -> list[float]:
    """Prioritize local state events, starts/ends, identity and protected regions.

    This is not a periodic timeline scan. A half-open window uses the last
    decodable instant just before its end rather than a frame from the next shot.
    """
    if not 1 <= max_images <= 5 or not windows:
        raise ValueError("Visual evidence requires windows and 1..5 images")
    if any(w.end > timeline.duration_seconds for w in windows):
        raise ValueError("Evidence window exceeds source duration")
    candidates: list[float] = []
    for span in timeline.spans:
        for at in (span.source.start, span.source.end):
            if any(w.start < at < w.end for w in windows):
                candidates.extend([at, max(0, at - 0.05)])
    for window in windows:
        candidates.extend([window.start, max(window.start, window.end - 0.05)])
    for sample in timeline.samples:
        if (sample.face_track_ids or sample.webcam_track_id or sample.text_regions
                or sample.persistent_rectangles):
            candidates.append(sample.at)
    selected: list[float] = []
    for at in candidates:
        if any(w.start <= at < w.end for w in windows) and at not in selected:
            selected.append(at)
            if len(selected) == max_images:
                break
    return selected


def associated_face_positions(
    timeline: SourceVisualTimeline, track: FaceTrackSummary,
) -> list[tuple[float, NormalizedRect, str]] | None:
    """Recover timestamps only from a complete ordered evidence-ID bijection.

    Face summaries aggregate positions and evidence IDs. Mere matching lengths,
    a track's broad first/last range or its identity do not timestamp positions.
    None means that association is unavailable, never an invented observation.
    """
    samples = [s for s in timeline.samples if track.track_id in s.face_track_ids]
    if (not samples or len(samples) != track.observation_count
            or len(track.positions) != len(samples) or len(track.evidence_ids) != len(samples)
            or len(set(track.evidence_ids)) != len(samples)
            or abs(samples[0].at - track.first_seen) > 1e-6
            or abs(samples[-1].at - track.last_seen) > .001):
        return None
    ids = set(track.evidence_ids)
    observed = [set(sample.evidence_ids) & ids for sample in samples]
    if any(len(evidence) != 1 for evidence in observed):
        return None
    ordered = [next(iter(evidence)) for evidence in observed]
    if ordered != track.evidence_ids:
        return None
    return [(sample.at, position, evidence) for sample, position, evidence
            in zip(samples, track.positions, ordered)]


@dataclass(frozen=True, slots=True)
class ObservedEvidenceContext:
    runtime: RenderRuntime
    origin: SourceTimeOrigin


def prepare_visual_evidence_context(source_video: Path) -> ObservedEvidenceContext:
    requirements = decode_requirements()
    runtime = resolve_render_runtime(requirements)
    if runtime.device_index is not None:
        raise ValueError("Director image evidence requires pinned CPU runtime")
    origin = probe_source(source_video, runtime)
    qualify_render_runtime(runtime, requirements)
    return ObservedEvidenceContext(runtime, origin)


_IMAGE_LIMITS = DecodeLimits(max_window=Fraction(3), max_retained_bytes=64_000_000)


def _points(windows: Sequence[TimeRange], timeline: SourceVisualTimeline,
            max_images: int) -> list[dict[str, Any]]:
    if type(max_images) is not int:
        raise ValueError("Image count must be an integer")
    return [{"requested": rational(Fraction(str(at))), "window_indices": [
        i for i, w in enumerate(windows) if w.start <= at < w.end]}
        for at in visual_frame_times(windows, timeline, max_images=max_images)]


def _batches(points: list[dict[str, Any]], origin: SourceTimeOrigin) -> list[tuple[Window, tuple[Fraction, ...]]]:
    """At most five requests, partitioned by exact span and conservative byte ceilings."""
    video = next(s for s in origin.streams if s.kind == "video")
    pixels = video.width * video.height
    bgr = pixels * 3
    if pixels > _IMAGE_LIMITS.max_pixels:
        raise ValueError("Director evidence pixel budget exceeded before work")
    groups: list[list[Fraction]] = []
    for at in sorted(exact(p["requested"]) for p in points):
        group = groups[-1] if groups else []
        count = len(group) + 1
        # Selected immutable bytes + predecessor/lookahead, PCM plus array/encode copies.
        fits = (not group or at - group[0] + Fraction(1,1000000) <= 2) and (
            bgr * (count + 2) + 96_000 <= 64_000_000
            and bgr * (3 * count + 6) <= 192_000_000)
        if not fits and group:
            groups.append([])
            group = groups[-1]
            count = 1
        if bgr * (count + 2) + 96_000 > 64_000_000 or bgr * (3 * count + 6) > 192_000_000:
            raise ValueError("Director evidence byte budget exceeded before work")
        if not groups:
            groups.append(group)
        group.append(at)
    result = []
    for group in groups:
        tiny = tuple(Window(at, min(origin.duration, at + Fraction(1, 1000000))) for at in group)
        # coalesce owns margin semantics; one bounded dispatch releases before the next.
        for window in coalesce(tiny, _IMAGE_LIMITS):
            selected = tuple(at for at in group if window.start <= at < window.end)
            if window.end - window.start > 2:
                raise ValueError("Director requested span budget exceeded")
            result.append((window, selected))
    return result


def _same(actual: Any, expected: Any, label: str) -> None:
    if canonical(actual) != canonical(expected):
        raise ValueError(f"Director evidence {label} mismatch")


def _fields(value: Any, fields: set[str], label: str) -> None:
    if type(value) is not dict or set(value) != fields:
        raise ValueError(f"Invalid Director evidence {label} schema")


def _artifact(root: Path, relative: str) -> dict[str, Any]:
    path = generation_file(root, relative)
    size = path.stat().st_size
    if not 0 < size <= MAX_ARTIFACT_BYTES:
        raise ValueError("Director evidence artifact byte quota exceeded")
    return {"path": relative, "bytes": size, "sha256": file_sha256(path)}


def _frame_record(frame: dict[str, Any], origin: SourceTimeOrigin) -> tuple[str, dict[str, Any]]:
    video = next(s for s in origin.streams if s.kind == "video")
    identity = {"origin_id": origin.identity, "decoder_id": origin.decoder_id,
        "stream_index": video.index, "raw_pts": frame["pts"], "time_base": frame["time_base"],
        "width": frame["width"], "height": frame["height"], "raw_bgr_sha256": frame["sha256"]}
    return digest(canonical(identity)), identity


def _check_source(source: Path, timeline: SourceVisualTimeline, context: ObservedEvidenceContext) -> None:
    origin = context.origin
    stat = source.stat()
    if (str(source.resolve()) != origin.path or source_fingerprint(source) != origin.fingerprint
            or origin.fingerprint != timeline.source_fingerprint or stat.st_size != origin.size
            or stat.st_mtime_ns != origin.mtime_ns):
        raise ValueError("source fingerprint changed before evidence validation")


def collect_visual_evidence(
    video_path: Path,
    windows: Sequence[TimeRange],
    *,
    timeline: SourceVisualTimeline,
    output_path: Path,
    max_images: int = 5,
    context: ObservedEvidenceContext | None = None,
    request: VisualEvidenceRequest | None = None,
) -> Path:
    """Immutable bounded observed BGR -> JPEG85 association, with incidental audio."""
    import cv2
    context = context or prepare_visual_evidence_context(video_path)
    _check_source(video_path, timeline, context)
    points = _points(windows, timeline, max_images)
    batches = _batches(points, context.origin)
    if request is not None:
        request = VisualEvidenceRequest.model_validate(request.model_dump())
        _same([w.model_dump(mode="json") for w in windows], [request.source.model_dump(mode="json")], "request windows")
    if output_path.exists():
        raise ValueError("Director evidence bundle already exists")
    frames_dir = output_path.parent / output_path.stem
    header={"version":1,"kind":"observed_director_images","source":str(video_path.resolve()),
        "source_fingerprint":timeline.source_fingerprint,"origin":context.origin.record(),
        "origin_id":context.origin.identity,"timeline_fingerprint":visual_timeline_fingerprint(timeline),
        "windows":[w.model_dump(mode="json") for w in windows],"max_images":max_images,
        "request":request.model_dump(mode="json") if request else None,"selected_requests":points}
    # Selected coordinates/window indices appear in the header, outcomes and refs.
    # Reserve those known bytes plus bounded frame/catalog metadata before work.
    reserved_json=(3*emitted_text_size(canonical(header))
        + max_images*(4096+2*len(str(frames_dir.resolve()).encode("utf-8"))))
    if reserved_json>MAX_ARTIFACT_BYTES:
        raise ValueError("Director bundle JSON quota exceeded before work")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=False)
    frames: dict[str, dict[str, Any]] = {}
    observations: list[dict[str,Any]] = []
    outcomes: dict[str, dict[str, Any]] = {}
    catalog: list[dict[str, Any]] = []
    total = 0

    def emit(relative: str, payload: bytes) -> dict[str, Any]:
        nonlocal total
        # Reserve final bundle and all remaining file slots before each write.
        if (not payload or len(payload) > MAX_ARTIFACT_BYTES or len(catalog) >= 2 * max_images
                or total + len(payload) + MAX_ARTIFACT_BYTES > (2 * max_images + 1) * MAX_ARTIFACT_BYTES):
            raise ValueError("Director evidence storage quota exceeded before write")
        target = generation_file(frames_dir, relative)
        with target.open("xb") as handle:
            handle.write(payload)
        row = _artifact(frames_dir, relative)
        if row["bytes"] != len(payload) or row["sha256"] != digest(payload):
            raise ValueError("Director evidence write postcondition failed")
        catalog.append(row)
        total += row["bytes"]
        return row

    for window, selected in batches:
        # File slots plus the final JSON are reserved before external work.
        if len(catalog) + 1 + len(selected) > 2 * max_images:
            raise ValueError("Director evidence artifact quota exceeded before work")
        results = observe_windows(context.runtime, context.origin, (window,), selected,
                                  audio_policy="analysis16k-mono", limits=_IMAGE_LIMITS)
        if len(results) != 1:
            raise ValueError("Director observation batch count mismatch")
        result = results[0]
        receipt = SourceDecodeReceipt.from_json(result.receipt.to_json(), origin=context.origin, policy=DECODE_POLICY)
        data = json.loads(receipt.payload_json)
        _same([o["requested"] for o in data["outcomes"]], [rational(t) for t in selected], "outcomes")
        if data["audio"][0]["policy"] != "analysis16k-mono" or digest(result.pcm) != data["audio"][0]["pcm_sha256"]:
            raise ValueError("Director incidental audio payload mismatch")
        offset=0
        for block in data["audio"][0]["output"]:
            piece=result.pcm[offset:offset+block["bytes"]]
            if digest(piece)!=block["sha256"] or f"{zlib.adler32(piece,0):08X}"!=block["checksum"]:
                raise ValueError("Director incidental PCM block mismatch")
            offset+=block["bytes"]
        if offset!=len(result.pcm):
            raise ValueError("Director incidental PCM byte count mismatch")
        del piece
        retained = {f.tag.n: f for f in result.frames}
        raw = None
        if len(retained) != len(result.frames) or set(retained) != {o["frame_n"] for o in data["outcomes"] if o["status"] == "covered"}:
            raise ValueError("Director raw frame association mismatch")
        observed: ObservedFrame | None = None
        for raw_n,observed in retained.items():
            metadata=data["video"][raw_n]
            if (observed.sha256!=metadata["sha256"] or observed.raw_end!=exact(metadata["coverage_end"])
                    or zlib.adler32(observed.bgr,0)!=observed.tag.checksum
                    or canonical(observed.tag.record())!=canonical({k:metadata[k] for k in observed.tag.record()})):
                raise ValueError("Director actual raw frame mismatch")
        observed=None
        relative = f"observation-{len(observations):02d}.json"
        observation = emit(relative, receipt.to_json().encode())
        observations.append({**observation, "observation_id": json.loads(receipt.to_json())["sha256"]})
        for outcome in data["outcomes"]:
            point = next(p for p in points if p["requested"] == outcome["requested"])
            ref = {**point, "receipt": relative, "frame_n": outcome["frame_n"]}
            frame_id = None
            if outcome["status"] == "covered":
                raw = retained[outcome["frame_n"]]
                metadata = data["video"][raw.tag.n]
                if (raw.sha256 != metadata["sha256"] or raw.raw_end != exact(metadata["coverage_end"])
                        or canonical(raw.tag.record()) != canonical({k: metadata[k] for k in raw.tag.record()})):
                    raise ValueError("Director actual raw frame mismatch")
                frame_id, _ = _frame_record(metadata, context.origin)
                start = rational(raw.raw_start - context.origin.origin)
                end = rational(raw.raw_end - context.origin.origin)
                if frame_id not in frames:
                    array = raw.array_copy()
                    ok, encoded = cv2.imencode(".jpg", array, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not ok:
                        raise ValueError("Cannot encode observed Director image")
                    jpeg_row = emit(f"frame-{frame_id}.jpg", encoded.tobytes())
                    del array, encoded
                    at = float(exact(point["requested"]))
                    frames[frame_id] = {"evidence_id": f"{output_path.stem}-frame-{frame_id[:16]}",
                        "at": at, "requested_at": at, "path": str((frames_dir/jpeg_row["path"]).resolve()),
                        "width": raw.tag.width, "height": raw.tag.height, "frame_id": frame_id,
                        "raw_bgr_sha256": raw.sha256, "raw_pts": raw.tag.pts,
                        "time_base": rational(raw.tag.time_base), "observed_source_start": start,
                        "observed_source_end": end, "jpeg": {"relative_path": jpeg_row["path"],
                            "sha256": jpeg_row["sha256"], "bytes": jpeg_row["bytes"], "quality": 85,
                            "resize": "none", "opencv_version": cv2.__version__}, "sample_refs": []}
                elif (frames[frame_id]["observed_source_start"], frames[frame_id]["observed_source_end"]) != (start,end):
                    raise ValueError("Conflicting same-PTS frame coverage")
                frames[frame_id]["sample_refs"].append(ref)
            outcomes[point["requested"]] = {**ref, "status": outcome["status"], "frame_id": frame_id}
        del retained, raw, observed, result, results  # no retained PCM/raw batch crosses next dispatch
    ordered_frames = sorted(frames.values(), key=lambda f: next(i for i,p in enumerate(points)
        if any(r["requested"] == p["requested"] for r in f["sample_refs"])))
    for frame in ordered_frames:
        frame["sample_refs"].sort(key=lambda r: next(i for i,p in enumerate(points) if p["requested"] == r["requested"]))
        frame["at"] = frame["requested_at"] = float(exact(frame["sample_refs"][0]["requested"]))
    bundle = {**header,"outcomes": [outcomes[p["requested"]] for p in points],
        "observations": observations, "frames": ordered_frames, "catalog": sorted(catalog,key=lambda r:r["path"])}
    text = canonical(bundle)
    if emitted_text_size(text) > MAX_ARTIFACT_BYTES:
        raise ValueError("Director bundle JSON quota exceeded before write")
    _check_source(video_path, timeline, context)
    atomic_write_text(output_path, text)
    load_visual_evidence_bundle(output_path, source_video=video_path, timeline=timeline, windows=windows,
                                context=context, max_images=max_images, request=request,
                                expected_bundle_sha256=digest(text))
    return output_path


def load_visual_evidence_bundle(bundle_path: Path, *, source_video: Path,
    timeline: SourceVisualTimeline, windows: Sequence[TimeRange], context: ObservedEvidenceContext,
    max_images: int = 5, request: VisualEvidenceRequest | None = None,
    expected_bundle_sha256: str | None = None) -> dict[str, Any]:
    """Pure bounded loading: no probe, qualification, observation, or model construction."""
    _check_source(source_video,timeline,context)
    if bundle_path.stat().st_size>MAX_ARTIFACT_BYTES:
        raise ValueError("Director bundle byte quota exceeded")
    if expected_bundle_sha256 is not None and file_sha256(bundle_path) != expected_bundle_sha256:
        raise ValueError("Director bundle captured SHA mismatch")
    data=read_generation_json(bundle_path)
    _fields(data,{"version","kind","source","source_fingerprint","origin","origin_id","timeline_fingerprint",
        "windows","max_images","request","selected_requests","outcomes","observations","frames","catalog"},"bundle")
    points=_points(windows,timeline,max_images)
    for key, expected in {"version":1,"kind":"observed_director_images","source":str(source_video.resolve()),
        "source_fingerprint":timeline.source_fingerprint,"origin":context.origin.record(),"origin_id":context.origin.identity,
        "timeline_fingerprint":visual_timeline_fingerprint(timeline),"windows":[w.model_dump(mode="json") for w in windows],
        "max_images":max_images,"request":request.model_dump(mode="json") if request else None,"selected_requests":points}.items():
        _same(data[key],expected,key)
    root=bundle_path.parent/bundle_path.stem
    if any(type(data[k]) is not list for k in ("catalog","observations","frames","outcomes")):
        raise ValueError("Invalid Director bundle containers")
    if len(data["catalog"]) > 2*max_images or len(data["observations"])>max_images or len(data["frames"])>max_images:
        raise ValueError("Director bundle catalog budget")
    catalog={}
    for row in data["catalog"]:
        _fields(row,{"path","bytes","sha256"},"catalog")
        generation_file(root,row["path"])
        if row["path"] in catalog:
            raise ValueError("Duplicate Director artifact")
        _same(row,_artifact(root,row["path"]),"catalog file")
        catalog[row["path"]]=row
    actual=set()
    for path in root.rglob("*"):
        if path.is_dir():
            raise ValueError("Unassociated Director artifact directory")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
            if len(actual)>2*max_images:
                raise ValueError("Director actual artifact quota exceeded")
    if actual != set(catalog):
        raise ValueError("Dangling Director artifacts")
    expected_outcomes={}
    expected_frames: dict[str,dict[str,Any]]={}
    associated=set()
    for index,observation in enumerate(data["observations"]):
        _fields(observation,{"path","bytes","sha256","observation_id"},"observation")
        relative=observation["path"]
        generation_file(root,relative)
        if relative != f"observation-{index:02d}.json" or relative in associated:
            raise ValueError("Invalid receipt path/association")
        _same({k:observation[k] for k in ("path","bytes","sha256")},catalog.get(relative),"receipt file")
        envelope=read_generation_json(generation_file(root,relative))
        receipt=SourceDecodeReceipt.from_json(canonical(envelope),origin=context.origin,policy=DECODE_POLICY)
        _same(observation["observation_id"],envelope["sha256"],"observation identity")
        payload=json.loads(receipt.payload_json)
        if payload["audio"][0]["policy"]!="analysis16k-mono":
            raise ValueError("Director incidental audio policy mismatch")
        for name,value in _IMAGE_LIMITS.record().items():
            _same(payload["limits"][name],value,"decode limits")
        if any(exact(w[1])-exact(w[0])>2 for w in payload["windows"]):
            raise ValueError("Director observation requested span limit")
        associated.add(relative)
        for outcome in payload["outcomes"]:
            coordinate=outcome["requested"]
            if coordinate in expected_outcomes or coordinate not in {p["requested"] for p in points}:
                raise ValueError("Director outcomes not bijective")
            point=next(p for p in points if p["requested"]==coordinate)
            ref={**point,"receipt":relative,"frame_n":outcome["frame_n"]}
            frame_id=None
            if outcome["status"]=="covered":
                raw=payload["video"][outcome["frame_n"]]
                frame_id,_=_frame_record(raw,context.origin)
                jpeg_path=f"frame-{frame_id}.jpg"
                if jpeg_path not in catalog:
                    raise ValueError("Missing Director JPEG")
                base={"evidence_id":f"{bundle_path.stem}-frame-{frame_id[:16]}","path":str(generation_file(root,jpeg_path).resolve()),
                    "width":raw["width"],"height":raw["height"],"frame_id":frame_id,"raw_bgr_sha256":raw["sha256"],
                    "raw_pts":raw["pts"],"time_base":raw["time_base"],
                    "observed_source_start":rational(raw["pts"]*exact(raw["time_base"])-context.origin.origin),
                    "observed_source_end":rational(exact(raw["coverage_end"])-context.origin.origin)}
                # A stream/PTS cannot acquire different pixels or coverage in another batch.
                for other in expected_frames.values():
                    if other["raw_pts"]==base["raw_pts"] and other["time_base"]==base["time_base"]:
                        _same({k:other[k] for k in base},base,"same PTS")
                if frame_id not in expected_frames:
                    expected_frames[frame_id]={**base,"sample_refs":[]}
                expected_frames[frame_id]["sample_refs"].append(ref)
                associated.add(jpeg_path)
            expected_outcomes[coordinate]={**ref,"status":outcome["status"],"frame_id":frame_id}
    if set(expected_outcomes)!={p["requested"] for p in points} or associated!=set(catalog):
        raise ValueError("Unassociated Director receipt/frame/outcome")
    _same(data["outcomes"],[expected_outcomes[p["requested"]] for p in points],"outcomes")
    expected_order=sorted(expected_frames,key=lambda fid:min(i for i,p in enumerate(points)
        if any(r["requested"]==p["requested"] for r in expected_frames[fid]["sample_refs"])))
    frame_fields={"evidence_id","at","requested_at","path","width","height","frame_id","raw_bgr_sha256",
            "raw_pts","time_base","observed_source_start","observed_source_end","jpeg","sample_refs"}
    for frame in data["frames"]:
        _fields(frame,frame_fields,"frame")
    if [f["frame_id"] for f in data["frames"]]!=expected_order:
        raise ValueError("Director frame identity/order mismatch")
    for frame in data["frames"]:
        _fields(frame,{"evidence_id","at","requested_at","path","width","height","frame_id","raw_bgr_sha256",
            "raw_pts","time_base","observed_source_start","observed_source_end","jpeg","sample_refs"},"frame")
        expected=expected_frames[frame["frame_id"]]
        refs=sorted(expected["sample_refs"],key=lambda r:next(i for i,p in enumerate(points) if p["requested"]==r["requested"]))
        for k,v in {**expected,"sample_refs":refs,"at":float(exact(refs[0]["requested"])),"requested_at":float(exact(refs[0]["requested"]))}.items():
            _same(frame[k],v,"frame association")
        jpeg=frame["jpeg"]
        _fields(jpeg,{"relative_path","sha256","bytes","quality","resize","opencv_version"},"JPEG")
        row=catalog[f"frame-{frame['frame_id']}.jpg"]
        _same({k:jpeg[k] for k in ("relative_path","sha256","bytes","quality","resize")},
              {"relative_path":row["path"],"sha256":row["sha256"],"bytes":row["bytes"],"quality":85,"resize":"none"},"JPEG identity")
        if type(jpeg["opencv_version"]) is not str or not jpeg["opencv_version"]:
            raise ValueError("Missing JPEG encoder identity")
    return data


def _best_transcript(job: JobStore) -> Path | None:
    if job.preferred_transcript_path.exists():
        return job.preferred_transcript_path
    return None


def _window_transcript(path: Path | None, start: float, end: float) -> list[dict[str, Any]]:
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [
        segment
        for segment in payload
        if float(segment.get("end", 0.0)) > start and float(segment.get("start", 0.0)) < end
    ]


def _load_optional(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_name(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in "-_" else "-" for character in value)
    return cleaned.strip("-")[:80] or "request"


def _request_fingerprint(request: EvidenceRequest) -> str:
    payload = json.dumps(
        request.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
