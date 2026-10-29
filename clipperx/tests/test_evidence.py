from __future__ import annotations

from fractions import Fraction as F
import json
from pathlib import Path
from typing import Any
import zlib

import cv2
import numpy as np
import pytest

from clipperx.evidence import visual_frame_times
from clipperx.planner_contracts import TimeRange
from clipperx.tests.test_visual_contracts import make_span, make_timeline


def install_observer(monkeypatch: pytest.MonkeyPatch, video: Path, *, duration: float = 100,
                     prefix: F = F(0), step: F = F(1,10), static: bool = False,
                     width: int = 192, height: int = 108) -> tuple[Any, list[Any]]:
    """Only external tool boundaries are fake: selection/receipts/pixels/files are real."""
    import clipperx.evidence as e
    from clipperx.render_clock import SourceTimeOrigin, StreamClock, SourceDecodeReceipt
    from clipperx.render_clock import digest, rational, audio_decode_argv
    from clipperx.source_decode import DecodedWindow, VideoTag, select_frames
    from clipperx.render_runtime import RenderRuntime
    from clipperx.local_paths import source_fingerprint
    stat = video.stat()
    origin = SourceTimeOrigin(str(video.resolve()), stat.st_size, stat.st_mtime_ns,
        source_fingerprint(video), F(0), F(str(duration)), (
            StreamClock(0, "video", F(1, 1000000), prefix, F(str(duration)), width, height),
            StreamClock(1, "audio", F(1, 48000), F(0), F(str(duration)),
                        sample_rate=48000, channels=1, sample_format="s16")), "HOST-fake", {})
    runtime = RenderRuntime(Path("HOST-ffmpeg"), Path("HOST-ffprobe"), "HOST", "a"*64,
                            "b"*64, None, None, "-filter_complex_script", None)
    calls: list[Any] = []
    monkeypatch.setattr(e, "resolve_render_runtime", lambda requirements: runtime, raising=False)
    monkeypatch.setattr(e, "probe_source", lambda source, pinned: origin, raising=False)
    monkeypatch.setattr(e, "qualify_render_runtime", lambda *a: calls.append("qualify"), raising=False)

    def observe(pinned: Any, selected_origin: Any, windows: Any, requests: Any, *,
                audio_policy: str, limits: Any) -> Any:
        assert selected_origin == origin and audio_policy == "analysis16k-mono"
        assert len(windows) == 1
        window = windows[0]
        calls.append((window, tuple(requests)))
        start = max(F(0), window.start - limits.preroll)
        end = min(origin.duration, window.end + limits.postroll)
        instants = sorted({max(prefix, prefix + ((t-prefix)//step)*step) for t in requests}
                          | {max(prefix, prefix + ((max(requests)-prefix)//step+1)*step)})
        pairs: list[tuple[VideoTag,bytes]] = []
        for n, at in enumerate(instants):
            if not start <= at < end:
                continue
            pixels = bytes([37 if static else int(at * 1000) % 251]) * (width * height * 3)
            tag = VideoTag(len(pairs), int(at * 1000000), F(1, 1000000), width, height,
                           zlib.adler32(pixels, 0))
            pairs.append((tag, pixels))
        selection = select_frames(pairs, origin, requests, limits)
        # Finite fake incidental PCM covers the actual decoder interval.
        a = F(int(start * 16000), 16000)
        count = int((end-a)*16000)
        pcm = bytes(count*2)
        first = dict(n=0, pts=int(a*48000), time_base="1/48000", rate=48000,
                     channels=1, samples=count*3, format="s16", checksum="00000000", stage="cxain")
        output = dict(first, pts=int(a*16000), time_base="1/16000", rate=16000, samples=count,
                      bytes=len(pcm), sha256=digest(pcm), stage="cxaout")
        covered = a <= window.start and a + F(count,16000) >= window.end
        payload = dict(windows=[[rational(window.start), rational(window.end)]],
            limits=dict(limits.record(), decoder_start=rational(start), decoder_end=rational(end)),
            video=[json.loads(row) for row in selection.observations],
            outcomes=[o.record() for o in selection.outcomes], audio=[dict(policy=audio_policy,
                input=[first], output=[output], samples=count, bytes=len(pcm), pcm_sha256=digest(pcm),
                sample_grid_origin=rational(a), source_phase=rational(a), coverage=[dict(
                    window=[rational(window.start), rational(window.end)],
                    status="covered" if covered else "unavailable")])],
            argv=[["HOST-video"], audio_decode_argv("HOST-ffmpeg", origin, start, end, audio_policy)])
        receipt = SourceDecodeReceipt(origin.identity, origin.decoder_id, e.DECODE_POLICY, payload)
        receipt.validate_origin(origin)
        return (DecodedWindow(receipt, selection.frames, pcm),)
    monkeypatch.setattr(e, "observe_windows", observe, raising=False)
    return origin, calls


def observed_case(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, prefix: F = F(59,1000),
                  static: bool = False) -> tuple[Any, ...]:
    import clipperx.evidence as e
    from clipperx.local_paths import source_fingerprint
    from clipperx.tests.test_visual_contracts import make_sample
    video = tmp_path / "source.identity"
    video.write_bytes(b"HOST identity placeholder")
    origin, calls = install_observer(monkeypatch, video, duration=2, prefix=prefix,
                                     step=F(1,10), static=static, width=16, height=12)
    timeline = make_timeline(source_fingerprint=source_fingerprint(video), duration_seconds=2,
        spans=[], face_tracks=[], webcam_tracks=[], samples=[make_sample(at=.08,
            sample_id="important", face_track_ids=[], webcam_track_id=None,
            persistent_rectangles=[dict(x=0,y=0,width=.1,height=.1)])])
    windows = [TimeRange(start=.07,end=.25)]
    assert hasattr(e, "prepare_visual_evidence_context"), "observed evidence preparation absent"
    context = e.prepare_visual_evidence_context(video)
    return video, timeline, windows, context, calls, origin


def test_observed_collector_binds_full_raw_jpeg_and_same_frame_aliases(monkeypatch: pytest.MonkeyPatch,
                                                                    tmp_path: Path) -> None:
    import clipperx.evidence as e
    from clipperx.render_clock import digest
    video, timeline, windows, context, calls, _ = observed_case(monkeypatch,tmp_path)
    path = e.collect_visual_evidence(video, windows, timeline=timeline, context=context,
                                     output_path=tmp_path/"initial.json", max_images=3)
    bundle = e.load_visual_evidence_bundle(path, source_video=video,timeline=timeline,
        windows=windows,context=context,max_images=3,expected_bundle_sha256=digest(path.read_bytes()))
    assert [r["requested"] for r in bundle["selected_requests"]] == ["7/100","1/5","2/25"]
    assert len(bundle["frames"]) == 2 and len(bundle["observations"]) == 1
    assert len([c for c in calls if c != "qualify"]) == 1
    assert calls.count("qualify") == 1
    assert [len(f["sample_refs"]) for f in bundle["frames"]] == [2,1]
    for frame, value in zip(bundle["frames"], [59,159]):
        raw = np.full((12,16,3),value,dtype=np.uint8)
        assert frame["raw_bgr_sha256"] == digest(raw.tobytes())
        ok, encoded = cv2.imencode(".jpg", raw, [cv2.IMWRITE_JPEG_QUALITY,85])
        assert ok and Path(frame["path"]).read_bytes() == encoded.tobytes()
        assert frame["jpeg"]["sha256"] == digest(encoded.tobytes())
    assert bundle["frames"][0]["at"] == .07
    assert bundle["frames"][0]["observed_source_start"] == "59/1000"


@pytest.mark.parametrize("prefix", [F(15,1000),F(59,1000)])
def test_observed_unavailable_prefix_never_clamps(monkeypatch: pytest.MonkeyPatch,tmp_path: Path,
                                               prefix: F) -> None:
    import clipperx.evidence as e
    video,timeline,_,context,_,_ = observed_case(monkeypatch,tmp_path,prefix=prefix)
    windows=[TimeRange(start=0,end=.1)]
    path=e.collect_visual_evidence(video,windows,timeline=timeline,context=context,
                                  output_path=tmp_path/"prefix.json",max_images=2)
    data=json.loads(path.read_text())
    assert data["outcomes"][0]["status"] == "before_video_start"
    assert data["outcomes"][0]["frame_id"] is None
    assert all(float(F(ref["requested"])) < .1 for f in data["frames"] for ref in f["sample_refs"])


def test_observed_followup_created_immutable_and_static_distinct_pts(monkeypatch: pytest.MonkeyPatch,
                                                                  tmp_path: Path) -> None:
    import clipperx.evidence as e
    from clipperx.visual_contracts import VisualEvidenceRequest
    video,timeline,windows,context,_,_=observed_case(monkeypatch,tmp_path,static=True)
    request=VisualEvidenceRequest.model_validate(dict(request_id="r1",clip_id="clip",source=windows[0],
        question="Boundary?",sample_interval_seconds=.25,reason="camera_boundary",iteration=1))
    path=e.collect_visual_evidence(video,windows,timeline=timeline,context=context,request=request,
                                  output_path=tmp_path/"followup.json",max_images=3)
    before=path.read_bytes()
    data=e.load_visual_evidence_bundle(path,source_video=video,timeline=timeline,windows=windows,
                                     context=context,request=request,max_images=3)
    assert path.read_bytes()==before and data["request"] == request.model_dump(mode="json")
    assert len({f["frame_id"] for f in data["frames"]})==2
    assert len({f["jpeg"]["sha256"] for f in data["frames"]})==1


@pytest.mark.parametrize("defect", ["hash","dangling","path","type","raw","receipt","policy"])
def test_observed_loader_rejects_drift_without_work(monkeypatch: pytest.MonkeyPatch,tmp_path: Path,
                                                  defect: str) -> None:
    import clipperx.evidence as e
    video,timeline,windows,context,calls,_=observed_case(monkeypatch,tmp_path)
    path=e.collect_visual_evidence(video,windows,timeline=timeline,context=context,
                                  output_path=tmp_path/"initial.json",max_images=3)
    data=json.loads(path.read_text())
    if defect=="hash":
        Path(data["frames"][0]["path"]).write_bytes(b"changed")
    elif defect=="dangling":
        data["observations"] = []
    elif defect=="path":
        data["catalog"][0]["path"]="../outside"
    elif defect=="type":
        data["version"]=True
    elif defect=="raw":
        data["frames"][0]["raw_pts"]+=1
    else:
        receipt_path=path.parent/path.stem/data["observations"][0]["path"]
        receipt=json.loads(receipt_path.read_text())
        if defect=="policy":
            receipt["policy"]="SELF-ASSERTED"
        else:
            receipt["payload"]["outcomes"][0]["frame_n"]=999
        receipt_path.write_text(json.dumps(receipt))
    path.write_text(json.dumps(data))
    before=list(calls)
    with pytest.raises(ValueError):
        e.load_visual_evidence_bundle(path,source_video=video,timeline=timeline,windows=windows,
                                     context=context,max_images=3)
    assert calls==before


def test_observed_quota_rejects_before_observation_and_any_output(monkeypatch: pytest.MonkeyPatch,
                                                                tmp_path: Path) -> None:
    import clipperx.evidence as e
    video,timeline,windows,context,calls,_=observed_case(monkeypatch,tmp_path)
    origin,_=install_observer(monkeypatch,video,duration=2,width=9000,height=9000)
    context=e.ObservedEvidenceContext(context.runtime,origin)
    before=list(calls)
    with pytest.raises(ValueError,match="budget"):
        e.collect_visual_evidence(video,windows,timeline=timeline,context=context,
                                  output_path=tmp_path/"quota.json",max_images=3)
    assert calls==before and not (tmp_path/"quota").exists()


def test_observed_repeated_windows_keep_all_half_open_aliases(monkeypatch: pytest.MonkeyPatch,
                                                           tmp_path: Path) -> None:
    import clipperx.evidence as e
    video,timeline,windows,context,_,_=observed_case(monkeypatch,tmp_path)
    windows*=2
    path=e.collect_visual_evidence(video,windows,timeline=timeline,context=context,
                                  output_path=tmp_path/"repeat.json",max_images=3)
    data=json.loads(path.read_text())
    assert all(r["window_indices"]==[0,1] for f in data["frames"] for r in f["sample_refs"])


def test_observed_releases_batch_and_array_before_next_dispatch(monkeypatch: pytest.MonkeyPatch,tmp_path: Path) -> None:
    import weakref
    import clipperx.evidence as e
    from clipperx.source_decode import ObservedFrame
    video=tmp_path/"source.identity"
    video.write_bytes(b"HOST")
    origin,_=install_observer(monkeypatch,video)
    timeline=make_timeline(source_fingerprint=origin.fingerprint,duration_seconds=100,spans=[],samples=[],
                           face_tracks=[],webcam_tracks=[])
    windows=[TimeRange(start=1,end=10),TimeRange(start=30,end=40)]
    previous: list[Any]=[]
    arrays: list[Any]=[]
    observe=e.observe_windows
    array_copy=ObservedFrame.array_copy
    class Holder:
        def __init__(self,result: Any) -> None:
            self.receipt,self.frames,self.pcm=result.receipt,result.frames,result.pcm
    def boundary(*args: Any,**kwargs: Any) -> Any:
        assert all(ref() is None for ref in previous+arrays)
        result=Holder(observe(*args,**kwargs)[0])
        previous.append(weakref.ref(result))
        return (result,)
    def copy(frame: Any) -> Any:
        result=array_copy(frame)
        arrays.append(weakref.ref(result))
        return result
    monkeypatch.setattr(e,"observe_windows",boundary)
    monkeypatch.setattr(ObservedFrame,"array_copy",copy)
    e.collect_visual_evidence(video,windows,timeline=timeline,
        output_path=tmp_path/"release.json",max_images=4)
    assert len(previous)==4 and all(ref() is None for ref in previous+arrays)


def test_observed_checks_encoded_quota_before_file_write(monkeypatch: pytest.MonkeyPatch,tmp_path: Path) -> None:
    import clipperx.evidence as e
    video,timeline,windows,context,_,_=observed_case(monkeypatch,tmp_path)
    monkeypatch.setattr(e,"MAX_ARTIFACT_BYTES",64)
    with pytest.raises(ValueError,match="quota"):
        e.collect_visual_evidence(video,windows,timeline=timeline,context=context,
                                  output_path=tmp_path/"quota.json",max_images=3)
    assert (not (tmp_path/"quota").exists() or not list((tmp_path/"quota").iterdir()))
    assert not (tmp_path/"quota.json").exists()


def test_observed_exact_coverage_endpoint_selects_next_frame(monkeypatch: pytest.MonkeyPatch,tmp_path: Path) -> None:
    import clipperx.evidence as e
    video,timeline,_,context,_,_=observed_case(monkeypatch,tmp_path)
    windows=[TimeRange(start=.159,end=.21)]
    path=e.collect_visual_evidence(video,windows,timeline=timeline,context=context,
                                  output_path=tmp_path/"endpoint.json",max_images=2)
    assert all(frame["raw_pts"]==159000 for frame in json.loads(path.read_text())["frames"])


def test_loader_rejects_self_asserted_policy_even_with_all_hashes_recomputed(monkeypatch: pytest.MonkeyPatch,
                                                                          tmp_path: Path) -> None:
    import clipperx.evidence as e
    from clipperx.render_clock import digest,canonical
    video,timeline,windows,context,_,_=observed_case(monkeypatch,tmp_path)
    path=e.collect_visual_evidence(video,windows,timeline=timeline,context=context,
                                  output_path=tmp_path/"policy.json",max_images=3)
    data=json.loads(path.read_text())
    observation=data["observations"][0]
    receipt_path=path.parent/path.stem/observation["path"]
    receipt=json.loads(receipt_path.read_text())
    receipt["policy"]="SELF-ASSERTED"
    receipt["sha256"]=digest(canonical({k:v for k,v in receipt.items() if k!="sha256"}))
    encoded=canonical(receipt).encode()
    receipt_path.write_bytes(encoded)
    for row in data["catalog"]+[observation]:
        if row["path"]==observation["path"]:
            row.update(bytes=len(encoded),sha256=digest(encoded))
    observation["observation_id"]=receipt["sha256"]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="policy"):
        e.load_visual_evidence_bundle(path,source_video=video,timeline=timeline,windows=windows,
                                     context=context,max_images=3)


@pytest.mark.parametrize("defect",["raw_checksum","pcm_block"])
def test_collector_validates_actual_immutable_result_before_writes(monkeypatch: pytest.MonkeyPatch,
                                                                tmp_path: Path,defect: str) -> None:
    import clipperx.evidence as e
    from clipperx.source_decode import DecodedWindow,ObservedFrame,VideoTag
    from clipperx.render_clock import SourceDecodeReceipt
    video,timeline,windows,context,_,_=observed_case(monkeypatch,tmp_path)
    original=e.observe_windows
    def forged(*args: Any,**kwargs: Any) -> Any:
        result=original(*args,**kwargs)[0]
        payload=json.loads(result.receipt.payload_json)
        frames=list(result.frames)
        if defect=="raw_checksum":
            frame=frames[0]
            tag=VideoTag(frame.tag.n,frame.tag.pts,frame.tag.time_base,frame.tag.width,frame.tag.height,
                         frame.tag.checksum^1)
            frames[0]=ObservedFrame(tag,frame.raw_end,frame.bgr)
            payload["video"][tag.n]["checksum"]=f"{tag.checksum:08X}"
        else:
            payload["audio"][0]["output"][0]["sha256"]="f"*64
        receipt=SourceDecodeReceipt(context.origin.identity,context.origin.decoder_id,e.DECODE_POLICY,payload)
        return (DecodedWindow(receipt,tuple(frames),result.pcm),)
    monkeypatch.setattr(e,"observe_windows",forged)
    with pytest.raises(ValueError):
        e.collect_visual_evidence(video,windows,timeline=timeline,context=context,
                                  output_path=tmp_path/"forged.json",max_images=3)


def test_partition_splits_before_tiny_interval_exceeds_two_seconds(monkeypatch: pytest.MonkeyPatch,tmp_path: Path) -> None:
    import clipperx.evidence as e
    video,_,_,context,_,_=observed_case(monkeypatch,tmp_path)
    origin,_=install_observer(monkeypatch,video,duration=5)
    batches=e._batches([dict(requested="1/1"),dict(requested="5999999/2000000")],origin)
    assert len(batches)==2 and all(w.end-w.start<=2 for w,_ in batches)


def test_collector_reserves_known_bundle_json_before_observation(monkeypatch: pytest.MonkeyPatch,tmp_path: Path) -> None:
    import clipperx.evidence as e
    video,timeline,windows,context,calls,_=observed_case(monkeypatch,tmp_path)
    before=list(calls)
    monkeypatch.setattr(e,"MAX_ARTIFACT_BYTES",512)
    with pytest.raises(ValueError,match="quota"):
        e.collect_visual_evidence(video,windows,timeline=timeline,context=context,
                                  output_path=tmp_path/"oversize.json",max_images=3)
    assert calls==before


def test_collector_checks_encoded_bytes_before_jpeg_write(monkeypatch: pytest.MonkeyPatch,tmp_path: Path) -> None:
    import clipperx.evidence as e
    from types import SimpleNamespace
    video,timeline,windows,context,_,_=observed_case(monkeypatch,tmp_path)
    class OversizeBytes(bytes):
        def __len__(self) -> int:
            return e.MAX_ARTIFACT_BYTES+1
    monkeypatch.setattr(cv2,"imencode",lambda *a:(True,SimpleNamespace(tobytes=lambda:OversizeBytes(b"synthetic quota sentinel"))))
    with pytest.raises(ValueError,match="quota"):
        e.collect_visual_evidence(video,windows,timeline=timeline,context=context,
                                  output_path=tmp_path/"encoded.json",max_images=3)
    assert not (tmp_path/"encoded.json").exists()
    assert not list((tmp_path/"encoded").glob("*.jpg"))


def test_visual_frame_times_prioritize_boundary_events_over_grid() -> None:
    timeline = make_timeline(samples=[], face_tracks=[], webcam_tracks=[], spans=[
        make_span(source=TimeRange(start=0, end=6), streamer_track_id=None,
                  identity_status="unavailable", webcam_track_id=None),
        make_span(span_id="next", source=TimeRange(start=6, end=10), streamer_track_id=None,
                  identity_status="unavailable", webcam_track_id=None)])
    times = visual_frame_times([TimeRange(start=4, end=9)], timeline, max_images=5)
    assert len(times) <= 5
    assert 6 in times and 4 in times
    assert all(4 <= at < 9 for at in times)


def test_visual_frame_times_reject_outside_source() -> None:
    with pytest.raises(ValueError):
        visual_frame_times([TimeRange(start=9, end=11)], make_timeline(), max_images=5)


def test_face_positions_require_ordered_evidence_bijection() -> None:
    from clipperx.evidence import associated_face_positions
    from clipperx.tests.test_visual_contracts import make_face_track, make_sample
    face = make_face_track(first_seen=1, last_seen=8, observation_count=3,
        positions=[dict(x=x, y=.1, width=.1, height=.1) for x in (.05, .6, .8)],
        evidence_ids=["a", "b", "c"])
    samples = [make_sample(sample_id=str(i), at=at, evidence_ids=[e], webcam_track_id=None)
               for i, (at, e) in enumerate([(1, "a"), (3, "b"), (8, "c")])]
    timeline = make_timeline(samples=samples, face_tracks=[face], webcam_tracks=[], spans=[])
    positions = associated_face_positions(timeline, face)
    assert positions is not None and [p[0] for p in positions] == [1, 3, 8]
    from clipperx.visual_director import _protected_regions
    from clipperx.tests.test_visual_contracts import make_shot
    shot = make_shot(source=TimeRange(start=2, end=4))
    protected = _protected_regions(shot, timeline, [])
    assert protected == [face.positions[1]]
    broken = face.model_copy(update={"evidence_ids": ["b", "a", "c"]})
    assert associated_face_positions(timeline, broken) is None
