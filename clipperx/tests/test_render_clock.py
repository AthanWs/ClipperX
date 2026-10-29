"""Exact common source clock, detached records and strict receipts."""

from dataclasses import FrozenInstanceError
from fractions import Fraction as F
import importlib.util
import json
from typing import Any

import pytest


def clock_module() -> Any:
    assert importlib.util.find_spec("clipperx.render_clock") is not None, "clock API absent"
    from clipperx import render_clock
    return render_clock


def make_origin() -> Any:
    c = clock_module()
    streams = [c.StreamClock(0, "video", F(1, 1000), F(5015, 1000), F(7), 2, 2),
               c.StreamClock(1, "audio", F(1, 44100), F(5), F(7),
                             sample_rate=44100, channels=1, sample_format="s16")]
    colors = {"color_range": "tv"}
    origin = c.SourceTimeOrigin("HOST", 12, 34, "0123456789abcdef", F(5), F(2),
                                streams, "paired-host", colors)
    streams.clear()
    colors["color_range"] = "pc"
    return origin


def stream_recipe() -> tuple[Any, Any, dict[str, Any], Any]:
    """Selected §11.5 constructor evidence, never an observed-media receipt."""
    import zlib
    from clipperx import source_decode as d

    c, origin = clock_module(), make_origin()
    limits = c.StreamDecodeLimits(
        preroll=F(0),
        postroll=F(0),
        max_window=F(1),
        timeout=3,
        max_frames=3,
        max_samples=4800,
        max_pixels=4,
        max_decoded_bytes=30000,
        window_timeout=6,
        cleanup_grace=1,
        max_requests=3,
        max_deliveries=2,
        io_chunk_bytes=64,
        max_live_payload_bytes=40000,
        max_selected_unique_bytes=30000,
        max_request_emission_bytes=30000,
        max_finalization_bytes=100000,
    )
    requests = (F(1, 50), F(1, 25), F(3, 50))
    old_limits = d.DecodeLimits(
        **{k: getattr(limits, k) for k in limits.operation_record()}, max_windows=1
    )
    bodies = [bytes([n]) * 12 for n in range(3)]
    tags = [
        d.VideoTag(n, pts, F(1, 1000), 2, 2, zlib.adler32(bodies[n], 0))
        for n, pts in enumerate((5015, 5050, 5090))
    ]
    selection = d.select_frames(zip(tags, bodies), origin, requests, old_limits)
    video = [json.loads(row) for row in selection.observations]
    pcm = bytes(9600)
    audio = {
        "policy": "render48k-mono",
        "input": [d.AudioTag(0, 220500, 44100, 1, 4410, "s16", 0, "cxain").record()],
        "output": [
            {
                **d.AudioTag(0, 240000, 48000, 1, 4800, "s16", 0, "cxaout").record(),
                "bytes": len(pcm),
                "sha256": c.digest(pcm),
            }
        ],
        "samples": 4800,
        "bytes": len(pcm),
        "pcm_sha256": c.digest(pcm),
        "sample_grid_origin": "5/1",
        "source_phase": "0/1",
        "coverage": [{"window": ["0/1", "1/10"], "status": "covered"}],
    }
    argv = [
        c.video_decode_argv("HOST-ffmpeg", origin, F(0), F(1, 10)),
        c.audio_decode_argv("HOST-ffmpeg", origin, F(0), F(1, 10), audio["policy"]),
    ]
    old_payload = {
        "windows": [["0/1", "1/10"]],
        "limits": {**old_limits.record(), "decoder_start": "0/1", "decoder_end": "1/10"},
        "video": video,
        "audio": [audio],
        "outcomes": [o.record() for o in selection.outcomes],
        "argv": argv,
    }
    old = c.SourceDecodeReceipt(origin.identity, origin.decoder_id, d.DECODE_POLICY, old_payload)
    old.validate_origin(origin)
    session = "00000000-0000-4000-8000-000000000001"
    payload: dict[str, Any] = {
        "window": ["0/1", "1/10"],
        "requests": [c.rational(t) for t in requests],
        "operation_limits": limits.operation_record(),
        "delivery_limits": limits.delivery_record(),
        "decoder_start": "0/1",
        "decoder_end": "1/10",
        "video": video,
        "audio": audio,
        "outcomes": old_payload["outcomes"],
        "argv": argv,
    }
    header = {
        k: payload[k]
        for k in (
            "window",
            "requests",
            "operation_limits",
            "delivery_limits",
            "decoder_start",
            "decoder_end",
        )
    }
    header.update(
        chain="selected-stream-v1",
        session_id=session,
        origin_id=origin.identity,
        decoder_id=origin.decoder_id,
        policy=c.STREAM_DECODE_POLICY,
        audio_policy=audio["policy"],
    )
    previous = c.digest(c.canonical(header))
    deliveries = []
    for i, ordinals in enumerate(((0, 1), (2,))):
        successor = {**video[i + 1], "coverage_end": None}
        provisional = d.ProvisionalSelection(
            session,
            i,
            ordinals,
            selection.frames[i],
            video[i],
            successor,
            previous_chain_sha256=previous,
        )
        deliveries.append(provisional.delivery_record())
        previous = provisional.delivery_chain_sha256
    payload["deliveries"] = deliveries
    payload["accounting"] = dict(
        decoded_video_frames=3,
        decoded_video_bytes=36,
        decoded_audio_output_bytes=9600,
        selected_unique_frames=2,
        selected_unique_bytes=24,
        offered_deliveries=2,
        offered_payload_bytes=24,
        acknowledged_deliveries=2,
        acknowledged_requests=3,
        acknowledged_request_bytes=36,
        reserved_payload_capacity_bytes=38400,
        peak_live_payload_bytes_accounted=28800,
        metadata_bytes=len(c.canonical(payload).encode()),
        video_log_bytes=600,
        audio_log_bytes=600,
    )
    payload["completion"] = {
        "children": [
            dict(
                kind=kind,
                exit_code=0,
                started_ns=i * 10**9,
                finished_ns=(i + 1) * 10**9,
                cleanup_complete=True,
            )
            for i, kind in enumerate(("video", "audio"))
        ],
        "source_fingerprint_after": origin.fingerprint,
        "pins_verified_before_children": True,
        "all_requests_acknowledged": True,
        "cleanup_complete": True,
        "elapsed_ns": 3 * 10**9,
    }
    return origin, limits, payload, old


def test_stream_limits_exact_records_and_host_ceiling() -> None:
    c = clock_module()
    assert hasattr(c, "StreamDecodeLimits"), "stream limits API absent"
    limits = c.StreamDecodeLimits()
    assert len(limits.__dataclass_fields__) == 19
    assert len(limits.operation_record()) == 9 and len(limits.delivery_record()) == 10
    assert (
        c.StreamDecodeLimits.from_records(limits.operation_record(), limits.delivery_record())
        == limits
    )
    for key in limits.__dataclass_fields__:
        ceiling = getattr(limits, key)
        for bad in (True, 1.5, "1", ceiling + 1):
            with pytest.raises(ValueError):
                c.StreamDecodeLimits(**{key: bad})
    with pytest.raises(ValueError):
        c.StreamDecodeLimits.from_records({}, limits.delivery_record())


def repair_stream_metadata(origin: Any, session: str, payload: dict[str, Any]) -> None:
    """Repair metadata identities only, so negative tests reach their owning rule."""
    c = clock_module()
    previous = c.digest(c.canonical(c._stream_header(origin, session, payload)))
    for delivery in payload["deliveries"]:
        n = delivery["frame_n"]
        previous = c._stream_link(previous, delivery, payload["video"][n], payload["video"][n + 1])
        delivery["chain_sha256"] = previous
    payload["accounting"]["metadata_bytes"] = len(c.canonical({k: v for k, v in payload.items()
        if k not in ("accounting", "completion")}).encode("utf-8"))


@pytest.mark.parametrize("mapping", ["dict", "proxy", "tight_cap"])
def test_stream_finalization_real_positive_recipe_and_chain(mapping: str) -> None:
    from dataclasses import replace
    from types import MappingProxyType

    c = clock_module()
    assert hasattr(c, "SourceStreamFinalization"), "stream finalization API absent"
    origin, limits, payload, old = stream_recipe()
    expected_limits = limits
    if mapping == "tight_cap":
        payload["delivery_limits"]["max_finalization_bytes"] = 20000
        repair_stream_metadata(origin, "00000000-0000-4000-8000-000000000001", payload)
        expected_limits = replace(limits, max_finalization_bytes=20000)
    receipt = c.SourceStreamFinalization(
        origin,
        "00000000-0000-4000-8000-000000000001",
        MappingProxyType(payload) if mapping == "proxy" else payload,
        limits=limits,
    )
    encoded = receipt.to_json()
    loaded = c.SourceStreamFinalization.from_json(
        encoded,
        origin=origin,
        limits=limits,
        policy=c.STREAM_DECODE_POLICY,
        expected_session_id=receipt.session_id,
    )
    assert loaded.record() == receipt.record() and loaded.sha256 == receipt.sha256
    assert loaded.requests == (F(1, 50), F(1, 25), F(3, 50))
    assert loaded.window == (F(0), F(1, 10)) and loaded.limits == expected_limits
    assert len(encoded.encode("utf-8")) <= expected_limits.max_finalization_bytes
    assert [v["coverage_end"] for v in payload["video"]] == ["101/20", "509/100", None]
    assert json.loads(old.payload_json)["outcomes"] == payload["outcomes"]
    payload["video"].clear()
    detached = loaded.record()
    detached["payload"].clear()
    assert loaded.to_json() == encoded


@pytest.mark.parametrize(
    "fault",
    [
        "schema",
        "kind",
        "version_bool",
        "session",
        "origin",
        "decoder",
        "policy",
        "hash",
        "duplicate",
        "float",
        "nan",
        "nesting",
        "utf8",
        "oversize",
        "rational",
        "rational_bits",
        "integer_bits",
        "bool_pts",
        "row_order",
        "row_dimension",
        "successor",
        "terminal",
        "unavailable",
        "request_order",
        "request_count",
        "limits",
        "host_limit",
        "argv",
        "argv_executable",
        "argv_count",
        "ordinal",
        "delivery_index",
        "chain",
        "resource",
        "metadata",
        "account_bool",
        "log_zero",
        "reservation",
        "peak",
        "selected_limit",
        "audio_count",
        "audio_phase",
        "audio_coverage",
        "audio_shape",
        "exit",
        "child_order",
        "child_deadline",
        "window_deadline",
        "cleanup",
        "completion_bool",
        "fingerprint",
        "effective_constructor", "effective_loader", "effective_padding",
    ],
)
def test_stream_finalization_rejects_schema_identity_and_chain_corruption(fault: str) -> None:
    from dataclasses import replace

    c = clock_module()
    origin, limits, payload, _ = stream_recipe()
    receipt = c.SourceStreamFinalization(
        origin, "00000000-0000-4000-8000-000000000001", payload, limits=limits
    )
    envelope = receipt.record()
    p = envelope["payload"]
    if fault.startswith("effective_"):
        cap = 20000 if fault == "effective_padding" else 1
        p["delivery_limits"]["max_finalization_bytes"] = cap
        repair_stream_metadata(origin, receipt.session_id, p)
        # Real non-envelope validation proves chain/association/accounting are coherent.
        assert (
            c._validate_stream_payload(origin, receipt.session_id, p, limits).max_finalization_bytes
            == cap
        )
        envelope.pop("sha256")
        encoded = c.canonical({**envelope, "sha256": c.digest(c.canonical(envelope))})
        if fault == "effective_padding":
            assert len(encoded.encode("utf-8")) < cap
            encoded = " " * (cap - len(encoded.encode("utf-8")) + 1) + encoded
        assert cap < len(encoded.encode("utf-8")) < limits.max_finalization_bytes
        with pytest.raises(ValueError, match="byte|expansion|string limit"):
            if fault == "effective_constructor":
                c.SourceStreamFinalization(origin, receipt.session_id, p, limits=limits)
            else:
                c.SourceStreamFinalization.from_json(
                    encoded,
                    origin=origin,
                    limits=limits,
                    policy=c.STREAM_DECODE_POLICY,
                    expected_session_id=receipt.session_id,
                )
        return
    host = limits
    if fault == "schema":
        p["unknown"] = 1
    elif fault == "kind":
        envelope["kind"] = "audio_only"
    elif fault == "version_bool":
        envelope["version"] = True
    elif fault == "session":
        envelope["session_id"] = "00000000-0000-4000-8000-000000000002"
    elif fault == "origin":
        envelope["origin_id"] = "0" * 64
    elif fault == "decoder":
        envelope["decoder_id"] = "stale"
    elif fault == "policy":
        envelope["policy"] = "HOST"
    elif fault == "rational":
        p["requests"][0] = "2/100"
    elif fault == "rational_bits":
        p["video"][0]["time_base"] = f"{2**4096}/{2**4096}"
    elif fault == "integer_bits":
        p["video"][0]["pts"] = 2**4096
    elif fault == "bool_pts":
        p["video"][0]["pts"] = True
    elif fault == "row_order":
        p["video"][1]["pts"] = 5014
    elif fault == "row_dimension":
        p["video"][1]["width"] = 1
    elif fault == "successor":
        p["video"][1]["sha256"] = "0" * 64
    elif fault == "terminal":
        p["video"][-1]["coverage_end"] = "6/1"
    elif fault == "unavailable":
        p["outcomes"][0] = dict(requested="1/50", status="unknown_terminal", frame_n=None)
    elif fault == "request_order":
        p["requests"][:2] = reversed(p["requests"][:2])
    elif fault == "request_count":
        p["requests"] *= 2
    elif fault == "limits":
        p["delivery_limits"]["max_live_selections"] = 2
    elif fault == "host_limit":
        host = replace(limits, max_requests=2)
    elif fault == "argv":
        p["argv"][0][-1] = "pipe:2"
    elif fault == "argv_executable":
        p["argv"][1][0] = "different"
    elif fault == "argv_count":
        p["argv"][0] += ["x"] * 128
    elif fault == "ordinal":
        p["deliveries"][0]["request_ordinals"] = [0, 2]
    elif fault == "delivery_index":
        p["deliveries"][0]["index"] = True
    elif fault == "chain":
        p["deliveries"][0]["chain_sha256"] = "0" * 64
    elif fault == "resource":
        p["accounting"]["offered_payload_bytes"] += 12
    elif fault == "metadata":
        p["accounting"]["metadata_bytes"] += 1
    elif fault == "account_bool":
        p["accounting"]["acknowledged_requests"] = True
    elif fault == "log_zero":
        p["accounting"]["audio_log_bytes"] = 0
    elif fault == "reservation":
        p["accounting"]["reserved_payload_capacity_bytes"] -= 1
    elif fault == "peak":
        p["accounting"]["peak_live_payload_bytes_accounted"] = 40001
    elif fault == "selected_limit":
        p["delivery_limits"]["max_selected_unique_bytes"] = 12
    elif fault == "audio_count":
        p["audio"]["input"][0]["samples"] -= 20
    elif fault == "audio_phase":
        p["audio"]["source_phase"] = "1/1"
    elif fault == "audio_coverage":
        p["audio"]["coverage"][0]["status"] = "unavailable"
    elif fault == "audio_shape":
        p["audio"]["input"] = {}
    elif fault == "exit":
        p["completion"]["children"][1]["exit_code"] = 1
    elif fault == "child_order":
        p["completion"]["children"].reverse()
    elif fault == "child_deadline":
        p["completion"]["children"][1]["finished_ns"] = 4 * 10**9
        p["completion"]["elapsed_ns"] = 5 * 10**9
    elif fault == "window_deadline":
        p["completion"]["elapsed_ns"] = 6 * 10**9
    elif fault == "cleanup":
        p["completion"]["children"][0]["cleanup_complete"] = False
    elif fault == "completion_bool":
        p["completion"]["cleanup_complete"] = 1
    elif fault == "fingerprint":
        p["completion"]["source_fingerprint_after"] = "f" * 16
    # Repair unrelated chain/metadata identities when probing deeper owning validators.
    if fault not in ("chain", "successor"):
        previous = c.digest(c.canonical(c._stream_header(origin, receipt.session_id, p)))
        for delivery in p["deliveries"]:
            n = delivery["frame_n"]
            previous = c._stream_link(previous, delivery, p["video"][n], p["video"][n + 1])
            delivery["chain_sha256"] = previous
    if fault != "metadata":
        p["accounting"]["metadata_bytes"] = len(
            c.canonical(
                {k: v for k, v in p.items() if k not in ("accounting", "completion")}
            ).encode()
        )
    # Rehash so rejection probes the real owning validator, rather than the outer checksum.
    envelope.pop("sha256")
    text = c.canonical({**envelope, "sha256": c.digest(c.canonical(envelope))})
    if fault == "hash":
        text = text.replace(receipt.origin_id, "f" * 64)
    elif fault == "duplicate":
        text = text.replace('"version":1', '"version":1,"version":1')
    elif fault == "float":
        text = text.replace('"version":1', '"version":1.0')
    elif fault == "nan":
        text = text.replace('"version":1', '"version":NaN')
    elif fault == "nesting":
        text = "[" * 11 + "0" + "]" * 11
    elif fault == "utf8":
        text = '"\ud800"'
    elif fault == "oversize":
        text = " " * (limits.max_finalization_bytes + 1)
    with pytest.raises(ValueError):
        c.SourceStreamFinalization.from_json(
            text,
            origin=origin,
            limits=host,
            policy=c.STREAM_DECODE_POLICY,
            expected_session_id=receipt.session_id,
        )


@pytest.mark.parametrize("placement", ["original", "legacy_window", "legacy_windows", "legacy_audio",
    "legacy_input", "legacy_output", "legacy_coverage", "legacy_coverage_window", "legacy_argv",
    "legacy_command", "stream_window", "stream_requests", "stream_audio_output"])
def test_stream_finalization_preserves_old_receipt_bytes_and_retention(placement: str) -> None:
    from clipperx import source_decode as d

    c = clock_module()
    if placement.startswith("legacy_"):
        origin = make_origin()
        audio = audio_payload()
        reference = c.SourceAudioDecodeReceipt(
            origin.identity, origin.decoder_id, d.AUDIO_DECODE_POLICY, audio
        )
        reference.validate_origin(origin)
        if placement == "legacy_window":
            audio["windows"][0] = tuple(audio["windows"][0])
        elif placement == "legacy_windows":
            audio["windows"] = tuple(audio["windows"])
        elif placement == "legacy_audio":
            audio["audio"] = tuple(audio["audio"])
        elif placement == "legacy_input":
            audio["audio"][0]["input"] = tuple(audio["audio"][0]["input"])
        elif placement == "legacy_output":
            audio["audio"][0]["output"] = tuple(audio["audio"][0]["output"])
        elif placement == "legacy_coverage":
            audio["audio"][0]["coverage"] = tuple(audio["audio"][0]["coverage"])
        elif placement == "legacy_coverage_window":
            audio["audio"][0]["coverage"][0]["window"] = tuple(
                audio["audio"][0]["coverage"][0]["window"]
            )
        elif placement == "legacy_argv":
            audio["argv"] = tuple(audio["argv"])
        elif placement == "legacy_command":
            audio["argv"][0] = tuple(audio["argv"][0])
        normalized = c.SourceAudioDecodeReceipt(
            origin.identity, origin.decoder_id, d.AUDIO_DECODE_POLICY, audio
        )
        normalized.validate_origin(origin)
        assert normalized.to_json() == reference.to_json()
        assert (
            c.digest(normalized.to_json())
            == "ba5dc2c418be997dca4314e46c47795dd23199a15d7e1fdb34022591b866857f"
        )
        return
    origin, limits, payload, old = stream_recipe()
    if placement.startswith("stream_"):
        if placement == "stream_window":
            payload["window"] = tuple(payload["window"])
        elif placement == "stream_requests":
            payload["requests"] = tuple(payload["requests"])
        elif placement == "stream_audio_output":
            payload["audio"]["output"] = tuple(payload["audio"]["output"])
        with pytest.raises(ValueError):
            c.SourceStreamFinalization(
                origin, "00000000-0000-4000-8000-000000000001", payload, limits=limits
            )
        return
    old_text = old.to_json()
    stream = c.SourceStreamFinalization(
        origin, "00000000-0000-4000-8000-000000000001", payload, limits=limits
    )
    assert (
        c.SourceDecodeReceipt.from_json(old_text, origin=origin, policy=d.DECODE_POLICY).to_json()
        == old_text
    )
    assert c.digest(old_text) == "9d45ad2573a5c95952c553f314501c945b6dd61e05e28340182fdb25c337e9fe"
    assert payload["argv"][0] == [
        "HOST-ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-nostats",
        "-loglevel",
        "info",
        "-copyts",
        "-t",
        "0.100000",
        "-threads",
        "2",
        "-i",
        "HOST",
        "-map",
        "0:0",
        "-an",
        "-sn",
        "-dn",
        "-vf",
        "trim=end=5.100000,format=bgr24,showinfo@cxv",
        "-fps_mode",
        "passthrough",
        "-enc_time_base",
        "filter",
        "-threads",
        "1",
        "-c:v",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]
    for loader, policy in (
        (c.SourceDecodeReceipt, d.DECODE_POLICY),
        (c.SourceAudioDecodeReceipt, d.AUDIO_DECODE_POLICY),
    ):
        with pytest.raises(ValueError):
            loader.from_json(stream.to_json(), origin=origin, policy=policy)
    with pytest.raises(ValueError):
        c.SourceStreamFinalization.from_json(
            old_text,
            origin=origin,
            policy=c.STREAM_DECODE_POLICY,
            limits=limits,
            expected_session_id=stream.session_id,
        )
    old_payload = json.loads(old.payload_json)
    old_payload["limits"]["max_retained_bytes"] = 9623
    with pytest.raises(ValueError):
        c.SourceDecodeReceipt(origin.identity, origin.decoder_id, d.DECODE_POLICY, old_payload)


def valid_payload() -> dict[str, Any]:
    return {"windows": [["0/1", "1/10"]], "limits": {
        "preroll": "0/1", "postroll": "0/1", "max_window": "1/1", "timeout": 3,
        "max_frames": 10, "max_samples": 10000, "max_pixels": 4,
        "max_decoded_bytes": 30000, "max_retained_bytes": 20000,
        "max_log_bytes": 2000000, "max_windows": 1, "decoder_start": "0/1", "decoder_end": "1/10"},
        "video": [{"n": n, "pts": pts, "time_base": "1/1000", "width": 2,
                   "height": 2, "format": "bgr24", "checksum": "00000000", "bytes": 12,
                   "sha256": "f" * 64, "coverage_end": "101/20" if n == 0 else None}
                  for n, pts in enumerate((5015, 5050))],
        "audio": [{"policy": "render48k-mono", "input": [{"n": 0, "pts": 220500,
            "time_base": "1/44100", "rate": 44100, "channels": 1, "samples": 4410,
            "format": "s16", "checksum": "00000000", "stage": "cxain"}],
            "output": [{"n": 0, "pts": 240000, "time_base": "1/48000", "rate": 48000,
                "channels": 1, "samples": 4800, "format": "s16", "checksum": "00000000",
                "stage": "cxaout", "bytes": 9600, "sha256": "f" * 64}],
            "samples": 4800, "bytes": 9600, "pcm_sha256": "f" * 64,
            "sample_grid_origin": "5/1", "source_phase": "0/1",
            "coverage": [{"window": ["0/1", "1/10"], "status": "covered"}]}],
        "outcomes": [{"requested": "1/50", "status": "covered", "frame_n": 0}],
        "argv": [["ffmpeg", "video"], ["ffmpeg", "audio"]]}


def envelope(payload: dict[str, Any]) -> str:
    c, origin = clock_module(), make_origin()
    body = {"version": 1, "scope": "window", "origin_id": origin.identity,
            "decoder_id": origin.decoder_id, "policy": "HOST", "payload": payload}
    return c.canonical({**body, "sha256": c.digest(c.canonical(body))})


@pytest.mark.parametrize("fault", ["limits_schema", "negative_limit", "timeout", "frame_budget",
    "sample_budget", "pixel_budget", "byte_budget", "retained_budget", "origin_window",
    "decoder_bounds", "input_rate", "input_channels", "input_format", "conversion_count",
    "initial_phase", "source_phase", "audio_outside_decoder", "coverage", "coverage_window", "coverage_missing",
    "bool_channels", "bool_n", "bool_version", "malformed_window", "malformed_coverage",
    "malformed_policy", "malformed_input", "missing_audio", "missing_argv"])
def test_valid_envelope_independent_contradictions_rejected(fault: str) -> None:
    c, origin = clock_module(), make_origin()
    payload = valid_payload()
    assert c.SourceDecodeReceipt.from_json(envelope(payload), origin=origin, policy="HOST")
    audio = payload["audio"][0]
    if fault == "limits_schema":
        payload["limits"] = {"frames": -1}
    elif fault in ("negative_limit", "timeout", "frame_budget", "sample_budget", "pixel_budget",
                   "byte_budget", "retained_budget"):
        key, value = {"negative_limit": ("max_frames", -1), "timeout": ("timeout", 46),
            "frame_budget": ("max_frames", 1), "sample_budget": ("max_samples", 100),
            "pixel_budget": ("max_pixels", 1), "byte_budget": ("max_decoded_bytes", 10),
            "retained_budget": ("max_retained_bytes", 100)}[fault]
        payload["limits"][key] = value
    elif fault == "origin_window":
        payload["windows"] = [["0/1", "3/1"]]
        payload["limits"].update(decoder_end="3/1", max_window="4/1")
        audio["coverage"] = [{"window": ["0/1", "3/1"], "status": "unavailable"}]
    elif fault == "decoder_bounds":
        payload["limits"]["decoder_end"] = "2/1"
    elif fault == "input_rate":
        audio["input"][0].update(rate=48000, time_base="1/48000", pts=240000, samples=4800)
    elif fault == "input_channels":
        audio["input"][0]["channels"] = 2
    elif fault == "input_format":
        audio["input"][0]["format"] = "fltp"
    elif fault == "conversion_count":
        audio["input"][0]["samples"] = 441
    elif fault == "initial_phase":
        audio["input"][0]["pts"] -= 441
    elif fault == "source_phase":
        audio["source_phase"] = "999/1"
    elif fault == "audio_outside_decoder":
        audio["input"][0]["pts"] += 44100
        audio["output"][0]["pts"] += 48000
        audio["sample_grid_origin"] = "6/1"
        audio["source_phase"] = "1/1"
        audio["coverage"][0]["status"] = "unavailable"
    elif fault == "coverage":
        audio["coverage"][0]["status"] = "unavailable"
    elif fault == "coverage_window":
        audio["coverage"][0]["window"] = ["0/1", "1/1"]
    elif fault == "coverage_missing":
        audio["coverage"] = []
    elif fault == "bool_channels":
        audio["input"][0]["channels"] = True
    elif fault == "bool_n":
        audio["input"][0]["n"] = False
    elif fault == "malformed_window":
        payload["windows"] = [None]
    elif fault == "malformed_coverage":
        audio["coverage"] = {"bad": []}
    elif fault == "malformed_policy":
        audio["policy"] = []
    elif fault == "malformed_input":
        audio["input"] = [None]
    elif fault == "missing_audio":
        payload["audio"] = []
    elif fault == "missing_argv":
        payload["argv"] = []
    text = envelope(payload)
    if fault == "bool_version":
        body = json.loads(text)
        body["version"] = True
        body.pop("sha256")
        text = c.canonical({**body, "sha256": c.digest(c.canonical(body))})
    with pytest.raises(ValueError):
        c.SourceDecodeReceipt.from_json(text, origin=origin, policy="HOST")


@pytest.mark.parametrize("field", ["sample_format", "kind", "path", "fingerprint", "decoder_id"])
def test_identity_scalars_reject_mutable_impostors(field: str) -> None:
    c = clock_module()
    mutable = ["s16"]
    with pytest.raises(ValueError):
        if field in ("sample_format", "kind"):
            kwargs: dict[str, Any] = {"sample_format": "s16", "kind": "audio"}
            kwargs[field] = mutable
            c.StreamClock(1, kwargs["kind"], F(1, 44100), F(5), F(7),
                          sample_rate=44100, channels=1, sample_format=kwargs["sample_format"])
        else:
            kwargs = {"path": "HOST", "fingerprint": "0123456789abcdef", "decoder_id": "paired-host"}
            kwargs[field] = mutable
            c.SourceTimeOrigin(kwargs["path"], 12, 34, kwargs["fingerprint"], F(5), F(2),
                               make_origin().streams, kwargs["decoder_id"], {})


def test_exact_common_origin_and_caller_isolation() -> None:
    origin = make_origin()
    assert origin.to_raw(F(15, 1000)) == F(5015, 1000)
    assert origin.to_source(F(5015, 1000)) == F(15, 1000)
    assert origin.streams[1].start == F(5)
    assert origin.color == (("color_range", "tv"),)
    assert origin.streams[0].start != origin.streams[1].start
    with pytest.raises(FrozenInstanceError):
        origin.duration = F(3)
    for bad in (F(-1), F(3), float("nan"), float("inf")):
        with pytest.raises(ValueError):
            origin.to_raw(bad)


def test_unknown_bounds_are_explicit_and_invalid_timebases_rejected() -> None:
    c = clock_module()
    stream = c.StreamClock(0, "video", F(1, 1000), None, None, 2, 2)
    assert stream.start is None and stream.end is None
    with pytest.raises(ValueError):
        c.StreamClock(0, "video", F(0), F(0), F(1), 2, 2)
    with pytest.raises(ValueError):
        c.StreamClock(0, "video", None, F(0), F(1), 2, 2)
    with pytest.raises(ValueError):
        c.StreamClock(0, "video", F(1), F(3), F(1), 2, 2)


def test_receipt_is_detached_canonical_and_rejects_corruption_or_stale_identity() -> None:
    c = clock_module()
    origin = make_origin()
    payload = valid_payload()
    receipt = c.SourceDecodeReceipt(origin.identity, origin.decoder_id, "HOST", payload)
    encoded = receipt.to_json()
    payload["video"].append({"bad": "mutation"})
    assert receipt.to_json() == encoded
    assert c.SourceDecodeReceipt.from_json(encoded, origin=origin, policy="HOST") == receipt
    with pytest.raises(ValueError, match="identity"):
        c.SourceDecodeReceipt.from_json(encoded.replace(origin.identity, "0" * 64),
                                        origin=origin, policy="HOST")
    data = json.loads(encoded)
    data["payload"]["limits"]["max_frames"] = 31
    with pytest.raises(ValueError, match="hash"):
        c.SourceDecodeReceipt.from_json(json.dumps(data), origin=origin, policy="HOST")
    payload["limits"]["max_frames"] = float("nan")
    with pytest.raises(ValueError):
        c.SourceDecodeReceipt(origin.identity, origin.decoder_id, "HOST", payload)


def test_observation_receipts_reject_invented_records_even_with_recomputed_hash() -> None:
    c = clock_module()
    origin = make_origin()
    payload = valid_payload()
    payload["outcomes"][0]["frame_n"] = 99
    with pytest.raises(ValueError, match="observation|outcome|coverage"):
        c.SourceDecodeReceipt(origin.identity, origin.decoder_id, "HOST", payload)


def test_origin_rejects_mutable_stream_impostors() -> None:
    c = clock_module()
    class Mutable:
        index = 0
    with pytest.raises(ValueError):
        c.SourceTimeOrigin("HOST", 12, 34, "0123456789abcdef", F(5), F(2),
                           [Mutable()], "paired-host", {})


def audio_payload() -> dict[str, Any]:
    from clipperx.source_decode import AudioDecodeLimits
    payload = valid_payload()
    payload.pop("video")
    payload.pop("outcomes")
    payload["limits"] = {**AudioDecodeLimits(preroll=F(0), postroll=F(0)).record(),
                         "decoder_start": "0/1", "decoder_end": "1/10"}
    payload["argv"] = [["ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-loglevel", "info",
        "-copyts", "-t", "0.100000", "-threads", "2", "-i", "HOST",
        "-map", "0:1", "-vn", "-sn", "-dn", "-af",
        "atrim=end=5.100000,asettb=1/44100,ashowinfo@cxain,aresample=48000:async=0,"
        "aformat=sample_fmts=s16:channel_layouts=mono,asettb=1/48000,ashowinfo@cxaout",
        "-c:a", "pcm_s16le", "-threads", "1", "-f", "s16le", "pipe:1"]]
    return payload


def audio_envelope(payload: dict[str, Any], **changes: Any) -> str:
    c, origin = clock_module(), make_origin()
    body = {"version": 1, "kind": "audio_only", "scope": "window",
        "origin_id": origin.identity, "decoder_id": origin.decoder_id,
        "policy": "HOST;audio-only;v1", "payload": payload, **changes}
    return c.canonical({**body, "sha256": c.digest(c.canonical(body))})


def test_audio_only_receipt_is_detached_and_cross_schema_loads_fail() -> None:
    c, origin = clock_module(), make_origin()
    payload = audio_payload()
    encoded_combined = envelope(valid_payload())
    combined = c.SourceDecodeReceipt.from_json(encoded_combined, origin=origin, policy="HOST")
    assert combined.to_json() == encoded_combined
    receipt = c.SourceAudioDecodeReceipt.from_json(audio_envelope(payload), origin=origin,
                                                 policy="HOST;audio-only;v1")
    payload["audio"][0]["samples"] = 9
    assert json.loads(receipt.payload_json)["audio"][0]["samples"] == 4800
    with pytest.raises(FrozenInstanceError):
        receipt.policy = "mutated"
    with pytest.raises(ValueError):
        c.SourceDecodeReceipt.from_json(receipt.to_json(), origin=origin, policy=receipt.policy)
    with pytest.raises(ValueError):
        c.SourceAudioDecodeReceipt.from_json(encoded_combined, origin=origin, policy="HOST")
    with pytest.raises(ValueError, match="Duplicate"):
        c.SourceAudioDecodeReceipt.from_json(receipt.to_json().replace('"version":1',
            '"version":1,"version":1'), origin=origin, policy=receipt.policy)
    with pytest.raises(ValueError):
        c.SourceAudioDecodeReceipt.from_json(receipt.to_json().replace('"samples":4800',
            '"samples":4801'), origin=origin, policy=receipt.policy)


@pytest.mark.parametrize("fault", ["kind", "version", "origin", "decoder", "policy", "video",
    "windows", "dummy_argv", "extra_argv", "argv_video", "argv_source", "argv_map",
    "decoder_bounds", "effective_limit", "origin_bounds", "limits_schema", "limit_bool",
    "limit_ceiling", "block_limit", "sample_limit", "byte_limit", "retain_limit",
    "input_rate", "input_channels", "input_format", "conversion", "phase", "source_phase",
    "coverage", "coverage_shape", "input_shape", "output_shape", "bool_n", "bool_samples",
    "timebase", "output_bytes", "output_hash", "aggregate_hash", "missing_audio"])
def test_audio_only_rehashed_contradictions_fail(fault: str) -> None:
    c, origin = clock_module(), make_origin()
    payload = audio_payload()
    assert c.SourceAudioDecodeReceipt.from_json(audio_envelope(payload), origin=origin,
                                               policy="HOST;audio-only;v1")
    audio = payload["audio"][0]
    changes: dict[str, Any] = {}
    if fault in ("kind", "version", "origin", "decoder", "policy"):
        key = {"origin": "origin_id", "decoder": "decoder_id"}.get(fault, fault)
        changes[key] = True if fault == "version" else "wrong"
    elif fault == "video":
        payload["video"] = []
    elif fault == "windows":
        payload["windows"] *= 2
    elif fault == "dummy_argv":
        payload["argv"] = [["ffmpeg", "audio"]]
    elif fault == "extra_argv":
        payload["argv"] *= 2
    elif fault == "argv_video":
        payload["argv"][0].extend(["-vf", "showinfo"])
    elif fault in ("argv_source", "argv_map"):
        flag = "-i" if fault == "argv_source" else "-map"
        payload["argv"][0][payload["argv"][0].index(flag) + 1] = "wrong"
    elif fault == "decoder_bounds":
        payload["limits"]["decoder_end"] = "1/5"
    elif fault == "effective_limit":
        payload["limits"]["max_window"] = "1/20"
    elif fault == "origin_bounds":
        payload["windows"][0][1] = "3/1"
        payload["limits"]["decoder_end"] = "3/1"
    elif fault == "limits_schema":
        payload["limits"]["max_pixels"] = 4
    elif fault in ("limit_bool", "limit_ceiling", "block_limit", "sample_limit", "byte_limit", "retain_limit"):
        key, value = {"limit_bool": ("timeout", True), "limit_ceiling": ("max_blocks", 65537),
            "block_limit": ("max_blocks", 0), "sample_limit": ("max_samples", 100),
            "byte_limit": ("max_decoded_bytes", 10), "retain_limit": ("max_retained_bytes", 10)}[fault]
        payload["limits"][key] = value
    elif fault == "input_rate":
        audio["input"][0].update(rate=48000, time_base="1/48000", pts=240000, samples=4800)
    elif fault == "input_channels":
        audio["input"][0]["channels"] = 2
    elif fault == "input_format":
        audio["input"][0]["format"] = "fltp"
    elif fault == "conversion":
        audio["input"][0]["samples"] = 441
    elif fault == "phase":
        audio["input"][0]["pts"] -= 441
    elif fault == "source_phase":
        audio["source_phase"] = "999/1"
    elif fault == "coverage":
        audio["coverage"][0]["status"] = "unavailable"
    elif fault == "coverage_shape":
        audio["coverage"] = [None]
    elif fault in ("input_shape", "output_shape"):
        audio[fault.split("_")[0]] = [None]
    elif fault == "bool_n":
        audio["output"][0]["n"] = False
    elif fault == "bool_samples":
        audio["input"][0]["samples"] = True
    elif fault == "timebase":
        audio["output"][0]["time_base"] = "1/44100"
    elif fault == "output_bytes":
        audio["output"][0]["bytes"] -= 1
    elif fault == "output_hash":
        audio["output"][0]["sha256"] = []
    elif fault == "aggregate_hash":
        audio["pcm_sha256"] = []
    elif fault == "missing_audio":
        payload["audio"] = []
    with pytest.raises(ValueError):
        c.SourceAudioDecodeReceipt.from_json(audio_envelope(payload, **changes), origin=origin,
                                             policy="HOST;audio-only;v1")


def test_audio_only_mutable_scalar_impostors_rejected() -> None:
    c, origin = clock_module(), make_origin()
    class MutableString(str):
        pass
    class MutableInt(int):
        pass
    for field, value in (("policy", MutableString("render48k-mono")), ("samples", MutableInt(4800))):
        payload = audio_payload()
        payload["audio"][0][field] = value
        with pytest.raises(ValueError):
            c.SourceAudioDecodeReceipt(origin.identity, origin.decoder_id, "HOST;audio-only;v1", payload)


def audio_seek_payload() -> dict[str, Any]:
    payload = audio_payload()
    payload["windows"] = [["1/2", "3/5"]]
    payload["limits"].update(decoder_start="1/2", decoder_end="3/5")
    audio = payload["audio"][0]
    audio["input"][0]["pts"] = 242550
    audio["output"][0]["pts"] = 264000
    audio.update(sample_grid_origin="11/2", source_phase="1/2",
        coverage=[{"window": ["1/2", "3/5"], "status": "covered"}])
    command = payload["argv"][0]
    command[command.index("-copyts") + 1:command.index("-copyts") + 1] = ["-ss", "0.500000"]
    graph_index = command.index("-af") + 1
    command[graph_index] = command[graph_index].replace("end=5.100000", "end=5.600000")
    return payload


@pytest.mark.parametrize("seek", [False, True])
def test_audio_only_canonical_receipt_accepts_zero_and_nonzero_seek(seek: bool) -> None:
    c, origin = clock_module(), make_origin()
    payload = audio_seek_payload() if seek else audio_payload()
    receipt = c.SourceAudioDecodeReceipt.from_json(audio_envelope(payload), origin=origin,
                                                 policy="HOST;audio-only;v1")
    assert json.loads(receipt.payload_json) == payload


@pytest.mark.parametrize("fault", ["resample_rate", "layout", "input_timebase", "output_timebase",
    "trim_end", "duration999", "seek", "missing_seek", "duplicate_seek", "gain_after",
    "padding_after", "async", "gain_before", "post_observation_pts", "extra_ar", "extra_ac",
    "extra_sample_format", "extra_output", "duplicate_t", "duplicate_input", "duplicate_map",
    "duplicate_af", "missing_t", "missing_map", "missing_vn", "missing_dn", "extra_codec",
    "extra_format", "threads", "trim_ignores_origin"])
def test_audio_only_command_contradictions_rehashed_rejected(fault: str) -> None:
    c, origin = clock_module(), make_origin()
    payload = audio_seek_payload()
    assert c.SourceAudioDecodeReceipt.from_json(audio_envelope(payload), origin=origin,
                                               policy="HOST;audio-only;v1")
    command = payload["argv"][0]
    graph_index = command.index("-af") + 1
    graph = command[graph_index]
    substitutions = {
        "resample_rate": ("aresample=48000:", "aresample=16000:"),
        "layout": ("channel_layouts=mono", "channel_layouts=stereo"),
        "input_timebase": ("asettb=1/44100", "asettb=1/48000"),
        "output_timebase": ("asettb=1/48000", "asettb=1/16000"),
        "trim_end": ("end=5.600000", "end=5.700000"),
        "trim_ignores_origin": ("end=5.600000", "end=0.600000"),
        "async": ("async=0", "async=1")}
    if fault in substitutions:
        command[graph_index] = graph.replace(*substitutions[fault])
    elif fault in ("gain_after", "padding_after", "post_observation_pts"):
        command[graph_index] += {"gain_after": ",volume=0", "padding_after": ",apad",
                                 "post_observation_pts": ",asetpts=PTS-STARTPTS"}[fault]
    elif fault == "gain_before":
        command[graph_index] = "volume=0," + graph
    elif fault in ("duration999", "seek", "threads"):
        flag, value = {"duration999": ("-t", "999.000000"), "seek": ("-ss", "0.250000"),
                        "threads": ("-threads", "8")}[fault]
        command[command.index(flag) + 1] = value
    elif fault.startswith("missing_"):
        flag = {"missing_seek": "-ss", "missing_t": "-t", "missing_map": "-map",
                "missing_vn": "-vn", "missing_dn": "-dn"}[fault]
        index = command.index(flag)
        del command[index:index + (1 if flag in ("-vn", "-dn") else 2)]
    elif fault.startswith("duplicate_"):
        flag = {"duplicate_seek": "-ss", "duplicate_t": "-t", "duplicate_input": "-i",
                "duplicate_map": "-map", "duplicate_af": "-af"}[fault]
        index = command.index(flag)
        command[-1:-1] = command[index:index + 2]
    else:
        command[-1:-1] = {"extra_ar": ["-ar", "16000"], "extra_ac": ["-ac", "2"],
            "extra_sample_format": ["-sample_fmt", "flt"], "extra_output": ["other-output.wav"],
            "extra_codec": ["-acodec", "pcm_s24le"], "extra_format": ["-f", "wav"]}[fault]
    with pytest.raises(ValueError):
        c.SourceAudioDecodeReceipt.from_json(audio_envelope(payload), origin=origin,
                                             policy="HOST;audio-only;v1")


@pytest.mark.parametrize("start,end,policy,rate,layout,seek,trim,duration", [
    (F(0), F(1, 10), "analysis16k-mono", 16000, "mono", [], "5.100000", "0.100000"),
    (F(1, 2), F(3, 5), "render48k-mono", 48000, "mono", ["-ss", "0.500000"], "5.600000", "0.100000"),
    (F(1, 3), F(2, 3), "render48k-stereo", 48000, "stereo", ["-ss", "0.333333"], "5.666666", "0.333333")])
def test_audio_canonical_argv_preserves_prior_literal(
        start: F, end: F, policy: str, rate: int, layout: str, seek: list[str],
        trim: str, duration: str) -> None:
    c, origin = clock_module(), make_origin()
    assert hasattr(c, "audio_decode_argv"), "Pure safe canonical audio command absent"
    # Literal prior execution command; not equality between two callers of a new builder.
    expected = ["ffmpeg", "-hide_banner", "-nostdin", "-nostats", "-loglevel", "info",
        "-copyts", *seek, "-t", duration, "-threads", "2", "-i", "HOST", "-map", "0:1",
        "-vn", "-sn", "-dn", "-af",
        f"atrim=end={trim},asettb=1/44100,ashowinfo@cxain,aresample={rate}:async=0,"
        f"aformat=sample_fmts=s16:channel_layouts={layout},asettb=1/{rate},ashowinfo@cxaout",
        "-c:a", "pcm_s16le", "-threads", "1", "-f", "s16le", "pipe:1"]
    assert c.audio_decode_argv("ffmpeg", origin, start, end, policy) == expected
def test_float_audio_limits_and_exact_argv_policy() -> None:
    from fractions import Fraction as F
    from clipperx.render_clock import (float_audio_decode_argv, source_decoder_identity,
                                      SourceTimeOrigin, StreamClock, audio_decode_argv)
    from clipperx.source_decode import FloatAudioDecodeLimits
    import pytest
    runtime = dict(runtime_id="HOST", ffmpeg="O:/HOST/ffmpeg.exe", ffprobe="O:/HOST/ffprobe.exe",
                   ffmpeg_sha256="A" * 64, ffprobe_sha256="B" * 64)
    origin = SourceTimeOrigin("O:/HOST/float.nut", 16, 1, "0123456789abcdef", F(0), F(1), (
        StreamClock(0, "video", F(1, 60), F(0), F(1), width=1, height=1),
        StreamClock(1, "audio", F(1, 48000), F(0), F(1), sample_rate=48000,
                    channels=1, sample_format="flt")), source_decoder_identity(runtime), {})
    limits = FloatAudioDecodeLimits()
    assert limits.max_window == 180 and limits.max_block_samples == 65536
    command = float_audio_decode_argv(runtime["ffmpeg"], origin, F(0), F(1, 12000),
                                      "render48k-f32-mono-v1")
    old = audio_decode_argv(runtime["ffmpeg"], origin, F(0), F(1, 12000), "render48k-mono")
    assert command[:18] == old[:18]
    assert command[-7:] == ["-c:a", "pcm_f32le", "-threads", "1", "-f", "f32le", "pipe:1"]
    assert command[command.index("-af") + 1] == (
        "atrim=end=0.000083,asettb=1/48000,ashowinfo@cxain,"
        "aresample=48000:async=0:osf=flt:tsf=fltp,"
        "aformat=sample_fmts=flt:channel_layouts=mono,asettb=1/48000,ashowinfo@cxaout")
    for kwargs in (dict(timeout=True), dict(max_blocks=16385), dict(preroll=F(3, 4)),
                   dict(max_window=F(181)), dict(io_chunk_bytes=65537), dict(preroll=0)):
        with pytest.raises(ValueError):
            FloatAudioDecodeLimits(**kwargs)
    with pytest.raises(ValueError):
        float_audio_decode_argv(runtime["ffmpeg"], origin, F(0), F(1, 10000000),
                                "render48k-f32-mono-v1")
    for kwargs in (dict(max_samples=1 << 4097), dict(timeout=0), dict(postroll=F(1, 2)),
                   dict(preroll=F(1, 1 << 4097))):
        with pytest.raises(ValueError):
            FloatAudioDecodeLimits(**kwargs)
    long_origin = SourceTimeOrigin(origin.path, origin.size, origin.mtime_ns, origin.fingerprint,
        F(0), F(1000), origin.streams, origin.decoder_id, {})
    assert "180.000000" in float_audio_decode_argv(runtime["ffmpeg"], long_origin, F(0), F(180),
                                                   "render48k-f32-mono-v1")
    with pytest.raises(ValueError):
        float_audio_decode_argv(runtime["ffmpeg"], long_origin, F(0), F(180001, 1000),
                                "render48k-f32-mono-v1")


def _float_recipe(channels: int = 1, input_rate: int = 48000,
                  input_format: str = "flt") -> tuple[Any, dict[str, Any], Any, bytes, dict[str, Any]]:
    from fractions import Fraction as F
    import zlib
    from clipperx.render_clock import (StreamClock, SourceTimeOrigin, source_decoder_identity,
                                      float_audio_decode_argv, digest)
    from clipperx.source_decode import FloatAudioDecodeLimits
    runtime = dict(runtime_id="HOST", ffmpeg="O:/HOST/ffmpeg.exe", ffprobe="O:/HOST/ffprobe.exe",
                   ffmpeg_sha256="A" * 64, ffprobe_sha256="B" * 64)
    origin = SourceTimeOrigin("O:/HOST/float.nut", 16, 1, "0123456789abcdef", F(0), F(1), (
        StreamClock(0, "video", F(1, 60), F(0), F(1), width=1, height=1),
        StreamClock(1, "audio", F(1, input_rate), F(0), F(1), sample_rate=input_rate,
                    channels=channels, sample_format=input_format)), source_decoder_identity(runtime), {})
    limits = FloatAudioDecodeLimits(preroll=F(0), postroll=F(0), max_window=F(1), timeout=1,
        max_blocks=2, max_samples=8, max_block_samples=8, max_decoded_bytes=32,
        max_retained_bytes=32, max_log_bytes=4096, max_metadata_bytes=8192, io_chunk_bytes=16)
    pcm = (bytes.fromhex("00000000 0000803f 000080bf 00008035") if channels == 1 else
           bytes.fromhex("00000000 00000080 0000803f 01000000 000080bf 0000c03f 00008035 000000b5"))
    policy = "render48k-f32-mono-v1" if channels == 1 else "render48k-f32-stereo-v1"
    block = dict(n=0, pts=0, time_base=f"1/{input_rate}", rate=input_rate, channels=channels, samples=4,
                 format=input_format, checksum=f"{zlib.adler32(pcm, 0):08X}", stage="cxain")
    audio = dict(policy=policy, input=[block], output=[dict(block, time_base="1/48000", rate=48000,
        format="flt", stage="cxaout", bytes=len(pcm), sha256=digest(pcm))], samples=4, bytes=len(pcm), pcm_sha256=digest(pcm),
        sample_grid_origin="0/1", source_phase="0/1",
        coverage=[dict(window=["0/1", "1/12000"], status="covered")])
    payload = dict(windows=[["0/1", "1/12000"]], limits=dict(limits.record(),
        decoder_start="0/1", decoder_end="1/12000"), audio=[audio],
        argv=[float_audio_decode_argv(runtime["ffmpeg"], origin, F(0), F(1, 12000), policy)])
    return origin, runtime, limits, pcm, payload


def _a4_input_grid_recipe() -> tuple[Any, dict[str, Any], Any, dict[str, Any]]:
    """A4 literal times/counts in aggregate metadata, not a native PCM proof."""
    from clipperx.source_decode import FloatAudioDecodeLimits

    c = clock_module()
    _, runtime, _, _, _ = _float_recipe(2, 44100, "fltp")
    origin = c.SourceTimeOrigin("HOST", 72618, 1, "0123456789abcdef", F(5, 4),
        F(151161, 50000), (
            c.StreamClock(0, "video", F(1, 44100), F(5, 4), F(17, 4), 2, 2),
            c.StreamClock(1, "audio", F(1, 44100), F(5, 4), F(188449, 44100),
                          sample_rate=44100, channels=2, sample_format="fltp")),
        c.source_decoder_identity(runtime), {})
    limits = FloatAudioDecodeLimits()
    window = ["1013/1000", "1713/1000"]
    block = dict(n=0, pts=77748, time_base="1/44100", rate=44100, channels=2,
                 samples=63945, format="fltp", checksum="00000000", stage="cxain")
    # Split the aggregate output to obey the existing per-block ceiling.
    output = [dict(block, n=n, pts=pts, time_base="1/48000", rate=48000,
                   samples=count, format="flt", stage="cxaout", bytes=count * 8,
                   sha256="f" * 64)
              for n, pts, count in ((0, 84624, 65536), (1, 150160, 4064))]
    audio = dict(policy="render48k-f32-stereo-v1", input=[block], output=output,
        samples=69600, bytes=556800, pcm_sha256="f" * 64,
        sample_grid_origin="1763/1000", source_phase="513/1000",
        coverage=[dict(window=window, status="covered")])
    payload = dict(windows=[window], limits=dict(limits.record(),
        decoder_start="513/1000", decoder_end="1963/1000"), audio=[audio],
        argv=[c.float_audio_decode_argv(runtime["ffmpeg"], origin, F(513, 1000),
                                      F(1963, 1000), audio["policy"])])
    return origin, runtime, limits, payload


def test_float_audio_a4_literal_input_grid_roundtrip() -> None:
    c = clock_module()
    origin, runtime, limits, payload = _a4_input_grid_recipe()
    assert F(1763, 1000) - F(77748, 44100) == F(1, 147000)
    command = payload["argv"][0]
    assert command[command.index("-ss") + 1] == "0.513000"
    receipt = c.SourceFloatAudioDecodeReceipt(origin, payload, runtime=runtime,
                                             limits=limits.record())
    wire = receipt.to_json()
    assert c.SourceFloatAudioDecodeReceipt.from_json(wire, origin=origin, runtime=runtime,
                                                    limits=limits.record()).to_json() == wire


def _input_grid_s16_recipe(rate: int, output_rate: int, raw_origin: F, start: F,
                           stream_start: F | None, first_pts: int) -> tuple[Any, dict[str, Any]]:
    """Small exact metadata recipe; the expected input coordinate comes from literals."""
    c = clock_module()
    origin = c.SourceTimeOrigin("HOST", 12, 34, "0123456789abcdef", raw_origin, F(2), (
        c.StreamClock(1, "audio", F(1, rate), stream_start, raw_origin + 2,
                      sample_rate=rate, channels=1, sample_format="s16"),), "paired-host", {})
    payload = audio_payload()
    end = start + F(1, 10)
    window = [c.rational(start), c.rational(end)]
    payload["windows"] = [window]
    payload["limits"].update(decoder_start=c.rational(start), decoder_end=c.rational(end))
    audio = payload["audio"][0]
    audio["policy"] = "analysis16k-mono" if output_rate == 16000 else "render48k-mono"
    audio["input"][0].update(pts=first_pts, rate=rate, time_base=f"1/{rate}", samples=rate // 20)
    # Pick an output grid point within the unchanged one-output-sample phase rule.
    output_pts = first_pts * output_rate // rate
    output_start = F(output_pts, output_rate)
    audio["output"][0].update(pts=output_pts, rate=output_rate, time_base=f"1/{output_rate}",
                              samples=output_rate // 20, bytes=output_rate // 10)
    audio.update(samples=output_rate // 20, bytes=output_rate // 10,
        sample_grid_origin=c.rational(output_start), source_phase=c.rational(output_start - raw_origin),
        coverage=[dict(window=window, status="unavailable")])
    payload["argv"] = [c.audio_decode_argv("ffmpeg", origin, start, end, audio["policy"])]
    return origin, payload


@pytest.mark.parametrize("rate", [44100, 48000])
@pytest.mark.parametrize("output_rate", [16000, 48000])
@pytest.mark.parametrize("case", ["positive_origin", "zero_origin", "negative_origin",
    "no_seek_positive", "no_seek_zero", "no_seek_negative", "later_stream", "unknown_stream",
    "positive_half", "negative_half", "below_positive_half", "above_positive_half",
    "below_negative_half", "above_negative_half", "relative_cli_floor", "stream_half",
    "negative_stream_half", "no_seek_positive_half", "no_seek_negative_half"])
def test_audio_input_grid_lower_bound_and_earlier_neighbor(rate: int, output_rate: int,
                                                           case: str) -> None:
    c = clock_module()
    start, raw_origin, stream_start = F(513, 1000), F(5, 4), None
    expected = 77748 if rate == 44100 else 84624
    if case == "positive_origin":
        stream_start = raw_origin
    elif case in ("zero_origin", "unknown_stream"):
        raw_origin = F(0)
        expected = 22623 if rate == 44100 else 24624
        stream_start = F(0) if case == "zero_origin" else None
    elif case == "negative_origin":
        raw_origin = F(-5, 4)
        stream_start = raw_origin
        expected = -32502 if rate == 44100 else -35376
    elif case in ("no_seek_positive_half", "no_seek_negative_half"):
        start = F(0)
        raw_origin = F(21 if case == "no_seek_positive_half" else -21, 2 * rate)
        stream_start = raw_origin
        expected = 11 if case == "no_seek_positive_half" else -11
    elif case.startswith("no_seek_"):
        start = F(0)
        raw_origin = {"no_seek_positive": F(5, 4), "no_seek_zero": F(0),
                      "no_seek_negative": F(-5, 4)}[case]
        stream_start = raw_origin
        expected = int(raw_origin * rate)
    elif case == "later_stream":
        raw_origin, stream_start = F(0), F(11, 20)
        expected = 24255 if rate == 44100 else 26400
    elif case == "stream_half":
        start, raw_origin = F(0), F(0)
        stream_start, expected = F(21, 2 * rate), 11
    elif case == "negative_stream_half":
        start, raw_origin = F(0), F(-100, rate)
        stream_start, expected = F(-21, 2 * rate), -11
    else:
        start = F(1, 2)
        target, expected = {
            "positive_half": (F(21, 2), 11),
            "negative_half": (F(-21, 2), -11),
            "below_positive_half": (F(1049, 100), 10),
            "above_positive_half": (F(1051, 100), 11),
            "below_negative_half": (F(-1051, 100), -11),
            "above_negative_half": (F(-1049, 100), -10),
            "relative_cli_floor": (F(21, 2), 10),
        }[case]
        raw_origin = target / rate - start
        if case == "relative_cli_floor":
            start = F(1, 3)
            raw_origin = target / rate - F(333333, 1000000) - F(1, 10000000)
            assert (raw_origin + start) * rate > F(21, 2)
    for delta in (0, 1, -1):
        origin, payload = _input_grid_s16_recipe(rate, output_rate, raw_origin, start,
                                               stream_start, expected + delta)
        receipt = c.SourceAudioDecodeReceipt(origin.identity, origin.decoder_id,
                                             "HOST;audio-only;v1", payload)
        if delta < 0:
            # Count, output phase and coverage are coherent even for the forged neighbor.
            with pytest.raises(ValueError, match="outside selected source/decoder bounds"):
                c.SourceAudioDecodeReceipt.from_json(receipt.to_json(), origin=origin,
                                                     policy=receipt.policy)
        else:
            assert c.SourceAudioDecodeReceipt.from_json(receipt.to_json(), origin=origin,
                                                        policy=receipt.policy) == receipt


@pytest.mark.parametrize("fault", ["earlier_input", "upper_input", "output_end", "count",
                                   "output_phase", "source_phase", "coverage"])
def test_float_audio_a4_input_grid_other_invariants_unchanged(fault: str) -> None:
    c = clock_module()
    origin, runtime, limits, payload = _a4_input_grid_recipe()
    receipt = c.SourceFloatAudioDecodeReceipt(origin, payload, runtime=runtime, limits=limits.record())
    envelope = receipt.record()
    audio = envelope["payload"]["audio"][0]
    match = "outside selected source/decoder bounds"
    if fault in ("earlier_input", "output_phase"):
        audio["input"][0]["pts"] -= 1 if fault == "earlier_input" else 2
        if fault == "earlier_input":
            for block in audio["output"]:
                block["pts"] -= 1
            audio.update(sample_grid_origin="84623/48000", source_phase="24623/48000")
        else:
            match = "conversion count/phase"
    elif fault == "upper_input":
        audio["input"][0]["samples"] += 2
        audio["samples"] += 2
        audio["bytes"] += 16
        audio["output"][-1]["samples"] += 2
        audio["output"][-1]["bytes"] += 16
    elif fault == "output_end":
        # Isolate this origin predicate; conversion count/phase also reject this
        # contradiction in the full loader before the redundant end check.
        for block in audio["output"]:
            block["pts"] += 2
        audio.update(sample_grid_origin=c.rational(F(84626, 48000)),
                     source_phase=c.rational(F(24626, 48000)))
        with pytest.raises(ValueError, match=match):
            c._validate_audio_origin(envelope["payload"], origin, origin.streams[1],
                                     F(513, 1000), F(1963, 1000), [(F(1013, 1000), F(1713, 1000))])
        match = "conversion count/phase"
    elif fault == "count":
        audio["input"][0]["samples"] -= 2
        match = "conversion count/phase"
    elif fault == "source_phase":
        audio["source_phase"] = "0/1"
        match = "source phase"
    elif fault == "coverage":
        audio["coverage"][0]["status"] = "unavailable"
        match = "coverage contradicts"
    envelope["sha256"] = c.digest(c.canonical({k: v for k, v in envelope.items() if k != "sha256"}))
    with pytest.raises(ValueError, match=match):
        c.SourceFloatAudioDecodeReceipt.from_json(c.canonical(envelope), origin=origin,
                                                 runtime=runtime, limits=limits.record())


def test_stream_audio_input_grid_rejects_consistently_shifted_earlier_neighbor() -> None:
    c = clock_module()
    origin, limits, payload, _ = stream_recipe()
    session = "00000000-0000-4000-8000-000000000001"
    receipt = c.SourceStreamFinalization(origin, session, payload, limits=limits)
    assert c.SourceStreamFinalization.from_json(receipt.to_json(), origin=origin, limits=limits,
        policy=c.STREAM_DECODE_POLICY, expected_session_id=session).to_json() == receipt.to_json()
    payload["audio"]["input"][0]["pts"] -= 1
    payload["audio"]["output"][0]["pts"] -= 1
    payload["audio"].update(sample_grid_origin="239999/48000", source_phase="-1/48000")
    repair_stream_metadata(origin, session, payload)
    with pytest.raises(ValueError, match="outside selected source/decoder bounds"):
        c.SourceStreamFinalization(origin, session, payload, limits=limits)


@pytest.mark.parametrize("channels,input_rate,input_format", [(1, 48000, "flt"), (2, 48000, "fltp"),
                                                             (1, 44100, "fltp"), (2, 44100, "flt")])
def test_float_audio_receipt_real_roundtrip_and_detachment(channels: int, input_rate: int, input_format: str) -> None:
    from clipperx.render_clock import SourceFloatAudioDecodeReceipt, digest, canonical
    origin, runtime, limits, pcm, payload = _float_recipe(channels, input_rate, input_format)
    receipt = SourceFloatAudioDecodeReceipt(origin, payload, runtime=runtime, limits=limits.record())
    wire = receipt.to_json()
    assert len(wire.encode()) <= 8192
    assert set(receipt.record()) == set("version kind scope origin_id origin_decoder_id operation_id policy runtime payload sha256".split())
    assert receipt.origin_decoder_id == origin.decoder_id
    assert receipt.operation_id == digest(canonical(dict(origin_decoder_id=origin.decoder_id,
        runtime=runtime, policy=receipt.policy, audio_policy=payload["audio"][0]["policy"])))
    loaded = SourceFloatAudioDecodeReceipt.from_json(wire, origin=origin, runtime=runtime,
                                                     limits=limits.record())
    assert loaded.to_json() == wire and loaded.sha256 == receipt.sha256
    payload["audio"][0]["output"][0]["bytes"] = 999
    runtime["runtime_id"] = "changed"
    detached = receipt.record()
    detached["payload"]["audio"].clear()
    assert receipt.to_json() == wire and len(pcm) == 16 * channels


def test_float_audio_receipt_rejects_runtime_origin_schema_or_budget_corruption() -> None:
    from fractions import Fraction
    import copy
    import json
    import pytest
    from clipperx.render_clock import SourceFloatAudioDecodeReceipt, canonical, digest
    origin, runtime, limits, _, payload = _float_recipe()
    receipt = SourceFloatAudioDecodeReceipt(origin, payload, runtime=runtime, limits=limits.record())
    def rejected(change: Any) -> None:
        value = receipt.record()
        change(value)
        value["sha256"] = digest(canonical({k: v for k, v in value.items() if k != "sha256"}))
        with pytest.raises(ValueError):
            SourceFloatAudioDecodeReceipt.from_json(canonical(value), origin=origin,
                runtime=runtime, limits=limits.record())
    for field in ("kind", "scope", "origin_id", "origin_decoder_id", "operation_id", "policy"):
        rejected(lambda v, field=field: v.__setitem__(field, "corrupt"))
    rejected(lambda v: v.__setitem__("version", True))
    # JSON bool and integer compare equal in Python; version requires a plain int
    # even when the original integer envelope hash is retained.
    with pytest.raises(ValueError):
        SourceFloatAudioDecodeReceipt.from_json(receipt.to_json().replace('"version":1', '"version":true'),
            origin=origin, runtime=runtime, limits=limits.record())
    rejected(lambda v: v["runtime"].__setitem__("ffmpeg", "other.exe"))
    rejected(lambda v: v["payload"]["argv"][0].__setitem__(0, "other.exe"))
    rejected(lambda v: v["payload"]["windows"][0].__setitem__(0, "0.0"))
    for field, bad in (("samples", True), ("bytes", 15), ("sample_grid_origin", "1/48000"),
                       ("source_phase", "1/48000"), ("pcm_sha256", "X" * 64)):
        rejected(lambda v, field=field, bad=bad: v["payload"]["audio"][0].__setitem__(field, bad))
    for field, bad in (("n", 1), ("pts", True), ("time_base", "2/96000"), ("format", "s16"),
                       ("rate", 44100), ("channels", 2), ("bytes", 8), ("checksum", "bad")):
        def corrupt_block(value: dict[str, Any], field: str = field, bad: Any = bad) -> None:
            block = value["payload"]["audio"][0]["output"][0]
            block[field] = bad
            if field == "rate":
                block["time_base"] = "1/44100"
            if field == "channels":
                block["bytes"] = value["payload"]["audio"][0]["bytes"] = 32
        rejected(corrupt_block)
    rejected(lambda v: v["payload"]["audio"][0]["coverage"][0].__setitem__("status", "unavailable"))
    rejected(lambda v: v["payload"]["audio"][0]["input"][0].__setitem__("format", "s16"))
    rejected(lambda v: v["payload"]["audio"].append(v["payload"]["audio"][0]))
    rejected(lambda v: v["payload"].__setitem__("video", []))
    rejected(lambda v: v["runtime"].__setitem__("ffmpeg_sha256", "a" * 64))
    rejected(lambda v: v["payload"]["limits"].__setitem__("decoder_end", "1/6000"))
    for time in ("1e999999999", "1/0", "01/2", "1/" + "1" * 2470):
        rejected(lambda v, time=time: v["payload"]["audio"][0].__setitem__("source_phase", time))
    # A different current runtime cannot load even the original intact envelope.
    with pytest.raises(ValueError):
        SourceFloatAudioDecodeReceipt.from_json(receipt.to_json(), origin=origin,
            runtime=dict(runtime, ffprobe="O:/OTHER/ffprobe.exe"), limits=limits.record())
    # Cross-schema loading remains strict in both directions.
    from clipperx.render_clock import SourceAudioDecodeReceipt
    with pytest.raises(ValueError):
        SourceAudioDecodeReceipt.from_json(receipt.to_json(), origin=origin, policy="HOST;audio-only;v1")
    with pytest.raises(ValueError):
        SourceFloatAudioDecodeReceipt.from_json(audio_envelope(audio_payload()), origin=origin,
                                                runtime=runtime, limits=limits.record())
    bad_payload = copy.deepcopy(payload)
    bad_payload["audio"][0]["input"][0]["samples"] = 9
    with pytest.raises(ValueError):
        SourceFloatAudioDecodeReceipt(origin, bad_payload, runtime=runtime, limits=limits.record())
    stricter = limits.record()
    stricter["max_samples"] = 4
    with pytest.raises(ValueError):
        SourceFloatAudioDecodeReceipt.from_json(receipt.to_json(), origin=origin, runtime=runtime,
                                                limits=stricter)
    # Unavailable is honest observed coverage, not an automatic receipt failure.
    unavailable = json.loads(canonical(payload))
    unavailable["windows"][0][1] = "1/6000"
    unavailable["limits"]["decoder_end"] = "1/6000"
    unavailable["audio"][0]["coverage"][0] = dict(window=["0/1", "1/6000"], status="unavailable")
    from clipperx.render_clock import float_audio_decode_argv
    unavailable["argv"] = [float_audio_decode_argv(runtime["ffmpeg"], origin, Fraction(0), Fraction(1, 6000),
                                                  "render48k-f32-mono-v1")]
    SourceFloatAudioDecodeReceipt(origin, unavailable, runtime=runtime, limits=limits.record())


def test_float_audio_receipt_caps_precede_parser_hash_or_arithmetic(monkeypatch: pytest.MonkeyPatch) -> None:
    import copy
    import pytest
    import clipperx.render_clock as clock
    origin, runtime, limits, _, payload = _float_recipe()
    receipt = clock.SourceFloatAudioDecodeReceipt(origin, payload, runtime=runtime, limits=limits.record())
    wire = receipt.to_json()
    malformed_json: tuple[Any, ...] = ("[" * 11 + "]" * 11, '{"a":1,"a":2}', '{"a":' + "1" * 1236 + "}",
                                     '{"a":1e999999999}', '{"a":0.1}', '{"a":NaN}', '\ud800', 1)
    for bad in malformed_json:
        with pytest.raises(ValueError):
            clock.SourceFloatAudioDecodeReceipt.from_json(bad, origin=origin, runtime=runtime,
                                                          limits=limits.record())
    malformed = copy.deepcopy(origin)
    object.__setattr__(malformed.streams[0], "time_base", None)
    with pytest.raises(ValueError):
        clock.SourceFloatAudioDecodeReceipt(malformed, payload, runtime=runtime, limits=limits.record())
    bypassed = copy.deepcopy(origin)
    object.__setattr__(bypassed, "duration", "1e999999999")
    with pytest.raises(ValueError):
        clock.SourceFloatAudioDecodeReceipt(bypassed, payload, runtime=runtime, limits=limits.record())
    tight = limits.record()
    tight["max_metadata_bytes"] = 10
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("parser/hash/clone reached before cap")
    monkeypatch.setattr(clock, "digest", forbidden)
    monkeypatch.setattr(clock.json, "loads", forbidden)
    with pytest.raises(ValueError):
        clock.SourceFloatAudioDecodeReceipt.from_json(wire, origin=origin, runtime=runtime, limits=tight)
    with pytest.raises(ValueError):
        clock.SourceFloatAudioDecodeReceipt(origin, payload, runtime=runtime, limits=tight)
    with pytest.raises(ValueError):
        receipt.validate_origin(bypassed, runtime=runtime)


def test_float_audio_shared_identity_and_stream_json_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    import pytest
    from dataclasses import replace
    from clipperx import render_clock as c, source_decode as d
    from clipperx.render_runtime import NATIVE_RUNTIME, FFMPEG_SHA256, FFPROBE_SHA256
    runtime = dict(runtime_id=NATIVE_RUNTIME, ffmpeg="O:/HOST/ffmpeg.exe", ffprobe="O:/HOST/ffprobe.exe",
                   ffmpeg_sha256=FFMPEG_SHA256, ffprobe_sha256=FFPROBE_SHA256)
    assert c.source_decoder_identity(runtime) == "a6374863a6199d4bc26d98c33eb8c6b3a8809104340d9664c12424e81c9f2642"
    assert c.DECODE_POLICY == d.DECODE_POLICY == "bgr24-showinfo;pcm-s16-ashowinfo;copyts;no-healing;v1"
    origin, limits, payload, _ = stream_recipe()
    session = "12345678-1234-4234-8234-123456789abc"
    repair_stream_metadata(origin, session, payload)
    receipt = c.SourceStreamFinalization(origin, session, payload, limits=limits)
    loaded = c.SourceStreamFinalization.from_json(receipt.to_json(), origin=origin,
        policy=c.STREAM_DECODE_POLICY, limits=limits, expected_session_id=session)
    assert loaded.to_json() == receipt.to_json()
    with pytest.raises(ValueError):
        c.SourceStreamFinalization.from_json(receipt.to_json(), origin=origin, policy=c.STREAM_DECODE_POLICY,
            limits=replace(limits, max_finalization_bytes=32), expected_session_id=session)
    with pytest.raises(ValueError):
        c.audio_format("render48k-f32-mono-v1")
    from pathlib import Path
    from clipperx.render_runtime import RenderRuntime
    host = RenderRuntime(Path(runtime["ffmpeg"]), Path(runtime["ffprobe"]), NATIVE_RUNTIME,
        FFMPEG_SHA256, FFPROBE_SHA256, None, None, "-filter_complex_script", None)
    monkeypatch.setattr(d, "source_decoder_identity", lambda record: "shared-digest")
    assert d._runtime_identity(host) == "shared-digest"

