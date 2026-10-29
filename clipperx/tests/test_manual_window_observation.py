"""Finite fake-only tests; no native observation or acceptance is asserted."""

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from fractions import Fraction as F
import hashlib
import importlib
from io import BytesIO
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
from typing import Any, Iterator
import zlib

import pytest
from PIL import Image


@pytest.fixture(autouse=True)
def no_native(monkeypatch: pytest.MonkeyPatch) -> None:
    import subprocess
    import threading

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("native child/worker forbidden")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(threading.Thread, "start", forbidden)


def module() -> Any:
    return importlib.import_module("scripts.observe_manual_windows")


def request_data() -> dict[str, Any]:
    windows = [
        [(".5", "1.6", [".5", "1", "1.5"])],
        [(".5", ".6", [".5"]), ("3.5", "3.6", ["3.5"]),
         ("6.45", "6.55", ["6.45"])],
        [("28", "28.1", ["28"]), ("30.7", "31.1", ["30.7", "30.75", "30.8", "31"]),
         ("31.95", "32.05", ["31.95"])]]
    return dict(version=1, run_id="fake_observe73", cases=[
        dict(id=name, source=f"O:/fake/{name}.mp4", source_fingerprint="a"*16,
             batches=[dict(window=[a, b], requests=r) for a, b, r in rows])
        for name, rows in zip(("gameplay", "camera", "transition"), windows)])


def test_default_and_missing_activation_have_no_actions(monkeypatch: pytest.MonkeyPatch,
                                                       capsys: Any) -> None:
    m = module()
    monkeypatch.setattr(m, "load_manifest", lambda *a: pytest.fail("authority touched"))
    for args in ([], ["--confirm-host"], ["--manifest", "absent"]):
        assert m.main(args) == 0
    assert "NOT RUN" in capsys.readouterr().out
    for args in (["--native"], ["--native", "--confirm-host"]):
        with pytest.raises(ValueError):
            m.main(args)


def test_exact_request_schema_and_bounds() -> None:
    m = module()
    data = request_data()
    assert m.parse_requests(data, data["run_id"]) == data
    for mutate in (
        lambda d: d.update(version=True), lambda d: d.update(extra=1),
        lambda d: d.update(run_id="../bad"), lambda d: d["cases"].pop(),
        lambda d: d["cases"][1].update(id="gameplay"),
        lambda d: d["cases"][0].update(source="relative.mp4"),
        lambda d: d["cases"][0].update(source_fingerprint="A"*16),
        lambda d: d["cases"][0]["batches"][0].update(window=["0", "2"]),
        lambda d: d["cases"][0]["batches"][0].update(requests=[0.5, "1", "1.5"]),
        lambda d: d["cases"][0]["batches"][0].update(requests=["nan", "1", "1.5"]),
        lambda d: d["cases"][0]["batches"][0].update(requests=["1"*129, "1", "1.5"]),
        lambda d: d["cases"][0]["batches"][0].update(requests=["1", ".5", "1.5"]),
        lambda d: d["cases"][0]["batches"][0].update(requests=[".5", "1", "1.6"]),
        lambda d: d["cases"][0]["batches"].append(d["cases"][0]["batches"][0]),
    ):
        value = request_data()
        mutate(value)
        with pytest.raises(ValueError):
            m.parse_requests(value, "fake_observe73")
    with pytest.raises(ValueError):
        m.parse_requests(data, "different")


def authority(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, dict[str, Any], dict[str, bytes]]:
    m = module()
    raw = json.dumps(request_data()).encode()
    payloads = {"requests.json": raw}
    manifest: dict[str, Any] = dict(pins={"requests.json": hashlib.sha256(raw).hexdigest()},
                                   gate_acceptance={}, runs={})
    for stage in ("prerequisites", "synthetic"):
        leaf = stage+".json"
        payloads[leaf] = json.dumps(dict(stage=stage, run_id=stage+"15",
            status="MEASURED_VISUAL_ACCEPTANCE_PENDING")).encode()
        manifest["runs"][stage] = stage+"15"
        manifest["gate_acceptance"][stage] = dict(path=leaf,
            sha256=hashlib.sha256(payloads[leaf]).hexdigest(), accepted_by_root=True)
    monkeypatch.setattr(m, "bounded_read", lambda p, cap: payloads[p.name])
    return m, manifest, payloads


@pytest.mark.parametrize("fault", ["pin", "requesthash", "run", "gate", "truthy", "gatehash",
                                    "gatestage", "gatestatus", "gaterun", "traversal"])
def test_authority_rejection_before_session(monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    m, manifest, payloads = authority(monkeypatch)
    if fault == "pin":
        manifest["pins"].clear()
    elif fault == "requesthash":
        manifest["pins"]["requests.json"] = "0"*64
    elif fault == "gate":
        manifest["gate_acceptance"].pop("prerequisites")
    elif fault == "truthy":
        manifest["gate_acceptance"]["prerequisites"]["accepted_by_root"] = 1
    elif fault == "gatehash":
        manifest["gate_acceptance"]["synthetic"]["sha256"] = "0"*64
    elif fault.startswith("gate"):
        prior = json.loads(payloads["synthetic.json"])
        prior[{"gatestage": "stage", "gatestatus": "status", "gaterun": "run_id"}[fault]] = "bad"
        payloads["synthetic.json"] = json.dumps(prior).encode()
        manifest["gate_acceptance"]["synthetic"]["sha256"] = hashlib.sha256(
            payloads["synthetic.json"]).hexdigest()
    with pytest.raises(ValueError):
        m.check_authority(Path("O:/fake"), manifest,
            "../requests.json" if fault == "traversal" else "requests.json",
            "different" if fault == "run" else "fake_observe73")


def test_authority_success(monkeypatch: pytest.MonkeyPatch) -> None:
    m, manifest, payloads = authority(monkeypatch)
    data, digest = m.check_authority(Path("O:/fake"), manifest, "requests.json", "fake_observe73")
    assert data == request_data()
    assert digest == hashlib.sha256(payloads["requests.json"]).hexdigest()


@dataclass
class Runtime:
    ffmpeg: str = "fake-ffmpeg"
    device_index: int | None = 0


def selection() -> Any:
    from clipperx.source_decode import ObservedFrame, ProvisionalSelection, VideoTag
    raw = bytes((3, 2, 1, 6, 5, 4))
    tag = VideoTag(0, 10, F(1, 10), 2, 1, zlib.adler32(raw, 0))
    frame = ObservedFrame(tag, F(11, 10), raw)
    row = dict(tag.record(), bytes=6, sha256=frame.sha256, coverage_end="11/10")
    successor = dict(row, n=1, pts=11, coverage_end=None)
    return ProvisionalSelection("aaaaaaaa-aaaa-4aaa-aaaa-aaaaaaaaaaaa", 0, (0, 1), frame, row, successor,
                                previous_chain_sha256="0"*64)


class FakeSession:
    def __init__(self) -> None:
        self.runtime = Runtime()
        self.files = object()
        self.raws: dict[Path, bytes] = {}
        self.encodes: list[Any] = []
        self.acks: list[int] = []
        self.finished = False
        self.retired = True
        self.fail_write = False
        self.channels = 2
        self.width = 2
        self.control = NS(ownership_snapshot=lambda: NS(phase="retired_ok" if self.retired else "active",
                            window_resources_zero=self.retired, cleanup_required=not self.retired))
        self.job = NS(new_observation=lambda *a, **k: self.control)

    def register(self, *args: Any, **kwargs: Any) -> None:
        assert kwargs == dict(owned=False)
        assert args[1:] == ("original", 2**48)

    def io_scope(self) -> Any:
        return nullcontext()

    def source_probe(self, source: Path) -> Any:
        return NS(streams=(NS(kind="video", width=self.width, height=1),
                           NS(kind="audio", channels=self.channels)), record=lambda: {"fake": True})

    def deadline(self, seconds: int) -> int:
        assert seconds == 45
        return 45

    def put(self, leaf: str, raw: bytes, **kwargs: Any) -> Path:
        assert kwargs == dict(role="raw", retain=False, cap=6)
        if self.fail_write:
            raise ValueError("fake write failed")
        path = Path(leaf)
        self.raws[path] = bytes(raw)
        return path

    def file_output(self, leaf: str, argv: Any, inputs: Any, **kwargs: Any) -> Any:
        assert self.finished and self.retired
        assert kwargs == dict(cap=12*1024**2, timeout=45, retain=True)
        self.encodes.append((argv, inputs))
        raw = self.raws[inputs[0]]
        with Image.frombytes("RGB", (2, 1), raw, "raw", "BGR") as image:
            out = BytesIO()
            image.save(out, format="PNG")
        self.raws[Path(leaf)] = out.getvalue()
        return Path(leaf), None

    def read(self, path: Path, cap: int) -> bytes:
        assert cap == 12*1024**2
        return self.raws[path]


@contextmanager
def fake_stream(session: FakeSession, *args: Any, **kwargs: Any) -> Iterator[Any]:
    assert args[0].device_index is None
    assert kwargs["_host"] is session.control and kwargs["_runtime_files"] is session.files
    assert kwargs["audio_policy"] == "render48k-stereo"
    assert kwargs["limits"].max_window == 2 and kwargs["limits"].max_requests == 5
    emitted = False
    def next_selection() -> Any:
        nonlocal emitted
        if emitted:
            return None
        emitted = True
        return selection()
    def acknowledge(index: int) -> None:
        # Inspect adapter caller: no borrowed selection/frame/bytes/view aliases may survive.
        locals_ = sys._getframe(1).f_locals
        from clipperx.source_decode import ObservedFrame, ProvisionalSelection
        assert not any(isinstance(v, (ObservedFrame, ProvisionalSelection, bytes, memoryview))
                       for v in locals_.values())
        session.acks.append(index)
    def finish() -> Any:
        assert session.acks == [0]
        session.finished = True
        return NS(record=lambda: dict(fake=True, completion={"children": 2}, audio={"bytes": 2}))
    yield NS(next_selection=next_selection, acknowledge=acknowledge, finish=finish)


def install_action(monkeypatch: pytest.MonkeyPatch, s: FakeSession) -> Any:
    from clipperx import source_decode
    m = module()
    monkeypatch.setattr(m, "source_fingerprint", lambda *a, **k: "a"*16)
    monkeypatch.setattr(source_decode, "stream_observed_window", lambda *a, **k: fake_stream(s, *a, **k))
    return m


def one_case() -> dict[str, Any]:
    return dict(id="gameplay", source="O:/fake/game.mp4", source_fingerprint="a"*16,
                batches=[dict(window=["1", "1.1"], requests=["1", "1.05"])])


def test_grouped_delivery_exact_pixels_alias_release_and_finish(monkeypatch: pytest.MonkeyPatch) -> None:
    s = FakeSession()
    m = install_action(monkeypatch, s)
    result = m.observe(s, [one_case()])
    assert len(s.encodes) == 1 and s.acks == [0] and s.finished
    row = result[0]["batches"][0]["images"][0]
    assert row["delivery"]["request_ordinals"] == [0, 1]
    assert row["selected"]["pts"] == 10 and row["selected"]["time_base"] == "1/10"
    assert row["selected_bgr_sha256"] == hashlib.sha256(bytes((3, 2, 1, 6, 5, 4))).hexdigest()
    assert result[0]["batches"][0]["finalization"]["completion"]["children"] == 2
    argv = s.encodes[0][0]
    assert "bgr24" in argv and "rgb24" in argv and "2:1" not in argv


@pytest.mark.parametrize("fault", ["write", "retire", "fingerprint", "channels", "pixels"])
def test_action_failures_stop_before_encode(monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    s = FakeSession()
    m = install_action(monkeypatch, s)
    if fault == "write":
        s.fail_write = True
    elif fault == "retire":
        s.retired = False
    elif fault == "fingerprint":
        monkeypatch.setattr(m, "source_fingerprint", lambda *a, **k: "b"*16)
        s.source_probe = lambda *a: pytest.fail("probe after fingerprint mismatch")  # type: ignore[method-assign]
    elif fault == "channels":
        s.channels = 6
    else:
        s.width = 1920*1080+1
    message = {"write": "fake write failed", "retire": "not certainly retired",
               "fingerprint": "fingerprint changed", "channels": "mono/stereo", "pixels": "bounded video"}[fault]
    with pytest.raises(ValueError, match=message):
        m.observe(s, [one_case()])
    assert not s.encodes
    if fault == "write":
        assert s.acks == [] and not s.finished


def test_png_exact_bgr_and_reject_wrong_pixels() -> None:
    m = module()
    raw = bytes((3, 2, 1, 6, 5, 4))
    with Image.frombytes("RGB", (2, 1), bytes((1, 2, 3, 4, 5, 6))) as image:
        out = BytesIO()
        image.save(out, format="PNG")
    m.verify_png(out.getvalue(), 2, 1, hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError):
        m.verify_png(out.getvalue(), 2, 1, "0"*64)


@pytest.mark.parametrize("fault", ["none", "size", "pixels", "convert"])
def test_png_size_before_conversion_and_deterministic_core_close(
        monkeypatch: pytest.MonkeyPatch, fault: str) -> None:
    m = module()
    fixture = Image.frombytes("RGB", (2, 1), bytes((1, 2, 3, 4, 5, 6)))
    out = BytesIO()
    try:
        fixture.save(out, format="PNG")
    finally:
        fixture.close()
    opened: list[Any] = []
    converted: list[Any] = []
    closed: list[Any] = []
    original_open, original_convert, original_close = Image.open, Image.Image.convert, Image.Image.close
    def open_(*args: Any, **kwargs: Any) -> Any:
        image = original_open(*args, **kwargs)
        assert image._im is None  # Real Pillow PNG open is lazy.
        opened.append(image)
        return image
    def convert(image: Any, *args: Any, **kwargs: Any) -> Any:
        if fault == "size":
            pytest.fail("conversion before exact size admission")
        if fault == "convert":
            image.load()
            raise ValueError("finite conversion failure")
        result = original_convert(image, *args, **kwargs)
        converted.append(result)
        return result
    def close(image: Any) -> None:
        original_close(image)
        closed.append(image)
        # Verify the actual image core was invalidated, not merely file-handle exit.
        with pytest.raises(ValueError, match="closed image"):
            image.getpixel((0, 0))
    monkeypatch.setattr(Image, "open", open_)
    monkeypatch.setattr(Image.Image, "convert", convert)
    monkeypatch.setattr(Image.Image, "close", close)
    expected = hashlib.sha256(bytes((3, 2, 1, 6, 5, 4))).hexdigest()
    if fault != "none":
        with pytest.raises(ValueError):
            m.verify_png(out.getvalue(), 1 if fault == "size" else 2, 1,
                         "0"*64 if fault == "pixels" else expected)
    else:
        m.verify_png(out.getvalue(), 2, 1, expected)
    assert len(opened) == 1
    assert closed == ([converted[0], opened[0]] if converted else opened)
    assert len(closed) == (2 if fault in ("none", "pixels") else 1)


def test_campaign_pre_admission_and_postreturn() -> None:
    m = module()
    c = m.ObservationCampaign()
    c.charge(32)
    with pytest.raises(ValueError):
        c.charge()
    assert c.children == 32
    c.start -= 2701*10**9
    with pytest.raises(ValueError):
        c.charge(0)


def test_aggregate_budget_rejects_before_write() -> None:
    m = module()
    b = m.EvidenceBudget()
    b.add_raw(128*1024**2)
    with pytest.raises(ValueError):
        b.add_raw(1)
    b.add_image(128*1024**2)
    with pytest.raises(ValueError):
        b.add_image(1)


@pytest.mark.parametrize("existing", ["base", "guard"])
def test_existing_base_or_guard_refused_before_mkdir(monkeypatch: pytest.MonkeyPatch,
                                                    existing: str) -> None:
    m, manifest, _ = authority(monkeypatch)
    monkeypatch.setattr(m, "load_manifest", lambda *a: (manifest, "f"*64))
    monkeypatch.setattr(Path, "is_dir", lambda p: True)
    monkeypatch.setattr(Path, "exists", lambda p: (p.name == ".guard") == (existing == "guard"))
    monkeypatch.setattr(Path, "mkdir", lambda *a, **k: pytest.fail("replayed guard created"))
    with pytest.raises(ValueError, match="consumed/replayed"):
        m.main(["--native", "--confirm-host", "--manifest", "fake-manifest.json",
                "--requests", "requests.json", "--run-id", "fake_observe73"])


@pytest.mark.parametrize("fail", [False, True])
def test_controller_writes_one_summary_or_error_exclusively(monkeypatch: pytest.MonkeyPatch,
                                                         fail: bool) -> None:
    m, manifest, _ = authority(monkeypatch)
    writes: dict[str, Any] = {}
    mkdirs = []
    monkeypatch.setattr(m, "load_manifest", lambda *a: (manifest, "f"*64))
    monkeypatch.setattr(Path, "is_dir", lambda p: True)
    monkeypatch.setattr(Path, "exists", lambda p: False)
    monkeypatch.setattr(Path, "mkdir", lambda p, **k: mkdirs.append((p, k)))
    class Capture(BytesIO):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name
        def __exit__(self, *a: Any) -> None:
            writes[self.name] = json.loads(self.getvalue())
            self.close()
    def open_(path: Path, mode: str) -> Any:
        assert mode == "xb" and path.name not in writes
        return Capture(path.name)
    monkeypatch.setattr(Path, "open", open_)
    def run(base: Path, manifest: Any, data: Any, campaign: Any) -> Any:
        assert list(writes) == [".guard"]
        campaign.records.append({"finite_fake_owner": "not actual proof"})
        if fail:
            raise ValueError("finite fake failure")
        return {"resources": campaign.records}
    monkeypatch.setattr(m, "run_observations", run)
    args = ["--native", "--confirm-host", "--manifest", "fake.json",
            "--requests", "requests.json", "--run-id", "fake_observe73"]
    if fail:
        with pytest.raises(ValueError, match="finite fake failure"):
            m.main(args)
    else:
        assert m.main(args) == 0
    assert len(mkdirs) == 1 and mkdirs[0][1] == dict(exist_ok=False)
    assert set(writes) == {".guard", "controller-error.json" if fail else "controller-summary.json"}
    for row in writes.values():
        assert row["run_id"] == "fake_observe73" and row["manifest_sha256"] == "f"*64
        assert len(row["requests_sha256"]) == 64


@pytest.mark.parametrize("oversized", ["summary", "error"])
def test_actual_serializer_rejection_never_opens_unbounded_terminal_destination(
        monkeypatch: pytest.MonkeyPatch, oversized: str) -> None:
    m, manifest, _ = authority(monkeypatch)
    opened: list[str] = []
    writes: dict[str, bytes] = {}
    monkeypatch.setattr(m, "load_manifest", lambda *a: (manifest, "f"*64))
    monkeypatch.setattr(Path, "is_dir", lambda p: True)
    monkeypatch.setattr(Path, "exists", lambda p: False)
    monkeypatch.setattr(Path, "mkdir", lambda p, **k: None)
    class Capture(BytesIO):
        def __init__(self, name: str) -> None:
            super().__init__()
            self.name = name
        def __exit__(self, *a: Any) -> None:
            writes[self.name] = self.getvalue()
            self.close()
    def open_(path: Path, mode: str) -> Any:
        assert mode == "xb" and path.name not in opened
        opened.append(path.name)
        return Capture(path.name)
    monkeypatch.setattr(Path, "open", open_)
    def run(base: Path, manifest: Any, data: Any, campaign: Any) -> Any:
        payload = "x"*(8*1024**2+1)
        if oversized == "error":
            campaign.records.append({"finite_fake_oversized_failure_evidence": payload})
            raise ValueError("finite fake action failure")
        return {"finite_fake_oversized_success_evidence": payload}
    monkeypatch.setattr(m, "run_observations", run)
    # Real reused json_bytes rejects the actual encoded >8MiB value; no serializer stub.
    with pytest.raises(ValueError, match="oracle/controller JSON ceiling"):
        m.main(["--native", "--confirm-host", "--manifest", "fake.json",
                "--requests", "requests.json", "--run-id", "fake_observe73"])
    assert "controller-summary.json" not in opened
    if oversized == "summary":
        assert opened == [".guard", "controller-error.json"]
        error = json.loads(writes["controller-error.json"])
        assert error["first_error"]["message"] == "oracle/controller JSON ceiling"
        assert 0 < len(writes["controller-error.json"]) <= 8*1024**2
    else:
        assert opened == [".guard"]  # Rejected error metadata cannot create an empty artifact.
    assert set(writes) == set(opened) and all(writes.values())


def test_all_authority_checked_before_guard_and_session(monkeypatch: pytest.MonkeyPatch) -> None:
    m, manifest, _ = authority(monkeypatch)
    manifest["pins"].clear()
    monkeypatch.setattr(m, "load_manifest", lambda *a: (manifest, "f"*64))
    monkeypatch.setattr(Path, "is_dir", lambda p: pytest.fail("filesystem setup before authority"))
    monkeypatch.setattr(m, "FixtureSession", lambda *a: pytest.fail("session before authority"))
    with pytest.raises(ValueError, match="manifest pin"):
        m.main(["--native", "--confirm-host", "--manifest", "fake.json",
                "--requests", "requests.json", "--run-id", "fake_observe73"])


def test_completed_session_resources_and_postreturn_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    m = module()
    seen = []
    class Session:
        def __init__(self, root: Path, manifest: Any, campaign: Any) -> None:
            seen.append(root)
            self.campaign = campaign
        def execute(self, action: Any) -> Any:
            self.campaign.records.append({"finite_fake_completed_snapshot": True})
            self.campaign.children = 33
            return []
    monkeypatch.setattr(m, "FixtureSession", Session)
    c = m.ObservationCampaign()
    with pytest.raises(ValueError, match="ceiling32"):
        m.run_observations(Path("O:/fake"), {}, request_data(), c)
    assert seen == [Path("O:/fake/observations")]
    assert c.records == [{"finite_fake_completed_snapshot": True}]


def test_session_completed_records_are_forwarded_without_owner_invention(monkeypatch: pytest.MonkeyPatch) -> None:
    m = module()
    evidence = {"finite_fake_completed_snapshot": "not native proof"}
    def session(root: Path, manifest: Any, campaign: Any) -> Any:
        def execute(action: Any) -> Any:
            campaign.records.append(evidence)
            return ["finite fake action result"]
        return NS(execute=execute)
    monkeypatch.setattr(m, "FixtureSession", session)
    c = m.ObservationCampaign()
    result = m.run_observations(Path("O:/fake"), {}, request_data(), c)
    assert result["resources"] is c.records and result["resources"][0] is evidence
    assert result["results"] == ["finite fake action result"]
    assert "own0" not in result and "files_clear" not in result


def test_scratch_rejection_precedes_selected_raw_write(monkeypatch: pytest.MonkeyPatch) -> None:
    s = FakeSession()
    m = install_action(monkeypatch, s)
    budget = m.EvidenceBudget(raw_bytes=128*1024**2)
    monkeypatch.setattr(m, "EvidenceBudget", lambda: budget)
    with pytest.raises(ValueError, match="RAW scratch ceiling"):
        m.observe(s, [one_case()])
    assert not s.raws and not s.acks and not s.encodes


def test_bad_png_pixels_stop_action(monkeypatch: pytest.MonkeyPatch) -> None:
    s = FakeSession()
    m = install_action(monkeypatch, s)
    read = s.read
    def corrupt(path: Path, cap: int) -> bytes:
        with Image.new("RGB", (2, 1), "black") as image:
            out = BytesIO()
            image.save(out, format="PNG")
        assert read(path, cap)
        return out.getvalue()
    s.read = corrupt  # type: ignore[method-assign]
    with pytest.raises(ValueError, match="BGR pixel identity"):
        m.observe(s, [one_case()])
    assert s.acks == [0] and s.finished and len(s.encodes) == 1
    b = m.EvidenceBudget()
    for _ in range(12):
        b.add_image(1)
    with pytest.raises(ValueError):
        b.add_image(1)
