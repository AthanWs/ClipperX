from __future__ import annotations

from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
from collections.abc import Sequence
from typing import Any, TypeAlias, TypeVar, cast

from pydantic import BaseModel
from PIL import Image
import pytest

from clipperx.local_paths import source_fingerprint
from clipperx.style_analysis import StyleEvidence
from clipperx.structured_client import StructuredResult
from clipperx.style_contracts import BackgroundTreatment, SceneFamily, StyleCardAnalysisResponse, StyleKeyframe
from clipperx.style_library import StyleCardBuildResult, StyleLibrary, build_style_card
from clipperx.visual_contracts import NormalizedRect


Inputs: TypeAlias = tuple[Path, Path, StyleEvidence]
ModelT = TypeVar("ModelT", bound=BaseModel)


class FakeClient:
    def __init__(self, evidence_refs: list[str] | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.evidence_refs = evidence_refs if evidence_refs is not None else ["frame_000"]

    def generate(
        self, output_model: type[ModelT], *, system_prompt: str, user_prompt: str,
        images: list[Path] | None = None, temperature: float = 0.2,
    ) -> StructuredResult[ModelT]:
        assert output_model is StyleCardAnalysisResponse
        self.calls.append({"system_prompt": system_prompt, "user_prompt": user_prompt, "images": images})
        return StructuredResult(
            value=output_model.model_validate(StyleCardAnalysisResponse(
                scene_family=SceneFamily.CONTENT_ONLY,
                camera_regions=[],
                content_regions=[],
                background_treatment=BackgroundTreatment.COVER,
                title_regions=[],
                brand_regions=[],
                subtitle_regions=[],
                subtitle_palette=[],
                special_techniques=["hard cuts"],
                description="Composición vertical visible.",
                evidence_refs=self.evidence_refs,
            ).model_dump()),
            usage={"total_tokens": 42, "estimated_input_tokens": 100},
            raw_response={"secret": "must never persist"},
        )


@pytest.fixture
def inputs(tmp_path: Path) -> Inputs:
    root = tmp_path / "library"
    source = tmp_path / "historical_edit.mp4"
    source.write_bytes(b"synthetic source identity, never decoded")
    keyframes = []
    hashes = {}
    for i in range(5):
        ref = f"evidence/frames/{i}.jpg"
        path = root / ref
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (36, 64), (i * 40, 20, 30)).save(path)
        keyframes.append(StyleKeyframe(timestamp=float(i), path=ref))
        hashes[ref] = hashlib.sha256(path.read_bytes()).hexdigest()
    evidence = StyleEvidence(
        analyzer_version="objective-style-1.2",
        clip_fingerprint=source_fingerprint(source),
        duration_seconds=5,
        media_properties={"width": 1080, "height": 1920},
        sampled_timestamps=[0, 1, 2, 3, 4],
        scene_change_timestamps=[2],
        keyframes=keyframes,
        keyframe_sha256=hashes,
        evidence_path="evidence/evidence.json",
        background_treatment=BackgroundTreatment.COVER,
        background_regions=[],
        panel_candidates=[],
        content_regions=[],
        title_regions=[],
        brand_regions=[],
        subtitle_regions=[],
        subtitle_palette=[],
        overlay_candidates=[],
        motion_candidates=[],
        cut_rate_per_minute=12,
        zoom_count=2,
        pan_count=0,
        visual_quality=0.8,
        quality_metrics={"sharpness": 123.0},
        limitations=["candidates are not semantics"],
    )
    return root, source, evidence


def build(inputs: Inputs, client: FakeClient, **kwargs: Any) -> StyleCardBuildResult:
    root, source, evidence = inputs
    return build_style_card(
        evidence,
        library_root=root,
        source_clip=source,
        client=client,
        model_id=kwargs.pop("model_id", "vision-test"),
        **kwargs,
    )


@pytest.mark.parametrize("count", [3, 4, 5, 8])
def test_card_builder_sends_three_to_five_saved_keyframes(inputs: Inputs, count: int) -> None:
    root, source, evidence = inputs
    if count > 5:
        # Existing eight-frame evidence: reuse neither path nor content.
        for i in range(5, count):
            path = root / f"evidence/frames/{i}.jpg"
            Image.new("RGB", (36, 64), (i * 30, 50, 80)).save(path)
            evidence.keyframes.append(
                type(evidence.keyframes[0])(
                    timestamp=3 + i / 4, path=path.relative_to(root).as_posix()
                )
            )
            evidence.keyframe_sha256[path.relative_to(root).as_posix()] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
        evidence.keyframes.sort(key=lambda f: f.timestamp)
    else:
        evidence.keyframes = evidence.keyframes[:count]
    client = FakeClient()
    result = build(inputs, client)
    card = result.card
    assert len(client.calls[0]["images"]) == min(count, 5)
    assert card.clip_fingerprint == evidence.clip_fingerprint
    assert card.duration == evidence.duration_seconds
    assert (card.cut_rate_per_minute, card.zoom_count, card.visual_quality) == (12, 2, 0.8)
    assert card.created_at is None and card.narrative_quality is None
    assert card.source_clip_ref == f"source_refs/{evidence.clip_fingerprint}.json"
    sidecar = json.loads((root / card.source_clip_ref).read_text())
    assert sidecar["source_clip_path"] == str(source.resolve())
    assert sidecar["provenance"] == "historical_edited_clip"
    assert source.read_bytes() == b"synthetic source identity, never decoded"
    assert result.usage["total_tokens"] == 42


def test_card_builder_rejects_absolute_keyframe_refs(inputs: Inputs) -> None:
    evidence = inputs[2]
    evidence.keyframes[0] = evidence.keyframes[0].model_copy(
        update={"path": "C:/outside/frame.jpg"}
    )
    client = FakeClient()
    with pytest.raises(ValueError, match="relative|portable"):
        build(inputs, client)
    assert not client.calls


@pytest.mark.parametrize("damage", ["missing", "corrupt", "hash", "escape", "symlink"])
def test_invalid_frames_rejected_before_upload(inputs: Inputs, tmp_path: Path, damage: str) -> None:
    root, _, evidence = inputs
    path = root / evidence.keyframes[0].path
    if damage == "missing":
        path.unlink()
    elif damage == "corrupt":
        path.write_bytes(b"not an image")
        evidence.keyframe_sha256[evidence.keyframes[0].path] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    elif damage == "hash":
        path.write_bytes(b"changed")
    elif damage == "escape":
        evidence.keyframes[0] = evidence.keyframes[0].model_copy(update={"path": "../outside.jpg"})
    else:
        outside = tmp_path / "outside.jpg"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        try:
            path.symlink_to(outside)
        except OSError:
            pytest.skip("symlink permission unavailable")
    client = FakeClient()
    with pytest.raises((ValueError, OSError)):
        build(inputs, client)
    assert not client.calls


@pytest.mark.parametrize("kind", ["too_few", "same_bytes", "same_path"])
def test_never_pads_duplicate_views(inputs: Inputs, kind: str) -> None:
    root, _, evidence = inputs
    evidence.keyframes = evidence.keyframes[:3]
    if kind == "too_few":
        evidence.keyframes = evidence.keyframes[:2]
    elif kind == "same_path":
        evidence.keyframes[1].path = evidence.keyframes[0].path
    else:
        target = root / evidence.keyframes[1].path
        target.write_bytes((root / evidence.keyframes[0].path).read_bytes())
        evidence.keyframe_sha256[evidence.keyframes[1].path] = hashlib.sha256(
            target.read_bytes()
        ).hexdigest()
    with pytest.raises(ValueError, match="distinct"):
        build(inputs, FakeClient())


def test_idempotence_durable_loading_and_host_label_refresh(inputs: Inputs) -> None:
    client = FakeClient()
    first = build(inputs, client, card_id="card-test")
    root, _, evidence = inputs
    library = StyleLibrary(root)
    fingerprint = library.fingerprint
    before = (root / "library_manifest.json").read_bytes()
    again = build(inputs, client, card_id="card-test")
    assert again.cache_hit and len(client.calls) == 1
    assert before == (root / "library_manifest.json").read_bytes()
    assert first.card == library.load_card("card-test")
    date = datetime(2026, 1, 1, tzinfo=timezone.utc)
    evidence.narrative_quality = 0.1  # not an authorized human label argument
    refreshed = build(
        inputs,
        client,
        card_id="card-test",
        approval="approved",
        narrative_quality=0.9,
        created_at=date,
    )
    assert len(client.calls) == 1 and refreshed.cache_hit
    assert refreshed.card.approval == "approved"
    assert refreshed.card.narrative_quality == 0.9 and refreshed.card.created_at == date
    assert [c.card_id for c in StyleLibrary(root).list_cards()] == ["card-test"]
    assert library.fingerprint != fingerprint
    assert "must never persist" not in "".join(p.read_text() for p in root.rglob("*.json"))


@pytest.mark.parametrize("change", ["model", "analyzer", "metadata", "metrics", "frame"])
def test_visual_cache_invalidates_objective_changes(inputs: Inputs, change: str) -> None:
    client = FakeClient()
    build(inputs, client)
    root, _, evidence = inputs
    kwargs = {}
    if change == "model":
        kwargs["model_id"] = "other-model"
    elif change == "analyzer":
        evidence.analyzer_version = "new-analyzer"
    elif change == "metadata":
        evidence.media_properties["width"] = 720
    elif change == "metrics":
        evidence.zoom_count = 3
    else:
        path = root / evidence.keyframes[0].path
        Image.new("RGB", (36, 64), (10, 90, 120)).save(path)
        evidence.keyframe_sha256[evidence.keyframes[0].path] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    build(inputs, client, **kwargs)
    assert len(client.calls) == 2


def test_cached_card_cannot_reuse_changed_source(inputs: Inputs) -> None:
    client = FakeClient()
    build(inputs, client)
    inputs[1].write_bytes(b"source has changed")
    with pytest.raises(ValueError, match="fingerprint"):
        build(inputs, client)
    assert len(client.calls) == 1


@pytest.mark.parametrize("refs", [["frame_999"], ["frame_000", "frame_000"]])
def test_unknown_or_duplicate_model_evidence_refs_rejected(inputs: Inputs, refs: list[str]) -> None:
    with pytest.raises(ValueError, match="evidence_ref"):
        build(inputs, FakeClient(refs))
    assert not (inputs[0] / "library_manifest.json").exists()


def test_corrupt_published_card_is_not_consumed(inputs: Inputs) -> None:
    build(inputs, FakeClient(), card_id="card-test")
    root = inputs[0]
    manifest = json.loads((root / "library_manifest.json").read_text())
    (root / manifest["cards"]["card-test"]["path"]).write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        StyleLibrary(root).list_cards()


def test_corrupt_visual_cache_is_not_consumed(inputs: Inputs) -> None:
    client = FakeClient()
    build(inputs, client)
    root = inputs[0]
    manifest = json.loads((root / "library_manifest.json").read_text())
    entry = next(iter(manifest["analyses"].values()))
    (root / entry["path"]).write_text("{}")
    with pytest.raises(ValueError, match="integrity"):
        build(inputs, client)
    assert len(client.calls) == 1


def test_failed_manifest_replace_preserves_previous_generation(inputs: Inputs, monkeypatch: pytest.MonkeyPatch) -> None:
    import clipperx.style_library as module

    build(inputs, FakeClient(), card_id="card-test")
    library = StyleLibrary(inputs[0])
    previous = library.load_card("card-test")
    original = module.atomic_write_text

    def fail_manifest(path: Path, text: str) -> Path:
        if path.name == "library_manifest.json":
            raise OSError("simulated interrupted publication")
        return original(path, text)

    monkeypatch.setattr(module, "atomic_write_text", fail_manifest)
    with pytest.raises(OSError, match="interrupted"):
        build(inputs, FakeClient(), card_id="card-test", approval="approved")
    assert library.load_card("card-test") == previous


def test_source_hashing_does_not_buffer_entire_original(inputs: Inputs, monkeypatch: pytest.MonkeyPatch) -> None:
    original = Path.read_bytes
    source = inputs[1]

    def guarded(path: Path) -> bytes:
        if path == source:
            raise AssertionError("source must use streaming hashing")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    build(inputs, FakeClient())


def test_stale_source_with_restored_stat_cannot_reuse_cache(inputs: Inputs) -> None:
    client = FakeClient()
    build(inputs, client)
    source = inputs[1]
    stat = source.stat()
    original = source.read_bytes()
    source.write_bytes(b"x" * len(original))
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert source_fingerprint(source) == inputs[2].clip_fingerprint
    with pytest.raises(ValueError, match="stale source"):
        build(inputs, client)
    assert len(client.calls) == 1


def test_mutation_during_client_call_is_not_published(inputs: Inputs) -> None:
    class MutatingClient(FakeClient):
        def generate(
            self, output_model: type[ModelT], *, system_prompt: str, user_prompt: str,
            images: list[Path] | None = None, temperature: float = 0.2,
        ) -> StructuredResult[ModelT]:
            result = super().generate(output_model, system_prompt=system_prompt, user_prompt=user_prompt,
                                      images=images, temperature=temperature)
            assert images is not None
            images[0].write_bytes(b"corrupted during request")
            return result

    with pytest.raises(ValueError, match="integrity"):
        build(inputs, MutatingClient())
    assert not (inputs[0] / "library_manifest.json").exists()


def test_source_changed_during_client_call_is_not_published(inputs: Inputs) -> None:
    class MutatingClient(FakeClient):
        def generate(
            self, output_model: type[ModelT], *, system_prompt: str, user_prompt: str,
            images: list[Path] | None = None, temperature: float = 0.2,
        ) -> StructuredResult[ModelT]:
            result = super().generate(output_model, system_prompt=system_prompt, user_prompt=user_prompt,
                                      images=images, temperature=temperature)
            inputs[1].write_bytes(b"changed during request")
            return result

    with pytest.raises(ValueError, match="changed during"):
        build(inputs, MutatingClient())
    assert not (inputs[0] / "library_manifest.json").exists()


def test_parallel_publication_keeps_all_cards_and_reuses_visual_cache(inputs: Inputs) -> None:
    client = FakeClient()
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda i: build(inputs, client, card_id=f"card-{i}"), range(3)))
    assert len(client.calls) == 1
    assert sum(result.cache_hit for result in results) == 2
    assert len(StyleLibrary(inputs[0]).list_cards()) == 3


def test_model_cannot_smuggle_host_fields(inputs: Inputs) -> None:
    class ForgedClient(FakeClient):
        def generate(
            self, output_model: type[ModelT], *, system_prompt: str, user_prompt: str,
            images: list[Path] | None = None, temperature: float = 0.2,
        ) -> StructuredResult[ModelT]:
            result = super().generate(output_model, system_prompt=system_prompt, user_prompt=user_prompt,
                                      images=images, temperature=temperature)
            value = result.value.model_dump()
            value["approval"] = "approved"
            class ForgedResponse(StyleCardAnalysisResponse):
                approval: str = "approved"

            # Deliberately wrong provider model: the host must reject the extra field.
            forged = cast(ModelT, ForgedResponse.model_validate(value))
            return StructuredResult(forged, result.usage, result.raw_response)

    with pytest.raises(ValueError, match="Unrequested fields"):
        build(inputs, ForgedClient())
    assert not (inputs[0] / "library_manifest.json").exists()


@pytest.mark.parametrize("labels", [{"approval": "invalid"}, {"narrative_quality": 2.0}])
def test_invalid_host_labels_fail_before_client_request(inputs: Inputs, labels: dict[str, str | float]) -> None:
    client = FakeClient()
    with pytest.raises(ValueError):
        build(inputs, client, **labels)
    assert not client.calls


def test_frames_of_previous_generation_survive_new_objective_evidence(inputs: Inputs) -> None:
    first = build(inputs, FakeClient())
    old_paths = [inputs[0] / frame.path for frame in first.card.keyframes]
    before = [path.read_bytes() for path in old_paths]
    path = inputs[0] / inputs[2].keyframes[0].path
    Image.new("RGB", (36, 64), (25, 90, 160)).save(path)
    inputs[2].keyframe_sha256[inputs[2].keyframes[0].path] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    build(inputs, FakeClient())
    assert [path.read_bytes() for path in old_paths] == before


@pytest.mark.parametrize("change", ["system_prompt", "user_payload", "response_schema"])
def test_visual_cache_invalidates_actual_request_and_preserves_artifacts(
    inputs: Inputs, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    import clipperx.style_library as module

    root = inputs[0]
    client = FakeClient()
    first = build(inputs, client, card_id="card-test")
    manifest = json.loads((root / "library_manifest.json").read_text())
    old_records = [*manifest["analyses"].values(), *manifest["cards"].values()]
    old_artifacts = {record["path"]: (root / record["path"]).read_bytes() for record in old_records}
    if change == "system_prompt":
        monkeypatch.setattr(
            module, "STYLE_CARD_SYSTEM_PROMPT", module.STYLE_CARD_SYSTEM_PROMPT + "\nNueva regla."
        )
    elif change == "user_payload":
        original = module.build_style_card_prompt

        def changed_payload(evidence: StyleEvidence, keyframes: Sequence[StyleKeyframe]) -> str:
            payload = json.loads(original(evidence, keyframes))
            payload["candidate_warning"] += "; revised request"
            return json.dumps(payload)

        monkeypatch.setattr(module, "build_style_card_prompt", changed_payload)
    else:
        original_schema = module.StyleCardAnalysisResponse.model_json_schema

        def changed_schema(cls: type[StyleCardAnalysisResponse], **kwargs: Any) -> dict[str, Any]:
            return {**original_schema(**kwargs), "description": "Revised response contract"}

        monkeypatch.setattr(
            module.StyleCardAnalysisResponse, "model_json_schema", classmethod(changed_schema)
        )
    changed = build(inputs, client, card_id="card-test")
    assert len(client.calls) == 2 and not changed.cache_hit
    assert changed.evidence_hash == first.evidence_hash
    assert changed.analysis_key != first.analysis_key
    if change == "system_prompt":
        assert client.calls[1]["system_prompt"].endswith("Nueva regla.")
    elif change == "user_payload":
        assert "revised request" in client.calls[1]["user_prompt"]
    again = build(inputs, client, card_id="card-test")
    assert again.cache_hit and again.analysis_key == changed.analysis_key
    refreshed = build(
        inputs, client, card_id="card-test", approval="approved", narrative_quality=0.8
    )
    assert refreshed.cache_hit and refreshed.analysis_key == changed.analysis_key
    assert len(client.calls) == 2
    assert refreshed.card.approval == "approved" and refreshed.card.narrative_quality == 0.8
    assert all((root / ref).read_bytes() == blob for ref, blob in old_artifacts.items())
    published = json.loads((root / "library_manifest.json").read_text())
    assert first.analysis_key in published["analyses"]
    assert changed.analysis_key in published["analyses"]


def test_legacy_visual_cache_is_preserved_but_not_reused(inputs: Inputs) -> None:
    root = inputs[0]
    client = FakeClient()
    first = build(inputs, client)
    manifest_path = root / "library_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    legacy_key = hashlib.sha256(
        json.dumps(
            {"model_id": "vision-test", "evidence_hash": first.evidence_hash},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    old_record = manifest["analyses"].pop(first.analysis_key)
    old_bytes = (root / old_record["path"]).read_bytes()
    manifest["analyses"][legacy_key] = old_record
    manifest["cards"][first.card.card_id]["analysis_key"] = legacy_key
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    corrected = build(inputs, client)
    assert not corrected.cache_hit and len(client.calls) == 2
    assert corrected.analysis_key != legacy_key
    assert build(inputs, client).cache_hit and len(client.calls) == 2
    published = json.loads(manifest_path.read_text())
    assert published["analyses"][legacy_key] == old_record
    assert (root / old_record["path"]).read_bytes() == old_bytes


@pytest.mark.parametrize("duplicate", [True, False])
def test_headline_brand_duplicate_fails_closed_but_other_overlap_is_valid(
    inputs: Inputs, duplicate: bool
) -> None:
    class RegionClient(FakeClient):
        def generate(
            self, output_model: type[ModelT], *, system_prompt: str, user_prompt: str,
            images: list[Path] | None = None, temperature: float = 0.2,
        ) -> StructuredResult[ModelT]:
            result = super().generate(output_model, system_prompt=system_prompt, user_prompt=user_prompt,
                                      images=images, temperature=temperature)
            title = NormalizedRect(x=0.1, y=0.1, width=0.6, height=0.2)
            brand = (
                title.model_copy()
                if duplicate
                else NormalizedRect(x=0.5, y=0.15, width=0.3, height=0.1)
            )
            assert isinstance(result.value, StyleCardAnalysisResponse)
            result.value.title_regions = [title]
            result.value.brand_regions = [brand]
            result.value.content_regions = [NormalizedRect(x=0, y=0, width=1, height=1)]
            result.value.camera_regions = [title.model_copy()]
            return result

    if duplicate:
        with pytest.raises(ValueError, match="title_regions.*brand_regions"):
            build(inputs, RegionClient())
        assert not (inputs[0] / "library_manifest.json").exists()
    else:
        client = RegionClient()
        built = build(inputs, client)
        assert built.card.title_regions and built.card.brand_regions
        assert build(inputs, client).cache_hit and len(client.calls) == 1
