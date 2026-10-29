from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import numpy as np
import pytest

from clipperx.style_contracts import SceneFamily, StyleCard
from clipperx.style_library import StyleLibrary
from clipperx.style_retrieval import (
    LocalStyleEncoder,
    StyleQuery,
    retrieve_style_cards,
)
from clipperx.visual_contracts import NormalizedRect

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
CAMERA = NormalizedRect(x=0, y=0, width=1, height=0.3)
CONTENT = NormalizedRect(x=0, y=0.3, width=1, height=0.7)


@dataclass
class EncoderSettings:
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    style_embedding_model_path: str = ""


@pytest.fixture
def encoder_settings(monkeypatch: pytest.MonkeyPatch) -> EncoderSettings:
    """Exercise constructor settings without loading .env or provider credentials."""
    settings = EncoderSettings()
    config = ModuleType("clipperx.config")
    setattr(config, "settings", settings)
    monkeypatch.setitem(sys.modules, "clipperx.config", config)
    return settings


class FakeEncoder:
    model_identity = "fake-encoder-v1"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(self, texts: list[str]) -> np.ndarray:
        self.calls.append(texts[:])
        return np.asarray([[1, 0] if t == "gameplay" else [0, 1] for t in texts], dtype=float)


def card(card_id: str, **changes: Any) -> StyleCard:
    data: dict[str, Any] = dict(
        card_id=card_id,
        clip_fingerprint=card_id,
        source_clip_ref="source_refs/test.json",
        duration=10,
        created_at=NOW,
        scene_family="gameplay_with_camera",
        keyframes=[dict(timestamp=0, path="frames/test.jpg")],
        camera_regions=[CAMERA],
        content_regions=[CONTENT],
        background_treatment="cover",
        title_regions=[],
        brand_regions=[],
        subtitle_regions=[],
        subtitle_palette=[],
        cut_rate_per_minute=0,
        zoom_count=0,
        special_techniques=[],
        visual_quality=1,
        narrative_quality=None,
        approval="unreviewed",
        description="gameplay",
        evidence=[],
    )
    data.update(changes)
    return StyleCard.model_validate(data)


def query(**changes: Any) -> StyleQuery:
    return StyleQuery.model_validate(
        dict(
            scene_family="gameplay_with_camera",
            has_camera=True,
            preserve_content=True,
            camera_regions=[CAMERA],
            content_regions=[CONTENT],
            description="gameplay",
            **changes,
        )
    )


def retrieve(cards: list[StyleCard], **kwargs: Any) -> Any:
    return retrieve_style_cards(
        query(),
        cards,
        encoder=kwargs.pop("encoder", FakeEncoder()),
        clock=lambda: NOW,
        **kwargs,
    )


def test_gameplay_query_excludes_full_camera_only_card() -> None:
    results = retrieve([card("gameplay"), card("full_camera", scene_family="full_camera")])
    assert [item.card.card_id for item in results] == ["gameplay"]
    assert results.coverage == "sparse"


def test_recent_approved_style_wins_only_after_compatibility() -> None:
    newer = card("new", scene_family="full_camera", approval="approved")
    older = card("old", created_at=NOW - timedelta(days=900), approval="approved")
    assert [item.card.card_id for item in retrieve([newer, older])] == ["old"]


@pytest.mark.parametrize(
    "changes",
    [
        {"scene_family": "unknown"},
        {"scene_family": "mixed"},
        {"camera_regions": []},
        {"content_regions": []},
        {"approval": "rejected"},
        {"scene_family": "browser_with_camera"},
        {"scene_family": "multi_person"},
    ],
)
def test_filter_precedes_embedding_and_never_reintroduces_rejected(changes: dict) -> None:
    encoder = FakeEncoder()
    result = retrieve([card("bad", **changes)], encoder=encoder)
    assert not result and result.coverage == "sparse"
    assert encoder.calls == []
    assert result.rejections[0].card_id == "bad"
    assert result.rejections[0].reasons


def test_camera_absence_is_respected_for_unknown_query() -> None:
    result = retrieve_style_cards(
        StyleQuery(scene_family=SceneFamily.UNKNOWN, has_camera=False, description="content"),
        [card("cam"), card("none", scene_family="content_only", camera_regions=[])],
        encoder=FakeEncoder(),
        clock=lambda: NOW,
    )
    assert [match.card.card_id for match in result] == ["none"]


def test_unknown_query_with_content_need_excludes_full_camera_even_with_rectangles() -> None:
    result = retrieve_style_cards(
        StyleQuery(preserve_content=True, description="gameplay"),
        [card("full", scene_family="full_camera")],
        encoder=FakeEncoder(),
    )
    assert not result


def test_query_camera_geometry_requires_camera_even_without_a_family_label() -> None:
    result = retrieve_style_cards(
        StyleQuery(camera_regions=[CAMERA], description="camera"),
        [card("none", scene_family="content_only", camera_regions=[])],
        encoder=FakeEncoder(),
    )
    assert not result


@pytest.mark.parametrize("copied_query", [False, True])
def test_camera_geometry_cannot_be_bypassed_by_explicit_absence_flag(copied_query: bool) -> None:
    encoder = FakeEncoder()
    with pytest.raises(ValueError, match="has_camera=False.*camera_regions"):
        if copied_query:
            contradictory = StyleQuery(has_camera=False, description="content").model_copy(
                update={"camera_regions": [CAMERA]}
            )
        else:
            contradictory = StyleQuery(
                has_camera=False, camera_regions=[CAMERA], description="content"
            )
        retrieve_style_cards(
            contradictory,
            [card("none", scene_family="content_only", camera_regions=[]), card("cam")],
            encoder=encoder,
        )
    assert encoder.calls == []


@pytest.mark.parametrize("count", [0, 1, 2, 3])
def test_exact_people_count_is_not_invented_from_card_geometry(count: int) -> None:
    result = retrieve_style_cards(
        StyleQuery(
            scene_family=SceneFamily.MULTI_PERSON, visible_people=count, description="people"
        ),
        [card("many", scene_family="multi_person")],
        encoder=FakeEncoder(),
    )
    assert not result and result.coverage == "sparse"
    assert "exact_people_count_unavailable" in result.rejections[0].reasons


def test_multiple_people_requirement_uses_only_explicit_family() -> None:
    result = retrieve_style_cards(
        StyleQuery(people_requirement="multiple", description="people"),
        [card("many", scene_family="multi_person"), card("camera")],
        encoder=FakeEncoder(),
    )
    assert [item.card.card_id for item in result] == ["many"]
    assert "people:multiple_family_only;exact_count_unknown" in result[0].reasons


def test_retrieval_returns_bounded_scores_and_auditable_components() -> None:
    result = retrieve([card("a")])[0]
    assert result.compatibility_score == 1
    assert result.style_score == pytest.approx(1)
    assert any(reason == "semantic=1.000000" for reason in result.reasons)
    assert all("O:" not in reason for reason in result.reasons)


def test_exact_six_term_weights_semantics_and_neutral_missing_date() -> None:
    result = retrieve([card("a", description="different", visual_quality=0, created_at=None)])[0]
    assert result.style_score == pytest.approx(0.35 + 0.20 + 0.15 + 0.05 * 0.5)
    assert "recency=0.500000" in result.reasons


def test_recency_is_bounded_and_approval_narrative_do_not_change_ranking() -> None:
    future = card("future", created_at=NOW + timedelta(days=3650), approval="unreviewed")
    now = card("now", approval="approved", narrative_quality=1)
    old = card("old", created_at=NOW - timedelta(days=3650))
    result = retrieve([old, now, future])
    assert [m.card.card_id for m in result] == ["future", "now", "old"]
    assert result[0].style_score == result[1].style_score == 1


def test_three_diverse_fingerprints_max_and_sparse_after_deduplication() -> None:
    cards = [card(str(i), clip_fingerprint=str(i // 2)) for i in range(10)]
    result = retrieve(cards, limit=100)
    assert len(result) == 3 and result.coverage == "sufficient"
    assert len({m.card.clip_fingerprint for m in result}) == 3
    assert len(result[:2]) == 2
    assert retrieve(cards[:2]).coverage == "sparse"
    assert retrieve(cards, limit=1).coverage == "sufficient"


def test_empty_library_is_sparse_without_loading_model(tmp_path: Path) -> None:
    result = retrieve_style_cards(query(), StyleLibrary(tmp_path / "empty"))
    assert not result and result.coverage == "sparse"


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_limit_rejected(limit: Any) -> None:
    with pytest.raises(ValueError, match="limit"):
        retrieve([], limit=limit)


def test_index_reuses_content_keys_across_reorder_and_invalidates_changes(tmp_path: Path) -> None:
    index = tmp_path / "style_index.npz"
    encoder = FakeEncoder()
    first = [card("a"), card("b", description="other")]
    retrieve(first, encoder=encoder, index_path=index)
    encoder.calls.clear()
    result = retrieve(first[::-1], encoder=encoder, index_path=index)
    assert [m.card.card_id for m in result] == ["a", "b"]
    assert encoder.calls == [["gameplay"]]
    encoder.calls.clear()
    retrieve([card("a", description="changed")], encoder=encoder, index_path=index)
    assert ["changed"] in encoder.calls
    encoder.model_identity = "fake-encoder-v2"
    encoder.calls.clear()
    retrieve(first, encoder=encoder, index_path=index)
    assert ["gameplay", "other"] in encoder.calls


@pytest.mark.parametrize("corruption", ["bytes", "nonfinite", "dimensions", "object", "keys"])
def test_corrupt_index_is_rebuilt_without_unpickling(tmp_path: Path, corruption: str) -> None:
    index = tmp_path / "style_index.npz"
    retrieve([card("a")], index_path=index)
    if corruption == "bytes":
        index.write_bytes(b"broken zip")
    else:
        with np.load(index, allow_pickle=False) as cached:
            values = {name: cached[name].copy() for name in cached.files}
        if corruption == "nonfinite":
            values["vectors"][0, 0] = np.nan
        elif corruption == "dimensions":
            values["vectors"] = np.ones((1, 5))
        elif corruption == "object":
            values["vectors"] = np.asarray([[object()]], dtype=object)
        else:
            values["keys"] = np.asarray(["wrong"])
        np.savez(index, **values)
    encoder = FakeEncoder()
    assert retrieve([card("a")], encoder=encoder, index_path=index)[0].style_score == 1
    assert encoder.calls == [["gameplay"], ["gameplay"]]


def test_concurrent_index_writers_preserve_valid_entries(tmp_path: Path) -> None:
    index = tmp_path / "style_index.npz"
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda i: retrieve([card(str(i))], index_path=index), range(2)))
    assert all(result[0].style_score == 1 for result in results)
    encoder = FakeEncoder()
    retrieve([card("0"), card("1")], encoder=encoder, index_path=index)
    assert encoder.calls == [["gameplay"]]
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize(
    "vectors", [np.asarray([[np.nan, 1]]), np.zeros((1, 2)), np.asarray([1, 2]), np.ones((1, 0))]
)
def test_invalid_encoder_output_fails_without_lexical_fallback(vectors: np.ndarray) -> None:
    class Broken(FakeEncoder):
        def encode(self, texts: list[str]) -> np.ndarray:
            return vectors

    with pytest.raises(ValueError, match="embedding"):
        retrieve([card("a")], encoder=Broken())


def test_missing_local_model_fails_actionably_without_import(
    tmp_path: Path, encoder_settings: EncoderSettings,
) -> None:
    with pytest.raises(FileNotFoundError, match="style_embedding_model_path.*download"):
        LocalStyleEncoder(model_path=tmp_path / "absent")


def test_library_integrity_failure_propagates_before_encoder(tmp_path: Path) -> None:
    (tmp_path / "library_manifest.json").write_text("{}broken")
    with pytest.raises(ValueError):
        retrieve_style_cards(query(), StyleLibrary(tmp_path), encoder=FakeEncoder())


def test_library_index_cannot_escape_its_integrity_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="containment"):
        retrieve_style_cards(
            query(),
            StyleLibrary(tmp_path / "library"),
            index_path=tmp_path / "outside.npz",
        )


def test_failed_atomic_publication_keeps_previous_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index = tmp_path / "style_index.npz"
    retrieve([card("old")], index_path=index)
    before = index.read_bytes()
    original = Path.replace

    def fail_temporary(path: Path, target: Path) -> Path:
        if path.suffix == ".tmp":
            raise OSError("synthetic publication failure")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", fail_temporary)
    with pytest.raises(OSError, match="publication failure"):
        retrieve([card("new")], index_path=index)
    assert index.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))
    assert retrieve([card("old")], index_path=index)[0].card.card_id == "old"


def test_encoder_dimension_mismatch_never_publishes_index(tmp_path: Path) -> None:
    class WrongDimension(FakeEncoder):
        def encode(self, texts: list[str]) -> np.ndarray:
            self.calls.append(texts)
            return np.ones((len(texts), 2 if len(self.calls) == 1 else 3))

    with pytest.raises(ValueError, match="embedding"):
        retrieve([card("a")], encoder=WrongDimension(), index_path=tmp_path / "style_index.npz")
    assert not (tmp_path / "style_index.npz").exists()


@pytest.mark.parametrize("explicit_path", [True, False])
def test_local_encoder_passes_offline_flags_and_hashes_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    encoder_settings: EncoderSettings, explicit_path: bool,
) -> None:
    for name in ["modules.json", "config.json", "model.safetensors", "tokenizer.json"]:
        (tmp_path / name).write_text("synthetic model, never loaded")
    constructors: list[tuple[str, dict]] = []
    encodings: list[dict] = []

    class StubModel:
        def __init__(self, path: str, **kwargs: Any) -> None:
            constructors.append((path, kwargs))

        def encode(self, texts: list[str], **kwargs: Any) -> np.ndarray:
            encodings.append(kwargs)
            return np.ones((len(texts), 2))

    encoder_settings.embedding_model = "test/local-encoder"
    encoder_settings.style_embedding_model_path = str(tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(
            SentenceTransformer=StubModel,
        ),
    )
    model_path = tmp_path if explicit_path else None
    encoder = LocalStyleEncoder(model_path=model_path)
    encoder.encode(["example"])
    assert constructors[0][0] == str(tmp_path.resolve())
    assert constructors[0][1]["local_files_only"] is True
    assert constructors[0][1]["trust_remote_code"] is False
    assert Path(constructors[0][1]["cache_folder"]).is_relative_to(
        Path(__file__).resolve().parents[2]
    )
    assert encodings[0]["normalize_embeddings"] is True
    assert encodings[0]["convert_to_numpy"] is True
    (tmp_path / "tokenizer.json").write_text("different tokenizer")
    assert LocalStyleEncoder(model_path=model_path).model_identity != encoder.model_identity


def test_default_minilm_rejects_unpinned_weights_before_model_import(
    tmp_path: Path, encoder_settings: EncoderSettings,
) -> None:
    for name in ["modules.json", "config.json", "model.safetensors", "tokenizer.json"]:
        (tmp_path / name).write_text("wrong weights")
    with pytest.raises(ValueError, match="pinned SHA-256"):
        LocalStyleEncoder(model_path=tmp_path)
