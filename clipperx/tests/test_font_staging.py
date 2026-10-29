from dataclasses import FrozenInstanceError, replace
import hashlib
import os
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from typing import Any, cast

import pytest

from clipperx import font_staging as staging
from clipperx.render_assets import RenderAssetRegistry


def _fixture(tmp_path: Path) -> tuple[RenderAssetRegistry, dict[str, str], Path]:
    sources = tmp_path / "sources"
    sources.mkdir()
    (tmp_path / "job").mkdir()
    data = {"headline_impact": b"fixture-impact-v1", "subtitle_bold": b"fixture-arialbold-v1"}
    paths = {key: sources / f"{key}.ttf" for key in data}
    for key, path in paths.items():
        path.write_bytes(data[key])
    return (RenderAssetRegistry({}, paths, {}),
            {key: hashlib.sha256(value).hexdigest() for key, value in data.items()},
            tmp_path / "job" / "fonts")


def _stage(registry: RenderAssetRegistry, expected: dict[str, str], destination: Path
           ) -> staging.FontStage:
    return staging._stage_fonts(registry, stage_dir=destination, expected_sha256=expected,
                                identity_authority="test-fixture")


def _link(source: Path, destination: Path, *, symbolic: bool = False, directory: bool = False
          ) -> None:
    try:
        if symbolic:
            destination.symlink_to(source, target_is_directory=directory)
        else:
            os.link(source, destination)
    except OSError as exc:
        # Missing privilege or unsupported operation only; other IO failures fail the test.
        if exc.errno in (1, 38, 95) or getattr(exc, "winerror", None) in (1, 50, 1314):
            pytest.skip(f"{'symlink' if symbolic else 'hardlink'} unsupported: {exc}")
        raise


def _stat_view(value: os.stat_result, **changes: object) -> Any:
    names = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_nlink",
             "st_mode", "st_file_attributes", "st_birthtime_ns")
    content = {name: getattr(value, name) for name in names if hasattr(value, name)}
    content.update(changes)
    return SimpleNamespace(**content)


def _guard_path_io(monkeypatch: pytest.MonkeyPatch, blocked: Path, *, deep_prefix: Path | None = None
                   ) -> None:
    for method in ("stat", "lstat", "resolve", "open"):
        original = getattr(Path, method)

        def fail_if_unchecked(path: Path, *args: Any, _original: Any = original,
                              **kwargs: Any) -> Any:
            if (path == blocked or not path.is_absolute() or path.drive.startswith("\\\\")
                    or (deep_prefix is not None and path.parts[:len(deep_prefix.parts)]
                        == deep_prefix.parts)):
                raise AssertionError("Syntax-invalid fixture path reached filesystem")
            return _original(path, *args, **kwargs)

        monkeypatch.setattr(Path, method, fail_if_unchecked)


def _lying_proxy(contents: dict[str, Any], *, reported: int, iterator_items: bool = False
                 ) -> tuple[Any, dict[str, int]]:
    observed = {"items": 0, "iteration": 0}

    class IteratorSentinel:
        def __iter__(self) -> Any:
            observed["iteration"] += 1
            raise AssertionError("Iterator-like proxy items must not be consumed")

    class LyingDictionary(dict[str, Any]):
        def __len__(self) -> int:
            return reported

        def __iter__(self) -> Any:
            observed["iteration"] += 1
            raise AssertionError("Proxy key iterator must not be consumed")

        def items(self) -> Any:
            observed["items"] += 1
            return IteratorSentinel() if iterator_items else dict.items(self)

    return MappingProxyType(LyingDictionary(contents)), observed


def _forbid_fixture_io(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Malformed container reached filesystem before rejection")

    for method in ("stat", "lstat", "resolve", "open"):
        monkeypatch.setattr(Path, method, fail)


def test_stage_fixture_copies_and_rehashes_two_originals_into_same_registry(tmp_path: Path) -> None:
    import hashlib
    from clipperx.font_staging import _stage_fonts, verify_font_stage
    from clipperx.render_assets import RenderAssetRegistry
    sources = tmp_path / "sources"
    job = tmp_path / "job"
    sources.mkdir()
    job.mkdir()
    literals = {"headline_impact": b"fixture-impact-v1",
                "subtitle_bold": b"fixture-arialbold-v1"}
    paths = {key: sources / (key + ".ttf") for key in literals}
    for key, path in paths.items():
        path.write_bytes(literals[key])
    expected = {key: hashlib.sha256(data).hexdigest()
                for key, data in literals.items()}
    registry = RenderAssetRegistry({}, paths, {})
    stage = _stage_fonts(registry, stage_dir=job / "fonts",
                         expected_sha256=expected, identity_authority="test-fixture")
    assert stage.identity_authority == "test-fixture"
    assert stage.selected_file_identity_status == "unverified"
    assert tuple(font.asset_id for font in stage.fonts) == tuple(literals)
    for font in stage.fonts:
        assert font.original_path.read_bytes() == literals[font.asset_id]
        assert font.staged_path.read_bytes() == literals[font.asset_id]
        assert font.original_sha256 == font.staged_sha256 == expected[font.asset_id]
        assert stage.registry.fonts[font.asset_id] == font.measurement_path == font.staged_path
        assert font.measurement_sha256 == font.staged_sha256
        assert font.face_index == font.measurement_face_index == 0
    verify_font_stage(stage)


def test_stage_preserves_brand_sfx_and_excludes_extra_font_paths(tmp_path: Path) -> None:
    registry, expected, destination = _fixture(tmp_path)
    brands = {"logo": tmp_path / "never-open-brand.png"}
    sounds = {"cue": Path("relative-never-open.wav")}
    fonts = dict(registry.fonts, extra=tmp_path / "never-open-font.ttf")
    registry = RenderAssetRegistry(brands, fonts, sounds)
    stage = _stage(registry, expected, destination)
    assert dict(stage.registry.brand_images) == brands
    assert dict(stage.registry.sfx) == sounds
    assert tuple(stage.registry.fonts) == tuple(expected)
    assert "extra" in registry.fonts
    assert stage.registry.brand_images is not registry.brand_images
    assert stage.registry.sfx is not registry.sfx
    with pytest.raises(ValueError, match="Unknown"):
        stage.registry.resolve("font", "extra")
    assert len(list(destination.iterdir())) == 2
    for font in stage.fonts:
        assert font.staged_stamp.size_bytes == len(font.staged_path.read_bytes())
        assert font.original_stamp.size_bytes == font.original_path.stat().st_size
        assert font.staged_stamp.inode == font.staged_path.stat().st_ino
        assert font.staged_stamp.link_count == 1


def test_public_stage_rejects_fixture_hashes_before_directory_creation(tmp_path: Path) -> None:
    registry, _, destination = _fixture(tmp_path)
    with pytest.raises(ValueError, match="preflight.*headline_impact.*fonts"):
        staging.stage_original_fonts(registry, stage_dir=destination)
    assert not destination.exists()


@pytest.mark.parametrize("policy", ["fixture", "missing", "extra", "empty", "wrong_type", "bad_hex",
                                    "proxy_excess", "proxy_iterator", "proxy_bounded"])
def test_approved_authority_rejects_arbitrary_expected_hashes(tmp_path: Path, policy: str,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
    registry, expected, destination = _fixture(tmp_path)
    if policy.startswith("proxy_"):
        entries = dict(expected)
        if policy == "proxy_excess":
            entries["extra"] = "0" * 64
        proxied, observed = _lying_proxy(entries, reported=2 if policy != "proxy_bounded" else 1,
                                          iterator_items=policy == "proxy_iterator")
        if policy == "proxy_bounded":
            stage = staging._stage_fonts(registry, stage_dir=destination,
                                         expected_sha256=proxied, identity_authority="test-fixture")
            staging.verify_font_stage(stage)
            assert observed == {"items": 1, "iteration": 0}
            return
        with monkeypatch.context() as scoped:
            _forbid_fixture_io(scoped)
            with pytest.raises(ValueError, match="preflight.*finite"):
                staging._stage_fonts(registry, stage_dir=destination,
                                     expected_sha256=proxied, identity_authority="test-fixture")
        assert observed == {"items": 1, "iteration": 0}
        assert not destination.exists()
        return
    supplied: Any = dict(expected)
    if policy == "missing":
        supplied.pop("subtitle_bold")
    elif policy == "extra":
        supplied["extra"] = "0" * 64
    elif policy == "empty":
        supplied["headline_impact"] = ""
    elif policy == "wrong_type":
        supplied = iter(expected.items())
    elif policy == "bad_hex":
        supplied["headline_impact"] = "g" * 64
    with pytest.raises(ValueError, match="preflight.*fonts"):
        staging._stage_fonts(registry, stage_dir=destination, expected_sha256=supplied,
                             identity_authority="approved-originals-v1")
    assert not destination.exists()
    with pytest.raises(ValueError):
        staging._stage_fonts(registry, stage_dir=destination, expected_sha256=expected,
                             identity_authority=cast(Any, "invalid"))
    assert not destination.exists()


@pytest.mark.parametrize("case", ["missing_id", "missing_file", "empty", "wrong_hash", "huge",
                                 "relative", "unc", "device", "ads", "deep", "long", "many_fonts",
                                 "many_brands", "many_sfx", "bad_table", "bad_id", "bad_value",
                                 "proxy_fonts_excess", "proxy_brand_images_excess", "proxy_sfx_excess",
                                 "proxy_fonts_iterator", "proxy_brand_images_iterator", "proxy_sfx_iterator",
                                 "proxy_fonts_bounded", "proxy_brand_images_bounded", "proxy_sfx_bounded"])
def test_stage_rejects_missing_required_empty_and_wrong_hash_sources(tmp_path: Path, case: str,
                                                                  monkeypatch: pytest.MonkeyPatch
                                                                  ) -> None:
    registry, expected, destination = _fixture(tmp_path)
    if case.startswith("proxy_"):
        field, mode = case.removeprefix("proxy_").rsplit("_", 1)
        entries = (dict(registry.fonts) if field == "fonts" else {"asset": tmp_path / "unopened"})
        if mode == "excess":
            entries.update({f"extra_{i}": tmp_path / "unopened" for i in range(65 - len(entries))})
        proxy, observed = _lying_proxy(entries, reported=1, iterator_items=mode == "iterator")
        object.__setattr__(registry, field, proxy)
        if mode == "bounded":
            stage = _stage(registry, expected, destination)
            staging.verify_font_stage(stage)
            assert observed == {"items": 1, "iteration": 0}
            return
        with monkeypatch.context() as scoped:
            _forbid_fixture_io(scoped)
            with pytest.raises(ValueError, match="preflight.*finite"):
                _stage(registry, expected, destination)
        assert observed == {"items": 1, "iteration": 0}
        assert not destination.exists()
        return
    fonts = dict(registry.fonts)
    if case == "missing_id":
        fonts.pop("headline_impact")
    elif case == "missing_file":
        fonts["headline_impact"].unlink()
    elif case == "empty":
        fonts["headline_impact"].write_bytes(b"")
    elif case == "wrong_hash":
        expected["headline_impact"] = "0" * 64
    elif case == "huge":
        with fonts["headline_impact"].open("r+b") as file:
            file.truncate(16 * 1024 * 1024 + 1)
    elif case in ("relative", "unc", "device", "ads", "deep", "long"):
        fonts["headline_impact"] = {
            "relative": Path("relative.ttf"), "unc": Path(r"\\server\share\font.ttf"),
            "device": Path(r"\\?\C:\fixture-font.ttf"),
            "ads": tmp_path / "font.ttf:stream", "deep": tmp_path.joinpath(*(["x"] * 33)),
            "long": tmp_path / ("x" * 2049),
        }[case]
    elif case == "many_fonts":
        fonts.update({f"extra_{index}": tmp_path / "never-open" for index in range(63)})
    registry = RenderAssetRegistry({}, fonts, {})
    if case in ("relative", "unc", "device", "ads", "deep", "long"):
        blocked = fonts["headline_impact"]
        _guard_path_io(monkeypatch, blocked, deep_prefix=tmp_path / "x" if case == "deep" else None)
    if case in ("many_brands", "many_sfx"):
        field = "brand_images" if case == "many_brands" else "sfx"
        object.__setattr__(registry, field, {f"item_{i}": Path("unopened") for i in range(65)})
    elif case == "bad_table":
        object.__setattr__(registry, "sfx", iter(()))
    elif case == "bad_id":
        object.__setattr__(registry, "sfx", {"a" * 65: Path("unopened")})
    elif case == "bad_value":
        object.__setattr__(registry, "brand_images", {"logo": object()})
    with pytest.raises((ValueError, OSError), match="preflight.*fonts"):
        _stage(registry, expected, destination)
    assert not destination.exists()


@pytest.mark.parametrize("case", ["bounds", "canonical", "raw_path_change", "raw_fd_change"])
def test_bounded_reader_rejects_65_bytes_at_64_byte_limit(tmp_path: Path, case: str,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "bounded.bin"
    if case != "bounds":
        path.write_bytes(b"fixture")
        named = path.lstat()
        os.utime(path, ns=(named.st_atime_ns, named.st_mtime_ns + 10_000_000))
        if case == "canonical":
            data, stamp = staging._read_checked(path, max_bytes=64)
            named = path.lstat()
            assert data == b"fixture"
            canonical = (getattr(named, "st_birthtime_ns", named.st_ctime_ns)
                         if os.name == "nt" else named.st_ctime_ns)
            assert stamp.ctime_ns == canonical
            fallback = _stat_view(named)
            if hasattr(fallback, "st_birthtime_ns"):
                del fallback.st_birthtime_ns
            assert staging._stamp(fallback).ctime_ns == named.st_ctime_ns
            if os.name == "nt" and hasattr(named, "st_birthtime_ns"):
                for invalid in (None, -1, True, 10 ** 33):
                    with pytest.raises(ValueError):
                        staging._stamp(_stat_view(named, st_birthtime_ns=invalid))
            return
        calls = 0
        if case == "raw_fd_change":
            real_fstat = os.fstat

            def changed_fstat(fd: int) -> Any:
                nonlocal calls
                value = real_fstat(fd)
                calls += 1
                return _stat_view(value, st_ctime_ns=value.st_ctime_ns + (calls > 1))

            monkeypatch.setattr(os, "fstat", changed_fstat)
        else:
            real_lstat = Path.lstat

            def changed_lstat(target: Path, *args: Any, **kwargs: Any) -> Any:
                nonlocal calls
                value = real_lstat(target, *args, **kwargs)
                if target == path:
                    calls += 1
                    return _stat_view(value, st_ctime_ns=value.st_ctime_ns + (calls > 1))
                return value

            monkeypatch.setattr(Path, "lstat", changed_lstat)
        with pytest.raises(ValueError):
            staging._read_checked(path, max_bytes=64)
        assert calls >= 2
        return
    path.write_bytes(b"x" * 65)
    with pytest.raises(ValueError):
        staging._read_checked(path, max_bytes=64)
    path.write_bytes(b"x" * 64)
    data, stamp = staging._read_checked(path, max_bytes=64)
    assert data == b"x" * 64 and stamp.size_bytes == 64
    for invalid in (0, -1, 16 * 1024 * 1024 + 1, True):
        with pytest.raises(ValueError):
            staging._read_checked(path, max_bytes=invalid)


@pytest.mark.parametrize("case", ["existing_directory", "existing_file", "missing_parent",
                                 "relative", "unc", "device", "ads"])
def test_stage_rejects_existing_directory_without_touching_its_files(tmp_path: Path, case: str,
                                                                   monkeypatch: pytest.MonkeyPatch
                                                                   ) -> None:
    registry, expected, destination = _fixture(tmp_path)
    if case != "existing_directory":
        attempted = {
            "existing_file": destination, "missing_parent": destination / "fonts",
            "relative": Path("fonts"), "unc": Path(r"\\server\share\fonts"),
            "device": Path(r"\\?\C:\fixture-fonts"), "ads": destination.with_name("fonts:stream"),
        }[case]
        if case == "existing_file":
            destination.write_bytes(b"untouched")
        elif case in ("relative", "unc", "device", "ads"):
            _guard_path_io(monkeypatch, attempted)
        with pytest.raises((ValueError, OSError), match="preflight.*stage_dir"):
            _stage(registry, expected, attempted)
        if case == "existing_file":
            assert destination.read_bytes() == b"untouched"
        else:
            assert not destination.exists()
        return
    destination.mkdir()
    sentinel = destination / "keep"
    sentinel.write_bytes(b"untouched")
    with pytest.raises((ValueError, OSError)):
        _stage(registry, expected, destination)
    assert sentinel.read_bytes() == b"untouched"
    assert list(destination.iterdir()) == [sentinel]


@pytest.mark.parametrize("hardlink", [False, True])
def test_stage_rejects_source_path_alias_and_source_hardlink_alias(tmp_path: Path, hardlink: bool
                                                                ) -> None:
    registry, expected, destination = _fixture(tmp_path)
    paths = dict(registry.fonts)
    if hardlink:
        alias = paths["headline_impact"].parent / "alias.ttf"
        _link(paths["headline_impact"], alias)
    else:
        alias = paths["headline_impact"]
    paths["subtitle_bold"] = alias
    expected["subtitle_bold"] = expected["headline_impact"]
    with pytest.raises(ValueError, match="alias"):
        _stage(RenderAssetRegistry({}, paths, {}), expected, destination)
    assert not destination.exists()


@pytest.mark.parametrize("component", ["source_leaf", "source_parent", "stage_parent"])
def test_stage_rejects_symlink_or_reparse_components_when_supported(tmp_path: Path,
                                                                  component: str) -> None:
    registry, expected, destination = _fixture(tmp_path)
    paths = dict(registry.fonts)
    link = tmp_path / "link"
    if component == "source_leaf":
        _link(paths["headline_impact"], link, symbolic=True)
        paths["headline_impact"] = link
    elif component == "source_parent":
        _link(paths["headline_impact"].parent, link, symbolic=True, directory=True)
        paths["headline_impact"] = link / paths["headline_impact"].name
    else:
        _link(destination.parent, link, symbolic=True, directory=True)
        destination = link / "fonts"
    with pytest.raises(ValueError, match="reparse|symlink"):
        _stage(RenderAssetRegistry({}, paths, {}), expected, destination)
    assert not destination.exists()


@pytest.mark.parametrize("case", ["after_copy", "primary_and_close", "close_only"])
def test_stage_failure_retains_partial_files_and_returns_no_registry(tmp_path: Path, case: str,
                                                                    monkeypatch: pytest.MonkeyPatch
                                                                    ) -> None:
    registry, expected, destination = _fixture(tmp_path)
    original_bytes = {key: path.read_bytes() for key, path in registry.fonts.items()}
    copy = staging._copy_exclusive

    def fail_after_copy(path: Path, data: bytes) -> None:
        copy(path, data)
        if case != "after_copy":
            real_open = Path.open

            class CloseFailure:
                def __init__(self, file: Any) -> None:
                    self.file = file

                def close(self) -> None:
                    self.file.close()
                    raise OSError("fixture close failure after real close")

            def fail_close(target: Path, mode: str) -> Any:
                assert target == path and mode == "rb"
                return CloseFailure(real_open(target, mode))

            with monkeypatch.context() as scoped:
                scoped.setattr(Path, "open", fail_close)
                with staging._owned_file(path, "rb"):
                    if case == "primary_and_close":
                        raise OSError(5, "fixture copy failure")
        raise OSError("fixture copy failure")

    monkeypatch.setattr(staging, "_copy_exclusive", fail_after_copy)
    with pytest.raises(OSError, match="copy.*headline_impact.*fonts") as failure:
        _stage(registry, expected, destination)
    assert isinstance(failure.value.__cause__, OSError)
    if case != "after_copy":
        assert "retirement uncertain" in str(failure.value)
        assert "close failure" in str(failure.value)
        if case == "primary_and_close":
            assert "fixture copy failure" in str(failure.value)
    files = list(destination.iterdir())
    assert len(files) == 1 and files[0].read_bytes() == original_bytes["headline_impact"]
    with files[0].open("r+b") as reopened:
        assert reopened.read() == original_bytes["headline_impact"]
    assert {key: path.read_bytes() for key, path in registry.fonts.items()} == original_bytes


def test_stage_rechecks_source_after_copy_and_rejects_changed_bytes(tmp_path: Path,
                                                                  monkeypatch: pytest.MonkeyPatch
                                                                  ) -> None:
    registry, expected, destination = _fixture(tmp_path)
    copy = staging._copy_exclusive
    other = registry.fonts["subtitle_bold"].read_bytes()
    calls = 0

    def change_after_copy(path: Path, data: bytes) -> None:
        nonlocal calls
        copy(path, data)
        calls += 1
        if calls == 1:
            registry.fonts["headline_impact"].write_bytes(b"fixture-corruption")

    monkeypatch.setattr(staging, "_copy_exclusive", change_after_copy)
    with pytest.raises(ValueError, match="source-recheck.*headline_impact.*fonts"):
        _stage(registry, expected, destination)
    assert calls == 2 and len(list(destination.iterdir())) == 2
    assert registry.fonts["subtitle_bold"].read_bytes() == other
    for path in destination.iterdir():
        with path.open("r+b") as reopened:
            assert reopened.read()


@pytest.mark.parametrize("case", ["source_bytes", "missing_source", "stage_bytes", "missing_stage"])
def test_verify_rejects_changed_or_missing_source_and_changed_stage(tmp_path: Path, case: str) -> None:
    registry, expected, destination = _fixture(tmp_path)
    stage = _stage(registry, expected, destination)
    font = stage.fonts[0]
    path = font.original_path if "source" in case else font.staged_path
    if case.startswith("missing"):
        path.unlink()
    else:
        path.write_bytes(b"fixture-corruption")
    with pytest.raises((ValueError, OSError), match="verify.*headline_impact.*fonts"):
        staging.verify_font_stage(stage)


@pytest.mark.parametrize("case", ["source_replace", "stage_replace", "stage_hardlink", "source_hardlink"])
def test_verify_rejects_replaced_equal_bytes_and_staged_hardlink(tmp_path: Path, case: str) -> None:
    registry, expected, destination = _fixture(tmp_path)
    if case == "source_hardlink":
        _link(registry.fonts["headline_impact"], tmp_path / "allowed-source-link.ttf")
    stage = _stage(registry, expected, destination)
    if case == "source_hardlink":
        staging.verify_font_stage(stage)
        assert stage.fonts[0].original_stamp.link_count == 2
        return
    font = stage.fonts[0]
    if case == "stage_hardlink":
        _link(font.staged_path, tmp_path / "forbidden-stage-link.ttf")
    else:
        path = font.original_path if case == "source_replace" else font.staged_path
        saved = path.with_suffix(".saved")
        path.rename(saved)
        path.write_bytes(saved.read_bytes())
    with pytest.raises(ValueError, match="verify.*headline_impact.*fonts"):
        staging.verify_font_stage(stage)


@pytest.mark.parametrize("case", ["detach", "face", "status", "registry", "authority", "pin",
                                 "count", "stamp", "directory", "expected", "staged_path",
                                 "bypassed_face", "bypassed_stamp", "bypassed_registry",
                                 "verify_readonly", "normalize", "bool_stamp", "status_type",
                                 "proxy_fonts_excess", "proxy_brand_images_excess", "proxy_sfx_excess",
                                 "proxy_fonts_iterator", "proxy_brand_images_iterator", "proxy_sfx_iterator",
                                 "nested_source_constructor", "nested_source_bypass"])
def test_manifest_detaches_input_and_rejects_registry_face_status_mismatch(tmp_path: Path,
                                                                        case: str,
                                                                        monkeypatch: pytest.MonkeyPatch
                                                                        ) -> None:
    registry, expected, destination = _fixture(tmp_path)
    stage = _stage(registry, expected, destination)
    font = stage.fonts[0]
    if case.startswith("proxy_"):
        field, mode = case.removeprefix("proxy_").rsplit("_", 1)
        entries = dict(stage.registry.fonts) if field == "fonts" else {"asset": tmp_path / "unopened"}
        if mode == "excess":
            entries.update({f"extra_{i}": tmp_path / "unopened" for i in range(65 - len(entries))})
        proxy, observed = _lying_proxy(entries, reported=1, iterator_items=mode == "iterator")
        object.__setattr__(stage.registry, field, proxy)
        with monkeypatch.context() as scoped:
            _forbid_fixture_io(scoped)
            with pytest.raises(ValueError, match="verify.*finite"):
                staging.verify_font_stage(stage)
        assert observed == {"items": 1, "iteration": 0}
        return
    if case.startswith("nested_source_"):
        nested = destination / "nested"
        nested.mkdir()
        moved = nested / "original.ttf"
        font.original_path.rename(moved)
        data, stamp = staging._read_checked(moved, max_bytes=64)
        assert hashlib.sha256(data).hexdigest() == font.original_sha256
        nested_font = replace(font, original_path=moved, original_stamp=stamp)
        with pytest.raises(ValueError, match="containment"):
            if case == "nested_source_constructor":
                replace(stage, fonts=(nested_font, stage.fonts[1]))
            else:
                object.__setattr__(stage, "fonts", (nested_font, stage.fonts[1]))
                staging.verify_font_stage(stage)
        return
    if case == "detach":
        fonts: Any = list(stage.fonts)
        identity: Any = list(stage.stage_directory_id)
        detached = replace(stage, fonts=fonts, stage_directory_id=identity)
        fonts.clear()
        identity.clear()
        assert len(detached.fonts) == 2 and len(detached.stage_directory_id) == 2
        with pytest.raises(FrozenInstanceError):
            setattr(detached, "identity_authority", "approved-originals-v1")
        with pytest.raises(TypeError):
            cast(Any, detached.registry.fonts)["extra"] = tmp_path
        return
    if case == "verify_readonly":
        before_registry = stage.registry
        before_fonts = stage.fonts
        staging.verify_font_stage(stage)
        assert stage.registry is before_registry and stage.fonts is before_fonts
        return
    if case == "normalize":
        normalized = replace(font, original_path=cast(Any, str(font.original_path)),
                             expected_original_sha256=font.expected_original_sha256.upper(),
                             original_sha256=font.original_sha256.upper(),
                             staged_sha256=font.staged_sha256.upper())
        assert normalized == font
        return
    with pytest.raises(ValueError):
        if case == "face":
            replace(font, face_index=cast(Any, 1))
        elif case == "status":
            replace(stage, selected_file_identity_status=cast(Any, "verified"))
        elif case == "registry":
            replace(stage, registry=registry)
        elif case == "authority":
            replace(stage, identity_authority=cast(Any, "invalid"))
        elif case == "pin":
            replace(stage, identity_authority="approved-originals-v1")
        elif case == "count":
            replace(stage, fonts=cast(Any, ()))
        elif case == "stamp":
            replace(font.original_stamp, inode=10 ** 33)
        elif case == "bool_stamp":
            replace(font.original_stamp, size_bytes=True)
        elif case == "status_type":
            class PretendStatus:
                def __eq__(self, other: object) -> bool:
                    return True

            replace(stage, selected_file_identity_status=cast(Any, PretendStatus()))
        elif case == "directory":
            replace(stage, stage_directory_id=(0, 0))
        elif case == "expected":
            replace(font, expected_original_sha256="0" * 64)
        elif case == "staged_path":
            replace(stage, fonts=(replace(font, staged_path=font.original_path), stage.fonts[1]))
        else:
            if case == "bypassed_face":
                object.__setattr__(font, "face_index", 1)
            elif case == "bypassed_stamp":
                object.__setattr__(font.original_stamp, "size_bytes", 0)
            else:
                object.__setattr__(stage.registry, "sfx", iter(()))
            staging.verify_font_stage(stage)


@pytest.mark.parametrize("nested", [False, True])
def test_stage_rejects_destination_under_original_parent(tmp_path: Path, nested: bool) -> None:
    registry, expected, _ = _fixture(tmp_path)
    parent = registry.fonts["headline_impact"].parent
    if nested:
        (parent / "job").mkdir()
    destination = parent / "job" / "fonts" if nested else parent
    with pytest.raises(ValueError, match="preflight.*Destination overlaps original parent"):
        _stage(registry, expected, destination)
    if nested:
        assert not destination.exists()


def test_verify_rejects_replaced_stage_directory_identity(tmp_path: Path) -> None:
    registry, expected, destination = _fixture(tmp_path)
    stage = _stage(registry, expected, destination)
    saved = destination.with_name("saved-fonts")
    destination.rename(saved)
    destination.mkdir()
    for font in stage.fonts:
        (saved / font.staged_path.name).rename(font.staged_path)
    assert all(font.staged_path.stat().st_ino == font.staged_stamp.inode for font in stage.fonts)
    with pytest.raises(ValueError, match="verify.*fonts.*directory identity"):
        staging.verify_font_stage(stage)
