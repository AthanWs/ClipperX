from pathlib import Path

import pytest

from clipperx.render_assets import (
    AssetKind, PreparedBrandAsset, RenderAssetRegistry, default_render_assets, validate_asset_loadability,
    prepare_brand_asset, place_brand_asset, compile_brand_layer,
)


def test_registry_is_detached_and_read_only() -> None:
    originals = {"kick_brand": Path("host.png")}
    registry = RenderAssetRegistry(brand_images=originals, fonts={}, sfx={})
    originals.clear()
    assert registry.brand_images["kick_brand"] == Path("host.png")
    with pytest.raises(TypeError):
        registry.brand_images["other"] = Path("other.png")  # type: ignore[index]


def test_default_host_registry_resolves_existing_named_assets() -> None:
    assets = default_render_assets()
    assert assets.brand_images["kick_brand"].name == "KickAthanW.png"
    assert assets.fonts == {
        "headline_impact": Path("C:/Windows/Fonts/impact.ttf"),
        "subtitle_bold": Path("C:/Windows/Fonts/arialbd.ttf"),
    }
    assert assets.sfx == {"whoosh_sa3_softgate_01":
        Path(__file__).resolve().parents[2] / "assets/sfx/whoosh_sa3_softgate_01.wav"}


@pytest.mark.parametrize("asset_id", ["../brand", "C:/brand.png", "https://host/image", "x.png"])
def test_registry_rejects_paths_as_ids(asset_id: str) -> None:
    with pytest.raises(ValueError, match="asset ID"):
        RenderAssetRegistry({asset_id: Path("host.png")}, {}, {})


@pytest.mark.parametrize("kind", ["brand", "font", "sfx"])
def test_asset_loadability_rejects_missing_and_invalid_files(
    tmp_path: Path, kind: AssetKind
) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="missing"):
        validate_asset_loadability(kind, missing)
    broken = tmp_path / "invalid"
    broken.write_bytes(b"" if kind == "sfx" else b"not an image or font")
    with pytest.raises(ValueError, match="not loadable"):
        validate_asset_loadability(kind, broken)


def test_default_assets_decode_without_modifying_originals() -> None:
    import hashlib

    assets = default_render_assets()
    for kind, table in (("brand", assets.brand_images), ("font", assets.fonts)):
        for path in table.values():
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            validate_asset_loadability(kind, path)  # type: ignore[arg-type]
            assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_derived_brand_trims_once_by_content_preserving_original(tmp_path: Path) -> None:
    import hashlib
    from PIL import Image

    source = tmp_path / "source.png"
    image = Image.new("RGBA", (100, 60))
    image.paste((20, 200, 30, 255), (15, 20, 85, 40))
    image.save(source)
    registry = RenderAssetRegistry({"host_brand": source}, {}, {})
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    cache = tmp_path / "cache"
    prepared = prepare_brand_asset(registry, "host_brand", cache_dir=cache)
    assert prepared.source_sha256 == before
    assert prepared.source_size == (100, 60)
    assert prepared.source_bbox == (15, 20, 85, 40)
    assert prepared.size == (70, 20)
    assert prepared.path.parent == cache
    stamp = prepared.path.stat().st_mtime_ns
    assert prepare_brand_asset(registry, "host_brand", cache_dir=cache) == prepared
    assert prepared.path.stat().st_mtime_ns == stamp
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
    with Image.open(prepared.path) as cropped:
        assert cropped.size == (70, 20) and cropped.mode == "RGBA"
        assert cropped.getchannel("A").getextrema() == (255, 255)
    image.putpixel((0, 0), (255, 0, 0, 255))
    image.save(source)
    changed = prepare_brand_asset(registry, "host_brand", cache_dir=cache)
    assert changed.path != prepared.path and prepared.path.is_file()


@pytest.mark.parametrize("corrupt", [False, True])
def test_empty_or_corrupt_brand_is_explicit_error(tmp_path: Path, corrupt: bool) -> None:
    from PIL import Image
    source = tmp_path / "source.png"
    if corrupt:
        source.write_bytes(b"broken")
    else:
        Image.new("RGBA", (10, 10)).save(source)
    registry = RenderAssetRegistry({"brand": source}, {}, {})
    with pytest.raises(ValueError, match="transparent|loadable"):
        prepare_brand_asset(registry, "brand", cache_dir=tmp_path / "cache")


def test_actual_brand_bbox_and_deterministic_visual_layer_hook(tmp_path: Path) -> None:
    from clipperx.filtergraph import PixelRect

    prepared = prepare_brand_asset(default_render_assets(), "kick_brand", cache_dir=tmp_path)
    assert prepared.source_bbox == (311, 157, 1102, 343)
    assert prepared.size == (791, 186)
    layer = place_brand_asset(prepared, destination=PixelRect(100, 1000, 800, 200),
                             start=2.25, end=3.75, z_index=7, shot_id="../host")
    assert layer.destination == PixelRect(100, 1000, 800, 200)
    assert layer.placement.width == 800 and layer.placement.height == 188
    assert layer.placement.x == 100 and layer.placement.y == 1006
    assert abs(layer.placement.width / layer.placement.height - 791 / 186) < .02
    graph = compile_brand_layer(layer, input_label="video", brand_label="1:v", output_label="out")
    assert "scale=w=800:h=188" in graph
    assert "overlay=x=100:y=1006" in graph
    assert "gte(t,2.25)*lt(t,3.75)" in graph
    assert "../host" not in graph and "drawtext" not in graph and "movie=" not in graph
    assert "[1:v]" in graph and "[video]" in graph and graph.endswith("[out]")
    with pytest.raises(ValueError, match="label"):
        compile_brand_layer(layer, input_label="v];movie=x", brand_label="1:v", output_label="out")
    collision = compile_brand_layer(layer, input_label="out_brand", brand_label="png",
                                    output_label="out")
    assert "format=rgba[out_brand_];[out_brand][out_brand_]" in collision


def test_finite_static_png_is_held_without_changing_authored_interval() -> None:
    from clipperx.filtergraph import PixelRect

    prepared = PreparedBrandAsset(
        source_path=Path("source.png"), path=Path("prepared.png"), source_sha256="0" * 64,
        source_size=(64, 32), source_bbox=(8, 4, 56, 28), size=(48, 24),
    )
    start, end = 2.25, 3.75
    layer = place_brand_asset(prepared, destination=PixelRect(100, 1000, 800, 200),
                              start=start, end=end, z_index=7, shot_id="host")
    graph = compile_brand_layer(layer, input_label="video", brand_label="png", output_label="out")

    assert graph == (
        "[png]scale=w=400:h=200,setsar=1,format=rgba[out_brand];"
        "[video][out_brand]overlay=x=300:y=1000:format=auto:"
        "enable='gte(t,2.25)*lt(t,3.75)':eof_action=repeat:repeatlast=1[out]"
    )
    checkpoints = {"K3": 2.0, "K4": 2.5, "K5": 3.5, "K6": 3.75}
    enabled = {name: start <= timestamp < end for name, timestamp in checkpoints.items()}
    assert enabled == {"K3": False, "K4": True, "K5": True, "K6": False}
    assert graph.count("[png]") == 1
    assert "shortest=1" not in graph and "endall" not in graph
    assert "loop" not in graph and "fps=" not in graph


def test_prepared_asset_detaches_accepted_dimension_containers(tmp_path: Path) -> None:
    from clipperx.render_assets import PreparedBrandAsset

    original = prepare_brand_asset(default_render_assets(), "kick_brand", cache_dir=tmp_path)
    size, source_size, bbox = list(original.size), list(original.source_size), list(original.source_bbox)
    detached = PreparedBrandAsset(original.source_path, original.path, original.source_sha256,
        source_size, bbox, size)  # type: ignore[arg-type]
    size.clear()
    source_size.clear()
    bbox.clear()
    assert detached == original


def test_corrupt_derived_cache_is_explicit_error(tmp_path: Path) -> None:
    assets = default_render_assets()
    prepared = prepare_brand_asset(assets, "kick_brand", cache_dir=tmp_path)
    prepared.path.write_bytes(b"corrupt cached PNG")
    with pytest.raises(ValueError, match="cache is not loadable"):
        prepare_brand_asset(assets, "kick_brand", cache_dir=tmp_path)


def test_decoded_png_hook_places_actual_trimmed_pixels_and_half_open_enable(tmp_path: Path) -> None:
    import subprocess
    from PIL import Image, ImageChops
    from clipperx.filtergraph import PixelRect

    prepared = prepare_brand_asset(default_render_assets(), "kick_brand", cache_dir=tmp_path)
    layer = place_brand_asset(prepared, destination=PixelRect(100, 1000, 800, 200),
                             start=2.25, end=3.75, z_index=4, shot_id="host")
    ffmpeg = Path(__file__).resolve().parents[2] / "data/tools/ffmpeg/9.0.2/bin/ffmpeg.exe"
    assert ffmpeg.is_file()
    hook = compile_brand_layer(layer, input_label="video", brand_label="png", output_label="out")
    for timestamp in (2.5, 3.75):
        graph = (f"[0:v]settb=1/60000000,setpts=PTS+{timestamp}/TB[video];"
                 f"[1:v]settb=1/60000000,setpts=PTS+{timestamp}/TB[png];" + hook)
        result = subprocess.run([
            str(ffmpeg), "-hide_banner", "-nostdin", "-f", "lavfi", "-i",
            "color=c=0x607080:s=1080x1920:r=10:d=0.1,format=rgba",
            "-loop", "1", "-framerate", "10", "-i", str(prepared.path),
            "-filter_complex", graph, "-map", "[out]", "-frames:v", "1",
            "-fps_mode", "passthrough", "-pix_fmt", "rgba", "-f", "rawvideo", "pipe:1",
        ], capture_output=True, check=True, timeout=30)
        assert len(result.stdout) == 1080 * 1920 * 4
        (tmp_path / f"decoded-brand-{timestamp}.log").write_bytes(result.stderr)
        image = Image.frombytes("RGBA", (1080, 1920), result.stdout)
        background = Image.new("RGBA", image.size, image.getpixel((0, 0)))
        bbox = ImageChops.difference(image, background).convert("RGB").getbbox()
        if timestamp == 3.75:
            assert bbox is None  # exact end is disabled
        else:
            # The original first alpha column has max opacity 1/255; rounding
            # against this background makes it invisible. Do not stretch/re-crop.
            with Image.open(prepared.path) as png:
                assert png.getchannel("A").crop((0, 0, 1, 186)).getextrema() == (0, 1)
            assert layer.placement == PixelRect(100, 1006, 800, 188)
            assert bbox == (101, 1006, 900, 1194)
            image.crop((100, 1000, 900, 1200)).save(tmp_path / "decoded-brand.png")
            print(f"decoded brand diagnostic: {tmp_path}; visible_bbox={bbox}")
