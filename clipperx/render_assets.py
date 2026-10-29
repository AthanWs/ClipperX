"""Explicit host asset IDs and read-only loadability checks."""

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
from types import MappingProxyType
from typing import Literal, TYPE_CHECKING, Iterator

from PIL import Image, ImageFont, PngImagePlugin

from .filtergraph import PixelRect
from .local_paths import paths
if TYPE_CHECKING:
    from ._retained_files import RetainedFiles

_ASSET_CAP = 16 * 1024**2
_PIXEL_CAP = 16777216


def _read_asset(path: Path, files: "RetainedFiles", cap: int = _ASSET_CAP) -> bytes:
    raw = bytearray()
    with files.open_binary(path) as stream:
        while len(raw) <= cap:
            chunk = stream.read(min(65536, cap + 1 - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
    if len(raw) > cap:
        raise ValueError("Encoded asset byte cap")
    return bytes(raw)


@contextmanager
def _image_input(path: Path, files: "RetainedFiles | None") -> Iterator[Path | BytesIO]:
    if files is None:
        yield path
    else:
        with BytesIO(_read_asset(path, files)) as memory:
            yield memory


def _image_cap(image: Image.Image) -> None:
    if image.width <= 0 or image.height <= 0 or image.width * image.height > _PIXEL_CAP:
        raise ValueError("Decoded PNG pixel cap")


AssetKind = Literal["brand", "font", "sfx"]
_ASSET_ID = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass(frozen=True, slots=True)
class RenderAssetRegistry:
    brand_images: Mapping[str, Path]
    fonts: Mapping[str, Path]
    sfx: Mapping[str, Path]

    def __post_init__(self) -> None:
        for name in ("brand_images", "fonts", "sfx"):
            supplied: Mapping[str, Path] = getattr(self, name)
            detached: dict[str, Path] = {}
            for asset_id, path in supplied.items():
                if not isinstance(asset_id, str) or not _ASSET_ID.fullmatch(asset_id):
                    raise ValueError("Registry keys require a safe asset ID, not a path or URI")
                detached[asset_id] = Path(path)
            object.__setattr__(self, name, MappingProxyType(detached))

    def resolve(self, kind: AssetKind, asset_id: str) -> Path:
        if not _ASSET_ID.fullmatch(asset_id):
            raise ValueError(f"{kind} requires an asset ID, not a path or URI")
        table = {"brand": self.brand_images, "font": self.fonts, "sfx": self.sfx}[kind]
        if asset_id not in table:
            raise ValueError(f"Unknown {kind} asset ID: {asset_id}")
        return table[asset_id]


def default_render_assets() -> RenderAssetRegistry:
    from .sfx_authority import approved_sfx_paths
    return RenderAssetRegistry(
        brand_images={"kick_brand": paths.base_dir / "assets" / "branding" / "KickAthanW.png"},
        fonts={"headline_impact": Path("C:/Windows/Fonts/impact.ttf"),
               "subtitle_bold": Path("C:/Windows/Fonts/arialbd.ttf")},
        sfx=approved_sfx_paths(),
    )


def validate_asset_loadability(kind: AssetKind, path: Path, *, _files: "RetainedFiles | None" = None) -> None:
    """Decode registered images/fonts without transforming or writing them.

    SFX codec loadability requires FFprobe and is checked by compiler preflight.
    """
    if not (path.is_file() if _files is None else stat.S_ISREG(_files.stat_path(path).st_mode)):
        raise ValueError(f"{kind} asset missing: {path}")
    try:
        if kind == "brand":
            with _image_input(path, _files) as image_input, Image.open(image_input) as image:
                if _files is not None:
                    _image_cap(image)
                image.load()
                if image.format != "PNG":
                    raise ValueError(f"brand asset must be PNG: {path}")
        elif kind == "font":
            if _files is None:
                ImageFont.truetype(str(path), 16)
            else:
                with BytesIO(_read_asset(path, _files)) as memory:
                    ImageFont.truetype(memory, 16, index=0)
        elif (path.stat() if _files is None else _files.stat_path(path)).st_size == 0:
            raise ValueError(f"sfx asset is empty: {path}")
    except (OSError, ValueError) as exc:
        raise ValueError(f"{kind} asset is not loadable: {path}: {exc}") from exc


def publish_render_bytes(data: bytes, *, output_dir: Path, suffix: Literal[".png", ".ass"],
                         _files: "RetainedFiles | None" = None) -> Path:
    """Publish only inside a host-supplied directory, with a content-derived name."""
    if suffix not in (".png", ".ass"):
        raise ValueError("Unsupported render artifact suffix")
    path = Path(output_dir) / (hashlib.sha256(data).hexdigest() + suffix)
    _atomic_write(path, data, **({} if _files is None else {"_files": _files}))
    return path


def _atomic_write(path: Path, data: bytes, *, _files: "RetainedFiles | None" = None) -> None:
    if _files is not None:
        if type(data) is not bytes or len(data) > _ASSET_CAP:
            raise ValueError("Encoded asset byte cap")
        _files.register(path, role="asset", owned=True, max_bytes=_ASSET_CAP)
        owned_temporary = path.with_suffix(path.suffix + ".tmp")
        _files.register(owned_temporary, role="temporary", owned=True, max_bytes=_ASSET_CAP)
        try:
            _files.path_action("stat", path)
        except FileNotFoundError:
            pass
        else:
            if _read_asset(path, _files) == data:
                return
            raise ValueError("Existing generated asset differs")
        _files.register(path.parent, role="directory", owned=True, max_bytes=0)
        try:
            _files.path_action("stat", path.parent)
        except FileNotFoundError:
            _files.path_action("mkdir", path.parent)
        with _files.open_file(owned_temporary, "xb") as file:
            for offset in range(0, len(data), 65536):
                view = memoryview(data)[offset:offset + 65536]
                if file.write(view) != len(view):
                    raise OSError("Incomplete render asset write")
            file.flush()
            file.fsync()
        _files.path_action("replace", owned_temporary, destination=path)
        return
    if path.is_file() and path.read_bytes() == data:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".render-", delete=False) as file:
            temporary = Path(file.name)
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


@dataclass(frozen=True, slots=True)
class PreparedBrandAsset:
    source_path: Path
    path: Path
    source_sha256: str
    source_size: tuple[int, int]
    source_bbox: tuple[int, int, int, int]
    size: tuple[int, int]

    def __post_init__(self) -> None:
        # Frozen annotations alone do not detach accepted list callers.
        for name in ("source_size", "source_bbox", "size"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, "source_path", Path(self.source_path))
        object.__setattr__(self, "path", Path(self.path))
        if (len(self.source_size) != 2 or len(self.size) != 2 or len(self.source_bbox) != 4
                or any(type(n) is not int for n in (*self.source_size, *self.size, *self.source_bbox))
                or min(*self.source_size, *self.size) <= 0
                or not re.fullmatch(r"[0-9a-f]{64}", self.source_sha256)):
            raise ValueError("Invalid prepared brand dimensions/identity")
        left, top, right, bottom = self.source_bbox
        if (not 0 <= left < right <= self.source_size[0]
                or not 0 <= top < bottom <= self.source_size[1]
                or self.size != (right - left, bottom - top)):
            raise ValueError("Invalid prepared brand bounding box")


def prepare_brand_asset(registry: RenderAssetRegistry, asset_id: str, *, cache_dir: Path,
                        _files: "RetainedFiles | None" = None) -> PreparedBrandAsset:
    """Trim alpha padding once per source-byte identity; never rewrite the source.

    The derived PNG carries its immutable crop metadata. Cache hits read/validate
    that PNG without decoding/cropping the original again. Files are atomically
    published; neither asset IDs nor model text become a path.
    """
    source = registry.resolve("brand", asset_id)
    try:
        data = source.read_bytes() if _files is None else _read_asset(source, _files)
        identity = hashlib.sha256(data).hexdigest()
        path = Path(cache_dir) / f"brand-v1-{identity}.png"
        if (path.resolve() == source.resolve() if _files is None else path == source):
            raise ValueError("Derived brand cannot overwrite its source")
        if _files is not None:
            _files.register(path, role="asset", owned=True, max_bytes=_ASSET_CAP)
            try:
                _files.path_action("stat", path)
                exists = True
            except FileNotFoundError:
                exists = False
        else:
            exists = path.exists()
        if exists:
            with _image_input(path, _files) as image_input, Image.open(image_input) as cached:
                if _files is not None:
                    _image_cap(cached)
                cached.load()
                metadata = json.loads(cached.info["clipperx_brand_v1"])
                if (cached.format != "PNG" or cached.mode != "RGBA"
                        or metadata["source_sha256"] != identity
                        or metadata["pixels_sha256"] != hashlib.sha256(cached.tobytes()).hexdigest()
                        or cached.getchannel("A").getbbox() != (0, 0, *cached.size)):
                    raise ValueError("Corrupt derived brand cache")
                return PreparedBrandAsset(source, path, identity,
                    tuple(metadata["source_size"]), tuple(metadata["source_bbox"]), cached.size)
        with BytesIO(data) as original_bytes, Image.open(original_bytes) as original:
            if _files is not None:
                _image_cap(original)
            original.load()
            if original.format != "PNG":
                raise ValueError("brand asset must be PNG")
            rgba = original.convert("RGBA")
            bbox = rgba.getchannel("A").getbbox()
            if bbox is None:
                raise ValueError("brand asset is all-transparent")
            cropped = rgba.crop(bbox)
            metadata = {"source_sha256": identity, "source_size": rgba.size,
                        "source_bbox": bbox,
                        "pixels_sha256": hashlib.sha256(cropped.tobytes()).hexdigest()}
            info = PngImagePlugin.PngInfo()
            info.add_text("clipperx_brand_v1", json.dumps(metadata, sort_keys=True))
            with BytesIO() as buffer:
                cropped.save(buffer, format="PNG", pnginfo=info)
                _atomic_write(path, buffer.getvalue(), **({} if _files is None else {"_files": _files}))
            return PreparedBrandAsset(source, path, identity, rgba.size, bbox, cropped.size)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise ValueError(f"brand asset/cache is not loadable: {source}: {exc}") from exc


@dataclass(frozen=True, slots=True)
class BrandVisualLayer:
    asset: PreparedBrandAsset
    destination: PixelRect
    placement: PixelRect
    start: float
    end: float
    z_index: int
    shot_id: str


def place_brand_asset(asset: PreparedBrandAsset, *, destination: PixelRect,
                      start: float, end: float, z_index: int, shot_id: str) -> BrandVisualLayer:
    """Contain the trimmed image with at most half-pixel resize rounding per edge."""
    if (not math.isfinite(start) or not math.isfinite(end) or not 0 <= start < end
            or type(z_index) is not int or z_index < 0):
        raise ValueError("Invalid brand layer timing/z_index")
    ratio = min(destination.width / asset.size[0], destination.height / asset.size[1])
    width = max(1, min(destination.width, math.floor(asset.size[0] * ratio + .5)))
    height = max(1, min(destination.height, math.floor(asset.size[1] * ratio + .5)))
    placement = PixelRect(destination.x + (destination.width - width) // 2,
                          destination.y + (destination.height - height) // 2, width, height)
    return BrandVisualLayer(asset, destination, placement, start, end, z_index, shot_id)


def compile_brand_layer(layer: BrandVisualLayer, *, input_label: str, brand_label: str,
                        output_label: str) -> str:
    """Assembly hook: host supplies one finite prepared PNG on the global output clock.

    The cached last frame covers the authored enable interval while the finite
    primary ends normally. Contains no file path or untrusted identity. Compose
    in z_index order with other visual layers; this helper does not execute/render a clip.
    """
    for label in (input_label, brand_label, output_label):
        if not re.fullmatch(r"[A-Za-z0-9_]+(?::[A-Za-z0-9_]+)*", label):
            raise ValueError("Invalid filter label")
    if len({input_label, brand_label, output_label}) != 3:
        raise ValueError("Filter labels must differ")
    p = layer.placement
    scaled = output_label.replace(":", "_") + "_brand"
    while scaled in (input_label, brand_label, output_label):
        scaled += "_"
    return (f"[{brand_label}]scale=w={p.width}:h={p.height},setsar=1,format=rgba[{scaled}];"
            f"[{input_label}][{scaled}]overlay=x={p.x}:y={p.y}:format=auto:"
            f"enable='gte(t,{layer.start!r})*lt(t,{layer.end!r})':eof_action=repeat:repeatlast=1[{output_label}]")
