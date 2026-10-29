"""Core package for ClipperX automated clipping pipeline."""

from importlib.metadata import version, PackageNotFoundError


def get_version() -> str:
    """Return installed package version or development placeholder."""
    try:
        return version("clipperx")
    except PackageNotFoundError:  # pragma: no cover - only during dev
        return "0.1.0-dev"


__all__ = ["get_version"]

