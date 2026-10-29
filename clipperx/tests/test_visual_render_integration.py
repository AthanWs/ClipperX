"""Explicit opt-in only. Default collection never creates guards, media or native work."""

from typing import Any

import pytest

from clipperx.tests.test_native_render_qualification import no_native as no_native


def test_default_cli_reports_actual_not_run_without_io(monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import render_visual_smoke as smoke
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("default mode performed setup IO")
    monkeypatch.setattr(smoke, "load_manifest", forbidden)
    assert smoke.main([]) == 0


def test_public_boundary_uses_literal_default_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path
    from clipperx import plan_render
    from clipperx.render_assets import RenderAssetRegistry
    from scripts import render_visual_smoke as smoke
    seen: list[Any] = []
    def boundary(*args: Any, **kwargs: Any) -> list[Path]:
        seen.append((args, kwargs))
        return [Path("out/clip.mp4")]
    monkeypatch.setattr(plan_render, "render_visual_composition_plan", boundary)
    assets = RenderAssetRegistry({}, {}, {})
    assert smoke.public_render(Path("source.mov"), Path("plan.json"), Path("out"),
                               assets=assets, transcript=None) == [Path("out/clip.mp4")]
    args, kwargs = seen[0]
    assert args == (Path("source.mov"), Path("plan.json"), Path("out"))
    assert kwargs["assets"] is assets and kwargs["transcript_path"] is None
    assert kwargs["profile"].video_encoder.value == "h264_nvenc"
    assert set(kwargs) == {"assets", "transcript_path", "profile"}


@pytest.mark.skip(reason="actual NOT RUN: invoke guarded script explicitly after Root source review")
@pytest.mark.parametrize("stage", ("prerequisites", "synthetic", "gameplay", "camera", "transition"))
def test_actual_qualification_disabled(stage: str) -> None:
    raise AssertionError(f"Use manifest/guard CLI for actual stage {stage}")
