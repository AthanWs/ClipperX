"""Admission for SFX through the default CLI, reusing director READY authority.

Explicit custom renderer registries remain a host API. This gate protects the
newly enabled default CLI; it does not replace native retention/preflight.
"""
from pathlib import Path

from .visual_compiler import _bounded_json_metadata
from .visual_contracts import VisualCompositionPlan


def validate_default_sfx_render_plan(source_video: Path, plan_path: Path) -> None:
    """Manual no-cue plans remain valid; cues require their current READY receipt."""
    # Match native HOST admission, including size, depth and duplicate-key guards.
    with plan_path.open("rb") as stream:
        raw = stream.read(8_000_001)
    payload = _bounded_json_metadata(raw.decode("utf-8"), cap=8_000_000, max_depth=24)
    if type(payload) is not dict or type(payload.get("contract_version")) is not int or payload["contract_version"] != 2:
        raise ValueError("render-visual-plan requires explicit contract_version 2")
    plan = VisualCompositionPlan.model_validate(payload)
    if not any(shot.audio.cues for shot in plan.shots):
        return
    from .visual_director import DirectorPublicationError, _load_terminal

    try:
        terminal = _load_terminal(plan_path.resolve().parent)
    except DirectorPublicationError as error:
        raise ValueError("SFX director READY authorization is invalid or no longer current") from error
    if (terminal is None or terminal[0].resolve() != plan_path.resolve()
            or terminal[1].get("status") != "ready"
            or Path(terminal[1]["source"]).resolve() != source_video.resolve()):
        raise ValueError("SFX requires the associated current director READY plan and source")
