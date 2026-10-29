"""Exact selected float buses and one prepared tail."""

import importlib.util
from fractions import Fraction

import pytest

from clipperx.tests.test_visual_compiler import pure_plan


def test_audio_executor_exists() -> None:
    assert importlib.util.find_spec("clipperx._visual_render_audio") is not None


@pytest.mark.parametrize("enabled", ["all", "mixed", "none"])
def test_enabled_intersections_use_schedule_sample_order(enabled: str) -> None:
    from clipperx._visual_render_audio import enabled_slices
    from clipperx.audio_mix import schedule_audio
    from clipperx.audio_selection import AudioSelectionGroup
    plan = pure_plan(("solid", "solid"), duration=.1)
    for index, shot in enumerate(plan.shots):
        shot.audio.normalize_dialogue = enabled == "all" or enabled == "mixed" and index == 1
    schedule = schedule_audio(plan)
    group = AudioSelectionGroup(index=0, shot_first=0, shot_last=2, schedule_first=0,
        schedule_last=len(schedule.spans), source_start=Fraction(0), source_end=Fraction(1, 5),
        start=Fraction(0), end=Fraction(1, 5), first=0, last=9600)
    slices = enabled_slices(schedule, group, 4)
    assert sum(end-start for start, end in slices) == schedule.enabled_samples*4
    if enabled == "mixed":
        assert slices == ((19200, 38400),)
