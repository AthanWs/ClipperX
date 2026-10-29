"""HOST-authored timeline fixtures; no model or media inference."""

import pytest
from fractions import Fraction as F

from clipperx.output_timeline import OutputTimeline, join_words
from clipperx.schema import TranscriptSegment, Word
from clipperx.visual_contracts import VisualCompositionPlan, VisualShot


def shot(source: tuple[float, float], output: tuple[float, float],
         shot_id: str = "a") -> VisualShot:
    return VisualShot.model_validate({
        "shot_id": shot_id, "source": dict(zip(("start", "end"), source)),
        "output": dict(zip(("start", "end"), output)), "primary_source": "content",
        "reason": "HOST fixture", "evidence_ids": ["host"], "confidence": 1,
        "layers": [{"layer_id": "source", "kind": "source_video", "source_ref": "source",
                    "destination": {"x": 0, "y": 0, "width": 1, "height": 1},
                    "fill": "cover", "z_index": 0}],
    })


def test_non_contiguous_source_words_map_to_contiguous_output() -> None:
    mapper = OutputTimeline.from_shots([shot((10, 12), (0, 2)),
                                        shot((20, 22), (2, 4), "b")])
    assert mapper.source_to_output(20.5) == pytest.approx(2.5)
    assert mapper.source_to_output(12) is None  # half-open source windows


def test_repeated_and_out_of_order_ranges_keep_every_occurrence() -> None:
    mapper = OutputTimeline.from_shots([shot((20, 22), (0, 2)),
        shot((10, 12), (2, 4), "b"), shot((20, 22), (4, 6), "c")])
    assert mapper.source_to_outputs(20.5) == pytest.approx((.5, 4.5))
    with pytest.raises(ValueError, match="ambiguous"):
        mapper.source_to_output(20.5)
    assert mapper.source_to_output(20.5, shot_id="c") == pytest.approx(4.5)
    words = [Word(20.25, 20.75, "hola")]
    events = mapper.remap([TranscriptSegment(20, 21, "hola", words)])
    assert [(e.shot_id, e.start, e.end, e.text) for e in events] == [
        ("a", .25, .75, "hola"), ("c", 4.25, 4.75, "hola")]


def test_words_clip_at_fractional_cuts_and_split_by_shot() -> None:
    mapper = OutputTimeline.from_shots([shot((10.125, 11.375), (0, 1.25)),
                                      shot((11.375, 12.125), (1.25, 2), "b")])
    events = mapper.remap([TranscriptSegment(10, 13, "uno dos tres fuera", [
        Word(10, 10.5, "uno"), Word(11.25, 11.5, "dos"), Word(12, 12.5, "tres"),
        Word(12.5, 13, "fuera")])])
    assert [e.text for e in events] == ["uno dos", "dos tres"]
    assert [(w.start, w.end) for e in events for w in e.words] == pytest.approx([
        (0, .375), (1.125, 1.25), (1.25, 1.375), (1.875, 2)])


def test_segment_only_text_is_clipped_without_inventing_word_times() -> None:
    mapper = OutputTimeline.from_shots([shot((10, 12), (0, 2))])
    event, = mapper.remap([TranscriptSegment(9, 11, "known text")])
    assert (event.start, event.end, event.text, event.words) == (0, 1, "known text", ())
    assert mapper.remap([]) == ()
    with pytest.raises(ValueError, match="Missing"):
        mapper.remap(None)


def test_results_detach_caller_shots_words_and_containers() -> None:
    shots = [shot((10, 12), (0, 2))]
    timeline = OutputTimeline.from_shots(shots)
    words = [Word(10, 11, "hello")]
    transcript = [TranscriptSegment(10, 12, "hello", words)]
    events = timeline.remap(transcript)
    words[0].word = "changed"
    words.clear()
    shots[0].source.start = 0
    shots.clear()
    assert timeline.source_to_output(10.5) == .5
    assert events[0].words[0].text == "hello"
    supplied = list(timeline.spans)
    detached = OutputTimeline(supplied)  # type: ignore[arg-type]
    supplied.clear()
    assert detached.spans == timeline.spans


@pytest.mark.parametrize("source,output", [((10, 12), (1, 3)), ((10, 12), (0, 1))])
def test_invalid_timeline_is_rejected(source: tuple[float, float],
                                      output: tuple[float, float]) -> None:
    with pytest.raises(ValueError):
        OutputTimeline.from_shots([shot(source, output)])


def test_punctuation_spacing_is_local_to_v2() -> None:
    assert join_words(["¿", "Hola", ",", "mundo", "?", "¡", "Sí", "!"]) == (
        "¿Hola, mundo? ¡Sí!")


def test_invalid_word_timing_is_rejected() -> None:
    timeline = OutputTimeline.from_shots([shot((10, 12), (0, 2))])
    with pytest.raises(ValueError, match="word"):
        timeline.remap([TranscriptSegment(10, 12, "bad", [Word(9, 11, "bad")])])


def test_native_point_words_keep_exact_acoustic_times_in_every_occurrence() -> None:
    timeline = OutputTimeline.from_shots([shot((10, 12), (0, 2)),
                                         shot((10, 12), (2, 4), "b")])
    transcript = [TranscriptSegment(10, 12, "uno dos", [
        Word(10.125, 10.125, "uno"), Word(10.25, 10.5, "dos")])]
    events = timeline.remap(transcript)
    assert [e.text for e in events] == ["uno dos", "uno dos"]
    assert [e.words[0].exact_interval for e in events] == [(F(1, 8), F(1, 8)),
                                                         (F(17, 8), F(17, 8))]
    assert [(w.start, w.end) for e in events for w in e.words] == [
        (.125, .125), (.25, .5), (2.125, 2.125), (2.25, 2.5)]
    assert transcript[0].words[0].start == transcript[0].words[0].end == 10.125


def test_point_words_at_a_cut_belong_only_to_the_right_hand_source_occurrence() -> None:
    timeline = OutputTimeline.from_shots([shot((10, 11), (0, 1)),
                                         shot((11, 12), (1, 2), "b")])
    event, = timeline.remap([TranscriptSegment(10, 11, "fin", [Word(11, 11, "fin")])])
    assert event.shot_id == "b" and event.text == "fin"
    assert event.words[0].exact_interval == (F(1), F(1))


def test_native_point_word_metadata_is_isolated_from_positive_event_intervals() -> None:
    from clipperx.output_timeline import MappedEvent, MappedWord

    point = [F(1, 8), F(1, 8)]
    word = MappedWord(.125, .125, "uno", point)  # type: ignore[arg-type]
    point.clear()
    assert word.exact_interval == (F(1, 8), F(1, 8))
    with pytest.raises(ValueError):
        MappedEvent("a", .125, .125, "uno", (word,), word.exact_interval)


def test_point_word_metadata_cannot_replace_a_positive_or_different_acoustic_projection() -> None:
    from clipperx.output_timeline import MappedWord

    for start, end in [(0, 1), (0, 0)]:
        with pytest.raises(ValueError, match="point"):
            MappedWord(start, end, "uno", (F(1, 8), F(1, 8)))


@pytest.mark.parametrize("mismatch", ["duration", "continuity"])
def test_schema_valid_submillisecond_mismatch_is_explicitly_unsupported(mismatch: str) -> None:
    if mismatch == "duration":
        shots = [shot((10, 12), (0, 2.0005))]
    else:
        shots = [shot((10, 12), (0, 2)), shot((20, 22), (2.0005, 4.0005), "b")]
    # The unchanged v2 schema permits 1ms differences. Compilation supports the
    # stricter 1e-9 exact-anchor subset and must never rescale/snap these endpoints.
    schema_valid = VisualCompositionPlan.model_validate({
        "clip_id": "host", "source_fingerprint": "host", "composition_fingerprint": "host",
        "source_timeline_fingerprint": "host", "identity_fingerprint": "host",
        "style_library_fingerprint": "host", "shots": [s.model_dump() for s in shots]})
    before = schema_valid.model_dump_json()
    with pytest.raises(ValueError, match="durations must match exactly|continuous output"):
        OutputTimeline.from_shots(schema_valid.shots)
    assert schema_valid.model_dump_json() == before


def test_raster_allocates_absolute_boundaries_not_each_duration() -> None:
    shots = [shot((float(F(i, 40)), float(F(i+1, 40))),
                  (float(F(i, 40)), float(F(i+1, 40))), f"s{i}")
             for i in range(40)]
    spans = OutputTimeline.from_shots(shots).raster_spans()
    assert [s.frame_count for s in spans] == [2, 1] * 20
    assert sum(s.frame_count for s in spans) == spans[-1].k_end == 60
    assert all(a.k_end == b.k_start for a, b in zip(spans, spans[1:]))


@pytest.mark.parametrize("residue", range(120))
def test_raster_halfway_ties_and_all_residues(residue: int) -> None:
    end = 1 + F(residue, 120)
    span, = OutputTimeline.from_shots([shot((0, float(end)), (0, float(end)))]).raster_spans()
    # Decimal validated operands are the authority; integer floor is independent.
    exact_end = F(str(float(end)))
    assert span.k_end == (60 * exact_end + F(1, 2)).numerator // (
        60 * exact_end + F(1, 2)).denominator
    assert span.start_delta == F(0)
    assert span.end_delta == F(span.k_end, 60) - exact_end
    assert span.duration_delta == F(span.frame_count, 60) - exact_end
    assert isinstance(span.start_delta, F) and isinstance(span.duration_delta, F)


def test_global_requests_keep_occurrence_and_crop_local_n() -> None:
    timeline = OutputTimeline.from_shots([shot((20, 22), (0, 2)),
        shot((10, 12), (2, 4), "b"), shot((20, 22), (4, 6), "c")])
    for k in (0, 119, 120, 239, 240, 359):
        request = timeline.video_request(k)
        index = k // 120
        assert (request.global_k, request.occurrence_index, request.local_n) == (k, index, k % 120)
        assert request.shot_id == ("a", "b", "c")[index]
        assert request.requested_source == F((20, 10, 20)[index]) + F(k % 120, 60)
        assert request.request_rule == "inverse_declared_mapper_at_global_pts"


def test_leading_negative_phase_one_cell_requests_declared_start() -> None:
    timeline = OutputTimeline.from_shots([shot((0, .020), (0, .020)),
                                        shot((2, 2.006), (.020, .026), "b")])
    request = timeline.video_request(1)
    assert (request.requested_source, request.local_n, request.occurrence_index) == (F(2), 0, 1)
    assert request.request_rule == "declared_start_for_leading_raster_cell"
    span = timeline.raster_spans()[1]
    assert (span.source_start, span.source_end, span.output_start, span.output_end) == (
        F(2), F(1003, 500), F(1, 50), F(13, 500))
    assert (span.start_delta, span.end_delta, span.duration_delta) == (
        -F(1, 300), F(11, 1500), F(4, 375))


def test_selected_global_rule_observes_independent_persistent_flash_interval() -> None:
    timeline = OutputTimeline.from_shots([shot((0, .025), (0, .025)),
                                        shot((2.007, 2.107), (.025, .125), "b")])
    request = timeline.video_request(2)
    flash_start, flash_end = F(2007, 1000) + F(1, 600), F(2107, 1000)
    assert flash_start <= request.requested_source < flash_end
    assert not flash_start <= F(2007, 1000) < flash_end  # rejected local candidate at n=0
    output_transition = F(1, 40) + F(1, 600)
    assert F(3, 60) - output_transition == F(7, 300)  # candidate L
    assert F(2, 60) - output_transition == F(1, 150)  # selected G


def test_zero_cell_and_exact_raster_discontinuity_reject_without_repair() -> None:
    timeline = OutputTimeline.from_shots([shot((0, .001), (0, .001))])
    with pytest.raises(ValueError, match="a.*0|0.*a"):
        timeline.raster_spans()
    timeline = OutputTimeline.from_shots([shot((0, .025), (0, .025)),
        shot((2, 2.1), (.0249999999, .1249999999), "b")])
    with pytest.raises(ValueError, match="raster.*continu|continu.*raster"):
        timeline.raster_spans()


def test_requests_check_exact_source_membership_despite_duration_tolerance() -> None:
    timeline = OutputTimeline.from_shots([shot((0, .9999999999), (0, 1))])
    assert timeline.video_request(59).requested_source == F(59, 60)
    assert timeline.raster_spans()[0].source_end == F(9999999999, 10**10)


@pytest.mark.parametrize("value", [True, False, -1, 60, 1.0, "1", None])
def test_video_requests_reject_invalid_global_indices(value: object) -> None:
    timeline = OutputTimeline.from_shots([shot((0, 1), (0, 1))])
    with pytest.raises(ValueError):
        timeline.video_request(value)  # type: ignore[arg-type]


def test_raster_windows_require_exact_contained_positive_cells() -> None:
    timeline = OutputTimeline.from_shots([shot((0, 1), (0, 1))])
    assert timeline.raster_window(F(1, 40), F(1, 20), shot_id="a") == (2, 3)
    assert timeline.raster_window(F(1, 120), F(1, 120) + F(1, 60), shot_id="a") == (1, 2)
    for start, end, identity in [(F(-1), F(1), "a"), (F(0), F(2), "a"),
                                (F(1), F(0), "a"), (F(0), F(1, 1000), "a"),
                                (F(0), F(1), "missing"), (0.0, F(1), "a")]:
        with pytest.raises(ValueError):
            timeline.raster_window(start, end, shot_id=identity)  # type: ignore[arg-type]


def test_exact_remap_interval_survives_its_lossy_float_projection() -> None:
    timeline = OutputTimeline.from_shots([
        shot((1, 1.15), (0, .15), "a"),
        shot((1e-18, .2), (.15, .35), "b"),
    ])
    mapped_b, = timeline.remap([
        TranscriptSegment(.075, .1, "x", [Word(.075, .1, "x")])
    ])
    expected = F(9, 40) - F(1, 10**18)
    interval = mapped_b.exact_interval
    assert interval is not None and interval == (expected, F(1, 4) - F(1, 10**18))
    assert mapped_b.words[0].exact_interval == interval
    assert mapped_b.start == float(expected)
    assert timeline.source_to_output(.075, shot_id="b") == float(expected)
    assert timeline.raster_window(*interval, shot_id="b")[0] == 13
    title_interval = timeline.overlay_output_interval("b", .075, .1)
    assert title_interval == (F(9, 40), F(1, 4))
    assert timeline.raster_window(*title_interval, shot_id="b") == (14, 15)


def test_exact_remap_clips_original_operands_for_repeated_out_of_order_occurrences() -> None:
    timeline = OutputTimeline.from_shots([shot((.075, .15), (0, .075)),
        shot((0, .075), (.075, .15), "b"), shot((.075, .15), (.15, .225), "c")])
    events = timeline.remap([TranscriptSegment(0, .2, "known", [
        Word(0, .1, "hola"), Word(.1, .2, "!")])])
    assert [(e.shot_id, e.exact_interval, e.text) for e in events] == [
        ("a", (F(0), F(3, 40)), "hola!"), ("b", (F(3, 40), F(3, 20)), "hola"),
        ("c", (F(3, 20), F(9, 40)), "hola!")]
    assert [w.exact_interval for w in events[0].words] == [(F(0), F(1, 40)),
                                                                (F(1, 40), F(3, 40))]
    plain, = OutputTimeline.from_shots([shot((.075, .15), (0, .075))]).remap([
        TranscriptSegment(0, .2, "known")])
    assert plain.exact_interval == (F(0), F(3, 40)) and plain.text == "known"


def test_exact_mapped_metadata_detaches_and_keeps_old_default_none() -> None:
    from dataclasses import FrozenInstanceError
    from clipperx.output_timeline import MappedEvent, MappedWord

    supplied = [F(0), F(1)]
    word = MappedWord(0, 1, "x", supplied)  # type: ignore[arg-type]
    event = MappedEvent("a", 0, 1, "x", [word], supplied)  # type: ignore[arg-type]
    supplied.clear()
    assert word.exact_interval == event.exact_interval == (F(0), F(1))
    assert MappedWord(0, 1, "x").exact_interval is None
    assert MappedEvent("a", 0, 1, "x").exact_interval is None
    with pytest.raises(FrozenInstanceError):
        word.exact_interval = None  # type: ignore[misc]
    with pytest.raises(ValueError):
        MappedEvent("a", 0, 0, "x", exact_interval=(F(0), F(0)))
    for interval in [(F(0), F(0)), (F(1), F(0)), (F(-1), F(1)),
                     (0.0, 1.0), (False, F(1)), (F(0),), "bad"]:
        with pytest.raises(ValueError):
            MappedWord(0, 1, "x", interval)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            MappedEvent("a", 0, 1, "x", exact_interval=interval)  # type: ignore[arg-type]


@pytest.mark.parametrize("start,end", [(True, 1), (0, False), (float("nan"), 1),
                                       (0, float("inf")), (-1, 1), (1, 0), (0, 2)])
def test_overlay_exact_interval_rejects_invalid_local_metadata(start: float, end: float) -> None:
    timeline = OutputTimeline.from_shots([shot((0, 1), (0, 1))])
    with pytest.raises(ValueError):
        timeline.overlay_output_interval("a", start, end)


def test_overlay_exact_interval_rejects_unknown_shot() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        OutputTimeline.from_shots([shot((0, 1), (0, 1))]).overlay_output_interval("missing", 0, 1)


@pytest.mark.parametrize("invalid", [True, float("nan"), float("inf"), -1, "1", None])
def test_source_and_transcript_reject_invalid_scalar_metadata(invalid: object) -> None:
    timeline = OutputTimeline.from_shots([shot((0, 1), (0, 1))])
    with pytest.raises(ValueError):
        timeline.source_to_outputs(invalid)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        timeline.remap([TranscriptSegment(invalid, 1, "x")])  # type: ignore[arg-type]


def test_raster_records_validate_exact_indices_identity_and_immutability() -> None:
    from dataclasses import FrozenInstanceError, replace
    from clipperx.output_timeline import TimelineSpan

    timeline = OutputTimeline.from_shots([shot((0, 1), (0, 1))])
    raster, = timeline.raster_spans()
    request = timeline.video_request(0)
    for changes in ({"occurrence_index": True}, {"k_start": False}, {"k_end": 1.0},
                    {"output_end": 1.0}, {"source_start": F(-1)}, {"shot_id": ""}):
        with pytest.raises(ValueError):
            replace(raster, **changes)
    for request_changes in ({"global_k": True}, {"local_n": 1.0}, {"occurrence_index": -1},
                    {"requested_source": 0.0}, {"request_rule": "local"}, {"shot_id": ""}):
        with pytest.raises(ValueError):
            replace(request, **request_changes)
    with pytest.raises(FrozenInstanceError):
        request.local_n = 3  # type: ignore[misc]
    with pytest.raises(ValueError):
        TimelineSpan("a", False, 1, 0, 1)
