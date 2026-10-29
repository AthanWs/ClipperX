"""The planner must carry the selected transcript into actual director custody."""
import hashlib
import json

import pytest

from clipperx import planner_engine as engine
from clipperx.tests.test_visual_director import FakeClient, ready
from clipperx.visual_contracts import SourceVisualSpan
from clipperx.visual_director import VisualDirectorLimits

pytest_plugins = ["clipperx.tests.test_visual_planner_integration"]


def prepare(inputs):
    inputs.timeline.spans = [SourceVisualSpan.model_validate(dict(span_id="source-span",
        source=dict(start=0, end=2), state="content_only", confidence=1,
        identity_status="unavailable", streamer_track_id=None, webcam_track_id=None,
        important_regions=[], text_regions=[], evidence_ids=["e-0"], change_reasons=[]))]
    inputs.timeline_path.write_text(inputs.timeline.model_dump_json(), encoding="utf-8")


def test_visual_planner_consumes_explicit_transcript_instead_of_missing_canonical(inputs):
    prepare(inputs)
    result = engine.run_visual_planner(inputs.video, inputs.plan,
        source_timeline_path=inputs.timeline_path, transcript_path=inputs.transcript,
        client=FakeClient([ready]), style_library=inputs.library, limits=VisualDirectorLimits(),
        jobs_root=inputs.root / "jobs")
    manifest = json.loads(result.visual_plan_manifest.read_text())
    entry, = manifest["clips"]
    assert entry["status"] == "ready"
    text = entry["host_context"]["text"]
    assert text["path"] == str(inputs.transcript.resolve())
    assert text["sha256"] == hashlib.sha256(inputs.transcript.read_bytes()).hexdigest()
    assert manifest["transcript_sha256"] == text["sha256"]


def test_visual_planner_rejects_transcript_changed_while_directing(inputs):
    prepare(inputs)
    def mutate(payload):
        response = ready(payload)
        inputs.transcript.write_text(inputs.transcript.read_text() + " ", encoding="utf-8")
        return response
    with pytest.raises(ValueError):
        engine.run_visual_planner(inputs.video, inputs.plan,
            source_timeline_path=inputs.timeline_path, transcript_path=inputs.transcript,
            client=FakeClient([mutate]), style_library=inputs.library, limits=VisualDirectorLimits(),
            jobs_root=inputs.root / "jobs")
