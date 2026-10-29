from __future__ import annotations

import inspect

from clipperx import timeline


def test_legacy_timeline_public_signature_and_payload_markers_remain_stable() -> None:
    signature = inspect.signature(timeline.build_timeline)
    source = inspect.getsource(timeline.build_timeline)

    assert list(signature.parameters) == [
        "video_path",
        "sample_interval",
        "include_audio",
        "force",
    ]
    assert '"kind": "coarse_timeline"' in source
    assert '"samples": samples' in source
    assert "return job.timeline_path" in source
