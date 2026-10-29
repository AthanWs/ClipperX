import json
from pathlib import Path

import pytest

from clipperx.planner_contracts import DiscoveryBatchResponse
from clipperx.planning_state import PlanningLedger


def test_ledger_enforces_sequential_idempotent_cursor(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "batches": [
                    {"batch_id": "a", "cursor_before": 0, "cursor_after": 5, "path": "a.json"},
                    {"batch_id": "b", "cursor_before": 5, "cursor_after": 9, "path": "b.json"},
                ]
            }
        ),
        encoding="utf-8",
    )
    ledger = PlanningLedger.create(manifest, tmp_path / "ledger.json")
    first = DiscoveryBatchResponse(batch_id="a", cursor_before=0, cursor_after=5)
    second = DiscoveryBatchResponse(batch_id="b", cursor_before=5, cursor_after=9)

    with pytest.raises(ValueError):
        ledger.apply_discovery(second)
    assert ledger.apply_discovery(first)
    assert not ledger.apply_discovery(first)
    assert ledger.apply_discovery(second)
    assert ledger.is_complete()
