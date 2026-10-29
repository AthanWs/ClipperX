"""Idempotent cursor ledger for sequential planner execution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .planner_contracts import DiscoveryBatchResponse


class PlanningLedger:
    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    def create(cls, manifest_path: Path, ledger_path: Path) -> "PlanningLedger":
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        batches = {
            item["batch_id"]: {
                "cursor_before": item["cursor_before"],
                "cursor_after": item["cursor_after"],
                "status": "pending",
                "path": item["path"],
            }
            for item in manifest.get("batches", [])
        }
        state = {
            "version": 1,
            "manifest": str(manifest_path.resolve()),
            "next_cursor": 0,
            "batches": batches,
            "candidates": {},
            "responses": {},
        }
        ledger = cls(ledger_path)
        ledger._write(state)
        return ledger

    def load(self) -> dict[str, Any]:
        return json.loads(self.path.read_text(encoding="utf-8"))

    def apply_discovery(self, response: DiscoveryBatchResponse) -> bool:
        state = self.load()
        batch = state.get("batches", {}).get(response.batch_id)
        if batch is None:
            raise ValueError(f"batch_id desconocido: {response.batch_id}")
        response_payload = response.model_dump(mode="json")
        response_hash = _payload_hash(response_payload)
        previous_hash = state.get("responses", {}).get(response.batch_id)
        if batch["status"] == "complete":
            if previous_hash != response_hash:
                raise ValueError("El batch ya fue completado con una respuesta diferente")
            return False
        if response.cursor_before != batch["cursor_before"] or response.cursor_after != batch["cursor_after"]:
            raise ValueError("La respuesta no coincide con los cursores del batch")
        if response.cursor_before != state["next_cursor"]:
            raise ValueError(
                f"Respuesta fuera de secuencia: esperado cursor {state['next_cursor']}, "
                f"recibido {response.cursor_before}"
            )
        candidates = state.setdefault("candidates", {})
        for candidate in response.candidates:
            payload = candidate.model_dump(mode="json")
            existing = candidates.get(candidate.candidate_id)
            if existing is not None and existing != payload:
                raise ValueError(f"candidate_id conflictivo: {candidate.candidate_id}")
            candidates[candidate.candidate_id] = payload
        batch["status"] = "complete"
        state["responses"][response.batch_id] = response_hash
        state["next_cursor"] = response.cursor_after
        self._write(state)
        return True

    def is_complete(self) -> bool:
        state = self.load()
        return all(item["status"] == "complete" for item in state.get("batches", {}).values())

    def _write(self, state: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.path)


def _payload_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
