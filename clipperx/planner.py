"""Prepare deterministic planner inputs without invoking a model."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .context_budget import ContextBudget, TiktokenCounter, build_transcript_batches
from .planner_contracts import DiscoveryBatchResponse
from .planner_prompts import DISCOVERY_SYSTEM_PROMPT
from .schema import TranscriptSegment


def prepare_transcript_batches(
    transcript_path: Path,
    output_dir: Path,
    *,
    model: str,
    context_window_tokens: int,
    reserved_output_tokens: int | None = None,
    utilization: float = 0.85,
    overlap_tokens: int = 1000,
) -> Path:
    raw_segments = json.loads(transcript_path.read_text(encoding="utf-8"))
    segments = [TranscriptSegment.from_dict(item) for item in raw_segments]
    counter = TiktokenCounter(model)
    prompt_scaffold = "\n".join(
        [
            DISCOVERY_SYSTEM_PROMPT,
            "CONTRACT_JSON_SCHEMA:",
            json.dumps(DiscoveryBatchResponse.model_json_schema(), ensure_ascii=False),
            "BATCH_METADATA: batch_id, cursor_before, cursor_after",
            "TRANSCRIPT:",
        ]
    )
    prompt_tokens = counter.count(prompt_scaffold)
    effective_reserved_output = reserved_output_tokens or min(
        4096,
        max(1024, context_window_tokens // 4),
    )
    budget = ContextBudget(
        context_window_tokens=context_window_tokens,
        prompt_tokens=prompt_tokens,
        reserved_output_tokens=effective_reserved_output,
        utilization=utilization,
        overlap_tokens=overlap_tokens,
    )
    effective_overlap = min(overlap_tokens, max(0, budget.input_token_budget // 3))
    if effective_overlap != overlap_tokens:
        budget = ContextBudget(
            context_window_tokens=context_window_tokens,
            prompt_tokens=prompt_tokens,
            reserved_output_tokens=effective_reserved_output,
            utilization=utilization,
            overlap_tokens=effective_overlap,
        )
    batches = build_transcript_batches(segments, counter=counter, budget=budget)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_records: list[dict[str, Any]] = []
    for index, batch in enumerate(batches):
        batch_path = output_dir / f"batch_{index:05d}_{batch.batch_id}.json"
        batch_payload = asdict(batch)
        batch_path.write_text(json.dumps(batch_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        record = batch.metadata()
        record["path"] = str(batch_path)
        batch_records.append(record)
    manifest = {
        "version": 1,
        "model": model,
        "tokenizer_encoding": counter.encoding_name,
        "exact_model_mapping": counter.exact_model_mapping,
        "context_budget": asdict(budget),
        "requested_reserved_output_tokens": reserved_output_tokens,
        "requested_overlap_tokens": overlap_tokens,
        "input_token_budget": budget.input_token_budget,
        "counting_note": (
            "Incluye system prompt, schema y scaffold. El safety_tokens absorbe overhead del protocolo."
        ),
        "transcript_segments": len(segments),
        "batches": batch_records,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path
