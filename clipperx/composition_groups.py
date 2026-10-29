"""Deterministic, context-bounded editorial groups and host-owned clip identities."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from .context_budget import TiktokenCounter
from .planner_contracts import GlobalCompositionResponse
from .structured_client import _strict_json_schema

# A request bound, never an output quota or a cap on total VOD coverage.
MAX_COMPOSITION_CANDIDATES = 12


def payload_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
        allow_nan=False).encode("utf-8")).hexdigest()


def connected_components(candidates: list[dict[str, Any]],
                         reviews: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Topic equality and review links both preserve whole narrative components."""
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("Duplicate candidate_id in coverage inputs")
    parents = {key: key for key in by_id}

    def find(key: str) -> str:
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def join(first: str, second: str) -> None:
        if first in parents and second in parents:
            parents[find(second)] = find(first)

    topics: dict[str, str] = {}
    for key in sorted(by_id):
        topic = by_id[key].get("topic_id")
        if topic:
            join(key, topics.setdefault(topic, key))
        for related in by_id[key].get("related_candidate_ids", []):
            join(key, related)
    for review in reviews:
        for related in review.get("related_candidate_ids", []):
            join(review["candidate_id"], related)
    components: dict[str, list[dict[str, Any]]] = {}
    for key in sorted(by_id):
        components.setdefault(find(key), []).append(by_id[key])
    return sorted(components.values(), key=lambda component: component[0]["candidate_id"])


def build_composition_groups(candidates: list[dict[str, Any]], reviews: list[dict[str, Any]],
        unresolved: list[dict[str, Any]], *, system_prompt: str,
        identity: dict[str, Any], run_limits: dict[str, Any], streamer_identity: Any,
        context_window_tokens: int, max_output_tokens: int, model: str,
        input_utilization: float = 0.85) -> list[dict[str, Any]]:
    """Pack independent components using the full actual request text and strict schema."""
    counter = TiktokenCounter(model)
    schema = _strict_json_schema(GlobalCompositionResponse)
    schema_text = json.dumps(schema, ensure_ascii=False)
    # Utilization already reserves 15% context; keep an additional message-envelope margin.
    input_limit = int(context_window_tokens * input_utilization) - max_output_tokens - 128
    known_ids = {candidate["candidate_id"] for candidate in candidates}

    def request(component: list[dict[str, Any]]) -> dict[str, Any]:
        ids = {candidate["candidate_id"] for candidate in component}
        payload = {"candidates": component,
            "boundary_reviews": [review for review in reviews if review["candidate_id"] in ids],
            "unresolved_evidence_due_to_limits": [item for item in unresolved
                if item["candidate_id"] in ids], "run_limits": run_limits,
            "streamer_identity": streamer_identity,
            "coverage_scope": "Grupo ofrecido; otros grupos se componen por separado"}
        user_prompt = json.dumps(payload, ensure_ascii=False)
        estimated = counter.count(f"{system_prompt}\n{user_prompt}\n{schema_text}")
        missing = sorted({related for item in [*component, *payload["boundary_reviews"]]
            for related in item.get("related_candidate_ids", [])} - known_ids)
        body = {"version": 1, "identity": identity, "payload": payload,
            "system_prompt": system_prompt, "user_prompt": user_prompt,
            "output_schema": schema, "temperature": 0.2,
            "context_window_tokens": context_window_tokens,
            "reserved_output_tokens": max_output_tokens, "safety_tokens": 128,
            "input_utilization": input_utilization, "estimated_input_tokens": estimated,
            "input_token_limit": input_limit,
            "tokenizer_encoding": counter.encoding_name,
            "exact_model_mapping": counter.exact_model_mapping,
            "max_candidates_per_request": MAX_COMPOSITION_CANDIDATES,
            "missing_related_candidate_ids": missing}
        sha = payload_hash(body)
        reason = ("missing_related_candidates" if missing
            else "candidate_bound" if len(component) > MAX_COMPOSITION_CANDIDATES
            else "context_budget" if estimated > input_limit else None)
        return {"group_id": sha, "request_sha256": sha, "request": body,
            "candidate_ids": sorted(ids), "deferred_reason": reason}

    groups: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    for component in connected_components(candidates, reviews):
        single = request(component)
        if single["deferred_reason"]:
            if pending:
                groups.append(request(pending))
                pending = []
            groups.append(single)
        elif pending and request(pending + component)["deferred_reason"]:
            groups.append(request(pending))
            pending = list(component)
        else:
            pending.extend(component)
    if pending:
        groups.append(request(pending))
    return groups


def _selection(ranges: list[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(ranges):
        if not math.isfinite(start) or not math.isfinite(end):
            raise ValueError("Nonfinite source selection")
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return tuple(merged)


def validate_group_composition(composition: GlobalCompositionResponse,
                               payload: dict[str, Any]) -> None:
    if len({clip.clip_id for clip in composition.clips}) != len(composition.clips):
        raise ValueError("Duplicate clip_id inside composition group")
    ids = {candidate["candidate_id"] for candidate in payload["candidates"]}
    if set(composition.rejected_candidate_ids) - ids:
        raise ValueError("Rejected candidate outside composition group")
    if any(item.candidate_id not in ids for item in composition.unresolved_requests):
        raise ValueError("Unresolved request outside composition group")
    offered = _selection([(float(item["start"]), float(item["end"]))
        for candidate in payload["candidates"] for item in candidate["transcript_ranges"]]
        + [(float(item["start"]), float(item["end"]))
            for review in payload["boundary_reviews"] for item in review.get("selected_ranges", [])])
    selections = set()
    for clip in composition.clips:
        ranges = [(segment.source.start, segment.source.end) for segment in clip.segments]
        selected = _selection(ranges)
        if sum(end - start for start, end in selected) < clip.duration - 1e-6:
            raise ValueError("Duplicate source selection inside clip")
        if selected in selections:
            raise ValueError("Duplicate source selection in composition group")
        selections.add(selected)
        if any(not any(start >= left and end <= right for left, right in offered)
               for start, end in ranges):
            raise ValueError("Source selection outside offered composition group")


def merge_group_compositions(groups: list[tuple[str, GlobalCompositionResponse]],
                             unresolved: list[dict[str, Any]]) -> GlobalCompositionResponse:
    clips = []
    rejected = []
    requests = {payload_hash(item): item for item in unresolved}
    selections = set()
    for group_id, composition in groups:
        for index, original in enumerate(composition.clips):
            clip = original.model_copy(deep=True)
            selected = _selection([(part.source.start, part.source.end) for part in clip.segments])
            if selected in selections:
                raise ValueError("Duplicate source selection across composition groups")
            selections.add(selected)
            clip.clip_id = f"coverage-{group_id}-{index:05d}"
            clips.append(clip)
        rejected.extend(composition.rejected_candidate_ids)
        for request in composition.unresolved_requests:
            payload = request.model_dump(mode="json")
            requests[payload_hash(payload)] = payload
    return GlobalCompositionResponse.model_validate({"contract_version": 1,
        "clips": [clip.model_dump(mode="json") for clip in clips],
        "rejected_candidate_ids": list(dict.fromkeys(rejected)),
        "unresolved_requests": list(requests.values())})
