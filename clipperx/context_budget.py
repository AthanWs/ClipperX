"""Token-aware sequential transcript batching."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Protocol, Sequence

from .schema import TranscriptSegment


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...


class TiktokenCounter:
    """GPT-family token counter with a documented encoding fallback."""

    def __init__(self, model: str) -> None:
        import tiktoken

        try:
            self.encoding = tiktoken.encoding_for_model(model)
            self.encoding_name = self.encoding.name
            self.exact_model_mapping = True
        except KeyError:
            self.encoding = tiktoken.get_encoding("o200k_base")
            self.encoding_name = self.encoding.name
            self.exact_model_mapping = False

    def count(self, text: str) -> int:
        return len(self.encoding.encode(text))


@dataclass(frozen=True, slots=True)
class ContextBudget:
    context_window_tokens: int
    prompt_tokens: int
    reserved_output_tokens: int = 4096
    safety_tokens: int = 1024
    utilization: float = 0.85
    overlap_tokens: int = 1000

    @property
    def input_token_budget(self) -> int:
        usable_context = int(self.context_window_tokens * self.utilization)
        available = usable_context - self.prompt_tokens - self.reserved_output_tokens - self.safety_tokens
        if available <= 0:
            raise ValueError("El presupuesto no deja tokens disponibles para transcript")
        return available

    def validate(self) -> None:
        if self.context_window_tokens < 4096:
            raise ValueError("context_window_tokens debe ser al menos 4096")
        if not 0.5 <= self.utilization <= 0.95:
            raise ValueError("utilization debe estar entre 0.5 y 0.95")
        if self.overlap_tokens < 0:
            raise ValueError("overlap_tokens no puede ser negativo")
        if self.overlap_tokens >= self.input_token_budget:
            raise ValueError("El solapamiento consumiría todo el input disponible")


@dataclass(frozen=True, slots=True)
class TranscriptBatch:
    batch_id: str
    context_start_index: int
    new_start_index: int
    new_end_index: int
    cursor_before: int
    cursor_after: int
    start_timestamp: float
    end_timestamp: float
    transcript_tokens: int
    text: str

    def metadata(self) -> dict[str, object]:
        payload = asdict(self)
        payload.pop("text")
        return payload


def build_transcript_batches(
    segments: Sequence[TranscriptSegment],
    *,
    counter: TokenCounter,
    budget: ContextBudget,
) -> list[TranscriptBatch]:
    budget.validate()
    if not segments:
        return []
    lines = [_segment_line(index, segment) for index, segment in enumerate(segments)]
    line_tokens = [counter.count(line) for line in lines]
    if any(tokens > budget.input_token_budget for tokens in line_tokens):
        raise ValueError("Un segmento individual excede el presupuesto de contexto")

    batches: list[TranscriptBatch] = []
    cursor = 0
    while cursor < len(segments):
        context_start = _overlap_start(cursor, line_tokens, budget.overlap_tokens)
        used_tokens = sum(line_tokens[context_start:cursor])
        end = cursor
        while end < len(segments):
            candidate_tokens = line_tokens[end]
            if end > cursor and used_tokens + candidate_tokens > budget.input_token_budget:
                break
            if end == cursor and used_tokens + candidate_tokens > budget.input_token_budget:
                context_start = cursor
                used_tokens = 0
            used_tokens += candidate_tokens
            end += 1
        if end == cursor:
            raise RuntimeError("El cursor no avanzó al construir batches")
        text = "\n".join(lines[context_start:end])
        transcript_tokens = counter.count(text)
        while transcript_tokens > budget.input_token_budget:
            if end > cursor + 1:
                end -= 1
            elif context_start < cursor:
                context_start = cursor
            else:
                raise ValueError("Un segmento individual excede el presupuesto de contexto")
            text = "\n".join(lines[context_start:end])
            transcript_tokens = counter.count(text)
        identity = f"{context_start}:{cursor}:{end}:{segments[cursor].start:.3f}:{segments[end - 1].end:.3f}"
        batches.append(
            TranscriptBatch(
                batch_id=hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12],
                context_start_index=context_start,
                new_start_index=cursor,
                new_end_index=end,
                cursor_before=cursor,
                cursor_after=end,
                start_timestamp=segments[context_start].start,
                end_timestamp=segments[end - 1].end,
                transcript_tokens=transcript_tokens,
                text=text,
            )
        )
        cursor = end
    return batches


def _overlap_start(cursor: int, line_tokens: Sequence[int], overlap_tokens: int) -> int:
    start = cursor
    used = 0
    while start > 0 and used + line_tokens[start - 1] <= overlap_tokens:
        start -= 1
        used += line_tokens[start]
    return start


def _segment_line(index: int, segment: TranscriptSegment) -> str:
    speaker = segment.speaker or "UNKNOWN"
    return f"[{index}|{segment.start:.3f}-{segment.end:.3f}|{speaker}] {segment.text.strip()}"
