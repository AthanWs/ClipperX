from clipperx.context_budget import ContextBudget, build_transcript_batches
from clipperx.schema import TranscriptSegment


class WordCounter:
    def count(self, text: str) -> int:
        return len(text.split()) * 100


class CharacterCounter:
    def count(self, text: str) -> int:
        return len(text)


def test_batches_advance_cursor_with_overlap() -> None:
    segments = [
        TranscriptSegment(start=float(index), end=float(index + 1), text="uno dos tres cuatro")
        for index in range(20)
    ]
    budget = ContextBudget(
        context_window_tokens=4096,
        prompt_tokens=200,
        reserved_output_tokens=512,
        safety_tokens=256,
        utilization=0.5,
        overlap_tokens=600,
    )

    batches = build_transcript_batches(segments, counter=WordCounter(), budget=budget)

    assert batches[0].cursor_before == 0
    assert batches[-1].cursor_after == len(segments)
    assert all(left.cursor_after == right.cursor_before for left, right in zip(batches, batches[1:]))
    assert any(batch.context_start_index < batch.new_start_index for batch in batches[1:])


def test_batches_validate_joined_text_against_budget() -> None:
    segments = [
        TranscriptSegment(start=float(index), end=float(index + 1), text="contenido")
        for index in range(8)
    ]
    budget = ContextBudget(
        context_window_tokens=4096,
        prompt_tokens=3580,
        reserved_output_tokens=100,
        safety_tokens=100,
        utilization=0.95,
        overlap_tokens=50,
    )

    batches = build_transcript_batches(segments, counter=CharacterCounter(), budget=budget)

    assert all(batch.transcript_tokens <= budget.input_token_budget for batch in batches)
    assert batches[-1].cursor_after == len(segments)
