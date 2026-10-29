# Decimal Overlay Boundary Implementation Plan

> For agentic workers: execute inline with TDD and one final fresh-context source review.

Goal: accept geometrically touching decimal overlay edges without accepting any positive overlap.

Architecture: keep the existing normalized rectangle models and every consumer; only evaluate strict intersection using exact decimal rationals. Spec: `docs/superpowers/specs/2026-10-21-decimal-overlay-boundaries-design.md`.

Global constraints: no epsilon or geometry rewrite; no inference beyond the two completed preview calls; no native build/candidate/production gate changes; preserve all originals. This checkout is not a functional Git repository (fresh rev-parse failed); do not initialize Git, commit, create a worktree or delete evidence. Preserve exact before-file snapshots instead.

## Task 1: exact overlap predicate

Files: `clipperx/visual_contracts.py`, `clipperx/tests/test_visual_contracts.py`.
Consumes/produces: unchanged `_rectangles_intersect(left: NormalizedRect, right: NormalizedRect) -> bool`; existing VisualShot overlay validation must retain strict positive-overlap rejection.

- [x] Snapshot the two current files before edits, with sizes and hashes.
- [x] Add parameterized real-contract cases for subtitles/title/brand, vertical/horizontal edges, exact touching at .82, an adjacent-representable gap, and an adjacent-representable real overlap. Test the real helper symmetrically as well.
- [x] Run `python -B -m pytest clipperx/tests/test_visual_contracts.py -q -k decimal_boundary` and observe exact-touch failures, not imports or syntax errors. Actual six touching failures, twelve other cases passed. A test-only decimal literal rounded to .82; replace it with `nextafter(.82, 0)` so the negative really overlaps. No production predicate workaround was made for that test typo.
- [x] Add `from fractions import Fraction`; change only the predicate to exact decimal comparisons. The four conditions stay strict `<`/`>` without tolerance. Example: `Fraction(str(left.x)) < Fraction(str(right.x)) + Fraction(str(right.width))`; apply symmetrically for right and bottom edges.
- [x] Run the focused cases, then the related contract/director/prompt/compiler tests once. Actual focused18 passed0.34s; related692 passed253.17s, exit0. Frozen Luna output now passes schema but retained host geometry rejects actual protected-region occlusion; original receipt stays review_required. Zero new inference.
- [x] Obtain one fresh source review of this finite diff using the user's required gpt-6.1-sol/xhigh; address any Important finding with RED/GREEN. Review source-review.md approved, no findings; source/test hashes still match after the full suite. Keep all snapshots; no Git history exists to recover them from.

Review focus: touching symmetry, genuine microscopic overlap rejection, integer coordinates, finite normalized values, public overlay validation, no epsilon or bypass, no misleading revalidation of the original failed call or production status.
