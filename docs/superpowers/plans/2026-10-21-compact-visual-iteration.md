# Compact Visual Iteration Implementation Plan

> For agentic workers: execute inline with TDD, then one fresh-context whole-change source review. User explicitly delegated execution and visual iterations; do not pause between routine steps.

Goal: improve the shown renderer's composition and timing, then visually iterate actual outputs before user review.

Architecture: reuse LayoutConfig, per-segment face crops, FFmpeg composition and ASS generation. Add only finite pure timing/placement helpers and the small ASS per-range placement option; use existing original source audio in the final mux. Keep the v2 route and its gates separate. Spec:2026-10-21-compact-visual-iteration-design.md.

Global constraints: preserve source/font/brand/original evidence; no custom native rebuild or production activation; no new external provider; no Git initialization/commit/worktree in this nonfunctional Git checkout. Snapshot changed source/test files first. All subagents, if required for review, gpt-6.1-sol/xhigh/fork none.

## Task1: reusable compact rendering and exact media timing

Files: clipperx/editing.py; clipperx/subtitles.py; clipperx/tests/test_editing.py; clipperx/tests/test_subtitles.py; focused new render-graph tests if needed.

Interfaces: keep render_clip/render_batch inputs. Add pure `_segment_frame_counts(segments)` cumulative Fraction(str(...))60Hz frame counts, `_source_audio_ranges(segments)` adjacent-only coalescing and `_caption_margin_for_segment(cfg, mode, crop, source_size)` placement. `clip_to_ass(..., margins_v=None)` optionally consumes one margin per output range, preserving existing callers.

- [x] Snapshot current files. Add real-graph/pure RED cases for full-width compact camera, <=180px bottom band, aspect-aware placement,0.1+0.1+0.1 clocks, nonchronological/gapped audio ranges,60Hz canvas/video and original-source final audio. Add ASS per-range margin tests and mismatch rejection. Run the tests and observe genuine old-behavior failures.
- [x] Update LayoutConfig compact defaults and camera sizing; render exact cumulative frame counts with explicit60FPS and zero-based segment PTS. Intermediates are video-only; final audio is assembled directly from ordered original-source ranges. Do not retime speech. Preserve B-roll video behavior.
- [x] Pass resolved per-segment caption margins using detached caption ranges. Use narrow caption band in split layouts; safe overlay placement for full content/camera. Re-run focused cases and related editing/subtitle/plan-render tests. Final related suite41passed5.28s, not full repository.

## Task2: actual visual iteration, including camera change

Files: fresh data/validation_visual/runs/compact-iteration-20261021-01/ scripts/plans/reports; project PROMPT_GESTOR.md workflow update.

- [x] Render the same37s source clip with new defaults first, preserving its narrative range. Inspect compactness, face/eyes/mouth, title/brand, captions and mobile obstruction risk. Add motivated per-segment crops only if visible source evidence supports them, and render another version when findings warrant it.
- [x] Independently decode all frames and count rate/duration, extract overview and detailed diagnostic frames, inspect actual rasters. Compare full source/output audio and local windows around cuts to detect cumulative offset rather than rely on stream metadata.
- [x] Render both actual camera/desktop transition directions and varied cases using evidence-backed local crops; inspect transition frames and original audio. Secondary cases have diagnostic titles but no invented transcript. Main sample validates actual caption bounds/cue timing; secondary cases do not establish speech/caption semantics. No claim stable-camera test proves arbitrary changes. Fix actual failures and repeat.
- [x] Record each actual iteration and remaining limits; update project guidance to make render/review/correct the standard process. Obtain one gpt-6.1-sol/xhigh read-only whole-change review and close Important findings with regression tests. Preserve all snapshots because no recoverable Git history exists.

Final checkpoint: actual render/QA/correction cycles completed; source review completed with three Important findings fixed via focused RED/GREEN. Actual caption-right-edge and title/logo boundary findings were corrected and re-rendered. Main08/matrix05 all11 outputs decoded(6683frames), measured whole/local audio lag0, actual final rasters inspected. Main08 is ready for user visual review; blank-source negative remains HOLD, derived/order controls remain separately labelled. See retained ledger and run RESULTS.md/final-measurements.json for exact identities/results/limits. Final geometry192px top guard plus72px header,52px default captions/210px side margins. Historical ASR lexical accuracy, phone UI, v2/native/full-VOD and autonomous-director acceptance are not claimed.

Review focus: tiny cuts/zero-frame rejection, nonchronological edits, no AAC-padding accumulation, changing camera aspect/position, caption position at cuts, no clipping/stretching/unverified ASR rewrite, no production/v2 claims from compatibility previews.
