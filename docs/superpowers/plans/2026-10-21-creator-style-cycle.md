# Creator Style Cycle Implementation Plan

> For agentic workers: implement inline with executing-plans/TDD, one finite fresh-context whole-change review. User already supplied direction and execution method; do not stop for routine approvals.

Goal: implement requested visual changes in application code and keep automatic-director intent explicit, then render/review/correct varied actual outputs.

Architecture: reuse LayoutConfig/render_clip/ASS and existing modern title fitter. Add appended camera_position/title persistence/title color fields, shared per-segment header geometry, measured title size parameter and model-facing creator preferences. Reuse cumulative60Hz clock/original ordered audio. Do not add a parallel editor backend.

Spec: docs/superpowers/specs/2026-10-21-creator-style-cycle-design.md.

Global constraints: preserve original VOD/fonts/branding and prior evidence; no native builds, publication, selector/key/account mutations. No fake transcript. Nonfunctional Git requires retained before snapshots, no Git initialization or deletion. Reviewer model gpt-6.1-sol/xhigh/fork none. Model-facing edits do not prove autonomous long-VOD success.

## Task1 — reusable layout and text

Files editing.py, overlay_ass.py, caption_templates.py, pipeline.py; tests test_editing_creator.py and affected geometry tests.

- [x] Snapshot files; write and observe REDs for camera-top order, optional bottom order, per-segment header/title/brand/caption geometry, persistent brand with title, actual measured title>46 retaining all words, vivid black-outlined ASS and lossless pipeline config replacement.
- [x] Implement shared `_header_height(cfg)` / `_header_geometry(cfg, mode, crop, source_size)`; use actual cumulative-frame enable intervals for overlays, not guessed source timestamps. `_title_event(... max_font_size=46)` preserves existing callers, compatibility requests configured88. Header uses chosen background color, title uses chosen text color. New fields appended. Main audio path/60Hz/SAR invariants unchanged.
- [x] Run focused and related suites; record intentional old-design test updates separately from genuine new REDs. Full suite3456passed/5skipped at its imported head; last prompt-only deltas separately verified by9prompt/116related tests, not attributed to earlier bytes. Mypy six production files clean.

## Task2 — automatic intent and actual visual cycles

Files visual_prompts.py, test_visual_prompts.py; fresh data/validation_visual/runs/creator-style-20261021-01 tooling/report; project guidance.

- [x] Add finite creator preference payload to existing director prompt; test actual serialized output without claiming prompt wording proves model compliance. Reuse existing safety/identity/contracts, no arbitrary source-range changes or SFX authorization. Existing narrative reasoning now actually transmitted after observed missing-brief RED/GREEN.
- [x] Render full main37s with clear selective punches plus dark/light and camera-order comparison snippets; inspect actual full-resolution images, title/brand/caption timing/width and source audio across cuts/tail. Keep versions; fix actual hair-crop finding and rerender. Corrected main is iteration02; light comparison iteration04 gets its own retained QA.
- [x] Exercise full-camera, browser, PIP/portrait, real mode changes and derived camera relocation. Blank input remains HOLD. Twelve configured final cases7163decodedframes, all whole/local audio measuredlag0; observed brand present in every fixture frame. Actual source/derived/control labels retained; not10h acceptance.
- [x] One fresh-context6.1Sol/xhigh source review, Important slightly-portrait crop fixed with RED/GREEN and actual control03. Two new integration deltas got bounded follow-ups, not repeat whole reviews. Inherited subframe ASS timing Minor remains deferred. Full report RESULTS.md; media handoff uses actual principal, not model-rejected output.

Automatic-editor trial outcome: four actual Luna trials/fivecalls221857reportedtokens. First technically ready plan rendered via normal v2 HOST and passed scopedclock/audio checks, but is editorially rejected for no captions/tiny webcam. Later brief transport fix made requested subtitle style reach actual proposals; forbidden geometry remained held. Final trial requested/received more source frames then returned review_required. No manual JSON-coordinate repair or gate bypass. Full-VOD all-worthy-moment coverage and reliable autonomous editorial acceptance remain unfinished, not hidden under this completed visual-step checklist.

Review focus: zero/very short output ranges at overlay boundaries; camera portrait/partially out-of-bounds crops; title/brand overlap when modes change; main default caption extents on both background colors; pipeline configuration must not silently lose newly appended fields. Each is covered by focused test or final media inspection.
