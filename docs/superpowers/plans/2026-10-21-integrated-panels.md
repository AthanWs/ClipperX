# Integrated panels implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans inline with one independent final review. User already approved autonomous integration; no additional approval checkpoint.

**Goal:** Integrate overlays without the central black strip and fix the verified small corner-webcam detection failure conservatively.

**Architecture:** Append a default integrated mode to LayoutConfig. Keep source content contained over full image blur, reserve only the sharp viewport where overlays need space, and use zero panel gap/header geometry in split mode. Detector corner sampling remains isolated in face.py.

**Tech Stack:** Existing Python, ffmpeg-python, FFmpeg, OpenCV, local MediaPipe model; no dependency changes.

**Spec:** docs/superpowers/specs/2026-10-21-integrated-panels-design.md

## Global constraints

Preserve source audio clock/60FPS/SAR and independent content crops. Preserve originals/fonts/brand. No Git metadata mutation; retain before snapshots instead. Existing local provider authorization does not require new calls for these geometry tests.

## Review focus

- Top/bottom camera and differing aspect ratios: touching panels, readable separate overlays.
- Full-camera/portrait/PIP: no opaque pad and no sharp-source overlap with title lane.
- Cut boundaries: title/brand present at both sides, original audio retained.
- Foreign/multiple faces: no global-largest content-face selection.
- Webcam relocation: reject mixed-side static crop rather than falsely averaging.

## Task 1: Integrated renderer (parent)

Files: clipperx/editing.py; clipperx/tests/test_editing{,_compact,_creator,_integrated}.py.
Interface: appended LayoutConfig.integrated_overlays=True; old explicit False retains old band layout. _compose_contained gains optional top_inset=0 without changing default behavior.

- [x] Add behavioral tests: top geometry (672,1248,0,672), bottom (0,1248,1248,672), no opaque final drawbox/pad, persistent independent overlays, protected contained portrait viewport. Observe RED.
- [x] Implement touching split panels, source-blur overlay space, stronger title outline, separate brand placement. Preserve source audio/cut logic. Observe GREEN and old geometry explicitly False.
- [x] Render pilot, inspect actual frame, correct if needed; render main + varied matrix and run whole-file frame/PCM diagnostics.

## Task 2: Corner detection (camera audit agent)

Files: clipperx/face.py and clipperx/tests/test_face.py only. Interface: existing detect_webcam_region -> WebcamRegion | None, unchanged confidence/model, conservative repeated corner support. Own cap/detector cleanup in finally.

- [x] RED/GREEN for TL/TR, global larger foreign face, isolated false positive, ambiguous/mixed locations, blank/full-camera, owned/injected cleanup.
- [x] Actual production-entry rerun on same eight bounded audited windows; retain before/after and inspect crop limits.

## Task 3: Verification and handoff

- [x] Freeze source; run targeted tests and full suite once, media QA independently. Retain exact commands/source/output identities. Two subsequently discovered visual deltas have fresh134targetedtests and current realrender QA; fullsuite3484/5 belongs to its imported head.
- [x] One fresh gpt-6.1-sol xhigh read-only whole-change review against snapshots. Resolve Important/Critical findings with RED/GREEN, scoped tests. No severity findings; parent-detected visual corrections narrowly reviewed.
- [x] Deliver exact preview and honest camera coverage/remaining limits; do not call controlled crops automatic detection or ten-hour readiness. See run RESULTS.md and finalsummary.
