# Autonomous vertical workflow implementation plan

> For agentic workers: use task-scoped implementation and independent review; continue through the tasks without routine human approval pauses. The user has approved autonomous development and the shorts-first product design.

Goal: one explicit operation advances a source through analysis, editorial selection, visual direction and resumable export, keeping unresolved work visible rather than claiming a finished editor.

Architecture: reuse JobStore, Parakeet stages, the existing editorial/visual planners, director READY receipts and public v2 renderer. Add thin coordination instead of another editor. Keep technical budgets distinct from editorial quantity. Add render-review iteration only on top of measured exported artifacts.

Global constraints:

- Vertical shorts only, maximum 179 seconds each. Preserve actual context/setup/payoff and original audio. No fabricated controversy, camera identity, timing or evidence.
- No fixed global clip quota in coverage mode. Existing bounded validation mode remains available.
- Preserve current source/media/fonts/branding, old evidence, pending statuses and user changes. No deletion, Git repair/init/commit/push, dependency installs, account/key provisioning or custom native build.
- GPT-6 Luna via existing CLIProxyAPI is the selected application model. Subagents must be gpt-6.1-sol/xhigh/fork_turns=none. Never print credentials. Default passive commands must not invoke a model or load provider credentials.
- Maintain strict v1 editorial and v2 visual contracts, exact ranges, budgets, source identity, READY custody and current SFX approval. Native qualification and perception are not proved by fake clients or a saved plan.
- Actual render is not editorial approval. No qa_passed/full_qa_passed/user_approved promotion without its corresponding evidence.
- File-exclusive parallel tasks; root owns CLI/workflow integration and docs. Snapshot each existing edited source before changes because Git is not functional. Use apply_patch and observed RED/GREEN tests.

## Task 1 — Coverage-mode editorial planning

Read `.superpowers/sdd/autonomous-flow-20261025/coverage-brief.md`.

- [ ] Add an explicit no-global-quota coverage mode and bounded composition groups with persistent validated caches.
- [ ] Preserve connected topics/review relationships and defer oversized unresolved groups explicitly.
- [ ] Exercise more than three justified clips, budget stops and resume without duplicate calls; keep the existing bounded mode tests.
- [ ] Independent source/spec review, focused regression and report before integration.

## Task 2 — Resumable export of visual manifests

Read `.superpowers/sdd/autonomous-flow-20261025/export-brief.md`.

- [ ] Consume ready/review/pending manifest entries using existing director authority and renderer; persist progress per clip.
- [ ] Reuse only byte-verified exports tied to unchanged inputs; preserve failed/older attempts and never render review receipts as plans.
- [ ] Verify multiple clips, partial failure, restart, tampering and duplicate invocation with real files and injected rendering.
- [ ] Independent source/spec review, focused regression and report before integration.

## Task 3 — Root workflow/CLI integration

Create `clipperx/autonomous_pipeline.py` and focused tests; modify CLI only after Tasks 1/2 publish their exact signatures.

- [ ] Explicit inference admission occurs before credentials/client construction; wire existing JobStore-backed transcript and visual timeline preparation.
- [ ] Run editorial coverage, visual planner and export coordinator; return a persistent run summary distinguishing pending/review/rendered-unreviewed/error, not just a list reported 100% done.
- [ ] Preserve all existing commands. Add `auto-edit` with explicit model/context/budget/confirmation and documented resumable inputs.
- [ ] Run a bounded real source through the actual public flow using the authorized Luna and existing runtime; inspect output pixels/audio if rendering succeeds. A runtime failure is persisted, diagnosed and corrected in scope, never hidden with legacy fallback.

## Task 4 — Render review and product continuation

- [ ] Read the source-grounded visual-quality audit, then enforce persistent brand and timed captions where feasible without weakening source protection.
- [ ] Add versioned render-review feedback and bounded retry only after export exists. Preserve original plans and distinguish technical, visual, timing and subjective checks.
- [ ] Expose the same backend operation to the existing UI after its progress/cancellation behavior is tested; do not enable accidental inference at startup.
- [ ] Validate varied real cases before scaling to a full VOD. Continue remaining tasks after each checkpoint; do not stop merely because focused tests passed.
