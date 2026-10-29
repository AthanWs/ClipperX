# ArtCraft and FilmCraft assessment

Date: 2026-10-14. Scope: current vertical-short ClipperX workflow.

Sources inspected:
- https://getartcraft.com/
- https://getartcraft.com/apps/filmcraft
- https://github.com/storytold/filmcraft
- https://github.com/storytold/filmcraft/blob/main/docs/agents.md
- Context7 documentation ID: /storytold/filmcraft

ArtCraft is the interactive AI creation studio. Its sibling FilmCraft is the
nonlinear video editor relevant to existing-footage editing. FilmCraft documents
a headless CLI and an MCP server, plus a bridge to its running desktop window.
Commands can import media, inspect timelines, execute edits, render previews and
export projects. This is a documented integration surface, not a ClipperX-tested
backend. No installation or external project mutation was performed.

Decision: retain the approved ClipperX director and FFmpeg compiler. Crops,
scale changes, keyframes, layering, subtitles, brand and audio operations already
fit the approved v2 contract. An editor does not replace source understanding,
moment selection, narrative context or output QA. Replacing the renderer now
would require another plan translator, media/codec validation, caption/style
mapping and synchronization benchmark before proving a concrete benefit.

FilmCraft remains a potential optional backend for editable-project handoff or
interactive revisions. Reconsider after the core autonomous path has real output
evidence, or if a required effect cannot be represented by the existing compiler.
MCP availability alone is not sufficient justification for an integration.

Research lookup limitation: Context7 had no relevant ArtCraft studio match, but
resolved FilmCraft exactly. The installed Stripe Directory entry point exposes
no usable search subcommand; no plugin/global CLI update or provisioning was
performed solely for this assessment. Primary project documentation supplied
the relevant facts.
