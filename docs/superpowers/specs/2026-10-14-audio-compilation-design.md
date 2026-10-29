# Mixed-track audio compiler — resumed Phase 3 authority

This technical clarification preserves the approved v2 plan and user objective.
It replaces the obsolete `sidechaincompress` example, not the planned audio
semantics. No new director, RenderProfile or UI fields are introduced.
Controller reviewed the complete planning proposal in
`.superpowers/sdd/phase3-audio-schedule-design.md`; its record table and PCM-cache
framework are recommendations, not eleven mandatory types or a new subsystem.
Implement the smallest immutable schedule/measurement/compiled-result interfaces
needed by Tasks 5 and 6. Keep `audio_mix.py` focused; escalate if a new file split
or architectural choice is needed instead of silently growing a framework.

## Required behavior

- Revalidate and detach the accepted v2 plan. Frozen results use detached tuples
  and finite scalar data, not aliases to mutable caller models/collections.
- Preserve original source windows, declared output order (including repeated
  selections), shot-local cue times, source gains and normalization/limiter flags.
  Source audio is the existing single mixed track, not isolated voice/music.
- Pure schedule/graph construction must not run tools, read credentials/settings,
  resolve a different FFmpeg or write files. Host measurement is an explicit
  operation. Use the same paired runtime/requirements and source-clock binding
  as rendering, not a second binary-selection policy.
- Adjacent shot boundaries share output sample boundaries computed from absolute
  output time at 48000 Hz, `Q(t)=floor(t*48000+1/2)`, using exact rational plan
  decimals. Do not accumulate independently rounded durations. Store declared
  seconds separately from integer sample representation. A planned nonempty
  interval mapping to zero samples fails explicitly.
- Task6 owns bounded source seeks, decoder-only pre-roll and raw PTS/source-time
  mapping. Measurement and render must share its immutable binding identity.
  No independent first-audio/first-video rebasing, hours of decode to reach a
  late window, source silence padding or async healing to hide missing coverage.
  Task5 may compile from original-PTS input labels and expose an explicit binding
  interface; it must not invent unqualified seek flags.

## Shared normalization authority

Build one bus containing all and only normalization-enabled selected source
samples, in output order. For the all-default plan it is the complete selected
source. For true/false/true flags, concatenate the true portions for normalization
and map them back to their recorded output positions; false portions bypass it.
Visual subdivisions, gain changes and cues do not create new normalization
instances. In particular, splitting one narrative interval at a visual-state
event must not reset volume normalization.

Measure the bus once with EBU R128, then supply measured two-pass `loudnorm`
parameters targeting `I=-14`, `LRA=11`, `TP=-1.5`. Retain measured I/LRA/TP,
threshold and target offset, source/schedule/binding/runtime identity, expected
sample count and channel layout. Missing, stale or mismatched evidence fails
before production rendering. No cache hit based only on a path or mtime.

For finite in-range statistics, let delta=-14-measured_I. Select measured linear
explicitly only if measured_LRA<=11 and measured_TP+delta<=-1.5; otherwise select
measured dynamic explicitly before pass two and record the reason. Both branches
provide the measured values/offset. Inspect actual `normalization_type` after
processing: a mismatch is an error, not permission to retry automatically or
accept the filter's implicit fallback. Dynamic processing must restore 48000 Hz
and retain the expected sample count. A lower observed LRA need not be expanded.

Use typed outcomes for insufficient duration (<19200 enabled samples), proven
digital silence, below-gate/undefined measurements, malformed/nonfinite stats,
mode mismatch and successful measured processing. `-inf` alone is never evidence
of silence; sub400ms real audio is not silence. Under this initial strict policy,
required normalization with unmeasurable stats is a typed preparation failure,
not fabricated finite statistics, silent passthrough or one-pass fallback.
Proven silence requires an independent all-zero check of the complete bus, not
a loudness string. A disabled authority needs no measurement or loudnorm.

Normalization targets apply to this enabled source bus BEFORE explicit planned
source gain, ducking, cue gain/mixing and limiting. Apply `source_gain_db` exactly
once downstream of normalization, so normalization cannot cancel that decision.
Record upstream observations and post-mix/delivered measurements with distinct
scope. Filter targets do not prove achieved final loudness/true peak; retain
observed values/residuals rather than inventing a `targets_met` claim or new
acceptance tolerance in this task. Metadata is `mixed_track_normalization`,
never `voice_normalization`.

## Explicit cues and exact source ducking

Only resolve cue IDs already in the v2 plan through `RenderAssetRegistry`.
`impact` is an internal ID that a host fixture may register to `impact.wav`;
do not add it to the default registry or invent a cue. Cue files are trimmed
or silence-padded only to their declared interval; source padding is different
and forbidden as coverage repair. Preserve source channels. A different cue
layout requires an explicit qualified host mapping, otherwise fail rather than
allow implicit downmix. Do not claim every channel layout is qualified.

For each cue's absolute output interval [a,b), duration d, use
A=min(.005,d/4), R=min(.010,d/4), S(x)=x*x*(3-2*x). Its ramp strength is zero
at/outside a,b, S((t-a)/A) in attack, 1 on the plateau and S((b-t)/R) in release.
Attenuation in dB is `duck_source_db * strength`. Across overlapping cues choose
the most negative CURRENT ramped value, not summed dB or unrequested additive
ducking. Convert that combined dB to amplitude once per sample/channel.
The full planned reduction holds on an uncontested plateau. No attack/release
tail may attenuate outside the explicit cue window. Zero-dB duck never reduces
the source. No cues means no duck evaluator.

Use per-sample `aeval` with `val(ch)` and `c=same` (or an equally qualified exact
implementation), not frame-evaluated `volume` for cue boundaries. A compressor
cannot guarantee the fixed planned dB/window; do not insert a no-op compressor
to satisfy the old example. These ramps are anti-click implementation details,
not new cue fades, source crossfades or an assurance that arbitrary edits cannot
click. Gain and cue values remain exactly authored.

Mix explicit cues with `amix` weights1, normalize0, dropout_transition0 and a
source-owned duration, enforcing exact output samples. Apply the limiter only
on maximal contiguous limiter-enabled runs, independent of visual splits.
Use `alimiter=limit=0.944:level=0:latency=1`, explicit level_in/out1, attack5ms,
release50ms, asc0. Disabled runs bypass it. Verify compensated sample timing
AND EOF rather than inferring correctness from the option string. The .944
sample threshold is not a claim of final -1.5 dBTP or AAC peak compliance.

## Task boundaries and verification

Task5 owns `audio_mix.py`, its tests and minimal `visual_compiler.py`/test
integration. It defines pure scheduling, strict measurement parsing/mode choice,
measurement/pass-two specifications, graph compilation and explicit host helpers
needed to prove them. Task6 owns production bounded extraction/materialization,
safe graph/input assembly, one video encode, metadata, sidecars and delivered
A/V acceptance. A bounded lossless float PCM scratch input may make both passes
consume identical selected samples; it is not a new asset or additional lossy
video encode. Do not build its general cache/orchestrator in Task5.

Test behavioral contracts, not graph strings alone: subdivision invariance,
true/false mapping, exact gains, mutable-input isolation, stale receipts,
finite-linear/dynamic mode choice, actual pinned-runtime measured processing,
undefined/quiet/nonzero short material, decoded fractional cue boundaries and
plateaus, overlap minimum, zero duck, source channels, cue padding/gain and
limiter no-boost/latency/EOF. Small synthetic PCM diagnostics are allowed;
no real source-video or full-VOD/model campaign here. Later decoded flash/click,
AAC, late-window and real-window gates remain open. Focused audio/compiler tests
plus scoped Ruff/mypy/compile; report imported historical typing debt separately.

## Documentation

Current Context7 `/websites/ffmpeg_documentation` queries returned official
[FFmpeg audio-filter documentation](https://ffmpeg.org/ffmpeg-all.html): measured
linear eligibility and implicit dynamic reversion; dynamic192kHz processing;
aeval sample/channel semantics; amix normalization defaults; and alimiter
auto-level/lookahead/EOF compensation. These are documented capabilities,
not evidence that the installed runtime or delivered MP4 has passed acceptance.
