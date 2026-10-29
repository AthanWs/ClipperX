# Audio-only observed preparation — Phase3 technical prerequisite

Status: adopted refinement of the approved automatic vertical-shorts plan.
No new creative behavior, model, UI/profile control, source range, threshold,
gain or dependency. Root read the 6a1 proposal and its complete audio-only
addendum, then inspected the actual approved 6a0 interfaces and sample-window use.

## Why this prerequisite

The analyzer's `_sample_window` runs to the next sample/end; default sparse
intervals are10s. `observe_windows` always launches video AND audio and its
combined-window ceiling is8s. Routing whole feature intervals through that API
would both reject ordinary intervals and risk decoding virtually the entire
source video merely to obtain audio.

Options considered: increasing combined limits would add unnecessary video work;
a second independent audio decoder would duplicate provenance/phase/error logic.
Selected: extract the existing audio preparation into ONE shared helper and add
an explicit audio-only entry. Preserve the combined observer and strict v1
receipts, including the successful R/S proofs, unchanged in meaning.

## Scope and interfaces

Only modify render_clock.py, source_decode.py and their two tests. Runtime,
analyzer, audio_features, JobStore, caches, models, audio_mix, CLI and originals
remain untouched. New unit name6a0-audio-seam precedes6a1-A/6a1-B.

In source_decode expose:

```python
@dataclass(frozen=True, slots=True)
class DecodedAudioWindow:
    receipt: SourceAudioDecodeReceipt
    pcm: bytes

def observe_audio_window(
    runtime: RenderRuntime,
    origin: SourceTimeOrigin,
    window: Window,
    *,
    audio_policy: str,
    limits: AudioDecodeLimits = AudioDecodeLimits(),
) -> DecodedAudioWindow: ...
```

`AudioDecodeLimits` is a detached immutable host policy, not a user control.
`DecodedAudioWindow` copies any supported byte-like caller input into immutable
bytes and checks its length/hash against its validated audio receipt. It must
not retain a mutable caller buffer or silently accept mismatched PCM identity.
Same selected pinned CPU runtime, source identity checks, exact common O,
microsecond-only decoder-margin serialization, three existing named conversion
policies and original44.1/48k mono/stereo support. Exactly ONE audio child per
call, with -vn/-sn/-dn, no video map/filter/raw pixels/hidden video child. No
normalization/gain/cues/limiter/apad/async healing. Decoder-only margins are not
changes to declared windows. Unsupported combinations fail before a child.

Extract the current audio argv, streaming, sequence/count/phase, observation and
bounded diagnostic code into one narrowly scoped private helper shared by the
combined and audio-only callers. No second decoder or duplicated audio algorithm.
Preserve cleanup/diagnostics through consumer and post-iteration errors. Combined
limits remain EXACTLY the existing8s policy; no new runtime selector or fallback.

## Separate finite audio-only limits

Defaults and hard upper ceilings: max_window300s INCLUDING effective decoder
margins, timeout45s, max_blocks65536 per input/output stage, max_samples14400000
per-channel per stage, max_decoded_bytes60000000, max_retained_bytes60000000,
max_log_bytes16000000. Positive integer fields reject bool and may only tighten
these ceilings. preroll=.5s/postroll=.25s by default, each exact in[0,1].
Windows are within the known container; the effective clipped decoder interval
must fit max_window before launching a child. Oversize intervals fail, never
silently split a feature across resampler restarts. One continuous resampler per
call; I/O may be chunked, resampling may not restart at an8s boundary.

Returned PCM is bounded in memory by the retained-payload limit; temporary
copies/metadata/pipe buffers do not constitute a measured total-RSS claim.
No unbounded subprocess capture or whole-source PCM. Future6a1 writes returned
PCM into its own generation and computes features once over exact selected
samples, not averages of per-chunk RMS/percentiles. No file writer/cache or
feature calculator is added to this prerequisite.

## Truthful distinct receipt

Add pure immutable `SourceAudioDecodeReceipt` in render_clock, with the same
constructor identity/payload pattern and `to_json`, `from_json(..., origin,
policy)` and `validate_origin` operations as the combined receipt.

Envelope fields: version1, kind=`audio_only`, scope=`window`, origin_id,
decoder_id, policy, payload and sha256. Payload fields EXACTLY windows, limits,
audio, argv. Exactly one declared window, one actual audio observation using the
existing audio record shape, and ONE nonempty argv list of actual executed string
arguments. No video rows, fake outcomes, dummy argv or inferred video coverage.
Use `AUDIO_DECODE_POLICY = DECODE_POLICY + ";audio-only;v1"` for the explicit
receipt operation; underlying origin decoder identity remains the same pinned
pair/pixel-and-audio conversion identity, not a different binary selection.

Both receipt loaders reject the other's schema. Combined v1 serialization and
strict acceptance rules stay compatible; do not upgrade/reshape old receipts.
Share narrow pure audio-record, origin/audio-bound and conversion validators so
both producers/loaders enforce the SAME rates/channels/formats, continuity,
count/phase, exact sample-grid/source phase, coverage and byte/budget invariants.
Reject duplicate keys, mutable impostors, malformed/nonfinite or stale/current-
origin conflicts and otherwise-valid rehashed contradictions. Hashing detects
corruption, not adversarial authenticity. Actual persisted PCM must later match
its supplied aggregate/block hashes; hashes are not reconstructed from hashes.

## Focused verification and preserved gates

TDD pure limits/type/receipt/cross-schema/current-origin/corruption checks, fake
child zero-video/one-audio argv/cleanup/diagnostic tests, then two tiny HOST cases.
One HOST audio interval is longer than8s and uses44.1k stereo to16k mono with
known impulses on both sides of the8s point and nonzero origin/seek; compare the
entire returned PCM to an independent continuous bounded FFmpeg reference
without the observation parser. One same-rate48k stereo control preserves
distinct channel impulses. Counts, actual PTS/phase and block hashes must agree.

Specific fixture allowance for this unit: synthetic audio<=12s, with at most
tiny<=320x180 video if the existing source-probe contract needs both streams;
CPU fixture creation only, not production encoder or edited MP4 qualification.
No original R/S observations/probes/full hashes or any guarded-runner rerun.
No300s benchmark. Use pure fake boundaries for maximum-policy exhaustion.

Run covering pure/new audio-only tests and two existing combined HOST cases that
exercise the extracted shared helper; avoid repeating unrelated eight-case
video/reorder matrix, runtime/GPU/font/normalization/full-project gates. Scoped
Ruff/mypy/compile on the FOUR paths. Independent spec+quality review after a
fresh snapshot34 closes this unit.6a1 still owns exact-index audio feature/cache
equivalence and persisted-generation associations; no new old-v7/ASR authority.
