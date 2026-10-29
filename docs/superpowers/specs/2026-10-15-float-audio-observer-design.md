# Versioned float audio observer — Task6b2 /42

ADOPTED after independent review41. This separately scoped source ownership is exactly
`clipperx/render_clock.py`, `clipperx/source_decode.py`,
`clipperx/tests/test_render_clock.py`, `clipperx/tests/test_source_decode.py`.
Pure/fake implementation gate only; no actual runtime/Windows41/media campaign.

Read controller-notes first: new float48k; old s16 intact; interval mapping later.
No mapper, prepared bus, normalization, mixer, compiler or future-owner API here.

## 1. Public boundaries and supported signal

`observe_audio_window`/`AudioDecodeLimits` live in source_decode. Preserve all old
s16/combined/streamed signatures, policies, literal argv, limits and receipts.

NEW in source_decode:

```python
@dataclass(frozen=True, slots=True, kw_only=True)
class FloatAudioDecodeLimits:
    # Exact fields/default ceilings are specified in section 3.
    def record(self) -> dict[str, object]: ...

@dataclass(frozen=True, slots=True, init=False)
class DecodedFloatAudioWindow:
    receipt: SourceFloatAudioDecodeReceipt
    pcm: bytes
    def __init__(self, receipt: SourceFloatAudioDecodeReceipt,
                 pcm: bytes | bytearray | memoryview) -> None: ...

def observe_float_audio_window(runtime: RenderRuntime, origin: SourceTimeOrigin,
    window: Window, *, audio_policy: str,
    limits: FloatAudioDecodeLimits) -> DecodedFloatAudioWindow: ...
```

Limits are mandatory; other types exist or are defined below.
Only `render48k-f32-mono-v1` / `render48k-f32-stereo-v1` are legal new policies.
Require original rate44100/48000, channels1/2, format flt/fltp; every input tag
must match. Preserve channels; no rematrix. Retain existing origin/probe's one
video+one mixed-audio restriction, without decoding video or adding a probe.

Reject integer/double/unknown formats before launch and on tags. One continuous
swr48k packed-float conversion: no integer stage, clipping, dither, compensation
or fallback. Finite>1/sub-s16 values remain legal; actual behavior needs HOST proof.

Finiteness scope is ONLY emitted f32: every little-endian four-byte word must
satisfy `(word & 0x7f800000) != 0x7f800000`, before retention/success. This rejects
NaN/Inf and preserves subnormals/signed0/>1 without another PCM copy. Input
ashowinfo cannot independently prove finiteness of uncaptured pre-resampler
bytes. No astats/filter/parser addition or broader finiteness claim.

## 2. Pure public operations, identities and exact envelope

NEW in render_clock; imports remain pure, no source_decode/render_runtime import:

```python
def float_audio_decode_argv(executable: str, origin: SourceTimeOrigin,
    start: Fraction, end: Fraction, audio_policy: str) -> list[str]: ...

def validate_float_audio_limits(limits: Mapping[str, object], *,
                               window: bool = False) -> None: ...

def source_decoder_identity(runtime: Mapping[str, object]) -> str: ...

@dataclass(frozen=True, slots=True, init=False)
class SourceFloatAudioDecodeReceipt:
    origin_id: str
    origin_decoder_id: str
    operation_id: str
    policy: str
    runtime_json: str
    payload_json: str
    def __init__(self, origin: SourceTimeOrigin, payload: Mapping[str, object], *,
        runtime: Mapping[str, object], limits: Mapping[str, object]) -> None: ...
    def record(self) -> dict[str, object]: ...
    def to_json(self) -> str: ...
    @property
    def sha256(self) -> str: ...
    @classmethod
    def from_json(cls, text: str, *, origin: SourceTimeOrigin,
        runtime: Mapping[str, object], limits: Mapping[str, object]
        ) -> SourceFloatAudioDecodeReceipt: ...
    def validate_origin(self, origin: SourceTimeOrigin, *,
                        runtime: Mapping[str, object]) -> None: ...
```

Constructor/loader limits is TRUSTED FloatAudioDecodeLimits.record(), never the
receipt's self-selected limits. Retain only canonical detached JSON/plain strings.

Runtime EXACT keys `runtime_id ffmpeg ffprobe ffmpeg_sha256 ffprobe_sha256`:
bounded nonempty plain strings,64-uppercase-hex runtime hashes; pure paths never resolved.
Guard record/origin before hashing. Observer derives it from supplied runtime
after existing `_runtime_identity` CPU/pair/device checks, `_verify_pins` and
source fingerprint. Loader requires exact equality with supplied current record.

Move unchanged DECODE_POLICY to render_clock, import/re-export in source_decode.
NEW source_decoder_identity(runtime) returns EXACT existing digest of canonical
`{"runtime":runtime_id,"ffmpeg":ffmpeg_sha256,"ffprobe":ffprobe_sha256,
"policy":DECODE_POLICY}`. `_runtime_identity` retains its pinned checks and
delegates only digest computation. Pure HOST identities never satisfy host pins.

NEW fixed `FLOAT_AUDIO_DECODE_POLICY` value:
`audio-only;pcm-f32le;48k;swr-fltp;copyts;no-healing;v1`.
Keep origin.decoder_id unchanged. Require it equals
source_decoder_identity(expected_runtime). NEW operation_id is digest(canonical
of EXACT keys `origin_decoder_id runtime policy audio_policy`), where runtime is
the above complete runtime record, policy is FLOAT_AUDIO_DECODE_POLICY and
audio_policy is the single audio record's named new policy. The operation ID
binds pinned pair/options separately from origin's legacy suite identity.
Source identity remains stat-derived, not full-content authentication.

Envelope EXACT keys/values:

```
version: 1 (plain int, not bool)
kind: "audio_only_float"
scope: "window"
origin_id: origin.identity
origin_decoder_id: origin.decoder_id
operation_id: computed above
policy: FLOAT_AUDIO_DECODE_POLICY
runtime: exact runtime record
payload: exact payload below
sha256: digest(canonical(all preceding envelope fields))
```

Payload EXACT keys `windows limits audio argv`, retaining the narrow existing
audio-only shape. One windows entry `[source_start,source_end]`, one audio record,
one actual executed argv list. `limits` contains all fields from section3 plus
`decoder_start decoder_end`. Every rational is canonical `numerator/denominator`
text, including zero as0/1. No float/exponent/decimal-string times in this wire.

Audio record EXACT keys:
`policy input output samples bytes pcm_sha256 sample_grid_origin source_phase coverage`.
`input` and `output` are nonempty lists. Shared block keys:
`n pts time_base rate channels samples format checksum stage`.
Output adds EXACT `bytes sha256`. n is consecutive starting0; samples>0;
time_base=1/rate; pts plain bounded signed integer; input stage=cxain and
output stage=cxaout. Input format flt/fltp as bound above. Output format=flt,
rate48000, unchanged channels. Output bytes=4*channels*samples, SHA256 lowercase
64hex, checksum8 uppercase hex. Input checksum is metadata only; output checksum
must equal `zlib.adler32(actual_payload, 0)` as current observer, NOT default1.

Within each stage, next_pts=prior_pts+prior_samples with unchanged rate/channel/
format. Total samples/output bytes equal sums; aggregate PCM hash is of actual
output concatenation. `sample_grid_origin=first_output_pts/48000`,
`source_phase=sample_grid_origin-origin.origin`. Coverage is one record with
EXACT `window status`; status covered iff observed full interval covers the
requested window under common origin, otherwise unavailable. Unavailable may
be a valid observation; the later preparation decides eligibility. No fabricated
silence, lower trim or altered window to make coverage covered.

Reuse validate_audio_conversion/_validate_audio_origin with unchanged<=1 sample
conversion allowances; these do not prove grid mapping/signal fidelity. Constructor
validates full origin/argv/limits immediately; validate_origin repeats current
association. Receipt is metadata consistency, not authenticated execution.
DecodedFloatAudioWindow checks exact receipt type and PCM nbytes/cap BEFORE copy,
detaches bytes, invokes existing `_validate_pcm`, checks finite words/final offset.
Only that wrapper establishes actual byte/finiteness association, not receipt alone.

## 3. Finite ceilings and guards BEFORE work

NEW defaults and hard maxima; every numeric field may only tighten:

| Field | Default/ceiling |
| --- | --- |
| preroll, postroll | Fraction(1,2), Fraction(1,4), respective maxima |
| max_window | Fraction(180), INCLUDING effective margins |
| timeout | 45 seconds |
| max_blocks | 16384 PER input/output stage |
| max_samples | 9000000 PER channel/stage |
| max_block_samples | 65536 PER input/output block |
| max_decoded_bytes, max_retained_bytes | 72000000 each |
| max_log_bytes | 16000000 |
| max_metadata_bytes | 8000000, COMPLETE canonical envelope |
| io_chunk_bytes | 65536 |

180s accommodates179s+0.75s margins where source permits. These are NEW unit
caps, not actual campaign/RSS guarantees. Count/byte caps are independent
failure limits, not permission to expand duration. Integers positive plain int,
no bool; Fractions plain/bounded, margins>=0, max_window>0. record() uses rational.

Use REAL39 `_stream_fraction/_stream_int/_stream_rational/_stream_guard/
_stream_keys/_stream_origin` and existing canonical/digest/_unique_pairs. Never
send untrusted text to exact/Fraction first: `1e999999999` fails rational syntax
before exponent work. Require actual Window and bounded plain Fraction fields
before arithmetic (its earlier construction is outside this entry); same for
argv bounds. `_stream_origin` precedes origin.record()/identity. No cloned engine.

Pre-child: validate actual limits; guard origin/runtime/window/policy; supported
format; exact effective interval=max(0,start-preroll)..min(duration,end+postroll)
within180s; budget argv/envelope; then real runtime/source checks and Popen.
Pure receipt guards precede canonical/hash. Presented limits<=trusted limits
FIELD BY FIELD; decoder bounds must equal derived bounds. Effective output cap:
min(max_decoded_bytes,max_retained_bytes,4*C*max_samples). Enforce metadata/log/
count independently during production, not only at finalization.

Extract existing39 SourceStreamFinalization.from_json's bounded JSON preflight/
parse into ONE private helper used by old39 and new float loader, preserving39:
plain str/UTF8 cap and depth<=10 BEFORE parse; bounded integer tokens, rejected
float/NaN/Infinity and duplicate keys. Then guard/keys/trusted limits and ENTIRE
envelope cap before canonical/hash. Every time uses `_stream_rational`, integer
uses `_stream_int` before multiplication. No second parser or s16-loader rewrite.

Reserve guarded envelope/argv/coverage/limits, maximum-width totals and hashes
before launch. Metadata producer alone charges `_stream_guard(record)` plus list
separator before retaining/queueing either stage; output bytes/hash placeholders
are charged before payload read. Consumer may replace them, never add uncharged
records. Final whole-envelope guard remains mandatory. Keep<=16384-byte tag
lines/log cap; guard integer token length BEFORE int(), then `_stream_int`.

Read chunks<=io_chunk_bytes; check block samples/bytes and remaining decoded/
retained cap BEFORE allocation/read. Retained cap is output payload, NOT RSS:
bytearray→bytes can double PCM plus block/queue/pipe overhead. Metadata cap is
serialized bytes, not Python/FFmpeg heap. No RSS/hard-cancellation claim.

## 4. Exact shared integration, no second decoder

Private immutable output-contract fields: format, codec, muxer, bytes_per_scalar,
policy→rate/channels table. Exactly two fixed instances: old s16/new float48k.
Old audio_format/audio_decode_argv accept only s16; new entry only float. No
caller-defined filter/codec. Share builder while retaining old literal argv.

New float graph EXACT form, with layout mono/stereo and original rate substituted:

```
atrim=end=<decoder_seconds(origin.to_raw(end))>,asettb=1/<input_rate>,ashowinfo@cxain,aresample=48000:async=0:osf=flt:tsf=fltp,aformat=sample_fmts=flt:channel_layouts=<layout>,asettb=1/48000,ashowinfo@cxaout
```

Use current decoder_input_argv prefix, selected0:<audio.index>, -vn/-sn/-dn,
-af graph, -c:a pcm_f32le -threads1 -f f32le pipe:1. No `first_pts`, lower
atrim, asetpts, source apad, async compensation, video filter/map or new flags.
Upper-only trim retains predecessor PCM/g. Compare ENTIRE actual argv with the
shared builder using expected executable/origin/bounds/policy, not substrings.

Share `_validate_audio_records` and `_validate_audio_argv` through a PRIVATE
contract argument defaulting to old s16, or one internal extracted core called
by strict wrappers. New wrapper performs bounded39 numeric/schema guards before
shared semantic checks, so existing exact() uses see only validated rational
text. New source format gate/input count and block ceiling are applied; old
policies remain exactly as before. `_validate_audio_origin`, conversion arithmetic,
canonical/digest and `_validate_pcm` remain single implementations.

In source_decode add a private keyword `_audio_contract` defaulting to the s16
instance to `stream_payloads`, `payload_stream` and `_audio_window`; extend their
limits unions with the defined FloatAudioDecodeLimits. Use new contract ONLY
from observe_float_audio_window. Default paths/old caller signatures behave as
before. `_audio_window` recognizes both materialized audio limit classes when
selecting max_blocks, retained-byte rules and payload_stream. The existing
`_owner` streamed39 branch explicitly rejects new float contract; no39 streaming
or Windows owner expansion belongs in this unit.

For materialized float operation:

1. Entry establishes/budgets origin/runtime/policy/limits and shared argv.
   `payload_stream` wraps the SAME `stream_payloads` generator and diagnostic
   lifecycle, passing private contract, expected audio clock and metadata budget.
2. Metadata producer uses existing parse_audio/tag grammar with bounded integer
   token guard for this contract. Input tags must match selected origin and
   supported flt/fltp; output tags must be flt48k/same channels. Charge both
   stages before retention. Sequence/gap/overlap and count caps stay shared.
3. Compute output payload size through a NEW private checked helper taking
   AudioTag+fixed contract, rather than trusting current AudioTag.byte_count's
   hardcoded2. Keep that property's old semantics for old users. Both normal
   materialized paths use the shared helper, preserving s16 results; streamed39
   path stays unchanged. Check exact size, checksum and finite float bits before
   `_audio_window` retains bytes. Reject trailing unassociated stdout bytes.
4. `_audio_window` retains its actual observed input/output records, phase,
   coverage/conversion checks and yield INSIDE payload_stream. New wrapper
   constructs SourceFloatAudioDecodeReceipt then DecodedFloatAudioWindow within
   that context, so post-EOF association errors retain bounded diagnostics and
   cleanup semantics. Check current source fingerprint again before returning.

Extract common real checks from `_check_observation_source`; contract-specific
format guard only. Same `_runtime_identity`/decode_requirements CPU pair/filters.
Receipt errors ValueError; child/protocol errors existing DecodeFailure; no retry.

Current stream_payloads owns Popen/metadata thread/watchdog/cleanup; no claim it
satisfies Windows41's stronger guarantees. HOST work remains separately gated
by root's ownership acceptance; no new owner API or cancellation design here.

## 5. Complete small positive recipe for FUTURE constructor tests

Not executed. Future tests use these real specified constructors, no mocked
validators/compiler import. Synthetic HOST identities cannot pass real host checks.

```python
from fractions import Fraction as F
import zlib
from clipperx.render_clock import (
    StreamClock, SourceTimeOrigin, SourceFloatAudioDecodeReceipt,
    source_decoder_identity, float_audio_decode_argv, digest,
)
from clipperx.source_decode import FloatAudioDecodeLimits, DecodedFloatAudioWindow

runtime = dict(runtime_id="HOST", ffmpeg="O:/HOST/ffmpeg.exe",
               ffprobe="O:/HOST/ffprobe.exe",
               ffmpeg_sha256="A" * 64, ffprobe_sha256="B" * 64)
origin = SourceTimeOrigin("O:/HOST/float.nut", 16, 1, "0123456789abcdef",
    F(0), F(1), (
        StreamClock(0, "video", F(1, 60), F(0), F(1), width=1, height=1),
        StreamClock(1, "audio", F(1, 48000), F(0), F(1),
                    sample_rate=48000, channels=1, sample_format="flt"),
    ), source_decoder_identity(runtime), {})
limits = FloatAudioDecodeLimits(preroll=F(0), postroll=F(0), max_window=F(1),
    timeout=1, max_blocks=2, max_samples=8, max_block_samples=8,
    max_decoded_bytes=32, max_retained_bytes=32, max_log_bytes=4096,
    max_metadata_bytes=8192, io_chunk_bytes=16)
policy = "render48k-f32-mono-v1"
pcm = bytes.fromhex("00000000 0000803f 000080bf 00008035")  # 0,1,-1,2**-20
checksum = f"{zlib.adler32(pcm, 0):08X}"
block = dict(n=0, pts=0, time_base="1/48000", rate=48000, channels=1,
             samples=4, format="flt", checksum=checksum, stage="cxain")
audio = dict(policy=policy, input=[block],
    output=[dict(block, stage="cxaout", bytes=16, sha256=digest(pcm))],
    samples=4, bytes=16, pcm_sha256=digest(pcm), sample_grid_origin="0/1",
    source_phase="0/1",
    coverage=[dict(window=["0/1", "1/12000"], status="covered")])
payload = dict(windows=[["0/1", "1/12000"]],
    limits=dict(limits.record(), decoder_start="0/1", decoder_end="1/12000"),
    audio=[audio], argv=[float_audio_decode_argv(runtime["ffmpeg"], origin,
                                              F(0), F(1, 12000), policy)])
receipt = SourceFloatAudioDecodeReceipt(origin, payload, runtime=runtime,
                                        limits=limits.record())
decoded = DecodedFloatAudioWindow(receipt, bytearray(pcm))
loaded = SourceFloatAudioDecodeReceipt.from_json(receipt.to_json(), origin=origin,
                                runtime=runtime, limits=limits.record())
assert decoded.pcm == pcm and loaded.to_json() == receipt.to_json()
assert decoded.receipt.origin_decoder_id == origin.decoder_id
assert len(decoded.pcm) == 4 * 1 * 4
```

Upper trim0.000083 versus4 samples=1/12000 fits the existing one-input-sample
edge allowance; no lower trim. Assert the complete envelope<=8192, never disable
its cap. This fixture establishes no real decode, input finiteness or final A/V.

## 6. Future verification manifests, separate from this assignment

Pure/fake: real recipe/detachment/roundtrip; exact-schema/cross-schema failures;
runtime/origin/operation/argv conflicts; format/rate/channel/count/hash/checksum;
finite-bit faults and preserved signed0/subnormal/>1; unavailable coverage and
phase/source-bound contradictions. At/above caps fail before relevant hash/
allocation/child work. Include exponent1e999999999, huge integer tokens, nesting,
duplicate keys, floats/constants and stricter HOST limits. Cap ENTIRE envelope.

Fake boundary tests use actual shared generator/context/helper: one audio/zero
video, short/partial/trailing bytes, bad tags/checksum/nonfinite output, limits,
post-EOF receipt/source mutation and diagnostics. No real OS/threads. Cover old
s16/combined/39 argv/receipt/shared-JSON behavior; no compiler/full-project suite.

Later HOST manifest, separately approved: <=2s48k mono/stereo float with>1,
sub-s16/channel markers, nonzero origin and preroll against independent complete
float reference; <=3s44.1k-fltp→48k with independent event/count/phase/EOF proof.
Require actual flt/fltp source; any tiny fixture video must be declared separately.
No fixture now, R/S repeat or179s benchmark. Broader source support/prepared
mapping/normalization/delivered A/V remain separate gates.

Consulted Context7 [official FFmpeg astats docs](https://ffmpeg.org/ffmpeg-all.html#astats)
only for input-finiteness scope. Controller chose output-byte checks, no astats.
No runtime proof. Exact activated Task42 brief/snapshot/review govern implementation.

## 7. Controller adoption clarifications — governing this unit

Root FULL-read the complete19862B background proposal, verified its SHA256
668DC7F9AC338DD191FC7F02E2236C2EAD6224D20DBA0F92650511964BA40A93, and
inspected the actual StreamClock, runtime identity, shared JSON preflight and
materialized audio helpers. This standalone adopted copy supersedes that
unadopted proposal for this unit. Public signatures and all exact fields above
are binding; private extraction/context arguments within the FOUR paths may
carry the selected StreamClock, limits and metadata ledger as needed, without
new owner/framework/public mode. Report those internal seams.

Runtime hashes use UPPERCASE64hex exactly like current RenderRuntime and its
pin checks; PCM/receipt/digest hashes remain lowercase as already stated.
The fixture uses A/B uppercase accordingly. This corrects a background-proposal
case inconsistency before source implementation. source_decoder_identity must
preserve the existing digest byte-for-byte: for the current NATIVE_RUNTIME and
pinned uppercase hashes the expected decoder identity is
a6374863a6199d4bc26d98c33eb8c6b3a8809104340d9664c12424e81c9f2642.
Root calculated it from the actual old canonical four-field expression, without
executing a native/runtime command. Include independent literal regression;
do not compute the expected value via the new helper itself.

StreamClock.sample_format already exists; no schema expansion is needed there.
Limits live in source_decode, pure validators in render_clock. No imports from
compiler/audio_mix/Windows adapter or model/settings code. Reuse old output/tag/
PCM/argv/origin/conversion/diagnostic machinery through exactly TWO fixed wire
contracts, preserving old valid byte strings, public API and helpers/tests.
Existing AudioTag.byte_count stays old2-byte semantics; checked contract helper
selects4 bytes only for the explicit new float route. Streamed39 owner rejects
the new contract instead of accidentally enabling float streaming.

Only emitted packed float bytes are checked for finiteness. Input observed
format/PTS/checksum is not independent input-finiteness evidence. No astats,
new detector or extra decode. No float prepared-map/normalization/effects/renderer
integration is included. New finite bit check accepts signedzero/subnormal and
finite>1; no silent clipping/quantization/normalization.

Pre-child generated decoder argv must also have a positive representable CLI
duration: a positive Fraction that collapses to zero under existing microsecond
decoder serialization fails explicitly, not via epsilon/range editing. Old
builders/serialization remain unchanged. Preserve all requested declared values
separately from decoder-margin strings and use actual observed coverage.

In this implementation gate every process/metadata worker/watchdog/timer is
fake or step-driven; standard in-memory buffers/queues/events are permitted,
no actual Thread.start/Timer.start/Popen/native call, media/source probe or
guarded campaign. Positive fake producer may replace ONLY HOST source-stat/pin/
runtime checks and child/I/O/clock/scheduling boundaries, never whole new
receipt/parser/tag/PCM validators. Synthetic facts are not actual observations.

Ordinary source regression is the SEVEN inspected old pure nodes only, plus
new float nodes in the exact brief. In particular do not run the misleading
audio_only_fake_child_protocol_and_cleanup or old pipe/host tests (real children),
nor old streamed-owner nodes that start real Python threads. Full
test_render_clock remains a pure regression gate for the shared JSON extraction.
All historical test/helper bodies stay unchanged; add new names/fixtures only.

The final parent179s product constraint, authored windows/styles, original fonts,
one final encode/five sidecars and Task7 actual A/V/visual gates remain binding.
This unit's180s effective decoder ceiling is not a product-duration extension.
No actual float/Windows/throughput/phase/DSP/encoded/model/human qualification is
conferred. Independent task review precedes the next source unit.

