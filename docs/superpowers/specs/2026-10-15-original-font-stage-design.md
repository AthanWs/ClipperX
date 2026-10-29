# Original-byte font staging — adopted Task6b4 design

Status: ADOPTED technical design. Exact Task44 brief activation governs source work. Governed
by the adopted font-staging-and-visual-validation design, global constraints and
preflight. Human choice of originals plus rendered validation is final.

## Choice and scope

1. Extend `render_assets.py`: `_atomic_write` overwrites; publisher is PNG/ASS.
2. **Choose two new files:** bounded copy/integrity and staged registry.
3. General font cache/renderer framework: unnecessary scope and native work.

Later implementation owns only `clipperx/font_staging.py` and
`clipperx/tests/test_font_staging.py`; host wiring/activation is separate.

Preserve original bytes/internal names/style/face data. No internal renaming,
subsetting, alias fonts, derivatives, substitution, install, cloud transfer,
embedding/redistribution or new renderer. Hash filenames are allowed. Equality
preserves internal data; staging does not certify family/style/glyph support.

## Existing seams and dependency direction

`RenderAssetRegistry` freezes paths, not bytes. `resolve("font", id)` and
`compile_overlay_ass(..., assets=...)` already supply the path seam. `_font`
uses Pillow index 0 and rejects italic/oblique and nonbold subtitles; cmap uses
FontTools fontNumber 0. Preserve these engines, wrapping, serializer, title cap
46 and all authored effects/timing/z-order/escaping.

Dependency: `font_staging -> render_assets`, never the reverse. Existing registry
imports Pillow; staging calls no loaders. Do not use loadability, brand helpers,
`_atomic_write` or `publish_render_bytes`. No circular import/new registry API.

## Authority and exact API

Production expected hashes are pinned in the new module under authority
`approved-originals-v1`, sourced from the root-provided historical identities:

| Required ID | Expected original SHA-256, lowercase | Face index |
|---|---|---|
| headline_impact | d8bbdb72e3f930157824b2e59677d3a912d4512c14a948e6656a2e3fb0c935ad | 0 |
| subtitle_bold | e8f4e3baf6cc35fed6fcce3a540e8b39e8f6cda1d22a28f2ec8f526fef7a43f5 | 0 |

These expected identities are historical, not fresh evidence. Changed sources
fail; new production hashes require separate authority. No fresh-hash approval
or path/stat-only acceptance.

Records use `@dataclass(frozen=True, slots=True)`: normalize Paths/64-hex hashes,
detach tuples, validate IDs/counts/status/index/cross-field consistency. They
are evidence containers, not unforgeable capabilities.

```python
FontFileStamp:
    device: int
    inode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int
    link_count: int

StagedFont:
    asset_id: Literal["headline_impact", "subtitle_bold"]
    original_path: Path             # checked absolute registered source
    expected_original_sha256: str
    original_sha256: str            # digest actually read, not copied from policy
    original_stamp: FontFileStamp
    staged_path: Path
    staged_sha256: str              # independently re-read from staged file
    staged_stamp: FontFileStamp
    face_index: Literal[0] = 0
    # Read-only properties, not independent caller-controlled fields:
    measurement_path -> staged_path
    measurement_sha256 -> staged_sha256
    measurement_face_index -> face_index

FontStage:
    stage_dir: Path
    identity_authority: Literal["approved-originals-v1", "test-fixture"]
    fonts: tuple[StagedFont, StagedFont]  # required-ID order shown above
    registry: RenderAssetRegistry
    selected_file_identity_status: Literal["unverified"] = "unverified"

stage_original_fonts(registry: RenderAssetRegistry, *, stage_dir: Path) -> FontStage
verify_font_stage(stage: FontStage) -> None
```

Public entry has no hash/policy override. It calls private
`_stage_fonts(registry, *, stage_dir, expected_sha256, identity_authority)`.
Detach/validate exactly two expected IDs. Even this private entry must enforce
the pins for `approved-originals-v1`; synthetic hashes require `test-fixture`.
Public entry always uses pinned authority. Integrity checks production records
against pins; fixture verification proves only fixture integrity. Production
host must reject fixture authority. Private APIs are not a security boundary.

At most 64 input font IDs; require both IDs. Extra fonts are not opened/staged/
passed through: output fonts are exactly the two staged files. Extra IDs then
fail existing resolve. Copy brand/SFX mappings unchanged with no file IO or
qualification; preserve input registry.

`FontStage` is the in-memory manifest plus usable registry; no disk JSON/cache.
Exactly two records, four file paths plus stage_dir, each <=2048 characters;
fixed hashes/IDs/status, no blobs/arbitrary metadata. Stamp integers nonnegative,
size/link count positive, decimal integers <=32 digits: finite contents.

`measurement_*` is the required measurement input, not proof measurement ran
or renderer-selected identity. No `fonts_verified` flag.

## Directory and finite IO contract

Host supplies absolute local `stage_dir`, a fresh child of an existing private
host-owned per-job directory. Create only that child with exclusive
`mkdir(exist_ok=False)`. Reject every preexisting child. No reuse/resume,
overwrite/replace/cleanup/ancestor creation. Host reserves it through encode.

Reject relative, UNC/network/device paths, Windows alternate data streams and
special files. Windows paths must be ordinary drive-qualified paths; this does
not prove locality. Host owns mapped-drive/cloud-sync avoidance and private
permissions/ACLs; helper does not establish these from path syntax.

Check source/stage-parent ancestors via lstat: <=32 components/path, <=2048
characters. Reject symlink/reparse components/junctions and nonregular fonts.
Resolve checked paths; reject source containment in stage and case-normalized
aliases or equal nonzero `(st_dev, st_ino)` between sources. Fail if usable file
identity is unavailable. Check containment structurally, not string prefixes.

Source hardlinks are allowed (Windows originals may have them): record stable
link counts/stamps and hash bytes; no source-isolation claim. Stage link count
must be 1 and object IDs distinct from both sources/other stage. Recheck at
readback/integrity; observed stage hardlinks/reparse/aliases fail.

Fixed bounds: two sources, each 1..16 MiB, total <=32 MiB; chunks <=64 KiB.
Read <=maximum+1 bytes; reject overflow/stat-size disagreement. No unbounded
read_bytes, directory enumeration, retries, recursion or font parsing. <=32
ancestor stats per checked path; two sources, one directory, two staged paths.
Each read adds pre-open lstat, fstat before/after, post-close lstat; reuse stamps
for alias checks. Finite IO is not a timeout against a stuck filesystem. No
threads/children/watchers.

Private `_read_checked(path, *, max_bytes) -> tuple[bytes, FontFileStamp]` uses
context-managed binary handles and matching pre/open/post stamps. Check path
components before opening. Positive max_bytes <=16 MiB; tiny direct tests use
64 bytes. Stamps detect observed changes; SHA-256 establishes byte identity.

## Ordered staging and failure behavior

1. Validate registry IDs, expected policy, all paths/limits/aliases and source
   sizes. Read each source once with `_read_checked`; hash real bytes and compare
   to pinned expectation. No output directory is created on a preflight failure.
   Keep the accepted two byte strings (total <=32 MiB) only for this call.
2. Exclusively create stage_dir. For each required ID, choose
   `<asset_id>-<expected_sha256>.ttf` and create with exclusive binary `xb`.
   Never open with truncation. Copy the accepted bytes with writes <=64 KiB;
   require successful complete writes, flush and fsync, close the handle.
   There are exactly two output files and <=32 MiB output bytes.
3. Independently reopen/read/hash both staged files with the bounded reader.
   Require byte count/digest equality with the verified original and expected
   identity, required containment, no reparse and no hardlinks/aliases.
   Do not hash only the original buffer and label it a staged-file observation.
4. Re-read/hash both original sources again after both copies; require original
   stamps and digests unchanged. Recheck stage directory identity and path
   components. Return FontStage only after every check succeeds. Construct its
   registry directly from the two checked staged paths and untouched other maps.

Staging: six read passes <=96 MiB+6 sentinel bytes; writes <=32 MiB. Release
readback buffers per file; buffers including read assembly <=64 MiB payload,
excluding bounded IO chunks/overhead. No manifest blobs/growing concatenations.

`ValueError` covers invalid policy/identity/structure/integrity; filesystem
`OSError` preserves cause. Include phase/ID/stage_dir. Failure retains partial
artifacts, closes handles and returns no success/registry. No deletion/repair/
retry; host owns later exact-directory cleanup. Retry needs a fresh child;
consumers must await success.

`verify_font_stage` revalidates structure, pinned production authority (when
applicable), registry-to-manifest equality and directory containment/components;
then re-reads/hashes both sources and both stages, compares recorded stamps,
lengths and digests, and rechecks alias/link rules. It has four file passes,
<=64 MiB + four sentinel bytes, no writes and no newly approved hash values.
Missing/changed source fails even if its old staged copy still matches. Same
bytes with changed object/stamp fail conservatively. Source or stage corruption
must not be silently repaired or accepted from a new digest.

Checks cannot defeat every replacement race, including before pathname-based
engine opens. Frozen paths are not immutable storage/locks or universal TOCTOU
protection. Require trusted host ownership/exclusive lifecycle; hostile concurrent
filesystem mutation is outside guarantee. No native handle broker.

## Later host contract; separate unexecuted gates

Stage before font loadability/glyph/measurement/wrap. Require production
authority; verify immediately before measurement and explicitly pass SAME
`stage.registry` to `compile_overlay_ass`/`build_overlay_ass`, never defaults.
Existing cmap/font/wrap/compile use these files at index 0. Record actual
measurement separately; no Pillow/FontTools duplication.

Verify immediately before encode and afterward before acceptance. Every
`OverlayCompilation.font_paths` entry must match staged registry (unused fonts
may be absent). Later host uses private stage in existing approved provider
setup; helper neither configures nor proves renderer choice. Retain files
through all consumers; host owns cleanup.

Actual-source/stage qualification and independent decoded validation are separate
unexecuted gates. Distinguish original/staged/measurement identity, scoped
family/provider/fallback logs, selected identity unverified, and decoded readable
bold/upright style, fit/wrap/safe regions/colors/outline/active word/timing/z-order.
Unsupported glyphs/unexpected visible fallback fail or remain unresolved. Pixels
or name logs never prove selected-file hash.

## Tiny filesystem TDD and scoped verification proposal

Complete positive recipe, only a new pytest `tmp_path`: create `sources` and
`job` children; write `b"fixture-impact-v1"` and `b"fixture-arialbold-v1"` to two
source files (invalid fonts, intentional byte fixtures). Compute SHA-256 from
those literal bytes. Build real RenderAssetRegistry with required paths plus
nonexistent extra-font/brand/SFX paths. Call real `_stage_fonts` with
`test-fixture`, these hashes and nonexistent `tmp_path / "job" / "fonts"`.
Assert real re-read copies/hashes/counts, index/status, unchanged originals,
exactly two font outputs, unchanged brand/SFX mappings and detached frozen
records. Run real `verify_font_stage`. No decoder. Fixture injection replaces
only expected identity; every positive guard/read/copy/hash remains real.

Exact proposed tests (all confined to newly created temp fixtures):

- `test_stage_fixture_copies_and_rehashes_two_originals_into_same_registry`
- `test_stage_preserves_brand_sfx_and_excludes_extra_font_paths`
- `test_public_stage_rejects_fixture_hashes_before_directory_creation`
- `test_approved_authority_rejects_arbitrary_expected_hashes`
- `test_stage_rejects_missing_required_empty_and_wrong_hash_sources`
- `test_bounded_reader_rejects_65_bytes_at_64_byte_limit`
- `test_stage_rejects_existing_directory_without_touching_its_files`
- `test_stage_rejects_source_path_alias_and_source_hardlink_alias`
- `test_stage_rejects_symlink_or_reparse_components_when_supported`
- `test_stage_failure_retains_partial_files_and_returns_no_registry`
- `test_stage_rechecks_source_after_copy_and_rejects_changed_bytes`
- `test_verify_rejects_changed_or_missing_source_and_changed_stage`
- `test_verify_rejects_replaced_equal_bytes_and_staged_hardlink`
- `test_manifest_detaches_input_and_rejects_registry_face_status_mismatch`

Negative recipes: corrupt one staged byte, remove fixture source, replace stage
with equal bytes/new inode, or create temp-only hardlinks/symlinks (skip if
unsupported, no elevation). For deterministic failure/source-change only, wrap
the real exclusive-copy helper: after first real copy raise OSError or mutate
source fixture. Never stub positive validators. Assert retained partial files,
unchanged originals except deliberate mutation, and released handles/reopening.

Later adopted TDD gate: failing new tests, minimum helper, only
`pytest -q clipperx/tests/test_font_staging.py`, then scoped lint/types on these two files.
No old asset/overlay HOST suites, real fonts/brand/media, native rendering,
subprocess/thread/model/network or actual renderer qualification. Not executed.

Self-review: no fresh-hash approval, selected-file claim, original overwrite,
partial success or new renderer/framework. Bounds/alias/race limits explicit.
Remaining gates: real host ownership/locality and actual original/decoded checks.


## Controller adoption clarifications — governing later activation44

Root FULL-read13956B original proposal SHA
6C64D51F7173FC6B364A6E50F074785E6B5C75D05C6F5DD4BD7199054516E552.
This adopted copy supersedes its proposal/future wording for the exact TWO-new-
path unit; it does NOT authorize a source writer until Task43 independent closure
and exact Task44 brief/absence snapshot/activation. No source/font/media/native
proof occurred during planning. Human's original-font/rendered-validation choice
is closed, no font mutation or selected-file attestation is reintroduced.

Exact paths are clipperx/font_staging.py and clipperx/tests/test_font_staging.py.
Stage exposes the exact two public functions and THREE frozen records above.
Add one REQUIRED field to FontStage: stage_directory_id: tuple[int,int], the
actual (st_dev,st_ino) observed for the exclusively created directory. Store only
usable bounded nonzero object identity; compare it at readback and verify. Do
not compare directory mtime/size to before child writes. This records the known
leaf identity without claiming every ancestor is immutably locked. Record font
file stamps exactly as specified; conservative same-bytes/stamp-change rejection
is an integrity policy, not a claim that every benign metadata change is damage.

Production pins are exact lowercase hashes above and NEVER learned from current
source bytes. Both public and private approved-originals-v1 routes enforce them.
The explicit private test-fixture route is only a tiny synthetic-byte test seam;
it returns identity_authority='test-fixture', never an approved style or production
manifest. No public override, loader/cache, new settings/provider/UI or source
registry mutation. Future production host must reject fixture authority before
using the stage. No need to embed pin provenance copies in every new record.

Also reject stage_dir equal to or beneath EITHER registered source's parent,
using resolved case-normalized component-aware containment before mkdir. This
keeps the default Windows Fonts locations and synthetic original directories
out of destination scope; do not make a broad directory write just because a
host supplied its path. The host still must provide a fresh child of an existing
private local job directory and avoid remote/mapped/cloud-synced locations.
Syntax/reparse checks do not establish filesystem locality or ACL/privacy.

max16MiB/source,32MiB accepted original buffers/outputs,64KiB read/write chunks,
32components/path,2048characters/path and32digit stamp scalars are binding.
Check stat/cap before buffer growth and retain exact source/stage object/byte
observations. Normalize expected hashes only AFTER checking type/length/hex.
Do not copy public whole-registry arbitrary fonts; require both IDs, <=64input
font IDs, return only the two staged fonts and untouched detached brand/SFX maps.
No file IO on extra fonts/brand/SFX. Scope original vs staged vs required
measurement input vs actual measurement/renderer facts remains explicit.

Bound untouched mapping projection BEFORE detachment too: each input registry
table has<=64entries, safe asset IDs<=64characters and path text<=2048characters;
reject unknown/iterator-like or bypassed malformed values before copying them.
The two source/stage paths require the full absolute/local/path-component
checks; untouched brand/SFX paths retain their existing values, including any
relative-path meaning, with no extra loadability/locality claim. These finite
count/text bounds prevent unbounded registry cloning without a second registry
or a generic JSON serializer. They do not create file IO on those assets.

If failure occurs after exclusive directory/file creation, retain those exact
job-private partial artifacts as evidence, close handles, return no stage. No
rollback deletion, overwrite, retry/resume or _atomic_write reuse. Error paths
must not obscure the primary failure with another handle cleanup exception;
include useful phase/ID/path context while retaining OSError cause/semantics.
If a close itself fails, attempt other already-open owned closes, retain the
primary error with cleanup failure context and never claim guaranteed resource
retirement from an exception. This helper does not issue a HOST/native own0
certificate or authorize another render after uncertain filesystem cleanup.
No promise that fsync/path operations have a hard time bound. Explicit bounded
payload/pass counts are not RSS, whole filesystem security or hard cancellation.

Future manifest is a TEMP-FILESYSTEM unit gate: real reads/copies/hash/stat only
inside new pytest temp fixtures and their needed path ancestors; no real font
bytes/load/brand/media/child/thread/network/native custom adapter/render. Negative
symlink/hardlink cases may skip ONLY actual unsupported filesystem capability,
with exact reason reported; do not skip positive copy/integrity behavior or
silently count skips as proven. All positive guards/copy/hash remain real.
The original14named tests plus a stage-under-source-parent rejection test and a
replaced-leaf-directory identity rejection test are the exact planned manifest.
Normal Python test runner/scoped static/compile are allowed, not subprocesses
inside tests. No old asset/overlay testfile execution or full repository suite.

This helper does not wire production overlay/preflight/encoder/metrics. Laterhost
must verify approved stage before measurement, encode and acceptance, pass SAME
registry and compare compilation font paths. Actual loadability/cmap/text style/
readability/wrap/safe-area/colors/active-word/timing/z-order and selected-file
identity='unverified' stay separate. No qualification is inferred from copying.

## Windows timestamp representation clarification during44 implementation

Synthetic fixture diagnostics found path lstat and descriptor fstat can report
different raw st_ctime_ns for the SAME unmodified Windows file, while their
st_birthtime_ns values agree. Root used Context7 /python/cpython then checked
exact configured-version primary source: CPython v3.12.6
[fileutils.c](https://github.com/python/cpython/blob/v3.12.6/Python/fileutils.c#L1027-L1045)
populates creation and metadata-change fields separately; its fstat route returns
those values. The
[path stat compatibility step](https://github.com/python/cpython/blob/v3.12.6/Modules/posixmodule.c#L2004-L2015)
replaces ctime with birthtime. This explains the observed API representation
disagreement, not a font mutation or a runtime-binary qualification.

The adopted record had not required raw st_ctime_ns specifically. Define its
ctime_ns semantics explicitly: on Windows WITH st_birthtime_ns, canonicalize
ctime_ns to that bounded integer creation timestamp for BOTH path/descriptor
stamps; otherwise use the bounded raw st_ctime_ns (older Windows/non-Windows).
Never round seconds/floats or silently accept malformed/missing present values.
Keep every device/inode/size/mtime/link/identity/hash comparison unchanged.

In ADDITION, compare raw path-API ctime_ns before/after the read with each other,
and raw descriptor-API ctime_ns before/after with each other. Thus canonicalizing
the cross-API field does not discard an observed within-API metadata change.
Do not require raw path ctime to equal raw descriptor ctime on this platform.
This remains finite stdlib observation, no custom native call/new record field.
No universal metadata-change detector or protection against all races is claimed.

Cover the actual synthetic discrepancy positive plus canonical-field semantics
and within-API raw-ctime-change rejection inside the existing16test names. Keep
real positive reads/copies/hashes, same-types/identity checks and Python3.10
fallback. This clarifies an unspecified platform representation; it does not
waive a required byte/stamp guard or introduce a human/style decision.

Negative forbidden-path tests MUST install fail-only stat/lstat/resolve/open
sentinels BEFORE exercising UNC/device/network syntax, including initial RED.
No actual remote path resolution/access is part of this temp-file gate. A missing
syntax guard must fail at a sentinel, not contact a remote path. Report any
earlier inadvertent path attempt and observed outcome exactly without recontact,
network diagnostics or an unsupported no-network/authentication-outcome claim.

