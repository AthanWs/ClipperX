# Bounded same-source HarfBuzz-FreeType recovery

2026-10-21. DESIGN ONLY. Root independently reviews this document and writes a task-sized plan before source/native work. No native/component acceptance or admission is implied.

## Decision

Use `O9=O:/Escritorio/ClipperX/.superpowers/sdd/upstream-motion/production-build-v9-hbft`, build ID `ffmpeg-9.0.2-clipperx-production-v9-hbft`. Preserve S3/S4/A5/A6/O7/O8, failures and partials. Rebuild ONLY HarfBuzz from accepted A4 commit `64e2b53235aa33b5bad5442449a092b5729be66a`, tree `2792379de069fcf35353475d9b300e1201ed4d4a`, with FreeType enabled against pinned A5 FT. Reuse O7 libraries and A5 FT/FriBidi; diagnose failed reuse before proposing broader work.

Task124 proves disabled FT/missing bridge/core-only configure probe. Fresh configure after bridge check is selected; header copying is insufficient, Make overrides add a contract, whole-stack rebuild lacks evidence. No protected edits/system HB/renderer change/disabled drawtext/acquisition/fallback.

Fresh Task125 delegation covers correction; Root disclosed ONLY accumulated configure1200->2400s, unchanged21600total/all other caps. This is delegated judgment, not separately typed numeric approval. Native/quality/media gates remain.

## Units and reuse

All following files are future implementation units:

| File | Interface/responsibility | Size guidance |
| --- | --- | --- |
| `O9/recovery.py` | Inert import/default NOT_RUN; `activate(request)`, authority, budget/predecessors, six stages, O9 consumer/publication checks | <=24KiB |
| `O9/hbft.py` | Inert helper; `plan(v5_lock,current,inputs)` returns six rows; `check_build(current,children)` returns check.files/proof; `selected_inputs(current,p,v5,hb_post)` replaces one HB | <=12KiB |
| `O9/recovery-lock.json` | Fixed contract below | <=16KiB |
| `.superpowers/sdd/task-125-run-reviewed-stage.ps1` | Narrow Task121 derivative; same public parameters, O9/six-stage identities | <=28KiB |
| `.superpowers/sdd/task-125-test-hbft-recovery.py`, `task-125-test-host.ps1` | Detached fake/source guards | finite focused files |

Load O8's actual pinned source/lock; preserve original modules/context. Reuse pure readers/Pin checks/host_elapsed/write_exclusive/finish_records. Call `_verify_old_history(c,p,old,current,original_o8_lock)` with ORIGINAL validated O8 lock and `c.load_context()` objects. No52KiB wrapper/ancestor-graph copy; size guidance never raises caps.

Reuse `p.task92_runner(current)`, original origin/freeze/receipt checks, `p.validate_link_policy,validate_map_inputs,validate_iconv_members,stage_outputs,copy_verified` and v5 static/component/map-member/PE/pair validators. HB constructors use `v5._meson_stage(v5_lock,O9,'hb',operation)`; NEW rows replace exactly one disabled freetype token with enabled and rename; originals remain unchanged.

O8 activation/layout/budget/predecessor/settlement/area/provenance/configure/inspect/publish bind O8: use narrow O9 equivalents. A6 `postcheck_native` FFmpeg-build hard-selects A5 HB; locally replace that input check, retain link/output validators. Adapt A5 disabled-FT graph/install rules explicitly for hb-ft.cc/hb-ft.h. No forged locks/globals/module roots/new runner.

Keep approved Python270104B/3470F7919170D235D7E6079691462C4B217745EC67EE612E745730E46D98F238, `-B -c`, cwdO7, WindowsA6-only PYTHONPATH, no user site/bytecode/PYTHONHOME/STARTUP. Preserve Task121 pre-import continuation SourceFileLoader/nonpackage/cache custody, normal import, actual O9/helper file/spec/code origins.

## Closed authority and record paths

`Pin={path,bytes,sha256}`: canonical absolute forward-slash path, positive non-bool bounded integer bytes, uppercase64hex expected SHA. Root externally supplies accepted identities after review; no discovery/hash-current-then-trust or invented future hashes.

Lock schema `clipperx-hbft-recovery-v1`, EXACT `schema,build_id,root,dependency_root,parent_root,original_recovery,helper,carry_anchor,carried,limits,stage_order,hb_recipe,correction`. Roots=O9/O7/A6. `original_recovery={source,lock}` pins accepted O8 recovery.py52641B/C742EE09579C1D25069B92312F406627D311AC02DB7DC199E953C03970F0743D and recovery-lock.json3148B/2372D60239750843D3BE1202B18B0918D04CC8F325DF9520A7213DF966984EAC. Helper=Pin(O9/hbft.py); stage_order=six names below. `hb_recipe={source_commit,source_tree,native_ini,options,children}`: A4 identities, Pin(A5/config/native.ini), original28HBoptions with ONLY freetype enabled, fixed child rows. `correction="same-source-harfbuzz-freetype-enabled;fresh-ffmpeg-configure"`.

External Root `.superpowers/sdd/task-125-hbft-carried-state.json`: schema `clipperx-hbft-carried-state-v1`, EXACT `schema,carried,limits,anchors,scope`; carried/limits=lock. Anchors EXACT role->Pin `delegation,diagnosis,root_failure,stage_book,a6_lock` select Task125 delegation, Task124 diagnosis/Root observation, Task123 stage book, A6 lock. `scope="historical debits included once in authoritative aggregate; group carry retained; failed O8 build consumed"`. Root observation binds failed O8 raw Pins; original validators bind older ancestry.

Root acceptance pins design/source/helper/lock/host/reviews externally. First charged host exclusively writes literal `O9/reviewed/recovery-admission.json`. Schema `clipperx-hbft-recovery-admission-v1`, EXACT `schema,build_id,root,source,lock,carry_anchor,carried,limits`, equal external activation/fixed lock. Runtime never authors expected Pins.

Keep `activate`'s nine fields `stage,source,lock,admission,predecessor,closure,host_origin_ns,host_pid,host_confirmed`; source/lock/admission fixed O9 paths above. Host external request stays first SIX fields; parameters remain `Stage,RequestBase64,AdmissionBodyBase64,ConfirmHost`. Only hbft_build requires first admission body/null predecessor; later stages forbid that body and require `O9/reviewed/<selected>.json`. Only publish requires `O9/reviewed/publication-closure.json`; other closure=null. No interface expansion.

Evidence=`O9/evidence/<stage>/`; raw files=`stages.json,<child>.json,<child>.log,postcheck.json,budget.json,completion.json`; failure=`validation.failure.json`. Operations=`.superpowers/sdd/task125-operations/<stage>-01/`, Root-created empty/single-use. Outer redirects outside. Retain Task121 launch/host-child/child.stdout/child.stderr/terminal/root.failure records with O9 identity. No partial reuse.

Readers retain same-stream type/size/hash/nlink/dev/inode/mtime pre/post checks and path recheck; reject reparse/hardlinks/duplicate/nonfinite/non-UTF8/extra fields. Caps: source1MiB; lock/closure/candidate/notice16KiB; admission/request64KiB; reviewed/post/inspection8MiB; budget/completion/failure4096B; logs/maps stagecap. New files ordinary one-link. Only fresh active raw records may lack external digest inside current transaction; predecessors require expected SHAs. No fallback.

Predecessor schema `clipperx-v9-hbft-predecessor-review-v1` keeps EXACT `schema,build_id,selected,admission_sha256,predecessors`; row EXACT `stage,stages_sha256,postcheck_sha256,budget_sha256,completion_sha256,outputs,settled_sha256`. Require ordered prior-stage prefix, completed statuses, cross-digests/provenance, unique confined outputs, current custody and Root settlement. Failure/partial records cannot become predecessors.

## Accounting and clock

Initial carried values: build6496.300s, logs2728961B, source179.784s, configure977.939s, pair523.345s, HB366.593s. Lock carried field names are `build_seconds,log_bytes,source_total_seconds,ffmpeg_configure_seconds,pair_build_seconds,hb_seconds`.

Root436ce0 reconciles original v3/v4 HB176.078+A5subset190.515=366.593 in immutable A6 `prior_cost.groups.hb`; Root976b6f checked all three A5 completions/budget SHAs. Retain original validators. Carry366.593 ONCE within2700, already included in6496.300;190.515alone would reset costs.

| Envelope | Cap | Initial remainder |
| --- | ---: | ---: |
| Aggregate incl host/prep/final work | 21600s | 15103.700s |
| Accumulated FFmpeg configure | 2400s | 1422.061s |
| HB dependency | 2700s | 2333.407s |
| FFmpeg+FFprobe pair | 7200s | 6676.655s |
| Inspection | 900s | 900s |

Original context limits stay exact/configure1200; active O9 differs ONLY in accumulated configure2400. Never mutate original context or feed active limits to its validators. Native configure child<=1200. Retain input/source/acquisition/per-child caps,2workers,64MiBstage/512MiBtotal logs,20GiBarea+20GiBfree,256MiBexecutables. Count seven roots once, reject aliases; no debit reset/acquisition.

Total left=21600-6496.300-prior O9 debits; group left subtracts fixed carry/prior O9 group debits. Allowance=min(total,group); publish uses total. Child timeout=floor(min(child ceiling,allowance-elapsed-1)); refuse<1. One conservative origin covers decoding/custody/import/admission/layout/children/final work. Debit hostelapsed+bounded1sfinal reserve,>=nativeelapsed. Recheck each child/write; failures/overruns remain charged.

Keep GetTickCount64()/0.015625/monotonictrue/adjustablefalse, compatible samples/first conservative origin; no QPC. Durable PIDcreation precedes stdin; stdin closed/both drains complete+disposed. Root reviews actual debit/final-write/scoped retirement; no OS hard-time/descendant guarantee.

## Six admissions and fixed children

Order EXACT `hbft_build,ffmpeg_configure,ffmpeg_build,ffprobe_build,inspect,publish`. Each returns stage_completed_only and never starts its successor. Original runner API stays `run_stages(stages,evidence,env,timeout_seconds,stage_log_cap,total_log_cap)`. Path aliases expand absolutely; descendant_cleanup_proven=false. Parse COFF per file.

HB creates O9 build/profile/cache/temp/config/prefix/bridge layout; copy exact pinned A5 native.ini to O9/config/native.ini. SIX singleton calls retain original runner: its one list-wide timeout/mkdir-exclusive evidence require separate child folders to clip configure/check between children.

| Child / evidence under hbft_build | Fixed argv constructor; ceiling |
| --- | --- |
| hbft_configure / children/01-hbft_configure | v5 `_meson_stage(v5_lock,O9,'hb','configure')`, ONLY freetype option replacement/new row name;1200s |
| hbft_compile / children/02-hbft_compile | same helper compile: pinned Python `-m mesonbuild.mesonmain compile -C O9/build/hb -j 2 harfbuzz:static_library`;1200s |
| hbft_install / children/03-hbft_install | same helper install: pinned Python `-m mesonbuild.mesonmain install -C O9/build/hb --no-rebuild`;120s |
| hbft_pkgconfig / children/04-hbft_pkgconfig | pinned pkg-config `--dont-define-prefix --static --cflags --libs harfbuzz freetype2 libass`, cwdO9;30s |
| hbft_bridge_link / children/05-hbft_bridge_link | pinned g++, `-O2 -static -static-libgcc -static-libstdc++ -IO9/prefix/include/harfbuzz -IA5/prefix/include/freetype2`, O9/config/hbft-bridge.cpp, `-o O9/build/bridge/hbft-bridge.exe`, `-Wl,-Map,O9/build/bridge/hbft-bridge.map`, original ordered absolute static group replacing ONLY HB, original ucrt_math/libstdc++/libwinpthread tail;120s |
| hbft_bridge_inspect / children/06-hbft_bridge_inspect | original pinned inspection tool `--file-header --symbols --coff-imports O9/prefix/lib/libharfbuzz.a O9/build/bridge/hbft-bridge.exe`, cwdO9;60s |

Clip each ceiling to shared remainder; sum grants no time. HB logs total<=64MiB/aggregate remainder. Stop on first uncertainty/nonzero/cap/custody failure. Check setup graph, compiled archive and installation before later children. No extra test/version/GPU/media child.

Deterministic pinned bridge.cpp includes hb-ft.h/FT; exact typed volatile observable function pointers bind BOTH hb_ft_font_changed/hb_ft_font_create_referenced; main reads them. Require surviving relocations/selected definitions despite-O2. Link only, never execute/font-load. Full original static group retains FT zlib/bzip2/cycle. No arbitrary flags/response/shell.

Outer HB stages.json: schema `clipperx-v9-hbft-transaction-v1`, EXACT `schema,status,children,descendant_cleanup_proven`; child EXACT `{name,stages,terminal,log}` Pins raw singleton stages.json/<child>.json/<child>.log. Completed=all6; failure=actual prefix. Never synthesize raw events. Postcheck check.files pins raw/generated/bridge/freeze; retained finish_records/p.stage_outputs apply. Budget group hbft_build maps to HB; settlement covers actual child PIDs.

FFmpeg configure: after Root HB review, compare full16-stage original A6 plan atO7 and actual49-token configure argv. Preserve all options+singleNVENC; cwdO9/build/ffmpeg/prefixO9/prefix; ONLY HB corrections below. One child<=1200. Protected configure unchanged; bridge precheck fills probe gap.

FFmpeg build, separately FFprobe build: original make/Makefile/SRC_PATH/SRC_LINK/SHELL/V=1/-j2 atO9; v5 `_append_map_to_argv` preserves LDEXEFLAGS plus separate ffmpeg-link.map/ffprobe-link.map. Validate actual link argv/both PE forms. No feature override/retry.

Inspect: FOUR original children ffmpeg_stripped,ffmpeg_unstripped,ffprobe_stripped,ffprobe_unstripped, original tool `--file-header --coff-imports`, O9 paths; one runner call clipped to inspection/total remainder. Retain maps/member/static/private-import/iconv/ucrt_math/_fdopen/PE checks, Root per-program symbol/CRT review.

Publish: non-native after external closure. Copied pair/ten unique ordered notices/SOURCE-NOTICE/candidate20keys/13outputs unchanged. Keep inherited lineage/HB-source path; truthful O9 build/publisher source/lock. Inspection/closure hashes bind HB proof; no extra candidate key or component/provider/media/selector action.

## Single-HB closure

HB env: v5 allowlist with O9 writable roots/no-user-site; PC order O9/O7/A5/original patched/shader/system, empty PKG_CONFIG_PATH; pinned tools/options/epoch. No Ragel/fallback/download. A5 FT.pc Requires harfbuzz may use historical A5 HB during setup bootstrap; record provenance, reject foreign/system/import alternatives. Verify old HB was not merged into new archive/exported final flags; bootstrap metadata is not final selection.

Fresh pkg-config must resolve O9 HB/FT->HB->FT cycle within fixed timeout. Capture actual PC/flags/header/config/archive/freeze. Prepend `-LO9/prefix/lib` to FFmpeg extra-ldflags and `-IO9/prefix/include/harfbuzz` to extra-cflags; replace ONLY absolute A5 HB inside original static group. Parse every HB absolute/-l and effective -L in PC output/config/actual link; reject old absolute or effective old selection. No PC edit/global substitution.

Verify `p.installed_static_inputs(current,O7,v5)` historically, replace exactly one HB row in a NEW selected-input list. Never mutate context/dependency_prefix/receipts. Group contains libass/A5FT/ONE new HB; verify paths, not basenames.

HB gates: enabled FT/HAVE_FREETYPE1/pinnedFT; selected hb-ft.cc/installed hb-ft.h; no amalgamation/subproject; actual commands/ABI/compiler; installed==built archive; ordinary no-alias files; x64 members/defined bridge symbols; newHB+A5FT bridge map and bounded PE/import/CRT closure. Frozen-prefix.json under HB evidence: closed `{schema,build_id,files}`, schema `clipperx-v9-hbft-prefix-v1`, exact sorted installed Pins, reverified pre/post later stages.

Final maps reject old/foreign HB LOAD/selected paths; FFmpeg drawtext definitions must come from new HB. Retain FFprobe required members, private DLL/import rejection, ucrt_math exception/_fdopen guard and separate Root CRT decisions. Strings/undefined references/configure success prove no runtime shaping/GPU behavior.

Inspection retains O8 payload with truthful O9 provenance plus `hbft_postcheck_sha256`. Closure schema `clipperx-v9-hbft-publication-closure-review-v1` keeps O8 closure fields plus that SAME single field; programs retain exact map/member/import digests and actual Root crt_reviewed. Publication requires it matches accepted HB predecessor and unchanged prefix. Candidate stays20keys; inspection/closure hashes bind HB proof.

## Verification and gates

Fake positives: six admissions/constructors/deadlines, untouched contexts/helper custody, injected clock, HB transaction/raw Pins,366.593HB/523.345paircarry,2400active/1200original-child configure, static cycle, confined records/publication shape/time-log subtraction on tiny fixtures.

Negatives/escape traps: origin/context/Pin/schema mutation; aliases/readrace/oversize; old HB precedence; string/undefined/elided symbol proof; wrongFT/CRT/import/maps; partial/failing/uncertain child; reuse/missing predecessor/settlement; forged final records; cost reset/time/log/area/free/PE caps; successor/unexpected process/network/media/owner call. Inject boundaries failing on native escape; inert import/main. Fakes never create Root authority.

Root checks actual Pins/clock/argv/log/config/output custody, PIDcreation/drains/terminal/caps/debit/final writes/retirement; alone writes successor authority. Failure consumes namespace; uncertainty holds. Settlement `clipperx-v9-hbft-process-settlement-v1`: O8 fields/seven roots/actual children/honest scope_limit.

Only after design/source acceptance, amend `scripts/qualify_production_runtime.py` and `clipperx/tests/test_production_runtime_qualification.py` to exact O9 build/root/source/lock/review. No fallback/discovery/schema/rawrole change. Existing153tests remain O8 evidence pending reviewed/proportionate mapping verification. Retain compact external acceptance/8rawrecords/10notices, ALL24real operations/row18verbose, media1200s, oracles/caps/owner/terminal quality, fonts/media preservation, selected_font_file_identity=unverified. No component acceptance.

## Self-review

Root436ce0 resolved HB scope. No unresolved authority/interface conflict; partition/identities/caps explicit. Design only.

Unmeasured gates: PC cycle, ABI/CRT reuse, prep time, bridge relocations, real media. Failure permits diagnosis only.

Context7 [official bridge API](https://github.com/harfbuzz/harfbuzz/blob/main/docs/harfbuzz-sections.txt); pinned recipe controls build.
