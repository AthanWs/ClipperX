# Compact vertical composition and visual iteration

The user approved the preceding critique (larger integrated camera, substantially less wasted black space, compact title/captions, motivated reframes) and explicitly required autonomous render/review/correct cycles, including image/audio sync and original-camera changes. This is execution of that direction, not another approval request.

## Deliverable and scope

Improve the actual compatibility rendering route currently used by the shown sample, retaining its explicit legacy identity; do not pretend it is a successful GPT-6 visual-director plan or bypass v2 safety gates. Changes must be reusable application code, not only a different pre-rendered MP4. Preserve all originals and previous attempts. No custom FFmpeg rebuild, selector activation, publication, cloud upload, account/key changes or extra UI control.

Initial visual target used64px top inset plus120px title/brand band. Actual visual iteration superseded it with a more prudent192px top guard plus72px measured title/brand band, while retaining the larger camera,80px caption gap and140px lower breathing space. The title uses original Impact measured to fit (max46px); the default subtitle style retains Montserrat SemiBold at52px with210px side margins. Use source geometry and protect the visible face, important gameplay action, phone-right UI-sensitive text and original aspect ratios. These are design defaults, not measured TikTok pixel guarantees. Choose motivated sample crops from real source evidence; do not apply a blind face zoom universally. Per-segment crop/position changes must actually affect rendering and captions.

## Time/audio invariants

- Deliver1080x1920 H264/AAC, CFR60, preserving narrative source range/order and <=179s.
- Generate the canvas at60FPS, not default25; partition video frame counts on one cumulative rational60Hz output clock so cuts do not add frames.
- Construct final audio directly from the original ordered source ranges, merging only exactly adjacent ranges. Do not accumulate AAC padding from individually encoded video subclips. No invented SFX or voice replacement.
- Captions use existing source word times and follow each segment's actual caption band. Do not silently rewrite uncertain ASR words. Preserve title/brand assets and source fonts. Selected runtime font identity remains unverified until demonstrated; visual glyph QA is separate.

## Review loop

Render a fresh version; decode/count the entire delivered file; check time/frame/audio stream properties and compare source/output PCM content/offset, including cuts and tail. Inspect source/output representative frames plus dense frames at cuts, changing camera, payoff and title/caption boundaries. Verify eyes/mouth, key action, text extents, consistent aspect, readable mobile text and no unwanted gaps/black/frozen frames. Correct actual findings, render again and recheck. Preserve iteration differences and exact actual results. Pixel metrics assist review; they do not automatically certify editorial or visual quality.

Exercise one real camera-state transition in addition to the primary chair clip; a single stable crop is not proof camera relocation works generally. Do not label the whole editor, all VODs, viral performance or the separate production24 qualification complete from these samples. Handoff only after agent review finds the scoped samples suitable for user review, with any honest remaining limitations stated.

## Evidence-led implementation clarifications

The user explicitly expanded varied edge-case coverage during implementation. The real browser example demonstrates that automatic whole-column removal for a corner webcam cuts visible paragraph text. Add an optional internal per-segment content ROI, separate from camera ROI; absent an explicit content ROI, retain full source. Append this optional dataclass field to preserve positional caller compatibility. This is not a new UI control or a model-authored region claim. For PIP, retain source camera aspect with a contained foreground and position it left/below center, above the lower mobile-control margin; caption placement follows its actual rectangle.

Actual cross-layout concat failed on sample aspect5440:5439 vs1:1, and intermediate NVENC packets reported a missing final-frame duration without explicit encoder rate. Normalize square pixels,60Hz time base and explicit encoder rate in every stage. Bound final video with its trim filter rather than stopping the entire mux with -vframes, which truncated the original audio tail. These correct actual measured failures; they do not relax source protection, duration bounds or the independent v2/native acceptance gates.

Reserve any opaque header before containing full-camera, portrait and PIP sharp foreground; do not erase source pixels with the later overlay. Use the same clamped PIP crop to compute actual camera and caption positions. New LayoutConfig fields must be appended so old positional construction remains valid. The review matrix now includes seven real source situations, two explicitly derived relocation/portrait fixtures and one nonchronological ordering control; blank source footage is a negative HOLD, not a publishable clip.
