# Integrated panels and conservative webcam detection

Approved direction: remove the opaque central title/brand strip. Camera and content must touch; retain the large measured headline, persistent original Kick/ATHANW branding, and bright yellow black-outlined captions. Titles overlay source-derived image/blur instead of an opaque block. Preserve sharp content, hair, source audio, exact 60 Hz cuts, and original assets. Keep explicit legacy band mode available.

Webcam detection must be checked independently of manually supplied render crops. Existing same-model real samples support corner-local detection. Require repeated consistent support, reject competing corners/side changes rather than averaging them or choosing a large foreign content face. This is a bounded corner candidate improvement, not general identity recognition or exact webcam-boundary detection. Mixed-side windows may safely return no crop; temporal relocation remains a separately disclosed limit.

Validation: failing behavioral tests before source edits; one real pilot followed by image inspection and corrections; current full clip plus varied full-camera/browser/reaction/negative/portrait/derived relocation cases; whole-file decoded-frame and source PCM diagnostics. No new provider, key, downloads, native build, Git initialization, or original-media writes.
