# Isolated FFmpeg candidate

2026-10-14 America/Bogota. Human authorized better implementation tools while
preserving the editor objective. This is a project-scoped diagnostic candidate,
not a global upgrade, production selection or validated rendering backend.

## Provenance and scope

Official FFmpeg download page links Gyan as a Windows binary provider:
https://ffmpeg.org/download.html . The provider currently lists release9.0.2:
https://www.gyan.dev/ffmpeg/builds/ . Package/source hashes are distinct from
upstream source-tarball signatures; no Windows code-signing claim is made.

Downloaded URL:
https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-9.0.2-full_build.7z

Published checksum:
https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-9.0.2-full_build.7z.sha256

HEAD response200,168971636bytes. Download SHA256 matches published value:
F0E46253C70DFE902BAC915DFB4224F0CF1B7C6EEAB9DA2CCC9A5581F9A71B13.
Archive is retained in
`.superpowers/sdd/toolchain-candidates/ffmpeg-9.0.2-full/`.

Windows tar inventory was checked for absolute/traversal paths. Extracted only
LICENSE, README.txt, bin/ffmpeg.exe and bin/ffprobe.exe into a new empty
`unpacked/ffmpeg-9.0.2-full_build/` child. The bundle's license is retained.
No installer, PATH/env/config/driver/service change, existing binary overwrite,
package-manager update, source-media mutation, credential or model call.
Project volume free space before download175319851008bytes.

## Exact binaries

Candidate prefix:
`O:/Escritorio/ClipperX/.superpowers/sdd/toolchain-candidates/ffmpeg-9.0.2-full/unpacked/ffmpeg-9.0.2-full_build/bin/`

- ffmpeg.exe version command exit0:9.0.2-full_build-www.gyan.dev.
  SHA256589E50B766D251AFDF181DD664D40BD94407E200B019989FD7468C7D118A28D0.
- ffprobe.exe version command exit0:9.0.2-full_build-www.gyan.dev.
  SHA25602264DFA4F56BAAA31E443ECE8AA16D397F6B008200E7FD184D22B9009E75FD3.

Existing global binaries remain untouched:

- C:/ffmpeg/bin/ffmpeg.exe,2024-10-21-git-baa23e40c1-full_build-www.gyan.dev,
  SHA256338BFA0630DF9E1CC5C4D38B6CBE7ED1B35403FACBD78A7060997AEED7D2A764.
- C:/ffmpeg/bin/ffprobe.exe,
  SHA256545895CA3B13CB5B582BD8365484D9AB34F84792EFB52EE301579FB7FF0DB148.

## Acceptance boundary

Version commands prove only executable startup. The old build's Vulkan controls
crashed0xC0000005; that does not prove the candidate fixes it, nor that NVENC
encoding is broken. No candidate GPU/filter/encoder/MP4 acceptance yet.
Any small synthetic comparison must invoke the candidate by exact path, keep
the original default untouched, record exit/elapsed/decoded output, and preserve
the same geometry/quality requirements. Native failure is an error, never an
implicit encoder/quality fallback. Source/runtime compatibility requires actual
tests before adoption. CPU optimization work on the original build continues
independently; extraction briefly overlapped its diagnostics and was disclosed.

## First actual candidate GPU control

Root ran `.superpowers/sdd/toolchain-candidates/probe-ffmpeg-candidate.ps1`.
It checks the pinned executable hash, changes only the executable path in the
old failed HOST lavfi control, drains both pipes and bounds its own child45s.
The saved candidate-vulkan-control.json records exit0, no timeout,1.4167194s,
one output frame. This exact control crashed before output on the old binary.
This proves that one filtering control works on the candidate, NOT that every
crop/fill/clock is correct, that quality is superior or that NVENC encodes work.
No old binary, driver, environment, global path or production selection changed.

Same architecture helper is evaluating bounded decoded geometry/quality and
representative1080controls before any adoption decision. Its report will be
`.superpowers/sdd/phase3-gpu-motion-candidate-design.md`; do not infer results
before that report is complete.

## Explicit implementation selection after bounded evidence

Controller fully read completed phase3-gpu-motion-candidate-design.md and selected
this pin for moving v2 crop graphs under the existing human authorization.
Exact exes were COPIED (not moved) into data/tools/ffmpeg/9.0.2/bin, with
LICENSE/README beside bin. Post-copy SHA256 values match both candidate pins.
No existing global binary/PATH/config was changed. Runtime requirements and
actual preflight integration are being implemented; this provision does not
itself close encoder, MP4, HDR or sustained-performance acceptance.
