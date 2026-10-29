# ClipperX

**ClipperX turns long livestream VODs into short vertical clips, edited the way a human clip editor would do it.**

You give it a 16:9 recording of a Twitch/Kick stream (30 minutes to 10 hours, one mixed audio track). ClipperX transcribes it, finds the moments that stand on their own (jokes, reactions, unexpected events, debates), keeps the setup and the payoff, and renders 9:16 clips of up to 179 seconds. Each clip gets an adaptive camera/content layout, word-timed captions, a title and persistent creator branding.

Everything heavy runs locally on a single consumer GPU (developed on an RTX 3060 12 GB, Windows 11). The only remote part is an optional, configurable LLM provider for editorial and visual decisions.

> Built and maintained with [Claude Code](https://claude.com/claude-code).

---

## How it works

```
VOD (16:9, mixed audio)
  │
  ├─ 1. Local indexing ─────────── sparse visual sampling + dense probes around changes,
  │                                face/webcam tracking (YuNet + SFace), audio energy/silence
  ├─ 2. Speech ─────────────────── ASR: NVIDIA Parakeet TDT 0.6B v3 (multilingual, native word timestamps)
  │                                Diarization: NVIDIA Nemotron (optional), voiceprint matching for the streamer
  ├─ 3. Editorial planner (LLM) ── token-budgeted transcript batches → candidate discovery →
  │                                evidence requests (frames/audio before, inside, after) →
  │                                boundary review (split / merge / extend) → global composition
  ├─ 4. Visual director (LLM) ──── per-shot layout (camera / content / split / full-cam), crops,
  │                                motivated zooms, title, captions, brand, optional SFX cues
  └─ 5. Render (FFmpeg) ────────── H.264/AAC 1080x1920, ASS captions, overlays, audio checks
                                   → clip.mp4 + clip.metadata.json
```

Design principles that run through the codebase:

- **Deterministic evidence, model decisions.** Indexing, ASR, geometry and rendering are deterministic and cached. The LLM only makes editorial and visual choices, and returns **strict JSON schemas** (Pydantic) that the host validates before anything is rendered.
- **Resumable and auditable.** Every stage persists cursors, receipts, fingerprints, token usage and decisions under a per-VOD job directory, so a 10-hour stream can be resumed after an interruption without paying for model calls twice.
- **No silent fallbacks.** If a backend, model asset or validation fails, the run stops with an explicit status (`review_required`, `partial`, `error`). It does not quietly degrade to a worse path or invent transcripts, identities or approvals.
- **Safe-zone aware layouts.** The director must keep faces, HUD and important regions visible and respect mobile UI margins. Overlays may only use space that is actually free.

### Main modules (`clipperx/`)

| Area | Modules |
|---|---|
| Jobs & evidence | `jobs.py`, `timeline.py`, `visual_timeline*.py`, `audio_features.py`, `evidence.py` |
| Speech | `parakeet.py`, `parakeet_worker.py`, `diarization.py`, `nemotron_worker.py`, `voiceprint.py`, `speaker_identity.py` |
| Editorial planner | `context_budget.py`, `planner_contracts.py`, `planner_prompts.py`, `planning_state.py`, `planner_engine.py`, `composition_groups.py` |
| Visual director | `visual_contracts.py`, `visual_prompts.py`, `visual_director.py`, `style_library.py` |
| Render & export | `render_*.py`, `editing.py`, `subtitles.py`, `overlay_ass.py`, `export_pipeline.py`, `sfx_authority.py` |
| Orchestration | `autonomous_pipeline.py`, `structured_client.py`, `cli.py`, `interface.py` |

The `clipperx/tests/` suite (pytest) uses real files, ledgers and schemas, and fakes only the external model boundary.

## Install

Requirements: Windows 10/11, Python 3.12, an NVIDIA GPU with CUDA, and FFmpeg on `PATH`.

```powershell
# Main environment (render, indexing, planner)
powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1

# Local speech stack (Parakeet ASR + Nemotron diarization), in its own environment
powershell -ExecutionPolicy Bypass -File scripts\setup_nvidia_speech.ps1 -DownloadModels

Copy-Item .env.example .env
```

### LLM provider

The LLM is **disabled by default** (`LLM_PROVIDER=disabled`). Indexing, transcription and the compatibility renderer work without it. To enable the planner and the director, point `.env` at any **OpenAI-compatible API** (hosted or local) whose model supports JSON-schema output and image input:

```ini
LLM_PROVIDER=openai_compatible
OPENAI_BASE_URL=https://your-endpoint/v1
OPENAI_API_KEY=...
OPENAI_MODEL=your-model-name
```

No inference is ever sent without the explicit `--confirm-inference` flag. Every run has hard limits on calls, images and tokens.

## Usage

```powershell
$py = ".\.venv-whisperx\Scripts\python.exe"

# Environment check
& $py -m clipperx.cli doctor

# Local, model-free steps
& $py -m clipperx.cli index-visual VOD.mp4
& $py -m clipperx.cli transcribe VOD.mp4
& $py -m clipperx.cli diarize VOD.mp4

# Full resumable workflow: prepare → editorial plan → visual plan → export
& $py -m clipperx.cli auto-edit VOD.mp4 --output-dir data\clips `
    --model your-model-name --context-window 65536 --encoder libx264 --confirm-inference

# Desktop UI
.\run_clipperx.bat
```

Run the tests:

```powershell
.\.venv-whisperx\Scripts\python.exe -m pytest clipperx/tests -q
```

## Status (honest)

ClipperX is an **active work in progress**, not a finished product.

**Working and tested:**
- Local visual and audio indexing with face/webcam identity and explicit uncertainty states (`confident` / `ambiguous` / `unavailable`).
- Parakeet ASR with native word timestamps and chunked long-form processing; optional Nemotron diarization.
- Staged editorial planner with resumable ledgers, evidence requests, boundary review and coverage mode (all justified moments, not a fixed quota).
- Visual director contracts and a "creator" publication gate: brand on every shot, title on the first shot, captions wherever speech exists.
- Resumable per-clip export coordinator. Compatibility renderer with styled captions and branding.

**Not yet proven:**
- End-to-end autonomous editing of a full multi-hour VOD with the chosen model.
- Editorial quality at scale (highlight precision, natural cuts) beyond bounded real-model samples.
- The v2 director placing overlays in genuinely free regions on every layout. 60 FPS native v2 render on real footage.

## Roadmap

1. Real end-to-end run on full VODs; human review loop feeding fixes back into prompts and validators.
2. Title and on-screen text styles learned from real short-form clip conventions (hooks, quotes, reaction framing).
3. Motivated zooms, punch-ins and sound design driven by detected events, not periodic effects.
4. Replace the legacy pipeline in the UI with the new autonomous workflow.
5. Synchronized chat as an additional signal once new streams are recorded.
6. Evaluate newer local speech models (e.g. larger multitalker ASR) as consumer-GPU builds become available.

### Research notes

`tools/` contains a closed experiment that used Meta's TRIBE v2 brain-encoding model to predict audience engagement from clips. A promising first batch (ρ≈0.54, n=21) **did not replicate** on fresh pre-registered data (ρ≈0.10, n=36), so it was not integrated. The model is not redistributed here.

---

## Resumen en español

ClipperX convierte directos largos (16:9, una pista de audio) en clips verticales 9:16 de hasta 179 s, editados como lo haría una persona. Indexa el vídeo de forma local, transcribe con Parakeet y diariza con Nemotron. Después, un planner editorial con LLM encuentra los momentos con contexto y desenlace, y un director visual decide la composición, los acercamientos, el título, los subtítulos y la marca. El render se hace con FFmpeg. Todo es reanudable y auditable, con contratos JSON estrictos y sin fallbacks silenciosos. Es un proyecto en desarrollo activo, hecho con Claude Code.
