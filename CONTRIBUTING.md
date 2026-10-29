# Contributing to ClipperX

Thanks for helping out. Changes go through a pull request that the maintainer reviews before merging into `main`.

## 1. Set up

Follow the [Quickstart](README.md#quickstart) section of the README. For docs-only changes you do not need the GPU environments.

## 2. Create a branch

```powershell
git checkout main
git pull
git checkout -b <type>/<short-description>   # e.g. docs/troubleshooting, fix/caption-offset
```

Keep each branch focused on one topic.

## 3. Make your change

- Do not change the pipeline's behaviour in a docs or test PR; open a separate PR for that.
- If you add or rename a setting in `clipperx/config.py`, document it in `.env.example`.
- Never commit `.env`, API keys, VODs, renders, logs, model weights or anything under `data/`. The `.gitignore` covers these; check `git status` before committing.

## 4. Run the tests

```powershell
.\.venv-whisperx\Scripts\python.exe -m pytest clipperx/tests -q
```

The suite fakes the external model boundary, so it never calls an LLM. Some test modules import the heavy runtime (Torch, WhisperX, pyannote) and only run inside the full `.venv-whisperx` environment. If you could not run part of the suite, say so in the PR.

## 5. Commit

Write commit messages in English, one logical change per commit, with a conventional prefix:

```
docs: add troubleshooting for FFmpeg on PATH
fix: keep captions inside the mobile safe zone
test: cover clip_text with multiple time ranges
chore: ignore the speech setup lock file
```

## 6. Open a pull request

```powershell
git push -u origin <your-branch>
gh pr create --base main
```

In the description, explain what changed and why, and list how you tested it (including anything you could not test).
