"""Explicitly provision pinned public model snapshots, never triggered by inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

MODELS={
    "parakeet":("nvidia/parakeet-tdt-0.6b-v3","541d1f99c6b0c3cd0b11a95167540bb8edefd82b"),
    "nemotron":("nvidia/Nemotron-3-Diarization","f667ed73aee57d40cc39428eb768b4fd87a0a29e"),
}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",choices=[*MODELS,"all"],default="all")
    args=parser.parse_args()
    from huggingface_hub import snapshot_download
    root=Path(__file__).resolve().parents[1]/"data/models/nvidia_speech"
    for name in MODELS if args.model=="all" else [args.model]:
        repo,revision=MODELS[name]
        destination=root/name
        snapshot_download(repo_id=repo,revision=revision,local_dir=destination,
            allow_patterns=["*.json","model.safetensors"],token=False,max_workers=2)
        files={}
        for path in sorted(destination.iterdir()):
            if path.is_file() and path.name != "clipperx-model.json":
                with path.open("rb") as stream:digest=hashlib.file_digest(stream,"sha256").hexdigest()
                files[path.name]=dict(sha256=digest,bytes=path.stat().st_size)
        receipt=dict(repo_id=repo,revision=revision,files=files)
        (destination/"clipperx-model.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
        print(json.dumps(dict(model=name,revision=revision,bytes=sum(r["bytes"] for r in files.values()))),flush=True)


if __name__=="__main__":
    main()
