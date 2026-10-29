import sys
from pathlib import Path
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from clipperx.transcription import load_model, WHISPERX_AVAILABLE
from clipperx.config import settings

print(f"WhisperX Available: {WHISPERX_AVAILABLE}")
print(f"Configured Model: {settings.whisper_model}")

try:
    print("Loading model via clipperx.transcription...")
    model = load_model()
    print(f"SUCCESS: Model loaded: {type(model)}")
    print(f"Compute type: {model.model.encode_options.get('compute_type', 'unknown')}")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"FAILURE: {e}")
