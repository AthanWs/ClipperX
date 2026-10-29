import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from clipperx.transcription import WHISPERX_AVAILABLE, load_whisperx_model, settings

print(f"WhisperX Available (Import): {WHISPERX_AVAILABLE}")

if WHISPERX_AVAILABLE:
    try:
        print(f"Testing whisperx.load_model with {settings.whisper_model}...")
        # Note: whisperx load_model signature might differ slightly or be picky about compute_type
        model = load_whisperx_model(settings.whisper_model, device="cuda", compute_type="int8")
        print("SUCCESS: WhisperX loaded the model.")
    except Exception as e:
        print(f"FAILURE calling load_whisperx_model: {e}")
else:
    print("Skipping WhisperX load test (not available).")
