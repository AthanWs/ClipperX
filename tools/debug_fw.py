from faster_whisper import WhisperModel
import os

model_id = "nekusu/faster-whisper-large-v3-turbo-latam-int8-ct2"
print(f"DEBUG: Testing direct faster-whisper load: {model_id}")

try:
    # Run on CPU first to avoid CUDA initialization noise if possible, or GPU if needed
    # But user has GPU.
    model = WhisperModel(model_id, device="cuda", compute_type="int8")
    print("SUCCESS: faster-whisper loaded model.")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"FAILURE: {e}")
