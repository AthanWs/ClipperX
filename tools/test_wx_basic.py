import whisperx
print("Testing WhisperX with 'tiny' model...")
try:
    model = whisperx.load_model("tiny", device="cuda", compute_type="float16")
    print("SUCCESS: WhisperX loaded 'tiny'.")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"FAILURE: {e}")
