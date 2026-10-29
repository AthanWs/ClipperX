import whisperx
import os

model_id = "marianbasti/whisper-large-v3-turbo-latam"
print(f"Attempting to load model: {model_id} with WhisperX...")

try:
    # whisperx.load_model signatures:
    # load_model(whisper_arch, device, device_index=0, compute_type="float16", asr_options=None, vad_options=None, language=None, vad_model=None, task="transcribe", download_root=None, threads=4)
    model = whisperx.load_model(model_id, device="cuda", compute_type="float16")
    print("Successfully loaded model!")
except Exception as e:
    print(f"Failed to load model: {e}")
