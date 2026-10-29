import whisperx
import time

model_id = "nekusu/faster-whisper-large-v3-turbo-latam-int8-ct2"
print(f"Testing load of model: {model_id}")
print("This may take a while to download on first run (approx 1.5GB)...")

start_time = time.time()
try:
    # Cargas el modelo especificando float16 o int8 se maneja auto por el modelo, 
    # pero este es INT8. WhisperX/Faster-Whisper usualmente detectan.
    # Forzamos compute_type='int8' para coincidir con el modelo si es necesario, 
    # o 'float16' si tenemos GPU decente (el modelo es int8 pero se puede correr en float16 compute).
    # Probaremos default.
    model = whisperx.load_model(model_id, device="cuda", compute_type="int8")
    end_time = time.time()
    print(f"\nSUCCESS: Model loaded in {end_time - start_time:.2f} seconds.")
except Exception as e:
    print(f"\nFAILURE: {e}")
